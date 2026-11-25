# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The inverses of the hoisting moves: an if, or the statements that compute a temporary, move into the loop or
doalls that use them. Built on DaCe's ``MoveIfIntoMap`` helpers; DaCe has no pass for either direction.

``push_if_into_maps``: ``if c: doall i: B`` becomes ``doall i: if c: B`` wherever the if sits (``MoveIfIntoMap``
needs an outer map and moves a guard chain; this moves only the condition).

``sink_into_loop`` and ``sink_into_map``: the statements that compute one named temporary move into the one loop or doall
that reads it, the inverse of loop-invariant code motion.

``t = a * b; for i: C[i] = t * X[i]`` becomes ``for i: {t = a * b; C[i] = t * X[i]}``. The statements then run once
per iteration instead of once, which trades recomputation for storage: ``t`` no longer lives across the loop, and a
map that read it from outside no longer has an input that ties it to the statements before it (so it can fuse with
a neighbor). The value is the same in every iteration, since the statements read nothing the loop writes.

Only plain statements sink: tasklets and the transients between them, in the state just before the loop or map, or
upstream of the map in its own state. A container computed by a map, a library call or a nested SDFG does not. Every
check that fails returns its reason as text, which a caller can show.
"""

import copy
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from dace import data
from dace import memlet as mm
from dace import sdfg as sd
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, LoopRegion, SDFGState
from dace.transformation.interstate.move_if_into_map import (
    GuardMatch,
    MoveIfIntoMap,
    descriptor_symbols,
    names_read,
    sibling_maps,
    single_body_nsdfg,
    single_meaningful_state,
)
from dace.transformation.passes.normalize_wcr import NormalizeWCR


def match_if_over_maps(cond_block: ConditionalBlock) -> GuardMatch | None:
    """The condition of ``cond_block`` as a guard :func:`push_if_into_maps` can move, or ``None``.

    Matches ``if c: <state of sibling maps>`` with no ``else``, anywhere: at the top of an SDFG, in a loop body or in
    a branch. Unlike :class:`MoveIfIntoMap`, no outer map and no guard chain are involved; only the condition moves.
    Refused when the branch writes what the condition reads (the condition would change between iterations), when
    the condition uses a symbol not defined at the maps, and when a map's body already uses one of its names.
    """
    if len(cond_block.branches) != 1 or cond_block.branches[0][0] is None:
        return None
    branch_state = single_meaningful_state(cond_block.branches[0][1])
    if branch_state is None or sibling_maps(branch_state) is None:
        return None
    sdfg = cond_block.sdfg
    reads, symbols = names_read(cond_block.branches[0][0].as_string, sdfg.arrays)
    symbols |= descriptor_symbols(sdfg.arrays, reads)
    body_writes = branch_state.read_and_write_sets()[1]
    defined = branch_state.symbols_defined_at(sibling_maps(branch_state)[0])
    if reads & body_writes or not symbols <= defined.keys():
        return None
    match = GuardMatch(sdfg, branch_state, [], reads, set(), set(), symbols)
    return match if all(MoveIfIntoMap.fits(match, me) for me in sibling_maps(branch_state)) else None


def push_if_into_maps(cond_block: ConditionalBlock) -> bool:
    """``if c: map i: B`` becomes ``map i: if c: B``, for every sibling map of the branch's one state.

    Each map body becomes a nested SDFG whose blocks run under a copy of the condition; the data the condition reads
    enters through the map, its symbols through ``symbol_mapping``. The condition is evaluated once per iteration and
    has the value it had once, since nothing in the branch writes what it reads. The inverse of
    :class:`~dace.transformation.interstate.move_map_invariant_if_up.MoveMapInvariantIfUp` for a single map.

    :param cond_block: The conditional block to push in.
    :returns: Whether the conditional block matched :func:`match_if_over_maps` and was pushed in.
    """
    match = match_if_over_maps(cond_block)
    if match is None:
        return False
    state, parent = match.branch_state, cond_block.parent_graph
    for entry in sibling_maps(state):
        # A loop variable is no symbol of the SDFG, so its type comes from where the map sits
        defined = state.symbols_defined_at(entry)
        nsdfg = single_body_nsdfg(state, entry)
        # The nest repeats a body reduction on its boundary; surface it the way NormalizeWCR does
        NormalizeWCR().normalize_nested_node(state, nsdfg)
        for name in sorted(match.symbols - nsdfg.sdfg.symbols.keys()):
            nsdfg.sdfg.add_symbol(name, defined[name])
        MoveIfIntoMap.guard_inner_body(match, cond_block, nsdfg)
        for name in sorted(match.reads):
            nsdfg.add_in_connector(name)
            memlet = mm.Memlet.from_array(name, match.enclosing.arrays[name])
            state.add_memlet_path(state.add_read(name), entry, nsdfg, memlet=memlet, dst_conn=name)

    was_start = parent.start_block is cond_block
    cond_block.branches[0][1].remove_node(state)
    parent.add_node(state, ensure_unique_name=True, is_start_block=was_start)
    for e in parent.in_edges(cond_block):
        parent.add_edge(e.src, state, e.data)
    for e in parent.out_edges(cond_block):
        parent.add_edge(state, e.dst, e.data)
    parent.remove_node(cond_block)
    return True


@dataclass(slots=True)
class SinkPlan:
    """What :func:`sink_into_loop` or :func:`sink_into_map` moves: the statements, the state they leave, the
    containers they compute, and those they read from outside (which stay where they are)."""

    source: SDFGState
    statements: set[nodes.Node]
    computed: set[str]
    inputs: set[str]


def mentions(code: str, name: str) -> bool:
    """Whether ``name`` appears in ``code`` as a whole word."""
    return re.search(rf"(?<![\w.]){re.escape(name)}(?!\w)", code) is not None


def block_codes(block: ControlFlowBlock) -> Iterable[str]:
    """The conditions and statements a control-flow block evaluates itself, outside its states."""
    if isinstance(block, LoopRegion):
        for code in (block.init_statement, block.loop_condition, block.update_statement):
            if code is not None:
                yield code.as_string
    elif isinstance(block, ConditionalBlock):
        for condition, branch in block.branches:
            if condition is not None:
                yield condition.as_string


def control_flow_reads(sdfg: sd.SDFG, name: str) -> str | None:
    """Where an interstate edge or a block's own condition or bounds uses ``name``, or ``None``."""
    for edge in sdfg.all_interstate_edges():
        assigned = edge.data.assignments
        if mentions(edge.data.condition.as_string, name) or name in assigned:
            return f"an interstate edge uses {name}."
        if any(mentions(value, name) for value in assigned.values()):
            return f"an interstate edge uses {name}."
    for block in sdfg.all_control_flow_blocks():
        if any(mentions(code, name) for code in block_codes(block)):
            return f"{block.label} reads {name} in its own condition or bounds."
    return None


def used_elsewhere(sdfg: sd.SDFG, name: str, source: SDFGState, readers: set[SDFGState]) -> str | None:
    """Why ``name`` cannot sink: a write in a state other than ``source``, a read in a state other than ``source``
    and the ``readers``, or a use in control flow. ``None`` when nothing else uses it."""
    for state in sdfg.states():
        if state is source:
            continue
        for node in state.data_nodes():
            if node.data != name:
                continue
            if state.in_degree(node) > 0:
                return f"{name} is also written in {state.label}; only its computation just before the loop sinks."
            if state not in readers:
                return f"{name} is also read in {state.label}, outside the loop."
    return control_flow_reads(sdfg, name)


def statements_writing(state: SDFGState, name: str) -> set[nodes.Node] | str:
    """The tasklets and access nodes of ``state`` that compute ``name``: everything upstream of its writes, up to
    the containers read from outside. A transient between two statements belongs to them when no other state and
    no control flow uses it."""
    sdfg = state.sdfg
    writes = [n for n in state.data_nodes() if n.data == name and state.in_degree(n) > 0]
    if not writes:
        return f"{state.label} does not compute {name}."
    found: set[nodes.Node] = set()
    stack: list[nodes.Node] = list(writes)
    while stack:
        node = stack.pop()
        if node in found:
            continue
        found.add(node)
        for edge in state.in_edges(node):
            src = edge.src
            if isinstance(src, nodes.Tasklet):
                if src.has_side_effects(sdfg):
                    return f"{src.label} has a side effect; it runs once and stays outside the loop."
                stack.append(src)
            elif isinstance(src, nodes.AccessNode):
                desc = sdfg.arrays[src.data]
                private = desc.transient and used_elsewhere(sdfg, src.data, state, set()) is None
                if src.data != name and private and state.in_degree(src) > 0:
                    stack.append(src)
            else:
                return f"{name} is computed by {src.label}, not by plain statements; only statements sink."
    return found


def container_refusal(sdfg: sd.SDFG, name: str) -> str | None:
    """Why ``name`` is no plain temporary of ``sdfg``, the only kind of container whose statements sink."""
    desc = sdfg.arrays.get(name)
    if desc is None:
        return f"{name} is no container of {sdfg.label}."
    if not desc.transient:
        return f"{name} is an argument; only a temporary sinks."
    if isinstance(desc, (data.Stream, data.View)):
        return f"{name} is a stream or a view; only plain temporaries sink."
    return None


def plan(
    source: SDFGState,
    name: str,
    readers: set[SDFGState],
    written: set[str],
    consumers: set[nodes.Node],
) -> SinkPlan | str:
    """The statements of ``source`` that compute ``name``, for a loop or map whose states ``readers`` read it, which
    writes the containers ``written``, and whose nodes in ``source`` are ``consumers``."""
    sdfg = source.sdfg
    why = container_refusal(sdfg, name)
    if why is not None:
        return why
    statements = statements_writing(source, name)
    if isinstance(statements, str):
        return statements
    reason = used_elsewhere(sdfg, name, source, readers)
    if reason is not None:
        return reason
    for node in statements:
        if any(e.dst not in statements and e.dst not in consumers for e in source.out_edges(node)):
            what = node.data if isinstance(node, nodes.AccessNode) else node.label
            return f"{what} is also used in {source.label} outside the loop; it must stay."
    computed = {n.data for n in statements if isinstance(n, nodes.AccessNode)}
    inputs = {
        e.src.data
        for n in statements
        for e in source.in_edges(n)
        if isinstance(e.src, nodes.AccessNode) and e.src not in statements
    }
    if inputs & written:
        changed = ", ".join(sorted(inputs & written))
        return f"the loop writes {changed}, which the statements read: they would compute another value."
    if computed & written:
        return f"the loop writes {', '.join(sorted(computed & written))}, which the statements compute."
    return SinkPlan(source, statements, computed, inputs)


def predecessor_state(block: ControlFlowBlock) -> SDFGState | None:
    """The one state that runs just before ``block``, through a plain edge, or ``None``."""
    edges = block.parent_graph.in_edges(block)
    if len(edges) != 1 or not edges[0].data.is_unconditional() or edges[0].data.assignments:
        return None
    return edges[0].src if isinstance(edges[0].src, SDFGState) else None


def plan_loop(loop: LoopRegion, name: str) -> SinkPlan | str:
    """What :func:`sink_into_loop` would move, or why it cannot."""
    source = predecessor_state(loop)
    if source is None:
        return f"{loop.label} has no single state just before it, through a plain edge, to sink from."
    return plan(source, name, set(loop.states()), loop.read_and_write_sets()[1], set())


def sink_into_loop(loop: LoopRegion, name: str) -> str | None:
    """Move the statements that compute ``name`` from the state before ``loop`` to the start of its body.

    :param loop: The loop that reads ``name``.
    :param name: The transient the statements compute.
    :returns: ``None`` once moved; else why nothing changed.
    """
    planned = plan_loop(loop, name)
    if isinstance(planned, str):
        return planned
    target = SDFGState(f"sunk_{name}")
    copy_statements(planned, target, {})
    start = loop.start_block
    loop.add_node(target, is_start_block=True, ensure_unique_name=True)
    loop.add_edge(target, start, sd.InterstateEdge())
    remove_statements(planned)
    return None


def map_read_refusal(state: SDFGState, entry: nodes.MapEntry, name: str) -> str | None:
    """Why the map ``entry`` takes no statements computing ``name``: it is nested, or it does not read all of
    ``name``."""
    if state.entry_node(entry) is not None:
        return f"{entry.map.label} is inside another map; only a top-level map takes statements."
    reads = [e for e in state.in_edges(entry) if e.data.data == name]
    if not reads:
        return f"{entry.map.label} does not read {name}."
    desc = state.sdfg.arrays.get(name)
    if desc is None:
        return f"{name} is no container of {state.sdfg.label}."
    whole = Memlet.from_array(name, desc).subset
    if any(e.data.subset != whole or e.data.wcr is not None for e in reads):
        return f"{entry.map.label} reads only a part of {name}; only a container it reads whole sinks."
    return None


def plan_map(state: SDFGState, entry: nodes.MapEntry, name: str) -> SinkPlan | str:
    """What :func:`sink_into_map` would move, or why it cannot."""
    why = map_read_refusal(state, entry, name)
    if why is not None:
        return why
    reads = [e for e in state.in_edges(entry) if e.data.data == name]
    written = {e.data.data for e in state.out_edges(state.exit_node(entry)) if e.data.data is not None}
    computed_here = any(state.in_degree(e.src) > 0 for e in reads)
    if computed_here:
        return plan(state, name, set(), written, {entry})
    source = predecessor_state(state)
    if source is None:
        return f"the state of {entry.map.label} has no single state just before it, through a plain edge."
    others = [
        n
        for n in state.data_nodes()
        if n.data == name and (state.in_degree(n) > 0 or any(e.dst is not entry for e in state.out_edges(n)))
    ]
    if others:
        return f"{name} is also used in {state.label} outside {entry.map.label}."
    return plan(source, name, {state}, written, set())


def sink_into_map(state: SDFGState, entry: nodes.MapEntry, name: str) -> str | None:
    """Move the statements that compute ``name`` into the body of the map ``entry``, as its first state.

    The body becomes a nested SDFG if it is not one. The containers the statements compute become its own
    transients (``name`` under the connector the body already reads it through), and the containers they read enter
    through the map, under new names.

    :param state: The state of the map.
    :param entry: The map that reads ``name``.
    :param name: The transient the statements compute.
    :returns: ``None`` once moved; else why nothing changed.
    """
    planned = plan_map(state, entry, name)
    if isinstance(planned, str):
        return planned
    sdfg = state.sdfg
    defined = state.symbols_defined_at(entry)
    nsdfg = single_body_nsdfg(state, entry)
    inner = nsdfg.sdfg
    rename: dict[str, str] = {}
    for edge in [e for e in state.in_edges(nsdfg) if e.data.data == name]:
        rename[name] = edge.dst_conn
        state.remove_memlet_path(edge)
        nsdfg.remove_in_connector(edge.dst_conn)
    for computed in sorted(planned.computed):
        desc = copy.deepcopy(sdfg.arrays[computed])
        desc.transient = True
        inner_name = rename.get(computed, computed)
        if inner_name in inner.arrays:
            inner.remove_data(inner_name, validate=False)
        rename[computed] = inner.add_datadesc(inner_name, desc, find_new_name=True)
    for outer in sorted(planned.inputs):
        desc = copy.deepcopy(sdfg.arrays[outer])
        desc.transient = False
        inner_name = inner.add_datadesc(f"{outer}_sunk", desc, find_new_name=True)
        rename[outer] = inner_name
        nsdfg.add_in_connector(inner_name)
        memlet = Memlet.from_array(outer, sdfg.arrays[outer])
        state.add_memlet_path(state.add_read(outer), entry, nsdfg, memlet=memlet, dst_conn=inner_name)
    target = SDFGState(f"sunk_{name}")
    copy_statements(planned, target, rename)
    start = inner.start_block
    inner.add_node(target, is_start_block=True, ensure_unique_name=True)
    inner.add_edge(target, start, sd.InterstateEdge())
    for symbol in sorted(target.free_symbols - inner.symbols.keys()):
        inner.add_symbol(symbol, defined[symbol])
        nsdfg.symbol_mapping[symbol] = symbol
    remove_statements(planned)
    return None


def copy_statements(planned: SinkPlan, target: SDFGState, rename: Mapping[str, str]) -> None:
    """Copy the statements, and an access node for each container they read, into ``target``, naming each
    container as ``rename`` says (its own name when absent)."""
    source = planned.source
    copies: dict[nodes.Node, nodes.Node] = {}
    for node in planned.statements:
        if isinstance(node, nodes.AccessNode):
            copies[node] = target.add_access(rename.get(node.data, node.data))
        else:
            copies[node] = copy.deepcopy(node)
            target.add_node(copies[node])
    for node in planned.statements:
        for edge in source.in_edges(node):
            if edge.src not in copies:
                copies[edge.src] = target.add_access(rename.get(edge.src.data, edge.src.data))
            memlet = copy.deepcopy(edge.data)
            if memlet.data is not None:
                memlet.data = rename.get(memlet.data, memlet.data)
            target.add_edge(copies[edge.src], edge.src_conn, copies[node], edge.dst_conn, memlet)


def remove_statements(planned: SinkPlan) -> None:
    """Remove the statements from their state, with the access nodes they leave without edges, and the containers
    they computed once nothing uses them."""
    source = planned.source
    feeding = {e.src for n in planned.statements for e in source.in_edges(n) if e.src not in planned.statements}
    # the access node of the map's read may have gone with its memlet path already
    source.remove_nodes_from([n for n in planned.statements if n in source.nodes()])
    source.remove_nodes_from([n for n in feeding if source.degree(n) == 0])
    sdfg = source.sdfg
    for computed in planned.computed:
        if not any(n.data == computed for s in sdfg.states() for n in s.data_nodes()):
            sdfg.remove_data(computed, validate=False)
