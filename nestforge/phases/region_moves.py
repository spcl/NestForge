# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scheduling moves on one named region that no single DaCe pattern transformation performs.

``LoopFission``, ``MoveIfIntoLoop`` and ``MoveLoopInvariantIfUp`` expose only a whole-program ``apply_pass``, so
these moves call the per-region match and rewrite steps those passes are built from. Map-loop interchange and
subgraph fission have no DaCe counterpart and compose DaCe helpers with ``MapFission``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import dace
from dace import symbolic
from dace.sdfg import nodes
from dace.sdfg.graph import SubgraphView
from dace.sdfg.state import BreakBlock, ConditionalBlock, ContinueBlock, ControlFlowBlock, LoopRegion, ReturnBlock
from dace.sdfg.state import SDFGState
from dace.sdfg.utils import set_nested_sdfg_parent_references
from dace.transformation.dataflow.map_fission import MapFission
from dace.transformation.helpers import nest_sdfg_subgraph
from dace.transformation.interstate import move_loop_invariant_if_up
from dace.transformation.interstate.multistate_inline import InlineMultistateSDFG
from dace.transformation.passes import loop_fission, move_if_into_loop

from nestforge.ir.extract import detach
from nestforge.ir.names import in_order

LOOP_FISSION_REFUSED = "blocked by LoopFission: the loop body has no two independent statement groups."
IF_INTO_LOOP_REFUSED = (
    "blocked by MoveIfIntoLoop: the branch is not a single no-else guard whose body ends in this loop, or the "
    "condition or its prep reads what the loop writes."
)
IF_OUT_OF_LOOP_REFUSED = (
    "blocked by MoveLoopInvariantIfUp: the guard is not the loop's only statement, or its condition reads the loop "
    "variable or data the loop writes."
)
SUBGRAPH_FISSION_REFUSED = (
    "blocked by MapFission: a half holds a conditional, or an interstate assignment reads a map parameter."
)


@dataclass(frozen=True, slots=True)
class Rewrite:
    """A legal move on the regions ``labels`` name; ``commit`` applies it and ``name`` says what ran."""

    name: str
    labels: tuple[str, ...]
    commit: Callable[[], None]


def finish(sdfg: dace.SDFG) -> None:
    """Restore the parent links and region ids a manual rewrite of ``sdfg``'s control flow leaves stale."""
    root = sdfg.root_sdfg
    set_nested_sdfg_parent_references(root)
    root.reset_cfg_list()


# loop fission


def plan_loop_fission(loop: LoopRegion) -> Rewrite | str:
    """Split ``loop`` into one loop per independent statement group, as ``LoopFission`` does program-wide."""
    sdfg = loop.sdfg
    compute = loop_fission._single_compute_state(loop)
    if compute is not None:
        # decided on a copy: the bridge rewrite is sound only once fission separates producer and consumer
        if not loop_fission._fissions_after_bridge_rewrite(loop, sdfg):
            return LOOP_FISSION_REFUSED

        def split_state() -> None:
            loop_fission._rewrite_per_iter_bridges(compute, loop.loop_variable, sdfg)
            loop_fission.LoopFission._fission(loop, compute, sdfg)
            finish(sdfg)

        return Rewrite("LoopFission", (loop.label,), split_state)
    groups = loop_fission._independent_block_groups(loop)
    if groups is None:
        return LOOP_FISSION_REFUSED

    def split_blocks() -> None:
        loop_fission.LoopFission._fission_blocks(loop, groups)
        finish(sdfg)

    return Rewrite("LoopFission", (loop.label,), split_blocks)


# if-loop interchange


def plan_if_into_loop(cond: ConditionalBlock, loop: LoopRegion) -> Rewrite | str:
    """Push the guard ``cond`` into the ``loop`` its branch ends in: ``if c: for i: B`` -> ``for i: if c: B``."""
    match = move_if_into_loop._match(cond)
    if match is None or match[0] is not cond or loop.parent_graph is not match[2] or match[2].out_degree(loop):
        return IF_INTO_LOOP_REFUSED

    def push() -> None:
        move_if_into_loop.MoveIfIntoLoop._move(*match)
        finish(loop.sdfg)

    return Rewrite("MoveIfIntoLoop", (cond.label, loop.label), push)


def plan_if_out_of_loop(loop: LoopRegion, cond: ConditionalBlock) -> Rewrite | str:
    """Hoist the loop-invariant guard ``cond`` out of ``loop``: ``for i: if c: B`` -> ``if c: for i: B``."""
    sdfg = loop.sdfg
    match = move_loop_invariant_if_up._match(loop)
    if match is None or match[0] is not loop or match[1] is not cond:
        return IF_OUT_OF_LOOP_REFUSED

    def hoist() -> None:
        move_loop_invariant_if_up.MoveLoopInvariantIfUp._move(*match)
        finish(sdfg)

    return Rewrite("MoveLoopInvariantIfUp", (loop.label, cond.label), hoist)


# map-loop interchange


def map_body(state: SDFGState, entry: nodes.MapEntry) -> nodes.NestedSDFG | None:
    """The nested SDFG that is the whole body of the map at ``entry``, if it is one."""
    targets = {edge.dst for edge in state.out_edges(entry)}
    body = next(iter(targets))
    return body if len(targets) == 1 and isinstance(body, nodes.NestedSDFG) else None


def free_names(expression: str) -> set[str]:
    return {str(s) for s in symbolic.pystr_to_symbolic(expression).free_symbols}


def carried_across_iterations(loop: LoopRegion, body: dace.SDFG) -> str | None:
    """What keeps a value from one iteration of ``loop`` to the next inside the map body, if anything does.

    Once the loop is outside, each iteration calls the body afresh, so its transients and symbols start over.
    """
    assigned = {name for edge in loop.all_interstate_edges() for name in edge.data.assignments}
    for edge in loop.all_interstate_edges():
        for name, rhs in edge.data.assignments.items():
            if free_names(rhs) & assigned:
                return f"symbol {name} is carried across iterations"
    for name, desc in body.arrays.items():
        states = [s for s in loop.all_states() if any(n.data == name for n in s.data_nodes())]
        if desc.transient and states and (len(states) > 1 or name in move_if_into_loop.upward_exposed_reads(states[0])):
            return f"transient {name} is carried across iterations"
    return None


def map_loop_refusal(state: SDFGState, entry: nodes.MapEntry, loop: LoopRegion) -> str | None:
    """Why the loop cannot move outside the map, or ``None``.

    Map iterations are independent, so running every map iteration per loop iteration keeps each one's order; the
    move is legal when the map body is exactly the loop, its trip count is the same for every map iteration, and
    nothing inside the body carries a value between loop iterations.
    """
    body = map_body(state, entry)
    if body is None or body.sdfg.nodes() != [loop]:
        return f"the body of {entry.map.label} is not exactly the loop {loop.label}."
    scope = state.scope_children()[None]
    if any(not isinstance(n, nodes.AccessNode) for n in scope if n is not entry and n is not state.exit_node(entry)):
        return f"the state of {entry.map.label} holds more than the map; the loop would repeat it."
    bounds = (loop.init_statement, loop.loop_condition, loop.update_statement)
    for name in {str(s) for code in bounds for s in code.get_free_symbols()} - {loop.loop_variable}:
        if name in entry.map.params or name in body.sdfg.arrays or str(body.symbol_mapping.get(name)) != name:
            return f"the loop bound {name} varies across map iterations."
    if any(isinstance(b, (BreakBlock, ContinueBlock, ReturnBlock)) for b in loop.all_control_flow_blocks()):
        return f"{loop.label} leaves an iteration early."
    return carried_across_iterations(loop, body.sdfg)


def interchange_map_loop(state: SDFGState, entry: nodes.MapEntry, loop: LoopRegion) -> None:
    """``map i: for t: B`` -> ``for t: map i: B``: splice the loop body into the map body, then wrap the state."""
    body = map_body(state, entry)
    assert body is not None, "map_loop_refusal checked the body"
    var = loop.loop_variable
    dtype = loop.new_symbols({})[var]
    blocks, edges, start = list(loop.nodes()), list(loop.edges()), loop.start_block
    body.sdfg.remove_node(loop)
    for block in blocks:
        loop.remove_node(block)
        body.sdfg.add_node(block, is_start_block=block is start)
    for edge in edges:
        body.sdfg.add_edge(edge.src, edge.dst, edge.data)
    if var not in body.sdfg.symbols:
        body.sdfg.add_symbol(var, dtype)
    body.symbol_mapping[var] = symbolic.symbol(var, dtype)
    parent = state.parent_graph
    ins, outs = parent.in_edges(state), parent.out_edges(state)
    parent.add_node(loop, is_start_block=parent.start_block is state, ensure_unique_name=True)
    for edge in ins:
        parent.add_edge(edge.src, loop, edge.data)
    for edge in outs:
        parent.add_edge(loop, edge.dst, edge.data)
    parent.remove_node(state)
    loop.add_node(state, is_start_block=True)
    finish(state.sdfg)


def plan_map_loop_interchange(state: SDFGState, entry: nodes.MapEntry, loop: LoopRegion) -> Rewrite | str:
    reason = map_loop_refusal(state, entry, loop)
    if reason is not None:
        return reason
    return Rewrite(
        "MapLoopInterchange", (entry.map.label, loop.label), lambda: interchange_map_loop(state, entry, loop)
    )


# subgraph fission


def drop_uninitialized_inputs(body: dace.SDFG) -> None:
    """Remove every nested-SDFG input that reads a transient no earlier state wrote.

    ``nest_sdfg_subgraph`` makes an input of every container the nest reads, including one it writes first; read
    from outside, such a transient is uninitialized.
    """
    written: set[str] = set()
    for state in in_order(body):
        for nest in [n for n in state.nodes() if isinstance(n, nodes.NestedSDFG)]:
            for edge in state.in_edges(nest):
                if body.arrays[edge.data.data].transient and edge.data.data not in written:
                    state.remove_edge(edge)
                    nest.remove_in_connector(edge.dst_conn)
                    if state.degree(edge.src) == 0:
                        state.remove_node(edge.src)
            written.update(edge.data.data for edge in state.out_edges(nest))


def split_body(body: dace.SDFG, cut: ControlFlowBlock) -> None:
    """Nest the blocks of ``body`` up to ``cut`` and the blocks after it into one nested SDFG each."""
    blocks = in_order(body)
    index = blocks.index(cut)
    for group in (blocks[: index + 1], blocks[index + 1 :]):
        if len(group) == 1 and isinstance(group[0], SDFGState):
            # nest_sdfg_subgraph leaves a lone state in place, and MapFission would split it per component
            group = [group[0], body.add_state_after(group[0])]
        nest_sdfg_subgraph(body, SubgraphView(body, group))
    drop_uninitialized_inputs(body)


def fission_at(state: SDFGState, entry: nodes.MapEntry, cut: ControlFlowBlock) -> bool:
    """Split the map at ``entry`` into two maps, its body before and after ``cut``; ``False`` when refused.

    ``MapFission`` widens every transient the halves share by the map's range, so each keeps its own element.
    """
    body = map_body(state, entry)
    assert body is not None, "plan_subgraph_fission checked the body"
    split_body(body.sdfg, cut)
    sdfg = state.sdfg
    if not MapFission.can_be_applied_to(sdfg, expr_index=1, map_entry=entry, nested_sdfg=body):
        return False
    # MapFission moves the two maps inside the nested SDFG; inlining it leaves them in two states
    MapFission.apply_to(sdfg, expr_index=1, map_entry=entry, nested_sdfg=body)
    InlineMultistateSDFG.apply_to(sdfg, nested_sdfg=body)
    finish(sdfg)
    return True


def plan_subgraph_fission(state: SDFGState, entry: nodes.MapEntry, cut: ControlFlowBlock) -> Rewrite | str:
    """Split the map at ``entry`` after the block ``cut`` of its body: ``map i: A; B`` -> ``map i: A; map i: B``."""
    body = map_body(state, entry)
    if body is None or cut.parent_graph is not body.sdfg:
        return f"{cut.label} is not a top-level block of the body of {entry.map.label}."
    outs = body.sdfg.out_edges(cut)
    if len(outs) != 1 or any(body.sdfg.in_degree(b) > 1 or body.sdfg.out_degree(b) > 1 for b in body.sdfg.nodes()):
        return f"{cut.label} is not followed by more of one straight-line body; name a block before the last."
    if not outs[0].data.is_unconditional() or outs[0].data.assignments:
        return f"the edge after {cut.label} assigns or branches; the halves would lose it."
    # the split nests blocks before MapFission judges it, so it is tried on a copy first
    sdfg = state.sdfg
    twin = detach(sdfg)
    twin_state = list(twin.all_states())[list(sdfg.all_states()).index(state)]
    twin_entry = twin_state.node(state.node_id(entry))
    twin_body = map_body(twin_state, twin_entry)
    assert twin_body is not None, "a deep copy keeps the body"
    twin_cut = twin_body.sdfg.node(body.sdfg.node_id(cut))
    if not fission_at(twin_state, twin_entry, twin_cut):
        return SUBGRAPH_FISSION_REFUSED

    def split() -> None:
        fission_at(state, entry, cut)

    return Rewrite("SubgraphFission", (entry.map.label, cut.label), split)


# enumeration


def map_entries(sdfg: dace.SDFG) -> Iterator[tuple[SDFGState, nodes.MapEntry]]:
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry) and isinstance(state, SDFGState):
            yield state, node


def region_blocks(sdfg: dace.SDFG, kind: type) -> Iterator:
    for region in sdfg.all_control_flow_regions(recursive=True):
        yield from (block for block in region.nodes() if isinstance(block, kind))


def rewrites(plans: Iterator[Rewrite | str]) -> Iterator[Rewrite]:
    return (plan for plan in plans if isinstance(plan, Rewrite))


def loop_fissions(sdfg: dace.SDFG) -> Iterator[Rewrite]:
    return rewrites(plan_loop_fission(loop) for loop in region_blocks(sdfg, LoopRegion))


def ifs_into_loops(sdfg: dace.SDFG) -> Iterator[Rewrite]:
    return rewrites(
        plan_if_into_loop(cond, branch.sink_nodes()[0])
        for cond in region_blocks(sdfg, ConditionalBlock)
        for _, branch in cond.branches[:1]
        if len(branch.sink_nodes()) == 1 and isinstance(branch.sink_nodes()[0], LoopRegion)
    )


def ifs_out_of_loops(sdfg: dace.SDFG) -> Iterator[Rewrite]:
    return rewrites(
        plan_if_out_of_loop(loop, cond)
        for loop in region_blocks(sdfg, LoopRegion)
        for cond in loop.nodes()
        if isinstance(cond, ConditionalBlock)
    )


def map_loop_interchanges(sdfg: dace.SDFG) -> Iterator[Rewrite]:
    return rewrites(
        plan_map_loop_interchange(state, entry, body.sdfg.nodes()[0])
        for state, entry in map_entries(sdfg)
        if (body := map_body(state, entry)) is not None and isinstance(body.sdfg.nodes()[0], LoopRegion)
    )


def subgraph_fissions(sdfg: dace.SDFG) -> Iterator[Rewrite]:
    return rewrites(
        plan_subgraph_fission(state, entry, block)
        for state, entry in map_entries(sdfg)
        if (body := map_body(state, entry)) is not None
        for block in in_order(body.sdfg)[:-1]
    )
