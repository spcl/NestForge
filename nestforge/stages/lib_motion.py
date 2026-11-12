# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``lib-before`` / ``lib-after`` / ``lib-into``: a library call whose memlets are its whole effect (a ``lib_copy``, a
reduce, a BLAS call), or a host ``Sync`` whose effect is the device work it waits for, leaves its state for a state of
its own before or after a block of the same or an enclosing region, or joins an existing state there, when no block it
crosses touches what it reads and writes."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, cast

import dace
from dace import data, dtypes
from dace.libraries.standard.nodes.external_call import ExternalCall
from dace.libraries.standard.nodes.sync import Sync, SyncScope
from dace.sdfg import nodes
from dace.sdfg.state import (
    BreakBlock,
    ContinueBlock,
    ControlFlowBlock,
    ControlFlowRegion,
    LoopRegion,
    ReturnBlock,
    SDFGState,
)
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import is_gpu_relevant_node

from nestforge.ir.introspect import Row

LIB_BEFORE = "lib-before"
LIB_AFTER = "lib-after"
LIB_INTO = "lib-into"
#: The barriers that wait on the host, outside any kernel, for the device work issued before them.
HOST_SYNCS = frozenset({SyncScope.DEVICE, SyncScope.STREAM})


@dataclass(frozen=True, slots=True)
class Effect:
    """What a library call reads and writes (container names), and the symbols its memlets and properties read."""

    reads: frozenset[str]
    writes: frozenset[str]
    symbols: frozenset[str]
    #: A host barrier: it may not cross, nor join ahead of, a block that issues device work.
    device: bool = False

    @property
    def touched(self) -> frozenset[str]:
        return self.reads | self.writes


def effect_refusal(state: SDFGState, node: nodes.LibraryNode) -> str | None:
    """Why ``node``'s memlets are not its whole effect, or ``None`` when they are."""
    sdfg = state.sdfg
    if state.entry_node(node) is not None:
        return f"{node.label} runs inside a doall; only a call at the top of a state moves."
    if isinstance(node, ExternalCall):
        return f"{node.label} is a kernel calling external code, whose effects its memlets do not show."
    if isinstance(node, Sync):
        if node.scope not in HOST_SYNCS or any(not e.data.is_empty() for e in state.all_edges(node)):
            return f"{node.label} is a {node.scope.value} barrier tied to a kernel or a named stream; it stays."
        return None
    if node.has_side_effects(sdfg):
        return (
            f"{node.label} ({type(node).__name__}) has side effects its memlets do not show (a barrier, a "
            "communication, a stateful call); it stays where it is."
        )
    for edge in state.all_edges(node):
        other = edge.src if edge.dst is node else edge.dst
        if edge.data.is_empty():
            continue
        if not isinstance(other, nodes.AccessNode):
            return f"{node.label} is wired to a {type(other).__name__}, not to arrays alone."
        desc = sdfg.arrays[other.data]
        if isinstance(desc, data.View) or (isinstance(desc, data.Array) and desc.may_alias):
            return f"{node.label} touches {other.data}, a view or an array that may alias another."
        if edge.data.wcr is not None:
            return f"{node.label} accumulates into {other.data}; an accumulation is not moved."
    return None


def device_work(state: SDFGState) -> list[nodes.Node]:
    """The nodes of ``state`` that issue or touch device work: kernel launches, GPU copies and fills, GPU arrays (DaCe's
    ``is_gpu_relevant_node``), and calls scheduled on the GPU."""
    return [
        n
        for n in state.nodes()
        if is_gpu_relevant_node(n, state.sdfg, state)
        or (isinstance(n, nodes.LibraryNode) and n.schedule in dtypes.GPU_SCHEDULES)
        or (isinstance(n, nodes.MapEntry) and n.map.schedule in dtypes.GPU_SCHEDULES)
    ]


def block_device_work(block: ControlFlowBlock) -> bool:
    states = [block] if isinstance(block, SDFGState) else list(cast(ControlFlowRegion, block).states())
    return any(device_work(state) for state in states)


def device_containers(sdfg: dace.SDFG) -> frozenset[str]:
    """The containers device work reads or writes anywhere in ``sdfg``: what a host ``Sync`` makes safe to touch."""
    names: set[str] = set()
    for state in sdfg.states():
        for node in device_work(state):
            ends = [node, state.exit_node(node)] if isinstance(node, nodes.EntryNode) else [node]
            for end in ends:
                for edge in state.all_edges(end):
                    names |= {n.data for n in (edge.src, edge.dst) if isinstance(n, nodes.AccessNode)}
    return frozenset(names)


def effect_of(state: SDFGState, node: nodes.LibraryNode) -> Effect:
    if isinstance(node, Sync):  # it reads and writes, in effect, what the device work it waits for touches
        touched = device_containers(state.sdfg)
        return Effect(touched, touched, frozenset(), device=True)
    reads = frozenset(str(e.data.data) for e in state.in_edges(node) if not e.data.is_empty())
    writes = frozenset(str(e.data.data) for e in state.out_edges(node) if not e.data.is_empty())
    symbols = set(node.free_symbols)
    for edge in state.all_edges(node):
        symbols |= {str(s) for s in edge.data.free_symbols}
    return Effect(reads, writes, frozenset(symbols - state.sdfg.arrays.keys()))


def reaching(graph: Any, start: Any, forward: bool) -> set[Any]:
    """The nodes reachable from ``start`` (exclusive) along the graph's edges, or against them."""
    seen: set[Any] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        nxt = graph.successors(node) if forward else graph.predecessors(node)
        for other in nxt:
            if other not in seen:
                seen.add(other)
                stack.append(other)
    return seen


def in_state_refusal(state: SDFGState, node: nodes.LibraryNode, effect: Effect, earlier: bool) -> str | None:
    """Whether ``node`` may leave ``state`` toward an earlier (``earlier``) or a later point: nothing it waits for, or
    nothing waiting for it, stays behind, and every other access that conflicts with it stays on the same side."""
    ordered = state.in_edges(node) if earlier else state.out_edges(node)
    if any(e.data.is_empty() for e in ordered):
        side = "after" if earlier else "before"
        return f"{node.label} is ordered {side} other nodes of {state.label} by empty edges."
    inputs = [e.src for e in state.in_edges(node) if not e.data.is_empty()]
    outputs = [e.dst for e in state.out_edges(node) if not e.data.is_empty()]
    if any(state.in_degree(n) != 1 for n in outputs):
        return f"something else in {state.label} writes the access node {node.label} writes."
    if earlier and any(state.in_degree(n) for n in inputs):
        return f"what {node.label} reads is computed (or ordered) in {state.label} first."
    if not earlier and any(state.out_degree(n) for n in outputs):
        return f"what {node.label} writes is read (or ordered) in {state.label} after it."
    side = reaching(state, node, forward=earlier)
    own = {node, *inputs, *outputs}
    for other in state.data_nodes():
        if other in own or other in side:
            continue
        writes = any(not e.data.is_empty() for e in state.in_edges(other))
        reads = any(not e.data.is_empty() for e in state.out_edges(other))
        if (writes and other.data in effect.touched) or (reads and other.data in effect.writes):
            return f"{state.label} accesses {other.data} out of order with {node.label}."
    return None


def accessed_beside(state: SDFGState, node: nodes.LibraryNode, names: frozenset[str], writes: bool) -> str | None:
    """A container of ``names`` that a node of ``state`` other than ``node``'s outputs writes (``writes``) or reads."""
    outputs = {e.dst for e in state.out_edges(node)}
    for other in state.data_nodes():
        edges = state.in_edges(other) if writes else state.out_edges(other)
        if other.data in names and other not in outputs and any(not e.data.is_empty() for e in edges):
            return other.data
    return None


def loop_state_refusal(
    state: SDFGState, node: nodes.LibraryNode, effect: Effect, loop: LoopRegion, earlier: bool
) -> str | None:
    """What the call's own state does every iteration of ``loop``: it may not change what the call touches, nor, when
    the call sinks after ``loop``, read what it writes."""
    written = accessed_beside(state, node, effect.touched, writes=True)
    if written is not None:
        return f"{state.label} in {loop.label} writes {written}, which the call touches."
    read = None if earlier else accessed_beside(state, node, effect.writes, writes=False)
    if read is not None:
        return f"{state.label} reads {read} every iteration; after {loop.label}, those reads would see the old value."
    return None


def interstate_refusal(edge_data: dace.InterstateEdge, effect: Effect, where: str) -> str | None:
    read = edge_data.read_symbols()
    if read & effect.writes:
        return f"a transition in {where} reads {sorted(read & effect.writes)[0]}, which the call writes."
    assigned = sorted(set(edge_data.assignments) & effect.symbols)
    if assigned:
        return f"a transition in {where} assigns {assigned[0]}, which the call reads."
    return None


def assigned_symbols(block: ControlFlowBlock) -> set[str]:
    """The symbols ``block`` assigns: on its transitions and as loop variables, nested regions included."""
    if not isinstance(block, ControlFlowRegion):
        return set()
    names = {str(k) for e in block.all_interstate_edges() for k in e.data.assignments}
    loops = [block, *block.all_control_flow_blocks(recursive=True)]
    return names | {b.loop_variable for b in loops if isinstance(b, LoopRegion) and b.loop_variable}


def crossed_refusal(block: ControlFlowBlock, effect: Effect) -> str | None:
    if effect.device and block_device_work(block):
        return f"{block.label} issues device work, which the barrier would no longer wait for (or wait for too early)."
    reads, writes = block.read_and_write_sets()
    if writes & effect.touched:
        return f"{block.label} writes {sorted(writes & effect.touched)[0]}, which the call touches."
    if reads & effect.writes:
        return f"{block.label} reads {sorted(reads & effect.writes)[0]}, which the call writes."
    if assigned_symbols(block) & effect.symbols:
        return f"{block.label} assigns {sorted(assigned_symbols(block) & effect.symbols)[0]}, which the call reads."
    return None


def loop_refusal(loop: LoopRegion, child: ControlFlowBlock, effect: Effect, earlier: bool) -> str | None:
    """Whether the call, in ``child`` of ``loop``, runs once before (``earlier``) or after ``loop`` with the effect of
    every iteration: it reads nothing the loop changes and nothing reads its writes ahead of it."""
    if not loop_analysis.provably_runs(loop, 1, loop.sdfg.facts()):
        return (
            f"{loop.label} is not proven to run once (its trip count may be zero); the call would run when it did not."
        )
    if effect.reads & effect.writes:
        return "the call reads what it writes, so running it once is not running it every iteration."
    if any(isinstance(b, (BreakBlock, ContinueBlock, ReturnBlock)) for b in loop.all_control_flow_blocks(True)):
        return f"{loop.label} breaks, continues or returns, so the call does not run every iteration."
    if effect.symbols & ({loop.loop_variable} | assigned_symbols(loop)):
        return f"the call reads a symbol {loop.label} changes."
    meta = set().union(*(c.get_free_symbols() for c in loop.get_meta_codeblocks()))
    if meta & effect.writes:
        return f"{loop.label}'s bounds read what the call writes."
    after = reaching(loop, child, forward=True)
    for block in loop.nodes():
        if block is child:
            continue
        reads, writes = block.read_and_write_sets()
        if writes & effect.touched:
            return f"{block.label} in {loop.label} writes {sorted(writes & effect.touched)[0]}, which the call touches."
        if reads & effect.writes and not (earlier and block in after):
            return f"{block.label} in {loop.label} reads {sorted(reads & effect.writes)[0]}, which the call writes."
    for edge in loop.edges():
        if (edge.data.read_symbols() & effect.writes) and not (earlier and (edge.src is child or edge.src in after)):
            return f"a transition in {loop.label} reads what the call writes."
    return None


def exit_chain(state: SDFGState, target: ControlFlowBlock) -> tuple[list[LoopRegion], ControlFlowBlock] | str:
    """The loops the call leaves to reach ``target``'s region, innermost first, and the block of that region that
    holds it; or why ``target``'s region does not enclose the call."""
    if target.sdfg is not state.sdfg:
        return f"{target.label} is in another SDFG than the call."
    region, anchor, loops = target.parent_graph, state, []
    while anchor.parent_graph is not region:
        parent = anchor.parent_graph
        if parent is None or isinstance(parent, dace.SDFG):
            return f"{target.label} is not in a region enclosing the call; a call moves only out, never into a block."
        if not isinstance(parent, LoopRegion):
            return f"the call would leave {parent.label}, which runs under a condition; it would run speculatively."
        loops.append(parent)
        anchor = parent
    return loops, anchor


def plan_lib_motion(row: Row, target_row: Row, place: str) -> Callable[[], object] | str:
    """The move of the library call ``row`` into a fresh state before or after (``place``) ``target_row``'s block,
    or ``into`` that state, or why it is illegal; nothing is mutated."""
    node, state = row
    target = target_row[0]
    assert isinstance(state, SDFGState) and isinstance(node, nodes.LibraryNode), "a library call row"
    why = effect_refusal(state, node)
    if why is not None:
        return why
    if target is state:
        return f"{node.label} is in {state.label} already; name another block."
    if place == "into" and not isinstance(target, SDFGState):
        return f"{target.label} is not a state; a call joins only a state."
    chain = exit_chain(state, target)
    if isinstance(chain, str):
        return chain
    loops, anchor = chain
    region = target.parent_graph
    assert region is not None, "exit_chain found the target's region"
    earlier = target in reaching(region, anchor, forward=False) or (target is anchor and place == "before")
    if not earlier and target is not anchor and target not in reaching(region, anchor, forward=True):
        return f"{target.label} is neither before nor after the call on any path."
    effect = effect_of(state, node)
    why = path_refusal(state, node, effect, loops, earlier)
    first, last = (target, anchor) if earlier else (anchor, target)
    span = (reaching(region, first, forward=True) & reaching(region, last, forward=False)) | {first, last}
    # the target is crossed when the call lands on its far side
    crossed = span - {anchor} - ({target} if place == "into" or earlier == (place == "after") else set())
    for block in crossed:
        why = why or crossed_refusal(block, effect)
    for edge in region.edges():
        if edge.src in span and edge.dst in span:
            why = why or interstate_refusal(edge.data, effect, region.label)
    if place != "into":
        return why or (lambda: move_call(state, node, target, place == "after"))
    joint = why or joint_of(cast(SDFGState, target), node, effect, earlier)
    if isinstance(joint, str):
        return joint
    return lambda: join_call(state, node, joint)


def path_refusal(
    state: SDFGState, node: nodes.LibraryNode, effect: Effect, loops: list[LoopRegion], earlier: bool
) -> str | None:
    """Whether the call may leave its state and the ``loops`` around it toward an earlier or a later point."""
    why = in_state_refusal(state, node, effect, earlier)
    if why is None and loops:
        why = loop_state_refusal(state, node, effect, loops[0], earlier)
    child: ControlFlowBlock = state
    for loop in loops:
        why = why or loop_refusal(loop, child, effect, earlier)
        child = loop
    return why


@dataclass(frozen=True, slots=True)
class Joint:
    """Where a call joins a state: the access nodes it reads from and writes to there, by container (new ones for the
    rest), and the nodes ordered before and after it by empty edges."""

    state: SDFGState
    sources: dict[str, nodes.AccessNode]
    sinks: dict[str, nodes.AccessNode]
    before: list[nodes.Node]
    after: list[nodes.Node]


def written(state: SDFGState, access: nodes.AccessNode) -> bool:
    return any(not e.data.is_empty() for e in state.in_edges(access))


def read(state: SDFGState, access: nodes.AccessNode) -> bool:
    return any(not e.data.is_empty() for e in state.out_edges(access))


def last_writer(state: SDFGState, name: str) -> nodes.AccessNode | None | str:
    """The access node of ``name`` that every other writer of it in ``state`` comes before; ``None`` when nothing
    writes it; why not, when two writers are unordered."""
    writers = [a for a in state.data_nodes() if a.data == name and written(state, a)]
    last = [w for w in writers if not reaching(state, w, forward=True) & set(writers)]
    if len(last) > 1 or any(w is not last[0] and last[0] not in reaching(state, w, True) for w in writers):
        return f"{state.label} writes {name} twice with no order between the writes."
    return last[0] if last else None


def producers(state: SDFGState, access: nodes.AccessNode) -> list[nodes.Node]:
    """The nodes that write ``access``: a map is ordered by its entry."""
    return [
        state.entry_node(e.src) if isinstance(e.src, nodes.ExitNode) else e.src
        for e in state.in_edges(access)
        if not e.data.is_empty()
    ]


def joint_of(target: SDFGState, node: nodes.LibraryNode, effect: Effect, earlier: bool) -> Joint | str:
    """How the call joins ``target``: after every access there that conflicts with it, when it comes from a later
    point (``earlier``), else before every one, so no cycle can form; refused for an accumulation or unordered
    writers."""
    clashing = [
        a
        for a in target.data_nodes()
        if (written(target, a) and a.data in effect.touched) or (read(target, a) and a.data in effect.writes)
    ]
    for access in clashing:
        if any(e.data.wcr is not None for e in target.in_edges(access)):
            return f"{target.label} accumulates into {access.data}; the call does not join an accumulation."
    for name in sorted({a.data for a in clashing if written(target, a)}):
        last = last_writer(target, name)
        if isinstance(last, str):
            return last
    if effect.device and not earlier and device_work(target):
        return f"{target.label} issues device work the barrier would come before; it joins only after such work."
    if earlier:
        sources = {}
        for name in () if effect.device else effect.reads:  # a barrier has no input to wire
            last = last_writer(target, name)
            if isinstance(last, nodes.AccessNode):
                sources[name] = last
        work = [target.exit_node(n) if isinstance(n, nodes.EntryNode) else n for n in device_work(target)]
        before = [a for a in clashing if a not in sources.values()] + (work if effect.device else [])
        return Joint(target, sources, {}, before, [])
    sinks = {}
    for name in effect.writes:
        fresh = [a for a in target.data_nodes() if a.data == name and target.in_degree(a) == 0]
        if fresh:
            sinks[name] = fresh[0]
    after: list[nodes.Node] = []
    for access in clashing:
        if access in sinks.values():
            continue
        after += producers(target, access) if written(target, access) else [access]
    return Joint(target, {}, sinks, [], after)


def join_call(state: SDFGState, node: nodes.LibraryNode, joint: Joint) -> SDFGState:
    """Take ``node`` out of ``state`` into ``joint.state``, wired and ordered as ``joint`` says."""
    target = joint.state
    edges = detach(state, node)
    target.add_node(node)
    for edge in edges:
        if edge.dst is node:
            source = joint.sources.get(edge.src.data) or target.add_access(edge.src.data)
            target.add_edge(source, edge.src_conn, node, edge.dst_conn, copy.deepcopy(edge.data))
        else:
            sink = joint.sinks.get(edge.dst.data) or target.add_access(edge.dst.data)
            target.add_edge(node, edge.src_conn, sink, edge.dst_conn, copy.deepcopy(edge.data))
    for other in dict.fromkeys(joint.before):
        target.add_nedge(other, node, dace.Memlet())
    for other in dict.fromkeys(joint.after):
        target.add_nedge(node, other, dace.Memlet())
    return target


def detach(state: SDFGState, node: nodes.LibraryNode) -> list[Any]:
    """Remove ``node`` from ``state`` with the access nodes only it used; returns its data edges."""
    edges = list(state.all_edges(node))
    state.remove_node(node)
    for edge in edges:
        for end in (edge.src, edge.dst):
            if (
                end is not node
                and end in state.nodes()
                and state.degree(end) == 0
                and isinstance(end, nodes.AccessNode)
            ):
                state.remove_node(end)
    return [e for e in edges if not e.data.is_empty()]


def move_call(state: SDFGState, node: nodes.LibraryNode, target: ControlFlowBlock, after: bool) -> SDFGState:
    """Take ``node`` and its access nodes out of ``state`` into a new state next to ``target``."""
    region = target.parent_graph
    label = f"{node.label}_moved"
    if after:
        moved = region.add_state_after(target, label)
    else:
        moved = region.add_state_before(target, label, is_start_block=region.start_block is target)
    return join_call(state, node, Joint(moved, {}, {}, [], []))


def movable_rows(sdfg: dace.SDFG, states_only: bool = False) -> Iterator[tuple[Row, Row]]:
    """Every top-level library call paired with every other block (or state) of its region and of each loop
    around it."""
    for state in sdfg.states():
        calls = [n for n in state.nodes() if isinstance(n, nodes.LibraryNode) and state.entry_node(n) is None]
        if not calls:
            continue
        regions, block = [], state
        while isinstance(block.parent_graph, ControlFlowRegion):
            regions.append(block.parent_graph)
            if not isinstance(block.parent_graph, LoopRegion):
                break
            block = block.parent_graph
        for call in calls:
            for region in regions:
                blocks = [b for b in region.nodes() if b is not state]
                if states_only:
                    blocks = [b for b in blocks if isinstance(b, SDFGState)]
                yield from (((call, state), (b, None)) for b in blocks)
