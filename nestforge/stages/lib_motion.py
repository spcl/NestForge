# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``lib-before`` / ``lib-after``: a library call whose memlets are its whole effect (a ``lib_copy``, a reduce, a
BLAS call) leaves its state for a state of its own before or after a block of the same or an enclosing region,
when no block it crosses touches what it reads and writes."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import dace
from dace import data
from dace.libraries.standard.nodes.external_call import ExternalCall
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

from nestforge.ir.introspect import Row

LIB_BEFORE = "lib-before"
LIB_AFTER = "lib-after"


@dataclass(frozen=True, slots=True)
class Effect:
    """What a library call reads and writes (container names), and the symbols its memlets and properties read."""

    reads: frozenset[str]
    writes: frozenset[str]
    symbols: frozenset[str]

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


def effect_of(state: SDFGState, node: nodes.LibraryNode) -> Effect:
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
    if any(e.data.is_empty() for e in state.all_edges(node)):
        return f"{node.label} is ordered against other nodes of {state.label} by empty edges."
    inputs = [e.src for e in state.in_edges(node)]
    outputs = [e.dst for e in state.out_edges(node)]
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


def plan_lib_motion(row: Row, target_row: Row, after: bool) -> Callable[[], object] | str:
    """The move of the library call ``row`` into a fresh state before (or ``after``) ``target_row``'s block, or why
    it is illegal; nothing is mutated."""
    node, state = row
    target = target_row[0]
    assert isinstance(state, SDFGState) and isinstance(node, nodes.LibraryNode), "a library call row"
    why = effect_refusal(state, node)
    if why is not None:
        return why
    if target is state:
        return f"{node.label} is in {state.label} already; name another block."
    chain = exit_chain(state, target)
    if isinstance(chain, str):
        return chain
    loops, anchor = chain
    region = target.parent_graph
    assert region is not None, "exit_chain found the target's region"
    behind = reaching(region, anchor, forward=False)
    earlier = target in behind or (target is anchor and not after)
    if not earlier and target is not anchor and target not in reaching(region, anchor, forward=True):
        return f"{target.label} is neither before nor after the call on any path."
    effect = effect_of(state, node)
    why = in_state_refusal(state, node, effect, earlier)
    if why is None and loops:
        why = loop_state_refusal(state, node, effect, loops[0], earlier)
    child: ControlFlowBlock = state
    for loop in loops:
        why = why or loop_refusal(loop, child, effect, earlier)
        child = loop
    if why is not None:
        return why
    first, last = (target, anchor) if earlier else (anchor, target)
    span = (reaching(region, first, forward=True) & reaching(region, last, forward=False)) | {first, last}
    crossed = span - {anchor} - ({target} if earlier == after else set())
    for block in crossed:
        why = why or crossed_refusal(block, effect)
    for edge in region.edges():
        if edge.src in span and edge.dst in span:
            why = why or interstate_refusal(edge.data, effect, region.label)
    if why is not None:
        return why
    return lambda: move_call(state, node, target, after)


def move_call(state: SDFGState, node: nodes.LibraryNode, target: ControlFlowBlock, after: bool) -> SDFGState:
    """Take ``node`` and its access nodes out of ``state`` into a new state next to ``target``."""
    region = target.parent_graph
    label = f"{node.label}_moved"
    if after:
        moved = region.add_state_after(target, label)
    else:
        moved = region.add_state_before(target, label, is_start_block=region.start_block is target)
    edges = list(state.all_edges(node))
    state.remove_node(node)
    moved.add_node(node)
    for edge in edges:
        if edge.dst is node:
            moved.add_edge(
                moved.add_access(edge.src.data), edge.src_conn, node, edge.dst_conn, copy.deepcopy(edge.data)
            )
        else:
            moved.add_edge(
                node, edge.src_conn, moved.add_access(edge.dst.data), edge.dst_conn, copy.deepcopy(edge.data)
            )
    for edge in edges:
        for end in (edge.src, edge.dst):
            if end is not node and end in state.nodes() and state.degree(end) == 0:
                state.remove_node(end)
    return moved


def movable_rows(sdfg: dace.SDFG) -> Iterator[tuple[Row, Row]]:
    """Every top-level library call paired with every other block of its region and of each loop around it."""
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
                yield from (((call, state), (b, None)) for b in region.nodes() if b is not state)
