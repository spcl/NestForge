# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scheduling moves on one named region, through the per-region entry points of DaCe's region transformations:
``LoopFission``, ``MoveIfIntoLoop``, ``MoveLoopInvariantIfUp``, ``MapLoopInterchange`` and ``SubgraphFission``."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, LoopRegion, SDFGState
from dace.sdfg.utils import set_nested_sdfg_parent_references
from dace.transformation.interstate import MapLoopInterchange, SubgraphFission
from dace.transformation.interstate.move_loop_invariant_if_up import MoveLoopInvariantIfUp, match_loop
from dace.transformation.passes.loop_fission import LoopFission
from dace.transformation.passes.move_if_into_loop import MoveIfIntoLoop, match_guard

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
    "blocked by SubgraphFission: the body branches, the edge after the cut assigns or branches, or MapFission "
    "refuses (a half holds a conditional, or an interstate assignment reads a map parameter)."
)


@dataclass(frozen=True, slots=True)
class Rewrite:
    """A legal move on the regions ``labels`` name; ``commit`` applies it and ``name`` says what ran."""

    name: str
    labels: tuple[str, ...]
    commit: Callable[[], None]


def finish(sdfg: dace.SDFG) -> None:
    """Restore the parent links and region ids a control-flow rewrite of ``sdfg`` leaves stale."""
    root = sdfg.root_sdfg
    set_nested_sdfg_parent_references(root)
    root.reset_cfg_list()


def plan_loop_fission(loop: LoopRegion) -> Rewrite | str:
    """Split ``loop`` into one loop per independent statement group, as ``LoopFission`` does program-wide."""
    if not LoopFission.can_fission(loop):
        return LOOP_FISSION_REFUSED
    sdfg = loop.sdfg

    def split() -> None:
        LoopFission.fission(loop)
        finish(sdfg)

    return Rewrite("LoopFission", (loop.label,), split)


def plan_if_into_loop(cond: ConditionalBlock, loop: LoopRegion) -> Rewrite | str:
    """Push the guard ``cond`` into the ``loop`` its branch ends in: ``if c: for i: B`` -> ``for i: if c: B``."""
    match = match_guard(cond)
    if match is None or loop.parent_graph is not match[2] or match[2].out_degree(loop):
        return IF_INTO_LOOP_REFUSED
    sdfg = loop.sdfg

    def push() -> None:
        MoveIfIntoLoop.push(cond)
        finish(sdfg)

    return Rewrite("MoveIfIntoLoop", (cond.label, loop.label), push)


def plan_if_out_of_loop(loop: LoopRegion, cond: ConditionalBlock) -> Rewrite | str:
    """Hoist the loop-invariant guard ``cond`` out of ``loop``: ``for i: if c: B`` -> ``if c: for i: B``."""
    sdfg = loop.sdfg
    match = match_loop(sdfg, loop)
    if match is None or match[1] is not cond:
        return IF_OUT_OF_LOOP_REFUSED

    def hoist() -> None:
        MoveLoopInvariantIfUp.hoist(loop)
        finish(sdfg)

    return Rewrite("MoveLoopInvariantIfUp", (loop.label, cond.label), hoist)


def map_body(state: SDFGState, entry: nodes.MapEntry) -> nodes.NestedSDFG | None:
    """The nested SDFG that is the whole body of the map at ``entry``, if it is one."""
    targets = {edge.dst for edge in state.out_edges(entry)}
    body = next(iter(targets))
    return body if len(targets) == 1 and isinstance(body, nodes.NestedSDFG) else None


def plan_map_loop_interchange(state: SDFGState, entry: nodes.MapEntry, loop: LoopRegion) -> Rewrite | str:
    """Move ``loop``, the whole body of the map at ``entry``, outside it: ``map i: for t: B`` -> ``for t: map i: B``."""
    body = map_body(state, entry)
    if body is None or body.sdfg.nodes() != [loop]:
        return f"the body of {entry.map.label} is not exactly the loop {loop.label}."
    xform = MapLoopInterchange()
    pattern = {MapLoopInterchange.map_entry: state.node_id(entry), MapLoopInterchange.nested_sdfg: state.node_id(body)}
    xform.setup_match(state.sdfg, state.parent_graph.cfg_id, state.block_id, pattern, 0)
    reason = xform.refusal(state)
    if reason is not None:
        return f"blocked by MapLoopInterchange: {reason}."
    return Rewrite("MapLoopInterchange", (entry.map.label, loop.label), lambda: xform.apply(state, state.sdfg))


def plan_subgraph_fission(state: SDFGState, entry: nodes.MapEntry, cut: ControlFlowBlock) -> Rewrite | str:
    """Split the map at ``entry`` after the block ``cut`` of its body: ``map i: A; B`` -> ``map i: A; map i: B``."""
    body = map_body(state, entry)
    if body is None or cut.parent_graph is not body.sdfg:
        return f"{cut.label} is not a top-level block of the body of {entry.map.label}."
    if not body.sdfg.out_edges(cut):
        return f"{cut.label} is the last block of the body of {entry.map.label}; name a block before the last."
    where = {"map_entry": entry, "nested_sdfg": body}
    options = {"cut": cut.label}
    if not SubgraphFission.can_be_applied_to(state.sdfg, options=options, **where):
        return SUBGRAPH_FISSION_REFUSED

    def split() -> None:
        SubgraphFission.apply_to(state.sdfg, options=options, annotate=False, save=False, **where)

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
        plan_map_loop_interchange(state, entry, loop)
        for state, entry in map_entries(sdfg)
        if (body := map_body(state, entry)) is not None and isinstance(loop := body.sdfg.nodes()[0], LoopRegion)
    )


def subgraph_fissions(sdfg: dace.SDFG) -> Iterator[Rewrite]:
    return rewrites(
        plan_subgraph_fission(state, entry, block)
        for state, entry in map_entries(sdfg)
        if (body := map_body(state, entry)) is not None
        for block in in_order(body.sdfg)[:-1]
    )
