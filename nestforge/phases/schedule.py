# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 1: the fusion, fission and interchange moves that shape kernels, their legality, and scope metrics."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from itertools import chain, product
from typing import Any, cast

import dace
import sympy
from dace.sdfg import nodes
from dace.sdfg.performance_evaluation import total_volume, work_depth
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, LoopRegion, SDFGState
from dace.transformation.dataflow.map_fission import MapFission
from dace.transformation.dataflow.map_fusion_horizontal import MapFusionHorizontal
from dace.transformation.dataflow.map_fusion_vertical import MapFusionVertical
from dace.transformation.dataflow.map_interchange import MapInterchange
from dace.transformation.interstate.loop_fusion import LoopFusion as FuseLoops
from dace.transformation.interstate.move_loop_into_map import MoveLoopIntoMap
from dace.transformation.interstate.state_fusion import StateFusion
from dace.transformation.passes.canonicalize import canonicalize, stage_labels
from dace.transformation.passes.canonicalize.split_statements import SplitStatements
from dace.transformation.passes.loop_fission import LoopFission

from nestforge.ir.extract import detach, detached_twin, extract_cfg_nest, extract_map_nest, find_state_of_node
from nestforge.ir.names import inline_top_level_nsdfgs
from nestforge.ir.introspect import Row
from nestforge.phases.normalize import FUSE_STAGE, Targets
from nestforge.phases.region_moves import (
    Rewrite,
    ifs_into_loops,
    ifs_out_of_loops,
    loop_fissions,
    map_loop_interchanges,
    plan_if_into_loop,
    plan_if_out_of_loop,
    plan_loop_fission,
    plan_map_loop_interchange,
    plan_subgraph_fission,
    subgraph_fissions,
)


#: A pattern transformation's ``PatternNode`` names and the nodes or blocks they match.
Where = dict[str, nodes.Node | ControlFlowBlock]


def map_exit(state: SDFGState, entry: nodes.MapEntry) -> nodes.MapExit:
    exit_node = state.exit_node(entry)
    assert isinstance(exit_node, nodes.MapExit), f"{entry} has no exit"
    return exit_node


@dataclass(slots=True)
class FusionMove:
    """One legal pattern transformation (a fusion or an interchange); ``where`` maps the transformation's
    ``PatternNode`` names to the matched nodes, as ``apply_to`` takes them."""

    kind: str
    where: Where
    xform: type = field(repr=False)
    sdfg: dace.SDFG = field(repr=False)  # the SDFG owning the matched nodes, nested or not

    def label(self) -> str:
        return f"{self.kind}({', '.join(str(n) for n in self.where.values())})"


def pattern_move(kind: str, xform: type, sdfg: dace.SDFG, where: Where) -> FusionMove | None:
    """The move ``xform`` makes at ``where`` when DaCe's own check accepts it, else ``None``."""
    return FusionMove(kind, where, xform, sdfg) if xform.can_be_applied_to(sdfg, **where) else None


def every_state(sdfg: dace.SDFG) -> Iterator[SDFGState]:
    """Every state of ``sdfg`` and of the SDFGs nested in it; ``state.sdfg`` is the one that owns it."""
    for owner in sdfg.all_sdfgs_recursive():
        yield from owner.all_states()


def loop_fusion_moves(sdfg: dace.SDFG) -> Iterator[FusionMove]:
    """Adjacent loop pairs ``LoopFusion`` accepts: one sequencing edge, same range, no carried dependency."""
    for cfg in sdfg.all_control_flow_regions(recursive=True):
        for first in cfg.nodes():
            if not isinstance(first, LoopRegion):
                continue
            out = cfg.out_edges(first)
            if len(out) != 1:
                continue
            second = out[0].dst
            if isinstance(second, LoopRegion) and second is not first:
                move = pattern_move("fuse-loops", FuseLoops, first.sdfg, {"first": first, "second": second})
                if move is not None:
                    yield move


def vertical_map_moves(sdfg: dace.SDFG) -> Iterator[FusionMove]:
    """Producer-consumer map pairs through a transient that ``MapFusionVertical`` accepts. A non-transient
    intermediate is a program output, which fusing would drop."""
    for state in every_state(sdfg):
        for node in state.data_nodes():
            if state.sdfg.arrays[node.data].transient:
                yield from vertical_moves_through(state, node)


def vertical_moves_through(state: SDFGState, node: nodes.AccessNode) -> Iterator[FusionMove]:
    producers = [e.src for e in state.in_edges(node) if isinstance(e.src, nodes.MapExit)]
    consumers = [e.dst for e in state.out_edges(node) if isinstance(e.dst, nodes.MapEntry)]
    yield from filter(None, (vertical_move(state.sdfg, x, node, entry) for x, entry in product(producers, consumers)))


def vertical_move(
    sdfg: dace.SDFG, exit_node: nodes.MapExit, array: nodes.AccessNode, entry: nodes.MapEntry
) -> FusionMove | None:
    where: Where = {"first_map_exit": exit_node, "array": array, "second_map_entry": entry}
    return pattern_move("fuse-map-vertical", MapFusionVertical, sdfg, where)


def horizontal_move(sdfg: dace.SDFG, first: nodes.MapEntry, second: nodes.MapEntry) -> FusionMove | None:
    where: Where = {"first_parallel_map_entry": first, "second_parallel_map_entry": second}
    return pattern_move("fuse-map-horizontal", MapFusionHorizontal, sdfg, where)


def map_interchange_move(sdfg: dace.SDFG, outer: nodes.MapEntry, inner: nodes.MapEntry) -> FusionMove | None:
    where: Where = {"outer_map_entry": outer, "inner_map_entry": inner}
    return pattern_move("interchange-map-map", MapInterchange, sdfg, where)


def loop_into_map_move(loop: LoopRegion) -> FusionMove | None:
    return pattern_move("interchange-loop-map", MoveLoopIntoMap, loop.sdfg, {"loop": loop})


def horizontal_map_moves(sdfg: dace.SDFG) -> Iterator[FusionMove]:
    """Sibling map pairs of one scope that ``MapFusionHorizontal`` accepts."""
    for state in every_state(sdfg):
        scope = state.scope_dict()
        entries = [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]
        for i, first in enumerate(entries):
            pairs = (horizontal_move(state.sdfg, first, b) for b in entries[i + 1 :] if scope[first] is scope[b])
            yield from filter(None, pairs)


def map_fusion_moves(sdfg: dace.SDFG) -> Iterator[FusionMove]:
    return chain(vertical_map_moves(sdfg), horizontal_map_moves(sdfg))


def enumerate_fusions(sdfg: dace.SDFG) -> list[FusionMove]:
    """Every legal fusion right now: loops, then vertical, then horizontal map pairs. Applying one invalidates
    the node references of the rest."""
    return list(chain(loop_fusion_moves(sdfg), map_fusion_moves(sdfg)))


def first_fusion(sdfg: dace.SDFG) -> FusionMove | None:
    """``enumerate_fusions(sdfg)[0]`` without scanning past it."""
    return next(chain(loop_fusion_moves(sdfg), map_fusion_moves(sdfg)), None)


def apply_fusion(move: FusionMove) -> None:
    """Commit a move from a current enumeration; the transformation re-verifies before applying."""
    move.xform.apply_to(move.sdfg, verify=True, annotate=False, save=False, **move.where)


STATE_BARRIER = (
    "nests are in different states, a control-flow dependency map fusion never crosses; merge the enclosing "
    "regions first (fuse_regions)."
)


def can_fuse(sdfg: dace.SDFG, first: object, second: object) -> str:
    """``"yes"`` when a fusion move for the pair exists, else a one-line reason."""
    if isinstance(first, LoopRegion) and isinstance(second, LoopRegion):
        return fuse_loops_reason(sdfg, first, second)
    if isinstance(first, nodes.MapEntry) and isinstance(second, nodes.MapEntry):
        return fuse_maps_reason(sdfg, first, second)
    return "cannot fuse a map-nest with a loop-nest; bring both to the same granularity first."


def fuse_loops_reason(sdfg: dace.SDFG, first: LoopRegion, second: LoopRegion) -> str:
    if first.parent_graph is not second.parent_graph:
        return "loops are in different control-flow regions; fuse the enclosing loops first."
    out = first.parent_graph.out_edges(first)
    if len(out) != 1 or out[0].dst is not second:
        return "loops are not adjacent: exactly one sequencing edge must lead from the first to the second."
    if FuseLoops.can_be_applied_to(sdfg, first=first, second=second):
        return "yes"
    return "blocked by FuseLoops: different iteration ranges, or a loop-carried dependency between the two."


def fuse_maps_reason(sdfg: dace.SDFG, first: nodes.MapEntry, second: nodes.MapEntry) -> str:
    state = find_state_of_node(sdfg, first)
    if find_state_of_node(sdfg, second) is not state:
        return STATE_BARRIER
    plan = plan_map_pair(sdfg, state, first, second)
    return plan if isinstance(plan, str) else "yes"


def intermediates(state: SDFGState, exit_node: nodes.MapExit, consumer: nodes.MapEntry) -> list[nodes.AccessNode]:
    """The access nodes ``exit_node`` writes and ``consumer`` reads: what a vertical fusion could fuse through."""
    written = dict.fromkeys(e.dst for e in state.out_edges(exit_node) if isinstance(e.dst, nodes.AccessNode))
    return [arr for arr in written if any(oe.dst is consumer for oe in state.out_edges(arr))]


def plan_map_pair(sdfg: dace.SDFG, state: SDFGState, first: nodes.MapEntry, second: nodes.MapEntry) -> FusionMove | str:
    """The fusion two maps of ``state`` admit, or why none: vertical through a transient in either data-flow order,
    else horizontal when no data links them."""
    # every intermediate counts: one fusable transient is a move, as vertical_map_moves offers it
    reasons: list[str] = []
    for producer, consumer in ((first, second), (second, first)):
        exit_node = map_exit(state, producer)
        for arr in intermediates(state, exit_node, consumer):
            if not sdfg.arrays[arr.data].transient:
                reasons.append(f"intermediate '{arr.data}' is a live output (non-transient); fusing would drop it")
                continue
            move = vertical_move(sdfg, exit_node, arr, consumer)
            if move is not None:
                return move
            reasons.append(f"blocked by MapFusionVertical on '{arr.data}': shape or dependency mismatch")
    if reasons:
        return "; ".join(reasons) + "."
    if state.scope_dict()[first] is not state.scope_dict()[second]:
        return "maps are in different scopes with no shared data; not a fusion pair."
    move = horizontal_move(sdfg, first, second)
    if move is not None:
        return move
    if first.map.range != second.map.range:
        return f"different map ranges {first.map.range} and {second.map.range}; horizontal fusion needs one range."
    return "blocked by MapFusionHorizontal: not both parallel-compatible, or a data dependency links them."


def fission_to_statements(sdfg: dace.SDFG) -> int:
    """Split ``sdfg`` in place to statement granularity, one map or loop per program output with local temporaries
    recomputed; returns how many fissions applied.

    ``MapFission`` alone is not enough: applied repeatedly it splits per tasklet and turns every local into an
    array. ``SplitStatements`` splits flat maps, ``LoopFission`` sequential loops, and ``MapFission`` only the
    nested-SDFG maps still writing several outputs.
    """
    applied = SplitStatements(split_maps=True).apply_pass(sdfg, {}) or 0
    applied += LoopFission().apply_pass(sdfg, {}) or 0
    return applied + fission_multi_output_maps(sdfg)


def multi_output_fission(sdfg: dace.SDFG, state: SDFGState, entry: nodes.MapEntry) -> dict[str, Any] | None:
    """The ``MapFission`` arguments that split a top-level map writing two or more outputs, if it applies."""
    outputs = {e.data.data for e in state.in_edges(map_exit(state, entry)) if e.data.data}
    if len(outputs) < 2:
        return None
    bodies = [n for n in state.scope_subgraph(entry, False, False).nodes() if isinstance(n, nodes.NestedSDFG)]
    # expr_index 1 matches a map whose body is one nested SDFG, 0 a map over several components
    if len(bodies) == 1 and MapFission.can_be_applied_to(sdfg, expr_index=1, map_entry=entry, nested_sdfg=bodies[0]):
        return {"expr_index": 1, "map_entry": entry, "nested_sdfg": bodies[0]}
    if MapFission.can_be_applied_to(sdfg, expr_index=0, map_entry=entry):
        return {"expr_index": 0, "map_entry": entry}
    return None


def next_multi_output_fission(sdfg: dace.SDFG) -> dict[str, Any] | None:
    for state in sdfg.all_states():
        scope = state.scope_dict()
        for entry in state.nodes():
            if isinstance(entry, nodes.MapEntry) and scope[entry] is None:
                target = multi_output_fission(sdfg, state, entry)
                if target is not None:
                    return target
    return None


def fission_multi_output_maps(sdfg: dace.SDFG) -> int:
    """Apply ``MapFission`` until no top-level map writes two outputs; a single-output map is already a statement."""
    applied = 0
    while (target := next_multi_output_fission(sdfg)) is not None:
        MapFission.apply_to(sdfg, **target)
        applied += 1
    return applied


@dataclass(slots=True)
class FissionMove:
    """Split ``map_entry``'s nested-SDFG body at ``nested_sdfg`` into its independent output groups."""

    map_entry: nodes.MapEntry
    nested_sdfg: nodes.NestedSDFG
    sdfg: dace.SDFG = field(repr=False)  # the SDFG owning the map

    def label(self) -> str:
        return f"fission-map({self.map_entry}): splits nested body {self.nested_sdfg} into independent output groups"


def map_fissions_at(sdfg: dace.SDFG, state: SDFGState, entry: nodes.MapEntry) -> list[FissionMove]:
    """The ``MapFission`` splits of the map at ``entry`` (the map-with-nested-SDFG pattern, ``expr_index=1``)."""
    # a map entry reaches its body over one edge per connector
    body = dict.fromkeys(e.dst for e in state.out_edges(entry) if isinstance(e.dst, nodes.NestedSDFG))
    return [
        FissionMove(entry, nsdfg, sdfg)
        for nsdfg in body
        if MapFission.can_be_applied_to(sdfg, expr_index=1, map_entry=entry, nested_sdfg=nsdfg)
    ]


def enumerate_map_fissions(sdfg: dace.SDFG) -> list[FissionMove]:
    return [
        move
        for state in every_state(sdfg)
        for node in state.nodes()
        if isinstance(node, nodes.MapEntry)
        for move in map_fissions_at(state.sdfg, state, node)
    ]


@dataclass(slots=True)
class RegionMove:
    """One legal ``StateFusion`` of two adjacent states."""

    first_state: SDFGState
    second_state: SDFGState

    def label(self) -> str:
        return f"fuse-states({self.first_state}, {self.second_state})"


def enumerate_region_fusions(sdfg: dace.SDFG) -> list[RegionMove]:
    """Adjacent state pairs ``StateFusion`` accepts, each judged in the SDFG owning it: merging them lets maps in
    both fuse."""
    return [
        RegionMove(edge.src, edge.dst)
        for cfg in sdfg.all_control_flow_regions(recursive=True)
        for edge in cfg.edges()
        if isinstance(edge.src, SDFGState)
        and isinstance(edge.dst, SDFGState)
        and edge.src is not edge.dst
        and StateFusion.can_be_applied_to(edge.src.sdfg, first_state=edge.src, second_state=edge.dst)
    ]


def apply_region_fusion(move: RegionMove) -> None:
    StateFusion.apply_to(
        move.first_state.sdfg,
        verify=True,
        annotate=False,
        save=False,
        first_state=move.first_state,
        second_state=move.second_state,
    )


def map_interchange_moves(sdfg: dace.SDFG) -> Iterator[FusionMove]:
    """Map pairs, the outer one feeding the map directly inside it, that ``MapInterchange`` accepts."""
    for state in every_state(sdfg):
        pairs = dict.fromkeys(
            (e.src, e.dst)
            for e in state.edges()
            if isinstance(e.src, nodes.MapEntry) and isinstance(e.dst, nodes.MapEntry)
        )
        yield from filter(None, (map_interchange_move(state.sdfg, outer, inner) for outer, inner in pairs))


def loop_map_interchange_moves(sdfg: dace.SDFG) -> Iterator[FusionMove]:
    """Loops ``MoveLoopIntoMap`` can move inside the one map they contain."""
    for cfg in sdfg.all_control_flow_regions(recursive=True):
        yield from filter(None, (loop_into_map_move(n) for n in cfg.nodes() if isinstance(n, LoopRegion)))


#: A legal move, and the SDFG owning its nodes.
Move = FusionMove | FissionMove | Rewrite

#: Every move kind, with the tree rows it takes in order.
MOVE_SHAPES: dict[str, tuple[type, ...]] = {
    "loop-fusion": (LoopRegion, LoopRegion),
    "loop-fission": (LoopRegion,),
    "map-fusion": (nodes.MapEntry, nodes.MapEntry),
    "map-fission": (nodes.MapEntry,),
    "subgraph-fission": (nodes.MapEntry, ControlFlowBlock),
    "interchange-loop-loop": (LoopRegion, LoopRegion),
    "interchange-loop-map": (LoopRegion, nodes.MapEntry),
    "interchange-map-loop": (nodes.MapEntry, LoopRegion),
    "interchange-map-map": (nodes.MapEntry, nodes.MapEntry),
    "interchange-if-loop": (ConditionalBlock, LoopRegion),
    "interchange-loop-if": (LoopRegion, ConditionalBlock),
}

#: Kinds NestForge does not implement, and why.
NOT_IMPLEMENTED: dict[str, str] = {
    "interchange-loop-loop": "no DaCe transformation interchanges two loops.",
}

MAP_FISSION_REFUSED = "blocked by MapFission: the map body is not one nested SDFG with independent output groups."
MAP_INTERCHANGE_REFUSED = (
    "blocked by MapInterchange: the inner range reads the outer parameter, or the outer map holds more than the "
    "inner map."
)
LOOP_INTO_MAP_REFUSED = (
    "blocked by MoveLoopIntoMap: the loop bounds are not analyzable, the body is more than one state around the map, "
    "or a dependence would cross map iterations once the map is outer."
)


def check_kind(kind: str) -> str:
    if kind not in MOVE_SHAPES:
        raise ValueError(f"unknown move kind {kind!r}; expected one of {list(MOVE_SHAPES)}")
    return kind


def tree_label(obj: Any) -> str:
    """The label a tree row prints for ``obj``."""
    return obj.map.label if isinstance(obj, (nodes.MapEntry, nodes.MapExit)) else obj.label


def loop_maps(loop: LoopRegion) -> list[nodes.MapEntry]:
    """The outermost maps of the states directly inside ``loop``."""
    return [
        n
        for block in loop.nodes()
        if isinstance(block, SDFGState)
        for n in block.scope_children()[None]
        if isinstance(n, nodes.MapEntry)
    ]


def move_labels(move: Move) -> tuple[str, ...]:
    """The tree labels an agent passes to request ``move``."""
    if isinstance(move, Rewrite):
        return move.labels
    if isinstance(move, FissionMove):
        return (move.map_entry.map.label,)
    if move.xform is MoveLoopIntoMap:
        loop = move.where["loop"]
        assert isinstance(loop, LoopRegion)
        return loop.label, loop_maps(loop)[0].map.label
    return tuple(dict.fromkeys(tree_label(n) for n in move.where.values() if not isinstance(n, nodes.AccessNode)))


#: Enumerators of the legal moves of each implemented kind.
MOVE_LISTINGS: dict[str, Callable[[dace.SDFG], Iterable[Move]]] = {
    "loop-fusion": loop_fusion_moves,
    "loop-fission": loop_fissions,
    "map-fusion": map_fusion_moves,
    "map-fission": enumerate_map_fissions,
    "subgraph-fission": subgraph_fissions,
    "interchange-loop-map": loop_map_interchange_moves,
    "interchange-map-loop": map_loop_interchanges,
    "interchange-map-map": map_interchange_moves,
    "interchange-if-loop": ifs_into_loops,
    "interchange-loop-if": ifs_out_of_loops,
}


def legal_moves(sdfg: dace.SDFG, kind: str | None = None) -> list[tuple[str, tuple[str, ...]]]:
    """``(kind, labels)`` of every legal move right now, of ``kind`` or of all kinds.

    :raises ValueError: ``kind`` is not a move kind.
    """
    kinds = list(MOVE_SHAPES) if kind is None else [check_kind(kind)]
    return [
        (name, labels)
        for name in kinds
        if name in MOVE_LISTINGS
        for labels in dict.fromkeys(move_labels(move) for move in MOVE_LISTINGS[name](sdfg))
    ]


def node_state(row: Row) -> SDFGState:
    """The state holding a node row; a block row has none."""
    state = row[1]
    if state is None:
        raise TypeError(f"{tree_label(row[0])!r} is a control-flow block, not a node of a state")
    return state


def plan_loop_fusion(first: Row, second: Row) -> Move | str:
    loop_a, loop_b = first[0], second[0]
    reason = fuse_loops_reason(loop_a.sdfg, loop_a, loop_b)
    if reason != "yes":
        return reason
    return FusionMove("fuse-loops", {"first": loop_a, "second": loop_b}, FuseLoops, loop_a.sdfg)


def plan_map_fusion(first: Row, second: Row) -> Move | str:
    state = node_state(first)
    if node_state(second) is not state:
        return STATE_BARRIER
    return plan_map_pair(state.sdfg, state, first[0], second[0])


def plan_map_fission(row: Row) -> Move | str:
    state = node_state(row)
    moves = map_fissions_at(state.sdfg, state, row[0])
    return moves[0] if moves else MAP_FISSION_REFUSED


def plan_map_interchange(outer_row: Row, inner_row: Row) -> Move | str:
    outer, inner, state = outer_row[0], inner_row[0], node_state(outer_row)
    if node_state(inner_row) is not state or state.entry_node(inner) is not outer:
        return (
            f"{inner.map.label} is not directly inside {outer.map.label}; name a map, then the map directly inside it."
        )
    return map_interchange_move(state.sdfg, outer, inner) or MAP_INTERCHANGE_REFUSED


def plan_loop_map_interchange(loop_row: Row, map_row: Row) -> Move | str:
    loop, entry = loop_row[0], map_row[0]
    inside = loop_maps(loop)
    if len(inside) != 1 or inside[0] is not entry:
        return f"{entry.map.label} is not the one map directly inside {loop.label}; MoveLoopIntoMap needs exactly that."
    return loop_into_map_move(loop) or LOOP_INTO_MAP_REFUSED


MOVE_PLANNERS: dict[str, Callable[..., Move | str]] = {
    "loop-fusion": plan_loop_fusion,
    "loop-fission": lambda row: plan_loop_fission(row[0]),
    "map-fusion": plan_map_fusion,
    "map-fission": plan_map_fission,
    "subgraph-fission": lambda entry, cut: plan_subgraph_fission(node_state(entry), entry[0], cut[0]),
    "interchange-loop-map": plan_loop_map_interchange,
    "interchange-map-loop": lambda entry, loop: plan_map_loop_interchange(node_state(entry), entry[0], loop[0]),
    "interchange-map-map": plan_map_interchange,
    "interchange-if-loop": lambda cond, loop: plan_if_into_loop(cond[0], loop[0]),
    "interchange-loop-if": lambda loop, cond: plan_if_out_of_loop(loop[0], cond[0]),
}


def plan_move(kind: str, rows: Sequence[Row]) -> Move | str:
    """The move ``kind`` makes of ``rows`` (one per label, of an implemented kind), or why it is illegal. Legality
    is DaCe's own check; nothing is mutated."""
    for (obj, _), wanted in zip(rows, MOVE_SHAPES[kind]):
        if not isinstance(obj, wanted):
            expected = ", ".join(t.__name__ for t in MOVE_SHAPES[kind])
            return f"{kind} takes ({expected}); {tree_label(obj)!r} is a {type(obj).__name__}."
    return MOVE_PLANNERS[kind](*rows)


def commit_move(move: Move) -> str:
    """Commit one move on the SDFG owning its nodes; returns what ran."""
    if isinstance(move, Rewrite):
        move.commit()
        return move.name
    if isinstance(move, FissionMove):
        MapFission.apply_to(
            move.sdfg, expr_index=1, annotate=False, save=False, map_entry=move.map_entry, nested_sdfg=move.nested_sdfg
        )
        return MapFission.__name__
    apply_fusion(move)
    return move.xform.__name__


CACHE_MODEL = "map_perfect_loop_none"


@dataclass(slots=True, frozen=True)
class ScopeMetrics:
    """Symbolic cost of one scope; ``oi`` is ``work / bytes``, ``None`` when no counted byte moves."""

    work: sympy.Expr
    depth: sympy.Expr
    bytes: sympy.Expr
    oi: sympy.Expr | None

    def suffix(self) -> str:
        if self.oi is None:
            oi = "-"
        else:
            oi = f"{float(self.oi):.4g}" if self.oi.is_number else str(self.oi)
        return f"work={self.work} depth={self.depth} bytes={self.bytes} OI={oi}"


def standalone_scope(sdfg: dace.SDFG, node: nodes.MapEntry | LoopRegion) -> dace.SDFG:
    if isinstance(node, nodes.MapEntry):
        state = find_state_of_node(sdfg, node)
        if state.entry_node(node) is not None:
            raise TypeError(f"map {node} is nested in another map; metrics are per top-level map")
        twin_sdfg, _, twin = detached_twin(sdfg, state, node)
        return extract_map_nest(twin_sdfg, twin).standalone_sdfg
    if isinstance(node, LoopRegion) and node.parent_graph is sdfg:
        twin_sdfg = detach(sdfg)
        twin_loop = twin_sdfg.nodes()[sdfg.nodes().index(node)]
        assert isinstance(twin_loop, LoopRegion), "a deep copy keeps block order"
        return extract_cfg_nest(twin_sdfg, twin_loop).standalone_sdfg
    raise TypeError(f"{node} is neither a top-level map of a state nor a loop at the top of the SDFG")


def scope_metrics(sdfg: dace.SDFG, node: nodes.MapEntry | LoopRegion) -> ScopeMetrics:
    """Work, depth, bytes moved and operational intensity of one scope, analyzed on a detached copy.

    :param sdfg: The program holding ``node``; never mutated.
    :param node: A top-level ``MapEntry`` of a state, or a ``LoopRegion`` block of ``sdfg`` itself.
    :returns: The metrics over ``sdfg``'s symbols, bytes under :data:`CACHE_MODEL`.
    """
    scope = standalone_scope(sdfg, node)
    # analyze_sdfg is unannotated and returns (work, depth) when not asked for average parallelism
    work, depth = cast(
        tuple[sympy.Expr, sympy.Expr], work_depth.analyze_sdfg(scope, {}, work_depth.get_tasklet_work_depth, [], False)
    )
    read, write = total_volume.analyze_sdfg(scope, cache_model=CACHE_MODEL)
    moved = cast(sympy.Expr, dace.symbolic.simplify(read + write))
    oi = cast(sympy.Expr, dace.symbolic.simplify(work / moved)) if moved != 0 else None
    return ScopeMetrics(work, depth, moved, oi)


def post_fusion_stages(targets: Targets) -> list[str]:
    """Canonicalization stages after :data:`FUSE_STAGE`; they run once the granularity is chosen."""
    labels = stage_labels(targets.canon_target)
    return labels[labels.index(FUSE_STAGE) + 1 :]


def full_fusion(sdfg: dace.SDFG, targets: Targets) -> dace.SDFG:
    """Deterministic phase 1 default: canonicalization's own fusion stage, then the post-fusion stages.

    :param sdfg: A normalized SDFG (phase 0 output), possibly with re-inlined kernels.
    :param targets: Picks the canonicalization preset.
    :returns: The same SDFG, fused.
    """
    # map fusion never descends into a nested SDFG, and a kernel re-inlined for feedback arrives nested
    inline_top_level_nsdfgs(sdfg)
    return canonicalize(sdfg, target=targets.canon_target, stages=[FUSE_STAGE, *post_fusion_stages(targets)])


def finish_schedule(sdfg: dace.SDFG, targets: Targets) -> dace.SDFG:
    """Run the post-fusion stages after a hand-chosen granularity."""
    return canonicalize(sdfg, target=targets.canon_target, stages=post_fusion_stages(targets))
