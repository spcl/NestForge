# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 2: fusion, fission and interchange moves on the tree rows labels name, judged and applied by DaCe's own
transformations, and the symbolic work and depth of a scope."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, cast

import dace
import sympy
from dace.sdfg import nodes
from dace.sdfg.graph import SubgraphView
from dace.sdfg.performance_evaluation import work_depth
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, LoopRegion, SDFGState
from dace.sdfg.utils import set_nested_sdfg_parent_references
from dace.transformation.dataflow.map_fission import MapFission
from dace.transformation.dataflow.map_fusion_horizontal import MapFusionHorizontal
from dace.transformation.dataflow.map_fusion_vertical import MapFusionVertical
from dace.transformation.dataflow.map_interchange import MapInterchange
from dace.transformation.dataflow.otf_map_fusion import OTFMapFusion
from dace.transformation.helpers import nest_state_subgraph
from dace.transformation.interstate import MapLoopInterchange, SubgraphFission
from dace.transformation.interstate.loop_fusion import LoopFusion
from dace.transformation.interstate.move_loop_into_map import MoveLoopIntoMap
from dace.transformation.interstate.move_loop_invariant_if_up import MoveLoopInvariantIfUp, match_loop
from dace.transformation.passes.loop_fission import LoopFission
from dace.transformation.passes.move_if_into_loop import MoveIfIntoLoop, match_guard
from dace.transformation.transformation import PatternTransformation

from nestforge.ir.extract import detach, detached_twin, extract_cfg_nest, extract_map_nest, find_state_of_node
from nestforge.ir.introspect import Row
from nestforge.ir.names import in_order, top_level_map_entries
from nestforge.records import Record


@dataclass(frozen=True, slots=True)
class Rewrite:
    """A legal move; ``commit`` applies it and ``name`` is the transformation that runs."""

    name: str
    commit: Callable[[], object]


#: Every move kind, with the tree rows it takes in order.
MOVE_SHAPES: dict[str, tuple[type, ...]] = {
    "loop-fusion": (LoopRegion, LoopRegion),
    "loop-fission": (LoopRegion,),
    "map-fusion": (nodes.MapEntry, nodes.MapEntry),
    "map-fission": (nodes.MapEntry,),
    "subgraph-fission": (nodes.MapEntry, ControlFlowBlock),
    "interchange-loop-map": (LoopRegion, nodes.MapEntry),
    "interchange-map-loop": (nodes.MapEntry, LoopRegion),
    "interchange-map-map": (nodes.MapEntry, nodes.MapEntry),
    "interchange-if-loop": (ConditionalBlock, LoopRegion),
    "interchange-loop-if": (LoopRegion, ConditionalBlock),
}

STATE_BARRIER = (
    "nests are in different states, a control-flow dependency map fusion never crosses; fuse the loops around "
    "them, or define one scope over their blocks."
)
LOOP_FISSION_REFUSED = "blocked by LoopFission: the loop body has no two independent statement groups."
MAP_FISSION_REFUSED = "blocked by MapFission: the map body is not one nested SDFG with independent output groups."
MAP_INTERCHANGE_REFUSED = (
    "blocked by MapInterchange: the inner range reads the outer parameter, or the outer map holds more than the "
    "inner map."
)
LOOP_INTO_MAP_REFUSED = (
    "blocked by MoveLoopIntoMap: the loop bounds are not analyzable, the body is more than one state around the map, "
    "or a dependence would cross map iterations once the map is outer."
)
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


def tree_label(obj: Any) -> str:
    """The label a tree row prints for ``obj``."""
    return obj.map.label if isinstance(obj, (nodes.MapEntry, nodes.MapExit)) else obj.label


def check_kind(kind: str) -> str:
    if kind not in MOVE_SHAPES:
        raise ValueError(f"unknown move kind {kind!r}; expected one of {list(MOVE_SHAPES)}")
    return kind


def pattern(
    xform: type[PatternTransformation], sdfg: dace.SDFG, where: dict[str, Any], **options: Any
) -> Rewrite | None:
    """The move ``xform`` makes at ``where`` when its own check accepts it."""
    if not xform.can_be_applied_to(sdfg, **options, **where):
        return None
    return Rewrite(xform.__name__, lambda: xform.apply_to(sdfg, annotate=False, save=False, **options, **where))


def region_rewrite(name: str, sdfg: dace.SDFG, rewrite: Callable[[], object]) -> Rewrite:
    """A control-flow rewrite, followed by restoring the parent links and region ids it leaves stale."""

    def commit() -> None:
        rewrite()
        root = sdfg.root_sdfg
        set_nested_sdfg_parent_references(root)
        root.reset_cfg_list()

    return Rewrite(name, commit)


def node_state(row: Row) -> SDFGState:
    state = row[1]
    if state is None:
        raise TypeError(f"{tree_label(row[0])!r} is a control-flow block, not a node of a state")
    return state


def map_body(state: SDFGState, entry: nodes.MapEntry) -> nodes.NestedSDFG | None:
    """The nested SDFG that is the whole body of the map at ``entry``, if it is one."""
    targets = dict.fromkeys(edge.dst for edge in state.out_edges(entry))
    body = next(iter(targets))
    return body if len(targets) == 1 and isinstance(body, nodes.NestedSDFG) else None


def loop_maps(loop: LoopRegion) -> list[nodes.MapEntry]:
    """The outermost maps of the states directly inside ``loop``."""
    return [n for block in loop.nodes() if isinstance(block, SDFGState) for n in top_level_map_entries(block)]


def intermediates(state: SDFGState, producer: nodes.MapEntry, consumer: nodes.MapEntry) -> list[nodes.AccessNode]:
    """The access nodes the ``producer`` map writes and the ``consumer`` map reads."""
    written = dict.fromkeys(e.dst for e in state.out_edges(state.exit_node(producer)))
    return [
        a for a in written if isinstance(a, nodes.AccessNode) and any(e.dst is consumer for e in state.out_edges(a))
    ]


def plan_loop_fusion(first: Row, second: Row) -> Rewrite | str:
    loop_a, loop_b = first[0], second[0]
    if loop_a.parent_graph is not loop_b.parent_graph:
        return "loops are in different control-flow regions; fuse the enclosing loops first."
    out = loop_a.parent_graph.out_edges(loop_a)
    if len(out) != 1 or out[0].dst is not loop_b:
        return "loops are not adjacent: exactly one sequencing edge must lead from the first to the second."
    move = pattern(LoopFusion, loop_a.sdfg, {"first": loop_a, "second": loop_b})
    return move or "blocked by LoopFusion: different iteration ranges, or a loop-carried dependency between the two."


def vertical_refusal(state: SDFGState, producer: nodes.MapEntry, arr: nodes.AccessNode) -> str:
    """Why ``MapFusionVertical`` likely refused to fuse through ``arr``, from the shape DaCe's checks reject."""
    if any(isinstance(e.src, nodes.AccessNode) and e.src.data == arr.data for e in state.in_edges(producer)):
        return (
            f"the first map also reads '{arr.data}' (a recurrence); DaCe fuses through a container only when the "
            "first map reads and writes the same elements of it"
        )
    if not state.sdfg.arrays[arr.data].transient:
        return f"'{arr.data}' is also a program output, which the fused map would have to write as well"
    return (
        f"the second map reads elements of '{arr.data}' that another iteration of the first map writes, or the two "
        "map ranges differ"
    )


def plan_map_fusion(first: Row, second: Row) -> Rewrite | str:
    """Vertical through the data linking the maps in either data-flow order (in place when the first map updates
    it), else on the fly (the second map recomputes what it reads of the first), else horizontal when no data links
    the maps."""
    state = node_state(first)
    if node_state(second) is not state:
        return STATE_BARRIER
    sdfg, a, b = state.sdfg, first[0], second[0]
    reasons: list[str] = []
    for producer, consumer in ((a, b), (b, a)):
        for arr in intermediates(state, producer, consumer):
            where = {"first_map_exit": state.exit_node(producer), "array": arr, "second_map_entry": consumer}
            move = pattern(MapFusionVertical, sdfg, where) or pattern(
                OTFMapFusion, sdfg, where, options={"recompute_windows": True}
            )
            if move is not None:
                return move
            reasons.append(
                f"blocked by MapFusionVertical and OTFMapFusion on '{arr.data}': {vertical_refusal(state, producer, arr)}"
            )
    if reasons:
        return "; ".join(reasons) + "."
    if state.entry_node(a) is not state.entry_node(b):
        return "maps are in different scopes with no shared data; not a fusion pair."
    move = pattern(MapFusionHorizontal, sdfg, {"first_parallel_map_entry": a, "second_parallel_map_entry": b})
    if move is not None:
        return move
    if a.map.range != b.map.range:
        return f"different map ranges {a.map.range} and {b.map.range}; horizontal fusion needs one range."
    return "blocked by MapFusionHorizontal: not both parallel-compatible, or a data dependency links them."


def statement_groups(state: SDFGState) -> list[list[nodes.Node]]:
    """The independent computations of a map body's state: its code nodes, joined through the locals (transients)
    they exchange, with those locals. A statement lowered to a chain of tasklets is one group."""
    leader: dict[nodes.Node, nodes.Node] = {
        n: n
        for n in state.nodes()
        if isinstance(n, nodes.CodeNode) or (isinstance(n, nodes.AccessNode) and n.desc(state.sdfg).transient)
    }

    def root(node: nodes.Node) -> nodes.Node:
        while leader[node] is not node:
            node = leader[node]
        return node

    for edge in state.edges():
        if edge.src in leader and edge.dst in leader:
            leader[root(edge.dst)] = root(edge.src)
    groups: dict[nodes.Node, list[nodes.Node]] = {}
    for node in leader:
        groups.setdefault(root(node), []).append(node)
    return [group for group in groups.values() if any(isinstance(n, nodes.CodeNode) for n in group)]


def computations(body: dace.SDFG) -> int:
    """The computations of a map body MapFission gives a map each: its control-flow regions, and the statement
    groups of its states."""
    return sum(1 if not isinstance(block, SDFGState) else len(statement_groups(block)) for block in body.nodes())


def group_statements(body: dace.SDFG) -> None:
    """Nest every statement group of more than one code node, so MapFission splits per statement, not per tasklet
    (which would turn its locals into arrays)."""
    for state in body.states():
        for group in statement_groups(state):
            if sum(isinstance(n, nodes.CodeNode) for n in group) > 1:
                nest_state_subgraph(body, state, SubgraphView(state, group))


def plan_map_fission(row: Row) -> Rewrite | str:
    """Split a map by its independent output groups: a nested-SDFG body by ``MapFission``'s nested pattern, a flat
    body writing two or more outputs by its component pattern (not per tasklet, which would turn locals into
    arrays)."""
    state, entry = node_state(row), row[0]
    body = map_body(state, entry)
    if body is not None and computations(body.sdfg) < 2:
        return "the map body is one computation; there is nothing to split."
    if body is not None:
        where = {"map_entry": entry, "nested_sdfg": body}
        move = pattern(MapFission, state.sdfg, where, expr_index=1)
        if move is None:
            return MAP_FISSION_REFUSED
        return Rewrite(move.name, lambda: (group_statements(body.sdfg), move.commit()))
    outputs = dict.fromkeys(e.data.data for e in state.in_edges(state.exit_node(entry)) if e.data.data)
    move = pattern(MapFission, state.sdfg, {"map_entry": entry}, expr_index=0) if len(outputs) > 1 else None
    return move or MAP_FISSION_REFUSED


def plan_loop_fission(row: Row) -> Rewrite | str:
    loop = row[0]
    if not LoopFission.can_fission(loop):
        return LOOP_FISSION_REFUSED
    return region_rewrite("LoopFission", loop.sdfg, lambda: LoopFission.fission(loop))


def plan_subgraph_fission(entry_row: Row, cut_row: Row) -> Rewrite | str:
    """``map i: A; B`` -> ``map i: A; map i: B``, cut after the block ``cut`` of the map's body."""
    state, entry, cut = node_state(entry_row), entry_row[0], cut_row[0]
    body = map_body(state, entry)
    if body is None or cut.parent_graph is not body.sdfg:
        return f"{cut.label} is not a top-level block of the body of {entry.map.label}."
    if not body.sdfg.out_edges(cut):
        return f"{cut.label} is the last block of the body of {entry.map.label}; name a block before the last."
    where = {"map_entry": entry, "nested_sdfg": body}
    return pattern(SubgraphFission, state.sdfg, where, options={"cut": cut.label}) or SUBGRAPH_FISSION_REFUSED


def plan_map_map(outer_row: Row, inner_row: Row) -> Rewrite | str:
    outer, inner, state = outer_row[0], inner_row[0], node_state(outer_row)
    if node_state(inner_row) is not state or state.entry_node(inner) is not outer:
        return (
            f"{inner.map.label} is not directly inside {outer.map.label}; name a map, then the map directly inside it."
        )
    where = {"outer_map_entry": outer, "inner_map_entry": inner}
    return pattern(MapInterchange, state.sdfg, where) or MAP_INTERCHANGE_REFUSED


def plan_loop_map(loop_row: Row, map_row: Row) -> Rewrite | str:
    loop, entry = loop_row[0], map_row[0]
    if loop_maps(loop) != [entry]:
        return f"{entry.map.label} is not the one map directly inside {loop.label}; MoveLoopIntoMap needs exactly that."
    return pattern(MoveLoopIntoMap, loop.sdfg, {"loop": loop}) or LOOP_INTO_MAP_REFUSED


def plan_map_loop(entry_row: Row, loop_row: Row) -> Rewrite | str:
    """``map i: for t: B`` -> ``for t: map i: B``, when the loop is the whole body of the map."""
    state, entry, loop = node_state(entry_row), entry_row[0], loop_row[0]
    body = map_body(state, entry)
    if body is None or body.sdfg.nodes() != [loop]:
        return (
            f"the body of {entry.map.label} is not exactly the loop {loop.label}; subgraph-fission the map around the "
            "loop first, so the loop is the whole body of a map of its own."
        )
    xform = MapLoopInterchange()
    match = {MapLoopInterchange.map_entry: state.node_id(entry), MapLoopInterchange.nested_sdfg: state.node_id(body)}
    xform.setup_match(state.sdfg, state.parent_graph.cfg_id, state.block_id, match, 0)
    reason = xform.refusal(state)
    if reason is not None:
        return f"blocked by MapLoopInterchange: {reason}."
    return Rewrite("MapLoopInterchange", lambda: xform.apply(state, state.sdfg))


def plan_if_loop(cond_row: Row, loop_row: Row) -> Rewrite | str:
    """``if c: for i: B`` -> ``for i: if c: B``."""
    cond, loop = cond_row[0], loop_row[0]
    match = match_guard(cond)
    if match is None or loop.parent_graph is not match[2] or match[2].out_degree(loop):
        return IF_INTO_LOOP_REFUSED
    return region_rewrite("MoveIfIntoLoop", loop.sdfg, lambda: MoveIfIntoLoop.push(cond))


def plan_loop_if(loop_row: Row, cond_row: Row) -> Rewrite | str:
    """``for i: if c: B`` -> ``if c: for i: B``, for a loop-invariant guard."""
    loop, cond = loop_row[0], cond_row[0]
    match = match_loop(loop.sdfg, loop)
    if match is None or match[1] is not cond:
        return IF_OUT_OF_LOOP_REFUSED
    return region_rewrite("MoveLoopInvariantIfUp", loop.sdfg, lambda: MoveLoopInvariantIfUp.hoist(loop))


PLANNERS: dict[str, Callable[..., Rewrite | str]] = {
    "loop-fusion": plan_loop_fusion,
    "loop-fission": plan_loop_fission,
    "map-fusion": plan_map_fusion,
    "map-fission": plan_map_fission,
    "subgraph-fission": plan_subgraph_fission,
    "interchange-loop-map": plan_loop_map,
    "interchange-map-loop": plan_map_loop,
    "interchange-map-map": plan_map_map,
    "interchange-if-loop": plan_if_loop,
    "interchange-loop-if": plan_loop_if,
}


class MoveKind(Record):
    """One move kind as the agent is told it: the tree rows it takes in order, the DaCe transformation it runs and
    what it does."""

    kind: str
    rows: tuple[str, ...]
    runs: str
    does: str


#: The tree keyword of each row type a move takes.
ROW_KEYWORDS: dict[type, str] = {
    LoopRegion: "for",
    nodes.MapEntry: "kernel",
    ControlFlowBlock: "state",
    ConditionalBlock: "if",
}

#: Kind -> (transformation, what it does); a test keeps the keys those of :data:`MOVE_SHAPES`.
MOVE_TEXT: dict[str, tuple[str, str]] = {
    "loop-fusion": ("LoopFusion", "two adjacent loops over the same range become one loop."),
    "loop-fission": ("LoopFission", "a loop of independent statement groups becomes one loop per group."),
    "map-fusion": (
        "MapFusionVertical / OTFMapFusion / MapFusionHorizontal",
        "two maps of one state become one: a producer and its consumer (their intermediate array becomes a scalar, "
        "or stays the array when the producer updates it in place), a consumer that recomputes the producer's "
        "values it reads on the fly (a stencil), or two independent maps over the same range.",
    ),
    "map-fission": (
        "MapFission",
        "split per computation: one map per independent output group. The pieces often want map-fusion afterwards: "
        "fuse back the ones that share data.",
    ),
    "subgraph-fission": (
        "SubgraphFission",
        "split per state: the map's body is a program of states (expand the kernel row to see them); the map is cut "
        "after the named state, the states up to it in one map and the rest in a second.",
    ),
    "interchange-loop-map": ("MoveLoopIntoMap", "for t: map i: B becomes map i: for t: B (the map goes outside)."),
    "interchange-map-loop": (
        "MapLoopInterchange",
        "map i: for t: B becomes for t: map i: B; the loop must be the whole body of the map.",
    ),
    "interchange-map-map": ("MapInterchange", "swap a map with the map directly inside it."),
    "interchange-if-loop": ("MoveIfIntoLoop", "if c: for i: B becomes for i: if c: B."),
    "interchange-loop-if": ("MoveLoopInvariantIfUp", "for i: if c: B becomes if c: for i: B, c loop-invariant."),
}


def move_kinds() -> list[MoveKind]:
    """Every move kind, in :data:`MOVE_SHAPES` order."""
    return [
        MoveKind(
            kind=kind, rows=tuple(ROW_KEYWORDS[t] for t in shape), runs=MOVE_TEXT[kind][0], does=MOVE_TEXT[kind][1]
        )
        for kind, shape in MOVE_SHAPES.items()
    ]


def plan_move(kind: str, rows: Sequence[Row]) -> Rewrite | str:
    """The move ``kind`` makes of ``rows``, or why it is illegal or not implemented; nothing is mutated."""
    for (obj, _), wanted in zip(rows, MOVE_SHAPES[kind]):
        if not isinstance(obj, wanted):
            expected = ", ".join(t.__name__ for t in MOVE_SHAPES[kind])
            return f"{kind} takes ({expected}); {tree_label(obj)!r} is a {type(obj).__name__}."
    return PLANNERS[kind](*rows)


# candidate rows per kind; plan_move judges them


def every_region_block[B: ControlFlowBlock](sdfg: dace.SDFG, kind: type[B]) -> Iterator[B]:
    for region in sdfg.all_control_flow_regions(recursive=True):
        yield from (block for block in region.nodes() if isinstance(block, kind))


def map_rows(sdfg: dace.SDFG) -> Iterator[tuple[SDFGState, nodes.MapEntry]]:
    for owner in sdfg.all_sdfgs_recursive():
        for state in owner.states():
            yield from ((state, n) for n in state.nodes() if isinstance(n, nodes.MapEntry))


def loop_pairs(sdfg: dace.SDFG) -> Iterator[tuple[Row, ...]]:
    for loop in every_region_block(sdfg, LoopRegion):
        out = loop.parent_graph.out_edges(loop)
        if len(out) == 1 and isinstance(out[0].dst, LoopRegion):
            yield (loop, None), (out[0].dst, None)


def map_pairs(sdfg: dace.SDFG) -> Iterator[tuple[Row, ...]]:
    """Map pairs of one state that share a data node, and each map with the next map of its scope, in node order.
    Only neighbors are offered unlinked: every pair of a scope would be quadratic (CloudSC lists 20k), and fusing
    neighbors one after another reaches any horizontal fusion."""
    for owner in sdfg.all_sdfgs_recursive():
        for state in owner.states():
            entries = [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]
            linked = {
                frozenset((state.entry_node(into.src), out.dst))
                for access in state.data_nodes()
                for into in state.in_edges(access)
                if isinstance(into.src, nodes.MapExit)
                for out in state.out_edges(access)
                if isinstance(out.dst, nodes.MapEntry)
            }
            for i, a in enumerate(entries):
                scope = state.entry_node(a)
                neighbor = next((b for b in entries[i + 1 :] if state.entry_node(b) is scope), None)
                for b in entries[i + 1 :]:
                    if b is neighbor or frozenset((a, b)) in linked:
                        yield (a, state), (b, state)


def body_cuts(sdfg: dace.SDFG) -> Iterator[tuple[Row, ...]]:
    for state, entry in map_rows(sdfg):
        body = map_body(state, entry)
        if body is not None:
            yield from (((entry, state), (block, None)) for block in in_order(body.sdfg)[:-1])


def map_loop_pairs(sdfg: dace.SDFG) -> Iterator[tuple[Row, ...]]:
    for state, entry in map_rows(sdfg):
        body = map_body(state, entry)
        if body is not None and isinstance(loop := body.sdfg.nodes()[0], LoopRegion):
            yield (entry, state), (loop, None)


def loop_map_pairs(sdfg: dace.SDFG) -> Iterator[tuple[Row, ...]]:
    for loop in every_region_block(sdfg, LoopRegion):
        inside = loop_maps(loop)
        if len(inside) == 1:
            yield (loop, None), (inside[0], find_state_of_node(loop.sdfg, inside[0]))


def nested_map_pairs(sdfg: dace.SDFG) -> Iterator[tuple[Row, ...]]:
    for owner in sdfg.all_sdfgs_recursive():
        for state in owner.states():
            pairs = dict.fromkeys(
                (e.src, e.dst)
                for e in state.edges()
                if isinstance(e.src, nodes.MapEntry) and isinstance(e.dst, nodes.MapEntry)
            )
            yield from (((outer, state), (inner, state)) for outer, inner in pairs)


def guard_loop_pairs(sdfg: dace.SDFG) -> Iterator[tuple[Row, ...]]:
    for cond in every_region_block(sdfg, ConditionalBlock):
        sinks = cond.branches[0][1].sink_nodes() if cond.branches else []
        if len(sinks) == 1 and isinstance(sinks[0], LoopRegion):
            yield (cond, None), (sinks[0], None)


def loop_guard_pairs(sdfg: dace.SDFG) -> Iterator[tuple[Row, ...]]:
    for loop in every_region_block(sdfg, LoopRegion):
        yield from (((loop, None), (cond, None)) for cond in loop.nodes() if isinstance(cond, ConditionalBlock))


CANDIDATES: dict[str, Callable[[dace.SDFG], Iterator[tuple[Row, ...]]]] = {
    "loop-fusion": loop_pairs,
    "loop-fission": lambda sdfg: (((loop, None),) for loop in every_region_block(sdfg, LoopRegion)),
    "map-fusion": map_pairs,
    "map-fission": lambda sdfg: (((entry, state),) for state, entry in map_rows(sdfg)),
    "subgraph-fission": body_cuts,
    "interchange-loop-map": loop_map_pairs,
    "interchange-map-loop": map_loop_pairs,
    "interchange-map-map": nested_map_pairs,
    "interchange-if-loop": guard_loop_pairs,
    "interchange-loop-if": loop_guard_pairs,
}


def legal_moves(sdfg: dace.SDFG, kind: str | None = None) -> list[tuple[str, tuple[str, ...]]]:
    """``(kind, labels)`` of every legal move right now, of ``kind`` or of every kind.

    :raises ValueError: ``kind`` is not a move kind.
    """
    kinds = list(CANDIDATES) if kind is None else [check_kind(kind)]
    found: dict[tuple[str, tuple[str, ...]], None] = {}
    for name in kinds:
        if name not in CANDIDATES:
            continue
        for rows in CANDIDATES[name](sdfg):
            if isinstance(plan_move(name, rows), Rewrite):
                found[(name, tuple(tree_label(obj) for obj, _ in rows))] = None
    return list(found)


@dataclass(frozen=True, slots=True)
class ScopeMetrics:
    """Symbolic cost of one scope: its work and its depth."""

    work: sympy.Expr
    depth: sympy.Expr

    def suffix(self) -> str:
        return f"work={self.work} depth={self.depth}"


def sdfg_metrics(scope: dace.SDFG) -> ScopeMetrics:
    """Work and depth of a standalone SDFG, by DaCe's work-depth analysis."""
    # analyze_sdfg is unannotated and returns (work, depth) when not asked for average parallelism
    work, depth = cast(
        tuple[sympy.Expr, sympy.Expr], work_depth.analyze_sdfg(scope, {}, work_depth.get_tasklet_work_depth, [], False)
    )
    return ScopeMetrics(work, depth)


def scope_metrics(sdfg: dace.SDFG, node: nodes.MapEntry | LoopRegion) -> ScopeMetrics:
    """:func:`sdfg_metrics` of a top-level map of a state, or of a loop block of ``sdfg`` itself, analyzed on a
    detached copy so ``sdfg`` is never mutated."""
    if isinstance(node, nodes.MapEntry):
        state = find_state_of_node(sdfg, node)
        if state.entry_node(node) is not None:
            raise TypeError(f"map {node} is nested in another map; metrics are per top-level map")
        twin_sdfg, _, twin = detached_twin(sdfg, state, node)
        return sdfg_metrics(extract_map_nest(twin_sdfg, twin).standalone_sdfg)
    if isinstance(node, LoopRegion) and node.parent_graph is sdfg:
        twin_sdfg = detach(sdfg)
        twin_loop = twin_sdfg.nodes()[sdfg.nodes().index(node)]
        assert isinstance(twin_loop, LoopRegion), "a deep copy keeps block order"
        return sdfg_metrics(extract_cfg_nest(twin_sdfg, twin_loop).standalone_sdfg)
    raise TypeError(f"{node} is neither a top-level map of a state nor a loop at the top of the SDFG")
