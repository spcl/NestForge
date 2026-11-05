# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 2: fusion, fission and interchange moves on the tree rows labels name, judged and applied by DaCe's own
transformations, and the symbolic work and depth of a scope."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, cast

import dace
import sympy
from dace.data import Scalar
from dace.libraries.standard.nodes.external_call import ExternalCall
from dace.sdfg import nodes
from dace.sdfg.graph import MultiConnectorEdge, SubgraphView
from dace.sdfg.performance_evaluation import work_depth
from dace.sdfg.propagation import propagate_memlets_sdfg
from dace.sdfg.state import BreakBlock, ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.sdfg.utils import set_nested_sdfg_parent_references
from dace.subsets import Range
from dace.transformation.dataflow.copy_to_map import CopyToMap
from dace.transformation.dataflow.map_collapse import MapCollapse
from dace.transformation.dataflow.map_expansion import MapExpansion
from dace.transformation.dataflow.map_fission import MapFission
from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.transformation.dataflow.map_fusion_horizontal import MapFusionHorizontal
from dace.transformation.dataflow.map_fusion_vertical import MapFusionVertical
from dace.transformation.dataflow.map_interchange import MapInterchange
from dace.transformation.dataflow.otf_map_fusion import OTFMapFusion
from dace.transformation.dataflow.tiling import MapTiling
from dace.transformation.dataflow.trivial_map_elimination import TrivialMapElimination
from dace.transformation.helpers import nest_state_subgraph
from dace.transformation.interstate import InlineSDFG, MapLoopInterchange, SubgraphFission
from dace.transformation.interstate.loop_fusion import LoopFusion
from dace.transformation.interstate.loop_tiling import LoopTiling
from dace.transformation.interstate.move_loop_into_map import MoveLoopIntoMap, analyze_lanes
from dace.transformation.interstate.move_loop_invariant_if_up import MoveLoopInvariantIfUp, match_loop
from dace.transformation.interstate.state_fusion import StateFusion
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize.untile_loops import (
    MapTilePair,
    UntileLoops,
    eliminate_pinned_maps,
    enclosing_loop_variables,
    exact_bounds,
    map_tile_pairs,
    untile_map_pair,
)
from dace.transformation.passes.cpu_specialization.interchange_loops_into_lane_maps import inside_a_map, lane_axis_of
from dace.transformation.passes.dead_dataflow_elimination import DeadDataflowElimination
from dace.transformation.passes.loop_fission import LoopFission
from dace.transformation.passes.move_if_into_loop import MoveIfIntoLoop, match_guard
from dace.transformation.transformation import PatternTransformation

from nestforge.ir.extract import detach, detached_twin, extract_cfg_nest, extract_map_nest, find_state_of_node
from nestforge.ir.introspect import Row
from nestforge.ir.names import in_order, top_level_map_entries
from nestforge.records import Record
from nestforge.stages.expand import expand_library, expansion, set_implementation


@dataclass(frozen=True, slots=True)
class Rewrite:
    """A legal move; ``commit`` applies it and ``name`` is the transformation that runs."""

    name: str
    commit: Callable[[], object]


#: The move kind that turns a library call into maps, which the session then canonicalizes.
EXPAND_LIBRARY = "expand-library"
#: The move kind that picks a library call's implementation; it takes ``implementation``.
SET_IMPLEMENTATION = "set-library-implementation"

#: Every move kind, with the tree rows it takes in order.
MOVE_SHAPES: dict[str, tuple[type, ...]] = {
    "for-fusion": (LoopRegion, LoopRegion),
    "for-fission": (LoopRegion,),
    "doall-fusion": (nodes.MapEntry, nodes.MapEntry),
    "doall-fission": (nodes.MapEntry,),
    "subgraph-fission": (nodes.MapEntry, ControlFlowBlock),
    "interchange-for-doall": (LoopRegion, nodes.MapEntry),
    "interchange-for-doalls": (LoopRegion,),
    "interchange-doall-for": (nodes.MapEntry, LoopRegion),
    "interchange-doall-doall": (nodes.MapEntry, nodes.MapEntry),
    "interchange-if-for": (ConditionalBlock, LoopRegion),
    "interchange-for-if": (LoopRegion, ConditionalBlock),
    EXPAND_LIBRARY: (nodes.LibraryNode,),
    SET_IMPLEMENTATION: (nodes.LibraryNode,),
    "tile-doall": (nodes.MapEntry,),
    "untile-doall": (nodes.MapEntry, nodes.MapEntry),
    "collapse-doalls": (nodes.MapEntry, nodes.MapEntry),
    "expand-doall": (nodes.MapEntry,),
    "doall-to-for": (nodes.MapEntry,),
    "copy-to-doall": (ControlFlowBlock,),
    "tile-for": (LoopRegion,),
    "untile-for": (LoopRegion,),
}

#: The move kinds that take ``tile_sizes``: one positive size per dimension of the kernel, in its order.
SIZED = frozenset({"tile-doall", "tile-for"})
#: The size a sized move is checked at when it is listed; the agent passes its own to apply it.
PROBE_TILE = 32


def probe_sizes(kind: str, rows: Sequence[Row]) -> tuple[int, ...]:
    """The tile sizes :func:`legal_moves` checks a sized move with: :data:`PROBE_TILE` in every dimension."""
    if kind not in SIZED:
        return ()
    head = rows[0][0]
    return (PROBE_TILE,) * (len(head.map.params) if isinstance(head, nodes.MapEntry) else 1)


def probe_implementation(kind: str, rows: Sequence[Row]) -> str | None:
    """The implementation :func:`legal_moves` checks :data:`SET_IMPLEMENTATION` with: any the node has; the session
    lists one move per implementation it may take."""
    if kind != SET_IMPLEMENTATION:
        return None
    return next(iter(cast(nodes.LibraryNode, rows[0][0]).implementations), None)


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

#: Every copy becomes a map: by default CopyToMap takes only copies whose two arrays differ in strides.
COPY_TO_MAP_OPTIONS = {"ignore_strides": True}
COPY_TO_MAP_REFUSED = (
    "blocked by CopyToMap: the state holds no copy between two arrays outside a doall that it can turn into a map "
    "(a copy between arrays of different rank is one only if it is a single element, which becomes a tasklet)."
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
    """The access nodes the ``producer`` map writes and the ``consumer`` map reads; an edge with an empty memlet only
    orders the two maps and reads nothing."""
    written = dict.fromkeys(e.dst for e in state.out_edges(state.exit_node(producer)) if not e.data.is_empty())
    return [
        a
        for a in written
        if isinstance(a, nodes.AccessNode)
        and any(e.dst is consumer and not e.data.is_empty() for e in state.out_edges(a))
    ]


#: LoopFusion without its guards against serializing a parallel loop, for loops whose parallelism is in maps inside.
UNGUARDED_LOOP_FUSION = {"defer_doall": False, "dont_mix_known_parallel": False}


def body_states(loop: LoopRegion) -> list[SDFGState]:
    return [block for block in loop.all_control_flow_blocks() if isinstance(block, SDFGState)]


def keeps_a_doall(loop: LoopRegion) -> bool:
    """Whether a map runs inside ``loop``'s body: fused and sequential, the loop keeps that map's parallelism."""
    return any(top_level_map_entries(state) for state in body_states(loop))


def fuse_body_states(loop: LoopRegion) -> int:
    """Merge consecutive states directly in ``loop`` while ``StateFusion`` allows, so the maps the two bodies brought
    meet in one state, where ``doall-fusion`` takes them; how many merges."""
    merged = 0
    while True:
        pair = next(
            (
                (edge.src, edge.dst)
                for edge in loop.edges()
                if isinstance(edge.src, SDFGState)
                and isinstance(edge.dst, SDFGState)
                and StateFusion.can_be_applied_to(loop.sdfg, first_state=edge.src, second_state=edge.dst)
            ),
            None,
        )
        if pair is None:
            return merged
        StateFusion.apply_to(loop.sdfg, first_state=pair[0], second_state=pair[1], verify=False)
        merged += 1


def plan_loop_fusion(first: Row, second: Row) -> Rewrite | str:
    """The two loops become one, then the states of the merged body become one where ``StateFusion`` allows. Loops
    that each hold a map fuse even when LoopFusion would rather leave one to ``LoopToMap``: sequential, the loop keeps
    the maps' parallelism (``doall-to-for`` makes such loops on purpose)."""
    loop_a, loop_b = first[0], second[0]
    if loop_a.parent_graph is not loop_b.parent_graph:
        return "loops are in different control-flow regions; fuse the enclosing loops first."
    out = loop_a.parent_graph.out_edges(loop_a)
    if len(out) != 1 or out[0].dst is not loop_b:
        return "loops are not adjacent: exactly one sequencing edge must lead from the first to the second."
    sdfg, where = loop_a.sdfg, {"first": loop_a, "second": loop_b}
    move = pattern(LoopFusion, sdfg, where)
    if move is None and keeps_a_doall(loop_a) and keeps_a_doall(loop_b):
        move = pattern(LoopFusion, sdfg, where, options=UNGUARDED_LOOP_FUSION)
    if move is None:
        return "blocked by LoopFusion: different iteration ranges, or a loop-carried dependency between the two."
    fused = move
    return region_rewrite(fused.name, sdfg, lambda: (fused.commit(), fuse_body_states(loop_a)))


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


def read_anywhere(sdfg: dace.SDFG, name: str) -> bool:
    """Whether any state of ``sdfg`` reads container ``name``, or an interstate edge names it."""
    if any(st.out_degree(n) > 0 for st in sdfg.states() for n in st.data_nodes() if n.data == name):
        return True
    return any(name in edge.data.free_symbols for edge in sdfg.all_interstate_edges())


def dropping_dead_writes(move: Rewrite, sdfg: dace.SDFG, name: str) -> Rewrite:
    """``move``, then DaCe's dead-dataflow elimination when the fusion left the intermediate ``name`` a transient no
    one reads: the fused map would otherwise still store the whole array (its last reader fused into the same map)."""

    def commit() -> None:
        move.commit()
        desc = sdfg.arrays.get(name)
        if desc is not None and desc.transient and not read_anywhere(sdfg, name):
            Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, {})

    return Rewrite(move.name, commit)


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
                return dropping_dead_writes(move, sdfg, arr.data)
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


def nests_scalar_output(body: dace.SDFG) -> bool:
    """Whether :func:`group_statements` would nest a group writing a scalar the map passes out: MapFission gives ``body``
    an array for that scalar and leaves the new nest's connector a scalar, which memlet propagation refuses."""
    for state in body.states():
        for group in statement_groups(state):
            if sum(isinstance(n, nodes.CodeNode) for n in group) < 2:
                continue
            for node in group:
                for edge in state.out_edges(node):
                    written = edge.dst.desc(body) if isinstance(edge.dst, nodes.AccessNode) else None
                    if isinstance(written, Scalar) and not written.transient:
                        return True
    return False


def derives_from_params(body: nodes.NestedSDFG, entry: nodes.MapEntry) -> bool:
    """Whether ``body`` binds a symbol of its own to an expression of the map's parameters (OTF fusion's ``_i0: i - 1``).
    MapFission rebinds only a parameter passed under its own name, so the binding would outlive the map defining it."""
    params = set(entry.map.params)
    return any(
        name not in params and params & {str(s) for s in dace.symbolic.pystr_to_symbolic(str(value)).free_symbols}
        for name, value in body.symbol_mapping.items()
    )


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
        if move is None or derives_from_params(body, entry) or nests_scalar_output(body.sdfg):
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


def scope_copies(state: SDFGState, entry: nodes.MapEntry) -> list[str]:
    """The arrays of access nodes inside ``entry``'s scope that no inner map reads or writes: a copy beside the map."""
    children = state.scope_children()[entry]
    inner = [n for n in children if isinstance(n, nodes.MapEntry)]
    ends = set(inner) | {state.exit_node(m) for m in inner}
    return [
        n.data
        for n in children
        if isinstance(n, nodes.AccessNode) and not any(e.src in ends or e.dst in ends for e in state.all_edges(n))
    ]


def flatten_map_body(state: SDFGState, entry: nodes.MapEntry) -> list[nodes.MapEntry]:
    """Inline the nested SDFG that is the body of ``entry`` and drop its one-iteration maps; the maps now directly
    inside ``entry``."""
    body = map_body(state, entry)
    assert body is not None, "the caller checked the body"
    InlineSDFG.apply_to(state.sdfg, nested_sdfg=body, verify=False)
    root = state.sdfg.root_sdfg
    set_nested_sdfg_parent_references(root)
    root.reset_cfg_list()
    for inner in [n for n in state.scope_children()[entry] if isinstance(n, nodes.MapEntry)]:
        if all(begin == end for begin, end, _ in inner.map.range):
            TrivialMapElimination.apply_to(state.sdfg, map_entry=inner, verify=False)
    return [n for n in state.scope_children()[entry] if isinstance(n, nodes.MapEntry)]


def body_swap_blocker(state: SDFGState, entry: nodes.MapEntry) -> str | None:
    """Why the map directly inside the body of ``entry`` cannot be swapped with it, or ``None``; this inlines the body
    in ``state``, so it runs on a detached twin."""
    body = map_body(state, entry)
    if body is None or not InlineSDFG.can_be_applied_to(state.sdfg, nested_sdfg=body):
        return f"blocked by InlineSDFG: the body of {entry.map.label} cannot be inlined."
    inner = flatten_map_body(state, entry)
    beside = (nodes.MapEntry, nodes.MapExit, nodes.AccessNode)
    holds = [n for n in state.scope_children()[entry] if not isinstance(n, beside)]
    if len(inner) != 1 or holds or scope_copies(state, entry):
        return (
            f"the body of {entry.map.label} holds more than one map or a computation or copy beside it; "
            "doall-fission or subgraph-fission it first."
        )
    where = {"outer_map_entry": entry, "inner_map_entry": inner[0]}
    return None if pattern(MapInterchange, state.sdfg, where) else MAP_INTERCHANGE_REFUSED


def plan_body_map_map(state: SDFGState, outer: nodes.MapEntry, inner_row: Row) -> Rewrite | str:
    """``map j: {nested SDFG: map k}`` -> ``map k: map j``: the body is inlined first (and its one-iteration maps
    dropped), so the maps are directly nested; legality is read on a copy."""
    inner, inner_state = inner_row[0], node_state(inner_row)
    body = map_body(state, outer)
    if body is None or inner_state.sdfg is not body.sdfg or len(body.sdfg.nodes()) != 1:
        return (
            f"{inner.map.label} is not directly inside {outer.map.label}, nor the one map in its body; name a map, "
            "then the map directly inside it."
        )
    _, twin_state, twin_entry = detached_twin(state.sdfg, state, outer)
    reason = body_swap_blocker(twin_state, twin_entry)
    if reason is not None:
        return reason

    def commit() -> None:
        (flat,) = flatten_map_body(state, outer)
        MapInterchange.apply_to(state.sdfg, outer_map_entry=outer, inner_map_entry=flat, verify=False)
        root = state.sdfg.root_sdfg
        set_nested_sdfg_parent_references(root)
        root.reset_cfg_list()

    return Rewrite("MapInterchange", commit)


def plan_map_map(outer_row: Row, inner_row: Row) -> Rewrite | str:
    outer, inner, state = outer_row[0], inner_row[0], node_state(outer_row)
    if node_state(inner_row) is not state:
        return plan_body_map_map(state, outer, inner_row)
    if state.entry_node(inner) is not outer:
        return (
            f"{inner.map.label} is not directly inside {outer.map.label}; name a map, then the map directly inside it."
        )
    if any(begin == end for begin, end, _ in inner.map.range):
        return plan_trimmed_map_map(state, outer, inner)
    where = {"outer_map_entry": outer, "inner_map_entry": inner}
    return pattern(MapInterchange, state.sdfg, where) or MAP_INTERCHANGE_REFUSED


def plan_trimmed_map_map(state: SDFGState, outer: nodes.MapEntry, inner: nodes.MapEntry) -> Rewrite | str:
    """``map j: map k, jj in j:j`` -> ``map k: map j``: the inner map's one-iteration dimensions (the one-column map a
    fission leaves, whose range reads ``j``) are dropped first, since ``MapInterchange`` refuses a range that reads
    the outer parameter; legality is read on a copy."""
    twin, twin_state, twin_outer = detached_twin(state.sdfg, state, outer)
    twin_inner = twin_state.node(state.node_id(inner))
    assert isinstance(twin_inner, nodes.MapEntry), "a deep copy keeps node ids"
    TrivialMapElimination.apply_to(twin, map_entry=twin_inner, verify=False)
    if twin_inner not in twin_state.nodes():
        return f"{inner.map.label} has one iteration only; there is nothing to swap with {outer.map.label}."
    where = {"outer_map_entry": twin_outer, "inner_map_entry": twin_inner}
    if pattern(MapInterchange, twin, where) is None:
        return MAP_INTERCHANGE_REFUSED

    def commit() -> None:
        TrivialMapElimination.apply_to(state.sdfg, map_entry=inner, verify=False)
        MapInterchange.apply_to(state.sdfg, outer_map_entry=outer, inner_map_entry=inner, verify=False)

    return Rewrite("MapInterchange", commit)


def plan_loop_map(loop_row: Row, map_row: Row) -> Rewrite | str:
    loop, entry = loop_row[0], map_row[0]
    if loop_maps(loop) != [entry]:
        return f"{entry.map.label} is not the one map directly inside {loop.label}; MoveLoopIntoMap needs exactly that."
    # MoveLoopIntoMap passes a nest only the SDFG's symbols and data, and raises on an outer loop's iterator
    read = {str(s) for s in loop.free_symbols} - set(loop.sdfg.symbols)
    if read & enclosing_loop_variables(loop):
        return f"blocked by MoveLoopIntoMap: {loop.label} reads the iterator of a loop around it; untile-for it first."
    return pattern(MoveLoopIntoMap, loop.sdfg, {"loop": loop}) or LOOP_INTO_MAP_REFUSED


LANE_HOIST_REFUSED = (
    "blocked by MoveLoopIntoMap: the loop bounds are not analyzable, or the body is not one component of states whose "
    "maps all span the lane axis."
)


def plan_loop_lanes(loop_row: Row) -> Rewrite | str:
    """``for t: { doall i: A; doall i: B; ... }`` -> ``doall i: for t: { A; B; ... }``: the loop moves inside one map
    over the axis every map of its body shares (DaCe's lane analysis, as ``InterchangeLoopsIntoLaneMaps`` runs it, but
    without its CPU gates: whether the hoist pays is the caller's to judge)."""
    loop = loop_row[0]
    if inside_a_map(loop):
        return f"{loop.label} is inside a doall; hoist the loop of the outermost doall's body instead."
    if lane_axis_of(loop) is None:
        return f"the body of {loop.label} does not hold two doalls over one shared axis; use interchange-for-doall."
    read = {str(s) for s in loop.free_symbols} - set(loop.sdfg.symbols)
    if read & enclosing_loop_variables(loop):
        return f"blocked by MoveLoopIntoMap: {loop.label} reads the iterator of a loop around it; untile-for it first."
    lanes = analyze_lanes(loop, loop.sdfg, True)
    if lanes.refusal is not None:
        return f"blocked by MoveLoopIntoMap: {lanes.refusal}"
    move = pattern(MoveLoopIntoMap, loop.sdfg, {"loop": loop}, options={"cfg_body": True, "lane_axis": True})
    if move is None:
        return LANE_HOIST_REFUSED
    sdfg = loop.sdfg
    return region_rewrite(move.name, sdfg, lambda: (move.commit(), propagate_memlets_sdfg(sdfg)))


def plan_map_loop(entry_row: Row, loop_row: Row) -> Rewrite | str:
    """``map i: for t: B`` -> ``for t: map i: B``, when the loop is the whole body of the map."""
    state, entry, loop = node_state(entry_row), entry_row[0], loop_row[0]
    body = map_body(state, entry)
    if body is None or body.sdfg.nodes() != [loop]:
        return (
            f"the body of {entry.map.label} is not exactly the loop {loop.label}; subgraph-fission the map around the "
            "loop first, so the loop is the whole body of a map of its own."
        )
    if state.entry_node(entry) is not None:
        return plan_nested_map_loop(state, entry)
    xform = map_loop_interchange(state, entry, body)
    reason = xform.refusal(state)
    if reason is not None:
        return f"blocked by MapLoopInterchange: {reason}."
    return Rewrite("MapLoopInterchange", lambda: xform.apply(state, state.sdfg))


def map_loop_interchange(state: SDFGState, entry: nodes.MapEntry, body: nodes.NestedSDFG) -> MapLoopInterchange:
    xform = MapLoopInterchange()
    match = {MapLoopInterchange.map_entry: state.node_id(entry), MapLoopInterchange.nested_sdfg: state.node_id(body)}
    xform.setup_match(state.sdfg, state.parent_graph.cfg_id, state.block_id, match, 0)
    return xform


def nest_map(state: SDFGState, entry: nodes.MapEntry) -> tuple[SDFGState, nodes.MapEntry, nodes.NestedSDFG]:
    """Wrap the map at ``entry`` (inside another map) and its body in a nested SDFG of its own, whose one state holds
    the map and its access nodes alone; that state, the map and its body there."""
    nest = nest_state_subgraph(state.sdfg, state, state.scope_subgraph(entry))
    root = state.sdfg.root_sdfg
    set_nested_sdfg_parent_references(root)
    root.reset_cfg_list()
    (inner,) = nest.sdfg.states()
    (inner_entry,) = top_level_map_entries(inner)
    inner_body = map_body(inner, inner_entry)
    assert inner_body is not None, "nesting keeps the map's body"
    return inner, inner_entry, inner_body


def plan_nested_map_loop(state: SDFGState, entry: nodes.MapEntry) -> Rewrite | str:
    """``map i0: map i: for t: B`` -> ``map i0: for t: map i: B`` (the point map of a tile): ``MapLoopInterchange``
    splits the state around the map it moves, which a map inside another map cannot be, so the inner map is nested
    first; legality is read on a copy."""
    twin, twin_state, twin_entry = detached_twin(state.sdfg, state, entry)
    twin_inner, twin_inner_entry, twin_body = nest_map(twin_state, twin_entry)
    reason = map_loop_interchange(twin_inner, twin_inner_entry, twin_body).refusal(twin_inner)
    if reason is not None:
        return f"blocked by MapLoopInterchange: {reason}."

    def commit() -> None:
        inner, inner_entry, body = nest_map(state, entry)
        map_loop_interchange(inner, inner_entry, body).apply(inner, inner.sdfg)

    return Rewrite("MapLoopInterchange", commit)


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


def plan_expand_library(row: Row) -> Rewrite | str:
    node, state = cast(nodes.LibraryNode, row[0]), node_state(row)
    if isinstance(node, ExternalCall):
        return f"{node.label} is a kernel NestForge built (a scope), not a library call."
    implementation = expansion(node)
    if implementation is None:
        return (
            f"{node.label} ({type(node).__name__}) has no vendor-free expansion (only {sorted(node.implementations)}); "
            "it stays a library call."
        )
    return Rewrite(f"{type(node).__name__} expansion ({implementation})", lambda: expand_library(state, node))


def plan_set_implementation(row: Row, implementation: str | None = None) -> Rewrite | str:
    node = cast(nodes.LibraryNode, row[0])
    if isinstance(node, ExternalCall):
        return f"{node.label} is a kernel NestForge built (a scope), not a library call."
    if implementation is None:
        return f"pass implementation, one of {sorted(node.implementations)}."
    if implementation not in node.implementations:
        return f"{node.label} ({type(node).__name__}) has no implementation {implementation!r}; it has {sorted(node.implementations)}."
    return Rewrite(
        f"{type(node).__name__} implementation {implementation}", lambda: set_implementation(node, implementation)
    )


def plan_tile_doall(row: Row, tile_sizes: Sequence[int] = ()) -> Rewrite | str:
    entry, state = cast(nodes.MapEntry, row[0]), node_state(row)
    dims = len(entry.map.params)
    if len(tile_sizes) != dims or any(size < 1 for size in tile_sizes) or max(tile_sizes) < 2:
        return (
            f"{entry.map.label} has {dims} dimension(s) {entry.map.params}: pass tile_sizes, one positive size each "
            "(1 leaves a dimension untiled), at least one of 2 or more."
        )
    options = {"tile_sizes": tuple(tile_sizes), "divides_evenly": False}
    move = pattern(MapTiling, state.sdfg, {"map_entry": entry}, options=options)
    if move is None:
        return f"blocked by MapTiling: {entry.map.label} cannot be tiled."

    def commit() -> None:
        move.commit()
        # MapTiling's approximate SymExpr bound is text no DaCe analysis reads back (work depth, the tile pair search)
        entry.map.range = Range([exact_bounds(rng) for rng in entry.map.range])

    return Rewrite(move.name, commit)


def one_element_copy(state: SDFGState, edge: MultiConnectorEdge) -> bool:
    """Whether ``edge`` copies exactly one element between two arrays of different rank (``s[n - 1, i] -> carry[i]``):
    ``CopyToMap`` raises on that (W45-59), and a one-element copy has no iteration to turn into a map."""
    src, dst = edge.src, edge.dst
    if not (isinstance(src, nodes.AccessNode) and isinstance(dst, nodes.AccessNode)):
        return False
    if len(src.desc(state.sdfg).shape) == len(dst.desc(state.sdfg).shape):
        return False
    sizes = (*edge.data.get_src_subset(edge, state).size(), *edge.data.get_dst_subset(edge, state).size())
    return all(size == 1 for size in sizes)


def copy_applies_on_twin(state: SDFGState, src: nodes.AccessNode, dst: nodes.AccessNode) -> bool:
    """Whether ``CopyToMap`` applies to the copy ``src`` -> ``dst`` of arrays of different rank, tried on a copy of the
    SDFG: its check says yes to copies it then raises on (W45-59)."""
    twin = detach(state.sdfg)
    twin_state = list(twin.states())[list(state.sdfg.states()).index(state)]
    twin_src, twin_dst = twin_state.node(state.node_id(src)), twin_state.node(state.node_id(dst))
    try:
        CopyToMap.apply_to(
            twin, options=COPY_TO_MAP_OPTIONS, annotate=False, save=False, a=twin_src, b=twin_dst, verify=False
        )
    except ValueError:  # raised from memlet_to_map by a copy between arrays of different rank
        return False
    return True


def plain_copies(state: SDFGState) -> list[tuple[nodes.AccessNode, nodes.AccessNode]]:
    """The copies between two arrays at the top of ``state`` (``a[:] = b``) that ``CopyToMap`` turns into a map, or
    that are one element between arrays of different rank (a tasklet); an edge with an empty memlet only orders the two
    nodes and copies nothing."""
    copies = dict.fromkeys(
        (e.src, e.dst)
        for e in state.edges()
        if not e.data.is_empty()
        and isinstance(e.src, nodes.AccessNode)
        and isinstance(e.dst, nodes.AccessNode)
        and state.entry_node(e.src) is None
        and CopyToMap.can_be_applied_to(state.sdfg, options=COPY_TO_MAP_OPTIONS, a=e.src, b=e.dst)
    )
    return [
        (src, dst)
        for src, dst in copies
        if len(src.desc(state.sdfg).shape) == len(dst.desc(state.sdfg).shape)
        or all(one_element_copy(state, e) for e in state.edges_between(src, dst) if not e.data.is_empty())
        or copy_applies_on_twin(state, src, dst)
    ]


def copy_to_tasklet(state: SDFGState, src: nodes.AccessNode, dst: nodes.AccessNode) -> None:
    """Each one-element copy between ``src`` and ``dst`` runs in a tasklet (``out = element``), so it is a computation
    like the others in the state."""
    for edge in list(state.edges_between(src, dst)):
        if edge.data.is_empty():
            continue
        read, write = edge.data.get_src_subset(edge, state), edge.data.get_dst_subset(edge, state)
        tasklet = state.add_tasklet("copy_element", {"element"}, {"out"}, "out = element")
        state.add_edge(src, None, tasklet, "element", dace.Memlet(data=src.data, subset=read))
        state.add_edge(tasklet, "out", dst, None, dace.Memlet(data=dst.data, subset=write, wcr=edge.data.wcr))
        state.remove_edge(edge)


def plan_copy_to_doall(row: Row) -> Rewrite | str:
    """Every plain copy at the top of a state becomes a doall, so tiling and the lane interchange see it like the
    doalls around it."""
    state = row[0]
    if not isinstance(state, SDFGState):
        return f"{tree_label(state)} is not a state; copy-to-doall takes the state that holds the copy."
    copies = plain_copies(state)
    if not copies:
        return COPY_TO_MAP_REFUSED

    def commit() -> None:
        for src, dst in copies:
            edges = [e for e in state.edges_between(src, dst) if not e.data.is_empty()]
            if all(one_element_copy(state, e) for e in edges):
                copy_to_tasklet(state, src, dst)
            else:
                CopyToMap.apply_to(state.sdfg, options=COPY_TO_MAP_OPTIONS, annotate=False, save=False, a=src, b=dst)

    return Rewrite("CopyToMap", commit)


def tile_pair(outer: nodes.MapEntry, inner: nodes.MapEntry, state: SDFGState) -> MapTilePair | None:
    return next((pair for pair in map_tile_pairs(state) if pair.outer is outer and pair.inner is inner), None)


def plan_untile_doall(outer_row: Row, inner_row: Row) -> Rewrite | str:
    outer, inner, state = cast(nodes.MapEntry, outer_row[0]), cast(nodes.MapEntry, inner_row[0]), node_state(outer_row)
    pair = tile_pair(outer, inner, state) if node_state(inner_row) is state else None
    if pair is None:
        return (
            f"{outer.map.label} is not a tile doall whose points are {inner.map.label}: name the doall stepping over "
            "tiles, then the doall over one tile's points."
        )

    def commit() -> None:
        # one tile dimension per pair: repeat until no tile kernel of these points is left
        current: MapTilePair | None = pair
        while current is not None:
            untile_map_pair(state.sdfg, state, current)
            eliminate_pinned_maps(state.sdfg, state, {current.outer})
            current = next((p for p in map_tile_pairs(state) if p.inner is inner), None)
        propagate_memlets_sdfg(state.sdfg.root_sdfg)

    return Rewrite("untile_map_pair", commit)


def plan_collapse_doalls(outer_row: Row, inner_row: Row) -> Rewrite | str:
    outer, inner, state = outer_row[0], inner_row[0], node_state(outer_row)
    if node_state(inner_row) is not state or state.entry_node(inner) is not outer:
        return f"{inner.map.label} is not directly inside {outer.map.label}; name a doall, then the one inside it."
    where = {"outer_map_entry": outer, "inner_map_entry": inner}
    return pattern(MapCollapse, state.sdfg, where) or (
        "blocked by MapCollapse: the outer doall holds more than the inner one, or the inner range reads the outer "
        "parameter."
    )


def plan_expand_doall(row: Row) -> Rewrite | str:
    entry, state = cast(nodes.MapEntry, row[0]), node_state(row)
    if len(entry.map.params) < 2:
        return f"{entry.map.label} has one dimension; there is nothing to expand."
    # every nested kernel keeps the kernel's own schedule: MapExpansion's default runs the inner ones sequentially
    options = {"inner_schedule": entry.map.schedule}
    return pattern(MapExpansion, state.sdfg, {"map_entry": entry}, options=options) or (
        f"blocked by MapExpansion: {entry.map.label} cannot be split into nested kernels."
    )


def holds_a_doall(state: SDFGState, entry: nodes.MapEntry) -> bool:
    """Whether another map runs inside the map at ``entry``, in its scope or in a nested SDFG of its body."""
    for node in state.scope_subgraph(entry).nodes():
        if isinstance(node, nodes.MapEntry) and node is not entry:
            return True
        if isinstance(node, nodes.NestedSDFG) and any(
            isinstance(inner, nodes.MapEntry) for inner, _ in node.sdfg.all_nodes_recursive()
        ):
            return True
    return False


def drop_empty_states(region: ControlFlowRegion, blocks: Sequence[ControlFlowBlock]) -> None:
    """Remove each of ``blocks`` that is an empty state between plain edges (no assignment, no condition)."""
    for block in blocks:
        if not isinstance(block, SDFGState) or block.nodes() or block not in region.nodes():
            continue
        ins, outs = region.in_edges(block), region.out_edges(block)
        if (
            len(outs) != 1
            or len(ins) > 1
            or any(e.data.assignments or not e.data.is_unconditional() for e in ins + outs)
        ):
            continue
        successor, was_start = outs[0].dst, region.start_block is block
        region.remove_node(block)
        if ins:
            region.add_edge(ins[0].src, successor, ins[0].data)
        elif was_start:
            region.start_block = region.node_id(successor)


def plan_doall_to_for(row: Row) -> Rewrite | str:
    """``doall k: doall j: B`` becomes ``for k: doall j: B``: the outer doall runs in order and the inner doalls stay
    parallel, so its body can fuse with a sequential loop over the same range (``for-fusion``)."""
    entry, state = cast(nodes.MapEntry, row[0]), node_state(row)
    if state.entry_node(entry) is not None:
        return f"{entry.map.label} is inside another doall; only an outermost doall becomes a loop."
    if len(entry.map.params) != 1:
        return f"{entry.map.label} has {len(entry.map.params)} dimensions; expand-doall it first."
    if not holds_a_doall(state, entry):
        return f"{entry.map.label} holds no doall inside: as a loop it would keep no parallelism."
    sdfg, region = state.sdfg, state.parent_graph
    move = pattern(MapToForLoop, sdfg, {"map_entry": entry})
    if move is None:
        return f"blocked by MapToForLoop: {entry.map.label} cannot become a loop."
    sequential = move

    def commit() -> None:
        before = set(region.nodes())
        sequential.commit()
        drop_empty_states(region, [state, *(b for b in region.nodes() if b not in before)])

    return region_rewrite(sequential.name, sdfg, commit)


def plan_tile_for(row: Row, tile_sizes: Sequence[int] = ()) -> Rewrite | str:
    loop = cast(LoopRegion, row[0])
    if len(tile_sizes) != 1 or tile_sizes[0] < 2:
        return f"{loop.label} is one for loop: pass tile_sizes with one size of at least 2."
    if any(isinstance(jump, BreakBlock) for jump in loop_analysis.loop_jumps(loop)):
        return f"{loop.label} holds a break, which would leave one tile and not the loop."
    where, options = {"loop": loop}, {"tile_size": tile_sizes[0]}
    if not LoopTiling.can_be_applied_to(loop.sdfg, options=options, **where):
        return f"blocked by LoopTiling: {loop.label} is not a counted loop of stride 1."
    return region_rewrite("LoopTiling", loop.sdfg, lambda: LoopTiling.apply_to(loop.sdfg, options=options, **where))


def plan_untile_for(row: Row) -> Rewrite | str:
    loop = cast(LoopRegion, row[0])
    # the untiling decides legality as it rewrites, so it is tried on a copy first
    trial = copy.deepcopy(loop.sdfg)
    twin = next(r for r in trial.all_control_flow_regions(recursive=True) if r.label == loop.label)
    if not isinstance(twin, LoopRegion) or not UntileLoops().untile_loop(twin, trial):
        return f"{loop.label} is not a loop over tiles holding the loop over one tile's points."

    def commit() -> None:
        UntileLoops().untile_loop(loop, loop.sdfg)
        propagate_memlets_sdfg(loop.sdfg.root_sdfg)

    return region_rewrite("UntileLoops.untile_loop", loop.sdfg, commit)


PLANNERS: dict[str, Callable[..., Rewrite | str]] = {
    "for-fusion": plan_loop_fusion,
    "for-fission": plan_loop_fission,
    "doall-fusion": plan_map_fusion,
    "doall-fission": plan_map_fission,
    "subgraph-fission": plan_subgraph_fission,
    "interchange-for-doall": plan_loop_map,
    "interchange-for-doalls": plan_loop_lanes,
    "interchange-doall-for": plan_map_loop,
    "interchange-doall-doall": plan_map_map,
    "interchange-if-for": plan_if_loop,
    "interchange-for-if": plan_loop_if,
    EXPAND_LIBRARY: plan_expand_library,
    SET_IMPLEMENTATION: plan_set_implementation,
    "tile-doall": plan_tile_doall,
    "untile-doall": plan_untile_doall,
    "collapse-doalls": plan_collapse_doalls,
    "expand-doall": plan_expand_doall,
    "doall-to-for": plan_doall_to_for,
    "copy-to-doall": plan_copy_to_doall,
    "tile-for": plan_tile_for,
    "untile-for": plan_untile_for,
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
    nodes.MapEntry: "doall",
    ControlFlowBlock: "state",
    ConditionalBlock: "if",
    nodes.LibraryNode: "lib_<family>",
}

#: Kind -> (transformation, what it does); a test keeps the keys those of :data:`MOVE_SHAPES`.
MOVE_TEXT: dict[str, tuple[str, str]] = {
    "for-fusion": ("LoopFusion", "two adjacent loops over the same range become one loop."),
    "for-fission": ("LoopFission", "a loop of independent statement groups becomes one loop per group."),
    "doall-fusion": (
        "MapFusionVertical / OTFMapFusion / MapFusionHorizontal",
        "two maps of one state become one: a producer and its consumer (their intermediate array becomes a scalar, "
        "or stays the array when the producer updates it in place), a consumer that recomputes the producer's "
        "values it reads on the fly (a stencil), or two independent maps over the same range.",
    ),
    "doall-fission": (
        "MapFission",
        "split per computation: one map per independent output group. The pieces often want doall-fusion afterwards: "
        "fuse back the ones that share data.",
    ),
    "subgraph-fission": (
        "SubgraphFission",
        "split per state: the map's body is a program of states (expand the doall row to see them); the map is cut "
        "after the named state, the states up to it in one map and the rest in a second.",
    ),
    "interchange-for-doall": (
        "MoveLoopIntoMap",
        "for t: doall i: B becomes doall i: for t: B (the doall goes outside).",
    ),
    "interchange-for-doalls": (
        "MoveLoopIntoMap",
        "for t: { doall i: A; doall i: B; ... } becomes doall i: for t: { A; B; ... } when every doall of the body spans "
        "the same axis i and no column touches another's elements: one parallel region instead of one per doall and "
        "step, and each column's data stays in cache across t.",
    ),
    "interchange-doall-for": (
        "MapLoopInterchange",
        "map i: for t: B becomes for t: map i: B; the loop must be the whole body of the map.",
    ),
    "interchange-doall-doall": (
        "MapInterchange",
        "swap a map with the map directly inside it; also the one map in the nested body of a map (the body is "
        "inlined and its one-iteration maps dropped first).",
    ),
    "interchange-if-for": ("MoveIfIntoLoop", "if c: for i: B becomes for i: if c: B."),
    "interchange-for-if": ("MoveLoopInvariantIfUp", "for i: if c: B becomes if c: for i: B, c loop-invariant."),
    EXPAND_LIBRARY: (
        "LibraryNode.expand (pure, else specialize), then canonicalization",
        "a library call (lib_blas, lib_fft, lib_reduce, lib_scan, ...) becomes the plain maps and "
        "tasklets that compute it, in canonical parallel form, so fusion, fission and interchange reach inside it; "
        "unexpanded, it stays a vendor or library call built on its own.",
    ),
    SET_IMPLEMENTATION: (
        "LibraryNode.implementation (implementation, one list_moves offers)",
        "a library call, a black-box doall, keeps its row and is built through the named implementation: loops "
        "(pure), the host vendor library (OpenBLAS, FFTW3) or, with a GPU target, the device one (cuBLAS, hipBLAS, "
        "...), which also decides the device its kernel runs on.",
    ),
    "tile-doall": (
        "MapTiling (tile_sizes, one per dimension)",
        "doall i: B becomes doall i0 (step = tile size): doall i (one tile): B; each tile's working set fits a "
        "cache or a thread block. A size need not divide the range.",
    ),
    "untile-doall": ("UntileLoops (map tile pairs)", "the inverse: a tile doall and its point doall become one."),
    "collapse-doalls": ("MapCollapse", "doall i: doall j: B becomes doall (i, j): B, one parallel space."),
    "expand-doall": ("MapExpansion", "the inverse: doall (i, j): B becomes doall i: doall j: B."),
    "doall-to-for": (
        "MapToForLoop",
        "doall k: doall j: B becomes for k: doall j: B: the outer doall runs in order and the doalls inside stay "
        "parallel, so its body can for-fusion with a sequential loop over the same range (a physics pass into the "
        "level scan it feeds).",
    ),
    "copy-to-doall": (
        "CopyToMap",
        "a plain copy between two arrays in the state (a[:] = b) becomes a doall over the copied elements, so "
        "tile-doall and the interchanges treat it like the doalls around it; a one-element copy between arrays of "
        "different rank (s[n - 1, i] -> carry[i]) becomes a tasklet, which doall-fission counts as a computation.",
    ),
    "tile-for": (
        "LoopTiling (tile_sizes, one size)",
        "for i: B becomes for i0 (step = tile size): for i (one tile): B, the same iterations in the same order.",
    ),
    "untile-for": ("UntileLoops", "the inverse: a loop over tiles and its loop over one tile's points become one."),
}


def move_kinds() -> list[MoveKind]:
    """Every move kind, in :data:`MOVE_SHAPES` order."""
    return [
        MoveKind(
            kind=kind, rows=tuple(ROW_KEYWORDS[t] for t in shape), runs=MOVE_TEXT[kind][0], does=MOVE_TEXT[kind][1]
        )
        for kind, shape in MOVE_SHAPES.items()
    ]


def plan_move(
    kind: str, rows: Sequence[Row], tile_sizes: Sequence[int] = (), implementation: str | None = None
) -> Rewrite | str:
    """The move ``kind`` makes of ``rows``, or why it is illegal or not implemented; nothing is mutated.
    ``tile_sizes`` goes to the :data:`SIZED` kinds, ``implementation`` to :data:`SET_IMPLEMENTATION`."""
    for (obj, _), wanted in zip(rows, MOVE_SHAPES[kind]):
        if not isinstance(obj, wanted):
            expected = ", ".join(t.__name__ for t in MOVE_SHAPES[kind])
            return f"{kind} takes ({expected}); {tree_label(obj)!r} is a {type(obj).__name__}."
    if kind in SIZED:
        return PLANNERS[kind](*rows, tile_sizes=tile_sizes)
    if kind == SET_IMPLEMENTATION:
        return PLANNERS[kind](*rows, implementation=implementation)
    return PLANNERS[kind](*rows)


# candidate rows per kind; plan_move judges them


def every_region_block[B: ControlFlowBlock](sdfg: dace.SDFG, kind: type[B]) -> Iterator[B]:
    for region in sdfg.all_control_flow_regions(recursive=True):
        yield from (block for block in region.nodes() if isinstance(block, kind))


def map_rows(sdfg: dace.SDFG) -> Iterator[tuple[SDFGState, nodes.MapEntry]]:
    for owner in sdfg.all_sdfgs_recursive():
        for state in owner.states():
            yield from ((state, n) for n in state.nodes() if isinstance(n, nodes.MapEntry))


def doall_rows(sdfg: dace.SDFG) -> Iterator[tuple[Row, ...]]:
    return (((entry, state),) for state, entry in map_rows(sdfg))


def for_rows(sdfg: dace.SDFG) -> Iterator[tuple[Row, ...]]:
    return (((loop, None),) for loop in every_region_block(sdfg, LoopRegion))


def starts_at(inner: LoopRegion, var: str) -> bool:
    """Whether ``inner`` starts at loop variable ``var``, as the loop over one tile's points does."""
    start = loop_analysis.get_init_assignment(inner)
    return start is not None and var in {str(s) for s in start.free_symbols}


def tile_loops(sdfg: dace.SDFG) -> Iterator[tuple[Row, ...]]:
    """Loops that step over tiles (a stride other than 1) and hold a loop starting at their variable: the only ones
    untile-for can collapse; the untiling itself, tried on a copy, decides."""
    for loop in every_region_block(sdfg, LoopRegion):
        stride = loop_analysis.get_loop_stride(loop)
        if stride is None or stride == 1 or not loop.loop_variable:
            continue
        if any(isinstance(b, LoopRegion) and starts_at(b, loop.loop_variable) for b in loop.nodes()):
            yield ((loop, None),)


def tile_pair_rows(sdfg: dace.SDFG) -> Iterator[tuple[Row, ...]]:
    for owner in sdfg.all_sdfgs_recursive():
        for state in owner.states():
            yield from (((pair.outer, state), (pair.inner, state)) for pair in map_tile_pairs(state))


def library_rows(sdfg: dace.SDFG) -> Iterator[tuple[Row, ...]]:
    for owner in sdfg.all_sdfgs_recursive():
        for state in owner.states():
            yield from (((n, state),) for n in state.nodes() if isinstance(n, nodes.LibraryNode))


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


def lane_loops(sdfg: dace.SDFG) -> Iterator[tuple[Row, ...]]:
    for loop in every_region_block(sdfg, LoopRegion):
        if len(loop_maps(loop)) > 1 and lane_axis_of(loop) is not None and not inside_a_map(loop):
            yield ((loop, None),)


def nested_map_pairs(sdfg: dace.SDFG) -> Iterator[tuple[Row, ...]]:
    for owner in sdfg.all_sdfgs_recursive():
        for state in owner.states():
            pairs = dict.fromkeys(
                (e.src, e.dst)
                for e in state.edges()
                if isinstance(e.src, nodes.MapEntry) and isinstance(e.dst, nodes.MapEntry)
            )
            yield from (((outer, state), (inner, state)) for outer, inner in pairs)


def body_map_pairs(sdfg: dace.SDFG) -> Iterator[tuple[Row, ...]]:
    """A map whose body is a nested SDFG of one state holding exactly one map, with that map."""
    for state, entry in map_rows(sdfg):
        body = map_body(state, entry)
        if body is None or len(body.sdfg.nodes()) != 1:
            continue
        inner_state = body.sdfg.nodes()[0]
        if isinstance(inner_state, SDFGState) and len(inside := list(top_level_map_entries(inner_state))) == 1:
            yield (entry, state), (inside[0], inner_state)


def interchange_map_pairs(sdfg: dace.SDFG) -> Iterator[tuple[Row, ...]]:
    yield from nested_map_pairs(sdfg)
    yield from body_map_pairs(sdfg)


def guard_loop_pairs(sdfg: dace.SDFG) -> Iterator[tuple[Row, ...]]:
    for cond in every_region_block(sdfg, ConditionalBlock):
        sinks = cond.branches[0][1].sink_nodes() if cond.branches else []
        if len(sinks) == 1 and isinstance(sinks[0], LoopRegion):
            yield (cond, None), (sinks[0], None)


def loop_guard_pairs(sdfg: dace.SDFG) -> Iterator[tuple[Row, ...]]:
    for loop in every_region_block(sdfg, LoopRegion):
        yield from (((loop, None), (cond, None)) for cond in loop.nodes() if isinstance(cond, ConditionalBlock))


CANDIDATES: dict[str, Callable[[dace.SDFG], Iterator[tuple[Row, ...]]]] = {
    "for-fusion": loop_pairs,
    "for-fission": for_rows,
    "doall-fusion": map_pairs,
    "doall-fission": doall_rows,
    "subgraph-fission": body_cuts,
    "interchange-for-doall": loop_map_pairs,
    "interchange-for-doalls": lane_loops,
    "interchange-doall-for": map_loop_pairs,
    "interchange-doall-doall": interchange_map_pairs,
    "interchange-if-for": guard_loop_pairs,
    "interchange-for-if": loop_guard_pairs,
    EXPAND_LIBRARY: library_rows,
    SET_IMPLEMENTATION: library_rows,
    "tile-doall": doall_rows,
    "untile-doall": tile_pair_rows,
    "collapse-doalls": nested_map_pairs,
    "expand-doall": lambda sdfg: (((e, st),) for st, e in map_rows(sdfg) if len(e.map.params) > 1),
    "doall-to-for": lambda sdfg: (((e, st),) for st, e in map_rows(sdfg) if len(e.map.params) == 1),
    "copy-to-doall": lambda sdfg: (((st, None),) for st in every_region_block(sdfg, SDFGState) if plain_copies(st)),
    "tile-for": for_rows,
    "untile-for": tile_loops,
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
            probe = plan_move(name, rows, probe_sizes(name, rows), probe_implementation(name, rows))
            if isinstance(probe, Rewrite):
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
