# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 2: fusion, fission and interchange moves on the tree rows labels name, judged and applied by DaCe's own
transformations, and the symbolic work and depth of a scope."""

from __future__ import annotations

import copy
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from math import prod
from pathlib import Path
from typing import Any, cast

import dace
import sympy
from dace import dtypes
from dace.data import Scalar
from dace.libraries.standard.nodes.external_call import ExternalCall, external_calls
from dace.sdfg import nodes
from dace.sdfg.graph import SubgraphView
from dace.sdfg.performance_evaluation import work_depth
from dace.sdfg.propagation import propagate_memlets_sdfg
from dace.sdfg.state import (
    BreakBlock,
    ConditionalBlock,
    ContinueBlock,
    ControlFlowBlock,
    ControlFlowRegion,
    LoopRegion,
    ReturnBlock,
    SDFGState,
)
from dace.sdfg.utils import set_nested_sdfg_parent_references, weakly_connected_component
from dace.subsets import Range
from dace.transformation.dataflow.map_collapse import MapCollapse
from dace.transformation.dataflow.map_expansion import MapExpansion
from dace.transformation.dataflow.map_fission import MapFission
from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.transformation.dataflow.map_fusion_horizontal import MapFusionHorizontal
from dace.transformation.dataflow.map_fusion_vertical import MapFusionVertical
from dace.transformation.dataflow.map_interchange import MapInterchange
from dace.transformation.dataflow.map_unroll import MapUnroll
from dace.transformation.dataflow.otf_map_fusion import OTFMapFusion
from dace.transformation.dataflow.stage_loads_shared_memory import StageLoadsThroughSharedMemory
from dace.transformation.dataflow.tiling import MapTiling
from dace.transformation.dataflow.trivial_map_elimination import TrivialMapElimination
from dace.transformation.helpers import nest_state_subgraph
from dace.transformation.interstate import InlineSDFG, MapLoopInterchange, SubgraphFission
from dace.transformation.interstate.loop_fusion import LoopFusion
from dace.transformation.interstate.loop_tiling import LoopTiling
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.interstate.loop_unroll import LoopUnroll
from dace.transformation.interstate.move_loop_into_map import MoveLoopIntoMap, analyze_lanes, single_map_body
from dace.transformation.interstate.move_loop_invariant_if_up import MoveLoopInvariantIfUp, match_loop
from dace.transformation.interstate.move_map_invariant_if_up import MoveMapInvariantIfUp
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
from dace.transformation.passes.config_propagation import ConfigPropagation
from dace.transformation.passes.cpu_specialization.interchange_loops_into_lane_maps import inside_a_map, lane_axis_of
from dace.transformation.passes.dead_dataflow_elimination import DeadDataflowElimination
from dace.transformation.passes.gpu_specialization.block_tile_kernels import is_kernel
from dace.transformation.passes.loop_fission import LoopFission
from dace.transformation.passes.loop_invariant_code_motion import (
    LoopInvariantCodeMotion,
    hoist_loop_region,
    hoist_map_scope,
    loop_has_invariant_code,
    map_has_invariant_code,
)
from dace.transformation.passes.move_if_into_loop import MoveIfIntoLoop, match_guard
from dace.transformation.passes.parallelization.wavefront.skew import WavefrontSkew
from dace.transformation.passes.parallelization.wavefront.statements import occurrence_index
from dace.transformation.passes.vectorization.vectorize_map import VectorizeMap
from dace.transformation.transformation import PatternTransformation

from nestforge.ir.extract import detach, detached_twin, extract_cfg_nest, extract_map_nest, find_state_of_node
from nestforge.ir.introspect import Row
from nestforge.ir.names import in_order, top_level_map_entries
from nestforge.records import CRASHES, Record, Refusal, Refused
from nestforge.stages.expand import expand_library, expansion, set_implementation
from nestforge.stages.layout import RELAYOUT
from nestforge.stages.lib_motion import LIB_AFTER, LIB_BEFORE, LIB_INTO, movable_rows, plan_lib_motion
from nestforge.stages.sinking import (
    SinkPlan,
    match_if_over_maps,
    predecessor_state,
    push_if_into_maps,
    sink_into_loop,
    sink_into_map,
)
from nestforge.stages.sinking import (
    plan_loop as plan_sink_loop,
)
from nestforge.stages.sinking import (
    plan_map as plan_sink_map,
)
from nestforge.stages.snapshot import REFUSED as SNAPSHOT_REFUSED
from nestforge.stages.snapshot import reads_another_doalls_write, snapshot, snapshot_applies
from nestforge.stages.vectorize import DEFAULT_LANES, VECTORIZE_DOALL, vectorize_config, vectorize_refusal


@dataclass(frozen=True, slots=True)
class Rewrite:
    """A legal move; ``commit`` applies it and ``name`` is the transformation that runs."""

    name: str
    commit: Callable[[], object]


#: The move kind that turns a library call into maps, which the session then canonicalizes.
INLINE_LIBRARY = "inline-library"
#: The move kind that picks a library call's implementation; it takes ``implementation``.
SET_IMPLEMENTATION = "set-library-implementation"
#: The move kind, on no row, that specializes the program for argument values; it takes ``config_values``.
PROPAGATE_CONFIG = "propagate-config"
#: The move, on no row, that fixes the program's floating-point mode; it takes ``fp_mode``.
SET_FP_MODE = "set-fp-mode"
#: The move that runs consecutive rows of kernel ``scope`` (for loops, states, doalls for their states) as one
#: persistent kernel; it takes one row or more.
PERSISTENT_KERNEL = "persistent-kernel"
#: The move, on no row, that stores one internal array in another floating-point type; it takes ``container`` and
#: ``dtype``.
CHANGE_PRECISION = "change-precision"
#: The move kind, on no row, that takes the last change back; the session keeps the programs it returns to.
ROLLBACK = "rollback"
#: The move kind that turns a sequential loop into a doall; it takes ``force``.
FOR_TO_DOALL = "for-to-doall"
#: What a forced ``for-to-doall`` answers as the transformation that ran.
FORCED_LOOP_TO_MAP = "LoopToMap, forced: the compiler did not prove the iterations independent"
#: The move kind that sets the OpenMP schedule of a doall inside a kernel placed on the CPU.
SET_SCHEDULE_CPU = "set-schedule-cpu"
#: The schedules ``set-schedule-cpu`` takes, by the name the agent sends.
CPU_SCHEDULES = {
    "default": dtypes.OMPScheduleType.Default,
    "static": dtypes.OMPScheduleType.Static,
    "dynamic": dtypes.OMPScheduleType.Dynamic,
    "guided": dtypes.OMPScheduleType.Guided,
}
#: The move kind that sets the thread block of a GPU kernel, or how the block's threads share a doall inside it.
SET_SCHEDULE_GPU = "set-schedule-gpu"
#: The schedules ``set-schedule-gpu`` takes on a doall inside a kernel, as DaCe schedules; ``guided`` waits for DaCe.
GPU_INNER_SCHEDULES = {
    "sequential": dtypes.ScheduleType.Sequential,
    "static": dtypes.ScheduleType.GPU_ThreadBlock,
    "dynamic": dtypes.ScheduleType.GPU_ThreadBlock_Dynamic,
}
#: The most threads a block has on every GPU NestForge targets.
MAX_BLOCK_THREADS = 1024
#: The move kinds that work only inside a kernel (a scope): a schedule belongs to the device it is placed on.
SCOPE_KINDS = frozenset({SET_SCHEDULE_CPU, SET_SCHEDULE_GPU})
#: The move kind that skews a loop nest into a wavefront: a sequential diagonal loop over a doall per diagonal.
SKEW_WAVEFRONT = "skew-wavefront"
#: The move kind that runs the loop-invariant statements of a for loop or a doall once, before it.
HOIST_INVARIANT_CODE = "hoist-invariant-code"
#: The move kind that moves a for loop's or a doall's invariant guard above it: ``if c`` then runs once.
HOIST_INVARIANT_GUARD = "hoist-invariant-guard"
#: The move kind that moves an if into the for loop or doalls it guards, the inverse of :data:`HOIST_INVARIANT_GUARD`.
SINK_INVARIANT_GUARD = "sink-invariant-guard"
#: The move kind that moves the statements computing one temporary into the for loop or doall that reads it, the
#: inverse of :data:`HOIST_INVARIANT_CODE`; it takes ``container``.
SINK_INVARIANT_CODE = "sink-invariant-code"
#: A row a move takes as either a for loop or a doall.
LOOP_ROW = (LoopRegion, nodes.MapEntry)
#: A row of a persistent run: a for loop, a state, or a doall standing for its state.
RUN_ROW = (LoopRegion, SDFGState, nodes.MapEntry)

#: Every move kind, with the tree rows it takes in order.
MOVE_SHAPES: dict[str, tuple[type | tuple[type, ...], ...]] = {
    "for-fusion": (LoopRegion, LoopRegion),
    "for-fission": (LoopRegion,),
    "doall-fusion": (nodes.MapEntry, nodes.MapEntry),
    "doall-fission": (nodes.MapEntry,),
    "subgraph-fission": (nodes.MapEntry, ControlFlowBlock),
    "interchange-for-doall": (LoopRegion, nodes.MapEntry),
    "interchange-doall-for": (nodes.MapEntry, LoopRegion),
    "interchange-doall-doall": (nodes.MapEntry, nodes.MapEntry),
    INLINE_LIBRARY: (nodes.LibraryNode,),
    SET_IMPLEMENTATION: (nodes.LibraryNode,),
    "tile-doall": (nodes.MapEntry,),
    "untile-doall": (nodes.MapEntry, nodes.MapEntry),
    "flatten-doalls": (nodes.MapEntry, nodes.MapEntry),
    "unflatten-doall": (nodes.MapEntry,),
    "doall-to-for": (nodes.MapEntry,),
    "snapshot-reads": (nodes.MapEntry,),
    "tile-for": (LoopRegion,),
    "untile-for": (LoopRegion,),
    "unroll-for": (LoopRegion,),
    "unroll-doall": (nodes.MapEntry,),
    HOIST_INVARIANT_CODE: (LOOP_ROW,),
    SINK_INVARIANT_CODE: (LOOP_ROW,),
    HOIST_INVARIANT_GUARD: (LOOP_ROW,),
    SINK_INVARIANT_GUARD: (ConditionalBlock,),
    VECTORIZE_DOALL: (nodes.MapEntry,),
    LIB_BEFORE: (nodes.CodeNode, ControlFlowBlock),
    LIB_AFTER: (nodes.CodeNode, ControlFlowBlock),
    LIB_INTO: (nodes.CodeNode, SDFGState),
    PROPAGATE_CONFIG: (),
    SET_FP_MODE: (),
    CHANGE_PRECISION: (),
    PERSISTENT_KERNEL: (RUN_ROW,),
    RELAYOUT: (),
    ROLLBACK: (),
    "stage-loads-shared": (nodes.MapEntry,),
    SKEW_WAVEFRONT: (LoopRegion,),
    FOR_TO_DOALL: (LoopRegion,),
    SET_SCHEDULE_CPU: (nodes.MapEntry,),
    SET_SCHEDULE_GPU: (nodes.MapEntry,),
}

#: The move kinds that rewrite a GPU kernel's offloaded body, inside a scope placed on the GPU.
DEVICE_KINDS = frozenset({"stage-loads-shared"})

#: The ISA ``vectorize-doall`` vectorizes for when none is given: the best one the host runs.
DEFAULT_ISA = "auto"

#: The move kinds listed with probe sizes: one positive size per dimension of the kernel, in its order (for
#: ``stage-loads-shared``, the thread block: one size per dimension, at most three).
SIZED = frozenset({"tile-doall", "tile-for", "stage-loads-shared"})
#: The size a sized move is checked at when it is listed; the agent passes its own to apply it.
PROBE_TILE = 32
#: How many tile extents ``skew-wavefront`` takes: along the skewed loop, and along the diagonal.
SKEW_TILE_DIMS = 2
#: The move kinds whose sizes are vector lanes (``lanes``), and those whose sizes are a GPU thread block (``block``);
#: every other sized kind takes ``tile_sizes``. One argument name, one meaning.
LANED = frozenset({VECTORIZE_DOALL})
BLOCKED = frozenset({"stage-loads-shared", SET_SCHEDULE_GPU})


def size_argument(kind: str) -> str:
    """The name of the sizes argument ``kind`` takes: ``lanes``, ``block`` or ``tile_sizes``."""
    return "lanes" if kind in LANED else "block" if kind in BLOCKED else "tile_sizes"


def chosen_sizes(
    kind: str, tile_sizes: Sequence[int], lanes: Sequence[int], block: Sequence[int]
) -> tuple[int, ...] | Refused:
    """The sizes ``kind`` was given, from the one argument it takes; refused when another one carries sizes."""
    given = {"tile_sizes": tuple(tile_sizes), "lanes": tuple(lanes), "block": tuple(block)}
    wanted = size_argument(kind)
    wrong = [name for name, sizes in given.items() if sizes and name != wanted]
    if wrong:
        return Refused(f"{kind} takes its sizes as {wanted}, not {', '.join(wrong)}.", Refusal.ARGUMENT)
    return given[wanted]


#: Move kinds renamed since runs were journaled: old name -> current one.
RENAMED_KINDS = {
    "collapse-doalls": "flatten-doalls",
    "expand-doall": "unflatten-doall",
    "expand-library": "inline-library",
    "auto-vectorize-scope": "vectorize-doall",
    "hoist-invariant-for": "hoist-invariant-code",
    "hoist-invariant-doall": "hoist-invariant-code",
}


def journaled_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """A journaled move's arguments in today's names: a renamed kind, and sizes under the argument the kind takes."""
    kind = arguments.get("kind")
    if not isinstance(kind, str):
        return arguments
    current = RENAMED_KINDS.get(kind, kind)
    upgraded = {**arguments, "kind": current}
    wanted = size_argument(current)
    if wanted != "tile_sizes" and upgraded.get("tile_sizes"):
        upgraded[wanted] = upgraded.pop("tile_sizes")
    return upgraded


#: The move kinds that take ``max_iterations``: the largest trip count the agent accepts to unroll.
UNROLLS = frozenset({"unroll-for", "unroll-doall"})
#: The cap an unroll is checked at when it is listed; the agent passes its own to apply it.
PROBE_UNROLL = 16


def probe_sizes(kind: str, rows: Sequence[Row]) -> tuple[int, ...]:
    """The tile sizes :func:`legal_moves` checks a sized move with: :data:`PROBE_TILE` in every dimension; the lanes
    ``vectorize-doall`` is listed with."""
    if kind == VECTORIZE_DOALL:
        return (DEFAULT_LANES,)
    if kind not in SIZED:
        return ()
    head = rows[0][0]
    dims = len(head.map.params) if isinstance(head, nodes.MapEntry) else 1
    return (PROBE_TILE,) * (min(dims, 3) if kind == "stage-loads-shared" else dims)


def probe_implementation(kind: str, rows: Sequence[Row]) -> str | None:
    """The implementation :func:`legal_moves` checks :data:`SET_IMPLEMENTATION` with: any the node has; the session
    lists one move per implementation it may take."""
    if kind != SET_IMPLEMENTATION:
        return None
    return next(iter(cast(nodes.LibraryNode, rows[0][0]).implementations), None)


STATE_BARRIER = Refused(
    "nests are in different states, a control-flow dependency map fusion never crosses; fuse the loops around "
    "them, or define one scope over their blocks.",
    Refusal.WRONG_ROW_KIND,
)
LOOP_FISSION_REFUSED = Refused(
    "blocked by LoopFission: the loop body has no two independent statement groups.", Refusal.DEPENDENCE
)
MAP_FISSION_REFUSED = Refused(
    "blocked by MapFission: the map body is not one nested SDFG with independent output groups.", Refusal.DEPENDENCE
)
MAP_INTERCHANGE_REFUSED = Refused(
    "blocked by MapInterchange: the inner range reads the outer parameter, or the outer map holds more than the "
    "inner map.",
    Refusal.DEPENDENCE,
)
LOOP_INTO_MAP_REFUSED = Refused(
    "blocked by MoveLoopIntoMap: the loop bounds are not analyzable, the body is more than one state around the map, "
    "or a dependence would cross map iterations once the map is outer.",
    Refusal.DEPENDENCE,
)
IF_INTO_LOOP_REFUSED = Refused(
    "blocked: the branch is not a single no-else guard whose body ends in one for loop (MoveIfIntoLoop) or is one "
    "state of doalls (push_if_into_maps), or the condition or its prep reads what the loop or doalls write.",
    Refusal.DEPENDENCE,
)
IF_OUT_OF_LOOP_REFUSED = Refused(
    "blocked by MoveLoopInvariantIfUp: the guard is not the loop's only statement, or its condition reads the loop "
    "variable or data the loop writes.",
    Refusal.DEPENDENCE,
)
SUBGRAPH_FISSION_REFUSED = Refused(
    "blocked by SubgraphFission: the body branches, the edge after the cut assigns or branches, or MapFission "
    "refuses (a half holds a conditional, or an interstate assignment reads a map parameter).",
    Refusal.DEPENDENCE,
)

MAP_IF_REFUSED = (
    "blocked by MoveMapInvariantIfUp: {map} holds no if that reads neither its parameters nor what its body writes."
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
        return Refused(
            "loops are in different control-flow regions; fuse the enclosing loops first.", Refusal.WRONG_ROW_KIND
        )
    out = loop_a.parent_graph.out_edges(loop_a)
    if len(out) != 1 or out[0].dst is not loop_b:
        return Refused(
            "loops are not adjacent: exactly one sequencing edge must lead from the first to the second.",
            Refusal.WRONG_ROW_KIND,
        )
    sdfg, where = loop_a.sdfg, {"first": loop_a, "second": loop_b}
    move = pattern(LoopFusion, sdfg, where)
    if move is None and keeps_a_doall(loop_a) and keeps_a_doall(loop_b):
        move = pattern(LoopFusion, sdfg, where, options=UNGUARDED_LOOP_FUSION)
    if move is None:
        return Refused(
            "blocked by LoopFusion: different iteration ranges, or a loop-carried dependency between the two.",
            Refusal.DEPENDENCE,
        )
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
        return Refused("; ".join(reasons) + ".", Refusal.DEPENDENCE)
    if state.entry_node(a) is not state.entry_node(b):
        return Refused("maps are in different scopes with no shared data; not a fusion pair.", Refusal.WRONG_ROW_KIND)
    move = pattern(MapFusionHorizontal, sdfg, {"first_parallel_map_entry": a, "second_parallel_map_entry": b})
    if move is not None:
        return move
    if a.map.range != b.map.range:
        return Refused(
            f"different map ranges {a.map.range} and {b.map.range}; horizontal fusion needs one range.",
            Refusal.DEPENDENCE,
        )
    return Refused(
        "blocked by MapFusionHorizontal: not both parallel-compatible, or a data dependency links them.",
        Refusal.DEPENDENCE,
    )


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
        return Refused("the map body is one computation; there is nothing to split.", Refusal.WRONG_ROW_KIND)
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
        return Refused(
            f"{cut.label} is not a top-level block of the body of {entry.map.label}.", Refusal.WRONG_ROW_KIND
        )
    if not body.sdfg.out_edges(cut):
        return Refused(
            f"{cut.label} is the last block of the body of {entry.map.label}; name a block before the last.",
            Refusal.WRONG_ROW_KIND,
        )
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
    """Inline the nested SDFG that is the body of ``entry`` and drop the one-iteration dimensions of its maps (a map
    left with none goes); the maps now directly inside ``entry``."""
    body = map_body(state, entry)
    assert body is not None, "the caller checked the body"
    InlineSDFG.apply_to(state.sdfg, nested_sdfg=body, verify=False)
    root = state.sdfg.root_sdfg
    set_nested_sdfg_parent_references(root)
    root.reset_cfg_list()
    for inner in [n for n in state.scope_children()[entry] if isinstance(n, nodes.MapEntry)]:
        if any(begin == end for begin, end, _ in inner.map.range):
            TrivialMapElimination.apply_to(state.sdfg, map_entry=inner, verify=False)
    return [n for n in state.scope_children()[entry] if isinstance(n, nodes.MapEntry)]


def body_swap_blocker(state: SDFGState, entry: nodes.MapEntry) -> str | None:
    """Why the map directly inside the body of ``entry`` cannot be swapped with it, or ``None``; this inlines the body
    in ``state``, so it runs on a detached twin."""
    body = map_body(state, entry)
    if body is None or not InlineSDFG.can_be_applied_to(state.sdfg, nested_sdfg=body):
        return Refused(f"blocked by InlineSDFG: the body of {entry.map.label} cannot be inlined.", Refusal.DEPENDENCE)
    inner = flatten_map_body(state, entry)
    beside = (nodes.MapEntry, nodes.MapExit, nodes.AccessNode)
    holds = [n for n in state.scope_children()[entry] if not isinstance(n, beside)]
    if len(inner) != 1 or holds or scope_copies(state, entry):
        return Refused(
            f"the body of {entry.map.label} holds more than one map or a computation or copy beside it; "
            "doall-fission or subgraph-fission it first.",
            Refusal.WRONG_ROW_KIND,
        )
    where = {"outer_map_entry": entry, "inner_map_entry": inner[0]}
    return None if pattern(MapInterchange, state.sdfg, where) else MAP_INTERCHANGE_REFUSED


def plan_body_map_map(state: SDFGState, outer: nodes.MapEntry, inner_row: Row) -> Rewrite | str:
    """``map j: {nested SDFG: map k}`` -> ``map k: map j``: the body is inlined first (and its one-iteration maps
    dropped), so the maps are directly nested; legality is read on a copy."""
    inner, inner_state = inner_row[0], node_state(inner_row)
    body = map_body(state, outer)
    if body is None or inner_state.sdfg is not body.sdfg or len(body.sdfg.nodes()) != 1:
        return Refused(
            f"{inner.map.label} is not directly inside {outer.map.label}, nor the one map in its body; name a map, "
            "then the map directly inside it.",
            Refusal.WRONG_ROW_KIND,
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
        return Refused(
            f"{inner.map.label} is not directly inside {outer.map.label}; name a map, then the map directly inside it.",
            Refusal.WRONG_ROW_KIND,
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
        return Refused(
            f"{inner.map.label} has one iteration only; there is nothing to swap with {outer.map.label}.",
            Refusal.WRONG_ROW_KIND,
        )
    where = {"outer_map_entry": twin_outer, "inner_map_entry": twin_inner}
    if pattern(MapInterchange, twin, where) is None:
        return MAP_INTERCHANGE_REFUSED

    def commit() -> None:
        TrivialMapElimination.apply_to(state.sdfg, map_entry=inner, verify=False)
        MapInterchange.apply_to(state.sdfg, outer_map_entry=outer, inner_map_entry=inner, verify=False)

    return Rewrite("MapInterchange", commit)


#: MoveLoopIntoMap's control-flow body: the maps of the loop's body, behind ifs or in several states, share one axis.
LANE_BODY = {"cfg_body": True, "lane_axis": True}


def loop_map_refusal(loop: LoopRegion, inside: list[nodes.MapEntry]) -> str:
    if inside_a_map(loop):
        return Refused(
            f"{loop.label} is inside a doall; move the loop of the outermost doall's body instead.",
            Refusal.WRONG_ROW_KIND,
        )
    if len(inside) > 1:
        if lane_axis_of(loop) is None:
            return Refused(
                f"the doalls inside {loop.label} share no axis; doall-fusion or doall-fission them first.",
                Refusal.WRONG_ROW_KIND,
            )
        lanes = analyze_lanes(loop, loop.sdfg, True)
        if lanes.refusal is not None:
            return Refused(f"blocked by MoveLoopIntoMap: {lanes.refusal}", Refusal.DEPENDENCE)
    return LOOP_INTO_MAP_REFUSED


def plan_loop_map(loop_row: Row, map_row: Row) -> Rewrite | str:
    """``for t: doall i: B`` -> ``doall i: for t: B``. A body of several maps over one shared axis, in several states
    or behind ifs, takes the same move (the lane analysis of ``MoveLoopIntoMap``): ``doall i: for t: {A; B; ...}``;
    ``map_row`` names one of the maps."""
    loop, entry = loop_row[0], map_row[0]
    inside = loop_maps(loop)
    if entry not in inside:
        return Refused(f"{entry.map.label} is not a doall directly inside {loop.label}.", Refusal.WRONG_ROW_KIND)
    # MoveLoopIntoMap passes a nest only the SDFG's symbols and data, and raises on an outer loop's iterator
    read = {str(s) for s in loop.free_symbols} - set(loop.sdfg.symbols)
    if read & enclosing_loop_variables(loop):
        return Refused(
            f"blocked by MoveLoopIntoMap: {loop.label} reads the iterator of a loop around it; untile-for it first.",
            Refusal.DEPENDENCE,
        )
    sdfg = loop.sdfg
    move = pattern(MoveLoopIntoMap, sdfg, {"loop": loop}, options=LANE_BODY)
    if move is None:
        return loop_map_refusal(loop, inside)
    if single_map_body(loop) is not None:
        return move
    return region_rewrite(move.name, sdfg, lambda: (move.commit(), propagate_memlets_sdfg(sdfg)))


def plan_map_loop(entry_row: Row, loop_row: Row) -> Rewrite | str:
    """``map i: for t: B`` -> ``for t: map i: B``, when the loop is the whole body of the map."""
    state, entry, loop = node_state(entry_row), entry_row[0], loop_row[0]
    body = map_body(state, entry)
    if body is None or body.sdfg.nodes() != [loop]:
        return Refused(
            f"the body of {entry.map.label} is not exactly the loop {loop.label}; subgraph-fission the map around the "
            "loop first, so the loop is the whole body of a map of its own.",
            Refusal.WRONG_ROW_KIND,
        )
    if state.entry_node(entry) is not None:
        return plan_nested_map_loop(state, entry)
    xform = map_loop_interchange(state, entry, body)
    reason = xform.refusal(state)
    if reason is not None:
        return Refused(f"blocked by MapLoopInterchange: {reason}.", Refusal.DEPENDENCE)
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
        return Refused(f"blocked by MapLoopInterchange: {reason}.", Refusal.DEPENDENCE)

    def commit() -> None:
        inner, inner_entry, body = nest_map(state, entry)
        map_loop_interchange(inner, inner_entry, body).apply(inner, inner.sdfg)

    return Rewrite("MapLoopInterchange", commit)


def plan_sink_invariant_guard(cond_row: Row) -> Rewrite | str:
    """``if c: for i: B`` -> ``for i: if c: B``, for the one loop the branch ends in; ``if c: doall i: B`` ->
    ``doall i: if c: B``, for every doall of the branch's one state."""
    cond = cast(ConditionalBlock, cond_row[0])
    if ends_in_one_loop(cond):
        return region_rewrite("MoveIfIntoLoop", cond.sdfg, lambda: MoveIfIntoLoop.push(cond))
    if match_if_over_maps(cond) is not None:
        return region_rewrite("push_if_into_maps", cond.sdfg, lambda: push_if_into_maps(cond))
    return IF_INTO_LOOP_REFUSED


def ends_in_one_loop(cond: ConditionalBlock) -> bool:
    """Whether ``cond`` is a guard MoveIfIntoLoop takes: one branch that ends in one for loop."""
    match = match_guard(cond)
    sinks = match[2].sink_nodes() if match is not None else []
    return len(sinks) == 1 and isinstance(sinks[0], LoopRegion)


def plan_hoist_invariant_guard(row: Row) -> Rewrite | str:
    """The invariant guard of the for loop or doall ``row`` names moves above it."""
    if isinstance(row[0], LoopRegion):
        return plan_loop_if(row)
    return plan_map_if(row)


def plan_loop_if(loop_row: Row) -> Rewrite | str:
    """``for i: if c: B`` -> ``if c: for i: B``, for a loop-invariant guard."""
    loop = loop_row[0]
    if match_loop(loop.sdfg, loop) is None:
        return IF_OUT_OF_LOOP_REFUSED
    return region_rewrite("MoveLoopInvariantIfUp", loop.sdfg, lambda: MoveLoopInvariantIfUp.hoist(loop))


def plan_inline_library(row: Row) -> Rewrite | str:
    node, state = cast(nodes.LibraryNode, row[0]), node_state(row)
    if isinstance(node, ExternalCall):
        return Refused(
            f"{node.label} is a kernel NestForge built (a scope), not a library call.", Refusal.WRONG_ROW_KIND
        )
    implementation = expansion(node)
    if implementation is None:
        return Refused(
            f"{node.label} ({type(node).__name__}) has no vendor-free expansion (only {sorted(node.implementations)}); "
            "it stays a library call.",
            Refusal.WRONG_ROW_KIND,
        )
    return Rewrite(f"{type(node).__name__} expansion ({implementation})", lambda: expand_library(state, node))


def plan_set_implementation(row: Row, implementation: str | None = None) -> Rewrite | str:
    node = cast(nodes.LibraryNode, row[0])
    if isinstance(node, ExternalCall):
        return Refused(
            f"{node.label} is a kernel NestForge built (a scope), not a library call.", Refusal.WRONG_ROW_KIND
        )
    if implementation is None:
        return Refused(f"pass implementation, one of {sorted(node.implementations)}.", Refusal.ARGUMENT)
    if implementation not in node.implementations:
        return Refused(
            f"{node.label} ({type(node).__name__}) has no implementation {implementation!r}; it has {sorted(node.implementations)}.",
            Refusal.ARGUMENT,
        )
    return Rewrite(
        f"{type(node).__name__} implementation {implementation}", lambda: set_implementation(node, implementation)
    )


def plan_tile_doall(row: Row, tile_sizes: Sequence[int] = ()) -> Rewrite | str:
    entry, state = cast(nodes.MapEntry, row[0]), node_state(row)
    dims = len(entry.map.params)
    if len(tile_sizes) != dims or any(size < 1 for size in tile_sizes) or max(tile_sizes) < 2:
        return Refused(
            f"{entry.map.label} has {dims} dimension(s) {entry.map.params}: pass tile_sizes, one positive size each "
            "(1 leaves a dimension untiled), at least one of 2 or more.",
            Refusal.ARGUMENT,
        )
    options = {"tile_sizes": tuple(tile_sizes), "divides_evenly": False}
    move = pattern(MapTiling, state.sdfg, {"map_entry": entry}, options=options)
    if move is None:
        return Refused(f"blocked by MapTiling: {entry.map.label} cannot be tiled.", Refusal.DEPENDENCE)

    def commit() -> None:
        move.commit()
        # MapTiling's approximate SymExpr bound is text no DaCe analysis reads back (work depth, the tile pair search)
        entry.map.range = Range([exact_bounds(rng) for rng in entry.map.range])

    return Rewrite(move.name, commit)


def plan_auto_vectorize(row: Row, tile_sizes: Sequence[int], isa: str) -> Rewrite | str:
    """Vectorize the top-level doall ``row`` for ``isa`` with ``tile_sizes`` lanes (:class:`VectorizeMap`), tried on a
    copy of the program first, so a refusal is an answer and changes nothing."""
    entry, state = cast(nodes.MapEntry, row[0]), node_state(row)
    config = vectorize_config(isa, tile_sizes)
    if isinstance(config, str):
        return config
    refused = vectorize_refusal(state, entry, config)
    if refused is not None:
        return refused
    return region_rewrite(
        "VectorizeMap", state.sdfg, lambda: VectorizeMap(config, state, entry).apply_pass(state.sdfg, {})
    )


def plan_stage_loads_shared(row: Row, tile_sizes: Sequence[int] = ()) -> Rewrite | str:
    """A GPU kernel stages the tiles of the arrays its threads reuse through shared memory, one thread block of
    ``tile_sizes`` threads per tile."""
    entry, state = cast(nodes.MapEntry, row[0]), node_state(row)
    dims = min(len(entry.map.params), 3)
    if len(tile_sizes) != dims or any(size < 1 for size in tile_sizes):
        return Refused(
            f"{entry.map.label} has {len(entry.map.params)} dimension(s) {entry.map.params}: pass block, the "
            f"thread block, {dims} positive size(s).",
            Refusal.ARGUMENT,
        )
    options = {"block_size": list(tile_sizes)}
    move = pattern(StageLoadsThroughSharedMemory, state.sdfg, {"map_entry": entry}, options=options)
    if move is None:
        return Refused(
            f"blocked by StageLoadsThroughSharedMemory: {entry.map.label} is no GPU kernel (place its scope on the GPU "
            "and insert the copies first), or reads no array it reuses with a constant footprint per thread.",
            Refusal.DEPENDENCE,
        )
    return move


def plan_snapshot_reads(row: Row) -> Rewrite | str:
    """A doall that reads a slice of an array another doall of its state writes copies the slice first, so the two
    stop touching one array and ``doall-fusion`` can merge them."""
    entry, state = cast(nodes.MapEntry, row[0]), node_state(row)
    if state.entry_node(entry) is not None:
        return Refused(
            f"{entry.map.label} is inside another doall; only an outermost doall snapshots its reads.",
            Refusal.WRONG_ROW_KIND,
        )
    if not snapshot_applies(state, entry):
        return SNAPSHOT_REFUSED
    return Rewrite("SnapshotReads", lambda: snapshot(state, entry))


def tile_pair(outer: nodes.MapEntry, inner: nodes.MapEntry, state: SDFGState) -> MapTilePair | None:
    return next((pair for pair in map_tile_pairs(state) if pair.outer is outer and pair.inner is inner), None)


def plan_untile_doall(outer_row: Row, inner_row: Row) -> Rewrite | str:
    outer, inner, state = cast(nodes.MapEntry, outer_row[0]), cast(nodes.MapEntry, inner_row[0]), node_state(outer_row)
    pair = tile_pair(outer, inner, state) if node_state(inner_row) is state else None
    if pair is None:
        return Refused(
            f"{outer.map.label} is not a tile doall whose points are {inner.map.label}: name the doall stepping over "
            "tiles, then the doall over one tile's points.",
            Refusal.WRONG_ROW_KIND,
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


def plan_flatten_doalls(outer_row: Row, inner_row: Row) -> Rewrite | str:
    outer, inner, state = outer_row[0], inner_row[0], node_state(outer_row)
    if node_state(inner_row) is not state or state.entry_node(inner) is not outer:
        return Refused(
            f"{inner.map.label} is not directly inside {outer.map.label}; name a doall, then the one inside it.",
            Refusal.WRONG_ROW_KIND,
        )
    where = {"outer_map_entry": outer, "inner_map_entry": inner}
    return pattern(MapCollapse, state.sdfg, where) or (
        "blocked by MapCollapse: the outer doall holds more than the inner one, or the inner range reads the outer "
        "parameter."
    )


def plan_unflatten_doall(row: Row) -> Rewrite | str:
    entry, state = cast(nodes.MapEntry, row[0]), node_state(row)
    if len(entry.map.params) < 2:
        return Refused(f"{entry.map.label} has one dimension; there is nothing to expand.", Refusal.WRONG_ROW_KIND)
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
        return Refused(
            f"{entry.map.label} is inside another doall; only an outermost doall becomes a loop.",
            Refusal.WRONG_ROW_KIND,
        )
    if len(entry.map.params) != 1:
        return Refused(
            f"{entry.map.label} has {len(entry.map.params)} dimensions; unflatten-doall it first.",
            Refusal.WRONG_ROW_KIND,
        )
    if not holds_a_doall(state, entry):
        return Refused(
            f"{entry.map.label} holds no doall inside: as a loop it would keep no parallelism.", Refusal.WRONG_ROW_KIND
        )
    sdfg, region = state.sdfg, state.parent_graph
    move = pattern(MapToForLoop, sdfg, {"map_entry": entry})
    if move is None:
        return Refused(f"blocked by MapToForLoop: {entry.map.label} cannot become a loop.", Refusal.DEPENDENCE)
    sequential = move

    def commit() -> None:
        before = set(region.nodes())
        sequential.commit()
        drop_empty_states(region, [state, *(b for b in region.nodes() if b not in before)])

    return region_rewrite(sequential.name, sdfg, commit)


def plan_tile_for(row: Row, tile_sizes: Sequence[int] = ()) -> Rewrite | str:
    loop = cast(LoopRegion, row[0])
    if len(tile_sizes) != 1 or tile_sizes[0] < 2:
        return Refused(f"{loop.label} is one for loop: pass tile_sizes with one size of at least 2.", Refusal.ARGUMENT)
    if any(isinstance(jump, BreakBlock) for jump in loop_analysis.loop_jumps(loop)):
        return Refused(
            f"{loop.label} holds a break, which would leave one tile and not the loop.", Refusal.WRONG_ROW_KIND
        )
    where, options = {"loop": loop}, {"tile_size": tile_sizes[0]}
    if not LoopTiling.can_be_applied_to(loop.sdfg, options=options, **where):
        return Refused(f"blocked by LoopTiling: {loop.label} is not a counted loop of stride 1.", Refusal.DEPENDENCE)
    return region_rewrite("LoopTiling", loop.sdfg, lambda: LoopTiling.apply_to(loop.sdfg, options=options, **where))


def plan_untile_for(row: Row) -> Rewrite | str:
    loop = cast(LoopRegion, row[0])
    # the untiling decides legality as it rewrites, so it is tried on a copy first
    trial = copy.deepcopy(loop.sdfg)
    twin = next(r for r in trial.all_control_flow_regions(recursive=True) if r.label == loop.label)
    if not isinstance(twin, LoopRegion) or not UntileLoops().untile_loop(twin, trial):
        return Refused(
            f"{loop.label} is not a loop over tiles holding the loop over one tile's points.", Refusal.WRONG_ROW_KIND
        )

    def commit() -> None:
        # the copy has no parent, so it can untile what the program, with its parent's symbols, refuses
        if not UntileLoops().untile_loop(loop, loop.sdfg):
            raise ValueError(f"UntileLoops refused {loop.label} in place, though it untiled a parentless copy.")
        propagate_memlets_sdfg(loop.sdfg.root_sdfg)

    return region_rewrite("UntileLoops.untile_loop", loop.sdfg, commit)


def constant(expr: Any, sdfg: dace.SDFG) -> int | None:
    """``expr`` evaluated with ``sdfg``'s constants, or ``None`` while it reads a symbol with no value."""
    try:
        return int(dace.symbolic.evaluate(expr, sdfg.constants))
    except TypeError:
        return None


def loop_trips(loop: LoopRegion) -> int | None:
    """How many iterations ``loop`` runs, counted as ``LoopUnroll`` counts them; ``None`` when not constant."""
    start, end = loop_analysis.get_init_assignment(loop), loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or stride is None:
        return None
    diff, step = constant(cast(sympy.Expr, end) - start, loop.sdfg), constant(stride, loop.sdfg)
    if diff is None or not step:
        return None
    return len(range(0, diff + (1 if step > 0 else -1), step))


def unroll_refusal(label: str, trips: int | None, max_iterations: int) -> str | None:
    if trips is None:
        return Refused(
            f"{label} has no constant trip count: its bounds read a symbol with no value; propagate-config a value "
            "for it first.",
            Refusal.WRONG_ROW_KIND,
        )
    if trips > max_iterations:
        return Refused(
            f"{label} runs {trips} iterations, more than max_iterations={max_iterations}; pass a larger cap.",
            Refusal.ARGUMENT,
        )
    return None


def plan_unroll_for(row: Row, max_iterations: int = PROBE_UNROLL) -> Rewrite | str:
    loop = cast(LoopRegion, row[0])
    reason = unroll_refusal(loop.label, loop_trips(loop), max_iterations)
    if reason is not None:
        return reason
    move = pattern(LoopUnroll, loop.sdfg, {"loop": loop})
    if move is None:
        return Refused(
            f"blocked by LoopUnroll: {loop.label} holds a break or continue, or is not a counted loop.",
            Refusal.DEPENDENCE,
        )
    return region_rewrite(move.name, loop.sdfg, move.commit)


def plan_unroll_doall(row: Row, max_iterations: int = PROBE_UNROLL) -> Rewrite | str:
    entry, state = cast(nodes.MapEntry, row[0]), node_state(row)
    if state.entry_node(entry) is not None:
        return Refused(
            f"{entry.map.label} is inside another doall of its state; MapUnroll unrolls only the outer one.",
            Refusal.WRONG_ROW_KIND,
        )
    reason = unroll_refusal(entry.map.label, constant(entry.map.range.num_elements_exact(), state.sdfg), max_iterations)
    if reason is not None:
        return reason
    # MapUnroll copies the whole connected part of the state: another computation in it would run once per copy
    scope = set(state.scope_subgraph(entry).nodes())
    beside = [n for n in weakly_connected_component(state, entry).nodes() if n not in scope]
    if any(not isinstance(n, nodes.AccessNode) for n in beside):
        return Refused(
            f"{entry.map.label} shares arrays with other computations of its state, which MapUnroll would copy with "
            "it; doall-fusion them first.",
            Refusal.WRONG_ROW_KIND,
        )
    return pattern(MapUnroll, state.sdfg, {"map_entry": entry}) or (
        f"blocked by MapUnroll: {entry.map.label} is a GPU doall."
    )


def config_names(sdfg: dace.SDFG) -> list[str]:
    """What ``propagate-config`` takes values for: the program's arguments and free symbols."""
    return [name for name, desc in sdfg.arrays.items() if not desc.transient] + sorted(map(str, sdfg.free_symbols))


def configurable(sdfg: dace.SDFG) -> bool:
    """Whether ``propagate-config`` is offered: ``sdfg`` has a scalar argument or a free symbol, and no kernel yet."""
    valued = bool(sdfg.free_symbols) or any(
        isinstance(desc, Scalar) and not desc.transient for desc in sdfg.arrays.values()
    )
    return valued and not external_calls(sdfg)


def loaded(root: dace.SDFG) -> dace.SDFG:
    """``root``, the program ConfigPropagation built, with the variants it calls loaded from their files."""
    for state in root.states():
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                node.load_external(state)
    return root


def plan_propagate_config(
    program: dace.SDFG, config_values: Mapping[str, int | float], output_dir: Path
) -> Rewrite | str:
    """A new program: ``if (args == values) <program specialized for them> else <program>``; the variants are written
    to ``output_dir``. ConfigPropagation leaves ``program`` as it is, so its refusal is read before anything changes."""
    if external_calls(program):
        return Refused(
            "the program has kernels: propagate-config specializes the program before its scopes.",
            Refusal.WRONG_ROW_KIND,
        )
    names = config_names(program)
    if not config_values:
        return Refused(f"pass config_values: a value for some of {names}.", Refusal.ARGUMENT)
    unknown = sorted(set(config_values) - set(names))
    if unknown:
        return Refused(f"{unknown} name no argument or free symbol of the program; it has {names}.", Refusal.ARGUMENT)
    domains: dict[str, Sequence[int | float]] = {name: [value] for name, value in config_values.items()}
    try:
        root = ConfigPropagation(domains=domains, output_dir=str(output_dir)).apply_pass(program, {})
    except ValueError as err:
        return Refused(f"blocked by ConfigPropagation: {err}.", Refusal.DEPENDENCE)
    return Rewrite("ConfigPropagation", lambda: loaded(root))


def state_of_map(sdfg: dace.SDFG, label: str) -> SDFGState:
    """The state holding the map labeled ``label``, anywhere in ``sdfg``."""
    for node, graph in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry) and node.map.label == label and isinstance(graph, SDFGState):
            return graph
    raise KeyError(label)


def guarded_doalls(sdfg: dace.SDFG) -> Iterator[tuple[Row, ...]]:
    """The doalls whose nested-SDFG body holds a conditional as one of its top-level blocks."""
    for state, entry in map_rows(sdfg):
        body = map_body(state, entry)
        if body is not None and any(isinstance(block, ConditionalBlock) for block in body.sdfg.nodes()):
            yield ((entry, state),)


def plan_map_if(row: Row) -> Rewrite | str:
    entry, state = cast(nodes.MapEntry, row[0]), node_state(row)
    # tried on a copy: a refused lift may still have split the doall's state off or nested it
    trial_state = state_of_map(copy.deepcopy(state.sdfg.root_sdfg), entry.map.label)
    (trial_entry,) = [
        n for n in trial_state.nodes() if isinstance(n, nodes.MapEntry) and n.map.label == entry.map.label
    ]
    if not MoveMapInvariantIfUp().apply_to_map(trial_state, trial_entry):
        return Refused(MAP_IF_REFUSED.format(map=entry.map.label), Refusal.DEPENDENCE)
    return region_rewrite("MoveMapInvariantIfUp", state.sdfg, lambda: MoveMapInvariantIfUp().apply_to_map(state, entry))


def plan_lib_move(row: Row, target: Row, place: str) -> Rewrite | str:
    plan = plan_lib_motion(row, target, place)
    if isinstance(plan, str):
        return plan
    return region_rewrite(f"lib-{place}", node_state(row).sdfg, plan)


def plan_skew_wavefront(row: Row, tile_sizes: Sequence[int] = (), target: str = "cpu") -> Rewrite | str:
    """Skew the nest of loop ``row`` so each diagonal runs as a doall (DaCe's :class:`WavefrontSkew`, which derives
    the skew from the dependences), tiled along the diagonal where it can; ``tile_sizes``, two extents, replace the
    tiling's defaults."""
    loop = cast(LoopRegion, row[0])
    if tile_sizes and (len(tile_sizes) != SKEW_TILE_DIMS or min(tile_sizes) < 1):
        return Refused(
            f"{SKEW_WAVEFRONT} takes tile_sizes as {SKEW_TILE_DIMS} positive extents (along {loop.label}, then along "
            "the diagonal), or none for DaCe's defaults.",
            Refusal.ARGUMENT,
        )
    skew = WavefrontSkew(target=target)
    if tile_sizes:
        if target == "gpu":
            skew.gpu_tile_i, skew.gpu_tile_j = tile_sizes
        else:
            skew.tile_i, skew.tile_j = tile_sizes
    sdfg = loop.sdfg
    plan = skew.plan_or_reason(loop, sdfg, occurrence_index(sdfg))
    if isinstance(plan, str):
        return Refused(f"blocked by WavefrontSkew: {loop.label}: {plan}.", Refusal.DEPENDENCE)
    return region_rewrite("WavefrontSkew", sdfg, lambda: skew.commit(plan, sdfg))


def early_exits(loop: LoopRegion) -> list[str]:
    """The blocks that end an iteration of ``loop`` early: its own break and continue (not those of a loop inside
    it), and a return anywhere in it."""
    found: list[str] = []
    pending: list[ControlFlowRegion] = [loop]
    while pending:
        region = pending.pop()
        for block in region.nodes():
            if isinstance(block, (BreakBlock, ContinueBlock, ReturnBlock)):
                found.append(type(block).__name__)
            elif isinstance(block, ConditionalBlock):
                pending.extend(branch for _, branch in block.branches)
            elif isinstance(block, LoopRegion):
                found += [
                    type(inner).__name__
                    for inner in block.all_control_flow_blocks(recursive=True)
                    if isinstance(inner, ReturnBlock)
                ]
            elif isinstance(block, ControlFlowRegion):
                pending.append(block)
    return found


def ordered_side_effect_in(region: ControlFlowRegion) -> str | None:
    """The label of a node in ``region`` whose side effect must keep its order (a communication call), or ``None``."""
    for node, parent in region.all_nodes_recursive():
        if isinstance(node, nodes.CodeNode) and isinstance(parent, SDFGState):
            if node.has_ordered_side_effects(parent.sdfg):
                return node.label
    return None


def plan_for_to_doall(row: Row, force: bool = False) -> Rewrite | str:
    """The sequential loop ``row`` becomes a doall where DaCe's :class:`LoopToMap` proves its iterations independent.
    With ``force`` the agent vouches for the containers: conflicts between iterations on them are waived, and every
    other condition still holds (bounds, ordered side effects, symbols one iteration hands to the next)."""
    loop = cast(LoopRegion, row[0])
    lift = LoopToMap()
    lift.loop = loop
    lift.keep_small_loops_sequential = False  # the agent asked for this loop
    name = "LoopToMap"
    if not lift.probe():
        refusal = lift.last_refusal
        why = "it gave no reason" if refusal is None else refusal.message
        if not force:
            return Refused(f"blocked by LoopToMap: {loop.label}: {why}.", Refusal.DEPENDENCE)
        exits = early_exits(loop)
        if exits:
            # permissive mode would map a body that leaves the loop early, which a map cannot
            return Refused(
                f"{loop.label} holds a {exits[0]}, which leaves the loop early; force does not waive that.",
                Refusal.WRONG_ROW_KIND,
            )
        ordered = ordered_side_effect_in(loop)
        if ordered is not None:
            # canonicalization orders such calls through data, a conflict that permissive mode would waive
            return Refused(
                f"{loop.label} holds {ordered}, whose side effect is ordered across iterations; force does not "
                "waive that.",
                Refusal.COMMUNICATION,
            )
        if not lift.probe(permissive=True):
            refusal = lift.last_refusal
            why = "it gave no reason" if refusal is None else refusal.message
            return Refused(
                f"blocked by LoopToMap, which force does not waive: {loop.label}: {why}.", Refusal.DEPENDENCE
            )
        name = FORCED_LOOP_TO_MAP
    return region_rewrite(name, loop.sdfg, lift.rewrite)


def in_parallel_doall(state: SDFGState, entry: nodes.MapEntry) -> bool:
    """Whether ``entry`` runs inside another doall, of its state or around the nested SDFG that holds it."""
    if state.entry_node(entry) is not None:
        return True
    sdfg = state.sdfg
    while sdfg.parent_nsdfg_node is not None:
        parent = cast(SDFGState, sdfg.parent)
        if parent.entry_node(sdfg.parent_nsdfg_node) is not None:
            return True
        sdfg = parent.sdfg
    return False


def schedule_text(entry: nodes.MapEntry) -> str:
    """The OpenMP schedule of ``entry`` as ``set-schedule-cpu`` takes it: ``dynamic,16 threads=8``."""
    found = entry.map.omp_schedule
    text = next(name for name, value in CPU_SCHEDULES.items() if value == found)
    if str(entry.map.omp_chunk_size) != "0":
        text += f",{entry.map.omp_chunk_size}"
    if entry.map.omp_num_threads > 0:
        text += f" threads={entry.map.omp_num_threads}"
    return text


def plan_set_schedule_cpu(row: Row, schedule: str | None, chunk: int | None, threads: int | None) -> Rewrite | str:
    """The OpenMP schedule of the outermost doall ``row``, its chunk and its team size (``None``: no clause)."""
    entry, state = cast(tuple[nodes.MapEntry, SDFGState], row)
    label = entry.map.label
    if in_parallel_doall(state, entry):
        return Refused(
            f"{label} runs inside another doall, whose threads each run it in order, so a schedule changes nothing; "
            "set the schedule of the outermost doall.",
            Refusal.WRONG_ROW_KIND,
        )
    if schedule is None or schedule not in CPU_SCHEDULES:
        return Refused(
            f"set-schedule-cpu takes schedule, one of {list(CPU_SCHEDULES)}; not {schedule!r}.", Refusal.ARGUMENT
        )
    if chunk is not None and (chunk < 1 or schedule == "default"):
        return Refused(
            f"a chunk is a positive count of iterations, with static, dynamic or guided; not {chunk} with {schedule}.",
            Refusal.ARGUMENT,
        )
    if threads is not None and threads < 1:
        return Refused(f"threads is a positive team size; not {threads}.", Refusal.ARGUMENT)

    def commit() -> None:
        entry.map.omp_schedule = CPU_SCHEDULES[schedule]
        entry.map.omp_chunk_size = 0 if chunk is None else chunk
        entry.map.omp_num_threads = 0 if threads is None else threads

    return Rewrite("OpenMP schedule", commit)


@dataclass(frozen=True, slots=True)
class GpuChoice:
    """What ``set-schedule-gpu`` chose for one doall: the thread block of a kernel doall, or the schedule of a doall
    inside one. The session keeps it per kernel and sets it after every placement, which resets both."""

    block: tuple[int, ...] = ()
    schedule: str = "sequential"

    def text(self) -> str:
        return f"block={list(self.block)}" if self.block else self.schedule


def gpu_role(state: SDFGState, entry: nodes.MapEntry) -> str | None:
    """``kernel`` for an outermost doall of a GPU kernel's body, ``inner`` for a doall directly inside one, else
    ``None``: deeper doalls run in order on each thread."""
    if in_parallel_doall(state, entry):
        outer = state.entry_node(entry)
        if isinstance(outer, nodes.MapEntry) and not in_parallel_doall(state, outer):
            return "inner"
        return None
    return "kernel"


def plan_set_schedule_gpu(
    row: Row, block: Sequence[int], schedule: str | None, chosen: Mapping[str, GpuChoice]
) -> GpuChoice | str:
    """The choice ``set-schedule-gpu`` makes for the doall ``row`` of a GPU kernel's body, given what the kernel's
    other doalls hold (``chosen``, by label), or why it is refused. DaCe's GPU code generator takes a kernel's thread
    block either from the kernel doall or from the doalls its threads share, never both; a dynamic doall's block size
    comes from DaCe's configuration (``compiler.cuda.dynamic_map_block_size``), and static and dynamic doalls cannot
    share a kernel."""
    entry, state = cast(tuple[nodes.MapEntry, SDFGState], row)
    label, role = entry.map.label, gpu_role(state, entry)
    if role is None:
        return Refused(
            f"{label} runs inside a doall inside the kernel, in order on each thread; set-schedule-gpu takes the kernel "
            "doall or a doall directly inside it.",
            Refusal.WRONG_ROW_KIND,
        )
    if role == "kernel":
        inner = [
            other.map.label
            for other in state.scope_children()[entry]
            if isinstance(other, nodes.MapEntry) and chosen.get(other.map.label, GpuChoice()).schedule != "sequential"
        ]
        if schedule is not None:
            return Refused(
                f"{label} is the kernel doall: it takes block, the threads of one block; schedule goes "
                "on a doall inside it.",
                Refusal.ARGUMENT,
            )
        last = int(dace.Config.get("compiler", "cuda", "block_size_lastdim_limit"))
        if (
            not 1 <= len(block) <= 3
            or any(size < 1 for size in block)
            or prod(block) > MAX_BLOCK_THREADS
            or (len(block) == 3 and block[2] > last)
        ):
            return Refused(
                f"block takes one to three positive sizes of at most {MAX_BLOCK_THREADS} threads in all, the third of "
                f"at most {last}; not {list(block)}.",
                Refusal.ARGUMENT,
            )
        if inner:
            return Refused(
                f"{', '.join(inner)} in {label} share the block's threads, so their schedule fixes the block; set them "
                "to sequential first.",
                Refusal.ARGUMENT,
            )
        return GpuChoice(block=tuple(block))
    kernel = cast(nodes.MapEntry, state.entry_node(entry))
    if block:
        return Refused(f"{label} is inside the kernel doall {kernel.map.label}; block goes on that.", Refusal.ARGUMENT)
    if schedule == "guided":
        return Refused("guided is not in DaCe's GPU code generator yet; take dynamic.", Refusal.ARGUMENT)
    if schedule not in GPU_INNER_SCHEDULES:
        return Refused(
            f"set-schedule-gpu takes schedule, one of {list(GPU_INNER_SCHEDULES)}, on {label}; not {schedule!r}.",
            Refusal.ARGUMENT,
        )
    if schedule == "sequential":
        return GpuChoice()
    if chosen.get(kernel.map.label, GpuChoice()).block:
        return Refused(
            f"the kernel doall {kernel.map.label} has a block of its own; set-schedule-gpu takes the block from one "
            "place only. Roll that move back first.",
            Refusal.ARGUMENT,
        )
    siblings = {
        other.map.label: chosen.get(other.map.label, GpuChoice()).schedule
        for other in state.scope_children()[kernel]
        if isinstance(other, nodes.MapEntry) and other is not entry
    }
    clash = [name for name, held in siblings.items() if held not in ("sequential", schedule)]
    if clash:
        return Refused(
            f"{', '.join(clash)} in the same kernel is shared another way; static and dynamic doalls cannot share a "
            "kernel.",
            Refusal.ARGUMENT,
        )
    if schedule == "dynamic" and len(entry.map.params) != 1:
        return Refused(f"a dynamic doall has one dimension; {label} has {len(entry.map.params)}.", Refusal.ARGUMENT)
    if schedule == "static":
        # bounded the way DaCe sizes a thread block: min(N, (i + 1) * 32) - i * 32 is 32, its last block guarded
        extent = prod(dace.symbolic.overapproximate(size) for size in entry.map.range.size())
        if dace.symbolic.issymbolic(extent) or int(extent) > MAX_BLOCK_THREADS:
            return Refused(
                f"a static doall gives each thread of the block one iteration, so its extent ({extent}) must be "
                f"bounded by a constant of at most {MAX_BLOCK_THREADS}; tile it first, or take dynamic.",
                Refusal.ARGUMENT,
            )
    return GpuChoice(schedule=schedule)


def set_gpu_choices(sdfg: dace.SDFG, choices: Mapping[str, Mapping[str, GpuChoice]]) -> None:
    """Set the choices of ``set-schedule-gpu`` on the placed GPU kernels of ``sdfg``, by kernel name and doall label;
    a label a later move retired is skipped."""
    for ext in external_calls(sdfg):
        held = choices.get(ext.name)
        if not held or ext.schedule not in dtypes.GPU_SCHEDULES or ext.standalone_sdfg is None:
            continue
        for node, _ in ext.standalone_sdfg.all_nodes_recursive():
            if isinstance(node, nodes.MapEntry) and node.map.label in held:
                choice = held[node.map.label]
                if choice.block:
                    node.map.gpu_block_size = list(choice.block)
                elif choice.schedule != "sequential":
                    node.map.schedule = GPU_INNER_SCHEDULES[choice.schedule]


def plan_hoist_invariant_code(row: Row) -> Rewrite | str:
    """LoopInvariantCodeMotion on the one for loop or doall ``row`` names."""
    if isinstance(row[0], LoopRegion):
        return plan_hoist_invariant_for(row)
    return plan_hoist_invariant_doall(row)


def plan_sink_invariant_code(row: Row, container: str | None) -> Rewrite | str:
    """The statements before the for loop or doall ``row`` names that compute ``container`` move into it."""
    obj = row[0]
    if container is None:
        return Refused(
            f"{SINK_INVARIANT_CODE} takes container=, the temporary whose statements sink; list_moves names them.",
            Refusal.ARGUMENT,
        )
    sdfg = obj.sdfg if isinstance(obj, LoopRegion) else node_state(row).sdfg
    desc = sdfg.arrays.get(container)
    if desc is None or not desc.transient:
        what = "no container" if desc is None else "an argument, not a temporary"
        return Refused(f"{container} is {what} of {sdfg.label}; only a temporary sinks.", Refusal.ARGUMENT)
    if isinstance(obj, LoopRegion):
        loop = obj
        planned = plan_sink_loop(loop, container)
        rewrite = region_rewrite("sink_into_loop", sdfg, lambda: sink_into_loop(loop, container))
    else:
        entry, state = cast(nodes.MapEntry, obj), node_state(row)
        planned = plan_sink_map(state, entry, container)
        rewrite = region_rewrite("sink_into_map", sdfg, lambda: sink_into_map(state, entry, container))
    if isinstance(planned, str):
        return Refused(planned, Refusal.DEPENDENCE)
    return rewrite


def sinkable_containers(row: Row) -> list[str]:
    """The temporaries whose statements can sink into the for loop or doall ``row`` names, sorted."""
    obj = row[0]
    if isinstance(obj, LoopRegion):
        source = predecessor_state(obj)
        if source is None:
            return []
        names = {n.data for n in source.data_nodes() if source.in_degree(n) > 0}
        return [name for name in sorted(names) if isinstance(plan_sink_loop(obj, name), SinkPlan)]
    entry, state = cast(nodes.MapEntry, obj), node_state(row)
    names = {e.data.data for e in state.in_edges(entry) if e.data.data is not None}
    return [name for name in sorted(names) if isinstance(plan_sink_map(state, entry, name), SinkPlan)]


def plan_hoist_invariant_for(row: Row) -> Rewrite | str:
    loop = cast(LoopRegion, row[0])
    if not loop_has_invariant_code(loop):
        return Refused(
            f"nothing in {loop.label} is loop-invariant: every statement reads what an iteration changes.",
            Refusal.DEPENDENCE,
        )
    return region_rewrite("LoopInvariantCodeMotion", loop.sdfg, lambda: hoist_loop_region(loop))


def plan_hoist_invariant_doall(row: Row) -> Rewrite | str:
    entry, state = cast(nodes.MapEntry, row[0]), node_state(row)
    # the map whose scope decides invariance: a hoist leaves the whole perfect nest or stays, as in canonical form
    root = LoopInvariantCodeMotion().variant_root(state, entry)
    if not map_has_invariant_code(state, entry, root):
        return Refused(
            f"nothing in {entry.map.label} is invariant: every statement reads what an iteration changes.",
            Refusal.DEPENDENCE,
        )
    return Rewrite("LoopInvariantCodeMotion", lambda: hoist_map_scope(state, entry, root))


PLANNERS: dict[str, Callable[..., Rewrite | str]] = {
    "for-fusion": plan_loop_fusion,
    "for-fission": plan_loop_fission,
    "doall-fusion": plan_map_fusion,
    "doall-fission": plan_map_fission,
    "subgraph-fission": plan_subgraph_fission,
    "interchange-for-doall": plan_loop_map,
    "interchange-doall-for": plan_map_loop,
    "interchange-doall-doall": plan_map_map,
    INLINE_LIBRARY: plan_inline_library,
    SET_IMPLEMENTATION: plan_set_implementation,
    "tile-doall": plan_tile_doall,
    "untile-doall": plan_untile_doall,
    "flatten-doalls": plan_flatten_doalls,
    "unflatten-doall": plan_unflatten_doall,
    "doall-to-for": plan_doall_to_for,
    "snapshot-reads": plan_snapshot_reads,
    "tile-for": plan_tile_for,
    "untile-for": plan_untile_for,
    "unroll-for": plan_unroll_for,
    "unroll-doall": plan_unroll_doall,
    HOIST_INVARIANT_CODE: plan_hoist_invariant_code,
    HOIST_INVARIANT_GUARD: plan_hoist_invariant_guard,
    SINK_INVARIANT_GUARD: plan_sink_invariant_guard,
    LIB_BEFORE: lambda row, target: plan_lib_move(row, target, "before"),
    LIB_AFTER: lambda row, target: plan_lib_move(row, target, "after"),
    LIB_INTO: lambda row, target: plan_lib_move(row, target, "into"),
    PROPAGATE_CONFIG: plan_propagate_config,
    "stage-loads-shared": plan_stage_loads_shared,
}


class MoveKind(Record):
    """One move kind as the agent is told it: the tree rows it takes in order, the DaCe transformation it runs and
    what it does."""

    kind: str
    rows: tuple[str, ...]
    runs: str
    does: str


#: The tree keyword of each row type a move takes.
ROW_KEYWORDS: dict[type | tuple[type, ...], str] = {
    LoopRegion: "for",
    nodes.MapEntry: "doall",
    ControlFlowBlock: "state",
    SDFGState: "state",
    ConditionalBlock: "if",
    nodes.LibraryNode: "lib_<family>",
    # a library call, or a one-element copy as the tasklet DaCe makes it (``names.is_call``)
    nodes.CodeNode: "lib_<family>",
    LOOP_ROW: "for|doall",
    RUN_ROW: "for|state|doall",
}
#: The kinds that take one row or more, each of their shape's one row kind.
VARIADIC = frozenset({PERSISTENT_KERNEL})

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
        "for t: doall i: B becomes doall i: for t: B (the doall goes outside). A body of several doalls over one shared "
        "axis, in several states or behind ifs, becomes doall i: for t: { A; B; ... } when no column touches another's "
        "elements; the row names one of the doalls. doall-fusion the ones that fuse first.",
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
    INLINE_LIBRARY: (
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
    "flatten-doalls": ("MapCollapse", "doall i: doall j: B becomes doall (i, j): B, one parallel space."),
    "unflatten-doall": ("MapExpansion", "the inverse: doall (i, j): B becomes doall i: doall j: B."),
    "doall-to-for": (
        "MapToForLoop",
        "doall k: doall j: B becomes for k: doall j: B: the outer doall runs in order and the doalls inside stay "
        "parallel, so its body can for-fusion with a sequential loop over the same range (a physics pass into the "
        "level scan it feeds).",
    ),
    FOR_TO_DOALL: (
        "LoopToMap (force, optional)",
        "for i: B becomes doall i: B where DaCe proves that no iteration reads or writes what another writes. With "
        "force, you vouch for the arrays: DaCe's checks of conflicting writes are waived, and some of its checks of "
        "carried reads, so a wrong program is yours to find. Force only what you can argue (an index array that is a "
        "permutation); bounds, a break, continue or return in the body, calls with ordered side effects and scalars "
        "carried between iterations are refused either way, and DaCe still refuses a read its memlet test ties to "
        "another iteration's write.",
    ),
    SET_SCHEDULE_CPU: (
        "the map's OpenMP schedule (schedule, chunk and threads; in a CPU kernel only)",
        "doall i: B keeps what it computes and runs under schedule(static|dynamic|guided[, chunk]) num_threads(t). "
        "Dynamic or guided pays when iterations differ in cost (triangular nests, sparse rows). Only an outermost "
        "doall of a kernel takes one.",
    ),
    SET_SCHEDULE_GPU: (
        "the kernel's thread block (gpu_block_size), or the schedule of a doall inside it (in a GPU kernel only)",
        "on the kernel doall, block=[bx, by] sets the threads of one block. On a doall directly inside it: sequential "
        "(each thread runs it whole), static (one iteration per thread of the block, GPU_ThreadBlock) or dynamic "
        "(the block's threads take iterations from a shared queue, GPU_ThreadBlock_Dynamic). The block comes from "
        "one place only: the kernel doall, or the doalls inside it.",
    ),
    "snapshot-reads": (
        "a doall of its own that copies each slice (SnapshotReads)",
        "a doall that reads a slice of an array (path[:, k], path[k, :]: one index per dimension, a map parameter or a "
        "value the doall does not change) which another doall of the state writes copies the slice into a transient "
        "first, so the two doalls no longer touch one array and doall-fusion can merge them.",
    ),
    SKEW_WAVEFRONT: (
        "WavefrontSkew (tile_sizes, optional: the extents along the loop and along the diagonal)",
        "a sequential loop whose iterations depend on each other along a diagonal (Gauss-Seidel, LU, dynamic "
        "programming) becomes a sequential loop over diagonals, each run as a parallel doall, or a diagonal of tiles "
        "where the dependences allow. It applies only where DaCe proves the diagonals independent; DaCe picks the skew.",
    ),
    "tile-for": (
        "LoopTiling (tile_sizes, one size)",
        "for i: B becomes for i0 (step = tile size): for i (one tile): B, the same iterations in the same order.",
    ),
    "untile-for": ("UntileLoops", "the inverse: a loop over tiles and its loop over one tile's points become one."),
    "unroll-for": (
        "LoopUnroll (max_iterations, the largest trip count you accept)",
        "for i in 0:3: B becomes B(0); B(1); B(2): one copy of the body per iteration, the loop gone. Only a loop "
        "with a constant trip count unrolls (propagate-config can give its bounds values).",
    ),
    "unroll-doall": (
        "MapUnroll (max_iterations, the largest trip count you accept)",
        "a doall with a constant range, not inside another doall of its state, becomes one copy of its body per "
        "iteration, the doall gone.",
    ),
    HOIST_INVARIANT_GUARD: (
        "MoveLoopInvariantIfUp on a for loop; MoveMapInvariantIfUp.apply_to_map on a doall (tried on a copy; lifts "
        "this doall's if only)",
        "for i: if c: B becomes if c: for i: B, and doall i: if c: B becomes if c: doall i: B, where c reads neither "
        "the loop variable nor what the loop writes; the test runs once. A for loop's if must be its only statement; "
        "a doall that also holds work outside the if is unswitched: if c: doall i: {A; B} else: doall i: A.",
    ),
    SINK_INVARIANT_GUARD: (
        "MoveIfIntoLoop on a branch that ends in one for loop; push_if_into_maps on a branch that is one state of "
        "doalls",
        "if c: for i: B becomes for i: if c: B, and if c: doall i: B becomes doall i: if c: B (each doall of the "
        "branch takes a copy of the test), the inverse of hoist-invariant-guard: the test runs every iteration, and "
        "the loop or doall can fuse with its neighbors.",
    ),
    SINK_INVARIANT_CODE: (
        "sink_into_loop or sink_into_map (container, the temporary to sink)",
        "the inverse of hoist-invariant-code for one temporary: t = a * b; for i: ... t ... becomes for i: {t = a * b; "
        "... t ...}. The statements run every iteration instead of once, t no longer lives across the loop, and a "
        "doall no longer has an input that ties it to the statements before it. Only plain statements sink, from "
        "the state just before the loop or doall (or above the doall in its state); t must be used nowhere else, and "
        "the loop must not write what the statements read.",
    ),
    HOIST_INVARIANT_CODE: (
        "LoopInvariantCodeMotion on this for loop or doall",
        "the loop's invariant statements (an expression or load whose inputs no iteration changes) run once, before "
        "the loop; for a for loop also whole sub-blocks that use neither the loop variable nor what the loop writes. "
        "The pass picks them; the loop stays. In a perfect doall nest, a statement invariant only in an inner doall "
        "stays, so the nest stays whole.",
    ),
    LIB_BEFORE: (
        "a state of its own, checked against the read and write sets of every block it crosses",
        "a library call whose memlets are its whole effect (lib_copy, lib_reduce, lib_blas, ...; not a "
        "communication or a kernel), or a host lib_sync (a device or stream barrier, which crosses no block that "
        "issues device work or touches what device work touches), leaves its state for a new state just before the named block, of the call's "
        "region or of a loop around it. Hoist a copy or call whose inputs a loop never changes out of it (it then "
        "runs once, not every iteration), or take one from between two loops so they become adjacent for "
        "for-fusion. Refused when a crossed block writes what the call touches or reads what it writes, when a loop "
        "it leaves is not proven to run at least once, or when it would leave a branch or enter a block.",
    ),
    LIB_AFTER: (
        "a state of its own, checked against the read and write sets of every block it crosses",
        "the same, into a new state just after the named block: sink a call to its consumer, or out of a loop "
        "when nothing in the loop reads what it writes.",
    ),
    LIB_INTO: (
        "the named state, ordered there by its access nodes and empty edges",
        "the same call joins an existing state of its region or of a loop around it instead of a new one: it reads "
        "the array from the access node that state last writes it to (or writes the access node that state reads), "
        "and runs after (when it comes from a later point) or before every other access there that conflicts with "
        "it. Refused for an accumulation there, two unordered writes of one array, a cycle, and, for a lib_sync, a "
        "state whose device work it would run ahead of.",
    ),
    ROLLBACK: (
        "the session's history (the last 8 changes)",
        "back to the program before the last move or default stage: its tree labels and applied_moves too; a label "
        "handed out since is never reused. Apply it again for the change before.",
    ),
    PROPAGATE_CONFIG: (
        "ConfigPropagation (config_values: the scalar arguments and free symbols you expect at run time)",
        "the program becomes if (arguments == values): the program specialized for them, else: the program as it "
        "was. The specialized branch runs only when the arguments equal the values; any other draw runs the generic "
        "program, so results stay correct. The moves reach inside the specialized branch, where bounds are constant "
        "(unroll-for, unroll-doall).",
    ),
    VECTORIZE_DOALL: (
        "VectorizeMap (lanes: per vectorized dimension, innermost last; isa)",
        "the doall's innermost maps run as tiles of lanes for isa (auto, avx512, avx2, sve, neon, scalar; "
        "sve takes its own vector length, so its lanes are ignored), with a masked tile for the remainder; the "
        "body becomes tile loads, tile operations and tile stores. Every other doall stays scalar. view_tree with c "
        "shows the body as the ISA's tile functions; judged when applied, on a copy first.",
    ),
    SET_FP_MODE: (
        "the compiler's floating-point flags of one kernel (scope; fp_mode: strict-ieee, contract-fma, fast-math)",
        "the kernel is built in that mode only: get_performance measures it and finalize's sweep keeps to it. "
        "strict-ieee keeps every rounding, contract-fma fuses multiply-adds, fast-math also reassociates and "
        "assumes no NaN or infinity. Unset, the kernel's sweep picks the fastest correct mode. The result is checked "
        "against the reference at get_performance, not proven.",
    ),
    PERSISTENT_KERNEL: (
        "MegakernelLoops (consecutive rows inside scope, the kernel)",
        "the rows (for loops, states, doalls for their states), consecutive in one region of the kernel's body, run as "
        "one persistent kernel with a barrier where a launch or an implicit barrier was: on the GPU one launch, "
        "behind a runtime test that the largest doall fits the resident threads (else the rows launch as before); on "
        "the CPU the doalls share one OpenMP team and synchronize explicitly. Pays when launches and barriers cost "
        "more than the doalls. A kernel holds one persistent run: to change it, roll back and apply a longer list.",
    ),
    CHANGE_PRECISION: (
        "a new element type of one internal array (container; dtype: float32, float64)",
        "the array is stored, read and written in dtype; every statement converts at the array, so the arithmetic "
        "around it is unchanged. Only an array the program makes itself, read and written by statements alone. The "
        "result is checked against the reference at get_performance, not proven; an argument's type changes with "
        "relayout cast.",
    ),
    RELAYOUT: (
        "relayout's apply_requests (operation: one operation of the bench's requests on an array argument)",
        "the program works on the argument as the operation lays it out (permute, block / unblock, split / unsplit, "
        "pad / unpad, shuffle, linear_shuffle, cast), and the operation goes to the bench as the submission's "
        "requests: the harness lays the argument out before the call and reads it back after it, outside the "
        "timing. Every operation so far is laid out again from the program before the first, so a later one names "
        "the digits the earlier ones left. Lay arguments out first: refused after any other change. An array the "
        "program makes itself is laid out in place at any point, zip and unzip included, with no request; "
        "linear_shuffle is the XOR swizzle.",
    ),
    "stage-loads-shared": (
        "StageLoadsThroughSharedMemory (block: the thread block)",
        "a GPU kernel inside a GPU scope runs one thread block per tile, and each block first copies the tile (halo "
        "included) of every array it only reads and reuses into shared memory, so global memory is read once.",
    ),
}


def move_kinds() -> list[MoveKind]:
    """Every move kind, in :data:`MOVE_SHAPES` order."""
    return [
        MoveKind(
            kind=kind,
            rows=tuple(ROW_KEYWORDS[t] for t in shape) + (("...",) if kind in VARIADIC else ()),
            runs=MOVE_TEXT[kind][0],
            does=MOVE_TEXT[kind][1],
        )
        for kind, shape in MOVE_SHAPES.items()
    ]


def plan_move(
    kind: str,
    rows: Sequence[Row],
    tile_sizes: Sequence[int] = (),
    implementation: str | None = None,
    max_iterations: int = PROBE_UNROLL,
    config_values: Mapping[str, int | float] | None = None,
    program: dace.SDFG | None = None,
    output_dir: Path | None = None,
    isa: str = DEFAULT_ISA,
    target: str = "cpu",
    force: bool = False,
    schedule: str | None = None,
    chunk: int | None = None,
    threads: int | None = None,
    container: str | None = None,
) -> Rewrite | str:
    """The move ``kind`` makes of ``rows``, or why it is illegal or not implemented; nothing is mutated.
    ``tile_sizes`` goes to the :data:`SIZED` kinds and :data:`SKEW_WAVEFRONT`, ``implementation`` to
    :data:`SET_IMPLEMENTATION`, ``max_iterations`` to the :data:`UNROLLS` kinds; :data:`PROPAGATE_CONFIG`, on no row,
    takes ``config_values`` for ``program`` and writes its variants to ``output_dir``. ``target`` (``cpu`` or ``gpu``)
    is the device the wavefront's tiles are for, ``force`` goes to :data:`FOR_TO_DOALL`, and ``schedule``, ``chunk``
    and ``threads`` go to :data:`SET_SCHEDULE_CPU`."""
    if kind == PROPAGATE_CONFIG:
        assert program is not None and output_dir is not None, "propagate-config is planned on a program"
        return plan_propagate_config(program, config_values or {}, output_dir)
    for (obj, _), wanted in zip(rows, MOVE_SHAPES[kind]):
        if not isinstance(obj, wanted):
            expected = ", ".join(ROW_KEYWORDS[t] for t in MOVE_SHAPES[kind])
            return Refused(
                f"{kind} takes ({expected}); {tree_label(obj)!r} is a {type(obj).__name__}.", Refusal.WRONG_ROW_KIND
            )
    if kind == VECTORIZE_DOALL:
        return plan_auto_vectorize(*rows, tile_sizes=tile_sizes, isa=isa)
    if kind == SKEW_WAVEFRONT:
        return plan_skew_wavefront(*rows, tile_sizes=tile_sizes, target=target)
    if kind == FOR_TO_DOALL:
        return plan_for_to_doall(*rows, force=force)
    if kind == SINK_INVARIANT_CODE:
        return plan_sink_invariant_code(*rows, container=container)
    if kind == SET_SCHEDULE_CPU:
        return plan_set_schedule_cpu(*rows, schedule=schedule, chunk=chunk, threads=threads)
    if kind in SIZED:
        return PLANNERS[kind](*rows, tile_sizes=tile_sizes)
    if kind == SET_IMPLEMENTATION:
        return PLANNERS[kind](*rows, implementation=implementation)
    if kind in UNROLLS:
        return PLANNERS[kind](*rows, max_iterations=max_iterations)
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
        lanes = len(inside) > 1 and lane_axis_of(loop) is not None and not inside_a_map(loop)
        if len(inside) == 1 or lanes:
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


def loop_guards(sdfg: dace.SDFG) -> Iterator[tuple[Row, ...]]:
    """The ifs whose first branch ends in one for loop, or is one state of doalls."""
    for cond in every_region_block(sdfg, ConditionalBlock):
        sinks = cond.branches[0][1].sink_nodes() if cond.branches else []
        if len(sinks) == 1 and isinstance(sinks[0], LoopRegion) or match_if_over_maps(cond) is not None:
            yield ((cond, None),)


def guarded_loops(sdfg: dace.SDFG) -> Iterator[tuple[Row, ...]]:
    """The for loops that hold a conditional as one of their top-level blocks."""
    for loop in every_region_block(sdfg, LoopRegion):
        if any(isinstance(block, ConditionalBlock) for block in loop.nodes()):
            yield ((loop, None),)


CANDIDATES: dict[str, Callable[[dace.SDFG], Iterator[tuple[Row, ...]]]] = {
    "for-fusion": loop_pairs,
    "for-fission": for_rows,
    "doall-fusion": map_pairs,
    "doall-fission": doall_rows,
    "subgraph-fission": body_cuts,
    "interchange-for-doall": loop_map_pairs,
    "interchange-doall-for": map_loop_pairs,
    "interchange-doall-doall": interchange_map_pairs,
    INLINE_LIBRARY: library_rows,
    SET_IMPLEMENTATION: library_rows,
    "tile-doall": doall_rows,
    "untile-doall": tile_pair_rows,
    "flatten-doalls": nested_map_pairs,
    "unflatten-doall": lambda sdfg: (((e, st),) for st, e in map_rows(sdfg) if len(e.map.params) > 1),
    "doall-to-for": lambda sdfg: (((e, st),) for st, e in map_rows(sdfg) if len(e.map.params) == 1),
    "snapshot-reads": lambda sdfg: (
        ((e, st),) for st, e in map_rows(sdfg) if st.entry_node(e) is None and reads_another_doalls_write(st, e)
    ),
    "tile-for": for_rows,
    "untile-for": tile_loops,
    "unroll-for": for_rows,
    "unroll-doall": lambda sdfg: (((e, st),) for st, e in map_rows(sdfg) if st.entry_node(e) is None),
    HOIST_INVARIANT_CODE: lambda sdfg: chain(for_rows(sdfg), doall_rows(sdfg)),
    SINK_INVARIANT_CODE: lambda sdfg: (
        rows for rows in chain(for_rows(sdfg), doall_rows(sdfg)) if sinkable_containers(rows[0])
    ),
    HOIST_INVARIANT_GUARD: lambda sdfg: chain(guarded_loops(sdfg), guarded_doalls(sdfg)),
    SINK_INVARIANT_GUARD: loop_guards,
    VECTORIZE_DOALL: lambda sdfg: (((e, st),) for st, e in map_rows(sdfg) if st.entry_node(e) is None),
    LIB_BEFORE: movable_rows,
    LIB_AFTER: movable_rows,
    LIB_INTO: lambda sdfg: movable_rows(sdfg, states_only=True),
    PROPAGATE_CONFIG: lambda sdfg: iter([()] if configurable(sdfg) else []),
    "stage-loads-shared": lambda sdfg: (((e, st),) for st, e in map_rows(sdfg) if is_kernel(st, e)),
    SKEW_WAVEFRONT: for_rows,
    FOR_TO_DOALL: for_rows,
    SET_SCHEDULE_CPU: lambda sdfg: (((e, st),) for st, e in map_rows(sdfg) if not in_parallel_doall(st, e)),
    SET_SCHEDULE_GPU: lambda sdfg: (((e, st),) for st, e in map_rows(sdfg) if gpu_role(st, e) is not None),
}


def legal_moves(sdfg: dace.SDFG, kind: str | None = None, target: str = "cpu") -> list[tuple[str, tuple[str, ...]]]:
    """``(kind, labels)`` of every legal move right now, of ``kind`` or of every kind.

    :raises ValueError: ``kind`` is not a move kind.
    """
    kinds = list(CANDIDATES) if kind is None else [check_kind(kind)]
    found: dict[tuple[str, tuple[str, ...]], None] = {}
    for name in kinds:
        if name not in CANDIDATES:
            continue
        for rows in CANDIDATES[name](sdfg):
            labels = tuple(tree_label(obj) for obj, _ in rows)
            if name in (
                PROPAGATE_CONFIG,
                VECTORIZE_DOALL,
                SET_SCHEDULE_CPU,
                SET_SCHEDULE_GPU,
                SINK_INVARIANT_CODE,
            ):  # no values to probe with, or too costly, or probed by its candidates: judged when applied
                found[(name, labels)] = None
                continue
            try:
                probe = plan_move(name, rows, probe_sizes(name, rows), probe_implementation(name, rows), target=target)
            except CRASHES as err:  # a check that raises on one candidate hides only that candidate
                print(f"nestforge: {name} {list(labels)} is not listed: {type(err).__name__}: {err}", file=sys.stderr)
                continue
            if isinstance(probe, Rewrite):
                found[(name, labels)] = None
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
