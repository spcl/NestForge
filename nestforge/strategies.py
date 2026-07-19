"""Pluggable detection strategies.

A strategy is ``Callable[[SDFG], List[Tuple[SDFG, NestNode]]]`` returning the nests to extract,
each paired with the **parent SDFG** it lives in (nested SDFGs are recursive, so the lowering
pass must know which SDFG to operate on). ``outer`` is the default. New strategies register via
:func:`register_strategy` and resolve by name with :func:`get_strategy`.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion

from nestforge.extract import NestNode

Strategy = Callable[[dace.SDFG], List[Tuple[dace.SDFG, NestNode]]]

_REGISTRY: Dict[str, Strategy] = {}


def register_strategy(name: str, fn: Strategy) -> None:
    _REGISTRY[name] = fn


def get_strategy(name: str) -> Strategy:
    if name not in _REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def strategy_names() -> List[str]:
    return sorted(_REGISTRY)


def top_level_map_entries(state: dace.SDFGState) -> List[nodes.MapEntry]:
    """MapEntry nodes at the top of a state's scope tree (not nested inside another map)."""
    return [n for n in state.scope_children()[None] if isinstance(n, nodes.MapEntry)]


def outer(sdfg: dace.SDFG) -> List[Tuple[dace.SDFG, NestNode]]:
    """Outermost nests of the root SDFG: top-level map-nests + top-level CFG loop regions.

    Does not descend into nested SDFGs (those are already 'inside'); other strategies may.
    """
    refs: List[Tuple[dace.SDFG, NestNode]] = []
    for block in sdfg.nodes():
        if isinstance(block, LoopRegion):
            refs.append((sdfg, block))
        elif isinstance(block, dace.SDFGState):
            for me in top_level_map_entries(block):
                refs.append((sdfg, me))
    return refs


_COMPUTE = (nodes.Tasklet, nodes.LibraryNode, nodes.NestedSDFG)


def direct_child_maps(state: dace.SDFGState, entry: nodes.MapEntry) -> List[nodes.MapEntry]:
    return [n for n in state.scope_children()[entry] if isinstance(n, nodes.MapEntry)]


def is_taskloop_map(state: dace.SDFGState, entry: nodes.MapEntry) -> bool:
    """A map whose body is *only* maps (no tasklet/library/nested compute) -- a scheduling wrapper."""
    kids = state.scope_children()[entry]
    has_map = any(isinstance(n, nodes.MapEntry) for n in kids)
    has_compute = any(isinstance(n, _COMPUTE) for n in kids)
    return has_map and not has_compute


def is_parallel_nest(node: NestNode) -> bool:
    """Whether an extracted nest is PARALLEL within the DaCe scope (its iterations are independent, so
    the emitted kernel may carry an OpenMP parallel scope) or SEQUENTIAL.

    A ``MapEntry`` is parallel unless its schedule is explicitly ``Sequential`` -- DaCe's ``LoopToMap``
    (run in the ``baseline``/``canonicalize`` build) only turns a *provably parallel* loop into a Map, so
    a Map is the parallel signal. A ``LoopRegion`` is a loop that stayed a loop (a loop-carried
    recurrence LoopToMap refused), hence sequential. A WCR reduction inside a parallel map is still
    parallel -- the OpenMP emitter carries it as a ``reduction(...)`` clause, not a serialization.
    """
    if isinstance(node, nodes.MapEntry):
        return node.map.schedule != dace.ScheduleType.Sequential
    return False  # LoopRegion (or anything non-Map): sequential


def is_taskloop_loop(loop: LoopRegion) -> bool:
    """A loop whose body is *only* maps: states with no free compute and no nested control flow."""
    if any(not isinstance(b, dace.SDFGState) for b in loop.nodes()):
        return False
    has_map = False
    for state in loop.nodes():
        for n in state.scope_children()[None]:
            if isinstance(n, _COMPUTE):
                return False
            if isinstance(n, nodes.MapEntry):
                has_map = True
    return has_map


def collect_skip_map(sdfg: dace.SDFG, state: dace.SDFGState, entry: nodes.MapEntry, refs: list) -> None:
    if is_taskloop_map(state, entry):
        for child in direct_child_maps(state, entry):
            collect_skip_map(sdfg, state, child, refs)
    else:
        refs.append((sdfg, entry))


def collect_skip_loop(sdfg: dace.SDFG, loop: LoopRegion, refs: list) -> None:
    if is_taskloop_loop(loop):
        for state in loop.nodes():
            for me in top_level_map_entries(state):
                collect_skip_map(sdfg, state, me, refs)
    else:
        refs.append((sdfg, loop))


def skip_taskloops(sdfg: dace.SDFG) -> List[Tuple[dace.SDFG, NestNode]]:
    """Like :func:`outer`, but never externalise a pure *taskloop* wrapper.

    A map whose body is only maps, or a loop whose body is only maps, is a scheduling construct with
    no compute of its own -- offloading it buys nothing. Such wrappers are skipped and the search
    descends to the first compute-bearing nest inside them (the actual kernel).
    """
    refs: List[Tuple[dace.SDFG, NestNode]] = []
    for block in sdfg.nodes():
        if isinstance(block, LoopRegion):
            collect_skip_loop(sdfg, block, refs)
        elif isinstance(block, dace.SDFGState):
            for me in top_level_map_entries(block):
                collect_skip_map(sdfg, block, me, refs)
    return refs


def region_has(region, node_types) -> bool:
    """True if any state anywhere in a control-flow region holds a node of the given types."""
    return any(
        isinstance(n, node_types) for block in region.all_control_flow_blocks() if isinstance(block, dace.SDFGState)
        for n in block.nodes())


def innermost(sdfg: dace.SDFG) -> List[Tuple[dace.SDFG, NestNode]]:
    """Every innermost nest -- a map or loop with no further nest inside it -- across all SDFGs.

    The vectorization-style unit, for both parallel and sequential compute leaves:

    * an **innermost map** has no map nested in its scope (a map cannot contain a loop region);
    * an **innermost loop** (``LoopRegion``) has no nested loop *and* no map inside -- if it held
      maps, those maps would be the innermost units, so the loop is a wrapper, not a leaf.

    A map or loop that wraps a ``NestedSDFG`` is *not* a leaf: the nested SDFG holds the real compute
    and is walked separately by ``all_sdfgs_recursive``, so selecting the wrapper too would offload
    the same compute twice. The two never overlap, so each compute leaf is returned exactly once.
    """
    refs: List[Tuple[dace.SDFG, NestNode]] = []
    for sub in sdfg.all_sdfgs_recursive():
        for state in sub.states():
            for entry in [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]:
                deeper = [
                    n for n in state.scope_subgraph(entry, include_entry=False).nodes()
                    if isinstance(n, (nodes.MapEntry, nodes.NestedSDFG))
                ]
                if not deeper:
                    refs.append((sub, entry))
        for region in sub.all_control_flow_regions():
            if not isinstance(region, LoopRegion):
                continue
            nested_loops = [
                r for r in region.all_control_flow_regions() if isinstance(r, LoopRegion) and r is not region
            ]
            if not nested_loops and not region_has(region, (nodes.MapEntry, nodes.NestedSDFG)):
                refs.append((sub, region))
    return refs


def empty_strategy_reason(sdfg: dace.SDFG) -> str:
    """Why a strategy found no nest to externalise -- distinguishing an honestly EMPTY kernel from one
    whose only compute is a **library node**. We deliberately do NOT offload library nodes: DaCe expands
    each to its fastest available library (BLAS/LAPACK/argreduce/...), so externalising it to a naive
    numpy->C loop would only lose performance. Such a kernel is legitimately skipped -- but with a reason
    that says so, instead of the misleading 'no compute nest'.
    """
    has_libnode = any(
        isinstance(n, nodes.LibraryNode) for sub in sdfg.all_sdfgs_recursive() for st in sub.states()
        for n in st.nodes())
    if has_libnode:
        return "only library-node compute (DaCe offloads it to its fastest library; no loop-nest to externalise)"
    return "no compute nest (strategy returned nothing)"


register_strategy("outer", outer)
register_strategy("skip-taskloops", skip_taskloops)
register_strategy("innermost", innermost)
