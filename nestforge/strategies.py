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


def _top_level_map_entries(state: dace.SDFGState) -> List[nodes.MapEntry]:
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
            for me in _top_level_map_entries(block):
                refs.append((sdfg, me))
    return refs


_COMPUTE = (nodes.Tasklet, nodes.LibraryNode, nodes.NestedSDFG)


def _direct_child_maps(state: dace.SDFGState, entry: nodes.MapEntry) -> List[nodes.MapEntry]:
    return [n for n in state.scope_children()[entry] if isinstance(n, nodes.MapEntry)]


def _is_taskloop_map(state: dace.SDFGState, entry: nodes.MapEntry) -> bool:
    """A map whose body is *only* maps (no tasklet/library/nested compute) -- a scheduling wrapper."""
    kids = state.scope_children()[entry]
    has_map = any(isinstance(n, nodes.MapEntry) for n in kids)
    has_compute = any(isinstance(n, _COMPUTE) for n in kids)
    return has_map and not has_compute


def _is_taskloop_loop(loop: LoopRegion) -> bool:
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


def _collect_skip_map(sdfg: dace.SDFG, state: dace.SDFGState, entry: nodes.MapEntry, refs: list) -> None:
    if _is_taskloop_map(state, entry):
        for child in _direct_child_maps(state, entry):
            _collect_skip_map(sdfg, state, child, refs)
    else:
        refs.append((sdfg, entry))


def _collect_skip_loop(sdfg: dace.SDFG, loop: LoopRegion, refs: list) -> None:
    if _is_taskloop_loop(loop):
        for state in loop.nodes():
            for me in _top_level_map_entries(state):
                _collect_skip_map(sdfg, state, me, refs)
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
            _collect_skip_loop(sdfg, block, refs)
        elif isinstance(block, dace.SDFGState):
            for me in _top_level_map_entries(block):
                _collect_skip_map(sdfg, block, me, refs)
    return refs


def _region_has_map(region) -> bool:
    """True if any state anywhere in a control-flow region holds a map."""
    return any(
        isinstance(n, nodes.MapEntry) for block in region.all_control_flow_blocks()
        if isinstance(block, dace.SDFGState) for n in block.nodes())


def innermost(sdfg: dace.SDFG) -> List[Tuple[dace.SDFG, NestNode]]:
    """Every innermost nest -- a map or loop with no further nest inside it -- across all SDFGs.

    The vectorization-style unit, for both parallel and sequential compute leaves:

    * an **innermost map** has no map nested in its scope (a map cannot contain a loop region);
    * an **innermost loop** (``LoopRegion``) has no nested loop *and* no map inside -- if it held
      maps, those maps would be the innermost units, so the loop is a wrapper, not a leaf.

    The two never overlap, so a kernel's compute leaves are returned exactly once each.
    """
    refs: List[Tuple[dace.SDFG, NestNode]] = []
    for sub in sdfg.all_sdfgs_recursive():
        for state in sub.states():
            for entry in [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]:
                inner_maps = [
                    n for n in state.scope_subgraph(entry, include_entry=False).nodes()
                    if isinstance(n, nodes.MapEntry)
                ]
                if not inner_maps:
                    refs.append((sub, entry))
        for region in sub.all_control_flow_regions():
            if not isinstance(region, LoopRegion):
                continue
            nested_loops = [
                r for r in region.all_control_flow_regions() if isinstance(r, LoopRegion) and r is not region
            ]
            if not nested_loops and not _region_has_map(region):
                refs.append((sub, region))
    return refs


register_strategy("outer", outer)
register_strategy("skip-taskloops", skip_taskloops)
register_strategy("innermost", innermost)
