"""Phase 1 of the 4-phase optimizer: set loop-/map-nest granularity by a fusion strategy.

The framework's first phase fixes the *granularity* the later phases optimize at. A
``FusionStrategy`` takes a ``SymbolPropagation``-normalized SDFG and mutates it in place to a chosen
granularity, returning the number of transformation steps applied. One strategy ships today --
:func:`maximal_fusion`, the deterministic Phase-1 default (fuse everything legal). New strategies
register via :func:`register_fusion_strategy` and resolve by name with :func:`get_fusion_strategy`,
mirroring :mod:`nestforge.strategies` (the Phase-2 offload-granularity registry).

Agent surface. Granularity is not one-shot -- the agent adjusts it move-by-move, then reoptimizes
(Phase 4). The per-move tools are re-exported here so Phase 1 is a single import:

  * :func:`enumerate_fusions` / :func:`apply_fusion` (from :mod:`nestforge.fusion_arms`) -- the legal
    fuse moves right now, one applied at a time. Applying one stales the rest; re-enumerate after each.
  * :func:`fission_to_statements` / :func:`map_fission_moves` (from :mod:`nestforge.fission_arms`) --
    the inverse: explode back to statement granularity, or fission one map.

A strategy is just a scripted policy over those same moves; :func:`maximal_fusion` is the "apply
every legal fuse to a fixed point" policy.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import dace
from dace.transformation.dataflow import MapFusionHorizontal, MapFusionVertical
from dace.transformation.interstate import LoopToMap

from nestforge.fission_arms import fission_to_statements, map_fission_moves
from nestforge.fusion_arms import FusionMove, apply_fusion, can_fuse, enumerate_fusions

#: A Phase-1 strategy: mutate the SDFG in place to a granularity, returning the step count.
FusionStrategy = Callable[[dace.SDFG], int]

_REGISTRY: Dict[str, FusionStrategy] = {}


def register_fusion_strategy(name: str, fn: FusionStrategy) -> None:
    _REGISTRY[name] = fn


def get_fusion_strategy(name: str) -> FusionStrategy:
    if name not in _REGISTRY:
        raise KeyError(f"unknown fusion strategy {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def fusion_strategy_names() -> List[str]:
    return sorted(_REGISTRY)


def maximal_fusion(sdfg: dace.SDFG) -> int:
    """Fuse everything legal: ``LoopToMap`` (loops -> parallel maps where sound) then ``MapFusion``
    (V+H) to a fixed point, then ``simplify``. The deterministic Phase-1 default -- the maximally-fused
    baseline granularity the agent fissions down from. Returns the number of transformation steps.

    The map-fusion fixed point is exactly what draining :func:`enumerate_fusions` reaches move-by-move;
    the batch form here is the deterministic policy, the arm surface the agent's per-move equivalent.
    """
    steps = sdfg.apply_transformations_repeated([LoopToMap]) or 0
    steps += sdfg.apply_transformations_repeated([MapFusionVertical, MapFusionHorizontal]) or 0
    sdfg.simplify()
    return steps


register_fusion_strategy("maximal-fusion", maximal_fusion)

__all__ = [
    "FusionStrategy",
    "register_fusion_strategy",
    "get_fusion_strategy",
    "fusion_strategy_names",
    "maximal_fusion",
    # agent per-move surface (re-exported)
    "FusionMove",
    "enumerate_fusions",
    "apply_fusion",
    "can_fuse",
    "fission_to_statements",
    "map_fission_moves",
]
