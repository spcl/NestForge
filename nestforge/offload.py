"""Phase 2 of the 4-phase optimizer: decide offload granularity.

Phase 1 fixed the fusion granularity; Phase 2 decides WHICH nests leave the SDFG as external library
calls. An offload *granularity* is a detection strategy (:mod:`nestforge.strategies`): it selects the
nests to externalize. The default is top-level compute nests -- the outermost nests, skipping pure
map/loop scheduling wrappers that carry no compute of their own (:data:`DEFAULT_GRANULARITY`).

Architectural invariant -- externalize BEFORE deciding offload. A nest is turned into a library call
FIRST; only THEN does each backend tool decide whether its kernel is offloadable (to GPU, say). No
lane may pre-decide offload before extraction, or an offload choice could shift the extraction
underneath it. So the Phase-2 commit is :func:`lower_nests_to_external_call`: every selected nest
becomes an ``ExternalCall`` node.

Agent surface. Granularity is a choice the agent inspects before committing:

  * :func:`offload_candidates` -- non-mutating: what a granularity WOULD externalize, each nest
    labeled and marked parallel/sequential. The Phase-2 analog of ``enumerate_fusions``.
  * :func:`lower_nests_to_external_call` (from :mod:`nestforge.pass_lower`) -- commit: swap every
    selected nest for an ``ExternalCall`` (stays runnable via the ``DaceReference`` expansion).
  * :func:`whole_program_boundary` (from :mod:`nestforge.extract`) -- the coarsest granularity: the
    whole un-split program as a single unit, no extraction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion

from nestforge.extract import NestNode, extract_nest_to_sdfg, whole_program_boundary
from nestforge.pass_lower import lower_nests_to_external_call
from nestforge.strategies import (Strategy, get_strategy, is_parallel_nest, register_strategy, strategy_names,
                                  top_level_map_entries)

#: A Phase-2 offload granularity is a detection strategy: SDFG -> the nests to externalize.
OffloadGranularity = Strategy

#: The default granularity: top-level compute nests (outermost, skipping scheduling wrappers).
DEFAULT_GRANULARITY = "skip-taskloops"


def label_nest(node: NestNode) -> str:
    """A short human/agent-readable label for an offload candidate."""
    if isinstance(node, nodes.MapEntry):
        return f"map[{', '.join(node.map.params)}] over {node.map.range}"
    if isinstance(node, LoopRegion):
        return f"loop {node.label}"
    if isinstance(node, dace.SDFGState):
        return f"state {node.label}"
    raise TypeError(f"not an offload candidate: {type(node).__name__}")


@dataclass
class OffloadCandidate:
    """One nest a granularity would externalize -- the parent SDFG it lives in, the nest node, a
    label, and whether its emitted kernel may carry an OpenMP parallel scope (see
    :func:`nestforge.strategies.is_parallel_nest`)."""
    parent_sdfg: dace.SDFG
    node: NestNode
    label: str
    parallel: bool


def offload_candidates(sdfg: dace.SDFG,
                       granularity: Union[str, OffloadGranularity] = DEFAULT_GRANULARITY) -> List[OffloadCandidate]:
    """The nests ``granularity`` would externalize, WITHOUT mutating ``sdfg``.

    Detection is read-only -- extraction happens later in :func:`lower_nests_to_external_call`. Lets
    the agent see the offload set (and each nest's parallel/sequential nature) before committing.
    """
    strat = get_strategy(granularity) if isinstance(granularity, str) else granularity
    return [OffloadCandidate(parent, node, label_nest(node), is_parallel_nest(node)) for parent, node in strat(sdfg)]


#: Offloading granularity UNITS (paper Axis 2), COARSE -> FINE. The structural unit each external call
#: wraps, from the graph itself: a whole ``cfg`` (a control-flow region -- a ``LoopRegion``), a whole
#: ``state`` (an ``SDFGState`` and all the maps it holds), or a single ``map`` (one ``MapEntry`` within a
#: state). Coarser wraps more compute per call; finer isolates one map. A DISTINCT decision from fusion
#: granularity (Axis 1, :mod:`nestforge.granularity`) and COMPOSES with it: a ``map`` offload over the
#: atoms partition puts each statement-atom in its own external call. The coarsest endpoint (no
#: decomposition, the whole program as one unit) is :func:`whole_program_boundary`.
OFFLOAD_UNITS = ("cfg", "state", "map")


def state_has_compute(state: dace.SDFGState) -> bool:
    """Whether a state holds real compute (a map, tasklet, library node, or nested SDFG) -- not a bare
    copy/access-only state, which there is nothing to externalize."""
    return any(
        isinstance(n, (nodes.MapEntry, nodes.Tasklet, nodes.LibraryNode, nodes.NestedSDFG)) for n in state.nodes())


def unit_refs(sdfg: dace.SDFG, unit: str) -> List[Tuple[dace.SDFG, NestNode]]:
    """The (parent-SDFG, node) pairs to externalize at one offloading UNIT level -- recursive over nested
    SDFGs. ``map`` = every top-level map-nest; ``cfg`` = every ``LoopRegion``; ``state`` = every
    compute-bearing state (externalized whole)."""
    if unit == "map":
        return [(sub, me) for sub in sdfg.all_sdfgs_recursive() for st in sub.all_states()
                for me in top_level_map_entries(st)]
    if unit == "cfg":
        # top-level LoopRegions only: extract_loop_nest needs the loop's parent to BE the SDFG
        # (SubgraphView(parent_sdfg, [loop])), so a loop nested inside another region is not a cfg unit.
        return [(sub, r) for sub in sdfg.all_sdfgs_recursive() for r in sub.nodes() if isinstance(r, LoopRegion)]
    if unit == "state":
        return [(sub, st) for sub in sdfg.all_sdfgs_recursive() for st in sub.all_states() if state_has_compute(st)]
    raise ValueError(f"unknown offload unit {unit!r}; known: {OFFLOAD_UNITS}")


def offload_unit_axis() -> List[str]:
    """The offloading-granularity axis, coarse -> fine (Axis 2). ``offload_candidates(sdfg, unit)`` previews
    a unit; ``lower_nests_to_external_call(sdfg, unit)`` commits it (each unit is a registered strategy)."""
    return list(OFFLOAD_UNITS)


def offload_coarseness(unit: str) -> int:
    """Rank of an offloading unit, 0 = coarsest (``cfg``). Lets a sweep order the axis and a search step one
    rung finer/coarser."""
    return OFFLOAD_UNITS.index(unit)


for _unit in OFFLOAD_UNITS:  # each unit level is also a detection strategy, so the existing lowering path works
    register_strategy(_unit, (lambda u: lambda sdfg: unit_refs(sdfg, u))(_unit))

__all__ = [
    "OffloadGranularity",
    "DEFAULT_GRANULARITY",
    "OffloadCandidate",
    "offload_candidates",
    "label_nest",
    # offloading granularity axis (Axis 2): cfg / state / map units
    "OFFLOAD_UNITS",
    "offload_unit_axis",
    "offload_coarseness",
    "unit_refs",
    "state_has_compute",
    # registry (from nestforge.strategies)
    "register_strategy",
    "get_strategy",
    "strategy_names",
    # commit + coarsest-granularity surface (re-exported)
    "lower_nests_to_external_call",
    "extract_nest_to_sdfg",
    "whole_program_boundary",
]
