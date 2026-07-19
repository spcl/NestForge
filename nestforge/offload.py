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
from typing import List, Union

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion

from nestforge.extract import NestNode, extract_nest_to_sdfg, whole_program_boundary
from nestforge.pass_lower import lower_nests_to_external_call
from nestforge.strategies import (Strategy, get_strategy, is_parallel_nest, register_strategy, strategy_names)

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


__all__ = [
    "OffloadGranularity",
    "DEFAULT_GRANULARITY",
    "OffloadCandidate",
    "offload_candidates",
    "label_nest",
    # registry (from nestforge.strategies)
    "register_strategy",
    "get_strategy",
    "strategy_names",
    # commit + coarsest-granularity surface (re-exported)
    "lower_nests_to_external_call",
    "extract_nest_to_sdfg",
    "whole_program_boundary",
]
