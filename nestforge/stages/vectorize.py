# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``auto-vectorize-scope``: DaCe's vectorizer on one top-level doall (``VectorizeMap``), every other doall left
scalar. Its innermost maps become tiles of ``tile_sizes`` lanes for ``isa``; the body becomes tile loads, tile
operations and tile stores, which the C view of the tree shows as the ISA's tile functions over its intrinsics."""

import copy
from collections.abc import Sequence
from typing import cast

from dace.sdfg import nodes
from dace.sdfg.state import SDFGState
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA
from dace.transformation.passes.vectorization.vectorize_map import VectorizeMap

from nestforge.records import CRASHES

__all__ = [
    "AUTO_VECTORIZE",
    "DEFAULT_LANES",
    "ISAS",
    "MAX_TILE_DIMS",
    "SCALABLE",
    "tile_lanes",
    "vectorize_config",
    "vectorize_refusal",
]

#: The move kind; it takes ``tile_sizes`` (lanes per vectorized dimension) and ``isa``.
AUTO_VECTORIZE = "auto-vectorize-scope"
#: The ISA names the move takes, by DaCe's tile-op ISA.
ISAS: dict[str, ISA] = {
    "auto": ISA.AUTO,
    "avx512": ISA.AVX512,
    "avx2": ISA.AVX2,
    "sve": ISA.ARM_SVE,
    "neon": ISA.ARM_NEON,
    "scalar": ISA.SCALAR,
}
#: The ISAs whose vector length is the hardware's: their tiles take :data:`DEFAULT_LANES`, never ``tile_sizes``.
SCALABLE = frozenset({"sve"})
#: The lanes of a tile listed by default, and of every tile of a scalable ISA.
DEFAULT_LANES = 8
#: The most dimensions DaCe's vectorizer tiles at once (the innermost ones).
MAX_TILE_DIMS = 3


def tile_lanes(isa: str, tile_sizes: Sequence[int]) -> tuple[int, ...] | str:
    """The lanes per vectorized dimension, innermost last, or why ``tile_sizes`` gives none."""
    if isa in SCALABLE:
        return (DEFAULT_LANES,) * max(1, min(len(tile_sizes), MAX_TILE_DIMS))
    if not 1 <= len(tile_sizes) <= MAX_TILE_DIMS or any(size < 1 for size in tile_sizes):
        return (
            f"pass tile_sizes: the lanes of each vectorized dimension, innermost last, 1 to {MAX_TILE_DIMS} sizes "
            f"(e.g. [{DEFAULT_LANES}]); a scalable ISA ({sorted(SCALABLE)}) takes its own vector length instead."
        )
    return tuple(tile_sizes)


def vectorize_config(isa: str, tile_sizes: Sequence[int]) -> VectorizeConfig | str:
    """The vectorizer configuration for ``isa`` and ``tile_sizes``, or why there is none."""
    if isa not in ISAS:
        return f"isa takes one of {list(ISAS)}; not {isa!r}."
    lanes = tile_lanes(isa, tile_sizes)
    # the tiles are expanded: the body is the ISA's tile functions over its intrinsics, as the C view prints it
    return (
        lanes if isinstance(lanes, str) else VectorizeConfig(widths=lanes, target_isa=ISAS[isa], expand_tile_nodes=True)
    )


def vectorize_refusal(state: SDFGState, entry: nodes.MapEntry, config: VectorizeConfig) -> str | None:
    """Why :class:`VectorizeMap` cannot vectorize ``entry``, found on a copy of the whole program; None when it can."""
    if state.entry_node(entry) is not None:
        return f"{entry.map.label} is inside another doall; vectorize the outermost one around it."
    trial = copy.deepcopy(state.sdfg.root_sdfg)
    trial_state = cast(SDFGState, trial.cfg_list[state.parent_graph.cfg_id].node(state.block_id))
    trial_entry = cast(nodes.MapEntry, trial_state.node(state.node_id(entry)))
    try:
        VectorizeMap(config, trial_state, trial_entry).apply_pass(trial_state.sdfg, {})
        trial.validate()
    except CRASHES as err:  # the vectorizer's refusal, naming what it cannot tile
        return f"blocked by VectorizeMap: {type(err).__name__}: {err}"
    return None
