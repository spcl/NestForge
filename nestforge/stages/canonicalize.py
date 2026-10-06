# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 1: DaCe canonicalization to the canonical parallel form, split at its ``fuse`` stage so stage 2 owns
the fusion decisions."""

from __future__ import annotations

from dataclasses import dataclass

import dace
from dace.sdfg import nodes
from dace.sdfg.propagation import propagate_memlets_nested_sdfg, propagate_memlets_state
from dace.transformation.passes.canonicalize import canonicalize as dace_canonicalize
from dace.transformation.passes.canonicalize import stage_labels
from dace.transformation.passes.symbol_propagation import SymbolPropagation

from nestforge.ir.names import inline_top_level_nsdfgs

FUSE_STAGE = "fuse"
FINAL_FUSE_STAGE = "fuse_final"


@dataclass(frozen=True, slots=True)
class Targets:
    """Devices the program may run on. The CPU is always a target; the GPU is opt-in."""

    gpu: bool = False

    @property
    def canon_target(self) -> str:
        return "gpu" if self.gpu else "cpu"


def canonicalize(sdfg: dace.SDFG, targets: Targets) -> dace.SDFG:
    """Canonicalize ``sdfg`` in place up to, excluding, the fusion stage."""
    # the frontend binds derived loop bounds to fresh interstate symbols that extraction cannot pass in
    SymbolPropagation().apply_pass(sdfg, {})
    labels = stage_labels(targets.canon_target)
    dace_canonicalize(sdfg, target=targets.canon_target, stages=labels[: labels.index(FUSE_STAGE)])
    narrow_nested_memlets(sdfg)
    return sdfg


def narrow_nested_memlets(sdfg: dace.SDFG) -> None:
    """A nest in a map takes whole arrays, and canonicalization leaves its outer memlets whole; narrowed to what one
    iteration touches, a fusion across the map sees the per-iteration access instead of the whole array."""
    for state in sdfg.states():
        nests = [n for n in state.nodes() if isinstance(n, nodes.NestedSDFG) and state.entry_node(n) is not None]
        for nest in nests:
            propagate_memlets_nested_sdfg(state.sdfg, state, nest)
        if nests:
            propagate_memlets_state(state.sdfg, state)


def after_fusion(targets: Targets) -> list[str]:
    labels = stage_labels(targets.canon_target)
    return labels[labels.index(FUSE_STAGE) + 1 :]


def fuse_and_finish(sdfg: dace.SDFG, targets: Targets) -> dace.SDFG:
    """Stage 2 default: canonicalization's own fusion stage, then the stages after it."""
    # map fusion never descends into a nested SDFG, and a re-inlined kernel arrives nested
    inline_top_level_nsdfgs(sdfg)
    return dace_canonicalize(sdfg, target=targets.canon_target, stages=[FUSE_STAGE, *after_fusion(targets)])


def finish(sdfg: dace.SDFG, targets: Targets) -> dace.SDFG:
    """The stages after fusion but the terminal re-fusion, once moves chose the granularity by hand."""
    stages = [label for label in after_fusion(targets) if label != FINAL_FUSE_STAGE]
    return dace_canonicalize(sdfg, target=targets.canon_target, stages=stages)
