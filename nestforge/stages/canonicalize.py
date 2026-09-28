# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 1: DaCe canonicalization to the canonical parallel form, split at its ``fuse`` stage so stage 2 owns
the fusion decisions."""

from __future__ import annotations

from dataclasses import dataclass

import dace
from dace.transformation.passes.canonicalize import canonicalize as dace_canonicalize
from dace.transformation.passes.canonicalize import stage_labels
from dace.transformation.passes.symbol_propagation import SymbolPropagation

from nestforge.ir.names import inline_top_level_nsdfgs

FUSE_STAGE = "fuse"


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
    return dace_canonicalize(sdfg, target=targets.canon_target, stages=labels[: labels.index(FUSE_STAGE)])


def after_fusion(targets: Targets) -> list[str]:
    labels = stage_labels(targets.canon_target)
    return labels[labels.index(FUSE_STAGE) + 1 :]


def fuse_and_finish(sdfg: dace.SDFG, targets: Targets) -> dace.SDFG:
    """Stage 2 default: canonicalization's own fusion stage, then the stages after it."""
    # map fusion never descends into a nested SDFG, and a re-inlined kernel arrives nested
    inline_top_level_nsdfgs(sdfg)
    return dace_canonicalize(sdfg, target=targets.canon_target, stages=[FUSE_STAGE, *after_fusion(targets)])


def finish(sdfg: dace.SDFG, targets: Targets) -> dace.SDFG:
    """The stages after fusion, once moves chose the granularity by hand."""
    return dace_canonicalize(sdfg, target=targets.canon_target, stages=after_fusion(targets))
