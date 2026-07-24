# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Baseline comparison lanes (paper C1/C2): the traditional optimizers the per-nest granularity search is
measured against. Each is an existing :class:`~nestforge.optimizers.Optimizer`, so no new build machinery:

  * ``gcc-O3`` / ``llvm-O3`` -- the production compilers' own auto-vectorization (external lane, sequential).
  * ``graphite`` / ``polly`` -- gcc Graphite / clang Polly AUTO-PARALLELIZATION (external lane, auto-par).
    Its polyhedral back end is probed; an absent one yields ``propose() -> None`` + a ``skip_reason``.
  * ``whole-program`` -- DaCe ``auto-opt`` over the WHOLE un-split program: the honest "what a
    whole-program optimizer already gets for free" floor a per-nest win must beat.
  * ``pluto`` -- the polyhedral source-to-source lane, gated on ``polycc`` (containerized in optarena,
    usually absent on a dev box); it skips with a recorded reason, never silently.

The point (C1/C2): show a measured per-nest granularity search beats every one of these, including on
non-affine kernels the polyhedral lanes cannot schedule.
"""
from __future__ import annotations

import shutil
from typing import List, Optional, Tuple

from nestforge.optimizers import ExternalOptimizer, Optimizer, WholeProgramOptimizer

#: The Pluto driver whose presence gates the polyhedral lane.
PLUTO_TOOL = "polycc"


def baseline_optimizers(gcc: str = "gcc", clang: str = "clang", nthreads: int = 1) -> List[Optimizer]:
    """The traditional baselines the search is compared against, as optimizers. The auto-par lanes carry a
    ``skip_reason`` (not a crash) when their polyhedral back end is absent; ``whole-program`` always
    proposes (DaCe). Pluto is separate (:func:`pluto_available`) -- it is not an ``Optimizer`` because its
    emit/compile path differs from the numpyto external lane."""
    return [
        ExternalOptimizer("c", "gnu", gcc, name="gcc-O3"),
        ExternalOptimizer("c", "llvm", clang, name="llvm-O3"),
        ExternalOptimizer("c", "gnu", gcc, parallel="auto-par", nthreads=nthreads, name="graphite"),
        ExternalOptimizer("c", "llvm", clang, parallel="auto-par", nthreads=nthreads, name="polly"),
        WholeProgramOptimizer(opt_mode="auto-opt", name="whole-program"),
    ]


def pluto_available() -> Tuple[bool, Optional[str]]:
    """Whether the Pluto polyhedral lane can run -- its source-to-source compiler ``polycc`` on ``PATH``.
    Returns ``(True, None)`` or ``(False, reason)``; the lane records the reason and skips, never
    silently truncating the baseline set."""
    if shutil.which(PLUTO_TOOL):
        return True, None
    return False, f"pluto unavailable: {PLUTO_TOOL!r} not on PATH (containerized in optarena)"


def baseline_names() -> List[str]:
    """Every baseline lane name, including the tool-gated ``pluto``. Ordered for a stable report column."""
    return [opt.name for opt in baseline_optimizers()] + ["pluto"]
