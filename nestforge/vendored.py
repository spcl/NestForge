# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Make the pinned hpcagent-bench checkout importable, so a bare clone has kernels.

The hpcagent_bench corpus is nest-forge's DEFAULT corpus (:func:`nestforge.tsvc.iter_tsvc_kernels`) and is
imported at the top of :mod:`nestforge.corpus` and :mod:`nestforge.tsvc` -- so without it those modules fail
at IMPORT time, and a clone can measure nothing at all. ``external/hpcagent-bench`` is a git submodule
tracking ``main``; ``git submodule update --init --remote`` checks it out.

Importing this module applies the path fix. That is deliberate side-effect-on-import: the fix has to land
before any ``import hpcagent_bench`` anywhere, and :mod:`nestforge` therefore imports this first (the same
shape as the ``dace.transformation.passes`` ordering directly below it).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

#: The submodule checkout. Kernels live under ``<here>/hpcagent_bench/benchmarks/``.
VENDORED_BENCH = Path(__file__).resolve().parent.parent / "external" / "hpcagent-bench"


def use_vendored_bench() -> bool:
    """Put the submodule on ``sys.path`` when hpcagent_bench is not installed; report whether it was used.

    An installed hpcagent_bench (editable or not) always WINS: a developer working across both repos must
    keep getting their own checkout, not a second copy pinned here. This only rescues the case the submodule
    exists for -- a fresh clone with no separate install.

    Returns ``False`` (rather than raising) when the submodule was never checked out. The failure then
    surfaces at the real import with hpcagent_bench's own name on it, which is the accurate message; raising
    here would blame the submodule for an environment that may simply have the package elsewhere."""
    if importlib.util.find_spec("hpcagent_bench") is not None:
        return False
    if not (VENDORED_BENCH / "hpcagent_bench" / "__init__.py").is_file():
        return False
    sys.path.insert(0, str(VENDORED_BENCH))
    return True


use_vendored_bench()
