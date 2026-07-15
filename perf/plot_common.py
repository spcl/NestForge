"""Shared helpers for the ``perf/plot_*.py`` result readers.

These scripts are dace-free: they read the per-kernel JSON a driver already wrote and render a chart,
never importing ``nestforge`` (which would pull the whole DaCe + optarena stack into a matplotlib
reader). So the three primitives every plot script needs live here, in ``perf/`` alongside them, rather
than in ``nestforge.perf.harness`` -- a plain ``from plot_common import ...`` resolves because ``perf/``
is ``sys.path[0]`` when a plot script is run as ``python perf/plot_<x>.py``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional


def finite(x) -> bool:
    """True only for a real, usable numeric time: a finite int/float. Rejects ``None`` (a non-finite value
    the driver mapped to ``null`` on write) and ``inf`` / ``nan``."""
    return isinstance(x, (int, float)) and math.isfinite(x)


def geomean(xs: List[float]) -> Optional[float]:
    """Geometric mean of the finite positive values; ``None`` when there are none."""
    vals = [x for x in xs if finite(x) and x > 0.0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def load_results(results_dir: Path) -> List[dict]:
    """Every per-kernel JSON in ``results_dir`` (``tables.md`` and unparseable files skipped)."""
    rows: List[dict] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "tables.md":
            continue
        try:
            rows.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError, ValueError):
            continue
    return rows
