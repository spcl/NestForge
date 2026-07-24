"""The experiment sweep matrix -- kept BOUNDED, and a measurement-cost ledger.

Auto-discovering the corpus (:func:`nestforge.tsvc.iter_tsvc_kernels`) and crossing it with the fusion
granularity ladder (:mod:`nestforge.granularity`) and the offloading units (:mod:`nestforge.offload`)
is a product that explodes. So the sweep is capped at every axis -- kernel count, granularity points per
kernel, offloading units -- and the caps are env-overridable but small by default. :func:`sweep_upper_bound`
is the analytic ceiling a test pins, so a careless widening cannot silently make the suite run for hours.

The :class:`MeasureLedger` counts how many whole-program measurements a search spent -- the cost side of
the paper's C4 comparison (a traditional exhaustive sweep measures every cell; an agentic search should
reach the same winner with far fewer).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Tuple

from nestforge import tsvc
from nestforge.granularity import granularity_ladder
from nestforge.offload import OFFLOAD_UNITS

#: Small by default; each is env-overridable for a real (cluster) sweep. The defaults keep the in-repo
#: experiment/test runnable in seconds, not hours.
DEFAULT_KERNEL_LIMIT = int(os.environ.get("NF_SWEEP_KERNELS", "3"))
DEFAULT_GRANULARITY_POINTS = int(os.environ.get("NF_SWEEP_GRAN_POINTS", "3"))


def parse_units(raw: str) -> Tuple[str, ...]:
    """Parse the ``NF_SWEEP_UNITS`` list, dropping blanks (a trailing comma or an empty value) and REJECTING
    an unknown unit up front. Without this an empty entry becomes a phantom unit that inflates
    :func:`sweep_upper_bound` and only fails later, deep in the sweep, as a ``get_strategy('')`` KeyError."""
    units = tuple(u.strip() for u in raw.split(",") if u.strip())
    unknown = [u for u in units if u not in OFFLOAD_UNITS]
    if unknown:
        raise ValueError(f"unknown offload unit(s) {unknown} in NF_SWEEP_UNITS={raw!r}; known: {OFFLOAD_UNITS}")
    return units or OFFLOAD_UNITS


DEFAULT_UNITS: Tuple[str, ...] = parse_units(os.environ.get("NF_SWEEP_UNITS", ",".join(OFFLOAD_UNITS)))


@dataclass(frozen=True)
class SweepCell:
    """One point of the bounded sweep: a kernel, a fusion-granularity ladder rung, an offloading unit."""
    kernel: str
    granularity: str
    unit: str


def bounded_kernels(corpus: str = "tsvc2",
                    only: Sequence[str] = None,
                    limit: int = DEFAULT_KERNEL_LIMIT) -> List[tsvc.TsvcKernel]:
    """Auto-discover the corpus, then CAP to ``limit`` kernels (or the explicit ``only`` set). The cap is
    applied after discovery so adding kernels to the corpus never silently grows the sweep."""
    kernels = tsvc.iter_tsvc_kernels(only=list(only) if only else None, corpus=corpus)
    return kernels[:limit]


def sweep_upper_bound(n_kernels: int,
                      max_granularity_points: int = DEFAULT_GRANULARITY_POINTS,
                      units: Sequence[str] = DEFAULT_UNITS) -> int:
    """The analytic ceiling on cell count: ``kernels x granularity-points x units``. The real
    :func:`sweep_cells` count is <= this (a kernel's ladder may be shorter than ``max_granularity_points``).
    A test pins this so the matrix cannot silently blow up."""
    return n_kernels * max_granularity_points * len(units)


def sweep_cells(kernels: Sequence[tsvc.TsvcKernel],
                max_granularity_points: int = DEFAULT_GRANULARITY_POINTS,
                units: Sequence[str] = DEFAULT_UNITS,
                opt_mode: str = "canonicalize") -> List[SweepCell]:
    """The bounded cell list: each kernel x its (subsampled) granularity ladder x each offloading unit.
    Per-kernel granularity is capped to ``max_granularity_points`` (:func:`granularity.granularity_ladder`),
    so the total is <= :func:`sweep_upper_bound`."""
    cells: List[SweepCell] = []
    for kernel in kernels:
        sdfg = tsvc.build_sdfg(kernel, opt_mode)
        for point in granularity_ladder(sdfg, max_points=max_granularity_points):
            for unit in units:
                cells.append(SweepCell(kernel.key, point.name, unit))
    return cells


@dataclass
class MeasureLedger:
    """Counts the whole-program measurements a search spent (the cost side of C4). ``tokens`` is the
    agent's extra budget, left 0 for a deterministic search."""
    measurements: int = 0
    tokens: int = 0
    seen: List[str] = field(default_factory=list)

    def measure(self, label: str, fn: Callable[[], float]) -> float:
        """Run one measurement, counting it. ``label`` records which cell was measured (for the ledger dump
        and to spot a search re-measuring a cell it already saw)."""
        self.measurements += 1
        self.seen.append(label)
        return fn()
