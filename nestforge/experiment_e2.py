"""E2 driver (paper C1/C2/C3): search speedup over the traditional baselines.

E1 and E3 establish that the optimal granularity and the optimal offloading unit MOVE with the backend.
E2 asks the question that makes those findings matter: does searching those axes beat what a production
compiler or a whole-program optimizer already gives for free? Each kernel's search winner (the fastest
valid cell any E1/E3 sweep produced) is divided by each baseline's time on the same kernel.

Comparability is the whole experiment, so the numbers are pinned rather than defaulted. Both sides measure
the WHOLE program, at the same ``preset`` and ``seed``, forked, and validated bit-exact against the numpy
oracle before any timing counts -- a speedup over a baseline that computed the wrong answer is not a
speedup. The search side comes in as cells the caller already measured (:func:`experiment_e1.run_e1` /
:func:`experiment_e3.run_e3`), so E2 never re-times it with a second set of knobs.

Lanes that cannot run are RECORDED, never dropped: pluto without ``polycc`` and the external
whole-program lanes (gcc/clang over the whole emitted source, still future work) come back as rows with a
reason and ``ok=False``. A baseline set that silently shrinks to the lanes that happened to be installed
would flatter the search by omission -- the missing lane must be visible in the table.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from nestforge import tsvc
from nestforge.baselines import pluto_available
from nestforge.experiment_e1 import E1Cell
from nestforge.optimizers import WholeProgramOptimizer
from nestforge.whole_program import measure_whole_program

#: Baselines whose lane exists but whose measurement path does not yet: the external compilers optimize a
#: single emitted nest, not the whole program (:func:`whole_program.measure_whole_program` builds the DaCe
#: scope). Listed so the table shows them as pending rather than omitting them.
EXTERNAL_WHOLE_PROGRAM_PENDING = ("gcc-O3", "llvm-O3", "graphite", "polly")


@dataclass(frozen=True)
class E2Row:
    """One (kernel, baseline) comparison. ``speedup`` is ``baseline_us / search_us`` -- above 1.0 means the
    search won. It is ``nan`` when either side has no valid measurement, and ``error`` says which."""
    kernel: str
    baseline: str
    baseline_us: float
    search_us: float
    speedup: float
    ok: bool
    error: Optional[str] = None


def search_best(cells: Sequence[E1Cell]) -> Dict[str, float]:
    """Per kernel, the fastest VALID time any search cell reached -- the search's answer for that kernel,
    whichever backend/granularity/unit produced it. Failed cells never enter, so a kernel whose every cell
    failed is absent rather than present at ``inf``."""
    best: Dict[str, float] = {}
    for c in cells:
        if not c.ok:
            continue
        if c.kernel not in best or c.median_us < best[c.kernel]:
            best[c.kernel] = c.median_us
    return best


def run_e2(kernels: Sequence[tsvc.TsvcKernel],
           cells: Sequence[E1Cell],
           out_dir,
           preset: str = "S",
           reps: int = 7,
           seed: int = 0,
           timeout: float = 900.0) -> List[E2Row]:
    """Compare the search winner against every baseline lane, per kernel.

    ``cells`` are the already-measured E1/E3 cells -- pass the same ``preset``/``seed`` they were swept at,
    or the ratio spans two problem sizes. ``kernels`` supplies the baseline side; a kernel with no valid
    search cell still yields rows, marked with the reason, so the corpus coverage stays visible.
    """
    out_dir = Path(out_dir)
    best = search_best(cells)
    rows: List[E2Row] = []
    # Broad for the sweep's reason (see run_e1): a baseline build meets dace, translator and toolchain
    # failures, and each is recorded on its row rather than ending the comparison.
    caught = Exception
    for kernel in kernels:
        search_us = best.get(kernel.key, float("inf"))
        missing = None if kernel.key in best else "no valid search cell for this kernel"
        try:
            res = measure_whole_program(WholeProgramOptimizer(opt_mode="auto-opt", name="whole-program"),
                                        kernel,
                                        out_dir / kernel.key / "whole-program",
                                        preset=preset,
                                        reps=reps,
                                        seed=seed,
                                        timeout=timeout)
            rows.append(compare(kernel.key, "whole-program", res.median_us, search_us, res.ok, res.error or missing))
        except caught as e:
            rows.append(E2Row(kernel.key, "whole-program", float("inf"), search_us, float("nan"), False, repr(e)))

        ok_pluto, why = pluto_available()
        # Gated lanes are rows with a reason, never absent rows: a baseline set that shrinks to whatever
        # happened to be installed flatters the search by omission.
        pluto_reason = why if not ok_pluto else "pluto lane not wired into the whole-program path yet"
        rows.append(E2Row(kernel.key, "pluto", float("inf"), search_us, float("nan"), False, pluto_reason))
        for name in EXTERNAL_WHOLE_PROGRAM_PENDING:
            rows.append(
                E2Row(kernel.key, name, float("inf"), search_us, float("nan"), False,
                      f"{name} optimizes one emitted nest; the external WHOLE-program lane is future work"))
    return rows


def compare(kernel: str, baseline: str, baseline_us: float, search_us: float, baseline_ok: bool,
            error: Optional[str]) -> E2Row:
    """One row from a measured baseline. The ratio needs BOTH sides valid and finite -- a baseline that
    failed validation or a kernel the search never solved yields ``nan``, never a fabricated ratio."""
    usable = baseline_ok and error is None and 0.0 < baseline_us < float("inf") and 0.0 < search_us < float("inf")
    speedup = baseline_us / search_us if usable else float("nan")
    return E2Row(kernel, baseline, baseline_us, search_us, speedup, usable, error)


def speedup_table(rows: Sequence[E2Row]) -> Dict[str, Dict[str, float]]:
    """The E2 figure: kernel -> baseline -> speedup, valid rows only. A baseline missing from a kernel's
    row set is a lane that could not run there; :func:`skipped_lanes` reports why."""
    table: Dict[str, Dict[str, float]] = {}
    for r in rows:
        if r.ok:
            table.setdefault(r.kernel, {})[r.baseline] = r.speedup
    return table


def skipped_lanes(rows: Sequence[E2Row]) -> Dict[str, str]:
    """Baseline -> why it produced no comparison, so the paper can state the gap instead of the table
    quietly having one fewer column."""
    return {r.baseline: r.error for r in rows if not r.ok and r.error}
