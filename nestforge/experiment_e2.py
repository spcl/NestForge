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
from typing import Dict, List, Optional, Sequence, Tuple, Union

from nestforge import tsvc
from nestforge.arena import discover_compilers
from nestforge.baselines import pluto_available
from nestforge.build import BuildOptions, DEFAULT_COMPILER, DEFAULT_FLAGS
from nestforge.experiment_e1 import E1Cell
from nestforge.optimizers import WholeProgramOptimizer
from nestforge.whole_program import measure_whole_program

#: Baselines whose lane exists but whose measurement path does not yet: the external compilers optimize a
#: single emitted nest, not the whole program (:func:`whole_program.measure_whole_program` builds the DaCe
#: scope). Listed so the table shows them as pending rather than omitting them.
EXTERNAL_WHOLE_PROGRAM_PENDING = ("gcc-O3", "llvm-O3", "graphite", "polly")

#: C compiler (what the arena builds nests with) -> the C++ driver that builds the whole program. The two
#: sides of the ratio must be the SAME toolchain: a search side taking min(gcc, clang) against a baseline
#: always built by g++ books part of a compiler swap as a granularity finding.
BASELINE_DRIVER = {"gcc": "g++", "clang": "clang++"}

#: The whole-program baseline is built at the search side's FP regime. build_backend_variants pins every
#: nest to ``ieee-strict`` (arena: ``-ffp-contract=off``), so a baseline left at the driver default forms
#: FMAs the search side is denied and the difference lands inside the ratio.
NO_FP_CONTRACT = "-ffp-contract=off"


@dataclass(frozen=True, slots=True)
class E2Row:
    """One (kernel, backend, baseline) comparison. ``speedup`` is ``baseline_us / search_us`` -- above 1.0
    means the search won. It is ``nan`` when either side has no valid measurement, and ``error`` says
    which."""
    kernel: str
    backend: str
    baseline: str
    baseline_us: float
    search_us: float
    speedup: float
    ok: bool
    error: Optional[str] = None


def search_best(cells: Sequence[E1Cell]) -> Dict[Tuple[str, str], float]:
    """Per (kernel, BACKEND), the fastest VALID time any search cell reached.

    Keyed by backend because the baseline is built by one specific toolchain: taking ``min`` across
    backends here and dividing by a single-compiler baseline would book a compiler swap as a granularity
    win. Failed cells never enter, so a pair whose every cell failed is absent rather than present at
    ``inf`` (which would divide to a speedup of 0.0 and read as a measured loss)."""
    best: Dict[Tuple[str, str], float] = {}
    for c in cells:
        if not c.ok:
            continue
        key = (c.kernel, c.backend)
        if key not in best or c.median_us < best[key]:
            best[key] = c.median_us
    return best


def run_e2(kernels: Sequence[tsvc.TsvcKernel],
           cells: Sequence[E1Cell],
           out_dir: Union[str, Path],
           preset: str = "S",
           reps: int = 7,
           seed: int = 0,
           timeout: float = 900.0,
           backends: Optional[Dict[str, str]] = None) -> List[E2Row]:
    """Compare the search winner against every baseline lane, per (kernel, backend).

    ``cells`` are the already-measured E1/E3 cells -- pass the same ``preset``/``seed`` they were swept at,
    or the ratio spans two problem sizes. The baseline is built by the SAME backend whose search cells it
    is divided by, at the same FP regime, so the ratio isolates the search axes. A (kernel, backend) pair
    with no valid search cell still yields rows, marked with the reason, so coverage stays visible.
    """
    out_dir = Path(out_dir)
    best = search_best(cells)
    backends = backends or discover_compilers()
    rows: List[E2Row] = []
    # Broad for the sweep's reason (see run_e1): a baseline build meets dace, translator and toolchain
    # failures, and each is recorded on its row rather than ending the comparison.
    caught = Exception
    for kernel in kernels:
        for backend in backends:
            search_us = best.get((kernel.key, backend), float("inf"))
            missing = None if (kernel.key, backend) in best else "no valid search cell for this kernel/backend"
            try:
                res = measure_whole_program(baseline_optimizer(backend),
                                            kernel,
                                            out_dir / kernel.key / backend / "whole-program",
                                            preset=preset,
                                            reps=reps,
                                            seed=seed,
                                            timeout=timeout)
                # The search-side gap is reported on its OWN field, never folded into the baseline's error:
                # a baseline that built, validated and timed fine must not be recorded as a lane that
                # could not run just because the search produced nothing to divide.
                rows.append(
                    compare(kernel.key, backend, "whole-program", res.median_us, search_us, res.ok, res.error, missing))
            except caught as e:
                rows.append(
                    E2Row(kernel.key, backend, "whole-program", float("inf"), search_us, float("nan"), False, repr(e)))

            ok_pluto, why = pluto_available()
            # Gated lanes are rows with a reason, never absent rows: a baseline set that shrinks to whatever
            # happened to be installed flatters the search by omission.
            pluto_reason = why if not ok_pluto else "pluto lane not wired into the whole-program path yet"
            rows.append(E2Row(kernel.key, backend, "pluto", float("inf"), search_us, float("nan"), False, pluto_reason))
            for name in EXTERNAL_WHOLE_PROGRAM_PENDING:
                rows.append(
                    E2Row(kernel.key, backend, name, float("inf"), search_us, float("nan"), False,
                          f"{name} optimizes one emitted nest; the external WHOLE-program lane is future work"))
    return rows


def baseline_optimizer(backend: str) -> WholeProgramOptimizer:
    """The whole-program baseline built by ``backend``'s C++ driver at the search side's FP regime.

    Both halves matter: the driver so the ratio is not part compiler swap, and ``-ffp-contract=off`` so the
    baseline is denied the FMAs every offloaded nest is denied (build_backend_variants pins ieee-strict)."""
    driver = BASELINE_DRIVER.get(backend, DEFAULT_COMPILER)
    return WholeProgramOptimizer(opt_mode="auto-opt",
                                 build=BuildOptions(compiler=driver, flags=DEFAULT_FLAGS + [NO_FP_CONTRACT]),
                                 name="whole-program")


def compare(kernel: str, backend: str, baseline: str, baseline_us: float, search_us: float, baseline_ok: bool,
            baseline_error: Optional[str], search_error: Optional[str]) -> E2Row:
    """One row from a measured baseline. The ratio needs BOTH sides valid and finite -- a baseline that
    failed validation, or a pair the search never solved, yields ``nan``, never a fabricated ratio.

    The two failure reasons stay distinguishable: ``baseline_error`` says the lane could not measure,
    ``search_error`` says there was nothing to divide. Folding them together makes a working baseline lane
    read as skipped."""
    usable = (baseline_ok and baseline_error is None and search_error is None and 0.0 < baseline_us < float("inf")
              and 0.0 < search_us < float("inf"))
    error = baseline_error or search_error
    speedup = baseline_us / search_us if usable else float("nan")
    return E2Row(kernel, backend, baseline, baseline_us, search_us, speedup, usable, error)


def speedup_table(rows: Sequence[E2Row]) -> Dict[Tuple[str, str], Dict[str, float]]:
    """The E2 figure: (kernel, backend) -> baseline -> speedup, valid rows only. A baseline missing from a
    pair's row set is a lane that could not run there; :func:`skipped_lanes` reports why."""
    table: Dict[Tuple[str, str], Dict[str, float]] = {}
    for r in rows:
        if r.ok:
            table.setdefault((r.kernel, r.backend), {})[r.baseline] = r.speedup
    return table


def skipped_lanes(rows: Sequence[E2Row]) -> Dict[Tuple[str, str, str], str]:
    """(kernel, backend, baseline) -> why it produced no comparison.

    Keyed per ROW, not per lane: a lane-only key lets the last kernel's reason overwrite every earlier
    one, so a lane that failed on 1 of 90 kernels reads identically to one that never ran at all --
    and the reason reported is whichever kernel happened to iterate last."""
    return {(r.kernel, r.backend, r.baseline): r.error for r in rows if not r.ok and r.error}
