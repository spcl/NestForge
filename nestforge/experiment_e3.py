"""E3 driver (paper C3): the offloading-granularity curve.

C3 says choosing WHICH fine regions to externalize pays -- that a monolithic whole-program compile leaves
performance on the table, and that offloading granularity is itself a search axis. E1 fixes the offloading
unit and sweeps the fusion partition (Axis 1); E3 is its transpose: fix the partition and sweep the unit
(Axis 2), coarse to fine, measuring the whole program in context at each rung.

The measurement is E1's, deliberately. :func:`nestforge.experiment_e1.run_e1_cell` already takes the unit
it lowers at, and :class:`~nestforge.experiment_e1.E1Cell` already records it, so E3 needs no new cell type
and no second measurement path -- the two axes stay directly comparable because the same code timed them.
What E3 adds is the read-off: the per-kernel curve over :data:`~nestforge.offload.OFFLOAD_UNITS` and the
argmin, which is where the offload boundary starts costing more than the finer region wins back.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from nestforge import tsvc
from nestforge.arena import discover_compilers
from nestforge.experiment_e1 import E1Cell, run_e1_cell
from nestforge.granularity import granularity_ladder
from nestforge.offload import OFFLOAD_UNITS, offload_coarseness


def run_e3(kernels: Sequence[tsvc.TsvcKernel],
           out_dir: Union[str, Path],
           units: Sequence[str] = OFFLOAD_UNITS,
           opt_mode: str = "canonicalize",
           backends: Optional[Dict[str, str]] = None,
           reps: int = 7,
           preset: str = "S",
           seed: int = 0) -> List[E1Cell]:
    """The bounded E3 sweep: every (kernel, backend, offloading unit) at ONE fixed granularity rung.

    The rung is the ladder's coarsest point, held constant so the unit is the only axis moving -- a curve
    that varied both would not attribute its shape to either. Cell count is
    ``kernels x backends x len(units)``.

    Units are measured coarse -> fine so the returned cells read as a curve. A kernel with no nest at a
    given unit (no ``LoopRegion`` for ``cfg``, say) comes back as a failed cell with the reason, exactly as
    in E1 -- a skip-with-reason, never a silent drop.
    """
    out_dir = Path(out_dir)
    backends = backends or discover_compilers()
    ordered = sorted(units, key=offload_coarseness)
    cells: List[E1Cell] = []
    # Broad on purpose, and for E1's reason: a corpus sweep meets dace InvalidSDFGError, TypeError from
    # extract, and more, and every failure is RECORDED on its cell rather than ending the sweep.
    caught = Exception
    for kernel in kernels:
        try:  # a kernel that cannot canonicalize has no rung to hold fixed, so every cell of it is a skip
            canonical = tsvc.build_sdfg(kernel, opt_mode)
            point = granularity_ladder(canonical, max_points=1)[0]
        except caught as e:
            for backend_name in backends:
                for unit in ordered:
                    cells.append(E1Cell(kernel.key, backend_name, "-", unit, float("inf"), False, repr(e)))
            continue
        for backend_name, backend_path in backends.items():
            for unit in ordered:
                cell_dir = out_dir / kernel.key / backend_name / unit
                try:
                    cells.append(
                        run_e1_cell(kernel,
                                    backend_name,
                                    backend_path,
                                    point,
                                    cell_dir,
                                    unit,
                                    opt_mode,
                                    reps,
                                    canonical=copy.deepcopy(canonical),
                                    preset=preset,
                                    seed=seed))
                except caught as e:
                    cells.append(E1Cell(kernel.key, backend_name, point.name, unit, float("inf"), False, repr(e)))
    return cells


def granularity_curve(cells: Sequence[E1Cell]) -> Dict[Tuple[str, str], List[Tuple[str, float]]]:
    """Per (kernel, backend), the measured time at each offloading unit, coarse -> fine. This is the E3
    figure: where the curve turns is where the offload boundary costs more than the finer region wins."""
    curve: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}
    for c in cells:
        if not c.ok:
            continue
        curve.setdefault((c.kernel, c.backend), []).append((c.unit, c.median_us))
    for series in curve.values():
        series.sort(key=lambda pair: offload_coarseness(pair[0]))
    return curve


def best_unit_per_backend(cells: Sequence[E1Cell]) -> Dict[Tuple[str, str], str]:
    """The C3 read-off: for each (kernel, backend), the offloading unit with the fastest valid time. A
    finest-unit win supports fine-grained externalization; a coarsest-unit win is the boundary cost
    dominating, which is the same claim measured from the other side."""
    best: Dict[Tuple[str, str], Tuple[float, str]] = {}
    for c in cells:
        if not c.ok:
            continue
        key = (c.kernel, c.backend)
        if key not in best or c.median_us < best[key][0]:
            best[key] = (c.median_us, c.unit)
    return {key: unit for key, (_us, unit) in best.items()}
