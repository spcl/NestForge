"""E5 driver (paper C2): the non-affine case study.

C2 says the search runs at SDFG level over the normal form, so it reaches kernels a polyhedral scheduler
cannot -- CloudSC and friends, where the subscripts are data-dependent or nonlinear and Pluto/PPCG have no
schedule to build. E5 is the evidence: classify each kernel, keep the ones the polyhedral model rejects,
and show a granularity choice still buys time there.

The baseline is deliberately internal. E2 already compares against whole-program and the polyhedral lanes;
here the polyhedral lane BY CONSTRUCTION cannot run, so dividing by it would be dividing by nothing. What
E5 divides by is the coarsest granularity rung -- "what you get if you do not search" -- which makes the
speedup attributable to the granularity choice alone and keeps the claim inside what was measured.

:func:`polyhedral_schedulable` is deliberately CONSERVATIVE: it answers "can a polyhedral scheduler
certainly handle this", so anything it cannot prove affine is reported as non-affine WITH the reason.
Over-reporting a kernel as non-affine would weaken E5 by putting an easy kernel in the hard set, which is
the safe direction to be wrong in -- and the reason string makes every classification auditable rather
than a bare boolean.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import sympy

import dace
from dace.sdfg.analysis.polyhedral_isl import to_isl

from nestforge import tsvc
from nestforge.arena import discover_compilers
from nestforge.experiment_e1 import E1Cell, run_e1_cell
from nestforge.granularity import granularity_ladder


def subset_indices(memlet: dace.Memlet) -> List[sympy.Expr]:
    """Every index expression in a memlet's subset, as sympy. A subset with no ranges (an empty memlet)
    contributes nothing rather than raising."""
    subset = memlet.subset
    if subset is None:
        return []
    return [sympy.sympify(part) for rng in subset.ndrange() for part in rng]


#: DaCe names the symbol carrying a loaded value ``__sym_<array>``, which is how an indirection survives
#: into a nested SDFG's subset (``A[idx[i]]`` -> ``__tmp_r[__sym___tmp_idx]``). Matching the bare array name
#: alone misses that spelling entirely.
VALUE_SYMBOL_PREFIX = "__sym_"


def data_dependent_on(expr: sympy.Expr, arrays: Set[str]) -> Optional[str]:
    """The array whose VALUE ``expr`` indexes through, or ``None`` when it indexes only symbols.

    Compares by NAME, never by symbol identity: a dace ``symbolic.symbol`` carries assumptions, so a
    same-name ``sympy.Symbol`` is a distinct object with a different hash and set operations between the
    two are silently empty (dace's own ``is_linear_in_param`` documents this trap).
    """
    for sym in expr.free_symbols:
        name = str(sym)
        base = name[len(VALUE_SYMBOL_PREFIX):] if name.startswith(VALUE_SYMBOL_PREFIX) else name
        if base in arrays:
            return base
    return None


def quasi_affine(expr: sympy.Expr) -> bool:
    """Whether ISL can represent ``expr`` exactly -- affine plus integer floor/ceil/mod.

    Delegates to dace's :func:`~dace.sdfg.analysis.polyhedral_isl.to_isl`, which raises on a nonlinear
    term, a non-integer coefficient, or a symbolic divisor. Using it rather than a degree test is what
    keeps TILED and STRIDED domains (``int_floor(N, 8)``) inside the affine fragment where they belong --
    a strict degree<=1 test rejects them, which would push ordinary tiled kernels into the case study.
    """
    try:
        to_isl(expr)
    except (ValueError, TypeError, AttributeError, NotImplementedError):
        return False
    return True


def polyhedral_schedulable(sdfg: dace.SDFG) -> Tuple[bool, str]:
    """Whether a polyhedral scheduler could certainly build a schedule for ``sdfg``.

    Two disqualifiers, each fatal to the polyhedral model:
      * **data-dependent subscripts** -- an index that reads an array's VALUE (``A[idx[i]]``), directly or
        through dace's ``__sym_<array>`` spelling inside a nested SDFG;
      * **non-quasi-affine subscripts** -- an index ISL cannot represent exactly (``A[i*j]``, ``A[i*i]``).

    Returns ``(True, "")`` or ``(False, reason)``. Conservative: an index that cannot be analyzed is
    reported as non-affine with that as the reason.

    Memlet VOLUME is deliberately not consulted. Volume counts total accesses across enclosing iterations,
    not distinct elements, so an ordinary affine read inside a loop reports a volume far LARGER than its
    subset spans (``a[j] = b[j] + 1`` over ``0:N`` carries volume ``N**2``). Testing subset-vs-volume
    therefore classifies every ordinary kernel as an indirection, which is why this reads the subscripts.
    """
    for sub in sdfg.all_sdfgs_recursive():
        arrays = set(sub.arrays)  # per-SDFG: a nested indirection indexes through that SDFG's own arrays
        for state in sub.all_states():
            for edge in state.edges():
                if edge.data is None or edge.data.data is None:
                    continue
                try:
                    exprs = subset_indices(edge.data)
                except (TypeError, ValueError, AttributeError, sympy.SympifyError) as e:
                    return False, f"subset of {edge.data.data!r} is not analyzable: {e!r}"
                for expr in exprs:
                    dependent = data_dependent_on(expr, arrays)
                    if dependent is not None:
                        return False, (f"data-dependent subscript on {edge.data.data!r}: index {expr} reads "
                                       f"the value of {dependent!r}")
                    if not quasi_affine(expr):
                        return False, (f"non-affine subscript on {edge.data.data!r}: index {expr} is outside "
                                       f"the affine fragment ISL can schedule")
    return True, ""


@dataclass(frozen=True, slots=True)
class E5Row:
    """One case-study kernel. ``schedulable`` is the polyhedral verdict (``reason`` says why not);
    ``speedup`` is ``coarsest_us / best_us`` -- what searching granularity bought over not searching."""
    kernel: str
    backend: str
    schedulable: bool
    reason: str
    best: str
    best_us: float
    coarsest: str
    coarsest_us: float
    speedup: float
    ok: bool
    error: Optional[str] = None


def run_e5(kernels: Sequence[tsvc.TsvcKernel],
           out_dir: Union[str, Path],
           unit: str = "map",
           max_granularity_points: int = 4,
           opt_mode: str = "canonicalize",
           backends: Optional[Dict[str, str]] = None,
           reps: int = 7,
           preset: str = "S",
           seed: int = 0,
           only_non_affine: bool = True) -> List[E5Row]:
    """Classify each kernel, then sweep granularity on the ones the polyhedral model rejects.

    ``only_non_affine`` keeps E5 to its claim; set it False to sweep the affine kernels too (useful to show
    the classifier is not simply labelling everything hard). A kernel the classifier ACCEPTS is still
    returned as a row -- with its verdict and no measurement -- so the case-study set is auditable rather
    than a filtered list with no record of what was dropped.
    """
    out_dir = Path(out_dir)
    backends = backends or discover_compilers()
    rows: List[E5Row] = []
    caught = Exception  # recorded per row (see run_e1)
    for kernel in kernels:
        try:
            canonical = tsvc.build_sdfg(kernel, opt_mode)
            schedulable, reason = polyhedral_schedulable(canonical)
            ladder = granularity_ladder(canonical, max_points=max_granularity_points)
        except caught as e:
            for backend_name in backends:
                rows.append(
                    E5Row(kernel.key, backend_name, False, "", "-", float("inf"), "-", float("inf"), 0.0, False,
                          repr(e)))
            continue
        if schedulable and only_non_affine:
            # Recorded, not silently dropped: the reader must see which kernels were excluded and why.
            for backend_name in backends:
                rows.append(
                    E5Row(kernel.key, backend_name, True, "affine: polyhedral lane applies", "-", float("inf"), "-",
                          float("inf"), 0.0, False, "excluded from the non-affine case study"))
            continue
        for backend_name, backend_path in backends.items():
            cells: List[E1Cell] = []
            for point in ladder:
                try:
                    cells.append(
                        run_e1_cell(kernel,
                                    backend_name,
                                    backend_path,
                                    point,
                                    out_dir / kernel.key / backend_name / point.name,
                                    unit,
                                    opt_mode,
                                    reps,
                                    canonical=copy.deepcopy(canonical),
                                    preset=preset,
                                    seed=seed))
                except caught as e:
                    cells.append(E1Cell(kernel.key, backend_name, point.name, unit, float("inf"), False, repr(e)))
            # ladder[-1], NOT ladder[0]: granularity_ladder runs k = 0 (atoms, the FINEST partition) up to
            # k = depth (maximal). "What you get if you do not search" is maximal fusion -- what a compiler
            # picks blindly -- so dividing by ladder[0] would divide by the fully-fissioned program instead.
            rows.append(summarize(kernel.key, backend_name, schedulable, reason, ladder[-1].name, cells))
    return rows


def summarize(kernel: str, backend: str, schedulable: bool, reason: str, coarsest: str,
              cells: Sequence[E1Cell]) -> E5Row:
    """Fold one kernel's granularity cells into a case-study row. The speedup needs the coarsest rung AND a
    winner to both have measured -- with no coarsest baseline there is nothing to attribute the gain to."""
    valid = {c.granularity: c.median_us for c in cells if c.ok}
    if not valid:
        return E5Row(kernel, backend, schedulable, reason, "-", float("inf"), coarsest, float("inf"), 0.0, False,
                     "no granularity rung measured")
    best = min(valid, key=valid.get)
    base_us = valid.get(coarsest, float("inf"))
    usable = 0.0 < base_us < float("inf")
    speedup = base_us / valid[best] if usable else 0.0
    error = None if usable else f"coarsest rung {coarsest!r} did not measure; no baseline to divide by"
    return E5Row(kernel, backend, schedulable, reason, best, valid[best], coarsest, base_us, speedup, usable, error)


def non_affine_findings(rows: Sequence[E5Row]) -> Dict[Tuple[str, str], float]:
    """The C2 read-off: (kernel, backend) -> speedup, over rows that are BOTH non-affine and measured.
    These are the kernels where no polyhedral scheduler could have found the win.

    Keyed by BACKEND too, like E1's and E3's read-offs: one row per (kernel, backend) is emitted, so a
    kernel-only key would silently publish whichever backend iterated last -- a headline number that
    changes when a compiler is installed or removed."""
    return {(r.kernel, r.backend): r.speedup for r in rows if r.ok and not r.schedulable}
