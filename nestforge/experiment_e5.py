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
from typing import Dict, List, Optional, Sequence, Tuple

import sympy

import dace
from dace.sdfg import nodes

from nestforge import tsvc
from nestforge.arena import discover_compilers
from nestforge.experiment_e1 import E1Cell, run_e1_cell
from nestforge.granularity import granularity_ladder


def subset_indices(memlet) -> List[sympy.Expr]:
    """Every index expression in a memlet's subset, as sympy. A subset with no ranges (an empty memlet)
    contributes nothing rather than raising."""
    subset = memlet.subset
    if subset is None:
        return []
    exprs: List[sympy.Expr] = []
    for rng in subset.ndrange():
        exprs.extend(sympy.sympify(part) for part in rng)
    return exprs


def partial_volume(memlet) -> bool:
    """Whether a memlet spans a region but touches strictly less of it than the region holds.

    This is how DaCe spells an indirection: ``A[idx[i]]`` passes the WHOLE array (``subset=0:N``) with
    ``volume=1``, because which element is read is known only at run time. A plain affine access carries
    ``volume == num_elements`` for the region it names. Conditional and dynamic writes land here too, and
    they are equally outside the affine fragment, so the over-approximation points the safe way.
    """
    if memlet.subset is None:
        return False
    try:
        surplus = sympy.simplify(sympy.sympify(memlet.subset.num_elements()) - sympy.sympify(memlet.volume))
    except (TypeError, ValueError, AttributeError, sympy.SympifyError):
        return True  # an unanalyzable volume is not a proof of affinity
    # Test for "provably EQUAL", not for "provably greater": with N carrying no positivity assumption,
    # `(N - 1).is_positive` is None, and a None read as False would pass every indirection through as
    # affine. An affine access gives exactly 0 here; anything else is unproven and treated as indirect.
    return surplus.is_zero is not True


def iterator_degree(expr: sympy.Expr, iterator: sympy.Symbol) -> Optional[int]:
    """Degree of ``expr`` in one loop iterator, or ``None`` when it is not a polynomial in it (``A[i//k]``,
    ``A[f(i)]``). ``None`` means "cannot prove affine", which the caller must treat as non-affine rather
    than as degree 0."""
    try:
        return int(sympy.Poly(expr, iterator).degree())
    except (sympy.PolynomialError, sympy.GeneratorsNeeded, TypeError, ValueError):
        return None


def polyhedral_schedulable(sdfg: dace.SDFG) -> Tuple[bool, str]:
    """Whether a polyhedral scheduler could certainly build a schedule for ``sdfg``.

    Three disqualifiers, each fatal to the polyhedral model:
      * **indirection** -- a memlet that names a region but touches less of it than the region holds
        (:func:`partial_volume`), which is how ``A[idx[i]]`` survives lowering: the subset is the whole
        array and the volume is one element, because the position is a run-time value;
      * **data-dependent subscripts** -- an index whose free symbols include an ARRAY name, the same thing
        spelled directly in a subset;
      * **nonlinear subscripts** -- an index of degree > 1 in the iterators (``A[i*j]``, ``A[i*i]``), which
        is outside the affine fragment.

    Returns ``(True, "")`` or ``(False, reason)``. Conservative by design: an index it cannot analyze is
    reported as non-affine with that as the reason.
    """
    arrays = set(sdfg.arrays)
    for sub in sdfg.all_sdfgs_recursive():
        for state in sub.all_states():
            iterators = {p for n in state.nodes() if isinstance(n, nodes.MapEntry) for p in n.map.params}
            symbols = {sympy.Symbol(i) for i in iterators}
            for edge in state.edges():
                if edge.data is None or edge.data.data is None:
                    continue
                if partial_volume(edge.data):
                    return False, (f"indirection on {edge.data.data!r}: memlet spans {edge.data.subset} but "
                                   f"touches volume {edge.data.volume} -- the position is a run-time value")
                try:
                    exprs = subset_indices(edge.data)
                except (TypeError, ValueError, AttributeError, sympy.SympifyError) as e:
                    return False, f"subset of {edge.data.data!r} is not analyzable: {e!r}"
                for expr in exprs:
                    names = {str(s) for s in expr.free_symbols}
                    dependent = names & arrays
                    if dependent:
                        return False, (f"data-dependent subscript on {edge.data.data!r}: index {expr} reads "
                                       f"{sorted(dependent)}")
                    for it in sorted(symbols & expr.free_symbols, key=str):
                        degree = iterator_degree(expr, it)
                        if degree is None:
                            return False, (f"subscript on {edge.data.data!r} is not polynomial in {it}: {expr}")
                        if degree > 1:
                            return False, f"nonlinear subscript on {edge.data.data!r}: index {expr} in {it}"
    return True, ""


@dataclass(frozen=True)
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
           out_dir,
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
            rows.append(summarize(kernel.key, backend_name, schedulable, reason, ladder[0].name, cells))
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


def non_affine_findings(rows: Sequence[E5Row]) -> Dict[str, float]:
    """The C2 read-off: kernel -> speedup, over rows that are BOTH non-affine and measured. These are the
    kernels where no polyhedral scheduler could have found the win."""
    return {r.kernel: r.speedup for r in rows if r.ok and not r.schedulable}
