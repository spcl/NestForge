"""E5 driver (C2): the non-affine case study.

The classifier is tested against REAL SDFGs built through the production path (``tsvc.build_sdfg`` on
corpus kernels, ``@dace.program`` fixtures for the hard cases), never hand-built memlet fakes. An earlier
version of this file tested a fake with hand-chosen volumes and passed while the classifier labelled every
ordinary TSVC kernel non-affine -- the whole case study inverted, green suite.
"""
import numpy as np
import pytest
import dace

from nestforge import tsvc
from nestforge.arena import discover_compilers
from nestforge.experiment_e1 import E1Cell
from nestforge.experiment_e5 import (data_dependent_on, non_affine_findings, polyhedral_schedulable, quasi_affine,
                                     run_e5, summarize)
from nestforge.granularity import granularity_ladder
from nestforge.tsvc import TsvcKernel

N = dace.symbol('N')


@dace.program
def affine_prog(A: dace.float64[N], C: dace.float64[N]):
    for k in dace.map[0:N]:
        C[k] = A[k] * 2.0


@dace.program
def gather_prog(A: dace.float64[N], idx: dace.int64[N], C: dace.float64[N]):
    for k in dace.map[0:N]:
        C[k] = A[idx[k]]


@dace.program
def two_map_prog(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    # TWO fusable maps, so the granularity ladder has a real depth: atoms != maximal. A single-map kernel
    # collapses the ladder to one rung, where ladder[0] and ladder[-1] are the same object and an
    # index-the-wrong-end bug is invisible.
    T = np.empty_like(A)
    for i in dace.map[0:N]:
        T[i] = A[i] + B[i]
    for i in dace.map[0:N]:
        C[i] = T[i] * 2.0


@dace.program
def nonlinear_prog(A: dace.float64[N], C: dace.float64[N]):
    # SEQUENTIAL loop on purpose: canonicalization leaves a loop-carried dependence as a LoopRegion, never
    # a Map, so a classifier that harvests iterators from MapEntry alone is blind to exactly this shape.
    for i in range(1, 10):
        C[i * i] = C[i - 1] + A[i]


def gather_kernel():
    return TsvcKernel(key="gather", program=gather_prog, regime="1d", params={}, corpus="tsvc2")


def test_ordinary_corpus_kernels_are_affine():
    """The regression that matters most: every plain TSVC kernel must classify AFFINE.

    Built through tsvc.build_sdfg, the path run_e5 uses. The previous classifier consulted memlet VOLUME,
    which counts accesses across enclosing iterations rather than distinct elements, so `a[j] = b[j] + 1`
    reported "spans 0:LEN_1D but touches volume LEN_1D**2" and every kernel entered the case study --
    C2 fabricated wholesale. Real kernels are the only fixture that can catch that.
    """
    verdicts = {}
    for kernel in list(tsvc.iter_tsvc_kernels())[:6]:
        ok, reason = polyhedral_schedulable(tsvc.build_sdfg(kernel, "canonicalize"))
        verdicts[kernel.key] = (ok, reason)
    assert all(ok for ok, _ in verdicts.values()), {k: r for k, (ok, r) in verdicts.items() if not ok}


def test_indirection_is_rejected_through_the_nested_sdfg_spelling():
    """A[idx[k]] lowers into a NESTED SDFG whose subset reads `__sym___tmp_...`; the index array is in THAT
    SDFG's descriptors, not the top level's. Capturing arrays from the outer SDFG only misses it."""
    ok, reason = polyhedral_schedulable(gather_prog.to_sdfg(simplify=True))
    assert not ok
    assert "data-dependent" in reason and "value of" in reason  # auditable: names what it indexes through


def test_nonlinear_subscript_in_a_sequential_loop_is_rejected():
    """C[i*i] in a `for` loop, i.e. a LoopRegion with no MapEntry at all. The reason must name the
    non-affinity, not some unrelated disqualifier -- the case-study table cites these strings."""
    ok, reason = polyhedral_schedulable(nonlinear_prog.to_sdfg(simplify=True))
    assert not ok
    assert "non-affine" in reason and "i**2" in reason


def test_affine_fixture_stays_affine():
    ok, reason = polyhedral_schedulable(affine_prog.to_sdfg(simplify=True))
    assert ok and reason == ""


def test_quasi_affine_admits_tiled_domains_and_refuses_nonlinear():
    """Tiled/strided bounds are inside the polyhedral fragment; a degree<=1 test wrongly rejects them and
    would push ordinary tiled kernels into the case study."""
    sym = dace.symbolic.pystr_to_symbolic
    assert quasi_affine(sym("i + 3"))
    assert quasi_affine(sym("int_floor(i, 8)"))  # tiled
    assert quasi_affine(sym("i % 4"))  # strided
    assert not quasi_affine(sym("i * j"))
    assert not quasi_affine(sym("i ** 2"))


def test_data_dependence_compares_names_not_symbol_identity():
    """dace's symbolic.symbol carries assumptions, so a same-name sympy.Symbol is a DIFFERENT object with a
    different hash. Set operations between the two are silently empty -- which made the previous nonlinear
    check unreachable dead code."""
    expr = dace.symbolic.pystr_to_symbolic("__sym_idx")
    assert data_dependent_on(expr, {"idx"}) == "idx"  # the __sym_<array> spelling
    assert data_dependent_on(dace.symbolic.pystr_to_symbolic("idx"), {"idx"}) == "idx"  # and the bare one
    assert data_dependent_on(dace.symbolic.pystr_to_symbolic("k + 1"), {"idx"}) is None


def test_speedup_divides_by_the_coarsest_rung_of_a_real_ladder():
    """The baseline label must be the ladder's COARSEST rung. granularity_ladder runs atoms (finest) ->
    maximal, so run_e5 must index ladder[-1]; ladder[0] divides by the fully-fissioned program instead.
    Uses a real ladder, because the label names ("atoms"/"maximal") only exist there -- the previous test
    passed hand-written "coarse"/"fine" labels that no ladder ever produces, so it could not see this.
    """
    ladder = granularity_ladder(two_map_prog.to_sdfg(simplify=True), max_points=4)
    assert len(ladder) > 1, "need a fusable kernel or the two ends coincide and nothing is tested"
    assert ladder[0].name == "atoms" and ladder[-1].name == "maximal"  # finest -> coarsest
    coarsest = ladder[-1].name
    # The C2 shape: the coarsest rung (what a compiler picks blindly) is SLOWER than a searched one.
    cells = [
        E1Cell("k", "gcc", ladder[0].name, "map", 8.0, True),  # searched winner
        E1Cell("k", "gcc", coarsest, "map", 10.0, True),  # the baseline to divide by
    ]
    row = summarize("k", "gcc", False, "indirection", coarsest, cells)
    assert row.coarsest == coarsest and row.coarsest_us == 10.0
    assert row.best == ladder[0].name and row.best_us == 8.0
    assert row.speedup == 1.25  # 10/8. Dividing by ladder[0] instead would give 8/8 == 1.0 and erase C2.


def test_findings_are_keyed_by_backend_so_none_are_overwritten():
    """One row per (kernel, backend): a kernel-only key publishes whichever backend iterated last, so the
    headline C2 number would change when a compiler is installed or removed."""
    rows = [
        summarize("k", "gcc", False, "indirection", "maximal",
                  [E1Cell("k", "gcc", "maximal", "map", 10.0, True),
                   E1Cell("k", "gcc", "atoms", "map", 5.0, True)]),
        summarize("k", "clang", False, "indirection", "maximal",
                  [E1Cell("k", "clang", "maximal", "map", 10.0, True),
                   E1Cell("k", "clang", "atoms", "map", 2.0, True)]),
    ]
    found = non_affine_findings(rows)
    assert found == {("k", "gcc"): 2.0, ("k", "clang"): 5.0}


def test_no_baseline_means_no_speedup_claim():
    cells = [E1Cell("k", "gcc", "atoms", "map", 4.0, True)]
    row = summarize("k", "gcc", False, "indirection", "maximal", cells)
    assert not row.ok and row.speedup == 0.0 and "no baseline" in row.error
    assert non_affine_findings([row]) == {}


def test_all_cells_failed_is_recorded_and_never_enters_the_table():
    row = summarize("k", "gcc", False, "indirection", "maximal",
                    [E1Cell("k", "gcc", "atoms", "map", float("inf"), False, "build failed")])
    assert not row.ok and "no granularity rung measured" in row.error
    assert non_affine_findings([row]) == {}


def test_affine_kernels_are_recorded_as_excluded_not_dropped(tmp_path, monkeypatch):
    import nestforge.experiment_e5 as e5
    monkeypatch.setattr(e5.tsvc, "build_sdfg", lambda k, m: affine_prog.to_sdfg(simplify=True))
    kernel = TsvcKernel(key="affine", program=affine_prog, regime="1d", params={}, corpus="tsvc2")
    rows = run_e5([kernel], tmp_path, backends={"gcc": "gcc"})
    assert len(rows) == 1 and rows[0].schedulable and not rows[0].ok
    assert "excluded" in rows[0].error
    assert non_affine_findings(rows) == {}


def test_run_e5_records_failures_without_crashing(tmp_path, monkeypatch):
    import nestforge.experiment_e5 as e5

    def boom(*a, **k):
        raise ValueError("cannot canonicalize")

    monkeypatch.setattr(e5.tsvc, "build_sdfg", boom)
    rows = run_e5([gather_kernel()], tmp_path, backends={"gcc": "gcc", "clang": "clang"})
    assert len(rows) == 2
    assert all(not r.ok and "cannot canonicalize" in r.error for r in rows)


@pytest.mark.integration
def test_run_e5_measures_a_non_affine_kernel_end_to_end(tmp_path, monkeypatch):
    """The real case study: classify, sweep granularity, build, swap, validate bit-exact, time.

    Asserts a MEASURED row (ok, finite time, a real ladder label), not merely that rows exist -- every
    driver records failures as rows, so 'rows are non-empty' is true even when nothing built.
    """
    backends = discover_compilers()
    assert backends, "need gcc/clang on PATH"
    one = dict([next(iter(backends.items()))])
    import nestforge.experiment_e5 as e5
    monkeypatch.setattr(e5.tsvc, "build_sdfg", lambda k, m: gather_prog.to_sdfg(simplify=True))

    rows = run_e5([gather_kernel()], tmp_path, max_granularity_points=2, reps=3, backends=one)
    assert len(rows) == 1
    row = rows[0]
    assert not row.schedulable and "data-dependent" in row.reason
    assert row.error is None, row.error
    assert row.ok and row.best_us < float("inf") and row.coarsest_us < float("inf")
    assert row.best in {"atoms", "maximal"} or row.best.startswith("fuse-")  # a real ladder label
    assert row.speedup == pytest.approx(row.coarsest_us / row.best_us)
