"""E5 driver (C2): the non-affine case study. The classifier and the read-off are unit tests (real SDFGs,
no compile); the measured granularity sweep on a non-affine kernel is integration."""
import pytest
import sympy
import dace

from nestforge.experiment_e1 import E1Cell
from nestforge.experiment_e5 import (iterator_degree, non_affine_findings, partial_volume, polyhedral_schedulable,
                                     run_e5, summarize)
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


def gather_kernel():
    return TsvcKernel(key="gather", program=gather_prog, regime="1d", params={}, corpus="tsvc2")


def test_indirection_is_detected_and_affine_code_is_not():
    # the C2 premise: A[idx[k]] is outside the polyhedral fragment, A[k]*2 is inside it. Getting this
    # backwards would put easy kernels in the hard set and invent the case study.
    ok, reason = polyhedral_schedulable(affine_prog.to_sdfg(simplify=True))
    assert ok and reason == ""
    ok, reason = polyhedral_schedulable(gather_prog.to_sdfg(simplify=True))
    assert not ok
    assert "indirection" in reason and "'A'" in reason  # auditable: names the array and what was seen


def test_partial_volume_tests_provably_equal_not_provably_greater():
    """A memlet spanning 0:N with volume 1 is an indirection. N carries no positivity assumption, so
    `(N - 1).is_positive` is None -- read as False that would pass every indirection through as affine."""
    assert (sympy.Symbol("N") - 1).is_positive is None  # the trap this guards

    class FakeSubset:

        def __init__(self, n):
            self.n = n

        def num_elements(self):
            return self.n

    class FakeMemlet:

        def __init__(self, subset, volume):
            self.subset, self.volume = subset, volume

    n = sympy.Symbol("N")
    assert partial_volume(FakeMemlet(FakeSubset(n), n)) is False  # whole region touched -> affine
    assert partial_volume(FakeMemlet(FakeSubset(n), 1)) is True  # region spanned, one element -> indirect
    assert partial_volume(FakeMemlet(None, 1)) is False  # no subset is not evidence of indirection


def test_iterator_degree_reports_none_for_the_unprovable():
    i = sympy.Symbol("i")
    assert iterator_degree(i + 3, i) == 1  # affine
    assert iterator_degree(i**2, i) == 2  # nonlinear
    assert iterator_degree(sympy.floor(i / 4), i) is None  # not a polynomial -> cannot prove affine


def test_speedup_is_over_the_coarsest_rung():
    # "what you get if you do not search" is the coarsest rung, so the gain is attributable to the
    # granularity choice alone -- the polyhedral lane cannot run here, so it cannot be the divisor.
    cells = [
        E1Cell("k", "gcc", "coarse", "map", 10.0, True),
        E1Cell("k", "gcc", "fine", "map", 4.0, True),
    ]
    row = summarize("k", "gcc", False, "indirection", "coarse", cells)
    assert row.ok and row.best == "fine" and row.speedup == 2.5
    assert non_affine_findings([row]) == {"k": 2.5}


def test_no_baseline_means_no_speedup_claim():
    # the coarsest rung failing leaves nothing to divide by; reporting the winner's time as a speedup
    # would compare it against itself.
    cells = [E1Cell("k", "gcc", "fine", "map", 4.0, True)]
    row = summarize("k", "gcc", False, "indirection", "coarse", cells)
    assert not row.ok and row.speedup == 0.0 and "no baseline" in row.error
    assert non_affine_findings([row]) == {}  # never enters the C2 table


def test_affine_kernels_are_recorded_as_excluded_not_dropped(tmp_path, monkeypatch):
    # the case-study set must be auditable: an affine kernel appears with its verdict, so the reader sees
    # what was excluded and why rather than a filtered list. No compile.
    import nestforge.experiment_e5 as e5

    monkeypatch.setattr(e5.tsvc, "build_sdfg", lambda k, m: affine_prog.to_sdfg(simplify=True))
    kernel = TsvcKernel(key="affine", program=affine_prog, regime="1d", params={}, corpus="tsvc2")
    rows = run_e5([kernel], tmp_path, backends={"gcc": "gcc"})
    assert len(rows) == 1
    assert rows[0].schedulable and not rows[0].ok
    assert "excluded" in rows[0].error
    assert non_affine_findings(rows) == {}  # an affine kernel never supports the C2 claim


def test_run_e5_records_failures_without_crashing(tmp_path, monkeypatch):
    import nestforge.experiment_e5 as e5

    def boom(*a, **k):
        raise ValueError("cannot canonicalize")

    monkeypatch.setattr(e5.tsvc, "build_sdfg", boom)
    rows = run_e5([gather_kernel()], tmp_path, backends={"gcc": "gcc", "clang": "clang"})
    assert len(rows) == 2
    assert all(not r.ok and "cannot canonicalize" in r.error for r in rows)


@pytest.mark.integration
def test_run_e5_sweeps_granularity_on_a_non_affine_kernel(tmp_path, monkeypatch):
    # the real case study: a kernel the polyhedral model rejects, swept for granularity, measured in
    # full-program context and validated bit-exact.
    import nestforge.experiment_e5 as e5

    monkeypatch.setattr(e5.tsvc, "build_sdfg", lambda k, m: gather_prog.to_sdfg(simplify=True))
    rows = run_e5([gather_kernel()], tmp_path, max_granularity_points=2, reps=3, backends={"gcc": "gcc"})
    assert len(rows) == 1
    row = rows[0]
    assert not row.schedulable and "indirection" in row.reason
    assert row.error is None, row.error
    assert row.ok and row.best_us < float("inf") and row.speedup > 0.0
