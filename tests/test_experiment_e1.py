# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""E1 driver (C1): the fusion-granularity x backend heatmap. The read-off logic is a unit test (synthetic
cells, no compile); the end-to-end per-backend variant build + swap + measure is an integration test that
compiles and forks, gated on a C toolchain."""
import pytest
import dace

from nestforge.arena import discover_compilers
from nestforge.experiment_e1 import E1Cell, best_granularity_per_backend, no_granularity_axis, run_e1, run_e1_cell
from nestforge.granularity import GranularityPoint, fuse_first_k
from nestforge.tsvc import TsvcKernel

N = dace.symbol('N')


@dace.program
def two_map(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], D: dace.float64[N]):
    """TWO global outputs on purpose. Statement granularity is one map per GLOBAL output
    (:func:`nestforge.fission_arms.fission_to_statements`), so a producer->consumer chain through a
    TRANSIENT is a single statement: it fissions to one nest, its ladder holds only ``atoms``, and an E1
    sweep over it measures no granularity axis at all. Writing C and D independently keeps the two maps
    separable, so ``atoms`` and ``maximal`` are genuinely different partitions to compare."""
    for i in dace.map[0:N]:
        C[i] = A[i] + B[i]
    for i in dace.map[0:N]:
        D[i] = A[i] * 2.0


def kernel():
    return TsvcKernel(key="two_map", program=two_map, regime="1d", params={}, corpus="tsvc2")


def test_best_granularity_per_backend_reads_the_argmin():
    # the C1 finding shape: the fastest rung differs by backend -> the optimum MOVES with the backend.
    cells = [
        E1Cell("k", "gcc", "atoms", "map", 10.0, True),
        E1Cell("k", "gcc", "maximal", "map", 4.0, True),  # gcc likes fused
        E1Cell("k", "clang", "atoms", "map", 3.0, True),  # clang likes split
        E1Cell("k", "clang", "maximal", "map", 7.0, True),
    ]
    best = best_granularity_per_backend(cells)
    assert best[("k", "gcc")] == "maximal"
    assert best[("k", "clang")] == "atoms"  # different winner => backend-dependent (C1)


def test_best_granularity_ignores_failed_cells():
    cells = [
        E1Cell("k", "gcc", "atoms", "map", float("inf"), False, "build failed"),
        E1Cell("k", "gcc", "maximal", "map", 5.0, True),
        E1Cell("k", "gcc", "fuse-1", "map", 9.0, True),
    ]
    assert best_granularity_per_backend(cells) == {("k", "gcc"): "maximal"}  # the failed rung is not the argmin


def test_single_surviving_rung_is_no_axis_not_a_winner():
    """One measured rung is not a preference: with the alternative unbuilt there was no choice to make, so
    the pair is excluded from the C1 table and reported as having no granularity axis."""
    cells = [
        E1Cell("k", "gcc", "atoms", "map", float("inf"), False, "build failed"),
        E1Cell("k", "gcc", "maximal", "map", 5.0, True),
    ]
    assert best_granularity_per_backend(cells) == {}
    assert no_granularity_axis(cells) == ["k | gcc"]


def test_one_rung_ladder_never_reports_a_best():
    """A single-statement kernel canonicalizes to ONE nest, so its ladder holds only ``atoms``. Reporting
    that as the backend's preferred granularity would fabricate a C1 finding from a table with no
    alternative in it -- the first TSVC kernels (s000, s111, ...) are all this shape."""
    cells = [
        E1Cell("s000", "gcc", "atoms", "map", 1.2, True),
        E1Cell("s000", "clang", "atoms", "map", 1.9, True),
    ]
    assert best_granularity_per_backend(cells) == {}
    assert no_granularity_axis(cells) == ["s000 | clang", "s000 | gcc"]


@pytest.mark.integration
def test_run_e1_cell_swaps_backend_variant_and_stays_bit_exact(tmp_path):
    # the swap path that variants={} never exercised: build a real per-backend archive, route the nests to
    # its extern-C symbols (implementation=ExternCall), run the whole program forked, validate bit-exact.
    compilers = discover_compilers()
    assert compilers, "need gcc/clang on PATH"
    name, path = next(iter(compilers.items()))
    atoms = GranularityPoint("atoms", fuse_first_k(0))
    cell = run_e1_cell(kernel(), name, path, atoms, tmp_path, unit="map", reps=3)
    assert isinstance(cell, E1Cell)
    assert cell.error is None, cell.error
    assert cell.ok and cell.median_us < float("inf")  # swapped-in native variant == oracle, bit-exact


def test_run_e1_records_a_failed_cell_instead_of_crashing(tmp_path):
    # a corpus kernel whose variant cannot emit/compile must be a recorded skip-with-reason, not a crash
    # that loses the rest of the sweep. Force a bogus compiler so every variant build fails.
    cells = run_e1([kernel()],
                   tmp_path,
                   unit="map",
                   max_granularity_points=2,
                   backends={"broken": "/nonexistent/cc"},
                   reps=2)
    assert cells and all(not c.ok and c.error for c in cells)  # every cell failed, none raised
    assert best_granularity_per_backend(cells) == {}  # no valid winner from all-failed cells


def test_run_e1_records_kernel_ladder_failure_without_crashing(monkeypatch, tmp_path):
    # a kernel whose ladder build raises must be a recorded skip for every backend, not a sweep-ending
    # crash (the ladder build sits before the per-cell try, so it needs its own guard). No compile.
    import nestforge.experiment_e1 as e1

    def boom(*a, **k):
        raise ValueError("cannot canonicalize")

    monkeypatch.setattr(e1, "granularity_ladder", boom)
    cells = run_e1([kernel()], tmp_path, backends={"gcc": "gcc", "clang": "clang"}, reps=2)
    assert len(cells) == 2  # one skip cell per backend
    assert all(not c.ok and "cannot canonicalize" in c.error for c in cells)


@pytest.mark.integration
def test_run_e1_sweeps_backends_and_granularity_bounded(tmp_path):
    cells = run_e1([kernel()], tmp_path, unit="map", max_granularity_points=2, reps=3)
    n_backends = len(discover_compilers())
    assert cells and len(cells) <= n_backends * 2  # kernels(1) x backends x <=2 granularity rungs
    assert all(c.ok and c.error is None for c in cells), [c.error for c in cells if not c.ok]
    # Guard the sweep against measuring nothing: a one-rung ladder would still fill in cells and, before
    # the read-off excluded it, would have reported "every backend prefers atoms" with no alternative ever
    # compiled. Both rungs must actually be present for the argmin below to mean anything.
    assert {c.granularity for c in cells} == {"atoms", "maximal"}
    best = best_granularity_per_backend(cells)
    assert set(b for _k, b in best) == set(discover_compilers())  # every backend produced a winner
    assert not no_granularity_axis(cells)


def test_unit_with_no_nest_is_a_skip_not_fabricated_data(tmp_path):
    # unit='cfg' on a flat kernel selects zero nests. Measuring anyway would time the ALL-REFERENCE program
    # under a backend label -- identical for every backend because no backend compiled anything -- which
    # fabricates "backend-independent" heatmap data. It must be a skip-with-reason instead. No compile.
    atoms = GranularityPoint("atoms", fuse_first_k(0))
    cell = run_e1_cell(kernel(), "gcc", "gcc", atoms, tmp_path, unit="cfg", reps=2)
    assert not cell.ok and cell.median_us == float("inf")
    assert "cfg" in cell.error and "no" in cell.error.lower()
    assert best_granularity_per_backend([cell]) == {}  # never enters the heatmap


def test_a_zero_argument_nest_is_reported_where_it_happens(tmp_path, monkeypatch):
    """An empty ABI order used to surface at expansion as "has no abi_order", which names the arena --
    the wrong place to look. It must fail where the empty signature is read, naming the nest's boundary."""
    from nestforge import experiment_e1

    class FakeBoundary:
        inputs, outputs, symbols = (), (), ()

    class FakeExt:
        name = "extcall_1"

    (tmp_path / "extcall_1").mkdir(parents=True)
    (tmp_path / "extcall_1" / "k.c").write_text("void extcall_1_fp64(void) {}\n")
    monkeypatch.setattr(experiment_e1, "prepare", lambda boundary, name, vdir: None)
    monkeypatch.setattr(experiment_e1, "emit_sources", lambda prep, vdir, target: [vdir / "k.c"])
    monkeypatch.setattr(experiment_e1, "signature_order", lambda text, symbol: [])

    with pytest.raises(ValueError, match="crosses no data"):
        experiment_e1.build_backend_variants([(FakeExt(), FakeBoundary())], "gcc", "/usr/bin/gcc", tmp_path)
