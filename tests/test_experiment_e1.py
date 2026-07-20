"""E1 driver (C1): the fusion-granularity x backend heatmap. The read-off logic is a unit test (synthetic
cells, no compile); the end-to-end per-backend variant build + swap + measure is an integration test that
compiles and forks, gated on a C toolchain."""
import numpy as np
import pytest
import dace

from nestforge.arena import discover_compilers
from nestforge.experiment_e1 import E1Cell, best_granularity_per_backend, run_e1, run_e1_cell
from nestforge.granularity import GranularityPoint, fuse_first_k
from nestforge.tsvc import TsvcKernel

N = dace.symbol('N')


@dace.program
def two_map(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    T = np.empty_like(A)
    for i in dace.map[0:N]:
        T[i] = A[i] + B[i]
    for i in dace.map[0:N]:
        C[i] = T[i] * 2.0


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
    ]
    assert best_granularity_per_backend(cells) == {("k", "gcc"): "maximal"}


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


@pytest.mark.integration
def test_run_e1_sweeps_backends_and_granularity_bounded(tmp_path):
    cells = run_e1([kernel()], tmp_path, unit="map", max_granularity_points=2, reps=3)
    n_backends = len(discover_compilers())
    assert cells and len(cells) <= n_backends * 2  # kernels(1) x backends x <=2 granularity rungs
    assert all(c.ok and c.error is None for c in cells), [c.error for c in cells if not c.ok]
    best = best_granularity_per_backend(cells)
    assert set(b for _k, b in best) == set(discover_compilers())  # every backend produced a winner
