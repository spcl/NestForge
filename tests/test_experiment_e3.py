# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""E3 driver (C3): the offloading-granularity curve. The read-off logic is a unit test (synthetic cells, no
compile); the end-to-end unit sweep is an integration test that compiles and forks, gated on a toolchain."""
import numpy as np
import pytest
import dace

from nestforge.arena import discover_compilers
from nestforge.experiment_e1 import E1Cell
from nestforge.experiment_e3 import best_unit_per_backend, granularity_curve, run_e3
from nestforge.offload import OFFLOAD_UNITS
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


def test_best_unit_reads_the_argmin_over_the_offload_axis():
    # the C3 finding shape: a finer offloading unit beats the coarse one -> fine-grained externalization
    # pays, and WHERE it stops paying differs by backend.
    cells = [
        E1Cell("k", "gcc", "g0", "cfg", 10.0, True),
        E1Cell("k", "gcc", "g0", "map", 4.0, True),  # gcc: finer wins
        E1Cell("k", "clang", "g0", "cfg", 3.0, True),  # clang: the offload boundary costs more than it wins
        E1Cell("k", "clang", "g0", "map", 8.0, True),
    ]
    best = best_unit_per_backend(cells)
    assert best[("k", "gcc")] == "map"
    assert best[("k", "clang")] == "cfg"


def test_curve_is_ordered_coarse_to_fine_and_skips_failures():
    cells = [
        E1Cell("k", "gcc", "g0", "map", 4.0, True),  # deliberately out of axis order
        E1Cell("k", "gcc", "g0", "cfg", 10.0, True),
        E1Cell("k", "gcc", "g0", "state", float("inf"), False, "no nest at unit 'state'"),
    ]
    curve = granularity_curve(cells)
    assert curve[("k", "gcc")] == [("cfg", 10.0), ("map", 4.0)]  # coarse -> fine, failed rung absent
    assert granularity_curve([cells[2]]) == {}  # an all-failed kernel contributes no curve


def test_run_e3_records_failed_cells_instead_of_crashing(tmp_path):
    # a bogus compiler fails every variant build: each must be a recorded skip-with-reason covering the
    # whole unit axis, not a crash that loses the sweep.
    cells = run_e3([kernel()], tmp_path, backends={"broken": "/nonexistent/cc"}, reps=2)
    assert len(cells) == len(OFFLOAD_UNITS)
    assert all(not c.ok and c.error for c in cells)
    assert {c.unit for c in cells} == set(OFFLOAD_UNITS)  # every rung accounted for, none dropped
    assert best_unit_per_backend(cells) == {}


def test_run_e3_records_kernel_ladder_failure_across_the_whole_axis(monkeypatch, tmp_path):
    # the fixed rung is chosen before the per-cell try, so a ladder failure needs its own guard -- and must
    # still emit one cell per (backend, unit) rather than silently shrinking the axis. No compile.
    import nestforge.experiment_e3 as e3

    def boom(*a, **k):
        raise ValueError("cannot canonicalize")

    monkeypatch.setattr(e3, "granularity_ladder", boom)
    cells = run_e3([kernel()], tmp_path, backends={"gcc": "gcc", "clang": "clang"}, reps=2)
    assert len(cells) == 2 * len(OFFLOAD_UNITS)
    assert all(not c.ok and "cannot canonicalize" in c.error for c in cells)


@pytest.mark.integration
def test_run_e3_sweeps_the_offload_axis_bounded(tmp_path):
    # the real thing: one kernel across the unit axis, every valid cell bit-exact against the oracle.
    backends = discover_compilers()
    assert backends, "need gcc/clang on PATH"
    cells = run_e3([kernel()], tmp_path, units=("cfg", "map"), reps=3)
    assert len(cells) == len(backends) * 2
    # 'cfg' selects no nest on this flat kernel -> a recorded skip; 'map' must measure for every backend.
    measured = [c for c in cells if c.ok]
    assert {c.unit for c in measured} == {"map"}
    assert {c.backend for c in measured} == set(backends)
    assert all(c.median_us < float("inf") for c in measured)
    assert set(best_unit_per_backend(cells).values()) == {"map"}
