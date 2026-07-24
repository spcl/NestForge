# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""E2 driver (C1/C2/C3): search speedup over the traditional baselines.

The ratio's validity is the experiment, so most of these tests are about what must be held EQUAL across
the two sides -- backend, FP regime, problem size -- and about keeping a search-side gap distinguishable
from a baseline-side one.
"""
import math

import numpy as np
import pytest
import dace

from nestforge.arena import FP_MODES, discover_compilers
from nestforge.experiment_e1 import E1Cell
from nestforge.experiment_e2 import (BASELINE_DRIVER, E2Row, EXTERNAL_WHOLE_PROGRAM_PENDING, NO_FP_CONTRACT,
                                     baseline_optimizer, compare, run_e2, search_best, skipped_lanes, speedup_table)
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


def test_search_best_is_keyed_by_backend_not_min_across_backends():
    """Taking min() across backends and dividing by a single-compiler baseline books part of a COMPILER
    SWAP as a granularity win -- the baseline is built by one toolchain, so the search side must be the
    same one."""
    cells = [
        E1Cell("k", "gcc", "atoms", "map", 10.0, True),
        E1Cell("k", "gcc", "maximal", "map", 6.0, True),
        E1Cell("k", "clang", "atoms", "map", 3.0, True),  # faster, but a DIFFERENT toolchain
    ]
    assert search_best(cells) == {("k", "gcc"): 6.0, ("k", "clang"): 3.0}


def test_search_best_omits_a_pair_whose_every_cell_failed():
    # absent, NOT inf: an inf entry divides into a speedup of 0.0 and reads as a measured loss rather
    # than a pair the search never solved.
    assert search_best([E1Cell("k", "gcc", "atoms", "map", float("inf"), False, "build failed")]) == {}


def test_baseline_is_built_by_the_same_toolchain_and_fp_regime_as_the_search_side():
    """Both halves of the comparability contract, asserted against the arena's own ieee-strict flags:
    the C++ driver matching the backend, and -ffp-contract=off. A baseline left at the driver default
    forms FMAs every offloaded nest is denied, and the difference lands inside the ratio."""
    assert NO_FP_CONTRACT in FP_MODES["ieee-strict"]  # the search side really is pinned this way
    for backend, driver in BASELINE_DRIVER.items():
        opt = baseline_optimizer(backend)
        assert opt.build.compiler == driver, backend
        assert NO_FP_CONTRACT in opt.build.flags, backend
    assert baseline_optimizer("gcc").build.compiler != baseline_optimizer("clang").build.compiler


def test_speedup_is_the_ratio_and_needs_both_sides_valid():
    won = compare("k", "gcc", "whole-program", 12.0, 4.0, True, None, None)
    assert won.ok and won.speedup == 3.0
    lost = compare("k", "gcc", "whole-program", 2.0, 4.0, True, None, None)
    assert lost.ok and lost.speedup == 0.5  # a loss is reported as a loss
    assert math.isnan(compare("k", "gcc", "whole-program", float("inf"), 4.0, False, "bad", None).speedup)
    assert math.isnan(compare("k", "gcc", "whole-program", 12.0, float("inf"), True, None, "none").speedup)


def test_a_search_side_gap_never_marks_the_baseline_lane_as_failed():
    """A baseline that built, validated and timed fine must not be recorded as a lane that could not run
    just because the search produced nothing to divide -- that states the gap on the wrong axis."""
    row = compare("k", "gcc", "whole-program", 12.0, float("inf"), True, None, "no valid search cell")
    assert not row.ok  # no ratio, correctly
    assert row.baseline_us == 12.0  # but the baseline's own measurement survives on the row
    assert "search" in row.error  # and the reason names the search side, not the baseline


def test_skipped_lanes_keeps_every_kernels_reason():
    """A lane-only key lets the last kernel's reason overwrite all the others, so a lane that failed on 1
    of 90 kernels reads identically to one that never ran."""
    rows = [
        E2Row("k1", "gcc", "whole-program", float("inf"), 4.0, float("nan"), False, "validation mismatch"),
        E2Row("k2", "gcc", "whole-program", float("inf"), 4.0, float("nan"), False, "toolchain absent"),
    ]
    skipped = skipped_lanes(rows)
    assert skipped[("k1", "gcc", "whole-program")] == "validation mismatch"
    assert skipped[("k2", "gcc", "whole-program")] == "toolchain absent"


def test_table_is_keyed_by_kernel_and_backend():
    rows = [
        E2Row("k", "gcc", "whole-program", 12.0, 4.0, 3.0, True),
        E2Row("k", "clang", "whole-program", 9.0, 3.0, 3.0, True),
    ]
    assert speedup_table(rows) == {("k", "gcc"): {"whole-program": 3.0}, ("k", "clang"): {"whole-program": 3.0}}


def test_unavailable_lanes_are_rows_with_reasons_for_every_backend(tmp_path, monkeypatch):
    import nestforge.experiment_e2 as e2

    def boom(*a, **k):
        raise RuntimeError("toolchain absent")

    monkeypatch.setattr(e2, "measure_whole_program", boom)
    cells = [E1Cell("two_map", "gcc", "atoms", "map", 5.0, True)]
    rows = run_e2([kernel()], cells, tmp_path, backends={"gcc": "gcc", "clang": "clang"})
    for backend in ("gcc", "clang"):
        lanes = {r.baseline for r in rows if r.backend == backend}
        assert lanes == {"whole-program", "pluto", *EXTERNAL_WHOLE_PROGRAM_PENDING}
    assert all(not r.ok and r.error for r in rows)
    assert speedup_table(rows) == {}


@pytest.mark.integration
def test_run_e2_measures_a_real_baseline_per_backend(tmp_path):
    """The real baseline: build + validate bit-exact + time the whole program, per backend, then ratio it.

    Asserts a MEASURED row -- ok, finite baseline_us, and the ratio equal to baseline/search -- rather
    than merely that rows exist. Every lane emits a row even when nothing built, so row-count assertions
    cannot tell a working pipeline from a dead one.
    """
    backends = discover_compilers()
    assert backends, "need gcc/clang on PATH"
    one = dict([next(iter(backends.items()))])
    name = next(iter(one))
    cells = [E1Cell("two_map", name, "atoms", "map", 5.0, True)]

    rows = run_e2([kernel()], cells, tmp_path, reps=3, backends=one)
    wp = next(r for r in rows if r.baseline == "whole-program")
    assert wp.error is None, wp.error
    assert wp.ok and 0.0 < wp.baseline_us < float("inf")
    assert wp.backend == name
    assert wp.speedup == pytest.approx(wp.baseline_us / 5.0)  # numerator/denominator not swapped
    assert set(r.baseline for r in rows if not r.ok) >= {"pluto", *EXTERNAL_WHOLE_PROGRAM_PENDING}
