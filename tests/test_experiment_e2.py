"""E2 driver (C1/C2/C3): search speedup over the traditional baselines. The ratio/read-off logic is a unit
test (synthetic rows, no compile); the whole-program baseline measurement is an integration test."""
import math

import numpy as np
import pytest
import dace

from nestforge.experiment_e1 import E1Cell
from nestforge.experiment_e2 import (E2Row, EXTERNAL_WHOLE_PROGRAM_PENDING, compare, run_e2, search_best, skipped_lanes,
                                     speedup_table)
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


def test_search_best_takes_the_argmin_across_every_axis():
    # the search's answer for a kernel is its fastest valid cell, whatever backend/granularity/unit got it.
    cells = [
        E1Cell("k", "gcc", "atoms", "map", 10.0, True),
        E1Cell("k", "clang", "maximal", "cfg", 4.0, True),  # winner, from a different backend AND unit
        E1Cell("k", "gcc", "maximal", "map", float("inf"), False, "build failed"),
    ]
    assert search_best(cells) == {"k": 4.0}


def test_search_best_omits_a_kernel_whose_every_cell_failed():
    # absent, NOT present at inf -- an inf entry would divide into a finite speedup of 0.0 and read as a
    # measured loss rather than a kernel the search never solved.
    cells = [E1Cell("k", "gcc", "atoms", "map", float("inf"), False, "build failed")]
    assert search_best(cells) == {}


def test_speedup_is_the_ratio_and_needs_both_sides_valid():
    won = compare("k", "whole-program", 12.0, 4.0, True, None)
    assert won.ok and won.speedup == 3.0  # search 3x faster than the baseline
    lost = compare("k", "whole-program", 2.0, 4.0, True, None)
    assert lost.ok and lost.speedup == 0.5  # a loss is reported as a loss, not hidden
    # a baseline that failed validation, and a kernel the search never solved, both yield nan -- never a
    # fabricated ratio from an inf operand.
    assert math.isnan(compare("k", "whole-program", float("inf"), 4.0, False, "did not validate").speedup)
    assert math.isnan(compare("k", "whole-program", 12.0, float("inf"), True, None).speedup)
    assert not compare("k", "whole-program", 12.0, float("inf"), True, None).ok


def test_table_keeps_valid_rows_and_reports_why_lanes_skipped():
    rows = [
        E2Row("k", "whole-program", 12.0, 4.0, 3.0, True),
        E2Row("k", "pluto", float("inf"), 4.0, float("nan"), False, "pluto unavailable: 'polycc' not on PATH"),
    ]
    assert speedup_table(rows) == {"k": {"whole-program": 3.0}}
    assert "polycc" in skipped_lanes(rows)["pluto"]  # the gap is stated, not silently a missing column


def test_unavailable_lanes_are_rows_with_reasons_not_absent_rows(tmp_path, monkeypatch):
    # a baseline set that shrinks to whatever is installed flatters the search by omission, so every lane
    # must appear -- including the ones that cannot run here. No compile: the whole-program build is stubbed.
    import nestforge.experiment_e2 as e2

    def boom(*a, **k):
        raise RuntimeError("toolchain absent")

    monkeypatch.setattr(e2, "measure_whole_program", boom)
    cells = [E1Cell("two_map", "gcc", "atoms", "map", 5.0, True)]
    rows = run_e2([kernel()], cells, tmp_path)
    lanes = {r.baseline for r in rows}
    assert lanes == {"whole-program", "pluto", *EXTERNAL_WHOLE_PROGRAM_PENDING}
    assert all(not r.ok and r.error for r in rows)  # every lane recorded, none dropped, none crashed
    assert speedup_table(rows) == {}


@pytest.mark.integration
def test_run_e2_measures_the_whole_program_baseline(tmp_path):
    # the real baseline: build + validate + time the whole program under DaCe auto-opt, then ratio it
    # against a search time. Only the whole-program lane can run on the runner; the rest are recorded skips.
    cells = [E1Cell("two_map", "gcc", "atoms", "map", 5.0, True)]
    rows = run_e2([kernel()], cells, tmp_path, reps=3)
    wp = next(r for r in rows if r.baseline == "whole-program")
    assert wp.error is None, wp.error
    assert wp.ok and wp.baseline_us < float("inf")
    assert wp.speedup == pytest.approx(wp.baseline_us / 5.0)
    assert set(skipped_lanes(rows)) >= {"pluto", *EXTERNAL_WHOLE_PROGRAM_PENDING}
