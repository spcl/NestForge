"""E4 driver (C4): quality vs cost, exhaustive oracle against the scoped agent. The scoring/read-off is a
unit test (synthetic results, a fake measure -- no compile); the real measured run is integration."""
import numpy as np
import pytest
import dace

from nestforge.arena import discover_compilers
from nestforge.experiment_e4 import (E4Row, ORACLE, cost_quality_table, run_e4, savings_vs_oracle, score)
from nestforge.policy import SearchResult, exhaustive_search, hillclimb_search
from nestforge.sweep import MeasureLedger
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


def test_quality_is_relative_to_the_oracle_and_cost_is_the_ledger():
    res = SearchResult("mid", 5.0, MeasureLedger(measurements=2, tokens=17))
    matched = score("k", "gcc", "hillclimb", res, oracle_us=5.0)
    assert matched.ok and matched.quality == 1.0  # found the optimum
    assert matched.measurements == 2 and matched.tokens == 17  # cost reported alongside, never omitted
    settled = score("k", "gcc", "hillclimb", res, oracle_us=2.5)
    assert settled.quality == 0.5  # settled for a rung twice as slow as the optimum


def test_quality_is_zero_when_there_was_no_optimum_to_be_near():
    # every rung failed to build -> inf. Reporting 1.0 would claim the strategy matched an oracle that
    # itself found nothing.
    dead = SearchResult("x", float("inf"), MeasureLedger(measurements=3))
    row = score("k", "gcc", "hillclimb", dead, oracle_us=float("inf"))
    assert not row.ok and row.quality == 0.0 and row.error


def test_savings_compares_only_runs_the_oracle_also_completed():
    rows = [
        E4Row("k", "gcc", ORACLE, "a", 4.0, 1.0, 10, 0, True),
        E4Row("k", "gcc", "hillclimb", "a", 4.0, 1.0, 4, 0, True),  # same answer, 40% of the cost
        E4Row("k2", "gcc", "hillclimb", "a", 4.0, 1.0, 2, 0, True),  # no oracle row for k2 -> not counted
    ]
    savings = savings_vs_oracle(rows)
    assert savings[ORACLE] == 1.0  # the oracle costs itself, by definition
    assert savings["hillclimb"] == pytest.approx(0.4)


def test_table_reports_quality_and_cost_together():
    rows = [
        E4Row("k", "gcc", ORACLE, "a", 4.0, 1.0, 10, 0, True),
        E4Row("k", "gcc", "hillclimb", "b", 5.0, 0.8, 3, 0, True),
        E4Row("k", "gcc", "hillclimb", "-", float("inf"), 0.0, 1, 0, False, "no valid rung measured"),
    ]
    table = cost_quality_table(rows)
    assert table["hillclimb"]["quality"] == 0.8 and table["hillclimb"]["measurements"] == 3.0
    assert table["hillclimb"]["runs"] == 1.0  # the failed row is excluded, not averaged in as a zero
    assert table[ORACLE]["quality"] == 1.0


def test_both_strategies_drive_the_same_surface_with_different_cost():
    # the C4 shape, against a fake measure so it needs no compiler: a ladder with a single minimum. The
    # oracle pays for every rung; the hillclimb walks to the same answer for less.
    labels = ["r0", "r1", "r2", "r3", "r4"]
    times = {"r0": 10.0, "r1": 8.0, "r2": 5.0, "r3": 7.0, "r4": 9.0}
    ex = exhaustive_search(labels, times.__getitem__, MeasureLedger())
    hc = hillclimb_search(labels, times.__getitem__, MeasureLedger())
    assert ex.best == hc.best == "r2"  # same surface, same optimum
    assert ex.ledger.measurements == len(labels)
    assert hc.ledger.measurements < ex.ledger.measurements  # near-oracle quality, fewer measurements


def test_run_e4_records_failures_for_every_strategy_without_crashing(tmp_path, monkeypatch):
    # a kernel whose ladder cannot be built must yield one recorded row per (backend, strategy), not a
    # crash and not a silently shorter table. No compile.
    import nestforge.experiment_e4 as e4

    def boom(*a, **k):
        raise ValueError("cannot canonicalize")

    monkeypatch.setattr(e4, "granularity_ladder", boom)
    rows = run_e4([kernel()], tmp_path, backends={"gcc": "gcc", "clang": "clang"})
    assert len(rows) == 2 * len(e4.STRATEGIES)
    assert all(not r.ok and "cannot canonicalize" in r.error for r in rows)
    assert cost_quality_table(rows) == {}


@pytest.mark.integration
def test_run_e4_measures_both_strategies_on_one_kernel(tmp_path):
    # the real thing: both strategies over a real granularity ladder, real builds, real times.
    backends = discover_compilers()
    assert backends, "need gcc/clang on PATH"
    one = {next(iter(backends)): backends[next(iter(backends))]}
    rows = run_e4([kernel()], tmp_path, max_granularity_points=3, reps=3, backends=one)
    assert {r.strategy for r in rows} == set(("exhaustive", "hillclimb"))
    assert all(r.error is None for r in rows), [r.error for r in rows if r.error]
    oracle = next(r for r in rows if r.strategy == ORACLE)
    assert oracle.ok and oracle.quality == 1.0  # the oracle matches itself
    for r in rows:
        assert r.measurements > 0  # a strategy that measured nothing cannot have found anything
        assert r.quality <= 1.0 + 1e-9  # nothing beats the exhaustive optimum
