# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The E1-E5 tables, checked as data.

``tests/data/experiments/`` holds the real tables the ``experiments`` workflow produced (run 29819716644).
The drivers record failures AS ROWS, so a table can be complete, well-formed and describe a pipeline that
measured nothing -- these tests read the tables the way the paper does and fail when they stop meaning
what the paper claims.

:func:`check_table` is applied both to those recorded tables (unit) and to a fresh bounded run
(integration), so a live run is held to the same contract.
"""
import json
from pathlib import Path

import pytest

TABLES = Path(__file__).resolve().parent / "data" / "experiments"

#: error substrings from bugs that are fixed. A live run producing one again is a regression, not a
#: kernel that happens to be unsupported.
FIXED_ERROR_CLASSES = ("is not Python (Language.CPP)", "StopIteration")

#: Kernels the recording still fails on, with the OPEN bug each one hits. Named so the assertions
#: below stay strict everywhere else; every entry is a defect to fix, not an accepted outcome.
#:   s1119 -- RecursionError in the granularity ladder (was StopIteration before dace 347ecb2a9)
OPEN_IN_RECORDING = {"s1119": "RecursionError"}

#: per table: the fields every row carries, and the derived aggregates the paper reads.
SCHEMA = {
    "e1": (("kernel", "backend", "granularity", "unit", "median_us", "ok"), ("best_granularity", )),
    "e2": (("kernel", "backend", "baseline", "baseline_us", "search_us", "speedup", "ok"), ("speedup", "skipped",
                                                                                            "search_cells_from")),
    "e3": (("kernel", "backend", "granularity", "unit", "median_us", "ok"), ("curve", "best_unit")),
    "e4": (("kernel", "backend", "strategy", "best", "best_us", "measurements", "tokens", "quality", "ok"),
           ("cost_quality", "savings")),
    "e5": (("kernel", "backend", "schedulable", "reason", "best", "best_us", "coarsest", "coarsest_us", "ok"),
           ("non_affine_speedup", )),
}


def recorded(name):
    return json.loads((TABLES / f"{name}.json").read_text())


def key_of(row):
    return f"{row['kernel']} | {row['backend']}"


def check_table(name, payload):
    """Structural + internal-consistency contract for one table. Shared by the recorded and live checks."""
    fields, aggregates = SCHEMA[name]
    assert payload["rows"], f"{name}: no rows at all"
    for aggregate in aggregates:
        assert aggregate in payload, f"{name}: missing aggregate {aggregate!r}"
    for row in payload["rows"]:
        missing = [f for f in fields if f not in row]
        assert not missing, f"{name}: row {row} missing {missing}"
        # ok and error are exclusive: a row may not claim success and carry a failure, nor fail silently.
        if row["ok"]:
            assert not row.get("error"), f"{name}: ok row carries an error: {row}"
        else:
            assert row.get("error"), f"{name}: failed row records no reason: {row}"


def check_fastest_rung_wins(payload, aggregate, axis):
    """The named winner must be the fastest MEASURED rung -- the direction is the whole claim, and an
    inverted comparison still produces a full, plausible table."""
    per_key = {}
    for row in payload["rows"]:
        if row["ok"]:
            per_key.setdefault(key_of(row), {})[row[axis]] = row["median_us"]
    assert per_key, "no measured rows to rank"
    assert set(payload[aggregate]) == set(per_key), "winner set does not match the measured cells"
    for key, timings in per_key.items():
        assert payload[aggregate][key] == min(timings, key=timings.get), f"{key}: {payload[aggregate][key]} named "\
                                                                         f"but {timings} measured"


@pytest.mark.parametrize("name", sorted(SCHEMA))
def test_recorded_table_matches_the_contract(name):
    check_table(name, recorded(name))


def test_the_open_list_matches_what_the_recording_actually_hit():
    """OPEN_IN_RECORDING must not become a dumping ground: each entry names a kernel that really
    fails, with the error it really fails on, so a fixed one has to be removed from the list."""
    for kernel, expected in OPEN_IN_RECORDING.items():
        errors = [r.get("error") or "" for n in SCHEMA for r in recorded(n)["rows"] if r["kernel"] == kernel]
        assert any(expected in e for e in errors), f"{kernel} no longer fails with {expected!r} -- drop it"


@pytest.mark.parametrize("name", ["e1", "e2", "e3", "e4"])
def test_recorded_table_measured_something(name):
    """E1-E4 must contain successful measurements. E5 is excluded: on an affine slice it correctly
    measures nothing (see test_e5_measures_nothing_only_by_exclusion)."""
    assert [r for r in recorded(name)["rows"] if r["ok"]], f"{name}: every row failed"


def test_e1_names_the_fastest_granularity():
    check_fastest_rung_wins(recorded("e1"), "best_granularity", "granularity")


def test_e3_names_the_fastest_offload_unit():
    check_fastest_rung_wins(recorded("e3"), "best_unit", "unit")


def test_e2_speedup_is_the_ratio_it_reports():
    """C1's headline number. A speedup that does not equal baseline/search means the table's rows and its
    summary disagree, and only the summary reaches the paper."""
    payload = recorded("e2")
    checked = 0
    for row in payload["rows"]:
        if not row["ok"]:
            continue
        reported = payload["speedup"][key_of(row)][row["baseline"]]
        assert reported == pytest.approx(row["baseline_us"] / row["search_us"]), row
        checked += 1
    assert checked, "no successful E2 rows to check"


def test_e2_records_a_reason_for_every_skipped_lane():
    payload = recorded("e2")
    for row in payload["rows"]:
        if not row["ok"]:
            assert f"{key_of(row)} | {row['baseline']}" in payload["skipped"], f"unexplained lane: {row}"


def test_e4_compares_both_strategies_over_the_same_runs():
    """The cost ledger is only comparable if every strategy is averaged over the same set of runs."""
    payload = recorded("e4")
    assert set(payload["cost_quality"]) == {r["strategy"] for r in payload["rows"]}
    runs = {s: v["runs"] for s, v in payload["cost_quality"].items()}
    assert len(set(runs.values())) == 1, f"strategies averaged over different run counts: {runs}"
    for strategy, stats in payload["cost_quality"].items():
        assert 0 < stats["quality"] <= 1.0, f"{strategy}: quality {stats['quality']} is not a fraction of oracle"


def test_e5_measures_nothing_only_by_exclusion():
    """E5 studies non-affine kernels; on the affine CI slice it must exclude them, not fail on them. A
    crash or a build failure here is a real defect wearing E5's empty-result costume."""
    payload = recorded("e5")
    broken = [
        r for r in payload["rows"]
        if not r["ok"] and "excluded" not in r["error"] and r["kernel"] not in OPEN_IN_RECORDING
    ]
    assert not broken, f"e5 rows failed rather than excluded: {broken[:2]}"
    for key in payload["non_affine_speedup"]:
        assert any(r["ok"] and key_of(r) == key for r in payload["rows"]), f"{key}: speedup without a measured row"


@pytest.mark.integration
def test_a_live_run_obeys_the_same_contract_and_hits_no_fixed_bug(tmp_path):
    """Bounded real run: same contract as the recorded tables, plus no error class we have already fixed."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_experiments import main

    assert main(["--out", str(tmp_path), "--kernels", "3", "--granularity-points", "3", "--reps", "3"]) == 0
    for name in SCHEMA:
        payload = json.loads((tmp_path / f"{name}.json").read_text())
        check_table(name, payload)
        for row in payload["rows"]:
            regressed = [c for c in FIXED_ERROR_CLASSES if c in (row.get("error") or "")]
            assert not regressed, f"{name}: {regressed} came back on {key_of(row)}"
