# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The E1-E5 runner script: argument handling and the JSON it writes. The table-shaping is unit-tested
(no compile); a real bounded run is integration."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_experiments import EXPERIMENTS, main, to_json_value  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_experiments.py"


def test_tuple_keys_survive_serialization():
    # tables are keyed by (kernel, backend); str(tuple) would render "('k', 'gcc')" and a naive flatten
    # would drop the backend half, silently merging every backend's row into one.
    table = {("k", "gcc"): {"quality": 1.0}, ("k", "clang"): {"quality": 0.5}}
    out = to_json_value(table)
    assert out == {"k | gcc": {"quality": 1.0}, "k | clang": {"quality": 0.5}}
    assert json.dumps(out)  # actually serializable, not merely reshaped


def test_non_finite_numbers_stay_visible():
    # inf/nan are not JSON. Coercing them to null would make a failed measurement indistinguishable from
    # one that was never attempted.
    out = to_json_value({"a": float("inf"), "b": float("nan"), "c": 1.5})
    assert out["a"] == "inf" and out["b"] == "nan" and out["c"] == 1.5
    assert json.dumps(out)


def test_dataclass_rows_serialize():
    from nestforge.experiment_e1 import E1Cell
    out = to_json_value([E1Cell("k", "gcc", "atoms", "map", 4.0, True)])
    assert out[0]["kernel"] == "k" and out[0]["median_us"] == 4.0
    assert json.dumps(out)


def test_unknown_experiment_is_rejected_not_silently_skipped(tmp_path):
    # a typo'd --experiments must fail loudly; silently running nothing would look like a clean run that
    # produced no results.
    with pytest.raises(SystemExit) as e:
        main(["--out", str(tmp_path), "--experiments", "e1,e9"])
    assert e.value.code != 0


def test_script_is_runnable_and_documents_its_flags():
    # the script is the cluster entry point, so --help must work from a plain checkout (no pytest path
    # tricks): a broken import here surfaces only when someone submits a job.
    proc = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    for flag in ("--out", "--experiments", "--kernels", "--reps", "--preset", "--seed"):
        assert flag in proc.stdout
    for name in EXPERIMENTS:
        assert name in proc.stdout


def measured_rows(payload):
    """Rows that actually MEASURED something. Every driver records failures as rows, so a non-empty `rows`
    list proves only that the sweep ran, not that anything built, validated or timed."""
    return [r for r in payload["rows"] if r.get("ok") and str(r.get("median_us", r.get("baseline_us", ""))) != "inf"]


@pytest.mark.integration
def test_bounded_run_produces_real_measurements_not_just_rows(tmp_path):
    """The pipeline proof. Asserts SUCCESSFUL measurements, because the previous version of this test --
    `rc == 0` and non-empty rows -- stayed green with the whole measurement path dead: main() returns 0
    unconditionally once a compiler is found, and every driver emits failure rows.

    A kernel with a real fusion depth and >=2 rungs, so the search surface is not a single point (on a
    1-rung ladder E4's oracle-vs-hillclimb comparison is degenerate: quality and savings are 1.0 by
    construction and a regression in either is invisible).
    """
    rc = main([
        "--out",
        str(tmp_path), "--kernels", "3", "--granularity-points", "3", "--reps", "3", "--experiments", "e1,e2,e3,e4,e5"
    ])
    assert rc == 0
    for name in EXPERIMENTS:
        path = tmp_path / f"{name}.json"
        assert path.exists(), f"{name} wrote no table"
        payload = json.loads(path.read_text())
        assert payload["rows"], f"{name} produced no rows"
        if name == "e5":
            # E5 studies NON-affine kernels, so on an all-affine slice (the first TSVC kernels are affine)
            # it correctly measures nothing and excludes them. A DEAD E5 is still caught: a crash or a
            # failed build is a row whose error is NOT the classifier's "excluded ..." verdict.
            not_excluded = [r for r in payload["rows"] if not r.get("ok") and "excluded" not in (r.get("error") or "")]
            assert not not_excluded, f"e5 rows failed rather than excluded: {not_excluded[:2]}"
            continue
        assert measured_rows(payload), f"{name} produced rows but measured NOTHING: {payload['rows'][:2]}"

    e1 = json.loads((tmp_path / "e1.json").read_text())
    assert e1["best_granularity"], "E1 measured cells but named no winner"
    e2 = json.loads((tmp_path / "e2.json").read_text())
    # provenance: which axes fed the search side, so a rerun with a different --experiments is explicable
    assert e2["search_cells_from"], "E2 did not record what fed its search side"
    assert e2["speedup"], "E2 wrote no speedup -- the baseline or the search side produced nothing"
    for lanes in e2["speedup"].values():
        assert all(v > 0 for v in lanes.values())


@pytest.mark.integration
def test_e2_alone_still_measures_a_search_side(tmp_path):
    # E2 divides a search time by a baseline; asking for it alone must still sweep the search axis rather
    # than emitting a table of bare baselines.
    rc = main(
        ["--out",
         str(tmp_path), "--kernels", "1", "--granularity-points", "2", "--reps", "3", "--experiments", "e2"])
    assert rc == 0
    payload = json.loads((tmp_path / "e2.json").read_text())
    assert payload["rows"]
    assert any(r["search_us"] not in ("inf", None) for r in payload["rows"])


def test_entry_point_forwards_argv_without_the_program_name():
    """`sys.exit(main(sys.argv[1:]))` -- the slice matters: passing sys.argv whole makes argparse choke on
    the script path with "unrecognized arguments" on every real cluster invocation.

    The previous test ran runpy with PYTEST's argv and asserted only SystemExit, which argparse raises for
    ANY bad flag -- so it passed against a deliberately broken entry point too. Assert the exit CODE for a
    known-good and a known-bad argv instead.
    """
    with pytest.raises(SystemExit) as bad:
        main(["--out", "/tmp/x", "--experiments", "e9"])  # argparse rejects the unknown experiment
    assert bad.value.code == 2
    src = SCRIPT.read_text()
    assert "sys.exit(main(sys.argv[1:]))" in src, "entry point must forward argv WITHOUT the program name"
