# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The E1-E5 runner script: argument handling and the JSON it writes. The table-shaping is unit-tested
(no compile); a real bounded run is integration."""
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_experiments import EXPERIMENTS, RunProvenance, main, provenance, to_json_value, write  # noqa: E402

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
    for flag in ("--out", "--experiments", "--kernels", "--reps", "--preset", "--seed", "--only", "--min-fusion-depth"):
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

    Kernels with a real fusion depth and >=2 rungs, so the search surface is not a single point (on a
    1-rung ladder E4's oracle-vs-hillclimb comparison is degenerate: quality and savings are 1.0 by
    construction and a regression in either is invisible).

    Named, not taken from the corpus head: the first six tsvc2 kernels all canonicalize to one
    statement-atom (measured ``fusion_depth`` 0), so ``--kernels 3`` swept three one-rung ladders and
    ``best_granularity`` -- which excludes them by construction -- was asserted non-empty against a table
    that could never hold a row. s1281 (depth 8) and s212 (depth 4) are the axis this asserts on.

    s1281 replaced s221 (depth 5): s221 fails EVERY rung with InvalidSDFGNodeError('Isolated node') -- a
    dace MapFission defect, not ours (it accepts the map, then leaves one of the two `a` AccessNodes at
    degree 0). This test stayed green on it because a slice's failures are invisible once any kernel in it
    measures, so half the named axis was dead and nothing said so.
    """
    rc = main([
        "--out",
        str(tmp_path), "--only", "s1281,s212", "--granularity-points", "3", "--reps", "3", "--experiments",
        "e1,e2,e3,e4,e5"
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
        measured = measured_rows(payload)
        assert measured, f"{name} produced rows but measured NOTHING: {payload['rows'][:2]}"
        # EVERY named kernel, not merely one of them: a slice's failures are invisible under a bare
        # `assert measured`, which is how s221 stayed dead at every rung here with the test green.
        assert {r["kernel"] for r in measured} == {"s1281", "s212"}, \
            f"{name} measured only {sorted({r['kernel'] for r in measured})} of the two named kernels"

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
    #
    # NAMED, not the corpus head, for the reason its sibling above gives -- and now doubly so: the default
    # corpus is hpcagent_bench, whose head is `argmax_value`, a conditional reduction with no map and so no
    # granularity axis at all. `--kernels 1` therefore asserted a measured search side against a kernel that
    # can never have one, and read as an E2 regression rather than a corpus-head change.
    #
    # s212, not the sibling's s221: foundation's s221 fails every rung with
    # InvalidSDFGNodeError('Isolated node'), so it has no valid search cell either. The sibling stays green
    # on it only because s212 measures alongside.
    rc = main(
        ["--out",
         str(tmp_path), "--only", "s212", "--granularity-points", "2", "--reps", "3", "--experiments", "e2"])
    assert rc == 0
    payload = json.loads((tmp_path / "e2.json").read_text())
    assert payload["rows"]
    assert payload["search_cells_from"] == ["e1-fallback"]  # e2 alone must MEASURE a search side, not skip it
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


def test_only_is_not_truncated_by_the_kernel_budget(monkeypatch, tmp_path):
    """--only names the sweep exactly. --kernels defaults to 1, and applying it on top swept just the
    first named key while printing a complete-looking run."""
    seen = []
    import run_experiments as re_mod
    monkeypatch.setattr(re_mod, "discover_compilers", lambda: {"gcc": "/usr/bin/gcc"})
    monkeypatch.setattr(re_mod, "run_e1", lambda kernels, *a, **kw: seen.extend(k.key for k in kernels) or [])
    main(["--out", str(tmp_path), "--only", "s221,s212", "--experiments", "e1"])
    assert seen == ["s212", "s221"]  # corpus order, both kept


def test_unknown_only_key_is_rejected(monkeypatch, tmp_path):
    # iter_tsvc_kernels filters a set, so a typo'd key silently shrinks the sweep instead of failing.
    import run_experiments as re_mod
    monkeypatch.setattr(re_mod, "discover_compilers", lambda: {"gcc": "/usr/bin/gcc"})
    with pytest.raises(SystemExit) as e:
        main(["--out", str(tmp_path), "--only", "s212,nosuchkernel", "--experiments", "e1"])
    assert e.value.code == 2


def test_every_result_file_records_what_produced_it(tmp_path):
    """A corpus sweep is many sbatch jobs, and merging their JSON concatenates rows that otherwise carry no
    record of their conditions. Two rows differing in preset, seed or reps -- or measured on different
    nodes -- are not two measurements of the same thing, so every file must name its own."""
    import argparse

    args = argparse.Namespace(preset="M", seed=3, reps=11, granularity_points=4)
    run = provenance(args, ["gcc", "clang"])
    assert (run.preset, run.seed, run.reps, run.granularity_points) == ("M", 3, 11, 4)
    assert run.backends == ["gcc", "clang"]
    assert run.host and run.cpu and run.dace_version, "host/cpu/dace version are what make two nodes comparable"

    path = write(tmp_path, "e1", [{"kernel": "s221"}], {"best_granularity": {}}, run)
    got = json.loads(path.read_text())
    assert got["run"] == dataclasses.asdict(run)
    assert got["rows"] == [{"kernel": "s221"}]


def test_a_table_cannot_overwrite_the_provenance_block(tmp_path):
    """Tables are splatted into the payload beside "run"/"rows". A table named either would replace the
    provenance with no sign it had, and the merged file would then attribute one job's rows to another
    job's conditions."""
    run = RunProvenance("S", 0, 5, 2, ["gcc"], "host", "cpu", "abc123", "1.0")
    with pytest.raises(ValueError, match="colliding with the reserved keys"):
        write(tmp_path, "e1", [], {"run": {"preset": "XL"}}, run)
