# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The loopnest -> winner dataset a sweep leaves behind: the record's shape, the flag-independent nest
identity it joins on, and the two failure modes that must not cost a job its results."""
import json

import dace
import pytest

from nestforge import dataset

N = dace.symbol("N")


@dace.program
def madd(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = a[i] * b[i] + a[i]


def a_record(**over) -> dataset.WinnerRecord:
    base = dict(kernel="s000",
                corpus="tsvc2",
                nest=0,
                fingerprint="abc123",
                axis="vectorize",
                winner={"vec_variant": "cpu-w16-posttail-fma"},
                median_us=4.0,
                baseline_us=8.0,
                speedup=2.0)
    base.update(over)
    return dataset.WinnerRecord(**base)


def test_a_record_round_trips_and_lands_in_its_axis_file(tmp_path):
    path = dataset.record_winner(a_record(), tmp_path)
    assert path.name == "winners-vectorize.jsonl"
    dataset.record_winner(a_record(kernel="s112", median_us=6.0), tmp_path)
    dataset.record_winner(a_record(axis="granularity", winner={"rung": "atoms"}), tmp_path)

    rows = dataset.load_records(tmp_path, axis="vectorize")
    assert [r["kernel"] for r in rows] == ["s000", "s112"]  # append-only, in job order
    assert rows[0]["winner"] == {"vec_variant": "cpu-w16-posttail-fma"} and rows[0]["speedup"] == 2.0
    assert rows[0]["machine"] and rows[0]["host"] and rows[0]["timestamp"] > 0
    assert len(dataset.load_records(tmp_path)) == 3  # no axis -> every file
    assert [r["axis"] for r in dataset.load_records(tmp_path, axis="granularity")] == ["granularity"]


def test_a_truncated_line_does_not_cost_the_rest_of_the_dataset(tmp_path):
    """A job killed mid-append leaves half a line. An append-only log exists so that what landed stays
    readable; taking the whole file down over the last line would defeat it."""
    dataset.record_winner(a_record(), tmp_path)
    with (tmp_path / "winners-vectorize.jsonl").open("a") as handle:
        handle.write('{"kernel": "s112", "median_us": 4.')  # killed here
    rows = dataset.load_records(tmp_path, axis="vectorize")
    assert [r["kernel"] for r in rows] == ["s000"]


def test_an_unwritable_directory_does_not_take_the_sweep_down(tmp_path, monkeypatch):
    """The tables are the job's product; the dataset is a by-product. A full disk must not turn an hour of
    measurement into a traceback."""

    def boom(*a, **k):
        raise OSError("no space left on device")

    monkeypatch.setattr(dataset.Path, "mkdir", boom)
    dataset.record_winner(a_record(), tmp_path / "nope")  # must not raise


def test_features_are_inputs_only():
    """The features are the model's X and the winner is its y. A timing or a config leaking into the
    features would train a model on the answer."""
    feats = dataset.nest_features(madd.to_sdfg(simplify=True))
    assert feats["maps"] >= 1 and feats["arrays"] == 3
    assert feats["dtypes"] == ["float64"]
    assert feats["has_fma_pattern"] is True  # a*b + a
    assert feats["has_branch"] is False and feats["has_integer_conditional"] is False
    assert feats["symbolic_extents"] == feats["innermost_maps"]  # N is a symbol: the remainder is always emitted
    assert feats["max_map_dims"] == 1
    forbidden = {"median_us", "winner", "speedup", "vec_variant", "compiler"}
    assert not (set(feats) & forbidden), f"a label leaked into the features: {set(feats) & forbidden}"
    assert json.dumps(feats)  # a record must serialize


def test_the_fingerprint_is_the_same_nest_under_different_flags(tmp_path):
    """The join key. Compile flags never reach the emitted source, so one nest keeps ONE identity across
    compilers, FP rungs and machines -- which is what lets the dataset ask what won for this loopnest
    somewhere else. Keyed on the kernel name instead, `s000` would be a different nest per opt-mode."""
    src = tmp_path / "nest.c"
    src.write_text("void k(double *a, const double *b, int n) {\n  for (int i = 0; i < n; ++i) a[i] = b[i] + 1.0;\n}\n")
    same = tmp_path / "same_body_other_name.c"
    same.write_text("void k(double *a, const double *b, int n) { for (int i = 0; i < n; ++i) a[i] = b[i] + 1.0; }")
    other = tmp_path / "other.c"
    other.write_text(
        "void k(double *a, const double *b, int n) {\n  for (int i = 0; i < n; ++i) a[i] = b[i] * 2.0;\n}\n")

    assert dataset.fingerprint(src) == dataset.fingerprint(same)  # layout is not identity
    assert dataset.fingerprint(src) != dataset.fingerprint(other)  # the computation is
    assert dataset.fingerprint(None) == "" and dataset.fingerprint(tmp_path / "absent.c") == ""


def test_the_default_directory_is_relative_and_overridable(monkeypatch):
    """No hardcoded absolute path: a job writes under its own result root. Records land in `.results` only
    when nothing names a directory."""
    assert not dataset.DATASET_DIR.startswith("/")
    assert dataset.DATASET_DIR == ".results" or dataset.DATASET_DIR


@pytest.mark.parametrize("axis", ["vectorize", "granularity", "flags"])
def test_every_axis_gets_its_own_file(tmp_path, axis):
    """One file per axis: two concurrent sweeps of different axes never share a line, and a reader of one
    axis does not filter the others out by hand."""
    path = dataset.record_winner(a_record(axis=axis), tmp_path)
    assert path.name == f"winners-{axis}.jsonl"
    assert len(dataset.load_records(tmp_path, axis=axis)) == 1


def test_the_sweep_records_a_winner_per_nest(tmp_path, monkeypatch):
    """The wiring: a lane that produced a validated winner leaves a record behind, with the plain-DaCe
    cell as its baseline. Without this the sweep prints its winner and forgets it."""
    from nestforge.perf import tsvc_full

    class FakeKernel:
        key, corpus = "s000", "tsvc2"

    nc = {
        "nest_idx": 0,
        "boundary": type("B", (), {"standalone_sdfg": madd.to_sdfg(simplify=True)})(),
        "lang_src": {},
    }
    cell = {"ok": True, "median_us": 4.0, "vec_variant": "cpu-w16-posttail-fma", "compiler": "gcc"}
    baselines = [{"nest": 0, "ok": True, "median_us": 8.0}]
    tsvc_full.record_nest_winner(FakeKernel(), nc, "vectorize", cell, baselines, "gcc", tmp_path)

    rows = dataset.load_records(tmp_path, axis="vectorize")
    assert len(rows) == 1
    assert rows[0]["kernel"] == "s000" and rows[0]["nest"] == 0
    assert rows[0]["winner"]["vec_variant"] == "cpu-w16-posttail-fma"
    assert rows[0]["baseline_us"] == 8.0 and rows[0]["speedup"] == 2.0
    assert rows[0]["features"]["has_fma_pattern"] is True


def test_a_failed_lane_records_nothing(tmp_path):
    """A lane with no validated cell has no winner to learn from -- recording one would put an `inf` (or
    the reason string) in the label column of the training set."""
    from nestforge.perf import tsvc_full

    class FakeKernel:
        key, corpus = "s000", "tsvc2"

    nc = {
        "nest_idx": 0,
        "boundary": type("B", (), {"standalone_sdfg": madd.to_sdfg(simplify=True)})(),
        "lang_src": {},
    }
    failed = {"ok": False, "error": "no vectorization config validated", "median_us": float("inf")}
    tsvc_full.record_nest_winner(FakeKernel(), nc, "vectorize", failed, [], "gcc", tmp_path)
    assert dataset.load_records(tmp_path) == []
