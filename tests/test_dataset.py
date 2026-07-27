# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The loopnest -> winner dataset a sweep leaves behind: the record's shape, the flag-independent nest
identity it joins on, and the two failure modes that must not cost a job its results."""
import json
from pathlib import Path

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


NEST_SRC = """def kernel(a, b, c, N):
    # a comment, and some deliberate blank lines below

    for i in range(N):
        c[i] = a[i] + 1.0
"""
NEST_REFORMATTED = "def kernel(a, b, c, N):\n    for i in range(N):\n        c[i] = a[i] + 1.0\n"
NEST_OTHER = "def kernel(a, b, c, N):\n    for i in range(N):\n        c[i] = a[i] * 2.0\n"


def test_the_fingerprint_is_the_numpy_nest_normalized_through_the_ast(tmp_path):
    """The join key, and why it is the NUMPY source: every emitted C/C++/Fortran variant is a translation
    of that one rendering, so it is the only form upstream of the language axis and independent of
    compiler, flags and machine. Keyed on the emitted C, a Fortran-only run recorded no identity at all.

    Through the AST so a comment or a blank line is not a different loopnest -- the dataset exists to join
    the same nest, met again in another job, to what won for it last time."""
    assert dataset.fingerprint(NEST_SRC) == dataset.fingerprint(NEST_REFORMATTED)
    assert dataset.fingerprint(NEST_SRC) != dataset.fingerprint(NEST_OTHER)  # the computation IS identity
    assert dataset.fingerprint(None) == "" and dataset.fingerprint("") == ""
    # unparseable source still keys (over-splitting costs a join; over-merging corrupts the labels)
    assert dataset.fingerprint("def kernel(:::") != ""


def test_the_nest_source_is_stored_once_and_reloadable(tmp_path):
    """Content-addressed: a nest a hundred jobs measured is ONE file, and a recorded winner stays
    reproducible -- the exact input the translator consumed sits next to it."""
    key = dataset.store_nest(NEST_SRC, tmp_path)
    assert key and (tmp_path / "nests" / f"{key}.py").is_file()
    assert dataset.store_nest(NEST_REFORMATTED, tmp_path) == key  # same nest -> same address
    assert len(list((tmp_path / "nests").glob("*.py"))) == 1
    assert dataset.load_nest(key, tmp_path) == NEST_SRC  # the first writer's bytes
    assert dataset.load_nest("never-seen", tmp_path) is None
    assert dataset.store_nest("", tmp_path) == ""


def test_the_default_directory_is_relative_and_overridable(monkeypatch, tmp_path):
    """No hardcoded absolute path: a job writes under its own result root. The previous version of this
    test asserted `DATASET_DIR == ".results" or DATASET_DIR` -- true for every non-empty string, so it
    could not fail -- and could not exercise NF_DATASET_DIR at all, since the value was read at import."""
    monkeypatch.delenv("NF_DATASET_DIR", raising=False)
    assert dataset.dataset_dir() == Path(".results")
    assert not dataset.DEFAULT_DIR.startswith("/")

    monkeypatch.setenv("NF_DATASET_DIR", str(tmp_path / "elsewhere"))
    assert dataset.dataset_dir() == tmp_path / "elsewhere"  # resolved per call, so the env var wins
    assert dataset.dataset_dir(tmp_path / "explicit") == tmp_path / "explicit"  # an argument beats it

    # and the override is REACHED: a record with no out_dir lands under the env directory
    dataset.record_winner(a_record())
    assert (tmp_path / "elsewhere" / "winners-vectorize.jsonl").is_file()


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
        "numpy_source": NEST_SRC,
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
    # the record points at the nest it was measured on, and that nest is on disk
    assert dataset.load_nest(rows[0]["fingerprint"], tmp_path) == NEST_SRC


def test_a_failed_lane_records_nothing(tmp_path):
    """A lane with no validated cell has no winner to learn from -- recording one would put an `inf` (or
    the reason string) in the label column of the training set."""
    from nestforge.perf import tsvc_full

    class FakeKernel:
        key, corpus = "s000", "tsvc2"

    nc = {
        "nest_idx": 0,
        "boundary": type("B", (), {"standalone_sdfg": madd.to_sdfg(simplify=True)})(),
        "numpy_source": NEST_SRC,
    }
    failed = {"ok": False, "error": "no vectorization config validated", "median_us": float("inf")}
    tsvc_full.record_nest_winner(FakeKernel(), nc, "vectorize", failed, [], "gcc", tmp_path)
    assert dataset.load_records(tmp_path) == []


def test_a_strided_extent_is_not_booked_as_symbolic():
    """`(end - begin + 1) / step` renders a step-2 range as "3.5", which fails isdigit() and books a fully
    CONSTANT extent as symbolic -- corrupting the one feature whose meaning is "the remainder is always
    emitted". int_ceil keeps it an integer, and the recorded trip count is the real one."""
    sdfg = dace.SDFG("strided")
    state = sdfg.add_state()
    sdfg.add_array("A", [8], dace.float64)
    entry, exit_ = state.add_map("m", {"i": dace.subsets.Range([(0, 6, 2)])})
    tasklet = state.add_tasklet("t", {}, {"o"}, "o = 1.0")
    src = state.add_access("A")
    state.add_edge(entry, None, tasklet, None, dace.Memlet())
    state.add_edge(tasklet, "o", exit_, "IN_A", dace.Memlet("A[i]"))
    state.add_edge(exit_, "OUT_A", src, None, dace.Memlet("A[0:8]"))
    exit_.add_in_connector("IN_A")
    exit_.add_out_connector("OUT_A")

    feats = dataset.nest_features(sdfg)
    assert feats["innermost_extents"] == ["4"], feats["innermost_extents"]  # ceil(7/2), not "3.5"
    assert feats["symbolic_extents"] == 0, "a constant strided extent was booked as symbolic"
