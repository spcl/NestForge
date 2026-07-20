"""The E1-E5 runner script: argument handling and the JSON it writes. The table-shaping is unit-tested
(no compile); a real bounded run is integration."""
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_experiments import EXPERIMENTS, jsonable, main  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_experiments.py"


def test_tuple_keys_survive_serialization():
    # tables are keyed by (kernel, backend); str(tuple) would render "('k', 'gcc')" and a naive flatten
    # would drop the backend half, silently merging every backend's row into one.
    table = {("k", "gcc"): {"quality": 1.0}, ("k", "clang"): {"quality": 0.5}}
    out = jsonable(table)
    assert out == {"k | gcc": {"quality": 1.0}, "k | clang": {"quality": 0.5}}
    assert json.dumps(out)  # actually serializable, not merely reshaped


def test_non_finite_numbers_stay_visible():
    # inf/nan are not JSON. Coercing them to null would make a failed measurement indistinguishable from
    # one that was never attempted.
    out = jsonable({"a": float("inf"), "b": float("nan"), "c": 1.5})
    assert out["a"] == "inf" and out["b"] == "nan" and out["c"] == 1.5
    assert json.dumps(out)


def test_dataclass_rows_serialize():
    from nestforge.experiment_e1 import E1Cell
    out = jsonable([E1Cell("k", "gcc", "atoms", "map", 4.0, True)])
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


@pytest.mark.integration
def test_bounded_run_writes_every_requested_table(tmp_path):
    # the real pipeline: one kernel, two rungs, every experiment -- each must leave a JSON file holding
    # both its rows and its read-off.
    rc = main([
        "--out",
        str(tmp_path), "--kernels", "1", "--granularity-points", "2", "--reps", "3", "--experiments", "e1,e2,e3,e4,e5"
    ])
    assert rc == 0
    for name in EXPERIMENTS:
        path = tmp_path / f"{name}.json"
        assert path.exists(), f"{name} wrote no table"
        payload = json.loads(path.read_text())
        assert "rows" in payload and payload["rows"], f"{name} produced no rows"


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


def test_module_entry_point_exists():
    # `python scripts/run_experiments.py` must reach main() -- runpy proves the __main__ guard is wired.
    with pytest.raises(SystemExit):
        runpy.run_path(str(SCRIPT), run_name="__main__")
