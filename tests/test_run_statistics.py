# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``scripts/run_statistics.py`` turns the plugin records a results DB kept into one row per episode: calls and
refusals by class, wasted attempts, and the tokens before and after the first ``write_kernel``, which add up."""

import csv
import pathlib
import subprocess
import sys

from hpcagent_bench.anticheat import Judgement
from hpcagent_bench.categories import Baseline, Language, Oracle, SourceMode
from hpcagent_bench.harness import plugin_records, recording
from hpcagent_bench.harness.envelope import Submission
from hpcagent_bench.harness.scoring import Score
from hpcagent_bench.harness.task import Task
from pydantic import JsonValue

from nestforge.paths import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "run_statistics.py"
KERNEL = "tsvc_2_s212"  # any real, fast-loading kernel
SETUP = "nestforge-llr"


def call(index: int, tool: str, status: str, tokens: int, refusal: str = "", forced: bool = False) -> JsonValue:
    return {"index": index, "tool": tool, "kind": None, "family": None, "status": status, "refusal": refusal,
            "forced": forced, "phase": "moves", "tokens": tokens}  # fmt: skip


def record(calls: list[JsonValue], first_write: int | None) -> dict[str, JsonValue]:
    return {"kernel": KERNEL, "disabled_families": [], "reference": None, "calls": calls, "journal": [],
            "first_write_kernel": first_write, "final": {"kernels": 1}}  # fmt: skip


def graded(db: pathlib.Path, folder: pathlib.Path, episode: str) -> None:
    """One credited /submit of ``KERNEL`` in ``episode``, whose folder holds the plugin record."""
    score = Score(
        correct=True, max_rel_error=0.0, native_ns=1000, build_ok=True, baseline_ns=2000, speedup=2.0,
        baseline=Baseline.NUMBA.text, public_correct=True, hidden_correct=True, hidden_passed=2, hidden_total=2,
        oracle=Oracle.AUTO.text,
    )  # fmt: skip
    submission = Submission(language=Language.C, source="/* x */", agent_root=str(folder))
    task = Task(KERNEL, SourceMode.RESTRICTED, Language.C)
    recording.record(score, submission, task, judgement=Judgement(), episode_id=episode, path=str(db))


def test_two_episodes_give_two_rows_whose_token_split_adds_up(tmp_path: pathlib.Path) -> None:
    db, out = tmp_path / "results.db", tmp_path / "stats.csv"
    moves_only = [call(0, "apply_move", "applied", 100), call(1, "apply_move", "illegal", 250, "dependence"),
                  call(2, "finalize", "ok", 300)]  # fmt: skip
    with_kernel = [call(0, "apply_move", "not-found", 50, "unknown-label"), call(1, "write_kernel", "build-failed", 400),
                   call(2, "write_kernel", "wrong", 700), call(3, "finalize", "ok", 1000)]  # fmt: skip
    for worker, calls, first_write in ((1, moves_only, None), (2, with_kernel, 1)):
        folder = tmp_path / f"w{worker}"
        plugin_records.keep(folder, "nestforge", 1, record(calls, first_write))
        graded(db, folder, f"{SETUP}.n0.p1.w{worker}")

    subprocess.run([sys.executable, str(SCRIPT), str(db), "--setup", SETUP, "--out", str(out)], check=True)

    rows = {row["episode"].rsplit(".", 1)[1]: row for row in csv.DictReader(out.open())}
    assert sorted(rows) == ["w1", "w2"]
    a, b = rows["w1"], rows["w2"]
    assert (a["applied"], a["refused"], a["refused_dependence"], a["refusal_rate"]) == ("1", "1", "1", "0.5")
    assert (a["tokens_before_write"], a["tokens_after_write"]) == ("", ""), "no kernel written: no split"
    assert (b["refused_unknown_label"], b["build_failures"], b["wrong_results"]) == ("1", "1", "1")
    assert (b["tokens_before_write"], b["tokens_after_write"], b["tokens_at_last_call"]) == ("400", "600", "1000")


def test_wrong_kernels_after_the_first_forced_move_are_counted_apart(tmp_path: pathlib.Path) -> None:
    db, out = tmp_path / "results.db", tmp_path / "stats.csv"
    calls = [call(0, "write_kernel", "wrong", 10), call(1, "apply_move", "applied", 20, forced=True),
             call(2, "write_kernel", "wrong", 30), call(3, "apply_move", "applied", 40, forced=True),
             call(4, "finalize", "ok", 50)]  # fmt: skip
    folder = tmp_path / "w1"
    plugin_records.keep(folder, "nestforge", 1, record(calls, 0))
    graded(db, folder, f"{SETUP}.n0.p1.w1")

    subprocess.run([sys.executable, str(SCRIPT), str(db), "--setup", SETUP, "--out", str(out)], check=True)

    (row,) = csv.DictReader(out.open())
    assert (row["forced"], row["wrong_results"], row["wrong_after_forced"]) == ("2", "2", "1")
