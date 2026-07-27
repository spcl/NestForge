# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the foundation-track sweep driver (nestforge.perf.foundation_sweep).

Only the merge step is covered here: it is the one part that runs after a long multi-rank sweep, where a
crash costs hours of already-finished measurement.
"""
import json

from nestforge.perf import foundation_sweep


def write_row(out_dir, name, **over):
    row = {"kernel": name, "seconds": 1.0, "nests": [{"error": None, "collapsed": []}]}
    row.update(over)
    (out_dir / f"{name}.json").write_text(json.dumps(row))


def test_merge_tables_counts_measured_and_failed_kernels(tmp_path, capsys):
    write_row(tmp_path, "good")
    write_row(tmp_path, "broken", nests=None, error="emit: UnsupportedNest")
    assert foundation_sweep.merge_tables(tmp_path) == 0
    out = capsys.readouterr().out
    assert "kernels: 2 | reached the arena: 1 | failed outright: 1" in out
    assert "FAILED broken" in out
    assert json.loads((tmp_path / "summary.json").read_text())["kernels"] == 2


def test_merge_tables_survives_a_truncated_json_and_names_it(tmp_path, capsys):
    """A rank killed mid-write (OOM, walltime) leaves a partial file. Parsed unguarded it raised and took
    the whole sweep's results down at the last step -- hours of measurement lost to the one file that did
    not matter. The bad file must be NAMED, so a short table is never read as a complete one."""
    write_row(tmp_path, "good")
    (tmp_path / "killed.json").write_text('{"kernel": "killed", "nests": [{"err')
    assert foundation_sweep.merge_tables(tmp_path) == 0
    out = capsys.readouterr().out
    assert "kernels: 1 | reached the arena: 1" in out  # the readable kernel still reported
    assert "UNREADABLE killed.json" in out
    assert json.loads((tmp_path / "summary.json").read_text())["unreadable"], "the reason must persist too"


def test_merge_tables_ignores_its_own_summary(tmp_path, capsys):
    """summary.json is an OUTPUT of this function; folding a previous run's copy back in would inflate the
    kernel count on every re-merge."""
    write_row(tmp_path, "good")
    foundation_sweep.merge_tables(tmp_path)
    capsys.readouterr()  # drop the first merge's output; only the RE-merge is under test
    foundation_sweep.merge_tables(tmp_path)
    assert "kernels: 1 |" in capsys.readouterr().out
