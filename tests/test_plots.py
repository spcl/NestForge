"""Compiler-free smoke tests for the two perf plotting readers (``perf/plot_overhead.py`` and
``perf/plot_winners.py``).

They write a few SYNTHETIC per-kernel JSON files (deliberately including ``Infinity`` and ``null``
timings, to prove the non-finite handling) into a tmp dir, invoke each script as a SUBPROCESS (matching
how the sbatch drivers call them -- the scripts live in ``perf/``, which is not a package, so this avoids
any import-by-path), and assert the PNG + CSV / markdown outputs exist and parse.

No compiler, no DaCe, fast; ``-n4``-safe (each test uses its own ``tmp_path``).
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PLOT_OVERHEAD = REPO_ROOT / "perf" / "plot_overhead.py"
PLOT_WINNERS = REPO_ROOT / "perf" / "plot_winners.py"


def run_script(script: Path, results_dir: Path) -> subprocess.CompletedProcess:
    """Invoke a plotting script exactly as the sbatch drivers do: as a subprocess with ``--results-dir``."""
    return subprocess.run(
        [sys.executable, str(script), "--results-dir", str(results_dir)], capture_output=True, text=True, check=True)


def test_plot_overhead_smoke(tmp_path):
    # a normal kernel, a fast-overhead kernel, and one with a NON-FINITE external time (Infinity) whose
    # ratio is null -> must be tolerated, not plotted or divided by.
    (tmp_path / "s000.json").write_text(
        json.dumps({
            "key": "s000",
            "compiler": "g++",
            "codegen_ms": 40.0,
            "monolithic_ms": 100.0,
            "external_ms": 130.0,
            "overhead_ratio": 1.3
        }))
    (tmp_path / "s111.json").write_text(
        json.dumps({
            "key": "s111",
            "compiler": "g++",
            "codegen_ms": 41.0,
            "monolithic_ms": 200.0,
            "external_ms": 220.0,
            "overhead_ratio": 1.1
        }))
    # json.dumps writes Infinity/None (Python's JSON extension); the reader must survive both.
    (tmp_path / "s112.json").write_text('{"key": "s112", "compiler": "g++", "codegen_ms": 40.0, '
                                        '"monolithic_ms": 150.0, "external_ms": Infinity, "overhead_ratio": null}')
    (tmp_path / "s113.json").write_text(json.dumps({"key": "s113", "compiler": "g++", "skipped": "no compute nest"}))

    run_script(PLOT_OVERHEAD, tmp_path)

    assert (tmp_path / "overhead.png").exists()
    csv_path = tmp_path / "overhead.csv"
    assert csv_path.exists()
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = {r["key"] for r in rows}
    assert {"s000", "s111", "s112"} <= keys  # completed kernels present
    assert "s113" not in keys  # the skipped kernel is dropped
    s112 = next(r for r in rows if r["key"] == "s112")
    assert s112["overhead_ratio"] == "" and s112["external_ms"] == ""  # non-finite -> empty, not a crash


def test_plot_winners_smoke(tmp_path):

    def cell(compiler,
             median,
             ok=True,
             role="timing",
             lang="c",
             parallel="sequential",
             cost="default",
             fp="default-fp"):
        return {
            "opt_mode": "baseline",
            "language": lang,
            "compiler": compiler,
            "parallel": parallel,
            "cost_model": cost,
            "fp_mode": fp,
            "role": role,
            "ok": ok,
            "maxdiff": 0.0,
            "median_us": median
        }

    # kernel A: baseline 100us; gcc best 50us (2x), clang best 25us (4x) -> clang wins globally.
    (tmp_path / "tsvc2_sA.json").write_text(
        json.dumps({
            "key": "sA",
            "corpus": "tsvc2",
            "dace_cpp": {
                "compiler": "gcc",
                "ok": True,
                "median_us": 100.0
            },
            "native": {
                "compiler": "gcc",
                "ok": True,
                "median_us": 90.0
            },
            "cells": [cell("gcc", 50.0), cell("clang", 25.0),
                      cell("clang", 1.0, ok=False)]
        }))
    # kernel B: dace_cpp median is null -> falls back to native (200us); gcc 40us (5x), clang 100us (2x).
    (tmp_path / "tsvc2_sB.json").write_text('{"key": "sB", "corpus": "tsvc2", '
                                            '"dace_cpp": {"compiler": "gcc", "ok": false, "median_us": null}, '
                                            '"native": {"compiler": "gcc", "ok": true, "median_us": 200.0}, '
                                            '"cells": [' + json.dumps(cell("gcc", 40.0)) + ', ' +
                                            json.dumps(cell("clang", 100.0)) + ', ' +
                                            json.dumps(cell("gcc", 999.0, role="gate")) + ']}')
    # kernel C: no valid timing cell (only a not-ok cell) -> listed as no-winner, must not crash.
    (tmp_path / "tsvc2_sC.json").write_text(
        json.dumps({
            "key": "sC",
            "corpus": "tsvc2",
            "dace_cpp": {
                "compiler": "gcc",
                "ok": True,
                "median_us": 100.0
            },
            "cells": [cell("gcc", 10.0, ok=False)]
        }))
    # a fully skipped kernel is ignored.
    (tmp_path / "tsvc2_sD.json").write_text(json.dumps({"key": "sD", "corpus": "tsvc2", "skipped": "no nest"}))

    proc = run_script(PLOT_WINNERS, tmp_path)

    assert (tmp_path / "winners.png").exists()
    assert (tmp_path / "winners_per_kernel.md").exists()
    csv_path = tmp_path / "winners_per_kernel.csv"
    assert csv_path.exists()
    with csv_path.open(newline="") as handle:
        rows = {r["key"]: r for r in csv.DictReader(handle)}
    assert set(rows) == {"sA", "sB", "sC"}  # sD skipped
    assert rows["sA"]["compiler"] == "clang"  # 4x beats gcc 2x
    assert rows["sB"]["compiler"] == "gcc"  # 5x beats clang 2x (baseline via native fallback)
    assert rows["sC"]["compiler"] == "—" and rows["sC"]["median_us"] == ""  # no valid cell -> no winner
    # C* single-compiler winner: geomean(gcc)=sqrt(2*5)=3.16 > geomean(clang)=sqrt(4*2)=2.83.
    assert "C* = gcc" in proc.stdout
