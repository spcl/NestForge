"""Plot the TSVC full-matrix winners (``nestforge.perf.tsvc_full``): the single best compiler if you had
to pick ONE for everything, versus letting nest-forge pick the best compiler+config PER kernel.

Reads the per-kernel JSON that :mod:`nestforge.perf.tsvc_full` writes (records with a ``skipped`` key are
ignored) and renders, WITHOUT recompiling anything:

Definitions
-----------
  * **baseline** per kernel = ``dace_cpp.median_us`` when finite (the strict-ieee DaCe-cpp lane -- the
    speedup denominator); else ``native.median_us``; else the kernel is dropped (no denominator).
  * a kernel's **best cell for compiler C** = the min ``median_us`` over its ``role=="timing"``,
    ``ok==true``, ``compiler==C`` cells with a finite median. speedup = baseline / that median.
  * **single-compiler winner** ``C*`` = the compiler whose GEOMEAN speedup over the kernels it could run
    is the largest ("pick ONE compiler for everything"). Every compiler's geomean is reported.
  * **nest-forge per-kernel best** = per kernel, the globally fastest timing cell across ALL compilers;
    speedup vs baseline, geomean over kernels ("let nest-forge pick the best compiler+config per kernel").

Outputs (next to the results; ``--out`` overrides the PNG path)
  * ``winners.png`` -- geomean speedup per compiler plus the nest-forge per-kernel-best bar, with ``C*``
    and the nest-forge bar highlighted.
  * ``winners_per_kernel.csv`` and ``winners_per_kernel.md`` -- the winning cell per kernel
    (compiler / language / parallel / cost_model / fp_mode / median_us / speedup), sorted by corpus,key.

Non-finite / missing timings (a median sanitized to ``null`` by the driver, an ``Infinity``) are treated
as "no data": never plotted, never divided by. A kernel with no valid timing cell is listed with ``--``
rather than crashing. This is a READER -- it never runs a kernel.

Usage::

    python perf/plot_winners.py --results-dir perf_results/tsvc_full
    python perf/plot_winners.py --results-dir perf_results/tsvc_full --out /tmp/winners.png
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless: pick the non-interactive backend BEFORE importing pyplot

import matplotlib.pyplot as plt  # noqa: E402 -- must follow matplotlib.use("Agg")

#: label for the "let nest-forge choose per kernel" bar in the chart / summary.
NESTFORGE_LABEL = "nestforge-best"


def finite(x) -> bool:
    """True only for a real, usable numeric time: a finite int/float. Rejects ``None`` (a non-finite value
    the driver mapped to ``null`` on write) and ``inf`` / ``nan``."""
    return isinstance(x, (int, float)) and math.isfinite(x)


def geomean(xs: List[float]) -> Optional[float]:
    """Geometric mean of the finite positive values; ``None`` when there are none."""
    vals = [x for x in xs if finite(x) and x > 0.0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def load_results(results_dir: Path) -> List[dict]:
    """Every per-kernel JSON in ``results_dir`` (``tables.md`` and unparseable files skipped)."""
    rows: List[dict] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "tables.md":
            continue
        try:
            rows.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError, ValueError):
            continue
    return rows


def baseline_us(kernel: dict) -> Optional[float]:
    """The speedup denominator: ``dace_cpp.median_us`` if finite, else ``native.median_us`` if finite,
    else ``None`` (kernel cannot be scored)."""
    for lane in ("dace_cpp", "native"):
        info = kernel.get(lane) or {}
        med = info.get("median_us")
        if finite(med):
            return float(med)
    return None


def timing_cells(kernel: dict) -> List[dict]:
    """The validated, finite-median lane-3 timing cells of one kernel."""
    return [
        c for c in (kernel.get("cells") or [])
        if c.get("role") == "timing" and c.get("ok") is True and finite(c.get("median_us"))
    ]


def best_for_compiler(cells: List[dict], compiler: str) -> Optional[dict]:
    """Fastest (min median) timing cell for one compiler, or ``None``."""
    grp = [c for c in cells if c.get("compiler") == compiler]
    return min(grp, key=lambda c: c["median_us"]) if grp else None


def best_cell(cells: List[dict]) -> Optional[dict]:
    """Globally fastest (min median) timing cell across all compilers, or ``None``."""
    return min(cells, key=lambda c: c["median_us"]) if cells else None


@dataclass
class WinnerRow:
    """The winning cell for one kernel (or a no-winner sentinel), for the per-kernel listing."""
    corpus: str
    key: str
    compiler: str = "—"
    language: str = "—"
    parallel: str = "—"
    cost_model: str = "—"
    fp_mode: str = "—"
    median_us: Optional[float] = None
    speedup: Optional[float] = None


@dataclass
class Winners:
    """The full analysis: per-kernel winner rows plus the aggregate geomeans."""
    rows: List[WinnerRow] = field(default_factory=list)
    compiler_geo: Dict[str, float] = field(default_factory=dict)  # compiler -> geomean speedup
    compiler_wins: Dict[str, int] = field(default_factory=dict)  # compiler -> #kernels it wins globally
    cstar: Optional[str] = None  # single-compiler winner
    nestforge_geo: Optional[float] = None
    n_scored: int = 0  # kernels with a baseline AND a valid winning cell


def analyze(kernels: List[dict]) -> Winners:
    """Score every non-skipped kernel: per-compiler best-speedup samples, the global per-kernel best, and
    the winner listing. Robust to missing baselines / cells -- such kernels list as ``--`` no-winner."""
    compiler_speedups: Dict[str, List[float]] = {}
    compiler_wins: Dict[str, int] = {}
    nestforge_speedups: List[float] = []
    rows: List[WinnerRow] = []
    n_scored = 0

    for kernel in kernels:
        if "skipped" in kernel:
            continue
        corpus = str(kernel.get("corpus", "?"))
        key = str(kernel.get("key", "?"))
        base = baseline_us(kernel)
        cells = timing_cells(kernel)
        winner = best_cell(cells)

        if base is not None:
            for compiler in {str(c.get("compiler")) for c in cells}:
                cbest = best_for_compiler(cells, compiler)
                if cbest is not None:
                    compiler_speedups.setdefault(compiler, []).append(base / cbest["median_us"])

        if base is None or winner is None:
            rows.append(WinnerRow(corpus=corpus, key=key))
            continue

        speedup = base / winner["median_us"]
        nestforge_speedups.append(speedup)
        wcomp = str(winner.get("compiler", "—"))
        compiler_wins[wcomp] = compiler_wins.get(wcomp, 0) + 1
        n_scored += 1
        rows.append(
            WinnerRow(corpus=corpus,
                      key=key,
                      compiler=wcomp,
                      language=str(winner.get("language", "—")),
                      parallel=str(winner.get("parallel", "—")),
                      cost_model=str(winner.get("cost_model", "—")),
                      fp_mode=str(winner.get("fp_mode", "—")),
                      median_us=float(winner["median_us"]),
                      speedup=speedup))

    compiler_geo = {c: g for c, sps in compiler_speedups.items() if (g := geomean(sps)) is not None}
    cstar = max(compiler_geo, key=lambda c: compiler_geo[c]) if compiler_geo else None
    rows.sort(key=lambda r: (r.corpus, r.key))
    return Winners(rows=rows,
                   compiler_geo=compiler_geo,
                   compiler_wins=compiler_wins,
                   cstar=cstar,
                   nestforge_geo=geomean(nestforge_speedups),
                   n_scored=n_scored)


def num(x: Optional[float], suffix: str = "") -> str:
    """Format a numeric for a table cell: ``—`` when absent, else 2dp + optional suffix."""
    return "—" if x is None else f"{x:.2f}{suffix}"


def write_per_kernel(csv_path: Path, md_path: Path, rows: List[WinnerRow]) -> None:
    """The per-kernel winner listing as both a CSV and a markdown table (rows already sorted)."""
    header = ["corpus", "key", "compiler", "language", "parallel", "cost_model", "fp_mode", "median_us", "speedup"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for r in rows:
            writer.writerow([
                r.corpus, r.key, r.compiler, r.language, r.parallel, r.cost_model, r.fp_mode,
                "" if r.median_us is None else repr(r.median_us), "" if r.speedup is None else repr(r.speedup)
            ])

    lines = [
        "# TSVC per-kernel winners (nest-forge best compiler+config per kernel)", "", "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header)
    ]
    for r in rows:
        lines.append(f"| {r.corpus} | {r.key} | {r.compiler} | {r.language} | {r.parallel} | {r.cost_model} "
                     f"| {r.fp_mode} | {num(r.median_us)} | {num(r.speedup, 'x')} |")
    md_path.write_text("\n".join(lines) + "\n")


def plot_winners(result: Winners, out_png: Path) -> None:
    """Bars of geomean speedup: one per compiler (single-compiler pick) plus the nest-forge per-kernel-best
    bar. ``C*`` and the nest-forge bar are highlighted; a dashed line marks the 1.0 (baseline) level."""
    compilers = sorted(result.compiler_geo, key=lambda c: result.compiler_geo[c], reverse=True)
    labels = list(compilers)
    heights = [result.compiler_geo[c] for c in compilers]
    if result.nestforge_geo is not None:
        labels.append(NESTFORGE_LABEL)
        heights.append(result.nestforge_geo)

    colors = []
    for lab in labels:
        if lab == NESTFORGE_LABEL:
            colors.append("#55a868")  # nest-forge per-kernel best
        elif lab == result.cstar:
            colors.append("#c44e52")  # single-compiler winner C*
        else:
            colors.append("#4c72b0")

    fig, ax = plt.subplots(figsize=(max(6.0, len(labels) * 1.1), 5.0))
    if not labels:
        ax.text(0.5,
                0.5,
                "no scored kernels (no baseline + valid timing cell)",
                ha="center",
                va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
    else:
        xs = list(range(len(labels)))
        ax.bar(xs, heights, color=colors, width=0.6)
        ax.axhline(1.0, linestyle="--", linewidth=0.8, color="#888888")
        for x, h in zip(xs, heights):
            ax.text(x, h, f"{h:.2f}x", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("geomean speedup vs DaCe-cpp baseline")
    star_txt = "n/a" if result.cstar is None else f"{result.cstar} ({num(result.compiler_geo.get(result.cstar))}x)"
    ax.set_title(
        f"Single-compiler winner C* = {star_txt}  |  nest-forge best = {num(result.nestforge_geo)}x "
        f"over {result.n_scored} kernels",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def print_summary(result: Winners) -> None:
    """Short stdout summary: C* + its geomean, nest-forge-best geomean, and per-compiler win counts."""
    star = "n/a" if result.cstar is None else f"{result.cstar} (geomean {num(result.compiler_geo.get(result.cstar))}x)"
    print(f"[plot-winners] single-compiler winner C* = {star}")
    print(f"[plot-winners] nest-forge per-kernel best geomean = {num(result.nestforge_geo)}x "
          f"over {result.n_scored} scored kernels")
    for compiler in sorted(result.compiler_geo, key=lambda c: result.compiler_geo[c], reverse=True):
        wins = result.compiler_wins.get(compiler, 0)
        print(f"[plot-winners]   {compiler}: geomean {result.compiler_geo[compiler]:.2f}x, wins {wins} kernels")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Plot the TSVC full-matrix compiler winners (reader, no builds)")
    ap.add_argument("--results-dir", required=True, help="dir of nestforge.perf.tsvc_full per-kernel JSON")
    ap.add_argument("--out", default=None, help="override the PNG path (default: <results-dir>/winners.png)")
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir)
    result = analyze(load_results(results_dir))

    out_png = Path(args.out) if args.out else results_dir / "winners.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    write_per_kernel(results_dir / "winners_per_kernel.csv", results_dir / "winners_per_kernel.md", result.rows)
    plot_winners(result, out_png)
    print_summary(result)
    print(f"[plot-winners] wrote {out_png}, {results_dir / 'winners_per_kernel.csv'}, "
          f"and {results_dir / 'winners_per_kernel.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
