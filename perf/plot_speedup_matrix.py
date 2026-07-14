"""The deliverable report: cross-language x compiler x flag **speedup matrix** vs the gcc and the llvm
vendor default, plus per-kernel speedup **scatter** plots vs each.

Reads the per-kernel JSON that :mod:`nestforge.perf.tsvc_full` writes (records with a ``skipped`` key are
ignored) and renders, WITHOUT recompiling anything.

Two baselines (the "default vector and fp flags" a user gets with no tuning), each its own tab:
  * **gcc default**  = the ``gcc`` compiler's C cell at ``sequential`` / ``default`` cost / ``default-fp``.
  * **llvm default** = the ``clang`` compiler's C cell at ``sequential`` / ``default`` cost / ``default-fp``.

Both are ``role=="timing"``, ``ok==true`` cells with a finite median. A kernel with no such baseline cell
(that compiler absent, or its default cell failed to validate) is dropped from THAT tab -- never divided by.

Outputs (next to the results; ``--out-dir`` overrides)
  * ``speedup_matrix.md`` -- for each baseline: a matrix whose rows are (compiler, language) and whose
    columns are the flag-combos (``parallel`` / ``cost`` / ``fp``), each cell the GEOMEAN over kernels of
    ``baseline_median / cell_median`` (>1 = faster than the vendor default). n = kernels contributing.
  * ``scatter_vs_gcc.png`` / ``scatter_vs_llvm.png`` -- per kernel, the speedup of nest-forge's BEST cell
    (fastest validated cell across every compiler+flag) over that vendor default, coloured by the winning
    compiler; a dashed 1.0 line marks parity. The headline "what you gain per kernel" plot.

This is a READER -- it never runs a kernel. Non-finite / missing timings are "no data" (never plotted,
never a denominator).

Usage::

    python perf/plot_speedup_matrix.py --results-dir perf_results/tsvc_full
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: pick the non-interactive backend BEFORE importing pyplot

import matplotlib.pyplot as plt  # noqa: E402 -- must follow matplotlib.use("Agg")

#: The two vendor-default baselines, keyed by the tab label -> the compiler name whose default cell is it.
BASELINES: Dict[str, str] = {"gcc": "gcc", "llvm": "clang"}

#: The axis coordinates of a "vendor default" cell: no parallelism, the compiler's own vectorizer cost
#: model, and its default FP. This is what "default vector and fp flags" means.
DEFAULT_PARALLEL, DEFAULT_COST, DEFAULT_FP, DEFAULT_LANG = "sequential", "default", "default-fp", "c"


def finite(x) -> bool:
    """True only for a real, usable numeric time (a finite int/float); rejects ``None`` / ``inf`` / ``nan``."""
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


def timing_cells(kernel: dict) -> List[dict]:
    """The validated, finite-median lane-3 timing cells of one kernel."""
    return [
        c for c in (kernel.get("cells") or [])
        if c.get("role") == "timing" and c.get("ok") is True and finite(c.get("median_us"))
    ]


def kernel_time_by_nest(cells: List[dict], predicate) -> Optional[float]:
    """A kernel's time for the cells matching ``predicate``, summed over its nests (each nest's own matching
    cell). ``None`` unless EVERY nest the kernel timed has a matching validated cell -- so a partial match is
    not mistaken for the whole kernel. A single-nest kernel is the usual case (one term)."""
    matched = [c for c in cells if predicate(c)]
    if not matched:
        return None
    nests = {c.get("nest", 0) for c in cells}
    total = 0.0
    for n in nests:
        per_nest = [c for c in matched if c.get("nest", 0) == n]
        if not per_nest:
            return None  # this nest has no matching cell -> the whole-kernel time is undefined for this combo
        total += min(c["median_us"] for c in per_nest)
    return total


def baseline_time(kernel: dict, compiler: str) -> Optional[float]:
    """The vendor-default whole-kernel time for ``compiler``: the C / sequential / default / default-fp cell,
    summed over nests. ``None`` if that compiler's default cell is absent or unvalidated for any nest."""
    cells = timing_cells(kernel)
    return kernel_time_by_nest(
        cells, lambda c: c.get("compiler") == compiler and c.get("language") == DEFAULT_LANG and c.get("parallel") ==
        DEFAULT_PARALLEL and c.get("cost_model") == DEFAULT_COST and c.get("fp_mode") == DEFAULT_FP)


def cell_axes(cell: dict) -> Tuple[str, str, str, str, str]:
    """The (compiler, language, parallel, cost, fp) coordinate of a cell -- a column/row key of the matrix."""
    return (str(cell.get("compiler")), str(cell.get("language")), str(cell.get("parallel")),
            str(cell.get("cost_model")), str(cell.get("fp_mode")))


@dataclass
class MatrixTab:
    """One baseline's speedup matrix: (compiler,lang) rows x (parallel/cost/fp) cols -> geomean speedup."""
    baseline: str  # "gcc" | "llvm"
    rows: List[Tuple[str, str]]  # (compiler, language), sorted
    cols: List[Tuple[str, str, str]]  # (parallel, cost, fp), sorted
    geo: Dict[Tuple, float]  # (compiler, lang, parallel, cost, fp) -> geomean speedup
    n: Dict[Tuple, int]  # same key -> #kernels contributing


def build_matrix(kernels: List[dict], baseline_compiler: str, baseline_label: str) -> MatrixTab:
    """For every (compiler, lang, parallel, cost, fp) coordinate, the geomean over kernels of
    ``baseline / cell`` (the cell's whole-kernel time vs the vendor default), plus the contributing count."""
    samples: Dict[Tuple, List[float]] = {}
    for kernel in kernels:
        if "skipped" in kernel:
            continue
        base = baseline_time(kernel, baseline_compiler)
        if base is None:
            continue
        cells = timing_cells(kernel)
        coords = {cell_axes(c) for c in cells}
        for coord in coords:
            comp, lang, par, cost, fp = coord
            t = kernel_time_by_nest(cells, lambda c, coord=coord: cell_axes(c) == coord)
            if t is not None and t > 0.0:
                samples.setdefault(coord, []).append(base / t)
    geo = {k: g for k, sps in samples.items() if (g := geomean(sps)) is not None}
    n = {k: len(sps) for k, sps in samples.items()}
    rows = sorted({(k[0], k[1]) for k in geo})
    cols = sorted({(k[2], k[3], k[4]) for k in geo})
    return MatrixTab(baseline=baseline_label, rows=rows, cols=cols, geo=geo, n=n)


def render_matrix_md(tabs: List[MatrixTab]) -> str:
    """A markdown report with one matrix section ('tab') per baseline: rows (compiler, language) x columns
    (parallel/cost/fp), each cell the geomean speedup (``>1`` beats the vendor default). ``—`` = no data."""
    lines = [
        "# TSVC speedup matrix — cross-language x compiler x flag vs the vendor default", "",
        "Each cell = **geomean over kernels** of `baseline_median / cell_median` (>1.0 = faster than that "
        "vendor's default `-O3 -march=native` at default vectorization + FP). A kernel counts toward a cell "
        "only when both its baseline and that cell validated on every nest.", ""
    ]
    for tab in tabs:
        lines += [f"## Speedup vs {tab.baseline} default (C, sequential, default cost, default FP)", ""]
        if not tab.rows or not tab.cols:
            lines += [
                f"_no data for the {tab.baseline} baseline (compiler absent or its default cell "
                "did not validate)_", ""
            ]
            continue
        head = "| compiler / lang | " + " | ".join(f"{p}/{c}/{f}" for p, c, f in tab.cols) + " |"
        lines += [head, "|" + "---|" * (len(tab.cols) + 1)]
        for comp, lang in tab.rows:
            cells = []
            for par, cost, fp in tab.cols:
                key = (comp, lang, par, cost, fp)
                g = tab.geo.get(key)
                cells.append("—" if g is None else f"{g:.2f}x")
            lines.append(f"| {comp} {lang} | " + " | ".join(cells) + " |")
        # the single best coordinate for this baseline (headline).
        best = max(tab.geo, key=lambda k: tab.geo[k])
        bc, bl, bp, bcost, bfp = best
        lines += [
            "", f"**Best coordinate vs {tab.baseline}:** `{bc} {bl} {bp}/{bcost}/{bfp}` at "
            f"**{tab.geo[best]:.2f}x** (n={tab.n.get(best, 0)}).", ""
        ]
    return "\n".join(lines) + "\n"


def scatter_speedups(kernels: List[dict], baseline_compiler: str) -> List[Tuple[str, float, str]]:
    """Per kernel: ``(key, best_speedup_vs_baseline, winning_compiler)`` -- nest-forge's fastest validated
    cell (whole-kernel, summed over nests) over that vendor default. Kernels lacking the baseline are skipped."""
    points: List[Tuple[str, float, str]] = []
    for kernel in kernels:
        if "skipped" in kernel:
            continue
        base = baseline_time(kernel, baseline_compiler)
        if base is None:
            continue
        cells = timing_cells(kernel)
        # nest-forge best = per compiler+config coordinate, the whole-kernel time; take the global min.
        coords = {cell_axes(c) for c in cells}
        best_t, best_comp = None, "—"
        for coord in coords:
            t = kernel_time_by_nest(cells, lambda c, coord=coord: cell_axes(c) == coord)
            if t is not None and t > 0.0 and (best_t is None or t < best_t):
                best_t, best_comp = t, coord[0]
        if best_t is not None:
            points.append((str(kernel.get("key", "?")), base / best_t, best_comp))
    return points


def plot_scatter(points: List[Tuple[str, float, str]], baseline_label: str, out_png: Path) -> Optional[float]:
    """A per-kernel speedup scatter vs one vendor default, coloured by winning compiler; returns the geomean.
    x = kernel rank (sorted by speedup), y = speedup (log scale); a dashed 1.0 line marks parity."""
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    if not points:
        ax.text(0.5,
                0.5,
                f"no kernels with a {baseline_label} baseline",
                ha="center",
                va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        return None
    pts = sorted(points, key=lambda p: p[1])
    palette = {"gcc": "#4c72b0", "clang": "#dd8452", "nvhpc": "#55a868", "intel": "#c44e52"}
    xs = list(range(len(pts)))
    ys = [p[1] for p in pts]
    cs = [palette.get(p[2], "#888888") for p in pts]
    ax.scatter(xs, ys, c=cs, s=18)
    ax.axhline(1.0, linestyle="--", linewidth=0.9, color="#888888")
    ax.set_yscale("log")
    ax.set_xlabel(f"kernel (sorted by speedup), n={len(pts)}")
    ax.set_ylabel(f"nest-forge best / {baseline_label} default")
    geo = geomean(ys)
    seen = {p[2] for p in pts}
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=palette.get(c, "#888888"), label=c) for c in sorted(seen)
    ]
    ax.legend(handles=handles, title="winning compiler", fontsize=8, loc="upper left")
    ax.set_title(
        f"Per-kernel speedup of nest-forge best vs {baseline_label} default  |  "
        f"geomean {('n/a' if geo is None else f'{geo:.2f}x')}",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return geo


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="TSVC speedup matrix + per-kernel scatter vs gcc / llvm defaults")
    ap.add_argument("--results-dir", required=True, help="dir of nestforge.perf.tsvc_full per-kernel JSON")
    ap.add_argument("--out-dir", default=None, help="override the output dir (default: the results dir)")
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    kernels = load_results(results_dir)

    tabs = [build_matrix(kernels, comp, label) for label, comp in BASELINES.items()]
    (out_dir / "speedup_matrix.md").write_text(render_matrix_md(tabs))

    scatter_png = {"gcc": out_dir / "scatter_vs_gcc.png", "llvm": out_dir / "scatter_vs_llvm.png"}
    for label, comp in BASELINES.items():
        geo = plot_scatter(scatter_speedups(kernels, comp), label, scatter_png[label])
        print(f"[speedup-matrix] per-kernel best vs {label} default: geomean "
              f"{('n/a' if geo is None else f'{geo:.2f}x')}")
    print(f"[speedup-matrix] wrote {out_dir / 'speedup_matrix.md'}, {scatter_png['gcc']}, {scatter_png['llvm']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
