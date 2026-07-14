"""Per-kernel speedup of **every framework** over a single-compiler, single-language baseline.

Motivation
----------
``plot_speedup_matrix`` answers "what does nest-forge's *single best* cell gain over the vendor C
default". This reader instead shows **all frameworks at once**: for each kernel it plots, as its own
series, the speedup of each compiler's *own best* configuration (across language / parallel / cost /
FP), plus the ``nest-forge`` best-of-all envelope -- all normalized to one **single-compiler,
single-language** baseline the way a user would get with no tuning.

Two baselines, one plot each (both ``role=="timing"``, ``ok==true``, finite-median cells):
  * **gcc C++**  = the ``gcc``  compiler's **c++** cell at ``sequential`` / ``default`` cost / ``default-fp``.
  * **llvm C++** = the ``clang`` compiler's **c++** cell at ``sequential`` / ``default`` cost / ``default-fp``.

A kernel with no such baseline cell (that compiler absent, or its default c++ cell failed to validate)
is dropped from THAT plot -- never divided by. A speedup point is drawn only where BOTH the baseline
and the framework's best cell are correct (validated) -- correctness gating, same rule as the dace plots.

Per kernel, per framework (a "framework" = one compiler, best over all its configs):
    speedup = baseline_median / framework_best_median      (>1 = faster than the single baseline)
``nest-forge`` = min over every compiler+config (the portfolio you get by letting nest-forge search).

Outputs (next to the results; ``--out-dir`` overrides)
  * ``framework_speedup_vs_gcc_cpp.png`` / ``framework_speedup_vs_llvm_cpp.png`` -- per-kernel scatter,
    kernel names on the x-axis (rotated 90deg), one coloured series per framework + nest-forge, log y,
    dashed 1.0 parity line; legend carries each series' geomean.
  * ``framework_speedup.md`` -- per-framework geomean speedup vs each baseline + the full per-kernel table.

This is a READER -- it never runs a kernel. Non-finite / missing timings are "no data" (never plotted,
never a denominator).

Usage::

    python perf/plot_framework_speedup.py --results-dir perf_results/tsvc_full
    python perf/plot_framework_speedup.py --results-dir perf_results/tsvc_full --baseline-lang c++
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: pick the non-interactive backend BEFORE importing pyplot

import matplotlib.pyplot as plt  # noqa: E402 -- must follow matplotlib.use("Agg")

#: The two single-compiler baselines: plot label -> the compiler whose default c++ cell is the baseline.
BASELINES: Dict[str, str] = {"gcc_cpp": "gcc", "llvm_cpp": "clang"}

#: The axis coordinates of a "single-toolchain default" cell: no parallelism, the compiler's own
#: vectorizer cost model, its default FP. Language is the ``--baseline-lang`` (default c++).
DEFAULT_PARALLEL, DEFAULT_COST, DEFAULT_FP = "sequential", "default", "default-fp"

#: Same categorical palette as plot_speedup_matrix / plot_vectorization so compiler hues match across
#: the whole report. ``nest-forge`` (the best-of envelope) gets a distinct dark series.
PALETTE: Dict[str, str] = {
    "gcc": "#4c72b0",
    "clang": "#dd8452",
    "nvhpc": "#55a868",
    "intel": "#c44e52",
    "nest-forge": "#222222"
}
MARKERS: Dict[str, str] = {"gcc": "o", "clang": "s", "nvhpc": "^", "intel": "D", "nest-forge": "*"}


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


def kernel_time_by_nest(cells: List[dict], predicate: Callable[[dict], bool]) -> Optional[float]:
    """A kernel's whole-kernel time for the cells matching ``predicate``, summed over its nests (the
    fastest matching cell per nest). ``None`` unless EVERY nest the kernel timed has a matching cell --
    so a partial match never masquerades as the whole kernel. A single-nest kernel is the usual case."""
    matched = [c for c in cells if predicate(c)]
    if not matched:
        return None
    nests = {c.get("nest", 0) for c in cells}
    total = 0.0
    for n in nests:
        per_nest = [c for c in matched if c.get("nest", 0) == n]
        if not per_nest:
            return None  # this nest has no matching cell -> whole-kernel time undefined for this combo
        total += min(c["median_us"] for c in per_nest)
    return total


def cell_axes(cell: dict) -> Tuple[str, str, str, str, str]:
    """The (compiler, language, parallel, cost, fp) coordinate of a cell."""
    return (str(cell.get("compiler")), str(cell.get("language")), str(cell.get("parallel")),
            str(cell.get("cost_model")), str(cell.get("fp_mode")))


def baseline_time(kernel: dict, compiler: str, lang: str) -> Optional[float]:
    """The single-toolchain-default whole-kernel time for ``compiler`` in ``lang``: the
    <lang> / sequential / default / default-fp cell, summed over nests. ``None`` if that cell is
    absent or unvalidated for any nest."""
    cells = timing_cells(kernel)
    return kernel_time_by_nest(
        cells, lambda c: c.get("compiler") == compiler and c.get("language") == lang and c.get("parallel") ==
        DEFAULT_PARALLEL and c.get("cost_model") == DEFAULT_COST and c.get("fp_mode") == DEFAULT_FP)


def best_time(cells: List[dict], predicate: Callable[[dict], bool]) -> Optional[float]:
    """The fastest whole-kernel time among the distinct config coordinates that satisfy ``predicate``
    (each summed over nests). ``None`` when no matching config validated on every nest."""
    coords = {cell_axes(c) for c in cells if predicate(c)}
    best: Optional[float] = None
    for coord in coords:
        t = kernel_time_by_nest(cells, lambda c, coord=coord: cell_axes(c) == coord)
        if t is not None and t > 0.0 and (best is None or t < best):
            best = t
    return best


def collect(kernels: List[dict], baseline_compiler: str, baseline_lang: str, frameworks: List[str]):
    """{kernel_id: {framework: speedup, 'nest-forge': speedup}} vs one baseline, plus exclusion counts.

    A framework's value is its best cell (over all its configs); nest-forge is the best over ALL
    compilers+configs. Emitted only where the baseline validated; a framework with no validated cell
    on this kernel is simply absent (not zero)."""
    data: Dict[str, Dict[str, float]] = {}
    excl = {"no_baseline": 0, "kernels": 0}
    for kernel in kernels:
        if "skipped" in kernel:
            continue
        excl["kernels"] += 1
        base = baseline_time(kernel, baseline_compiler, baseline_lang)
        if base is None:  # no validated single-toolchain baseline -> speedup undefined, drop kernel
            excl["no_baseline"] += 1
            continue
        cells = timing_cells(kernel)
        row: Dict[str, float] = {}
        for fw in frameworks:
            t = best_time(cells, lambda c, fw=fw: c.get("compiler") == fw)
            if t is not None and t > 0.0:
                row[fw] = base / t
        nf = best_time(cells, lambda c: True)
        if nf is not None and nf > 0.0:
            row["nest-forge"] = base / nf
        if row:
            corpus = str(kernel.get("corpus", "?"))
            data[f"{corpus}/{kernel.get('key', '?')}"] = row
    return data, excl


def sort_key(row: Dict[str, float]) -> float:
    """Order kernels by the nest-forge best speedup (falls back to the max series present)."""
    return row.get("nest-forge") or (max(row.values()) if row else 0.0)


def series_order(frameworks: List[str]) -> List[str]:
    """Frameworks first (data palette order), nest-forge envelope drawn last/on top."""
    return frameworks + ["nest-forge"]


def plot_baseline(label: str, data: Dict[str, Dict[str, float]], frameworks: List[str], out_png: Path):
    """One scatter vs a single baseline; returns {series: geomean}. Kernel names on x (rotated 90deg)."""
    kernels = sorted(data, key=lambda k: sort_key(data[k]), reverse=True)
    series = [s for s in series_order(frameworks) if any(s in data[k] for k in kernels)]
    geos: Dict[str, Optional[float]] = {}
    if not kernels or not series:
        print(f"[framework-speedup] {label}: nothing to plot")
        return geos

    fig_w = max(7.0, min(30.0, len(kernels) * 0.16))
    fig, ax = plt.subplots(figsize=(fig_w, 6.4))
    for s in series:
        xs = [i for i, k in enumerate(kernels) if s in data[k]]
        ys = [data[k][s] for k in kernels if s in data[k]]
        geos[s] = geomean(ys)
        gm = geos[s]
        label = f"{s}  (gm {gm:.2f}x, n={len(ys)})" if gm else s
        if s == "nest-forge":
            # the best-of-all portfolio == the per-compiler envelope; drawn as a line UNDER the
            # framework points so gcc/clang markers (which often sit on it) stay visible.
            ax.plot(xs, ys, "-", color=PALETTE[s], lw=1.1, alpha=0.85, zorder=1, label=label)
        else:
            ax.scatter(xs,
                       ys,
                       s=26,
                       marker=MARKERS.get(s, "o"),
                       c=PALETTE.get(s, "#888888"),
                       alpha=0.8,
                       edgecolors="none",
                       zorder=2,
                       label=label)

    ax.axhline(1.0, linestyle="--", linewidth=1.0, color="#8a8a86")
    ax.set_yscale("log")
    ax.set_ylabel(f"speedup vs {label.replace('_', ' ')} default  (log, >1 = faster)")
    ax.set_title(
        f"Per-kernel speedup of all frameworks vs {label.replace('_', ' ')} default "
        f"(sorted by nest-forge best)",
        fontsize=11)
    ax.set_xticks(range(len(kernels)))
    fs = max(3.0, min(7.0, 620.0 / max(1, len(kernels))))
    ax.set_xticklabels(kernels, rotation=90, fontsize=fs)
    ax.set_xlabel(f"kernel (n={len(kernels)})")
    ax.margins(x=0.005)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out_pdf = out_png.with_suffix(".pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")  # high-dpi raster
    fig.savefig(out_pdf, bbox_inches="tight")  # vector (scales losslessly)
    plt.close(fig)
    print(f"[framework-speedup] wrote {out_png} + {out_pdf} ({len(kernels)} kernels, series={series})")
    return geos


def render_md(results, out_path: Path) -> None:
    """Per-baseline geomean summary + full per-kernel speedup table."""
    lines = ["# Per-framework speedup vs single-compiler single-language baseline", ""]
    for label, base_comp, base_lang, data, geos, frameworks, excl in results:
        pretty = label.replace("_", " ")
        lines += [f"## vs {pretty} default  ({base_comp}, {base_lang}, sequential, default cost, default FP)", ""]
        lines.append(f"Correctness gating: {excl['no_baseline']} of {excl['kernels']} kernels dropped "
                     f"(no validated {base_comp} {base_lang} baseline cell); speedup shown only where "
                     f"both the baseline and the framework's best cell validated.")
        lines.append("")
        series = [s for s in series_order(frameworks) if geos.get(s)]
        if not series:
            lines += ["_no measured kernels_", ""]
            continue
        lines += ["| framework | geomean speedup | kernels | % faster than baseline |", "|---|---|---|---|"]
        for s in series:
            vals = [data[k][s] for k in data if s in data[k]]
            nf = sum(1 for v in vals if v > 1.0)
            pct = f"{100.0 * nf / len(vals):.0f}%" if vals else "n/a"
            lines.append(f"| {s} | {geos[s]:.2f}x | {len(vals)} | {pct} |")
        lines.append("")
        cols = [s for s in series_order(frameworks) if any(s in data[k] for k in data)]
        lines += ["| kernel | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
        for k in sorted(data, key=lambda k: sort_key(data[k]), reverse=True):
            cells = [f"{data[k][s]:.2f}x" if s in data[k] else "—" for s in cols]
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[framework-speedup] wrote {out_path}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Per-framework speedup vs a single-compiler C++ baseline")
    ap.add_argument("--results-dir", required=True, help="dir of nestforge.perf.tsvc_full per-kernel JSON")
    ap.add_argument("--out-dir", default=None, help="override the output dir (default: the results dir)")
    ap.add_argument("--baseline-lang", default="c++", help="single-toolchain baseline language (default: c++)")
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    kernels = load_results(results_dir)

    # frameworks = every compiler that produced a validated timing cell anywhere (data palette order).
    present = {str(c.get("compiler")) for k in kernels if "skipped" not in k for c in timing_cells(k)}
    frameworks = [c for c in ("gcc", "clang", "nvhpc", "intel") if c in present]
    frameworks += sorted(present - set(frameworks))  # any unexpected compiler, appended stably

    results = []
    for label, base_comp in BASELINES.items():
        data, excl = collect(kernels, base_comp, args.baseline_lang, frameworks)
        geos = plot_baseline(label, data, frameworks, out_dir / f"framework_speedup_vs_{label}.png")
        for s in series_order(frameworks):
            if geos.get(s):
                print(f"[framework-speedup]   vs {label}: {s} geomean {geos[s]:.2f}x")
        results.append((label, base_comp, args.baseline_lang, data, geos, frameworks, excl))
    render_md(results, out_dir / "framework_speedup.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
