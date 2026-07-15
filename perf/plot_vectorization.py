"""Single-core vectorization report: on ONE core, how much each compiler's auto-vectorizer helps per
kernel, and — the headline — where **one compiler's vectorization beats another's by a lot**.

Motivation
----------
Averaged, parallel, best-of-everything views (``plot_winners`` / ``plot_speedup_matrix``) hide the
single-thread vectorizer story: gcc and clang routinely diverge by 1.5–3x on the SAME C source at
``-O3 -march=native`` purely because one backend vectorizes a loop the other leaves scalar (or
mis-vectorizes it and regresses). This reader isolates that, holding everything but the compiler
fixed: the **sequential** lane, one language (``--lang``, default ``c``), one FP mode
(``--fp``, default ``default-fp``), the compiler's ``default`` (vectorized) cost model vs its
``no-vec`` (scalar floor) cost model.

Per kernel, per compiler (summed over the kernel's nests, like the other readers):
  * ``vec``    = sequential / default-cost / <fp> / <lang>   -- the compiler's own vectorized time.
  * ``scalar`` = sequential / no-vec / <fp> / <lang>          -- the same source, vectorizer OFF.
  * ``gain``   = scalar / vec                                 -- >1 the vectorizer helped, <1 it HURT.

Two cross-compiler quantities that "capture where one beats the other":
  * ``gap``      = max(vec over compilers) / min(vec over compilers) -- the single-core spread. A big
    gap on a kernel means one compiler's single-thread codegen is much faster than another's.
  * the *fastest* and *slowest* compiler at that vectorized single-core time.

Outputs (next to the results; ``--out-dir`` overrides)
  * ``vectorization_single_core.md`` -- per kernel: each compiler's vec/scalar/gain, the fastest
    compiler, and the cross-compiler gap; sorted by gap DESC so the big divergences float to the top.
    Plus a per-compiler geomean vec-gain summary and the list of kernels over ``--gap-threshold``.
  * ``vectorization_gain.png``       -- per-compiler geomean single-core vectorization gain (bars).
  * ``vectorization_gap.png``        -- per-kernel cross-compiler single-core gap (sorted, coloured by
    the winning compiler, with the ``--gap-threshold`` line): the "one compiler wins big" chart.

This is a READER -- it never runs a kernel. Non-finite / missing timings are "no data": never plotted,
never a denominator. A kernel that never timed the sequential lane for >=2 compilers contributes no gap.

Usage::

    python perf/plot_vectorization.py --results-dir perf_results/tsvc_full
    python perf/plot_vectorization.py --results-dir perf_results/tsvc_full --lang c --gap-threshold 1.5
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless: pick the non-interactive backend BEFORE importing pyplot

import matplotlib.pyplot as plt  # noqa: E402 -- must follow matplotlib.use("Agg")

from plot_common import finite, geomean, load_results  # noqa: E402

#: The single-core coordinate this reader lives on. Vectorized = the compiler's own cost model; the
#: scalar floor = ``no-vec``. Both on the ``sequential`` (one-core) lane so the only difference that
#: moves the number is the compiler's auto-vectorizer.
SEQ, VEC_COST, SCALAR_COST = "sequential", "default", "no-vec"

#: Same palette as plot_speedup_matrix so the winning-compiler colours match across the report.
PALETTE: Dict[str, str] = {"gcc": "#4c72b0", "clang": "#dd8452", "nvhpc": "#55a868", "intel": "#c44e52"}


def timing_cells(kernel: dict) -> List[dict]:
    """The validated, finite-median lane-3 timing cells of one kernel."""
    return [
        c for c in (kernel.get("cells") or [])
        if c.get("role") == "timing" and c.get("ok") is True and finite(c.get("median_us"))
    ]


def time_by_nest(cells: List[dict], predicate) -> Optional[float]:
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


@dataclass
class CompilerSC:
    """One compiler's single-core numbers on one kernel."""
    vec: Optional[float] = None  # vectorized (default cost)
    scalar: Optional[float] = None  # scalar floor (no-vec)
    gain: Optional[float] = None  # scalar / vec  (>1 = vectorizer helped)


@dataclass
class KernelSC:
    """One kernel's single-core picture: per-compiler numbers plus the cross-compiler gap."""
    corpus: str
    key: str
    by_compiler: Dict[str, CompilerSC] = field(default_factory=dict)
    fastest: Optional[str] = None  # compiler with the min vectorized single-core time
    slowest: Optional[str] = None
    gap: Optional[float] = None  # slowest_vec / fastest_vec  (>=1)


def single_core(kernel: dict, lang: str, fp: str) -> Optional[KernelSC]:
    """The single-core vec/scalar/gain per compiler for one kernel, plus the cross-compiler gap. ``None``
    when no compiler produced a vectorized sequential time (nothing to say about this kernel)."""
    if "skipped" in kernel:
        return None
    cells = timing_cells(kernel)
    seq = [c for c in cells if c.get("parallel") == SEQ and c.get("language") == lang and c.get("fp_mode") == fp]
    if not seq:
        return None
    compilers = sorted({str(c.get("compiler")) for c in seq})
    sc = KernelSC(corpus=str(kernel.get("corpus", "?")), key=str(kernel.get("key", "?")))
    for comp in compilers:
        vec = time_by_nest(seq, lambda c, comp=comp: c.get("compiler") == comp and c.get("cost_model") == VEC_COST)
        scalar = time_by_nest(seq,
                              lambda c, comp=comp: c.get("compiler") == comp and c.get("cost_model") == SCALAR_COST)
        gain = (scalar / vec) if (finite(vec) and finite(scalar) and vec > 0.0) else None
        sc.by_compiler[comp] = CompilerSC(vec=vec, scalar=scalar, gain=gain)
    vecs = {comp: cs.vec for comp, cs in sc.by_compiler.items() if finite(cs.vec) and cs.vec > 0.0}
    if not vecs:
        return None
    sc.fastest = min(vecs, key=lambda c: vecs[c])
    sc.slowest = max(vecs, key=lambda c: vecs[c])
    sc.gap = vecs[sc.slowest] / vecs[sc.fastest]
    return sc


def num(x: Optional[float], suffix: str = "") -> str:
    """``—`` when absent, else 2dp + optional suffix."""
    return "—" if not finite(x) else f"{x:.2f}{suffix}"


def render_md(kernels: List[KernelSC], compilers: List[str], gap_threshold: float, lang: str, fp: str) -> str:
    """Per-kernel table (sorted by cross-compiler gap DESC) + per-compiler geomean gain + the over-threshold
    divergence list. ``compilers`` is the union column order."""
    lines = [
        f"# Single-core vectorization report ({lang}, {fp}, sequential lane)",
        "",
        "Everything but the **compiler** is held fixed (one core, one language, one FP mode). "
        "`vec` = the compiler's own vectorized time (µs); `gain` = `no-vec / vec` (>1 = its vectorizer "
        "helped, <1 = it HURT). `gap` = slowest-vec / fastest-vec across compilers — the single-core "
        "spread; a big gap is a kernel where one compiler's vectorization beats another's by a lot.",
        "",
    ]
    # per-compiler geomean vectorization gain (how much each vectorizer helps on average, single core)
    lines += ["## Per-compiler single-core vectorization gain (geomean `no-vec / vec`)", ""]
    lines += ["| compiler | geomean gain | n kernels |", "|---|---|---|"]
    for comp in compilers:
        gains = [k.by_compiler[comp].gain for k in kernels if comp in k.by_compiler]
        g = geomean([x for x in gains if finite(x)])
        n = sum(1 for x in gains if finite(x))
        lines.append(f"| {comp} | {num(g, 'x')} | {n} |")
    lines += [""]
    # the headline: kernels where the compilers diverge most on single core
    diverged = [k for k in kernels if finite(k.gap) and k.gap >= gap_threshold]
    diverged.sort(key=lambda k: k.gap, reverse=True)
    lines += [f"## Single-core divergences ≥ {gap_threshold:.2f}x  (one compiler beats another by a lot)", ""]
    if not diverged:
        lines += [f"_none: no kernel had a cross-compiler single-core gap ≥ {gap_threshold:.2f}x_", ""]
    else:
        lines += ["| corpus | key | fastest | slowest | gap |", "|---|---|---|---|---|"]
        for k in diverged:
            lines.append(f"| {k.corpus} | {k.key} | {k.fastest} | {k.slowest} | {num(k.gap, 'x')} |")
        lines += [""]
    # full per-kernel table, sorted by gap DESC
    lines += ["## All kernels (sorted by cross-compiler single-core gap, DESC)", ""]
    head = "| corpus | key | " + " | ".join(f"{c} vec / gain" for c in compilers) + " | fastest | gap |"
    lines += [head, "|" + "---|" * (len(compilers) + 4)]
    for k in sorted(kernels, key=lambda k: (-(k.gap if finite(k.gap) else 0.0), k.corpus, k.key)):
        cols = []
        for comp in compilers:
            cs = k.by_compiler.get(comp)
            cols.append("—" if cs is None else f"{num(cs.vec)} / {num(cs.gain, 'x')}")
        lines.append(f"| {k.corpus} | {k.key} | " + " | ".join(cols) + f" | {k.fastest or '—'} | {num(k.gap, 'x')} |")
    return "\n".join(lines) + "\n"


def plot_gain(kernels: List[KernelSC], compilers: List[str], out_png: Path) -> None:
    """Bars: per-compiler geomean single-core vectorization gain (`no-vec / vec`). A dashed 1.0 line marks
    'vectorization made no difference'; a bar below it means that compiler's vectorizer HURT on average."""
    heights, labels, colors = [], [], []
    for comp in compilers:
        gains = [k.by_compiler[comp].gain for k in kernels if comp in k.by_compiler]
        g = geomean([x for x in gains if finite(x)])
        if g is None:
            continue
        labels.append(comp)
        heights.append(g)
        colors.append(PALETTE.get(comp, "#888888"))
    fig, ax = plt.subplots(figsize=(max(5.0, len(labels) * 1.4), 5.0))
    if not labels:
        ax.text(0.5, 0.5, "no single-core vec/no-vec pairs", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        xs = list(range(len(labels)))
        ax.bar(xs, heights, color=colors, width=0.55)
        ax.axhline(1.0, linestyle="--", linewidth=0.8, color="#888888")
        for x, h in zip(xs, heights):
            ax.text(x, h, f"{h:.2f}x", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_ylabel("geomean single-core gain (no-vec / vec)")
        ax.set_title("Single-core vectorization gain per compiler (>1 = vectorizer helps)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_gap(kernels: List[KernelSC], gap_threshold: float, out_png: Path) -> None:
    """Per-kernel cross-compiler single-core gap (slowest-vec / fastest-vec), sorted ascending, each bar
    coloured by the FASTEST compiler on that kernel. The ``gap_threshold`` line marks 'significant
    divergence' -- everything to the right is a kernel where one compiler wins single core by a lot."""
    pts = [k for k in kernels if finite(k.gap)]
    pts.sort(key=lambda k: k.gap)
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    if not pts:
        ax.text(0.5, 0.5, "no cross-compiler single-core gaps", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        return
    xs = list(range(len(pts)))
    ys = [k.gap for k in pts]
    cs = [PALETTE.get(k.fastest, "#888888") for k in pts]
    ax.bar(xs, ys, color=cs, width=0.9)
    ax.axhline(gap_threshold, linestyle="--", linewidth=0.9, color="#444444")
    ax.text(0, gap_threshold, f" {gap_threshold:.2f}x threshold", va="bottom", ha="left", fontsize=8, color="#444444")
    ax.set_ylim(bottom=1.0)
    ax.set_xlabel(f"kernel (sorted by single-core gap), n={len(pts)}")
    ax.set_ylabel("slowest-vec / fastest-vec (single core)")
    over = sum(1 for k in pts if k.gap >= gap_threshold)
    seen = {k.fastest for k in pts}
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=PALETTE.get(c, "#888888"), label=c) for c in sorted(seen)
    ]
    ax.legend(handles=handles, title="fastest compiler", fontsize=8, loc="upper left")
    ax.set_title(f"Per-kernel single-core compiler gap  |  {over}/{len(pts)} kernels ≥ {gap_threshold:.2f}x",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def print_summary(kernels: List[KernelSC], compilers: List[str], gap_threshold: float) -> None:
    """Short stdout summary: per-compiler geomean vec-gain and the count / top single-core divergences."""
    for comp in compilers:
        gains = [k.by_compiler[comp].gain for k in kernels if comp in k.by_compiler]
        g = geomean([x for x in gains if finite(x)])
        print(f"[plot-vectorization]   {comp}: single-core geomean vec-gain {num(g, 'x')} "
              f"(n={sum(1 for x in gains if finite(x))})")
    diverged = sorted((k for k in kernels if finite(k.gap) and k.gap >= gap_threshold),
                      key=lambda k: k.gap,
                      reverse=True)
    print(f"[plot-vectorization] {len(diverged)} kernels with a single-core gap ≥ {gap_threshold:.2f}x")
    for k in diverged[:5]:
        print(f"[plot-vectorization]   {k.corpus}/{k.key}: {k.gap:.2f}x  (fastest {k.fastest}, slowest {k.slowest})")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Single-core vectorization report (reader, no builds)")
    ap.add_argument("--results-dir", required=True, help="dir of nestforge.perf.tsvc_full per-kernel JSON")
    ap.add_argument("--out-dir", default=None, help="override the output dir (default: the results dir)")
    ap.add_argument("--lang", default="c", help="language to hold fixed (default: c -- the canonical source)")
    ap.add_argument("--fp", default="default-fp", help="FP mode to hold fixed (default: default-fp)")
    ap.add_argument("--gap-threshold",
                    type=float,
                    default=1.3,
                    help="a cross-compiler single-core gap at/above this is a 'divergence' (default 1.3x)")
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_results(results_dir)
    kernels = [sc for k in raw if (sc := single_core(k, args.lang, args.fp)) is not None]
    compilers = sorted({c for k in kernels for c in k.by_compiler})

    md = out_dir / "vectorization_single_core.md"
    md.write_text(render_md(kernels, compilers, args.gap_threshold, args.lang, args.fp))
    gain_png, gap_png = out_dir / "vectorization_gain.png", out_dir / "vectorization_gap.png"
    plot_gain(kernels, compilers, gain_png)
    plot_gap(kernels, args.gap_threshold, gap_png)

    print_summary(kernels, compilers, args.gap_threshold)
    print(f"[plot-vectorization] wrote {md}, {gain_png}, {gap_png} "
          f"({len(kernels)} kernels, compilers: {', '.join(compilers) or 'none'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
