"""Plot the runtime call-overhead job output (``nestforge.perf.calloverhead``).

Reads the per-kernel JSON that :mod:`nestforge.perf.calloverhead` writes (one ``<corpus>_<key>.json`` per
kernel, plus a ``tables.md`` that is skipped) and renders, WITHOUT rebuilding anything:

  * a grouped bar chart per kernel of ``inline_us`` / ``external_lto_us`` / ``external_us`` (per-call
    microseconds), sorted by ``call_overhead`` (external / inline) descending, and
  * the geomean ``call_overhead`` (external / inline) and ``lto_overhead`` (external-lto / inline) in the
    title -- ~1.0 means the external ``.a`` call is as cheap as inlining (and LTO recovered it).

Two files land next to the results (or ``--out`` overrides the PNG): ``calloverhead.png`` and
``calloverhead.csv`` (``key, inline_us, external_lto_us, external_us, call_overhead, lto_overhead``).

Non-finite / missing timings (``null`` from the driver, a variant a compiler could not build) are treated
as "no data": never plotted, never divided by. This is a READER -- it never runs a kernel.

Usage::

    python perf/plot_calloverhead.py --results-dir perf_results/calloverhead
    python perf/plot_calloverhead.py --results-dir perf_results/calloverhead --out /tmp/calloverhead.png
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: pick the non-interactive backend BEFORE importing pyplot

import matplotlib.pyplot as plt  # noqa: E402 -- must follow matplotlib.use("Agg")

#: key, inline us, external-lto us, external us, call overhead, lto overhead. Any numeric may be None.
CallRow = Tuple[str, Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]


def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def geomean(xs: List[float]) -> Optional[float]:
    vals = [x for x in xs if finite(x) and x > 0.0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else None


def load_results(results_dir: Path) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "tables.md":
            continue
        try:
            rows.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError, ValueError):
            continue
    return rows


def call_row(record: dict) -> Optional[CallRow]:
    """Reduce a raw record to a :data:`CallRow`, or None for a skipped / un-timed kernel. Non-finite
    numerics become None; ratios are taken from the record when finite, else recomputed."""
    if "skipped" in record or record.get("inline_us") is None:
        return None
    key = str(record.get("key", "?"))
    inl = record.get("inline_us")
    extl = record.get("external_lto_us")
    ext = record.get("external_us")
    inl = inl if finite(inl) else None
    extl = extl if finite(extl) else None
    ext = ext if finite(ext) else None
    co = record.get("call_overhead")
    lo = record.get("lto_overhead")
    co = co if finite(co) else ((ext / inl) if (inl and ext is not None) else None)
    lo = lo if finite(lo) else ((extl / inl) if (inl and extl is not None) else None)
    return key, inl, extl, ext, co, lo


def write_csv(path: Path, rows: List[CallRow]) -> None:

    def cell(x: Optional[float]) -> str:
        return "" if x is None else repr(float(x))

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "inline_us", "external_lto_us", "external_us", "call_overhead", "lto_overhead"])
        for key, inl, extl, ext, co, lo in rows:
            writer.writerow([key, cell(inl), cell(extl), cell(ext), cell(co), cell(lo)])


def plot_calloverhead(rows: List[CallRow], geo_call: Optional[float], geo_lto: Optional[float],
                      out_png: Path) -> None:
    """Grouped inline / external-lto / external bars per kernel, sorted by call overhead desc. Only kernels
    with a finite inline time are plotted; the two geomeans go in the title."""
    plottable = [(k, i, el, e, c) for (k, i, el, e, c, _l) in rows if i is not None]
    plottable.sort(key=lambda t: (t[4] if t[4] is not None else float("-inf")), reverse=True)

    def txt(g):
        return "n/a" if g is None else f"{g:.3f}x"

    title = (f"Runtime call overhead: geomean external/inline = {txt(geo_call)}, "
             f"external-lto/inline = {txt(geo_lto)} over {len(plottable)} kernels")

    fig, ax = plt.subplots(figsize=(max(8.0, len(plottable) * 0.4), 5.0))
    if not plottable:
        ax.text(0.5, 0.5, "no kernels with a finite inline timing", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        keys = [k for (k, *_rest) in plottable]
        inls = [(i if i is not None else 0.0) for (_k, i, _el, _e, _c) in plottable]
        extls = [(el if el is not None else 0.0) for (_k, _i, el, _e, _c) in plottable]
        exts = [(e if e is not None else 0.0) for (_k, _i, _el, e, _c) in plottable]
        xs = list(range(len(plottable)))
        w = 0.28
        ax.bar([x - w for x in xs], inls, width=w, label="inline (#include)", color="#4c72b0")
        ax.bar(xs, extls, width=w, label="external-lto .a", color="#55a868")
        ax.bar([x + w for x in xs], exts, width=w, label="external .a (no LTO)", color="#dd8452")
        ax.set_xticks(xs)
        ax.set_xticklabels(keys, rotation=90, fontsize=6)
        ax.set_ylabel("per-call time (us)")
        ax.legend(loc="upper right")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Plot the runtime call-overhead job output (reader, no builds)")
    ap.add_argument("--results-dir", required=True, help="dir of nestforge.perf.calloverhead per-kernel JSON")
    ap.add_argument("--out", default=None, help="override the PNG path (default: <results-dir>/calloverhead.png)")
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir)
    rows = [row for row in (call_row(r) for r in load_results(results_dir)) if row is not None]
    geo_call = geomean([r[4] for r in rows])
    geo_lto = geomean([r[5] for r in rows])

    out_png = Path(args.out) if args.out else results_dir / "calloverhead.png"
    out_csv = results_dir / "calloverhead.csv"
    plot_calloverhead(rows, geo_call, geo_lto, out_png)
    write_csv(out_csv, rows)
    print(f"[plot-calloverhead] {len(rows)} kernels -> {out_png} + {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
