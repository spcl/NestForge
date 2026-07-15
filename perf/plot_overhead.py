"""Plot the static-lib COMPILE-overhead job output (``nestforge.perf.staticlib_overhead``).

Reads the per-kernel JSON that :mod:`nestforge.perf.staticlib_overhead` writes (one ``<key>.json`` per
kernel, plus a ``tables.md`` that is skipped) and renders, WITHOUT recompiling anything:

  * a grouped bar chart per kernel of ``monolithic_ms`` vs ``external_ms``, sorted by ``overhead_ratio``
    (``external / monolithic``, the static-lib assembly compile-time overhead) descending, and
  * the geomean ``overhead_ratio`` across kernels, shown in the chart title.

Two files land next to the results (or ``--out`` overrides the PNG path): ``overhead.png`` and
``overhead.csv`` (``key, monolithic_ms, external_ms, overhead_ratio``).

Non-finite / missing timings (a median sanitized to ``null`` by the driver, an ``Infinity`` a compiler
never produced) are treated as "no data": never plotted, never divided by. This is a READER -- it never
runs a kernel.

Usage::

    python perf/plot_overhead.py --results-dir perf_results/staticlib
    python perf/plot_overhead.py --results-dir perf_results/staticlib --out /tmp/overhead.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: pick the non-interactive backend BEFORE importing pyplot

import matplotlib.pyplot as plt  # noqa: E402 -- must follow matplotlib.use("Agg")

from plot_common import finite, geomean, load_results  # noqa: E402

#: One completed kernel: key, monolithic ms, external ms, overhead ratio. Any numeric may be ``None``
#: (non-finite / missing in the source JSON).
OverheadRow = Tuple[str, Optional[float], Optional[float], Optional[float]]


def overhead_row(record: dict) -> Optional[OverheadRow]:
    """Reduce one raw JSON record to an :data:`OverheadRow`, or ``None`` if it is not a completed kernel
    (a ``skipped`` record, or one missing ``monolithic_ms``). Numerics that are non-finite become
    ``None``; the ratio is taken from the record when finite, else recomputed from the two times."""
    if "skipped" in record or "monolithic_ms" not in record:
        return None
    key = str(record.get("key", "?"))
    mono = record.get("monolithic_ms")
    ext = record.get("external_ms")
    ratio = record.get("overhead_ratio")
    mono = mono if finite(mono) else None
    ext = ext if finite(ext) else None
    if not finite(ratio):
        # recompute only when both operands are usable and the denominator is nonzero
        ratio = (ext / mono) if (mono and ext is not None) else None
    return key, mono, ext, ratio


def write_csv(path: Path, rows: List[OverheadRow]) -> None:
    """``key, monolithic_ms, external_ms, overhead_ratio`` -- one line per completed kernel; a ``None``
    numeric is written as an empty cell."""

    def cell(x: Optional[float]) -> str:
        return "" if x is None else repr(float(x))

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "monolithic_ms", "external_ms", "overhead_ratio"])
        for key, mono, ext, ratio in rows:
            writer.writerow([key, cell(mono), cell(ext), cell(ratio)])


def plot_overhead(rows: List[OverheadRow], geo: Optional[float], out_png: Path) -> None:
    """Grouped ``monolithic`` vs ``external`` bars per kernel, sorted by overhead ratio desc. Only kernels
    with BOTH times finite are plotted; the geomean overhead goes in the title."""
    plottable = [(k, m, e, r) for (k, m, e, r) in rows if m is not None and e is not None]
    # ratio desc; a plottable kernel without a ratio sorts last (treated as -inf)
    plottable.sort(key=lambda t: (t[3] if t[3] is not None else float("-inf")), reverse=True)

    geo_txt = "n/a" if geo is None else f"{geo:.3f}x"
    title = f"Static-lib compile overhead: geomean external/monolithic = {geo_txt} over {len(plottable)} kernels"

    fig, ax = plt.subplots(figsize=(max(8.0, len(plottable) * 0.34), 5.0))
    if not plottable:
        ax.text(0.5,
                0.5,
                "no kernels with finite monolithic + external timings",
                ha="center",
                va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
    else:
        keys = [k for (k, _m, _e, _r) in plottable]
        monos = [m for (_k, m, _e, _r) in plottable]
        exts = [e for (_k, _m, e, _r) in plottable]
        xs = list(range(len(plottable)))
        width = 0.4
        ax.bar([x - width / 2 for x in xs], monos, width=width, label="monolithic (single TU)", color="#4c72b0")
        ax.bar([x + width / 2 for x in xs], exts, width=width, label="external .a (per-kernel link)", color="#dd8452")
        ax.set_xticks(xs)
        ax.set_xticklabels(keys, rotation=90, fontsize=6)
        ax.set_ylabel("compile time (ms)")
        ax.legend(loc="upper right")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Plot the static-lib compile-overhead job output (reader, no builds)")
    ap.add_argument("--results-dir", required=True, help="dir of nestforge.perf.staticlib_overhead per-kernel JSON")
    ap.add_argument("--out", default=None, help="override the PNG path (default: <results-dir>/overhead.png)")
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir)
    rows = [row for row in (overhead_row(r) for r in load_results(results_dir)) if row is not None]
    geo = geomean([r for (_k, _m, _e, r) in rows])

    out_png = Path(args.out) if args.out else results_dir / "overhead.png"
    out_csv = results_dir / "overhead.csv"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_csv, rows)
    plot_overhead(rows, geo, out_png)

    geo_txt = "n/a" if geo is None else f"{geo:.3f}x"
    print(f"[plot-overhead] {len(rows)} kernels; geomean overhead {geo_txt}")
    print(f"[plot-overhead] wrote {out_png} and {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
