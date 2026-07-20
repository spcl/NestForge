"""Run the paper's experiments (E1-E5) and write their tables as JSON.

One entry point for every experiment, because the drivers must agree on the knobs that make their numbers
comparable -- ``preset``, ``seed``, ``reps`` and the backend set feed all five from here. E2 in particular
DIVIDES an E1/E3 time by a baseline measured through another path, so a run that swept those axes at one
problem size and measured the baseline at another would produce a ratio that means nothing.

Bounded by default (few kernels, few granularity rungs) so a smoke run finishes in CI; widen with the
flags for a corpus run on a cluster. Every experiment writes ``<out>/<name>.json`` holding both its raw
rows and its read-off table, and failures are recorded in those rows rather than aborting the run -- a
sweep that dies on kernel 3 of 90 must not lose kernels 1 and 2.

  python scripts/run_experiments.py --out results/ --experiments e1,e3
  python scripts/run_experiments.py --out results/ --kernels 20 --granularity-points 5 --reps 15
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Dict, List

# Python puts THIS file's directory on sys.path, not the repo root, so a plain
# `python scripts/run_experiments.py` from a checkout cannot import nestforge unless it happens to be
# pip-installed. This is the cluster entry point -- it must run from a bare clone.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nestforge import tsvc  # noqa: E402
from nestforge.arena import discover_compilers  # noqa: E402
from nestforge.experiment_e1 import best_granularity_per_backend, run_e1  # noqa: E402
from nestforge.experiment_e2 import run_e2, skipped_lanes, speedup_table  # noqa: E402
from nestforge.experiment_e3 import best_unit_per_backend, granularity_curve, run_e3  # noqa: E402
from nestforge.experiment_e4 import cost_quality_table, run_e4, savings_vs_oracle  # noqa: E402
from nestforge.experiment_e5 import non_affine_findings, run_e5  # noqa: E402

EXPERIMENTS = ("e1", "e2", "e3", "e4", "e5")


def to_json_value(value):
    """Tables are keyed by tuples (kernel, backend), which JSON cannot express -- join them so the file
    stays readable instead of silently losing the backend half of the key.

    Deliberately NOT ``nestforge.perf.harness.jsonable``, and deliberately not named the same: that one
    maps non-finite floats to ``null``, which makes a failed measurement indistinguishable from one that
    was never attempted. These tables keep "inf"/"nan" visible as strings, and carry tuple keys and
    dataclass rows that the harness helper does not handle.
    """
    if isinstance(value, dict):
        return {(" | ".join(k) if isinstance(k, tuple) else str(k)): to_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_value(v) for v in value]
    if dataclasses.is_dataclass(value):
        return to_json_value(dataclasses.asdict(value))
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return str(value)  # inf/nan are not JSON; keep them visible rather than coercing to null
    return value


def write(out_dir: Path, name: str, rows, tables: Dict) -> Path:
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps({"rows": to_json_value(rows), **to_json_value(tables)}, indent=2, sort_keys=True))
    return path


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="directory for the JSON tables and build trees")
    ap.add_argument("--experiments", default=",".join(EXPERIMENTS), help=f"comma-separated subset of {EXPERIMENTS}")
    ap.add_argument("--kernels", type=int, default=1, help="how many corpus kernels to sweep (0 = all)")
    ap.add_argument("--granularity-points", type=int, default=2, help="granularity rungs per kernel")
    ap.add_argument("--reps", type=int, default=5, help="timed repetitions per measurement")
    ap.add_argument("--preset", default="S", help="problem-size preset, shared by every experiment")
    ap.add_argument("--seed", type=int, default=0, help="input seed, shared by every experiment")
    ap.add_argument("--unit", default="map", help="offloading unit for the fixed-unit experiments")
    args = ap.parse_args(argv)

    names = [e.strip().lower() for e in args.experiments.split(",") if e.strip()]
    unknown = sorted(set(names) - set(EXPERIMENTS))
    if unknown:
        ap.error(f"unknown experiments {unknown}; known: {list(EXPERIMENTS)}")

    args.out.mkdir(parents=True, exist_ok=True)
    backends = discover_compilers()
    if not backends:
        print("no compiler found by discover_compilers(); nothing can be measured", file=sys.stderr)
        return 1

    kernels = list(tsvc.iter_tsvc_kernels())
    if args.kernels:
        kernels = kernels[:args.kernels]
    shared = dict(reps=args.reps, preset=args.preset, seed=args.seed, backends=backends)
    print(f"{len(kernels)} kernel(s) x {len(backends)} backend(s): {', '.join(sorted(backends))}")

    # E1/E3 cells feed E2, so they are kept even when E2 alone was requested -- E2 without them would have
    # no search side to divide by, and reporting the baseline alone is not the experiment.
    cells = []
    fed_by: List[str] = []  # which axes fed E2's search side -- see the provenance note where e2 is written
    if "e1" in names:
        rows = run_e1(kernels,
                      args.out / "e1",
                      unit=args.unit,
                      max_granularity_points=args.granularity_points,
                      **shared)
        cells += rows
        fed_by.append("e1")
        print("e1 ->", write(args.out, "e1", rows, {"best_granularity": best_granularity_per_backend(rows)}))
    if "e3" in names:
        rows = run_e3(kernels, args.out / "e3", **shared)
        cells += rows
        fed_by.append("e3")
        print("e3 ->",
              write(args.out, "e3", rows, {
                  "curve": granularity_curve(rows),
                  "best_unit": best_unit_per_backend(rows)
              }))
    if "e2" in names:
        if not cells:  # E2 needs a search side; measure the E1 axis for it rather than reporting half a table
            fed_by.append("e1-fallback")
            cells = run_e1(kernels,
                           args.out / "e2-search",
                           unit=args.unit,
                           max_granularity_points=args.granularity_points,
                           **shared)
        rows = run_e2(kernels,
                      cells,
                      args.out / "e2",
                      preset=args.preset,
                      reps=args.reps,
                      seed=args.seed,
                      backends=backends)
        # Record WHICH axes fed the search side: search_best takes a min over whatever cells it is
        # handed, so `--experiments e2` divides by a smaller candidate set than `e1,e2,e3` does.
        # Without this field two runs report different speedups with nothing to explain the gap.
        print(
            "e2 ->",
            write(args.out, "e2", rows, {
                "speedup": speedup_table(rows),
                "skipped": skipped_lanes(rows),
                "search_cells_from": fed_by,
            }))
    if "e4" in names:
        rows = run_e4(kernels,
                      args.out / "e4",
                      unit=args.unit,
                      max_granularity_points=args.granularity_points,
                      **shared)
        print(
            "e4 ->",
            write(args.out, "e4", rows, {
                "cost_quality": cost_quality_table(rows),
                "savings": savings_vs_oracle(rows)
            }))
    if "e5" in names:
        rows = run_e5(kernels,
                      args.out / "e5",
                      unit=args.unit,
                      max_granularity_points=args.granularity_points,
                      **shared)
        print("e5 ->", write(args.out, "e5", rows, {"non_affine_speedup": non_affine_findings(rows)}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
