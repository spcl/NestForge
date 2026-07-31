# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Sweep the hpcagent_bench ``foundation`` track through the arena, one JSON per kernel.

Every kernel is lowered to its nests and each nest is measured across the arena's compiler x FP-level
matrix, deduped by built artifact (:mod:`nestforge.dedup`), and validated against the numpy oracle before
any timing counts. Ranks self-partition via ``SLURM_PROCID``/``SLURM_NTASKS`` (:func:`harness.my_slice`),
so ``srun`` needs no extra coordination and a rank that dies takes only its own slice with it.

A kernel that cannot lower, emit or build is recorded with its reason and the sweep continues -- across
corpus kernels some will not translate, and losing the run to one of them is the failure mode this
exists to avoid. Results are written per kernel, so a killed sweep keeps everything already finished.

The measurement itself forks (:func:`nestforge.isolation.run_isolated`), so a segfaulting candidate costs
one cell.

This drives the ARENA axes (compiler x fp-level x cost-model x veclib). The richer axis matrix that
``perf/tsvc_full.py`` sweeps -- language, parallelism, opt-mode -- is a separate driver over the TSVC
corpora; pointing it at this track needs its corpus to become a parameter.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from nestforge.corpus import CorpusKernel, iter_dace_kernels, set_precision_fp64
from nestforge.isolation import run_isolated
from nestforge.perf.harness import jsonable, my_slice, rank_and_size

#: Problem-size scale. ``S`` fits cache and is the dev-box default; ``L``/``XL`` are the memory-bound
#: sizes a real result needs. Named presets, not a raw integer -- the manifest already sizes every symbol
#: per preset, and a uniform guess silently measures something else (it also does not name the symbols).
PRESETS = ("S", "M", "L", "XL")
DEFAULT_PRESET = "M"


def kernel_sizes(kernel: CorpusKernel, preset: str) -> Dict[str, int]:
    """The manifest's own value for every free symbol at ``preset``."""
    parameters = kernel.spec.parameters
    if preset not in parameters:
        raise KeyError(f"{kernel.short_name} has no {preset!r} preset (has: {sorted(parameters)})")
    return dict(parameters[preset])


def sweep_kernel(kernel: CorpusKernel, out_dir: Path, preset: str, reps: int, seed: int) -> Dict[str, object]:
    """Measure one corpus kernel; never raises -- a failure is a recorded row.

    Lowering runs dace passes that can hang or die natively, so the WHOLE thing is forked, not just the
    measurement inside it.
    """

    def work() -> Dict[str, object]:
        from nestforge.run import optimize_program  # imported in the child: dace codegen stays out of the parent
        set_precision_fp64()
        report = optimize_program(kernel.program().to_sdfg(simplify=True),
                                  sizes=kernel_sizes(kernel, preset),
                                  out_dir=out_dir / "build",
                                  reps=reps,
                                  seed=seed)
        return {
            "nests": [{
                "name": n.name,
                "error": n.error,
                "winners": {
                    mode: jsonable(dataclasses.asdict(cell))
                    for mode, cell in (n.arena.winners.items() if n.arena else ())
                },
                "collapsed": list(n.arena.collapsed) if n.arena else [],
            } for n in report.nests]
        }

    started = time.perf_counter()
    res = run_isolated(work, timeout=1800.0)
    return {
        "kernel": kernel.short_name,
        "preset": preset,
        "reps": reps,
        "seconds": time.perf_counter() - started,
        **res,
    }


def run_sweep(out_dir: Path, preset: str, reps: int, seed: int, limit: Optional[int],
              only: Optional[Sequence[str]]) -> int:
    kernels: List[CorpusKernel] = sorted(iter_dace_kernels("foundation"), key=lambda k: k.short_name)
    if only:
        wanted = set(only)
        kernels = [k for k in kernels if k.short_name in wanted or k.short_name.rsplit("/", 1)[-1] in wanted]
    if limit:
        kernels = kernels[:limit]
    procid, ntasks = rank_and_size()
    mine = my_slice(kernels, procid, ntasks)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"rank {procid}/{ntasks}: {len(mine)} of {len(kernels)} foundation kernels", flush=True)
    for kernel in mine:
        name = kernel.short_name.replace("/", "__")
        row = sweep_kernel(kernel, out_dir / name, preset, reps, seed)
        (out_dir / f"{name}.json").write_text(json.dumps(row, indent=1))
        nests = row.get("nests")
        ok = sum(1 for n in nests if not n["error"]) if isinstance(nests, list) else 0
        print(
            f"  {kernel.short_name}: {ok} nest(s) measured, {row['seconds']:.1f}s"
            f"{'  ERROR: ' + str(row['error'])[:80] if 'error' in row else ''}",
            flush=True)
    return 0


def merge_tables(out_dir: Path) -> int:
    """Fold every rank's per-kernel JSON into one summary on stdout."""
    # A rank killed mid-write (OOM, walltime) leaves a truncated JSON. Parsed unguarded, that one file
    # raised and lost the whole sweep's results at the very last step; the run costs hours to repeat.
    # Named on stdout rather than dropped, so a short table is never mistaken for a complete one.
    rows, unreadable = [], []
    for p in sorted(out_dir.glob("*.json")):
        if p.name == "summary.json":
            continue
        try:
            rows.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError) as e:
            unreadable.append(f"{p.name}: {type(e).__name__}: {str(e)[:80]}")
    measured = [r for r in rows if isinstance(r.get("nests"), list)]
    failed = [r for r in rows if not isinstance(r.get("nests"), list)]
    nests = [n for r in measured for n in r["nests"]]
    print(f"kernels: {len(rows)} | reached the arena: {len(measured)} | failed outright: {len(failed)}")
    print(f"nests: {len(nests)} | measured: {sum(1 for n in nests if not n['error'])}")
    collapsed = sum(len(n["collapsed"]) for n in nests)
    print(f"duplicate-artifact groups collapsed: {collapsed}")
    for r in failed:
        print(f"  FAILED {r['kernel']}: {str(r.get('error'))[:110]}")
    for u in unreadable:
        print(f"  UNREADABLE {u}")
    (out_dir / "summary.json").write_text(
        json.dumps({
            "kernels": len(rows),
            "nests": len(nests),
            "unreadable": unreadable
        }, indent=1))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.environ.get("NF_OUT", "perf_results/foundation"))
    parser.add_argument("--preset", choices=PRESETS, default=os.environ.get("NF_PRESET", DEFAULT_PRESET))
    parser.add_argument("--reps", type=int, default=int(os.environ.get("NF_REPS", "15")))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="first N kernels only (smoke runs)")
    parser.add_argument("--only", nargs="*", default=None, help="explicit kernel names")
    parser.add_argument("--tables-only", action="store_true", help="merge existing per-kernel JSON, measure nothing")
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    if args.tables_only:
        return merge_tables(out_dir)
    return run_sweep(out_dir, args.preset, args.reps, args.seed, args.limit, args.only)


if __name__ == "__main__":
    raise SystemExit(main())
