# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``python -m nestforge KERNEL --size N=1048576`` -- measure every nest of a kernel and print the report.

The user-facing front door onto :func:`nestforge.run.optimize_program`. Sizes are explicit because a
guessed one silently changes what was measured.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from nestforge.run import optimize_program


def parse_sizes(pairs: Sequence[str]) -> Dict[str, int]:
    """``["N=1024", "M=512"]`` -> ``{"N": 1024, "M": 512}``."""
    sizes: Dict[str, int] = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep or not name.strip():
            raise argparse.ArgumentTypeError(f"--size expects SYMBOL=INT, got {pair!r}")
        try:
            sizes[name.strip()] = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"--size value for {name.strip()!r} is not an integer: {value!r}") from None
    return sizes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nestforge",
                                     description="Compile every semantics-preserving variant of a kernel's "
                                     "nests, validate each against the numpy oracle, and report the fastest.")
    parser.add_argument("kernel", help="a .py holding a @dace.program, or a serialized .sdfg/.sdfgz")
    parser.add_argument("--size",
                        "-s",
                        action="append",
                        default=[],
                        metavar="SYM=INT",
                        help="value for one free symbol; repeat per symbol (required if the kernel has any)")
    parser.add_argument("--out", "-o", default=None, help="build tree (default: a kept temp dir)")
    parser.add_argument("--func", default=None, help="which @dace.program, when the file defines several")
    parser.add_argument("--strategy", default="skip-taskloops", help="offload granularity (default: skip-taskloops)")
    parser.add_argument("--reps", type=int, default=50, help="timed repetitions per variant (default: 50)")
    parser.add_argument("--seed", type=int, default=0, help="input-generation seed (default: 0)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = optimize_program(Path(args.kernel),
                              sizes=parse_sizes(args.size),
                              out_dir=args.out,
                              func_name=args.func,
                              strategy=args.strategy,
                              reps=args.reps,
                              seed=args.seed)
    print(report.markdown())
    # A nest that never reached the arena is a failure of the run, not a report to exit 0 on
    return 1 if any(n.arena is None for n in report.nests) else 0


if __name__ == "__main__":
    sys.exit(main())
