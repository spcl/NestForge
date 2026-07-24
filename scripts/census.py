# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Census: emit a standalone numpy kernel for every optarena dace kernel and classify the outcome.

For each corpus kernel we build its SDFG and run ``sdfg_to_numpy`` (library nodes -> numpy ops, maps
-> ``for`` loops). The buckets show, on real npbench/polybench kernels, exactly what the emitter
handles today and what remains:

  OK           emitted a standalone numpy kernel that parses
  UNSUPPORTED  a construct the emitter rejects (control-flow region, unmapped library node, WCR ...)
  BUILD_FAIL   ``to_sdfg`` raised (frontend issue, upstream of nest-forge)
  EMIT_FAIL    emitter raised something other than a clean rejection, or output does not parse

Run: ``python scripts/census.py [track]``  (track = hpc | ml | foundation; default all).
"""
import ast
import sys
from collections import defaultdict

from nestforge.corpus import iter_dace_kernels
from nestforge.emit_libnode import UnsupportedLibraryNode
from nestforge.emit_numpy import UnsupportedNest, sdfg_to_numpy


def classify(kernel):
    try:
        sdfg = kernel.to_sdfg(simplify=True)
    except Exception as exc:
        return "BUILD_FAIL", f"{type(exc).__name__}: {str(exc)[:70]}"
    try:
        source = sdfg_to_numpy(sdfg, kernel.short_name.rsplit("/", 1)[-1])
    except (UnsupportedNest, UnsupportedLibraryNode) as exc:
        return "UNSUPPORTED", str(exc)[:70]
    except Exception as exc:
        return "EMIT_FAIL", f"{type(exc).__name__}: {str(exc)[:70]}"
    try:
        ast.parse(source)
    except SyntaxError as exc:
        return "EMIT_FAIL", f"SyntaxError: {exc}"
    return "OK", ""


def main(track=None):
    buckets = defaultdict(list)
    for kernel in iter_dace_kernels(track):
        outcome, detail = classify(kernel)
        buckets[outcome].append((kernel.short_name, detail))
        print(f"  [{outcome:11}] {kernel.short_name}" + (f"  -- {detail}" if detail else ""))

    total = sum(len(v) for v in buckets.values())
    print(f"\n=== census summary ({total} dace kernels{f', track={track}' if track else ''}) ===")
    for outcome in ("OK", "UNSUPPORTED", "EMIT_FAIL", "BUILD_FAIL"):
        rows = buckets.get(outcome, [])
        if rows:
            print(f"  {outcome:11}: {len(rows):3}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
