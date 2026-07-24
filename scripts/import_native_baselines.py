# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Import the TSVC native C++ baselines from the upstream VectraArtifacts tree into the HPCAgent-Bench
foundation, where :attr:`nestforge.tsvc.TsvcKernel.native_cpp` resolves them.

Each foundation kernel wants a ``<stem>_native.cpp`` next to its ``<stem>.yaml`` (the arena's native lane
compiles it and times it as the speedup denominator). The authoritative source is the single-kernel variant
under ``VectraArtifacts``:

  * ``tsvc2``   -> ``tsvc_2/tsvc_cpp_microkernels/<key>/<key>_d_single.cpp``  (self-timing single-run variant)
  * ``tsvc2_5`` -> ``tsvc_2_5/tsvc_2_5_cpp_microkernels/<key>/<key>_d.cpp``   (no ``_single`` variant exists)

Both carry a trailing ``std::int64_t* time_ns`` self-timing out-param and a ``<chrono>`` clock; the arena times
the baseline itself, so :func:`strip_timing` removes the instrumentation on import -- the ``_native.cpp`` is a
pure kernel. The upstream-attribution header is added by HPCAgent-Bench's own tooling, not here, so this only
bootstraps NEW kernels -- do not re-run it over the curated foundation files (it would drop their header).
Reports every copy and every kernel with no upstream source (never silent).

Usage: ``python scripts/import_native_baselines.py [--vectra DIR] [--dry-run]``.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from nestforge import tsvc

#: Per-corpus upstream layout: (cpp-microkernels subdir under the VectraArtifacts root, per-key filename).
CORPUS_SOURCE = {
    "tsvc2": ("tsvc_2/tsvc_cpp_microkernels", "{key}_d_single.cpp"),
    "tsvc2_5": ("tsvc_2_5/tsvc_2_5_cpp_microkernels", "{key}_d.cpp"),
}

#: The upstream single-kernel sources self-time via a trailing ``std::int64_t* time_ns`` out-param and a
#: ``<chrono>`` clock. The arena times the baseline itself (perf_counter around the ctypes call), so the
#: instrumentation is stripped on import -- the imported ``_native.cpp`` is a pure kernel with no timing
#: param. Each pattern removes one piece; ``\s`` spans newlines, so every formatting variant is covered.
TIME_STRIP = [
    (re.compile(r"#include <chrono>\n"), ""),
    (re.compile(r"using clock_highres[^\n]*\n"), ""),
    (re.compile(r",\s*std::int64_t \* __restrict__ time_ns\)"), ")"),
    (re.compile(r"\s*auto t[12] = clock_highres::now\(\);"), ""),
    (re.compile(r"\s*std::int64_t ns =\s*std::chrono::duration_cast<std::chrono::nanoseconds>\(t2 - t1\)"
                r"\.count\(\);"), ""),
    (re.compile(r"\s*(?:time_ns\[0\]|\*time_ns) =\s*"
                r"std::chrono::duration_cast<std::chrono::nanoseconds>\(t2 - t1\)\.count\(\);"), ""),
    (re.compile(r"\s*(?:time_ns\[0\]|\*time_ns) = ns;"), ""),
]


def strip_timing(text: str) -> str:
    """Remove the ``time_ns`` self-timing instrumentation (param, ``<chrono>`` include, clock reads, ns
    write) from an upstream single-kernel source, leaving the pure kernel the arena compiles."""
    for pat, repl in TIME_STRIP:
        text = pat.sub(repl, text)
    return text


def upstream_source(vectra: Path, corpus: str, key: str) -> Optional[Path]:
    """The VectraArtifacts ``.cpp`` for one kernel, or ``None`` when the corpus/key has no upstream file."""
    layout = CORPUS_SOURCE.get(corpus)
    if layout is None:
        return None
    subdir, fname = layout
    src = vectra / subdir / key / fname.format(key=key)
    return src if src.exists() else None


def plan_imports(vectra: Path) -> Tuple[List[Tuple[Path, Path]], List[str]]:
    """Resolve every TSVC kernel to a (source, destination) copy; return ``(copies, missing_keys)``.

    Destination is exactly what :attr:`TsvcKernel.native_cpp` expects: ``<stem>_native.cpp`` beside the
    kernel's foundation manifest. A kernel with no foundation entry or no upstream source is reported, not
    copied.
    """
    copies: List[Tuple[Path, Path]] = []
    missing: List[str] = []
    for corpus in CORPUS_SOURCE:
        for kernel in tsvc.iter_tsvc_kernels(corpus=corpus):
            entry = kernel.foundation_entry
            src = upstream_source(vectra, corpus, kernel.key)
            if entry is None or src is None:
                missing.append(f"{corpus}/{kernel.key} ("
                               f"{'no foundation entry' if entry is None else 'no upstream .cpp'})")
                continue
            copies.append((src, entry.with_name(f"{entry.stem}_native.cpp")))
    return copies, missing


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vectra",
                    type=Path,
                    default=Path.home() / "Work" / "VectraArtifacts",
                    help="VectraArtifacts root (default: ~/Work/VectraArtifacts)")
    ap.add_argument("--dry-run", action="store_true", help="report the plan without writing")
    args = ap.parse_args(argv)

    if not args.vectra.is_dir():
        print(f"error: VectraArtifacts root not found: {args.vectra}", file=sys.stderr)
        return 2

    copies, missing = plan_imports(args.vectra)
    for src, dst in copies:
        if not args.dry_run:
            dst.write_text(strip_timing(src.read_text()))
        print(f"{'would copy' if args.dry_run else 'copied'}: {src.name} -> {dst}")
    for m in missing:
        print(f"skip (no source): {m}")
    print(f"\n{len(copies)} baseline(s) {'planned' if args.dry_run else 'imported'}, {len(missing)} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
