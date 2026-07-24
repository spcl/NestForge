# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Import the TSVC native C++ baselines from the upstream VectraArtifacts tree into the HPCAgent-Bench
foundation, where :attr:`nestforge.tsvc.TsvcKernel.native_cpp` resolves them.

Each foundation kernel wants a ``<stem>_original.cpp`` next to its ``<stem>.yaml`` (the arena's native lane
compiles it and times it as the speedup denominator). The authoritative source is the single-kernel variant
under ``VectraArtifacts``:

  * ``tsvc2``   -> ``tsvc_2/tsvc_cpp_microkernels/<key>/<key>_d_single.cpp``  (self-timing single-run variant)
  * ``tsvc2_5`` -> ``tsvc_2_5/tsvc_2_5_cpp_microkernels/<key>/<key>_d.cpp``   (no ``_single`` variant exists)

Both carry a trailing ``std::int64_t* time_ns`` self-timing out-param; the arena binds it to scratch and keeps
its own timing (see :data:`nestforge.perf.harness.NATIVE_TIME_PARAM`). Idempotent: overwrites the destination
so a re-run tracks the upstream. Reports every copy and every kernel with no upstream source (never silent).

Usage: ``python scripts/import_native_baselines.py [--vectra DIR] [--dry-run]``.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from nestforge import tsvc

#: Per-corpus upstream layout: (cpp-microkernels subdir under the VectraArtifacts root, per-key filename).
CORPUS_SOURCE = {
    "tsvc2": ("tsvc_2/tsvc_cpp_microkernels", "{key}_d_single.cpp"),
    "tsvc2_5": ("tsvc_2_5/tsvc_2_5_cpp_microkernels", "{key}_d.cpp"),
}


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

    Destination is exactly what :attr:`TsvcKernel.native_cpp` expects: ``<stem>_original.cpp`` beside the
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
            copies.append((src, entry.with_name(f"{entry.stem}_original.cpp")))
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
            shutil.copyfile(src, dst)
        print(f"{'would copy' if args.dry_run else 'copied'}: {src.name} -> {dst}")
    for m in missing:
        print(f"skip (no source): {m}")
    print(f"\n{len(copies)} baseline(s) {'planned' if args.dry_run else 'imported'}, {len(missing)} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
