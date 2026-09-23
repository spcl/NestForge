#!/usr/bin/env python
# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fail when a staged file exceeds ``--max-kb`` KiB, so build artifacts and datasets stay out of git.

pre-commit passes the staged files; without arguments, ``git diff --cached`` names them.
"""

import argparse
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

DEFAULT_MAX_KB = 500


def staged_files() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"], capture_output=True, text=True
    )
    return [line for line in out.stdout.splitlines() if line.strip()] if out.returncode == 0 else []


def oversized(paths: list[str], max_bytes: int) -> Iterator[tuple[str, int]]:
    """``(path, size)`` of every existing file larger than ``max_bytes``."""
    for rel in paths:
        path = Path(rel)
        if path.is_file() and path.stat().st_size > max_bytes:
            yield rel, path.stat().st_size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-kb", type=int, default=DEFAULT_MAX_KB, help="size limit in KiB (default: 500)")
    parser.add_argument("files", nargs="*", help="files to check (default: the staged set)")
    args = parser.parse_args(argv)
    offenders = sorted(oversized(args.files or staged_files(), args.max_kb * 1024))
    if not offenders:
        return 0
    print(f"error: {len(offenders)} staged file(s) exceed the {args.max_kb} KiB limit:", file=sys.stderr)
    for rel, size in offenders:
        print(f"  {rel}  ({size / 1024:.0f} KiB)", file=sys.stderr)
    print("Keep large artifacts out of git, or raise --max-kb deliberately.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
