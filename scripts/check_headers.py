#!/usr/bin/env python
# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Require the copyright and SPDX header on every Python file under nestforge/, scripts/ and tests/.

The header follows an optional shebang and coding line. A header with another year or project name is accepted;
``--fix`` adds the canonical one to files that have none. Without file arguments, every tracked file is checked.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPDX_LINE = "# SPDX-License-Identifier: GPL-3.0-or-later"
HEADER = ("# Copyright 2021 ETH Zurich and the NestForge authors.", SPDX_LINE)
COPYRIGHT_RE = re.compile(r"^# Copyright \d{4} ETH Zurich and the [\w.-]+ authors\.$")
CODING_RE = re.compile(r"^[ \t\f]*#.*?coding[:=]")
SCOPE_PREFIXES = ("nestforge/", "scripts/", "tests/")


def in_scope(rel: str) -> bool:
    posix = rel.replace("\\", "/")
    return posix.endswith(".py") and posix.startswith(SCOPE_PREFIXES)


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True)
    return [line for line in out.stdout.splitlines() if line.strip()] if out.returncode == 0 else []


def prefix_len(lines: list[str]) -> int:
    """How many leading lines (a shebang, then a coding line) precede the header."""
    index = 1 if lines and lines[0].startswith("#!") else 0
    return index + 1 if index < len(lines) and CODING_RE.match(lines[index]) else index


def has_header(lines: list[str]) -> bool:
    start = prefix_len(lines)
    head = lines[start : start + 2]
    return len(head) == 2 and COPYRIGHT_RE.match(head[0]) is not None and head[1] == SPDX_LINE


def insert_header(path: Path) -> bool:
    """Add the header after any shebang and coding line; ``False`` when it was already there."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    bare = [line.rstrip("\r\n") for line in lines]
    if has_header(bare):
        return False
    newline = "\r\n" if lines and lines[0].endswith("\r\n") else "\n"
    start = prefix_len(bare)
    path.write_text("".join([*lines[:start], *(line + newline for line in HEADER), *lines[start:]]), encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true", help="insert missing headers instead of failing")
    parser.add_argument("files", nargs="*", help="files to check (default: every tracked file in scope)")
    args = parser.parse_args(argv)
    candidates = args.files or tracked_files()
    targets = [rel for rel in sorted(set(candidates)) if in_scope(rel) and (REPO_ROOT / rel).is_file()]
    if args.fix:
        fixed = [rel for rel in targets if insert_header(REPO_ROOT / rel)]
        print(f"check-headers: inserted the header into {len(fixed)} of {len(targets)} file(s)")
        return 0
    missing = [rel for rel in targets if not has_header((REPO_ROOT / rel).read_text(encoding="utf-8").splitlines())]
    if not missing:
        print(f"check-headers: {len(targets)} file(s) OK")
        return 0
    print(f"check-headers: {len(missing)} of {len(targets)} file(s) lack the copyright/SPDX header:")
    for rel in missing:
        print(f"  {rel}")
    print("Fix with: python scripts/check_headers.py --fix")
    return 1


if __name__ == "__main__":
    sys.exit(main())
