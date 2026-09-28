#!/usr/bin/env python
# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pin DaCe and HPCAgent-Bench in pyproject.toml to the current tip of their tracked branches.

Run after DaCe extended gains a transformation or fix NestForge needs, then reinstall and test:
``python scripts/bump_deps.py && pip install -e ".[dev]" && pytest``.
"""

import re
import subprocess
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"
#: package -> (repository, tracked branch)
TRACKED = {
    "dace": ("https://github.com/spcl/dace.git", "extended"),
    "hpcagent_bench": ("https://github.com/spcl/HPCAgent-Bench.git", "main"),
}


def tip(repo: str, branch: str) -> str:
    out = subprocess.run(["git", "ls-remote", repo, f"refs/heads/{branch}"], capture_output=True, text=True, check=True)
    return out.stdout.split()[0]


def main() -> None:
    text = PYPROJECT.read_text()
    for package, (repo, branch) in TRACKED.items():
        sha = tip(repo, branch)
        pattern = rf'"{package} @ git\+{re.escape(repo)}@[0-9a-f]{{40}}"'
        text, count = re.subn(pattern, f'"{package} @ git+{repo}@{sha}"', text)
        if count != 1:
            raise SystemExit(f"expected one pinned {package} requirement in {PYPROJECT}, found {count}")
        print(f"{package}: {branch} @ {sha}")
    PYPROJECT.write_text(text)


if __name__ == "__main__":
    main()
