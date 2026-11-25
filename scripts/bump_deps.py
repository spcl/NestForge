#!/usr/bin/env python
# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Point a checkout at the tip of a HPCAgent-Bench branch (:data:`BENCH_BRANCH` unless one is named) in place of the
PyPI release: writes ``hpcagent-bench`` into pyproject.toml's ``[tool.uv.sources]`` (adding the table when absent),
which a checkout's ``uv sync`` or ``uv pip install -e .`` reads. CI's bench-tip job runs it; main itself installs the
bench, DaCe extended, dace-relayout and FP-Arena from PyPI. Then reinstall and test:
``python scripts/bump_deps.py && uv pip install -e '.[bench]' --group dev && pytest``.
"""

import re
import subprocess
import sys
from pathlib import Path

# not tactics.paths: importing tactics loads DaCe, which a fresh checkout has not installed yet
PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
BENCH = "https://github.com/spcl/HPCAgent-Bench.git"
#: The bench branch Tactics tracks.
BENCH_BRANCH = "main"


def tip(repo: str, branch: str) -> str:
    out = subprocess.run(["git", "ls-remote", repo, f"refs/heads/{branch}"], capture_output=True, text=True, check=True)
    return out.stdout.split()[0]


def main(bench_branch: str) -> None:
    sha = tip(BENCH, bench_branch)
    source = f'hpcagent-bench = {{ git = "{BENCH}", rev = "{sha}" }}'
    text = PYPROJECT.read_text()
    text, count = re.subn(r"^hpcagent-bench = \{ git = .*$", source, text, flags=re.MULTILINE)
    if count == 0 and "[tool.uv.sources]\n" in text:
        text = text.replace("[tool.uv.sources]\n", f"[tool.uv.sources]\n{source}\n", 1)
    elif count == 0:
        text = text.replace("[tool.uv]\n", f"[tool.uv.sources]\n{source}\n\n[tool.uv]\n", 1)
    if source not in text:
        raise SystemExit(f"could not write the hpcagent-bench source into {PYPROJECT}")
    print(f"hpcagent-bench: {sha}")
    PYPROJECT.write_text(text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else BENCH_BRANCH)
