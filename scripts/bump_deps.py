#!/usr/bin/env python
# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pin HPCAgent-Bench (and its agent package) in pyproject.toml's ``[tool.uv.sources]`` to the tip of a bench branch
(:data:`BENCH_BRANCH` unless one is named), and dace-relayout to the commit that bench commit pins. DaCe extended
follows its PyPI releases (``dace-extended>=X.Y.Z``), so this script leaves it alone.

``[project].dependencies`` name plain PyPI packages (PyPI refuses a direct URL), so the exact commits live only in
``[tool.uv.sources]``, which a checkout's ``uv sync`` or ``uv pip install -e .`` reads. Then reinstall and test:
``python scripts/bump_deps.py && uv pip install -e ".[dev]" && pytest``.
"""

import re
import subprocess
import sys
import tomllib
import urllib.request
from pathlib import Path

# not tactics.paths: importing tactics loads DaCe, which a fresh checkout has not installed yet
PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
BENCH = "https://github.com/spcl/HPCAgent-Bench.git"
#: The bench's pyproject.toml at one commit.
BENCH_PYPROJECT = "https://raw.githubusercontent.com/spcl/HPCAgent-Bench/{sha}/pyproject.toml"
#: The bench branch NestForge tracks.
BENCH_BRANCH = "main"
#: The bench's ``[tool.uv.sources]`` key of relayout once it pins the PyPI-renamed package. A bench that still keys it
#: ``relayout`` pins a commit whose project is not yet called dace-relayout, so ours stays at its release commit.
RELAYOUT_KEY = "dace-relayout"


def tip(repo: str, branch: str) -> str:
    out = subprocess.run(["git", "ls-remote", repo, f"refs/heads/{branch}"], capture_output=True, text=True, check=True)
    return out.stdout.split()[0]


def bench_sources(bench: str) -> dict[str, str]:
    """The commit of every git source in the bench's ``[tool.uv.sources]`` at ``bench``, by package."""
    with urllib.request.urlopen(BENCH_PYPROJECT.format(sha=bench)) as response:
        sources = tomllib.load(response)["tool"]["uv"]["sources"]
    return {name: source["rev"] for name, source in sources.items() if isinstance(source, dict) and "rev" in source}


def main(bench_branch: str) -> None:
    bench = tip(BENCH, bench_branch)
    sources = bench_sources(bench)
    pins = {"hpcagent-bench": bench, "hpcagent-agent": bench}
    if RELAYOUT_KEY in sources:
        pins[RELAYOUT_KEY] = sources[RELAYOUT_KEY]
    else:
        print(f"{RELAYOUT_KEY}: kept (the bench at {bench} does not pin it yet)")
    text = PYPROJECT.read_text()
    for package, sha in pins.items():
        pattern = rf'^({re.escape(package)} = {{ git = "[^"]+", rev = ")[0-9a-f]{{40}}(")'
        text, count = re.subn(pattern, rf"\g<1>{sha}\g<2>", text, flags=re.MULTILINE)
        if count != 1:
            raise SystemExit(f"expected one {package} source in [tool.uv.sources] of {PYPROJECT}, found {count}")
        print(f"{package}: {sha}")
    PYPROJECT.write_text(text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else BENCH_BRANCH)
