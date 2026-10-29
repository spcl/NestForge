#!/usr/bin/env python
# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pin HPCAgent-Bench (and its agent package) in pyproject.toml to the tip of a bench branch (:data:`BENCH_BRANCH`
unless one is named), and DaCe and relayout to the commits that bench commit pins: the bench owns both pins.

The bench pins DaCe and relayout only in ``[tool.uv.sources]``, which neither pip nor a uv install of another project
reads (PyPI's ``relayout`` is an unrelated project), so each is written twice: as the dependency and as the
``[tool.uv]`` override. Then reinstall and test:
``python scripts/bump_deps.py && uv pip install -e ".[dev]" && pytest``.
"""

import re
import subprocess
import sys
import tomllib
import urllib.request
from pathlib import Path

# not nestforge.paths: importing nestforge loads DaCe, which a fresh checkout has not installed yet
PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
DACE = "https://github.com/spcl/dace.git"
BENCH = "https://github.com/spcl/HPCAgent-Bench.git"
RELAYOUT = "https://github.com/ThrudPrimrose/relayout.git"
#: The bench's pyproject.toml at one commit.
BENCH_PYPROJECT = "https://raw.githubusercontent.com/spcl/HPCAgent-Bench/{sha}/pyproject.toml"
#: The bench branch NestForge tracks: the release it ships with (main lacks its phase routes until the release merges).
BENCH_BRANCH = "release-v0.4.0"
#: Where each package's pin is written: the dependency, and for DaCe and relayout also the uv override.
PIN_COUNT = {"hpcagent_bench": 1, "hpcagent-agent": 1, "dace": 2, "relayout": 2}


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
    pins = {
        "hpcagent_bench": (BENCH, bench),
        "hpcagent-agent": (BENCH, bench),
        "dace": (DACE, sources["dace"]),
        "relayout": (RELAYOUT, sources["relayout"]),
    }
    text = PYPROJECT.read_text()
    for package, (repo, sha) in pins.items():
        pattern = rf'"{package} @ git\+{re.escape(repo)}@[0-9a-f]{{40}}(#[^"]*)?"'
        text, count = re.subn(pattern, lambda m: f'"{package} @ git+{repo}@{sha}{m.group(1) or ""}"', text)
        if count != PIN_COUNT[package]:
            raise SystemExit(
                f"expected {PIN_COUNT[package]} pinned {package} requirement(s) in {PYPROJECT}, found {count}"
            )
        print(f"{package}: {sha}")
    PYPROJECT.write_text(text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else BENCH_BRANCH)
