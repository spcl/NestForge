#!/usr/bin/env python
# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pin HPCAgent-Bench (and its agent package) in pyproject.toml to the tip of a bench branch (:data:`BENCH_BRANCH`
unless one is named) and DaCe to the tip of extended.

HPCAgent-Bench requires its own DaCe commit, so the DaCe pin is written twice: as the dependency and as the
``[tool.uv]`` override that wins over the bench's. Then reinstall and test:
``python scripts/bump_deps.py && uv pip install -e ".[dev]" && pytest``.
"""

import re
import subprocess
import sys
from pathlib import Path

# not nestforge.paths: importing nestforge loads DaCe, which a fresh checkout has not installed yet
PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
DACE = "https://github.com/spcl/dace.git"
BENCH = "https://github.com/spcl/HPCAgent-Bench.git"
#: The bench branch NestForge tracks: the release the plugin ships with.
BENCH_BRANCH = "release-v0.3.0"
#: Where each package's pin is written: the dependency, and for DaCe also the uv override.
PIN_COUNT = {"hpcagent_bench": 1, "hpcagent-agent": 1, "dace": 2}


def tip(repo: str, branch: str) -> str:
    out = subprocess.run(["git", "ls-remote", repo, f"refs/heads/{branch}"], capture_output=True, text=True, check=True)
    return out.stdout.split()[0]


def main(bench_branch: str) -> None:
    bench = tip(BENCH, bench_branch)
    pins = {"hpcagent_bench": (BENCH, bench), "hpcagent-agent": (BENCH, bench), "dace": (DACE, tip(DACE, "extended"))}
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
