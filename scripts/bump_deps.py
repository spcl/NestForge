#!/usr/bin/env python
# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pin HPCAgent-Bench in pyproject.toml to the tip of its main branch, and DaCe to the commit that tip pins.

The two DaCe pins must agree: HPCAgent-Bench requires its own DaCe commit, and a resolver refuses two URLs for one
package. A newer DaCe than the bench's arrives here once the bench moves to it. Then reinstall and test:
``python scripts/bump_deps.py && pip install -e ".[dev]" && pytest``.
"""

import re
import subprocess
import urllib.request

from nestforge.paths import REPO_ROOT

PYPROJECT = REPO_ROOT / "pyproject.toml"
DACE = "https://github.com/spcl/dace.git"
BENCH = "https://github.com/spcl/HPCAgent-Bench.git"
#: The bench's pyproject at one commit, where its DaCe pin is written.
BENCH_PYPROJECT = "https://raw.githubusercontent.com/spcl/HPCAgent-Bench/{sha}/pyproject.toml"


def tip(repo: str, branch: str) -> str:
    out = subprocess.run(["git", "ls-remote", repo, f"refs/heads/{branch}"], capture_output=True, text=True, check=True)
    return out.stdout.split()[0]


def bench_dace(bench: str) -> str:
    """The DaCe commit HPCAgent-Bench ``bench`` pins."""
    with urllib.request.urlopen(BENCH_PYPROJECT.format(sha=bench), timeout=30) as response:
        text = response.read().decode()
    match = re.search(rf'{re.escape(DACE)}", rev = "([0-9a-f]{{40}})"', text)
    if match is None:
        raise SystemExit(f"HPCAgent-Bench {bench} pins no DaCe commit")
    return match.group(1)


def main() -> None:
    bench = tip(BENCH, "main")
    pins = {"hpcagent_bench": (BENCH, bench), "dace": (DACE, bench_dace(bench))}
    text = PYPROJECT.read_text()
    for package, (repo, sha) in pins.items():
        pattern = rf'"{package} @ git\+{re.escape(repo)}@[0-9a-f]{{40}}"'
        text, count = re.subn(pattern, f'"{package} @ git+{repo}@{sha}"', text)
        if count != 1:
            raise SystemExit(f"expected one pinned {package} requirement in {PYPROJECT}, found {count}")
        print(f"{package}: {sha}")
    PYPROJECT.write_text(text)


if __name__ == "__main__":
    main()
