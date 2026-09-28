# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""HPCAgent-Bench imports NestForge, so importing NestForge must never load HPCAgent-Bench."""

import subprocess
import sys

from nestforge.paths import REPO_ROOT


def test_importing_nestforge_never_loads_hpcagent_bench():
    """HPCAgent-Bench drives NestForge, so a top-level import the other way would close an import cycle."""
    script = (
        "import importlib, pkgutil, sys\n"
        "import nestforge\n"
        "for info in pkgutil.walk_packages(nestforge.__path__, nestforge.__name__ + '.'):\n"
        "    importlib.import_module(info.name)\n"
        "leaked = sorted(m for m in sys.modules if m.startswith(('hpcagent_bench', 'numpyto')))\n"
        "sys.stdout.write(','.join(leaked))\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr
    leaked = [name for name in result.stdout.strip().split(",") if name]
    assert not leaked, f"importing nestforge modules loaded: {leaked}"
