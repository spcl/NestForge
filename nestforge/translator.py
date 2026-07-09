"""Native surface for optarena's **numpy translator** -- one of the two pieces of optarena that
nest-forge exposes as first-class (the other is :mod:`nestforge.corpus`, the kernel corpus).

optarena itself is vendored as a git submodule (``external/optarena``); nest-forge reaches into it
for exactly two things -- this translator and the corpus -- and wraps them here so the rest of the
codebase depends on ``nestforge.translator`` / ``nestforge.corpus`` rather than optarena internals.

The translator turns a ``*_numpy.py`` kernel plus its ``BenchSpec`` manifest into C / C++ / Fortran
source, via optarena's ``numpyto`` driver.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

from optarena import emit_bridge
from optarena.spec import BenchSpec

#: optarena's numpy -> {C, C++, Fortran} translator CLI entry point.
DRIVER = "numpyto_common.cli"

__all__ = ["BenchSpec", "DRIVER", "translate"]


def translate(spec: BenchSpec, numpy_path, name: str, out_dir, target: str = "c",
              precision: str = "float64") -> List[Path]:
    """Translate the ``*_numpy.py`` kernel at ``numpy_path`` into ``target`` source under ``out_dir``.

    :param spec: the kernel's :class:`BenchSpec` (shapes, dtypes, signature).
    :param name: the kernel base name (the generated files are ``<name>_<variant>.<ext>``).
    :returns: the generated source files, C then C++ then Fortran.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with emit_bridge.bench_info_tempfile(spec) as bench_info:
        cmd = [sys.executable, "-m", DRIVER, "--target", target, "--kernel", str(numpy_path),
               "--bench-info", str(bench_info), "--out", str(out), "--precision", precision]
        res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"numpyto failed for {name} (target={target}):\n{res.stderr[-2000:]}")
    return (sorted(out.glob(f"{name}_*.c")) + sorted(out.glob(f"{name}_*.cpp"))
            + sorted(out.glob(f"{name}_*.f90")))
