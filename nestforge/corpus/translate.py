# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""An extracted kernel written as NumPy and a manifest, and translated to C, C++ or Fortran by HPCAgent-Bench.
hpcagent_bench is imported inside functions, so importing nestforge never loads it."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from nestforge.ir.emit_numpy import nest_to_numpy
from nestforge.ir.emit_yaml import manifest_dict
from nestforge.ir.extract import Boundary
from nestforge.build.toolchain import COMPILE_TIMEOUT_S

if TYPE_CHECKING:
    from hpcagent_bench.spec import BenchSpec


@dataclass(slots=True)
class Prepared:
    """A kernel written as files the translator consumes."""

    name: str
    numpy_path: Path
    yaml_path: Path
    numpy_source: str
    manifest: dict[str, Any]
    spec: BenchSpec


def prepare(
    boundary: Boundary,
    name: str,
    out_dir: str | Path,
    sizes: dict[str, int] | None = None,
) -> Prepared:
    """Write ``<name>_numpy.py`` + ``<name>.yaml`` for one extracted kernel and build its ``BenchSpec``."""
    from hpcagent_bench.spec import BenchSpec

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    numpy_source = nest_to_numpy(boundary, fn_name=name)
    manifest = manifest_dict(boundary, name, sizes=sizes)
    numpy_path = out / f"{name}_numpy.py"
    numpy_path.write_text(numpy_source)
    yaml_path = out / f"{name}.yaml"
    yaml_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    spec = BenchSpec.from_yaml(dict(manifest), source=str(yaml_path))
    return Prepared(name, numpy_path, yaml_path, numpy_source, manifest, spec)


def emit_sources(prep: Prepared, out_dir: str | Path, target: str = "c") -> list[Path]:
    """Translate the kernel's NumPy source to ``target`` with HPCAgent-Bench's ``numpyto`` driver.

    :returns: the generated source files, C then C++ then Fortran.
    """
    from hpcagent_bench import emit_bridge

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with emit_bridge.bench_info_tempfile(prep.spec) as bench_info:
        cmd = [sys.executable, "-m", "numpyto_common.cli", "--target", target, "--kernel", str(prep.numpy_path)]
        cmd += ["--bench-info", str(bench_info), "--out", str(out), "--precision", "float64"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"numpyto timed out for {prep.name} (target={target}); the ceiling is NF_COMPILE_TIMEOUT"
            )
    if res.returncode != 0:
        raise RuntimeError(f"numpyto failed for {prep.name} (target={target}):\n{res.stderr[-2000:]}")
    name = prep.name
    return sorted(out.glob(f"{name}_*.c")) + sorted(out.glob(f"{name}_*.cpp")) + sorted(out.glob(f"{name}_*.f90"))
