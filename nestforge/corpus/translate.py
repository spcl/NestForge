# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""An extracted kernel written as Python and a manifest, and translated to C, C++ or Fortran by HPCAgent-Bench.
hpcagent_bench is imported inside functions, so importing nestforge never loads it."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from dace import symbolic

from nestforge.build.toolchain import COMPILE_TIMEOUT_S
from nestforge.ir.emit_python import kernel_args, kernel_arrays, oracle_sdfg, render
from nestforge.ir.extract import Boundary

if TYPE_CHECKING:
    from hpcagent_bench.spec import BenchSpec

#: Size of an integer symbol the caller gave no value for.
DEFAULT_SIZE = 1 << 16
#: Characters of the translator's stderr kept in a failure message.
STDERR_TAIL = 2000


@dataclass(slots=True)
class Prepared:
    """A kernel written as files the translator consumes."""

    name: str
    numpy_path: Path
    yaml_path: Path
    numpy_source: str
    manifest: dict[str, Any]
    spec: BenchSpec


def python_and_manifest(
    boundary: Boundary, name: str, sizes: dict[str, int] | None = None
) -> tuple[str, dict[str, Any]]:
    """The kernel's Python oracle and its argument manifest, which share one signature."""
    sdfg = oracle_sdfg(boundary)
    arrays = kernel_arrays(boundary, sdfg)
    args = kernel_args(boundary, arrays)
    shapes: dict[str, dict[str, str]] = {}
    for a in arrays:
        dims = [symbolic.symstr(d, cpp_mode=False) for d in sdfg.arrays[a].shape]
        shape = "(" + ", ".join(dims) + ("," if len(dims) == 1 else "") + ")"
        shapes[a] = {"shape": shape, "dtype": np.dtype(sdfg.arrays[a].dtype.type).name}
    int_params: dict[str, int] = {}
    float_scalars: dict[str, float] = {}
    for s in boundary.symbols:
        # a float symbol is a staged value; the translator declares it double, not int64
        if s in sdfg.symbols and np.dtype(sdfg.symbols[s].type).kind == "f":
            float_scalars[s] = 0.0
        else:
            int_params[s] = int((sizes or {}).get(s, DEFAULT_SIZE))
    init: dict[str, Any] = {"arrays": shapes} | ({"scalars": float_scalars} if float_scalars else {})
    manifest = {
        "name": name,
        "func_name": name,
        "relative_path": "extended",
        "level": 1,
        "parameters": {"S": int_params},
        "input_args": args,
        "array_args": arrays,
        "output_args": list(boundary.outputs),
        "init": init,
    }
    return render(name, args, sdfg), manifest


def prepare(boundary: Boundary, name: str, out_dir: str | Path, sizes: dict[str, int] | None = None) -> Prepared:
    """Write ``<name>_numpy.py`` + ``<name>.yaml`` for one extracted kernel and build its ``BenchSpec``."""
    from hpcagent_bench.spec import BenchSpec

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    source, manifest = python_and_manifest(boundary, name, sizes)
    numpy_path = out / f"{name}_numpy.py"
    numpy_path.write_text(source)
    yaml_path = out / f"{name}.yaml"
    yaml_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    spec = BenchSpec.from_yaml(dict(manifest), source=str(yaml_path))
    return Prepared(name, numpy_path, yaml_path, source, manifest, spec)


def emit_sources(prep: Prepared, out_dir: str | Path, target: str = "c") -> list[Path]:
    """Translate the kernel's Python source to ``target`` with HPCAgent-Bench's ``numpyto`` driver.

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
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"numpyto timed out for {prep.name} (target={target}); see NF_COMPILE_TIMEOUT") from exc
    if res.returncode != 0:
        raise RuntimeError(f"numpyto failed for {prep.name} (target={target}):\n{res.stderr[-STDERR_TAIL:]}")
    name = prep.name
    return sorted(out.glob(f"{name}_*.c")) + sorted(out.glob(f"{name}_*.cpp")) + sorted(out.glob(f"{name}_*.f90"))
