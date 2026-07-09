"""Drive OptArena's translator: extracted nest -> numpy + manifest -> C/C++/Fortran sources.

Uses OptArena as a library (``BenchSpec.from_yaml`` + the ``numpyto`` driver) without needing
the kernel registered in OptArena's benchmark registry.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml

from optarena.spec import BenchSpec
from optarena import emit_bridge

from nestforge.emit_numpy import nest_to_numpy
from nestforge.emit_yaml import manifest_dict
from nestforge.extract import Boundary

_DRIVER = "numpyto_common.cli"


@dataclass
class Prepared:
    """A nest turned into files OptArena can translate."""
    name: str
    numpy_path: Path
    yaml_path: Path
    numpy_source: str
    manifest: Dict
    spec: BenchSpec


def prepare(boundary: Boundary, name: str, out_dir: os.PathLike, sizes: Dict[str, int] = None,
            preset: str = "S") -> Prepared:
    """Write ``<name>_numpy.py`` + ``<name>.yaml`` and build the OptArena ``BenchSpec``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    numpy_source = nest_to_numpy(boundary, fn_name=name)
    manifest = manifest_dict(boundary, name, sizes=sizes, preset=preset)

    numpy_path = out / f"{name}_numpy.py"
    numpy_path.write_text(numpy_source)
    yaml_path = out / f"{name}.yaml"
    yaml_path.write_text(yaml.safe_dump(manifest, sort_keys=False))

    spec = BenchSpec.from_yaml(dict(manifest), source=str(yaml_path))
    return Prepared(name=name, numpy_path=numpy_path, yaml_path=yaml_path,
                    numpy_source=numpy_source, manifest=manifest, spec=spec)


def emit_sources(prep: Prepared, out_dir: os.PathLike, target: str = "c",
                 precision: str = "float64") -> List[Path]:
    """Run the ``numpyto`` translator; return the generated source files."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with emit_bridge.bench_info_tempfile(prep.spec) as bi:
        cmd = [sys.executable, "-m", _DRIVER, "--target", target,
               "--kernel", str(prep.numpy_path), "--bench-info", str(bi),
               "--out", str(out), "--precision", precision]
        res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"numpyto failed for {prep.name} (target={target}):\n{res.stderr[-2000:]}")
    return sorted(out.glob(f"{prep.name}_*.c")) + sorted(out.glob(f"{prep.name}_*.cpp")) \
        + sorted(out.glob(f"{prep.name}_*.f90"))
