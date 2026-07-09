"""Drive the numpy translator: extracted nest -> numpy + manifest -> C/C++/Fortran sources.

The translation step goes through :mod:`nestforge.translator` (nest-forge's native surface over
optarena's ``numpyto`` driver), not optarena directly, so the kernel need not be registered in
optarena's benchmark registry.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml

from nestforge.emit_numpy import nest_to_numpy
from nestforge.emit_yaml import manifest_dict
from nestforge.extract import Boundary
from nestforge.translator import BenchSpec, translate


@dataclass
class Prepared:
    """A nest turned into files the translator can consume."""
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
    """Run the numpy translator; return the generated source files."""
    return translate(prep.spec, prep.numpy_path, prep.name, out_dir, target=target, precision=precision)
