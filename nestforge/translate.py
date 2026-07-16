"""Drive the numpy translator: extracted nest -> numpy + manifest -> C/C++/Fortran sources.

The translation step goes through :mod:`nestforge.translator` (nest-forge's native surface over
optarena's ``numpyto`` driver), not optarena directly, so the kernel need not be registered in
optarena's benchmark registry.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

import dace

from nestforge.emit_numpy import nest_to_numpy, sdfg_to_numpy
from nestforge.emit_yaml import manifest_dict
from nestforge.extract import Boundary, detach, whole_program_boundary
from nestforge.split_unsupported import region_to_standalone, whole_program_regions
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


def build_prepared(name: str, out: Path, numpy_source: str, manifest: Dict) -> Prepared:
    """Write ``<name>_numpy.py`` + ``<name>.yaml`` for an already-rendered numpy source + manifest and build
    the OptArena ``BenchSpec``. Shared tail of :func:`prepare` (per-nest) and :func:`prepare_whole_program`."""
    out.mkdir(parents=True, exist_ok=True)
    numpy_path = out / f"{name}_numpy.py"
    numpy_path.write_text(numpy_source)
    yaml_path = out / f"{name}.yaml"
    yaml_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    spec = BenchSpec.from_yaml(dict(manifest), source=str(yaml_path))
    return Prepared(name=name,
                    numpy_path=numpy_path,
                    yaml_path=yaml_path,
                    numpy_source=numpy_source,
                    manifest=manifest,
                    spec=spec)


def prepare(boundary: Boundary,
            name: str,
            out_dir: os.PathLike,
            sizes: Dict[str, int] = None,
            preset: str = "S") -> Prepared:
    """Write ``<name>_numpy.py`` + ``<name>.yaml`` for ONE extracted nest and build the OptArena ``BenchSpec``."""
    out = Path(out_dir)
    numpy_source = nest_to_numpy(boundary, fn_name=name)
    manifest = manifest_dict(boundary, name, sizes=sizes, preset=preset)
    return build_prepared(name, out, numpy_source, manifest)


def prepare_whole_program(sdfg: dace.SDFG,
                          name: str,
                          out_dir: os.PathLike,
                          sizes: Dict[str, int] = None,
                          preset: str = "S") -> Prepared:
    """Whole-program-scope analogue of :func:`prepare`: emit the ENTIRE (un-split) kernel SDFG via
    :func:`sdfg_to_numpy` (the corpus whole-SDFG emitter -- handles multiple nests + ``__return`` + scratch)
    and build its manifest from a :func:`whole_program_boundary`. The rest (files + ``BenchSpec``) is
    identical to the per-nest path, so the same ``emit_sources`` / oracle / compile machinery drives it --
    the only difference is scope. May raise ``UnsupportedNest`` when the whole program cannot be externalized
    (early return / orphan break); the caller records that as a whole-program skip."""
    out = Path(out_dir)
    boundary = whole_program_boundary(sdfg)
    numpy_source = sdfg_to_numpy(boundary.standalone_sdfg, fn_name=name)
    manifest = manifest_dict(boundary, name, sizes=sizes, preset=preset)
    return build_prepared(name, out, numpy_source, manifest)


def prepare_regions(sdfg: dace.SDFG,
                    name: str,
                    out_dir: os.PathLike,
                    sizes: Dict[str, int] = None,
                    preset: str = "S") -> Tuple[List[Prepared], List[dace.SDFGState]]:
    """Split-around-unsupported whole-program preparation: isolate every non-emittable library node
    (MPI/pblas, sparse, ...) into its own state, then emit a :class:`Prepared` for each externalizable
    REGION -- the pure-compute states before/after each island -- leaving the island states native.

    Returns ``(prepared, islands)``: one :class:`Prepared` per region (each a standalone numpy kernel over
    that region, with cross-boundary transients surfaced as its inputs/outputs) and the list of native
    island states the caller must run through the DaCe path, not externalize. When the whole program is
    pure this is a single region == the whole program (equivalent to :func:`prepare_whole_program`).

    Runs on a detached copy, so ``sdfg`` is not mutated. May raise ``UnsupportedNest`` when a region cannot
    be emitted (e.g. an early return) -- the caller records that region as a whole-program skip."""
    out = Path(out_dir)
    work = detach(sdfg)
    regions, islands = whole_program_regions(work)
    prepared: List[Prepared] = []
    for i, region in enumerate(regions):
        region_name = f"{name}_region{i}" if len(regions) > 1 else name
        standalone = region_to_standalone(work, region, region_name)
        boundary = whole_program_boundary(standalone)
        numpy_source = sdfg_to_numpy(standalone, fn_name=region_name)
        manifest = manifest_dict(boundary, region_name, sizes=sizes, preset=preset)
        prepared.append(build_prepared(region_name, out, numpy_source, manifest))
    return prepared, islands


def emit_sources(prep: Prepared, out_dir: os.PathLike, target: str = "c", precision: str = "float64") -> List[Path]:
    """Run the numpy translator; return the generated source files."""
    return translate(prep.spec, prep.numpy_path, prep.name, out_dir, target=target, precision=precision)
