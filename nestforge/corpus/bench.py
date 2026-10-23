# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""HPCAgent-Bench's kernels as SDFGs, the NestForge corpus. A kernel's ``_dace.py`` reads HPCAgent-Bench's precision
global, which is set to float64 before it is imported.
hpcagent_bench is imported inside functions, so importing nestforge never loads it."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dace
import yaml

from nestforge.paths import REPO_ROOT

if TYPE_CHECKING:
    from types import ModuleType

    from hpcagent_bench.spec import BenchSpec

#: Tracks whose ``_dace.py`` this module generates on demand (gitignored, never committed).
DACE_TRACKS = ("loop_level_reasoning", "scientific_computing", "machine_learning")
#: The track name of the repo-local kernels, which are not part of hpcagent_bench's corpus.
LOCAL_TRACK = "benchmarking_creation"
#: Where repo-local kernels live: one folder per kernel with its ``<name>.yaml`` and a committed ``<name>_dace.py``.
LOCAL_ROOT = REPO_ROOT / LOCAL_TRACK


@dataclass(slots=True)
class CorpusKernel:
    """One corpus kernel with a ``@dace.program`` implementation."""

    short_name: str  # registry key, e.g. "hpc/dense_linear_algebra/gemm/gemm"
    module_path: str  # canonical dotted name, used only as the sys.modules cache key
    dace_file: Path  # the kernel's ``_dace.py`` on disk (source of truth)
    spec: BenchSpec

    def module(self) -> ModuleType:
        """Imports the kernel's ``_dace.py`` by file path, sidestepping ``hpcagent_bench.benchmarks``
        namespace-package resolution (which can bind a stray duplicate ``benchmarks/`` root)."""
        if self.module_path in sys.modules:
            return sys.modules[self.module_path]
        spec = importlib.util.spec_from_file_location(self.module_path, self.dace_file)
        assert spec is not None and spec.loader is not None, f"{self.dace_file} is not importable"
        module = importlib.util.module_from_spec(spec)
        sys.modules[self.module_path] = module
        spec.loader.exec_module(module)
        return module

    def program(self) -> dace.frontend.python.parser.DaceProgram:
        """The kernel's entry ``@dace.program``, named by the spec's ``func_name``; without one, the last
        program the module defines, since helpers precede the entry."""
        import hpcagent_bench.frameworks.dace_framework as dfw

        # the kernel module reads HPCAgent-Bench's dtype globals when it is imported
        dfw.dc_float = dace.float64
        dfw.dc_complex_float = dace.complex128
        module = self.module()
        entry = vars(module).get(self.spec.func_name)
        if isinstance(entry, dace.frontend.python.parser.DaceProgram):
            return entry
        programs = [v for v in vars(module).values() if isinstance(v, dace.frontend.python.parser.DaceProgram)]
        if not programs:
            raise LookupError(f"no @dace.program found in {self.dace_file}")
        return programs[-1]

    def to_sdfg(self, simplify: bool = True) -> dace.SDFG:
        """The program's SDFG with the manifest's small symbols declared small, as the bench's CPF route declares
        them: unless declared, a symbol is assumed big, and the defaults choose schedules by that."""
        from hpcagent_bench.cpf.canonical import declare_small, small_symbols

        sdfg = self.program().to_sdfg(simplify=simplify)
        declare_small(sdfg, small_symbols(self.spec))
        return sdfg


def module_path(short_name: str) -> str:
    """Canonical dotted name for a kernel's ``_dace.py`` (a stable sys.modules cache key)."""
    *dirs, module_name = short_name.split("/")
    prefix = "" if dirs[0] == LOCAL_TRACK else "hpcagent_bench.benchmarks."
    return f"{prefix}{'.'.join(dirs)}.{module_name}_dace"


def track_names(track: str | None) -> list[str]:
    """Short names of the corpus kernels of ``track``, or of every track."""
    from hpcagent_bench.spec import KERNELS

    return [name for name in KERNELS if track is None or name.startswith(f"{track}/")]


def generate_dace_file(short_name: str) -> None:
    """Regenerate hpcagent_bench's gitignored ``_dace.py`` of a :data:`DACE_TRACKS` kernel if it is missing."""
    from hpcagent_bench import autogen

    if short_name.split("/", 1)[0] in DACE_TRACKS:
        autogen.ensure(short_name, ("dace",))


def local_names() -> dict[str, Path]:
    """Registry key to manifest path of every repo-local kernel, in name order."""
    return {
        f"{LOCAL_TRACK}/{path.stem}/{path.stem}": path
        for path in sorted(LOCAL_ROOT.glob("*/*.yaml"))
        if path.parent.name == path.stem
    }


def corpus_kernel(name: str) -> CorpusKernel | None:
    """The corpus kernel ``name`` (registry key, manifest stem or directory) names, or None without a ``_dace.py``."""
    from hpcagent_bench.spec import KERNELS, BenchSpec

    short_name = KERNELS.path_key(name)
    if short_name is None:
        return None
    dace_file = KERNELS[short_name].parent / f"{short_name.rsplit('/', 1)[-1]}_dace.py"
    if not dace_file.exists():
        generate_dace_file(short_name)
    if not dace_file.exists():
        return None
    return CorpusKernel(short_name, module_path(short_name), dace_file, BenchSpec.load(short_name))


def local_spec(yaml_path: Path) -> BenchSpec:
    """The spec of a repo-local manifest. ``BenchSpec.from_yaml`` derives the reference's folder and the track from
    a location under hpcagent_bench's ``benchmarks/``, so this takes its steps with the kernel's own folder as
    ``relative_path`` (absolute, which hpcagent_bench joins onto its corpus root unchanged) and the default track."""
    from hpcagent_bench import manifest_schema
    from hpcagent_bench import spec as hb

    source = str(yaml_path)
    raw: dict[str, Any] = yaml.safe_load(yaml_path.read_text())
    manifest = manifest_schema.validate_manifest(raw, source)
    raw.update(relative_path=str(yaml_path.parent.resolve()), short_name=yaml_path.stem)
    raw.setdefault("module_name", yaml_path.stem)
    raw.setdefault("name", yaml_path.stem)
    func_name = hb.derive_func_name(raw["relative_path"], raw["module_name"])
    if func_name is not None:
        raw["func_name"] = func_name
    module = yaml_path.with_name(f"{raw['module_name']}.py")
    initializer = module if hb.defines_function(module, hb.INITIALIZER) else None
    hb.expand_args(raw, manifest.args, initializer, source)
    return hb.BenchSpec.from_dict(raw, source)


def iter_local_kernels() -> Iterator[CorpusKernel]:
    """Every repo-local kernel. Its ``_dace.py`` is committed with it, so a missing one is an authoring error."""
    for short_name, yaml_path in local_names().items():
        dace_file = yaml_path.with_name(f"{yaml_path.stem}_dace.py")
        if not dace_file.exists():
            raise FileNotFoundError(f"{dace_file} is missing; generate it with scripts/emit_local_dace.py")
        yield CorpusKernel(
            short_name=short_name,
            module_path=module_path(short_name),
            dace_file=dace_file,
            spec=local_spec(yaml_path),
        )


def iter_dace_kernels(track: str | None = None) -> Iterator[CorpusKernel]:
    """Every kernel with a ``_dace.py``, of ``track`` or of all tracks: the corpus's, then the repo-local ones."""
    if track != LOCAL_TRACK:
        for short_name in track_names(track):
            kernel = corpus_kernel(short_name)
            if kernel is not None:
                yield kernel
    if track in (None, LOCAL_TRACK):
        yield from iter_local_kernels()


def materialize_dace_corpus() -> None:
    """Generate every missing ``_dace.py``; run it once before parallel workers, which would race the write."""
    for short_name in track_names(None):
        generate_dace_file(short_name)


def preset_sizes(kernel: CorpusKernel, preset: str) -> dict[str, int]:
    """The integer symbol sizes of one preset in the kernel's manifest; config knobs merged into it are not sizes."""
    rung = kernel.spec.parameters.get(preset, {})
    knobs = kernel.spec.config_names
    return {
        sym: size
        for sym, size in rung.items()
        if sym not in knobs and isinstance(size, int) and not isinstance(size, bool)
    }
