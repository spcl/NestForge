# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""HPCAgent-Bench's kernels as SDFGs, the NestForge corpus. A kernel's ``_dace.py`` reads HPCAgent-Bench's precision
global, which is set to float64 before it is imported.
hpcagent_bench is imported inside functions, so importing nestforge never loads it."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from collections.abc import Iterator

import dace

if TYPE_CHECKING:
    from types import ModuleType

    from hpcagent_bench.spec import BenchSpec

#: Tracks whose ``_dace.py`` this module generates on demand (gitignored, never committed).
DACE_TRACKS = ("loop_level_reasoning", "scientific_computing", "machine_learning")


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
        """The kernel's entry ``@dace.program``, named by the manifest's ``func_name``; without one, the last
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
        return self.program().to_sdfg(simplify=simplify)


def module_path(short_name: str) -> str:
    """Canonical dotted name for a kernel's ``_dace.py`` (a stable sys.modules cache key)."""
    *dirs, module_name = short_name.split("/")
    return f"hpcagent_bench.benchmarks.{'.'.join(dirs)}.{module_name}_dace"


def track_names(track: str | None) -> list[str]:
    """Short names of the corpus kernels of ``track``, or of every track."""
    from hpcagent_bench.spec import KERNELS

    return [name for name in KERNELS if track is None or name.startswith(f"{track}/")]


def generate_dace_file(short_name: str) -> None:
    """Regenerate hpcagent_bench's gitignored ``_dace.py`` of a :data:`DACE_TRACKS` kernel if it is missing."""
    from hpcagent_bench import autogen

    if short_name.split("/", 1)[0] in DACE_TRACKS:
        autogen.ensure(short_name, ("dace",))


def iter_dace_kernels(track: str | None = None) -> Iterator[CorpusKernel]:
    """Every corpus kernel with a ``_dace.py``, of ``track`` or of all tracks."""
    from hpcagent_bench.spec import KERNELS, BenchSpec

    for short_name in track_names(track):
        module_name = short_name.rsplit("/", 1)[-1]
        dace_file = KERNELS[short_name].parent / f"{module_name}_dace.py"
        if not dace_file.exists():
            generate_dace_file(short_name)
        if not dace_file.exists():
            continue
        yield CorpusKernel(
            short_name=short_name,
            module_path=module_path(short_name),
            dace_file=dace_file,
            spec=BenchSpec.load(short_name),
        )


def materialize_dace_corpus() -> None:
    """Generate every missing ``_dace.py``; run it once before parallel workers, which would race the write."""
    for short_name in track_names(None):
        generate_dace_file(short_name)


def preset_sizes(kernel: CorpusKernel, preset: str) -> dict[str, int]:
    """The integer symbol sizes of one preset in the kernel's manifest."""
    rung = kernel.spec.parameters.get(preset, {})
    return {sym: size for sym, size in rung.items() if isinstance(size, int) and not isinstance(size, bool)}
