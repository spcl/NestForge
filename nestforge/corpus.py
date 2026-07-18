"""Load real npbench/polybench kernels from the installed ``optarena`` package as SDFGs.

optarena ships each kernel as ``<name>_numpy.py`` (oracle) + ``<name>.yaml`` (BenchSpec) and, for the
HPC/ML tracks, a ``<name>_dace.py`` holding a ``@dace.program`` -- import it, ``to_sdfg`` it, feed it
to the lowering pass.

Kernels bind optarena's ``dc_float`` precision global at import time, so it must be stamped to fp64
before any kernel module imports.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import dace

from optarena import autogen
from optarena.spec import KERNELS, BenchSpec

#: Tracks whose ``_dace.py`` optarena generates on demand (gitignored, never committed). foundation
#: (TSVC) is sourced through :mod:`nestforge.tsvc`, not this corpus, so it's never materialized here.
DACE_TRACKS = ("hpc", "ml")


def set_precision_fp64() -> None:
    """Fix optarena's kernel dtype global to float64 before kernel modules import it."""
    import optarena.frameworks.dace_framework as dfw
    dfw.dc_float = dace.float64
    dfw.dc_complex_float = dace.complex128


@dataclass
class CorpusKernel:
    """One optarena kernel that ships a ``@dace.program`` dace impl."""
    short_name: str  # registry key, e.g. "hpc/dense_linear_algebra/gemm/gemm"
    module_path: str  # canonical dotted name, used only as the sys.modules cache key
    dace_file: Path  # the kernel's ``_dace.py`` on disk (source of truth)
    spec: BenchSpec

    def module(self):
        """Import the kernel's ``_dace.py`` by file path.

        Loading by path (not ``import_module``) sidesteps ``optarena.benchmarks`` namespace-package
        resolution, which can non-deterministically bind to a stray duplicate ``benchmarks/`` root.
        """
        if self.module_path in sys.modules:
            return sys.modules[self.module_path]
        spec = importlib.util.spec_from_file_location(self.module_path, self.dace_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[self.module_path] = module
        spec.loader.exec_module(module)
        return module

    def program(self):
        """The kernel's *entry* ``@dace.program``, selected by the manifest's ``func_name``.

        A module often defines helper programs before it and a ``*_gpu`` variant after, so neither
        "first" nor "last" is reliable; mirrors optarena's own ``_import_kernel``.
        """
        set_precision_fp64()
        module = self.module()
        entry = vars(module).get(self.spec.func_name)
        if isinstance(entry, dace.frontend.python.parser.DaceProgram):
            return entry
        programs = [v for v in vars(module).values() if isinstance(v, dace.frontend.python.parser.DaceProgram)]
        if not programs:
            raise LookupError(f"no @dace.program found in {self.dace_file}")
        return programs[-1]  # entry is defined after its helpers

    def to_sdfg(self, simplify: bool = True) -> dace.SDFG:
        return self.program().to_sdfg(simplify=simplify)


def module_path(short_name: str) -> str:
    """Canonical dotted name for a kernel's ``_dace.py`` (a stable sys.modules cache key)."""
    *dirs, module_name = short_name.split("/")
    return f"optarena.benchmarks.{'.'.join(dirs)}.{module_name}_dace"


def iter_dace_kernels(track: Optional[str] = None) -> Iterator[CorpusKernel]:
    """Yield every corpus kernel that ships a ``_dace.py`` impl, optionally filtered by track.

    :param track: ``"hpc"``, ``"ml"``, ``"foundation"``, or ``None`` for all.
    """
    for short_name in KERNELS:
        if track is not None and not short_name.startswith(f"{track}/"):
            continue
        module_name = short_name.rsplit("/", 1)[-1]
        dace_file = KERNELS[short_name].parent / f"{module_name}_dace.py"
        if not dace_file.exists() and short_name.split("/", 1)[0] in DACE_TRACKS:
            autogen.ensure(short_name, ("dace", ))  # regenerate optarena's gitignored _dace.py on demand
        if not dace_file.exists():
            continue
        yield CorpusKernel(short_name=short_name,
                           module_path=module_path(short_name),
                           dace_file=dace_file,
                           spec=BenchSpec.load(short_name))


def materialize_dace_corpus(track: Optional[str] = None) -> None:
    """Generate every missing ``_dace.py`` for the dace-bearing tracks up front (gitignored, so a fresh
    checkout has none). Safe to call repeatedly. Call ONCE, serially, before a parallel test run:
    :func:`autogen.ensure` writes non-atomically, so concurrent xdist workers must not race the same
    kernel."""
    for short_name in KERNELS:
        if short_name.split("/", 1)[0] not in DACE_TRACKS:
            continue
        if track is not None and not short_name.startswith(f"{track}/"):
            continue
        autogen.ensure(short_name, ("dace", ))


def dace_kernel_names(track: Optional[str] = None) -> List[str]:
    return [k.short_name for k in iter_dace_kernels(track)]
