"""Load real npbench/polybench kernels from the installed ``optarena`` package as SDFGs.

optarena ships each kernel as ``<name>_numpy.py`` (oracle) + ``<name>.yaml`` (BenchSpec) and, for
the HPC/ML tracks, a ready ``<name>_dace.py`` holding a ``@dace.program``. Those dace impls are the
zero-friction corpus for nest-forge: import the program, ``to_sdfg`` it, and feed it to the lowering
pass. The numpy oracle + yaml already match what nest-forge itself emits, so the same kernel doubles
as a correctness reference.

``dc_float`` in optarena's dace kernels is a module global fixed at a chosen precision; we stamp it
to fp64 before importing any kernel module (kernels bind it at import time via ``from ... import``).
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import dace

from optarena.spec import KERNELS, BenchSpec


def _set_precision_fp64() -> None:
    """Fix optarena's kernel dtype global to float64 before kernel modules import it."""
    import optarena.infrastructure.dace_framework as dfw
    dfw.dc_float = dace.float64
    dfw.dc_complex_float = dace.complex128


@dataclass
class CorpusKernel:
    """One optarena kernel that ships a ``@dace.program`` dace impl."""
    short_name: str  # registry key, e.g. "hpc/dense_linear_algebra/gemm/gemm"
    module_path: str  # canonical dotted name, used only as the sys.modules cache key
    dace_file: Path  # the kernel's ``_dace.py`` on disk (source of truth)
    spec: BenchSpec

    def _module(self):
        """Import the kernel's ``_dace.py`` by file path.

        Loading by path (not ``import_module``) sidesteps ``optarena.benchmarks`` namespace-package
        resolution, which can non-deterministically bind to a stray/duplicate ``benchmarks/`` root
        that lacks the kernel's subpackage. The kernel's own ``from optarena.infrastructure ...``
        imports are unambiguous and still resolve normally.
        """
        if self.module_path in sys.modules:
            return sys.modules[self.module_path]
        spec = importlib.util.spec_from_file_location(self.module_path, self.dace_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[self.module_path] = module
        spec.loader.exec_module(module)
        return module

    def program(self):
        """The kernel's *entry* ``@dace.program`` object.

        A dace module often defines helper programs (``relu``, ``conv2d``, ...) before the kernel,
        and sometimes a ``*_gpu`` variant after it, so the entry is selected by the manifest's
        ``func_name`` (mirrors optarena's own ``_import_kernel``) rather than "the first/last one".
        """
        _set_precision_fp64()
        module = self._module()
        entry = vars(module).get(self.spec.func_name)
        if isinstance(entry, dace.frontend.python.parser.DaceProgram):
            return entry
        programs = [v for v in vars(module).values() if isinstance(v, dace.frontend.python.parser.DaceProgram)]
        if not programs:
            raise LookupError(f"no @dace.program found in {self.dace_file}")
        return programs[-1]  # entry is defined after its helpers

    def to_sdfg(self, simplify: bool = True) -> dace.SDFG:
        return self.program().to_sdfg(simplify=simplify)


def _module_path(short_name: str) -> str:
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
        if not dace_file.exists():
            continue
        yield CorpusKernel(short_name=short_name,
                           module_path=_module_path(short_name),
                           dace_file=dace_file,
                           spec=BenchSpec.load(short_name))


def dace_kernel_names(track: Optional[str] = None) -> List[str]:
    return [k.short_name for k in iter_dace_kernels(track)]
