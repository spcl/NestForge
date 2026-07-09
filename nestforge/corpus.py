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

import importlib
from dataclasses import dataclass
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
    module_path: str  # importable module, e.g. "optarena.benchmarks...gemm.gemm_dace"
    spec: BenchSpec

    def program(self):
        """The kernel's ``@dace.program`` object."""
        _set_precision_fp64()
        module = importlib.import_module(self.module_path)
        for value in vars(module).values():
            if isinstance(value, dace.frontend.python.parser.DaceProgram):
                return value
        raise LookupError(f"no @dace.program found in {self.module_path}")

    def to_sdfg(self, simplify: bool = True) -> dace.SDFG:
        return self.program().to_sdfg(simplify=simplify)


def _module_path(short_name: str) -> str:
    """Dotted import path of a kernel's ``_dace.py`` from its registry short name."""
    ypath = KERNELS[short_name]
    rel = ypath.parent.relative_to(_benchmarks_root())
    module_name = short_name.rsplit("/", 1)[-1]
    dotted = ".".join(rel.parts)
    return f"optarena.benchmarks.{dotted}.{module_name}_dace"


def _benchmarks_root():
    import optarena.benchmarks as b
    import pathlib
    return pathlib.Path(next(iter(b.__path__)))


def iter_dace_kernels(track: Optional[str] = None) -> Iterator[CorpusKernel]:
    """Yield every corpus kernel that ships a ``_dace.py`` impl, optionally filtered by track.

    :param track: ``"hpc"``, ``"ml"``, ``"foundation"``, or ``None`` for all.
    """
    for short_name in KERNELS:
        if track is not None and not short_name.startswith(f"{track}/"):
            continue
        ypath = KERNELS[short_name]
        module_name = short_name.rsplit("/", 1)[-1]
        if not (ypath.parent / f"{module_name}_dace.py").exists():
            continue
        yield CorpusKernel(short_name=short_name, module_path=_module_path(short_name), spec=BenchSpec.load(short_name))


def dace_kernel_names(track: Optional[str] = None) -> List[str]:
    return [k.short_name for k in iter_dace_kernels(track)]
