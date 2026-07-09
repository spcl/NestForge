"""nest-forge: extract DaCe loop-/map-nests and offload them to external compilers via an arena.

Two pieces of the vendored optarena submodule are surfaced natively: :mod:`nestforge.translator`
(the numpy -> C/C++/Fortran translator) and :mod:`nestforge.corpus` (the npbench/polybench kernel
corpus). Everything else is nest-forge's own.
"""

from nestforge.corpus import CorpusKernel, dace_kernel_names, iter_dace_kernels
from nestforge.extract import Boundary, extract_nest_to_sdfg
from nestforge.strategies import Strategy, get_strategy, outer, register_strategy
from nestforge.translator import BenchSpec, translate

__all__ = [
    "Boundary",
    "extract_nest_to_sdfg",
    "Strategy",
    "outer",
    "register_strategy",
    "get_strategy",
    # native optarena surfaces
    "translate",
    "BenchSpec",
    "CorpusKernel",
    "iter_dace_kernels",
    "dace_kernel_names",
]
