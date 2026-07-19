"""nest-forge: extract DaCe loop-/map-nests and offload them to external compilers via an arena.

Two pieces of the optarena dependency are surfaced natively: :mod:`nestforge.translator`
(the numpy -> C/C++/Fortran translator) and :mod:`nestforge.corpus` (the npbench/polybench kernel
corpus). Everything else is nest-forge's own.
"""

# Pre-warm dace's ``passes`` package before any nest-forge submodule pulls ``dace.transformation.
# interstate``: on the extended branch ``passes.canonicalize -> vectorization -> interstate`` forms an
# import cycle that only resolves when ``passes`` starts loading first -- importing ``interstate`` first
# dies on a partially initialized module (``cannot import name 'InlineMultistateSDFG'``). Fixing the
# order here once covers every entry point (bare ``import nestforge.tsvc``, the perf drivers, tests).
import dace.transformation.passes  # noqa: F401

from nestforge.corpus import CorpusKernel, dace_kernel_names, iter_dace_kernels
from nestforge.extract import Boundary, extract_nest_to_sdfg
from nestforge.fusion import (FusionMove, FusionStrategy, apply_fusion, enumerate_fusions, fission_to_statements,
                              fusion_strategy_names, get_fusion_strategy, maximal_fusion, map_fission_moves,
                              register_fusion_strategy)
from nestforge.strategies import Strategy, get_strategy, outer, register_strategy
from nestforge.translator import BenchSpec, translate

__all__ = [
    "Boundary",
    "extract_nest_to_sdfg",
    # Phase 1: fusion granularity
    "FusionStrategy",
    "register_fusion_strategy",
    "get_fusion_strategy",
    "fusion_strategy_names",
    "maximal_fusion",
    "FusionMove",
    "enumerate_fusions",
    "apply_fusion",
    "fission_to_statements",
    "map_fission_moves",
    # Phase 2: offload granularity
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
