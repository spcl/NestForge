"""nest-forge owns the DaCe build (BUILD.md): generate DaCe's C++, compile+link it ourselves with a
chosen compiler+flags, and call it via ctypes with manual init/program/exit -- not ``dace.compile``.

These tests build real corpus nests through :mod:`nestforge.build` and check the owned-built kernel
matches the numpy oracle, exercising: the ``generate_program_folder`` source-tree layout + our own
compile, the ``__dace_init``/``__program``/``__dace_exit`` call sequence, and per-parameter ctype
marshaling (a size symbol is ``int`` or ``int64_t`` per its DaCe dtype; a Scalar passes by value).
"""
import ctypes.util
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("optarena")
pytestmark = pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not on PATH")

from nestforge.corpus import iter_dace_kernels
from nestforge.strategies import get_strategy
from nestforge.extract import extract_nest_to_sdfg
from nestforge.translate import prepare
from nestforge.arena import make_inputs, run_oracle
from nestforge.build import build_sdfg, dace_runtime_include, OpenMPRuntime, compiler_family


def _kernels():
    return {k.short_name: k for k in iter_dace_kernels()}


def _first_nest(short):
    sdfg = _kernels()[short].to_sdfg(simplify=True)
    parent, node = get_strategy("skip-taskloops")(sdfg)[0]
    return extract_nest_to_sdfg(parent, node, name="nest")


def _owned_build_matches_oracle(short, size=48):
    boundary = _first_nest(short)
    shape_syms = {s for s in boundary.symbols
                  if any(s in str(d.shape) for d in boundary.standalone_sdfg.arrays.values())}
    sizes = {s: (size if s in shape_syms else 0) for s in boundary.symbols}
    inputs = make_inputs(boundary, sizes, seed=0)
    prep = prepare(boundary, "k", Path(tempfile.mkdtemp()))
    oracle = run_oracle(prep, boundary, inputs, sizes)

    built = build_sdfg(boundary.standalone_sdfg, Path(tempfile.mkdtemp(prefix="nf_build_")))
    buf = {k: v.copy() for k, v in inputs.items()}
    built.run(buf, sizes)  # init -> program -> exit
    for o in oracle:
        np.testing.assert_allclose(buf[o], oracle[o], rtol=1e-9, atol=1e-9, equal_nan=True)


def test_dace_runtime_include_exists():
    assert (dace_runtime_include() / "dace" / "dace.h").exists()


def test_owned_build_gemm_matches_oracle():
    """gemm: int64_t size symbols + a Scalar (alpha/beta) passed by value through the owned build."""
    _owned_build_matches_oracle("hpc/dense_linear_algebra/gemm/gemm")


def test_owned_build_jacobi_matches_oracle():
    """jacobi_1d: an ``int`` (not int64_t) size symbol -- guards the per-parameter ctype marshaling."""
    _owned_build_matches_oracle("hpc/structured_grids/jacobi_1d/jacobi_1d")


def test_openmp_runtime_is_a_separate_per_compiler_flag_axis():
    """The OpenMP runtime is one configurable knob that maps to the right flag PER COMPILER, so a set of
    node libraries built with different compilers all target the SAME runtime -- the mixed-compiler
    single-runtime contract from PARALLEL.md."""
    rt = OpenMPRuntime()  # default libomp
    assert compiler_family("gfortran") == "gnu" and compiler_family("flang") == "llvm"
    assert compiler_family("icx") == "llvm" and compiler_family("icc") == "intel-classic"
    # LLVM family selects the runtime BY NAME (the user's example: flang -fopenmp=libomp).
    assert rt.compile_flags("flang") == ["-fopenmp=libomp"]
    assert rt.compile_flags("clang++") == ["-fopenmp=libomp"]
    assert rt.compile_flags("icx") == ["-fopenmp=libomp"]
    # gnu emits GOMP calls at compile and links the mandated runtime explicitly (not -fopenmp -> libgomp).
    assert rt.compile_flags("g++") == ["-fopenmp"] and rt.link_flags("g++") == ["-lomp"]
    # intel-classic / nvidia use their own spellings.
    assert rt.compile_flags("icc") == ["-qopenmp"] and rt.compile_flags("nvc") == ["-mp"]
    # a lib_dir threads onto the link line.
    assert "-L/opt/omp/lib" in OpenMPRuntime(lib_dir="/opt/omp/lib").link_flags("g++")


def test_openmp_runtime_registry_covers_the_popular_runtimes():
    """The four popular runtimes are ready knobs: libgomp (GNU), libomp (LLVM), libiomp5 (Intel; ABI-
    compatible with libomp), libnvomp (NVIDIA HPC, via nvc -mp only)."""
    from nestforge.build import OPENMP_RUNTIMES, LIBGOMP, LIBIOMP5
    assert set(OPENMP_RUNTIMES) == {"libomp", "libgomp", "libiomp5", "libnvomp"}
    assert LIBIOMP5.link_flags("g++") == ["-liomp5"]  # gcc object on Intel's runtime (has GOMP compat)
    assert LIBGOMP.link_flags("g++") == ["-lgomp"]


def test_openmp_abi_compatibility_is_enforced():
    """A runtime is usable with a compiler only if it implements the ABI the compiler emits. The kmpc
    compilers (clang/flang/icx AND nvc++) emit ``__kmpc_*`` -> they can use libomp/libiomp5/libnvomp but
    NOT libgomp (GOMP-only); gcc emits ``GOMP_*`` -> it can use all four. Mismatches raise, not silently
    mis-link."""
    from nestforge.build import LIBOMP, LIBGOMP, LIBIOMP5, LIBNVOMP
    # nvc++ (and clang) can use the kmpc runtimes...
    assert LIBOMP.compatible("nvc++") and LIBIOMP5.compatible("nvc++") and LIBNVOMP.compatible("nvc++")
    assert LIBOMP.compatible("clang++")
    # ...but NOT libgomp (it lacks __kmpc_*).
    assert not LIBGOMP.compatible("nvc++") and not LIBGOMP.compatible("clang++")
    with pytest.raises(ValueError, match="kmpc"):
        LIBGOMP.compile_flags("nvc++")
    with pytest.raises(ValueError, match="kmpc"):
        LIBGOMP.link_flags("clang++")
    # gcc (GOMP) works against every runtime, since libomp/libiomp5/libnvomp carry a GOMP-compat layer.
    for rt in (LIBOMP, LIBGOMP, LIBIOMP5, LIBNVOMP):
        assert rt.compatible("g++")


@pytest.mark.skipif(ctypes.util.find_library("omp") is None, reason="libomp not installed")
def test_gcc_compiled_kernel_links_against_libomp():
    """A g++-compiled kernel (which emits GOMP_* calls under -fopenmp) links + runs against libomp via
    libomp's GOMP-compat ABI -- the concrete mixed-compiler / one-runtime case: a GCC node library can
    share the same libomp a clang/flang node library uses. Builds gemm with openmp=OpenMPRuntime() on
    g++ and checks it still matches the oracle."""
    boundary = _first_nest("hpc/dense_linear_algebra/gemm/gemm")
    shape_syms = {s for s in boundary.symbols
                  if any(s in str(d.shape) for d in boundary.standalone_sdfg.arrays.values())}
    sizes = {s: (32 if s in shape_syms else 0) for s in boundary.symbols}
    inputs = make_inputs(boundary, sizes, seed=0)
    prep = prepare(boundary, "k", Path(tempfile.mkdtemp()))
    oracle = run_oracle(prep, boundary, inputs, sizes)
    built = build_sdfg(boundary.standalone_sdfg, Path(tempfile.mkdtemp(prefix="nf_omp_")),
                       compiler="g++", openmp=OpenMPRuntime())  # gcc object on libomp
    buf = {k: v.copy() for k, v in inputs.items()}
    built.run(buf, sizes)
    for o in oracle:
        np.testing.assert_allclose(buf[o], oracle[o], rtol=1e-9, atol=1e-9, equal_nan=True)


def test_owned_build_reusable_handle_program():
    """After one init, __program can be called repeatedly in place (the timing path) on one handle."""
    boundary = _first_nest("hpc/dense_linear_algebra/gemm/gemm")
    shape_syms = {s for s in boundary.symbols
                  if any(s in str(d.shape) for d in boundary.standalone_sdfg.arrays.values())}
    sizes = {s: (32 if s in shape_syms else 0) for s in boundary.symbols}
    inputs = make_inputs(boundary, sizes, seed=1)
    built = build_sdfg(boundary.standalone_sdfg, Path(tempfile.mkdtemp(prefix="nf_build_")))
    buf = {k: v.copy() for k, v in inputs.items()}
    built._init(sizes)
    try:
        for _ in range(5):
            built.program(buf, sizes)  # repeated in-place calls on the same state handle
    finally:
        built.close()
