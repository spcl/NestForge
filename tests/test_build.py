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

import dace

pytest.importorskip("optarena")
pytestmark = pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not on PATH")

from nestforge.corpus import iter_dace_kernels
from nestforge.strategies import get_strategy
from nestforge.extract import extract_nest_to_sdfg
from nestforge.translate import prepare
from nestforge.arena import make_inputs, run_oracle
from nestforge.build import (build_sdfg, dace_runtime_include, OpenMPRuntime, compiler_family, LIBOMP,
                             ArenaConfig, PrunedConfig, prune_to_valid_combinations, resolve_runtime)


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


def _parallel_axpy_sdfg(name="paxpy"):
    """A minimal SDFG with ONE genuinely parallel map (``CPU_Multicore`` -> ``#pragma omp parallel
    for``): ``Z[i] = X[i] + Y[i]``. Hermetic (no corpus dependency) so the cross-compiler OpenMP link
    matrix is tested on a guaranteed-parallel loop, not on whatever schedule a corpus kernel happens to
    carry."""
    N = dace.symbol("N", dace.int64)
    sdfg = dace.SDFG(name)
    for a in ("X", "Y", "Z"):
        sdfg.add_array(a, [N], dace.float64)
    st = sdfg.add_state()
    me, mx = st.add_map("m", {"i": "0:N"}, schedule=dace.ScheduleType.CPU_Multicore)
    t = st.add_tasklet("t", {"x", "y"}, {"z"}, "z = x + y")
    st.add_memlet_path(st.add_read("X"), me, t, dst_conn="x", memlet=dace.Memlet("X[i]"))
    st.add_memlet_path(st.add_read("Y"), me, t, dst_conn="y", memlet=dace.Memlet("Y[i]"))
    st.add_memlet_path(t, mx, st.add_write("Z"), src_conn="z", memlet=dace.Memlet("Z[i]"))
    return sdfg


def test_parallel_map_emits_omp_pragma():
    """The sanity nest is actually parallel: DaCe lowers the ``CPU_Multicore`` map to an OpenMP pragma in
    the generated C++ (so the cross-compiler tests below really do exercise the OpenMP runtime link)."""
    from nestforge.build import generate_program_folder
    frame, _ = generate_program_folder(_parallel_axpy_sdfg(), Path(tempfile.mkdtemp(prefix="nf_omp_src_")))
    assert "#pragma omp parallel for" in frame.read_text()


# gcc is the driver; each of these compilers builds the SAME parallel nest as a node library, and every
# node library links against the one mandated OpenMP runtime (libomp / its ABI-equal native kmpc runtime
# for nvc++). This is the mixed-compiler / single-runtime sanity matrix: prove each compiler emits a
# correct parallel loop that links + runs on libomp. Missing toolchains skip (icpx needs oneapi setvars).
@pytest.mark.parametrize("compiler", ["g++", "clang++", "nvc++", "icpx"])
def test_parallel_loop_links_on_libomp_across_compilers(compiler):
    if shutil.which(compiler) is None:
        pytest.skip(f"{compiler} not on PATH")
    assert LIBOMP.compatible(compiler), f"{compiler} must be able to link libomp (kmpc/gomp)"
    n = 256
    x, y = np.random.default_rng(0).random(n), np.random.default_rng(1).random(n)
    buf = {"X": x.copy(), "Y": y.copy(), "Z": np.zeros(n)}
    # Compiler-neutral flags only (no -march=native: nvc++ spells it -tp); OpenMP is the separate axis.
    built = build_sdfg(_parallel_axpy_sdfg(), Path(tempfile.mkdtemp(prefix="nf_par_")), compiler=compiler,
                       flags=["-O2", "-fPIC", "-shared", "-std=c++14"], openmp=LIBOMP)
    built.run(buf, {"N": n})
    np.testing.assert_allclose(buf["Z"], x + y, rtol=1e-12, atol=1e-12)


def test_prune_selecting_libgomp_discards_kmpc_compilers():
    """User rule: select libgomp (GOMP-only) and every kmpc compiler is discarded -- clang++ AND nvc++
    both emit ``__kmpc_*`` which libgomp lacks; only gcc (which emits ``GOMP_*``) survives. Each drop
    warns. (probe off: pure ABI logic, independent of what is installed.)"""
    cfg = ArenaConfig(compilers=["g++", "clang++", "nvc++"], runtimes=["libgomp"])
    with pytest.warns(UserWarning, match="nvc"):
        pruned = prune_to_valid_combinations(cfg, probe_compilers=False, probe_runtimes=False)
    assert pruned.compilers == ["g++"]
    assert pruned.runtimes == ["libgomp"]
    assert pruned.combos == [("g++", "libgomp")]


def test_prune_removes_runtime_with_no_compatible_compiler():
    """The "remove runtimes by default" step: with only nvc++ (kmpc) as a compiler, a libgomp in the
    runtime list is compatible with no remaining compiler and is dropped with a warning; the kmpc libomp
    stays. So nvc++ is forced onto libomp -- the "nvc++ -> must use libomp" rule."""
    cfg = ArenaConfig(compilers=["nvc++"], runtimes=["libomp", "libgomp"])
    with pytest.warns(UserWarning, match="incompatible"):
        pruned = prune_to_valid_combinations(cfg, probe_compilers=False, probe_runtimes=False)
    assert pruned.runtimes == ["libomp"]
    assert pruned.combos == [("nvc++", "libomp")]
    # invariant: every surviving combo is ABI-valid.
    assert all(resolve_runtime(r).compatible(c) for c, r in pruned.combos)


def test_prune_removes_uninstalled_compiler_with_warning():
    """A compiler not on PATH is dropped with a warning (real probe: the bogus name cannot exist)."""
    cfg = ArenaConfig(compilers=["g++", "nf-not-a-real-compiler"], runtimes=["libomp"])
    with pytest.warns(UserWarning, match="not on PATH"):
        pruned = prune_to_valid_combinations(cfg, probe_compilers=True, probe_runtimes=False)
    assert pruned.compilers == ["g++"]


def test_prune_removes_uninstalled_runtime_with_warning(monkeypatch):
    """A runtime whose library is not found is dropped with a warning. libnvomp is marked absent to make
    the test host-independent."""
    import nestforge.build as B
    real = B.runtime_installed
    monkeypatch.setattr(B, "runtime_installed", lambda rt: rt.soname != "nvomp" and real(rt))
    cfg = ArenaConfig(compilers=["nvc++"], runtimes=["libomp", "libnvomp"])
    with pytest.warns(UserWarning, match="not installed"):
        pruned = prune_to_valid_combinations(cfg, probe_compilers=False, probe_runtimes=True)
    assert "libnvomp" not in pruned.runtimes and "libomp" in pruned.runtimes


def test_prune_default_config_yields_gcc_on_libomp():
    """The default arena (g++/clang++/nvc++/icpx x libomp) pruned on this machine keeps at least the
    gcc-on-libomp combo; uninstalled toolchains (e.g. icpx without setvars) just warn and drop out (not
    asserted here since which toolchains are present is host-dependent)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pruned = prune_to_valid_combinations(ArenaConfig())
    assert ("g++", "libomp") in pruned.combos
    assert all(resolve_runtime(r).compatible(c) for c, r in pruned.combos)


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
