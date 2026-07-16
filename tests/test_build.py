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
from dace.sdfg import nodes
from nestforge.build import (build_sdfg, dace_runtime_include, OpenMPRuntime, compiler_family, LIBOMP, LIBNVOMP,
                             compare_link_modes, LinkTimings, available_linkers, fastest_linker, linker_supported,
                             VECTOR_LIBS, SLEEF, LIBMVEC, SVML, vectorlib_installed, BuildOptions, set_fast_libnodes,
                             runtime_installed, config_has, codegen_impls_available, codegen_config,
                             default_codegen_impl, CODEGEN_IMPLS, parse_params)


def kernels():
    return {k.short_name: k for k in iter_dace_kernels()}


def first_nest(short):
    sdfg = kernels()[short].to_sdfg(simplify=True)
    parent, node = get_strategy("skip-taskloops")(sdfg)[0]
    return extract_nest_to_sdfg(parent, node, name="nest")


def owned_build_matches_oracle(short, size=48, opts=None):
    boundary = first_nest(short)
    shape_syms = {
        s
        for s in boundary.symbols if any(s in str(d.shape) for d in boundary.standalone_sdfg.arrays.values())
    }
    sizes = {s: (size if s in shape_syms else 0) for s in boundary.symbols}
    inputs = make_inputs(boundary, sizes, seed=0)
    prep = prepare(boundary, "k", Path(tempfile.mkdtemp()))
    oracle = run_oracle(prep, boundary, inputs, sizes)

    built = build_sdfg(boundary.standalone_sdfg, Path(tempfile.mkdtemp(prefix="nf_build_")), opts)
    buf = {k: v.copy() for k, v in inputs.items()}
    built.run(buf, sizes)  # init -> program -> exit
    for o in oracle:
        np.testing.assert_allclose(buf[o], oracle[o], rtol=1e-9, atol=1e-9, equal_nan=True)
    return built


def test_dace_runtime_include_exists():
    assert (dace_runtime_include() / "dace" / "dace.h").exists()


def test_owned_build_gemm_matches_oracle():
    """gemm: int64_t size symbols + a Scalar (alpha/beta) passed by value through the owned build."""
    owned_build_matches_oracle("hpc/dense_linear_algebra/gemm/gemm")


def test_owned_build_jacobi_matches_oracle():
    """jacobi_1d: an ``int`` (not int64_t) size symbol -- guards the per-parameter ctype marshaling."""
    owned_build_matches_oracle("hpc/structured_grids/jacobi_1d/jacobi_1d")


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
    # intel-classic and nvidia link ONLY their native runtimes (icc -qopenmp -> libiomp5, nvc -mp ->
    # libnvomp), so libomp is not compatible with either -- their spellings are checked on those runtimes.
    from nestforge.build import LIBIOMP5
    assert LIBIOMP5.compile_flags("icc") == ["-qopenmp"] and LIBIOMP5.link_flags("icc") == ["-qopenmp"]
    assert LIBNVOMP.compile_flags("nvc") == ["-mp"] and LIBNVOMP.link_flags("nvc") == ["-mp"]
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
    """A runtime is usable with a compiler only if the compiler can actually LINK it -- which depends on
    HOW the family selects a runtime, not ABI alone. gcc links any gomp-capable runtime via -l<soname>.
    LLVM (clang/flang/icx) name-selects only libomp/libgomp/libiomp5, restricted to its kmpc ABI -> libomp/
    libiomp5, NOT libgomp (no __kmpc_*) and NOT libnvomp (not name-selectable). Classic icc (-qopenmp) and
    nvc++ (-mp) hard-link their native libiomp5 / libnvomp ALONE. Mismatches raise, not silently mis-link."""
    from nestforge.build import LIBOMP, LIBGOMP, LIBIOMP5, LIBNVOMP
    # nvc++ / icc link ONLY their native runtimes.
    assert LIBNVOMP.compatible("nvc++") and not LIBOMP.compatible("nvc++") and not LIBIOMP5.compatible("nvc++")
    assert LIBIOMP5.compatible("icc") and not LIBOMP.compatible("icc") and not LIBGOMP.compatible("icc")
    # clang name-selects libomp/libiomp5 but NOT libgomp (no __kmpc_*) and NOT libnvomp (unreachable by name).
    assert LIBOMP.compatible("clang++") and LIBIOMP5.compatible("clang++")
    assert not LIBGOMP.compatible("clang++") and not LIBNVOMP.compatible("clang++")
    with pytest.raises(ValueError, match="libnvomp"):  # nvidia gets the -mp / libnvomp-only message
        LIBGOMP.compile_flags("nvc++")
    with pytest.raises(ValueError, match="kmpc"):  # clang + libgomp: wrong ABI
        LIBGOMP.link_flags("clang++")
    with pytest.raises(ValueError, match="name-selectable"):  # clang + libnvomp: right ABI, not name-selectable
        LIBNVOMP.compile_flags("clang++")
    # gcc (GOMP) works against every runtime, since libomp/libiomp5/libnvomp carry a GOMP-compat layer.
    for rt in (LIBOMP, LIBGOMP, LIBIOMP5, LIBNVOMP):
        assert rt.compatible("g++")


@pytest.mark.skipif(ctypes.util.find_library("omp") is None, reason="libomp not installed")
def test_gcc_compiled_kernel_links_against_libomp():
    """A g++-compiled kernel (which emits GOMP_* calls under -fopenmp) links + runs against libomp via
    libomp's GOMP-compat ABI -- the concrete mixed-compiler / one-runtime case: a GCC node library can
    share the same libomp a clang/flang node library uses. Builds gemm with openmp=OpenMPRuntime() on
    g++ and checks it still matches the oracle."""
    boundary = first_nest("hpc/dense_linear_algebra/gemm/gemm")
    shape_syms = {
        s
        for s in boundary.symbols if any(s in str(d.shape) for d in boundary.standalone_sdfg.arrays.values())
    }
    sizes = {s: (32 if s in shape_syms else 0) for s in boundary.symbols}
    inputs = make_inputs(boundary, sizes, seed=0)
    prep = prepare(boundary, "k", Path(tempfile.mkdtemp()))
    oracle = run_oracle(prep, boundary, inputs, sizes)
    built = build_sdfg(boundary.standalone_sdfg, Path(tempfile.mkdtemp(prefix="nf_omp_")),
                       BuildOptions(compiler="g++", openmp=OpenMPRuntime()))  # gcc object on libomp
    buf = {k: v.copy() for k, v in inputs.items()}
    built.run(buf, sizes)
    for o in oracle:
        np.testing.assert_allclose(buf[o], oracle[o], rtol=1e-9, atol=1e-9, equal_nan=True)


def parallel_axpy_sdfg(name="paxpy"):
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
    frame, _ = generate_program_folder(parallel_axpy_sdfg(), Path(tempfile.mkdtemp(prefix="nf_omp_src_")))
    assert "#pragma omp parallel for" in frame.read_text()


# gcc is the driver; each of these compilers builds the SAME parallel nest as a node library, each
# linking the ONE runtime it can: libomp for gcc/clang/icx (kmpc+gomp), libnvomp for nvc++ (which links
# only its native runtime via -mp -- see the C2 fix). This is the mixed-compiler / single-runtime sanity
# matrix: prove each compiler emits a correct parallel loop that links + runs. Missing toolchains skip.
@pytest.mark.parametrize(
    "compiler",
    [
        "g++",
        "clang++",
        pytest.param("nvc++", marks=pytest.mark.integration),  # vendor compiler: absent on the CI runner
        pytest.param("icpx", marks=pytest.mark.integration),  # vendor compiler: absent on the CI runner
    ])
def test_parallel_loop_links_openmp_across_compilers(compiler):
    if shutil.which(compiler) is None:
        pytest.skip(f"{compiler} not on PATH")
    # nvc++ links only libnvomp (its -mp native runtime); everyone else uses the mandated libomp.
    rt = LIBNVOMP if compiler_family(compiler) == "nvidia" else LIBOMP
    if not runtime_installed(rt):
        pytest.skip(f"{rt.name} not installed here (no OpenMP runtime on PATH/LD_LIBRARY_PATH/ldconfig)")
    assert rt.compatible(compiler), f"{compiler} must be able to link {rt.name}"
    n = 256
    x, y = np.random.default_rng(0).random(n), np.random.default_rng(1).random(n)
    buf = {"X": x.copy(), "Y": y.copy(), "Z": np.zeros(n)}
    # Compiler-neutral flags only (no -march=native: nvc++ spells it -tp); OpenMP is the separate axis.
    built = build_sdfg(parallel_axpy_sdfg(), Path(tempfile.mkdtemp(prefix="nf_par_")),
                       BuildOptions(compiler=compiler, flags=["-O2", "-fPIC", "-shared", "-std=c++20"], openmp=rt))
    built.run(buf, {"N": n})
    np.testing.assert_allclose(buf["Z"], x + y, rtol=1e-12, atol=1e-12)


def test_build_tracks_optimization_and_compile_time():
    """Every owned build records the optimization time (DaCe codegen) and the post-optimization compile
    time (the toolchain subprocess), so both are trackable per build."""
    built = build_sdfg(parallel_axpy_sdfg(), Path(tempfile.mkdtemp(prefix="nf_time_")))
    assert built.codegen_seconds > 0.0
    assert built.compile_seconds > 0.0


def test_external_linking_build_is_correct():
    """The nest built as a SEPARATE static ``.a`` (link_external) and linked into the ``.so`` runs
    identically to the monolithic build -- the external-linking path is correct, not merely timeable."""
    built = owned_build_matches_oracle("hpc/dense_linear_algebra/gemm/gemm", opts=BuildOptions(link_external=True))
    assert built.compile_seconds > 0.0
    assert (built.so_path.parent / f"lib{built.name}_nest.a").exists()  # the static node lib was produced


def test_compare_link_modes_tracks_compile_time_with_and_without_external_linking():
    """One optimization (codegen) pass, then the same frame compiled two ways: WITHOUT external linking
    (monolithic single TU) and WITH external linking (static ``.a`` -> ``.so``). All three times are
    tracked and positive -- this is the with/without-external-linking compile-time comparison."""
    t = compare_link_modes(parallel_axpy_sdfg(), Path(tempfile.mkdtemp(prefix="nf_linkmodes_")))
    assert isinstance(t, LinkTimings)
    assert t.codegen_seconds > 0.0
    assert t.compile_seconds_monolithic > 0.0
    assert t.compile_seconds_external > 0.0


def test_external_linking_with_lto_is_correct():
    """External linking + ``-flto`` (via the LTO-aware ``ar``) still matches the oracle -- LTO is the
    knob meant to recover the cross-TU inlining that external linking otherwise costs."""
    if shutil.which("gcc-ar") is None:
        pytest.skip("gcc-ar (LTO-aware archiver) not on PATH")
    owned_build_matches_oracle("hpc/dense_linear_algebra/gemm/gemm", opts=BuildOptions(link_external=True, lto=True))


def test_available_linkers_and_fastest_pick():
    """Linker discovery reports the installed fast linkers (fastest first); the picker chooses the fastest
    one the compiler is NEW ENOUGH to accept (version-gated), and never touches nvc/nvc++ (no -fuse-ld)."""
    av = available_linkers()
    assert all(Path(p).exists() for p in av.values())  # every reported linker really is on disk
    assert set(av) <= {"mold", "lld", "gold"}
    picked = fastest_linker("g++")
    if picked:
        # the pick is the fastest INSTALLED linker g++ actually supports (may skip mold on an old gcc).
        ld = picked[0].split("=", 1)[1]
        assert ld in av and linker_supported("g++", ld)
        supported = [x for x in av if linker_supported("g++", x)]
        assert ld == supported[0]  # fastest-first among the supported ones
    else:
        assert not any(linker_supported("g++", x) for x in av)  # nothing installed is supported
    assert fastest_linker("nvc++") == []  # NVIDIA keeps its default linker


def test_fastest_linker_version_gate_skips_unsupported(monkeypatch):
    """The pick is VERSION-gated: an old compiler that predates -fuse-ld=mold must not be handed mold even
    when mold is installed. Force the compiler version below mold's floor and assert mold is skipped (this
    fails if the version gate is dropped, unlike the host-dependent check above)."""
    import nestforge.build as B
    monkeypatch.setattr(B, "compiler_version", lambda c: (9, 0))  # gcc 9 < mold's (12,1) floor; >= lld/gold
    assert not B.linker_supported("g++", "mold")
    picked = B.fastest_linker("g++")
    if picked:  # whatever it fell back to (lld/gold), g++ at v9 must actually support it, and it isn't mold
        assert picked != ["-fuse-ld=mold"]
        assert B.linker_supported("g++", picked[0].split("=", 1)[1])


def test_veclib_flag_mapping_and_compatibility():
    """Each vector-math library maps to the right per-compiler-family flag, and an incompatible pairing
    raises rather than silently emitting nothing."""
    assert set(VECTOR_LIBS) == {"sleef", "libmvec", "svml"}
    # SLEEF: clang/llvm via -fveclib; gcc unsupported (no -fveclib / -mveclibabi for it).
    assert SLEEF.compile_flags("clang++") == ["-fveclib=SLEEF"] and SLEEF.link_flags("icx") == ["-lsleef"]
    assert not SLEEF.compatible("g++")
    with pytest.raises(ValueError):
        SLEEF.compile_flags("g++")
    # libmvec: clang names it; gcc uses it automatically (no compile flag) but links -lmvec.
    assert LIBMVEC.compile_flags("clang++") == ["-fveclib=libmvec"]
    assert LIBMVEC.compatible("g++") and LIBMVEC.compile_flags("g++") == [] and LIBMVEC.link_flags("g++") == ["-lmvec"]
    # SVML: icx via -fveclib, gcc via -mveclibabi.
    assert SVML.compile_flags("icx") == ["-fveclib=SVML"] and SVML.compile_flags("g++") == ["-mveclibabi=svml"]
    # NVIDIA cannot use any of these.
    assert not any(vl.compatible("nvc++") for vl in VECTOR_LIBS.values())


def test_veclib_link_flags_come_after_the_source_in_every_link_mode(monkeypatch, tmp_path):
    """ld resolves left-to-right: a -l placed BEFORE the object that references it contributes nothing, so
    the veclib would silently not be linked and its 'speedup' would just be libm. Assert the composed
    command ORDER (no compile needed) for every branch of :func:`compile`."""
    import nestforge.build as B
    cmds = []
    monkeypatch.setattr(B, "run", lambda cmd, **kw: cmds.append(list(cmd)))
    frame = tmp_path / "src" / "cpu" / "k.cpp"
    frame.parent.mkdir(parents=True)
    frame.write_text("")
    for opts in (BuildOptions(compiler="clang++",
                              veclib=SLEEF), BuildOptions(compiler="clang++", veclib=SLEEF, openmp=LIBOMP),
                 BuildOptions(compiler="clang++", veclib=SLEEF, link_external=True)):
        cmds.clear()
        B.compile(frame, tmp_path, "k", opts)
        link = [c for c in cmds if "-lsleef" in c]
        assert len(link) == 1, opts
        cmd = link[0]
        # whichever input carries the code that references the veclib symbols (source / object / archive)
        inputs = [str(frame), str(tmp_path / "k.o"), str(tmp_path / "libk_nest.a")]
        pos = [cmd.index(i) for i in inputs if i in cmd]
        assert pos and max(pos) < cmd.index("-lsleef"), cmd


def test_parse_params_strips_the_const_qualifier_only_as_a_word():
    """``const`` is a QUALIFIER, not a substring: a parameter merely NAMED ``constant`` / ``const_term``
    must keep its name, or the ctypes bind looks the array up under a mangled key."""
    params = parse_params("k_state_t *__state, const double * __restrict__ constant, const int const_term")
    assert [p.name for p in params] == ["constant", "const_term"]
    assert params[0].is_pointer and params[0].ctype == ctypes.POINTER(ctypes.c_double)
    assert not params[1].is_pointer and params[1].ctype == ctypes.c_int


def test_parse_params_refuses_an_unmapped_by_value_scalar_type():
    """An unmapped by-value type must fail LOUD: defaulting it to int64 puts a float in a GP register on
    the SysV ABI, and the callee then reads garbage with no ctypes error."""
    with pytest.raises(ValueError, match="uint64_t"):
        parse_params("k_state_t *__state, uint64_t n")


@pytest.mark.skipif(not vectorlib_installed(LIBMVEC), reason="glibc libmvec not found")
def test_veclib_libmvec_build_is_correct():
    """Building against glibc's libmvec (g++: -lmvec, no compile flag) links + runs correctly -- the
    veclib axis threads through the owned build without breaking it."""
    n = 128
    x, y = np.random.default_rng(2).random(n), np.random.default_rng(3).random(n)
    buf = {"X": x.copy(), "Y": y.copy(), "Z": np.zeros(n)}
    built = build_sdfg(parallel_axpy_sdfg(), Path(tempfile.mkdtemp(prefix="nf_vec_")),
                       BuildOptions(compiler="g++", flags=["-O3", "-march=native", "-fPIC", "-shared"], veclib=LIBMVEC))
    built.run(buf, {"N": n})
    np.testing.assert_allclose(buf["Z"], x + y, rtol=1e-12, atol=1e-12)


def test_owned_build_reusable_handle_program():
    """After one init, __program can be called repeatedly in place (the timing path) on one handle."""
    boundary = first_nest("hpc/dense_linear_algebra/gemm/gemm")
    shape_syms = {
        s
        for s in boundary.symbols if any(s in str(d.shape) for d in boundary.standalone_sdfg.arrays.values())
    }
    sizes = {s: (32 if s in shape_syms else 0) for s in boundary.symbols}
    inputs = make_inputs(boundary, sizes, seed=1)
    built = build_sdfg(boundary.standalone_sdfg, Path(tempfile.mkdtemp(prefix="nf_build_")))
    buf = {k: v.copy() for k, v in inputs.items()}
    built.init(sizes)
    try:
        for _ in range(5):
            built.program(buf, sizes)  # repeated in-place calls on the same state handle
    finally:
        built.close()


def test_set_fast_libnodes_selects_implementation():
    """set_fast_libnodes picks a concrete library-node implementation (OpenBLAS/MKL when a BLAS env is
    available on the extended branch, else the pure fallback) instead of expanding to naive loops -- the
    node keeps its library form with an implementation set, which is what fast_libnodes relies on."""
    N = dace.symbol("N")

    @dace.program
    def mm(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
        C[:] = A @ B

    sdfg = mm.to_sdfg(simplify=True)
    libnodes = [n for s in sdfg.all_states() for n in s.nodes() if isinstance(n, nodes.LibraryNode)]
    assert libnodes, "gemm should lower to a library node before expansion"
    set_fast_libnodes(sdfg)  # must not raise, and must not expand the node away
    still = [n for s in sdfg.all_states() for n in s.nodes() if isinstance(n, nodes.LibraryNode)]
    assert still and all(n.implementation for n in still)  # every node carries a chosen implementation


# --- codegen-implementation axis (legacy | experimental) ---------------------------------------------
def test_config_has_reflects_schema():
    """config_has answers whether the running DaCe schema DEFINES a key -- true for a core key, false for
    a bogus one -- without getattr/hasattr, so the codegen axis can degrade on a build lacking the key."""
    assert config_has("compiler", "build_type")  # a core key every DaCe schema has
    assert not config_has("compiler", "cpu", "definitely_not_a_real_key_zzz")


def test_codegen_impls_available_default_first_and_consistent():
    """The toggleable axis always offers legacy, lists the default first, and default_codegen_impl agrees
    with the first entry. On this (readable-codegen) DaCe build both impls are available, new first."""
    impls = codegen_impls_available()
    assert "legacy" in impls
    assert impls[0] == default_codegen_impl()  # default-first ordering
    assert set(impls) <= set(CODEGEN_IMPLS)
    # A plain build defaults to whatever is available first -- experimental where the key exists.
    assert BuildOptions().codegen_impl == default_codegen_impl()


def test_codegen_config_degrades_gracefully_without_the_key(monkeypatch):
    """On a DaCe build WITHOUT compiler.cpu.implementation (simulated), the default is legacy, a legacy
    scope is a no-op that still runs, and an explicit experimental request RAISES rather than silently
    emitting legacy and mislabelling it."""
    monkeypatch.setattr("nestforge.build.config_has", lambda *path: False)
    assert default_codegen_impl() == "legacy"
    assert codegen_impls_available() == ("legacy", )
    with codegen_config("legacy"):
        pass  # no key to set; the emit_tree_reductions pin is harmless
    with pytest.raises(ValueError):
        with codegen_config("experimental"):
            pass


@pytest.mark.parametrize("impl", codegen_impls_available())
def test_both_codegen_impls_build_and_match_oracle(impl):
    """Every toggleable codegen impl builds the same nest to a working kernel that matches the oracle --
    the axis is genuinely selectable, not just a stamped label."""
    owned_build_matches_oracle("hpc/structured_grids/jacobi_1d/jacobi_1d", opts=BuildOptions(codegen_impl=impl))


def test_vectorized_owned_build_matches_oracle():
    """The DaCe multi-dim tile-op vectorizer plugs into the owned build: a VectorizeConfig on BuildOptions
    is applied before codegen and the vectorized kernel still matches the numpy oracle (AUTO resolves to
    the host ISA, so this is host-agnostic)."""
    from dace.transformation.passes.vectorization.config import VectorizeConfig
    owned_build_matches_oracle("hpc/structured_grids/jacobi_1d/jacobi_1d",
                               size=256,
                               opts=BuildOptions(vectorize=VectorizeConfig(widths=(8, ), target_isa="AUTO")))
