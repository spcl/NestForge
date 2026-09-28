# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""nest-forge owns the DaCe build (docs/build.md): generate DaCe's C++, compile+link it ourselves, call it via
ctypes with manual init/program/exit -- not ``dace.compile``.

Tests build real corpus nests through :mod:`nestforge.build.sdfg` and check the owned-built kernel matches the
numpy oracle: source-tree layout, the init/program/exit call sequence, and per-parameter ctype marshaling.
"""

import ctypes.util
import os
import shutil
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pytest

import dace

import nestforge.build.sdfg as build_mod
import nestforge.build.toolchain as toolchain_mod
from nestforge.build.toolchain import lib_linkable, runtime_library

assert shutil.which("g++") is not None, "g++ not on PATH (setup_apt.sh installs it)"

from nestforge.stages.scopes import parallel_top_level_maps
from nestforge.ir.extract import extract_nest_to_sdfg
from nestforge.corpus.translate import prepare
from nestforge.build.arena import make_inputs, run_oracle
from nestforge.build.sdfg import BuildOptions, build_sdfg, dace_runtime_include
from nestforge.build.toolchain import (
    compiler_family,
    driver_lib_path,
    library_flags,
    openmp_compile_flags,
    openmp_link_flags,
    parse_params,
)
from helpers import corpus_kernel


def first_nest(short):
    sdfg = corpus_kernel(short).to_sdfg(simplify=True)
    parent, node = parallel_top_level_maps(sdfg)[0]
    return extract_nest_to_sdfg(parent, node, name="nest")


def prepared(short, size, seed, out_dir):
    """The first nest of ``short``, its sizes (shape symbols at ``size``, the rest 0), inputs and NumPy oracle."""
    boundary = first_nest(short)
    shape_syms = {
        s for s in boundary.symbols if any(s in str(d.shape) for d in boundary.standalone_sdfg.arrays.values())
    }
    sizes = {s: (size if s in shape_syms else 0) for s in boundary.symbols}
    inputs = make_inputs(boundary, sizes, seed=seed)
    oracle = run_oracle(prepare(boundary, "k", out_dir / "k"), boundary, inputs, sizes)
    return boundary, sizes, inputs, oracle


def owned_build_matches_oracle(tmp_path, short, size=48, opts=None):
    boundary, sizes, inputs, oracle = prepared(short, size, 0, tmp_path)
    built = build_sdfg(boundary.standalone_sdfg, tmp_path / "build", opts)
    buf = {k: v.copy() for k, v in inputs.items()}
    built.run(buf, sizes)  # init -> program -> exit
    for o in oracle:
        np.testing.assert_allclose(buf[o], oracle[o], rtol=1e-9, atol=1e-9, equal_nan=True)
    return built


def test_dace_runtime_include_exists():
    assert (dace_runtime_include() / "dace" / "dace.h").exists()


def test_owned_build_gemm_matches_oracle(tmp_path):
    """gemm: int64_t size symbols + a Scalar (alpha/beta) passed by value through the owned build."""
    owned_build_matches_oracle(tmp_path, "scientific_computing/dense_linear_algebra/gemm/gemm")


def test_owned_build_jacobi_matches_oracle(tmp_path):
    """jacobi_1d: an ``int`` (not int64_t) size symbol -- guards the per-parameter ctype marshaling."""
    owned_build_matches_oracle(tmp_path, "scientific_computing/structured_grids/jacobi_1d/jacobi_1d")


def without_search_paths(flags):
    """``flags`` without ``-L`` and ``-Wl,-rpath,`` entries: which runtime, not where it was found."""
    return [f for f in flags if not f.startswith("-L") and not f.startswith("-Wl,-rpath")]


def test_openmp_flags_select_libomp_for_every_compiler_family():
    """An LLVM compiler selects libomp by name; gcc compiles its GOMP calls and links libomp explicitly, never
    the libgomp a bare -fopenmp would pull in, so mixed-compiler builds share one runtime."""
    assert compiler_family("gfortran") == "gnu" and compiler_family("flang") == "llvm"
    assert compiler_family("icx") == "llvm"
    assert openmp_compile_flags("flang") == openmp_compile_flags("clang++") == ["-fopenmp=libomp"]
    assert openmp_compile_flags("g++") == ["-fopenmp"]
    linked = without_search_paths(openmp_link_flags("g++"))
    assert len(linked) == 1 and linked[0] in ("-lomp", "-l:libomp.so.5"), linked
    assert without_search_paths(openmp_link_flags("clang++")) == ["-fopenmp=libomp"]


def test_gcc_compiled_kernel_links_against_libomp(tmp_path):
    """A g++-compiled kernel (GOMP_* calls under -fopenmp) links + runs against libomp via its GOMP-compat
    ABI -- proof a GCC node library can share the same libomp a clang/flang node library uses."""
    assert ctypes.util.find_library("omp") is not None, "libomp not installed (setup_apt.sh: libomp-dev)"
    owned_build_matches_oracle(
        tmp_path,
        "scientific_computing/dense_linear_algebra/gemm/gemm",
        size=32,
        opts=BuildOptions(compiler="g++"),
    )


def parallel_axpy_sdfg(name="paxpy"):
    """A minimal SDFG with one genuinely parallel map (``CPU_Multicore`` -> ``#pragma omp parallel for``):
    ``Z[i] = X[i] + Y[i]``. Hermetic, so the OpenMP link matrix tests a guaranteed-parallel loop."""
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


def test_link_flags_pin_a_runtime_that_is_off_the_default_linker_path(tmp_path, monkeypatch, cold_lookup_caches):
    """The linker and the loader search different places, so "installed" does not imply "-l<soname>
    resolves": Ubuntu's libomp-dev moves the library off the default linker path across releases."""
    (tmp_path / "libfakeomp.so").write_bytes(b"")  # a linkable lib, deliberately off the default path
    # LD_LIBRARY_PATH (not LIBRARY_PATH): the loader searches it, the linker does not
    monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path))
    assert driver_lib_path("fakeomp", "g++") is None, "premise: the linker cannot find it unaided"
    assert library_flags("fakeomp", "g++") == [f"-L{tmp_path}", f"-Wl,-rpath,{tmp_path}", "-lfakeomp"]
    assert lib_linkable("fakeomp", "g++")


def test_link_flags_add_no_search_path_when_the_linker_already_finds_the_runtime(tmp_path, cold_lookup_caches):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (tmp_path / "libfakeomp.so").write_bytes(b"")
    write_tool(bin_dir, "fake-g++", f"{tmp_path}/libfakeomp.so")  # answers the way g++ does when it finds it
    assert library_flags("fakeomp", str(bin_dir / "fake-g++")) == ["-lfakeomp"]


def test_driver_lib_path_normalises_the_answer_without_following_the_symlink(tmp_path):
    """``libomp.so`` IS a symlink (-> ``libomp.so.5``) and the two can live in different directories, so the
    answer needs normalising without following it: ``resolve()`` would follow the symlink to a directory
    with no ``libomp.so``, so lexical normalisation is used instead.
    """
    link_dir, target_dir = tmp_path / "linkdir", tmp_path / "targetdir"
    link_dir.mkdir()
    target_dir.mkdir()
    (target_dir / "libsplit.so.5").write_bytes(b"")
    (link_dir / "libsplit.so").symlink_to(target_dir / "libsplit.so.5")  # the real distro layout

    fake_cc = tmp_path / "fake-cc"  # a driver that answers the way gcc does: full of ".." segments
    fake_cc.write_text(f'#!/bin/sh\necho "{link_dir}/../linkdir/libsplit.so"\n')
    fake_cc.chmod(0o755)

    got = driver_lib_path("split", str(fake_cc))
    assert got == link_dir / "libsplit.so", f"expected the symlink itself, got {got}"
    assert got.parent == link_dir, "the -L must be the symlink's own dir, never its target's"


def test_parallel_map_emits_omp_pragma(tmp_path):
    """The sanity nest is actually parallel: DaCe lowers ``CPU_Multicore`` to an OpenMP pragma in the
    generated C++ (so the cross-compiler tests below really exercise the runtime link)."""
    from nestforge.build.sdfg import generate_program_folder

    frame = generate_program_folder(parallel_axpy_sdfg(), tmp_path / "nf_omp_src")
    assert "#pragma omp parallel for" in frame.read_text()


# Each compiler builds the same parallel nest, linking the one mandated runtime (libomp) -- the
# mixed-compiler / single-runtime sanity matrix. icpx is a vendor compiler, only ever present in a
# vendor-configured environment (setup_apt.sh --oneapi).
@pytest.mark.parametrize(
    "compiler",
    [
        "g++",
        "clang++",
        pytest.param("icpx", marks=pytest.mark.vendor),  # vendor compiler: absent on the CI runner
    ],
)
def test_parallel_loop_links_openmp_across_compilers(tmp_path, compiler):
    assert shutil.which(compiler) is not None, f"{compiler} not on PATH"
    assert lib_linkable("omp", compiler), f"libomp is not linkable by {compiler} (setup_apt.sh installs libomp)"
    n = 256
    x, y = np.random.default_rng(0).random(n), np.random.default_rng(1).random(n)
    buf = {"X": x.copy(), "Y": y.copy(), "Z": np.zeros(n)}
    built = build_sdfg(
        parallel_axpy_sdfg(),
        tmp_path / "nf_par",
        BuildOptions(compiler=compiler, flags=["-O2", "-fPIC", "-shared", "-std=c++20"]),
    )
    built.run(buf, {"N": n})
    np.testing.assert_allclose(buf["Z"], x + y, rtol=1e-12, atol=1e-12)


def test_build_tracks_optimization_and_compile_time(tmp_path):
    """Every owned build records both the codegen (optimization) time and the compile (toolchain) time."""
    built = build_sdfg(parallel_axpy_sdfg(), tmp_path / "nf_time")
    assert built.codegen_seconds > 0.0
    assert built.compile_seconds > 0.0


def test_unload_after_close_is_a_noop(tmp_path):
    """The documented lifecycle -- ``run()`` (init -> program -> close) then ``unload()`` once the sweep is
    done with a kernel -- must not raise: by the time ``unload()`` runs, ``close()`` has already dropped the
    handle, so there is nothing left for ``unload`` to reconcile."""
    built = build_sdfg(parallel_axpy_sdfg(), tmp_path / "nf_unload")
    n = 8
    buf = {"X": np.zeros(n), "Y": np.zeros(n), "Z": np.zeros(n)}
    built.run(buf, {"N": n})
    assert built.handle is None
    built.unload()
    assert built.lib is None


def test_close_after_unload_raises_when_a_handle_is_still_open(tmp_path):
    """Misuse case: unloading the library while a handle from ``init()`` is still open leaves nothing able
    to run ``__dace_exit`` on that handle. This must fail loudly rather than silently leak the handle or
    crash on a null CDLL lookup."""
    built = build_sdfg(parallel_axpy_sdfg(), tmp_path / "nf_unload2")
    built.init({"N": 8})
    built.unload()
    with pytest.raises(RuntimeError, match="unload"):
        built.close()


def test_external_linking_build_is_correct(tmp_path):
    """A nest built as a separate static ``.a`` (link_external) and linked into the ``.so`` runs identically
    to the monolithic build -- external linking is correct, not merely timeable."""
    built = owned_build_matches_oracle(
        tmp_path, "scientific_computing/dense_linear_algebra/gemm/gemm", opts=BuildOptions(link_external=True)
    )
    assert built.compile_seconds > 0.0
    assert (built.so_path.parent / f"lib{built.name}_nest.a").exists()  # the static node lib was produced


def test_parse_params_strips_the_const_qualifier_only_as_a_word():
    """``const`` is a qualifier, not a substring: params literally named ``constant``/``const_term`` must
    keep their name, or the ctypes bind looks them up under a mangled key."""
    params = parse_params("k_state_t *__state, const double * __restrict__ constant, const int const_term")
    assert [p.name for p in params] == ["constant", "const_term"]
    assert [p.ctype for p in params] == [ctypes.POINTER(ctypes.c_double), ctypes.c_int]


def test_parse_params_refuses_an_unmapped_by_value_scalar_type():
    """An unmapped by-value type must fail loud: defaulting to int64 puts a float in a GP register (SysV
    ABI), so the callee reads garbage with no ctypes error."""
    with pytest.raises(ValueError, match="uint64_t"):
        parse_params("k_state_t *__state, uint64_t n")


def test_parse_params_refuses_an_unmapped_pointer_base_type():
    """An unmapped pointer base type must fail loud too, the same as the scalar branch: silently defaulting
    to ``double*`` marshals a differently-sized element through the ABI with no ctypes error."""
    with pytest.raises(ValueError, match="uint64_t"):
        parse_params("k_state_t *__state, uint64_t *n")


def test_int_and_complex_pointers_bind_by_address():
    """xsbench's entry takes ``int*`` and scattering_self_energies' ``dace::complex128*``. A pointer carries only
    the buffer address, so each binds as a pointer to its element (or complex component) type."""
    params = parse_params("const int * __restrict__ index_grid, dace::complex128 * __restrict__ D, int64_t N")
    assert [(p.name, p.ctype) for p in params] == [
        ("index_grid", ctypes.POINTER(ctypes.c_int)),
        ("D", ctypes.POINTER(ctypes.c_double)),
        ("N", ctypes.c_int64),
    ]


def test_owned_build_reusable_handle_program(tmp_path):
    """After one init, __program can be called repeatedly in place (the timing path) on one handle, and
    every call still computes the right answer (this nest's output does not read its own prior value, so
    repeating the call is idempotent and one oracle run covers every rep)."""
    boundary, sizes, inputs, oracle = prepared("scientific_computing/dense_linear_algebra/gemm/gemm", 32, 1, tmp_path)
    built = build_sdfg(boundary.standalone_sdfg, tmp_path / "build")
    buf = {k: v.copy() for k, v in inputs.items()}
    built.init(sizes)
    try:
        for _ in range(5):
            built.program(buf, sizes)  # repeated in-place calls on the same state handle
    finally:
        built.close()
    for o in oracle:
        np.testing.assert_allclose(buf[o], oracle[o], rtol=1e-9, atol=1e-9, equal_nan=True)


def test_toolchain_is_importable_without_dace():
    """Asking which OpenMP runtime a compiler links must not load the DaCe code generator."""
    # a fresh interpreter: this one imported dace long ago
    path = Path(__file__).resolve().parents[1] / "nestforge" / "build" / "toolchain.py"
    probe = textwrap.dedent(f"""
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("nf_toolchain_isolated", {str(path)!r})
        module = importlib.util.module_from_spec(spec)
        sys.modules["nf_toolchain_isolated"] = module
        spec.loader.exec_module(module)
        assert "dace" not in sys.modules, "importing nestforge.build.toolchain pulled in dace"
        assert module.compiler_family("icx") == "llvm"          # a real probe, not just an import
        assert module.C_PTR["dace::complex128"] is module.C_SCALAR["double"]
    """)
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-800:]


# compiler diagnostics on the DaCe-generated C++
def test_resolved_flags_guarantee_the_standard_and_warnings_without_forcing_them():
    """Both are filled IN, not appended blindly: nearly every caller passes its own ``flags`` for one axis
    (an -O level, an FP mode) and would otherwise lose them. ``-Werror`` is deliberately absent -- this
    compiles generated C++ we do not own, so a warning is a codegen signal, not a failed measurement."""
    assert BuildOptions().resolved_flags()[-1] == "-Wall"
    assert BuildOptions(flags=["-O2"]).resolved_flags() == ["-O2", "-std=c++20", "-Wall"]
    # an explicit choice wins in both directions: -w silences, -Wextra is not duplicated into -Wall
    assert "-Wall" not in BuildOptions(flags=["-O2", "-w"]).resolved_flags()
    assert BuildOptions(flags=["-O2", "-Wall", "-Wextra"]).resolved_flags().count("-Wall") == 1
    assert not any(f.startswith("-Werror") for f in BuildOptions().resolved_flags())


def test_a_succeeding_compile_surfaces_its_warnings(tmp_path):
    """-Wall is inert unless the diagnostics are read: toolchain.run captured stderr and dropped it on
    success, so every warning from the generated C++ went to /dev/null."""
    src = tmp_path / "warn.cpp"
    src.write_text("int main() { int unused_variable = 1; return 0; }\n")
    with pytest.warns(UserWarning, match="unused_variable"):
        toolchain_mod.run(["g++", "-Wall", "-c", str(src), "-o", str(tmp_path / "warn.o")])


def test_a_clean_compile_warns_about_nothing(tmp_path):
    """The sweep compiles thousands of cells; a run that warned unconditionally would bury the real ones."""
    src = tmp_path / "clean.cpp"
    src.write_text("int main() { return 0; }\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning at all fails this test
        toolchain_mod.run(["g++", "-Wall", "-c", str(src), "-o", str(tmp_path / "clean.o")])


def test_a_cpu_build_links_an_openmp_runtime_by_default(tmp_path, monkeypatch):
    """DaCe emits ``#pragma omp parallel for`` for every multicore map; a build without an OpenMP flag drops the
    pragma and runs serially, silently, so a build that names no runtime gets the resolved one."""
    seen = []
    monkeypatch.setattr(build_mod, "run", lambda cmd, **k: seen.append(list(cmd)))
    monkeypatch.setattr(build_mod, "lib_linkable", lambda soname, compiler: True)
    src = tmp_path / "x.cpp"
    src.write_text("int main() { return 0; }\n")

    build_mod.compile(src, tmp_path, "x", BuildOptions())

    issued = " ".join(t for cmd in seen for t in cmd)
    assert "-fopenmp" in issued and "omp" in issued.replace("-fopenmp", ""), issued


def test_the_runtime_is_named_never_a_bare_fopenmp():
    """A bare -fopenmp lets each family link its own default (gcc->libgomp, clang->libomp), so a sweep spanning
    compilers ends up with two thread pools in one process."""
    assert openmp_link_flags("g++") != ["-fopenmp"] and "-fopenmp" not in openmp_link_flags("g++")
    assert openmp_link_flags("clang++")[0] == "-fopenmp=libomp"


def test_compiler_warnings_are_reported_but_bounded():
    """-Wall fires on nearly every generated cell, each with its own paths, so warnings dedup by kind and stop
    past a budget of kinds."""
    toolchain_mod.WARNED.clear()
    try:
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            for cell in range(50):  # the same kind, 50 different files
                toolchain_mod.warn_once("g++", f"/build/cell{cell}/x.cpp:{cell}:9: warning: unused [-Wunused-variable]")
        assert len(seen) == 1, f"one warning kind reported {len(seen)} times"

        # a genuinely new kind is still reported, up to the budget
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            for kind in range(10):
                toolchain_mod.warn_once("g++", f"x.cpp:1:1: warning: kind {kind} [-Wkind{kind}]")
        assert len(seen) == toolchain_mod.WARN_BUDGET - 1, len(seen)
    finally:
        toolchain_mod.WARNED.clear()


def write_tool(bin_dir: Path, name: str, answer: str) -> None:
    """A fake driver on PATH that answers every query with ``answer``."""
    tool = bin_dir / name
    tool.write_text(f'#!/bin/sh\necho "{answer}"\n')
    tool.chmod(0o755)


@pytest.fixture
def cold_lookup_caches():
    """Runtime lookups are cached per name; a fake PATH must neither read nor leave a cached answer."""
    caches = (toolchain_mod.driver_lib_path, toolchain_mod.runtime_library)
    for cache in caches:
        cache.cache_clear()
    yield
    for cache in caches:
        cache.cache_clear()


def isolate_lookup(tmp_path: Path, monkeypatch) -> Path:
    """A PATH holding only the fake drivers a test writes, and no library path in the environment."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("LIBRARY_PATH", raising=False)
    # a test may have looked up the real runtime before isolating, so drop what that cached
    for cache in (toolchain_mod.driver_lib_path, toolchain_mod.runtime_library):
        cache.cache_clear()
    return bin_dir


@pytest.mark.parametrize(
    "gxx_answers, clang_answers, found_by",
    [(True, True, "gxx"), (False, True, "clang"), (False, False, "llvm-config")],
    ids=["the_compiler_first", "then_an_llvm_driver", "then_llvm_config_with_only_a_versioned_so"],
)
def test_libomp_lookup_asks_the_compiler_then_an_llvm_driver_then_llvm_config(
    tmp_path, monkeypatch, cold_lookup_caches, gxx_answers, clang_answers, found_by
):
    """Ubuntu installs libomp under the LLVM prefix, where g++ does not search: the lookup must reach it through
    clang or llvm-config, in that order, and accept libomp.so.5 when the unversioned symlink is missing."""
    bin_dir = isolate_lookup(tmp_path, monkeypatch)
    dirs = {name: tmp_path / name for name in ("gxx", "clang", "llvm-config")}
    for directory in dirs.values():
        directory.mkdir()
    (dirs["gxx"] / "libomp.so").write_bytes(b"")
    (dirs["clang"] / "libomp.so").write_bytes(b"")
    (dirs["llvm-config"] / "libomp.so.5").write_bytes(b"")
    write_tool(bin_dir, "g++", f"{dirs['gxx']}/libomp.so" if gxx_answers else "libomp.so")
    write_tool(bin_dir, "clang", f"{dirs['clang']}/libomp.so" if clang_answers else "libomp.so")
    write_tool(bin_dir, "llvm-config", str(dirs["llvm-config"]))
    expected = {"gxx": dirs["gxx"] / "libomp.so", "clang": dirs["clang"] / "libomp.so"}.get(
        found_by, dirs["llvm-config"] / "libomp.so.5"
    )

    assert runtime_library("omp", "g++") == expected


OMP_KERNEL_CXX = """extern "C" double kern(const double *a, int n) {
  double s = 0.0;
  #pragma omp parallel for reduction(+:s)
  for (int i = 0; i < n; i++) s += a[i];
  return s;
}
"""

LOAD_AND_CALL = """import ctypes, sys
lib = ctypes.CDLL(sys.argv[1])
lib.kern.restype = ctypes.c_double
n = 1000
values = (ctypes.c_double * n)(*range(n))
print(lib.kern(values, n))
"""


@pytest.mark.e2e
def test_a_libomp_only_llvm_config_finds_links_and_loads_from_its_own_directory(
    tmp_path, monkeypatch, cold_lookup_caches
):
    """The CI layout, rebuilt in a temp dir: libomp only as libomp.so.5 under an LLVM prefix g++ does not search.
    The discovered directory serves the link (-L, -l:libomp.so.5) and the load (rpath), with nothing on the
    loader path."""
    real_gxx, real_libomp = shutil.which("g++"), runtime_library("omp", "g++")
    assert real_gxx and real_libomp, "g++ and libomp are installed (setup_apt.sh)"
    real_env = dict(os.environ)  # the build and the load see the real PATH; only the lookup sees the fake one
    llvm_lib = tmp_path / "llvm" / "lib"
    llvm_lib.mkdir(parents=True)
    (llvm_lib / "libomp.so.5").symlink_to(os.path.realpath(real_libomp))
    bin_dir = isolate_lookup(tmp_path, monkeypatch)
    write_tool(bin_dir, "g++", "libomp.so")
    write_tool(bin_dir, "llvm-config", str(llvm_lib))

    flags = openmp_link_flags("g++")

    assert flags == [f"-L{llvm_lib}", f"-Wl,-rpath,{llvm_lib}", "-l:libomp.so.5"]
    source, obj, shared = tmp_path / "kern.cpp", tmp_path / "kern.o", tmp_path / "libkern.so"
    source.write_text(OMP_KERNEL_CXX)
    subprocess.run([real_gxx, "-O2", "-fPIC", "-fopenmp", "-c", str(source), "-o", str(obj)], check=True, env=real_env)
    subprocess.run(
        [real_gxx, "-shared", "-Wl,--as-needed", str(obj), *flags, "-o", str(shared)], check=True, env=real_env
    )
    dynamic = subprocess.run(
        ["readelf", "-d", str(shared)], capture_output=True, text=True, check=True, env=real_env
    ).stdout
    assert "[libomp.so.5]" in dynamic and str(llvm_lib) in dynamic
    loaded = subprocess.run(
        [sys.executable, "-c", LOAD_AND_CALL, str(shared)], capture_output=True, text=True, env=real_env
    )
    assert loaded.returncode == 0, loaded.stderr[-1500:]
    assert float(loaded.stdout) == 999 * 1000 / 2
