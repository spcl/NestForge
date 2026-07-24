# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The empirical support matrix: discover compilers + runtimes, find which (compilers, ONE runtime) combos
actually build, link, load and run. Unit tests here exercise the LOGIC (ranking, nvhpc drop, cache) with
synthetic cells; one integration test builds the real matrix on whatever compilers are present.
"""
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from nestforge.perf import flags
from nestforge.perf.support_matrix import (MachineCompat, MatrixCell, VeclibCell, build_support_matrix, loop_source,
                                           machine_compat, machine_config, render_matrix, resolve_tool_paths,
                                           surviving_runtimes, try_veclib, vectorized_via)
from nestforge.perf.tsvc_arena import Toolchain, discover_toolchains


def cell(runtime, compilers, correct=True):
    return MatrixCell(runtime=runtime, compilers=tuple(compilers), ok=True, loads=True, parallel=True, correct=correct)


def test_surviving_runtimes_ranks_by_cross_compiler_support():
    cells = [
        cell("libomp", ("gcc", "clang")),
        cell("libomp", ("clang", "gcc")),
        cell("libgomp", ("gcc", "gcc")),  # same-compiler only -- does NOT prove a shared runtime
        cell("libiomp5", ("gcc", "clang")),
    ]
    ranked = surviving_runtimes(cells)
    assert ranked[0] == "libomp", "libomp supports the most cross-compiler combos and must rank first"
    assert "libgomp" not in ranked, "a runtime with only same-compiler cells does not prove sharing"


def test_ties_prefer_the_portable_default_over_a_symlinked_equivalent():
    # libomp and libiomp5 tie (libiomp5 -> libomp via ABI symlink); the portable name wins.
    cells = [cell("libiomp5", ("gcc", "clang")), cell("libomp", ("gcc", "clang"))]
    assert surviving_runtimes(cells)[0] == flags.DEFAULT_OPENMP_RUNTIME.name


def test_a_vendor_only_runtime_never_ranks_as_cross_compiler():
    # nvc can only ever link libnvomp, so it has no cross-compiler cell and must not be offered as shared.
    cells = [cell("libnvomp", ("nvhpc", "nvhpc")), cell("libomp", ("gcc", "clang"))]
    assert surviving_runtimes(cells) == ["libomp"]


def test_render_names_the_best_runtime_and_notes():
    text = render_matrix([cell("libomp", ("gcc", "clang"))], ["dropped nvhpc"])
    assert "libomp" in text and "gcc+clang" in text and "note: dropped nvhpc" in text


def test_loop_source_names_are_unique_so_n_nests_link_together():
    # distinct symbol per index -- N nests must coexist in one program without a duplicate-symbol link error
    assert "nest0(" in loop_source(0) and "nest1(" in loop_source(1)
    assert loop_source(0) != loop_source(1)


# --- the cache: probe once, load thereafter ----------------------------------------------------------
def test_machine_config_writes_then_loads_without_reprobing(tmp_path, monkeypatch):
    cache = tmp_path / "toolchains.json"
    calls = {"n": 0}
    fake = [Toolchain("gcc", cc="/usr/bin/gcc", cxx="/usr/bin/g++", version=(15, 0), source="path")]

    def fake_discover(_req="auto"):
        calls["n"] += 1
        return fake

    import nestforge.perf.support_matrix as sm
    monkeypatch.setattr(sm, "discover_toolchains", fake_discover)
    monkeypatch.setattr(sm, "build_support_matrix", lambda tcs: ([cell("libomp", ("gcc", "gcc"))], []))
    monkeypatch.setattr(sm, "probe_vector_libs", lambda tcs: [])  # keep synthetic: no real veclib compiles

    first = machine_config(cache=cache)
    assert cache.exists() and calls["n"] == 1
    assert first["default_openmp_runtime"] == flags.DEFAULT_OPENMP_RUNTIME.name
    assert first["compilers"]["gcc"]["cc"] == "/usr/bin/gcc"  # absolute path persisted

    second = machine_config(cache=cache)  # must LOAD, not re-probe
    assert calls["n"] == 1, "a present cache must be loaded, never re-probed"
    # Compare through JSON: tuples vs. their serialised lists -- semantic identity is what the round-trip
    # preserves.
    assert second == json.loads(json.dumps(first))


def test_machine_config_refresh_reprobes(tmp_path, monkeypatch):
    cache = tmp_path / "toolchains.json"
    cache.write_text(json.dumps({"stale": True}))
    import nestforge.perf.support_matrix as sm
    monkeypatch.setattr(sm,
                        "discover_toolchains",
                        lambda _r="auto": [Toolchain("gcc", "/usr/bin/gcc", "/usr/bin/g++", (15, 0), "path")])
    monkeypatch.setattr(sm, "build_support_matrix", lambda tcs: ([cell("libomp", ("gcc", "gcc"))], []))
    monkeypatch.setattr(sm, "probe_vector_libs", lambda tcs: [])  # keep synthetic: no real veclib compiles
    cfg = machine_config(cache=cache, refresh=True)
    assert "stale" not in cfg and "compilers" in cfg


def test_a_corrupt_cache_reprobes_instead_of_crashing(tmp_path, monkeypatch):
    cache = tmp_path / "toolchains.json"
    cache.write_text("{ not valid json")
    import nestforge.perf.support_matrix as sm
    monkeypatch.setattr(sm,
                        "discover_toolchains",
                        lambda _r="auto": [Toolchain("gcc", "/usr/bin/gcc", "/usr/bin/g++", (15, 0), "path")])
    monkeypatch.setattr(sm, "build_support_matrix", lambda tcs: ([cell("libomp", ("gcc", "gcc"))], []))
    monkeypatch.setattr(sm, "probe_vector_libs", lambda tcs: [])  # keep synthetic: no real veclib compiles
    cfg = machine_config(cache=cache)  # must not raise on the bad file
    assert "compilers" in cfg


# --- dynamic compatibility: query the discovered matrix, prune per-machine ---------------------------
def compat_from(cells, default_runtime="libomp", surviving=("libomp", "libiomp5")):
    """A MachineCompat over a synthetic config -- no toolchain needed to test the query logic."""
    return MachineCompat({
        "default_openmp_runtime": default_runtime,
        "surviving_runtimes": list(surviving),
        "support_matrix": [vars(c) for c in cells],  # every caller passes MatrixCell, a dataclass
    })


# synthetic machine: libomp works for gcc/clang/intel, libgomp gcc-only, nvc islanded.
THIS_MACHINE = [
    cell("libomp", ("gcc", "clang")),
    cell("libomp", ("gcc", "intel")),
    cell("libomp", ("clang", "intel")),
    cell("libgomp", ("gcc", "gcc")),
    cell("libnvomp", ("nvhpc", "nvhpc")),
]


def test_default_runtime_is_the_machine_survivor_not_a_hardcoded_constant():
    compat = compat_from(THIS_MACHINE, default_runtime="libiomp5")
    assert compat.default_runtime().name == "libiomp5"  # follows the CACHE, not flags.DEFAULT_OPENMP_RUNTIME
    # a bare machine with no discovery falls back to the portable default rather than crashing.
    assert MachineCompat({}).default_runtime().name == flags.DEFAULT_OPENMP_RUNTIME.name


def test_is_supported_reflects_the_empirical_cells_not_the_abi_table():
    compat = compat_from(THIS_MACHINE)
    assert compat.is_supported("gcc", "libomp") and compat.is_supported("intel", "libomp")
    # clang+libgomp is ABI-IMPOSSIBLE and never in the matrix -- the query must say so from evidence.
    assert not compat.is_supported("clang", "libgomp")
    # nvc only ever appears with libnvomp.
    assert compat.is_supported("nvhpc", "libnvomp") and not compat.is_supported("nvhpc", "libomp")


def test_an_inert_cell_is_not_supported_even_though_it_ran():
    # a cell that produced the right answer but never parallelised (Polly inert) must not count as support.
    inert = cell("libomp", ("clang", "clang"))
    inert.parallel = False
    compat = compat_from([inert])
    assert not compat.is_supported("clang", "libomp"), "a serially-run cell is not parallel support"


def test_supported_runtimes_ranks_the_machine_default_first():
    compat = compat_from(THIS_MACHINE, default_runtime="libomp", surviving=("libomp", "libiomp5"))
    assert compat.supported_runtimes("gcc")[0] == "libomp"  # default first, so a fallback stays portable
    assert "libgomp" in compat.supported_runtimes("gcc")  # gcc really can use it here
    assert compat.supported_runtimes("clang") == ["libomp"]  # clang cannot use libgomp/libnvomp


def test_runtime_for_keeps_the_sweep_on_one_runtime_when_possible():
    compat = compat_from(THIS_MACHINE)
    # gcc, clang and intel all support the default -> all get the SAME runtime (the single-runtime contract).
    assert compat.runtime_for("gcc").name == "libomp"
    assert compat.runtime_for("clang").name == "libomp"
    assert compat.runtime_for("intel").name == "libomp"


def test_runtime_for_falls_back_to_a_compilers_own_runtime_off_the_default():
    # nvc cannot link the shared libomp default, so it gets its own libnvomp rather than being dropped.
    compat = compat_from(THIS_MACHINE)
    assert compat.runtime_for("nvhpc").name == "libnvomp"
    # a compiler that parallelises against NOTHING here is dropped from the parallel lanes (None).
    assert compat.runtime_for("unknownfamily") is None


def test_supported_compilers_names_who_can_share_a_runtime():
    compat = compat_from(THIS_MACHINE)
    assert set(compat.supported_compilers("libomp")) == {"gcc", "clang", "intel"}
    assert compat.supported_compilers("libnvomp") == ["nvhpc"]  # islanded
    assert compat.supported_compilers("libgomp") == ["gcc"]


def test_machine_compat_reads_the_cache(tmp_path, monkeypatch):
    cache = tmp_path / "toolchains.json"
    import nestforge.perf.support_matrix as sm
    monkeypatch.setattr(sm,
                        "discover_toolchains",
                        lambda _r="auto": [Toolchain("gcc", "/usr/bin/gcc", "/usr/bin/g++", (15, 0), "path")])
    monkeypatch.setattr(sm, "build_support_matrix", lambda tcs: ([cell("libomp", ("gcc", "gcc"))], []))
    monkeypatch.setattr(sm, "probe_vector_libs", lambda tcs: [])  # keep synthetic: no real veclib compiles
    compat = machine_compat(cache=cache)  # probes once, writes cache
    assert compat.default_runtime().name == flags.DEFAULT_OPENMP_RUNTIME.name
    assert cache.exists()


def test_cached_default_runtime_never_probes(tmp_path, monkeypatch):
    """The hot-path accessor is safe: with no cache it returns the static default without triggering a
    discovery build."""
    import nestforge.perf.support_matrix as sm

    def explode(*a, **k):
        raise AssertionError("cached_default_runtime probed -- it must never build")

    monkeypatch.setattr(sm, "discover_toolchains", explode)
    monkeypatch.setattr(sm, "build_support_matrix", explode)
    assert sm.cached_default_runtime(tmp_path / "absent.json").name == flags.DEFAULT_OPENMP_RUNTIME.name


def test_cached_default_runtime_uses_the_discovered_runtime_when_a_cache_exists(tmp_path):
    import json as _json
    import nestforge.perf.support_matrix as sm
    cache = tmp_path / "toolchains.json"
    cache.write_text(
        _json.dumps({
            "default_openmp_runtime": "libiomp5",
            "surviving_runtimes": ["libiomp5"],
            "support_matrix": []
        }))
    assert sm.cached_default_runtime(cache).name == "libiomp5"  # follows the cache, no probing


def test_cached_default_runtime_survives_a_corrupt_cache(tmp_path):
    import nestforge.perf.support_matrix as sm
    cache = tmp_path / "toolchains.json"
    cache.write_text("{ broken")
    assert sm.cached_default_runtime(cache).name == flags.DEFAULT_OPENMP_RUNTIME.name  # falls back, no raise


# --- integration: the real matrix on whatever is installed -------------------------------------------
@pytest.mark.integration  # compiles a few dozen tiny programs
def test_real_support_matrix_finds_a_cross_compiler_runtime():
    """On any box with >=2 C compilers, >=1 runtime must support a real cross-compiler program -- and it
    must be libomp or libiomp5 (its ABI-compat twin), never libgomp (clang cannot emit its ABI)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tcs = discover_toolchains("auto")
    families = {t.name for t in tcs}
    if len({f for f in families if f in ("gcc", "clang", "intel")}) < 2:
        pytest.skip(f"need two kmpc/gomp-compatible C compilers, found {families}")
    cells, notes = build_support_matrix(tcs)
    cross = surviving_runtimes(cells)
    assert cross, f"no cross-compiler runtime survived; notes={notes}"
    assert cross[0] in ("libomp", "libiomp5"), f"unexpected shared runtime {cross[0]}"
    # every surviving cell must have actually parallelised -- no silently-serial program counts.
    assert all(c.parallel for c in cells if c.correct), "a 'surviving' cell ran serially"


@pytest.mark.integration
def test_resolve_tool_paths_returns_absolute_paths():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tcs = discover_toolchains("gcc")
    if not tcs:
        pytest.skip("no gcc")
    paths = resolve_tool_paths(tcs[0])
    assert paths.compiler.startswith("/")
    for d in paths.openmp.values():
        assert d.startswith("/")  # absolute runtime dirs, as promised


# --- the vector-math-library probe: query logic + the symbol detector (all synthetic, no compiling) ---
def vcell(veclib, compiler, vectorized=True, correct=True, loads=True, ok=True):
    return VeclibCell(veclib=veclib, compiler=compiler, ok=ok, loads=loads, vectorized=vectorized, correct=correct)


def veclib_compat(cells=(), supported=None):
    """A MachineCompat over a synthetic veclib config -- no toolchain, to test the query logic alone."""
    cfg = {"support_matrix": [], "veclib_matrix": [asdict(c) for c in cells]}
    if supported is not None:
        cfg["supported_veclibs"] = supported
    return MachineCompat(cfg)


def test_supported_veclibs_reads_the_config_field():
    compat = veclib_compat(supported={"gnu": ["none", "libmvec"], "llvm": ["none", "sleef", "libmvec"]})
    assert compat.supported_veclibs("gnu") == ["none", "libmvec"]
    assert compat.supported_veclibs("llvm") == ["none", "sleef", "libmvec"]
    assert compat.supported_veclibs("intel-classic") == []  # a family with no correct veclib -> nothing
    assert MachineCompat({}).supported_veclibs("gnu") == []  # bare config -> empty, never a crash


def test_veclib_vectorizes_needs_both_vectorized_and_correct():
    compat = veclib_compat([
        vcell("libmvec", "gnu", vectorized=True, correct=True),  # the real thing
        vcell("none", "gnu", vectorized=False, correct=True),  # scalar baseline: correct but NOT vectorized
        vcell("svml", "gnu", vectorized=True, correct=False),  # called the packed sin but diverged -> unusable
    ])
    assert compat.veclib_vectorizes("gnu", "libmvec")
    assert not compat.veclib_vectorizes("gnu", "none")  # a scalar baseline is never 'vectorized'
    assert not compat.veclib_vectorizes("gnu", "svml")  # ran the packed call but wrong answer
    assert not compat.veclib_vectorizes("gnu", "sleef")  # no such cell at all
    assert not compat.veclib_vectorizes("llvm", "libmvec")  # right veclib, wrong family


def test_vectorized_via_matches_the_per_library_symbol_fingerprint(monkeypatch):
    """The detector reads ``nm -u`` and matches each library's undefined-symbol fingerprint, driven with
    hard-coded nm output so it needs no compiling."""
    import nestforge.perf.support_matrix as sm
    zgv = "                 U sin\n                 U _ZGVdN4v_sin@GLIBC_2.22\n"  # glibc GNU-vector-ABI
    masked = "                 U _ZGVeM8v_sin\n"  # AVX512 masked variant (omp-simd)
    svml = "                 U __svml_sin4\n"
    scalar = "                 U sin\n"

    def feed(out):
        monkeypatch.setattr(sm.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=out, returncode=0))

    feed(zgv)
    # libmvec AND sleef both emit the glibc _ZGV* names on x86, differing only in the linked library.
    assert vectorized_via("libmvec", "x.o") and vectorized_via("sleef", "x.o")
    assert not vectorized_via("svml", "x.o")  # the _ZGV names are not svml's __svml_*
    assert not vectorized_via("none", "x.o")  # none is scalar-by-definition even when vector syms are present

    feed(masked)
    assert vectorized_via("libmvec", "x.o") and vectorized_via("sleef", "x.o")  # masked _ZGV* also detected

    feed(svml)
    assert vectorized_via("svml", "x.o")
    assert not vectorized_via("libmvec", "x.o") and not vectorized_via("sleef", "x.o")

    feed(scalar)
    assert not any(vectorized_via(v, "x.o") for v in ("libmvec", "svml", "sleef"))  # only scalar sin -> nothing fired


# --- integration: the real veclib probe on local gcc (compiles two tiny sin loops) -------------------
@pytest.mark.integration
def test_try_veclib_none_libmvec_and_sleef_on_local_gcc(tmp_path):
    """The scalar ``none`` baseline proves the harness works end to end; libmvec (glibc) vectorizes and
    matches numpy; SLEEF, when ``libsleefgnuabi`` is present, shares the same ``_ZGV*`` emission but binds
    SLEEF's lib -- absent that lib it must fail HONESTLY at link, never silently pass as libmvec."""
    import warnings

    from nestforge import build
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tcs = discover_toolchains("gcc")
    assert tcs, "no gcc discovered -- this box is expected to have gcc"
    gcc = tcs[0]

    baseline = try_veclib(gcc, "none", tmp_path)
    assert baseline.ok and baseline.loads and baseline.correct, f"scalar baseline failed: {baseline.reason}"
    assert not baseline.vectorized, "the 'none' baseline must report scalar, never vectorized"

    mvec = try_veclib(gcc, "libmvec", tmp_path)
    assert mvec.ok and mvec.loads and mvec.correct, f"libmvec cell failed: {mvec.reason}"
    assert mvec.vectorized, "gcc libmvec must emit a packed vector sin here (e.g. _ZGVdN4v_sin)"

    sleef = try_veclib(gcc, "sleef", tmp_path)
    if build.veclib_lib_dir("sleefgnuabi", gcc.cc) is not None:
        assert sleef.ok and sleef.loads and sleef.correct, f"gcc sleef cell failed: {sleef.reason}"
        assert sleef.vectorized, "gcc sleef must emit a packed vector sin (the same _ZGV* glibc ABI)"
    else:  # no libsleefgnuabi installed: a clean link failure, not a silent fallback to libmvec
        assert not sleef.ok and "link" in sleef.reason.lower(), f"expected a link failure, got: {sleef!r}"
