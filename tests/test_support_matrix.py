"""The empirical support matrix: discover compilers + runtimes, then find which (compilers, ONE runtime)
combinations actually build, link, load and run an N-loopnest program compiled by DIFFERENT compilers.

The unit tests here exercise the LOGIC (ranking, the nvhpc drop, the cache) with synthetic cells so they
need no exotic toolchain. One integration test builds the real matrix on whatever compilers are present.
"""
import json

import pytest

from nestforge.perf import flags
from nestforge.perf.support_matrix import (MatrixCell, build_support_matrix, loop_source, machine_config,
                                           render_matrix, resolve_tool_paths, surviving_runtimes)
from nestforge.perf.tsvc_arena import Toolchain, discover_toolchains


def cell(runtime, compilers, correct=True):
    return MatrixCell(runtime=runtime, compilers=tuple(compilers), ok=True, loads=True, parallel=True,
                      correct=correct)


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
    # libomp and libiomp5 tie on this machine (libiomp5 -> libomp via ABI symlink); the portable name wins.
    cells = [cell("libiomp5", ("gcc", "clang")), cell("libomp", ("gcc", "clang"))]
    assert surviving_runtimes(cells)[0] == flags.DEFAULT_OPENMP_RUNTIME.name


def test_a_vendor_only_runtime_never_ranks_as_cross_compiler():
    # nvc can only ever link libnvomp, so libnvomp has no cross-compiler cell -- it must not be offered as
    # a shared runtime even though nvhpc+nvhpc works.
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

    first = machine_config(cache=cache)
    assert cache.exists() and calls["n"] == 1
    assert first["default_openmp_runtime"] == flags.DEFAULT_OPENMP_RUNTIME.name
    assert first["compilers"]["gcc"]["cc"] == "/usr/bin/gcc"  # absolute path persisted

    second = machine_config(cache=cache)  # must LOAD, not re-probe
    assert calls["n"] == 1, "a present cache must be loaded, never re-probed"
    # Compare through JSON: the first result still holds in-memory tuples, the loaded one holds the lists
    # they serialised to. Semantic identity is what matters, and it is what the cache round-trip preserves.
    assert second == json.loads(json.dumps(first))


def test_machine_config_refresh_reprobes(tmp_path, monkeypatch):
    cache = tmp_path / "toolchains.json"
    cache.write_text(json.dumps({"stale": True}))
    import nestforge.perf.support_matrix as sm
    monkeypatch.setattr(sm, "discover_toolchains",
                        lambda _r="auto": [Toolchain("gcc", "/usr/bin/gcc", "/usr/bin/g++", (15, 0), "path")])
    monkeypatch.setattr(sm, "build_support_matrix", lambda tcs: ([cell("libomp", ("gcc", "gcc"))], []))
    cfg = machine_config(cache=cache, refresh=True)
    assert "stale" not in cfg and "compilers" in cfg


def test_a_corrupt_cache_reprobes_instead_of_crashing(tmp_path, monkeypatch):
    cache = tmp_path / "toolchains.json"
    cache.write_text("{ not valid json")
    import nestforge.perf.support_matrix as sm
    monkeypatch.setattr(sm, "discover_toolchains",
                        lambda _r="auto": [Toolchain("gcc", "/usr/bin/gcc", "/usr/bin/g++", (15, 0), "path")])
    monkeypatch.setattr(sm, "build_support_matrix", lambda tcs: ([cell("libomp", ("gcc", "gcc"))], []))
    cfg = machine_config(cache=cache)  # must not raise on the bad file
    assert "compilers" in cfg


# --- integration: the real matrix on whatever is installed -------------------------------------------
@pytest.mark.integration  # compiles a few dozen tiny programs
def test_real_support_matrix_finds_a_cross_compiler_runtime():
    """On any box with >=2 C compilers, at least one runtime must support a real cross-compiler program --
    and it must be libomp or libiomp5 (its ABI-compat twin), never libgomp (clang cannot emit its ABI)."""
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
