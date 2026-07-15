"""Graceful full-matrix degradation (compiler-free): when a compiler / language target is absent, the
lane-3 enumeration must SHRINK the matrix -- recording an error cell for the missing coordinate -- and
keep emitting real compile jobs for the coordinates that ARE available, never raise. Exercises the pure
enumeration path (:func:`nestforge.perf.tsvc_full.enumerate_cells` / :func:`compiler_for`) with dummy
source Paths, since enumeration never reads the source files.
"""
from pathlib import Path

import pytest

from nestforge.perf import flags, tsvc_full
from nestforge.perf.tsvc_arena import Toolchain, discover_toolchains


def test_compiler_for_routes_by_language_and_none_when_absent():
    tc = Toolchain("gcc", cc="gcc", cxx="g++", version=(13, 0), source="path")
    assert tsvc_full.compiler_for("c", tc, {}) == "gcc"  # C -> the C compiler
    assert tsvc_full.compiler_for("c++", tc, {}) == "g++"  # C++ -> the C++ frontend
    assert tsvc_full.compiler_for("fortran", tc, {"gcc": "gfortran"}) == "gfortran"  # fortran -> the family map
    # absent cxx / absent fortran-map entry -> None (the cell degrades, never crashes)
    noc = Toolchain("clang", cc="clang", cxx=None, version=(18, 0), source="path")
    assert tsvc_full.compiler_for("c++", noc, {}) is None
    assert tsvc_full.compiler_for("fortran", noc, {}) is None


def _synthetic_opt_ctx():
    """A lane-3 context with three language sources but dummy Paths -- enumerate_cells never reads them."""
    return {
        "lang_src": {
            "c": (Path("x.c"), ["a", "N"], [None, None]),
            "c++": (Path("w.cpp"), ["a", "N"], [None, None]),
            "fortran": (Path("x.f90"), ["a", "N"], [None, None]),
        },
        "symbol": "s000_fp64",
    }


def _axes():
    return {
        "opt_mode": "simplify-parallel",
        "parallelism": ["sequential", "auto-par"],
        "cost_models": ["default"],
        "fp_modes": ["default-fp"],
        "gate": True,
    }


def test_enumerate_cells_shrinks_matrix_when_compiler_absent(tmp_path):
    # a clang-only toolchain: no C++ frontend, no fortran in the (empty) family map.
    tc = Toolchain("clang", cc="clang", cxx=None, version=(18, 0), source="path")
    pendings, jobs = tsvc_full.enumerate_cells(_synthetic_opt_ctx(), [tc], {}, _axes(), 4, flags.CXX_STD, tmp_path)

    def by_lang(lang):
        return [p for p in pendings if p.cell.language == lang]

    # (1) c++ has no compiler -> a single error cell with NO compile job, error names the missing compiler.
    cpp = by_lang("c++")
    assert cpp and all(p.compile_key is None for p in cpp)
    assert all(p.cell.error and "no c++ compiler" in p.cell.error for p in cpp)

    # (2) fortran has no compiler in the family map -> same degradation.
    ftn = by_lang("fortran")
    assert ftn and all(p.compile_key is None for p in ftn)
    assert all(p.cell.error and "no fortran compiler" in p.cell.error for p in ftn)

    # (3) c DOES compile -> the matrix shrank, it did not break: real compile jobs exist for C.
    c_cells = by_lang("c")
    c_jobs = [p for p in c_cells if p.compile_key is not None]
    assert c_jobs and jobs  # deduped compile jobs were produced for the C lane

    # (4) clang/llvm + auto-par is unsupported -> recorded error cell (no job) while its sequential
    #     cells still compile. The matrix shrinks per-coordinate, it is never dropped silently.
    c_autopar = [p for p in c_cells if p.cell.parallel == "auto-par"]
    assert c_autopar and all(p.compile_key is None for p in c_autopar)
    assert all(p.cell.error and "unsupported" in p.cell.error for p in c_autopar)
    c_seq = [p for p in c_cells if p.cell.parallel == "sequential" and p.cell.role == "timing"]
    assert c_seq and all(p.compile_key is not None for p in c_seq)


def test_enumerate_cells_with_present_c_compiler_does_not_raise(tmp_path):
    """Sanity: a full gcc toolchain (cc + cxx present) enumerates C and C++ compile jobs and only fortran
    (absent from the empty family map) degrades to an error cell -- no exception at any coordinate."""
    tc = Toolchain("gcc", cc="gcc", cxx="g++", version=(13, 0), source="path")
    pendings, jobs = tsvc_full.enumerate_cells(_synthetic_opt_ctx(), [tc], {}, _axes(), 8, flags.CXX_STD, tmp_path)
    langs_with_jobs = {p.cell.language for p in pendings if p.compile_key is not None}
    assert {"c", "c++"} <= langs_with_jobs  # both real C/C++ compile jobs
    ftn = [p for p in pendings if p.cell.language == "fortran"]
    assert ftn and all(p.compile_key is None and "no fortran compiler" in p.cell.error for p in ftn)


def test_discover_toolchains_unknown_token_warns_and_returns_empty():
    with pytest.warns(UserWarning, match="unknown compiler token"):
        tcs = discover_toolchains("definitely-not-a-compiler")
    assert tcs == []  # a bogus request warns and yields nothing, it never raises
