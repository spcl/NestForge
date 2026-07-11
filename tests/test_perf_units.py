"""Compile-free unit tests for the perf/arena plumbing: signature parsing, the FP-precision x
cost-model flag composition, winner selection, and the markdown reporters.

These exercise the pure logic on synthetic inputs (hand-written C/Fortran signatures, synthetic result
JSON), so they run fast and without any compiler on PATH -- unlike the end-to-end driver tests in
``test_tsvc_arena.py`` which compile and run real kernels (and skip when a toolchain is absent). The two
together cover both the wiring and the numbers-under-load.
"""
import json

import pytest

from nestforge import tsvc
from nestforge.isolation import run_isolated
from nestforge.perf import crosslang_xl, flags, tsvc_arena


# --- native-baseline signature parsing (tsvc.native_signature) ----------------------------------------
def test_native_signature_strips_qualifiers_and_types():
    cpp = 'extern "C" void s000_d(double* restrict a, const double * b, int64_t LEN_1D, int n1) {'
    sig = tsvc.native_signature(cpp, "s000_d")
    assert sig == [("a", "double", True), ("b", "double", True), ("LEN_1D", "int64_t", False), ("n1", "int", False)]


def test_native_signature_float_and_missing_symbol():
    assert tsvc.native_signature("void k(float* x)", "k") == [("x", "float", True)]
    with pytest.raises(LookupError):
        tsvc.native_signature("void other(double* x)", "k")


def test_native_symbol_fallback_to_first_kernel():
    # The convention symbol is used when present; otherwise the first `void <name>(` is taken.
    assert tsvc_arena.native_symbol("void s000_d(double* a)", "s000_d") == "s000_d"
    assert tsvc_arena.native_symbol("void renamed_kernel(double* a)", "s000_d") == "renamed_kernel"
    with pytest.raises(LookupError):
        tsvc_arena.native_symbol("int not_a_kernel;", "s000_d")


# --- emitted-source signature order (crosslang_xl.signature_order / tsvc_arena.abi_order) --------------
def test_signature_order_c_and_fortran_multiline():
    csrc = "void s000_fp64(double* a, double* out, int64_t N) {"
    assert crosslang_xl.signature_order(csrc, "s000_fp64", "c") == ["a", "out", "N"]
    # a long Fortran arg list wraps with `&` continuations; they must be stripped, not become arg names.
    ftn = "subroutine s1115_fp64(aa, &\n  & bb_slice, cc, &\n  & LEN_2D) bind(c, name='s1115_fp64')\n"
    assert crosslang_xl.signature_order(ftn, "s1115_fp64", "fortran") == ["aa", "bb_slice", "cc", "LEN_2D"]


def test_fortran_unmunge_multiple_and_no_underscore():
    # a leading `_` munges to `x`; a non-underscore name is unchanged; both reverse cleanly.
    order = ["x_a", "xb", "LEN_1D"]
    names = ["__a", "xb", "LEN_1D"]
    assert crosslang_xl.fortran_unmunge(order, names) == ["__a", "xb", "LEN_1D"]


def test_abi_order_pointer_star_stripped():
    assert tsvc_arena.abi_order("void k_fp64(double *a, double* b, int64_t N) {", "k_fp64") == ["a", "b", "N"]


# --- flag composition (flags.*) -----------------------------------------------------------------------
def test_base_flags_native_tuning_per_family():
    assert flags.base_flags("gnu") == ["-O3", "-march=native", "-fPIC", "-shared"]
    assert flags.base_flags("nvidia")[1] == "-tp=native"  # nvc uses -tp=native, not -march=native


def test_fortran_fp_flags_strip_unsupported_and_add_gfortran_guards():
    # gfortran rejects -fexcess-precision=standard and -fno-math-errno; they must be dropped.
    strict_f = flags.fp_flags("gnu", "strict-ieee", "fortran")
    assert "-fexcess-precision=standard" not in strict_f and "-fno-math-errno" not in strict_f
    assert "-fno-frontend-optimize" in strict_f  # gfortran reassociates at -O without this
    assert "-fno-protect-parens" in flags.fp_flags("gnu", "fast-math", "fortran")  # only at the fast rung
    # the C spelling keeps the flags the Fortran frontend rejects.
    assert "-fexcess-precision=standard" in flags.fp_flags("gnu", "strict-ieee", "c")


def test_cost_flags_no_vec_and_cheap_collapse():
    assert flags.cost_flags("gnu", "no-vec") == ["-fno-tree-vectorize"]
    assert flags.cost_flags("llvm", "no-vec") == ["-fno-vectorize", "-fno-slp-vectorize"]
    assert flags.cost_flags("nvidia", "no-vec") == ["-Mnovect"]
    assert flags.cost_flags("gnu", "cheap") == ["-fvect-cost-model=cheap"]
    assert flags.cost_flags("llvm", "cheap") == []  # clang has no cheap knob -> collapses to default
    assert flags.cost_flags("gnu", "default") == []


def test_flag_matrix_atol_covers_every_level():
    # every emitted level has a validation tolerance, and strict is the tightest.
    assert set(flags.FP_ATOL) == set(flags.FP_LEVELS)
    assert flags.FP_ATOL["strict-ieee"] < flags.FP_ATOL["fast-math"]
    for level, model, cflags in flags.flag_matrix("gnu"):
        assert cflags[:1] == ["-O3"] and level in flags.FP_LEVELS and model in flags.COST_MODELS


def test_family_of_maps_labels_to_fp_families():
    assert crosslang_xl.family_of("gcc") == "gnu"
    assert crosslang_xl.family_of("clang") == "llvm"
    assert crosslang_xl.family_of("nvhpc") == "nvidia"
    assert crosslang_xl.family_of("intel") == "intel"
    assert crosslang_xl.family_of("unknown") == "gnu"  # safe default


# --- winner selection ---------------------------------------------------------------------------------
def _cell(ok, t, fp="strict-ieee", cost="default"):
    return {"ok": ok, "time_us": t, "fp_level": fp, "cost_model": cost, "maxdiff": 0.0}


def test_cells_winner_picks_fastest_ok_only():
    cells = [_cell(True, 5.0), _cell(True, 2.0, "fast-math"), _cell(False, 1.0)]  # the 1.0 is not ok
    assert crosslang_xl.cells_winner(cells)["time_us"] == 2.0
    assert crosslang_xl.cells_winner([_cell(False, 1.0)]) is None  # nothing valid -> no winner
    assert crosslang_xl.cells_winner([_cell(True, float("inf"))]) is None  # inf is not a real time


def test_global_winner_across_toolchains_carries_compiler():
    k = {
        "rows": [
            {
                "compiler": "gcc",
                "winner": {
                    "time_us": 9.0,
                    "flags": ["-O3"],
                    "label": "a"
                }
            },
            {
                "compiler": "clang",
                "winner": {
                    "time_us": 4.0,
                    "flags": ["-O3"],
                    "label": "b"
                }
            },
            {
                "compiler": "nvhpc",
                "winner": None
            },
        ]
    }
    win = tsvc_arena.global_winner(k)
    assert win["time_us"] == 4.0 and win["compiler"] == "clang"  # compiler label overrides the cell's own
    assert tsvc_arena.global_winner({"rows": [{"compiler": "gcc", "winner": None}]}) is None


# --- report math (render_tables) ----------------------------------------------------------------------
def _tsvc_row(nat, win):

    def cell(t, label):
        return {
            "ok": True,
            "time_us": t,
            "maxdiff": 0.0,
            "label": label,
            "flags": [],
            "compile_us": 0.0,
            "error": None,
            "compiler": "gcc"
        }

    return {
        "compiler": "gcc",
        "version": [15, 0],
        "source": "path",
        "native": cell(nat, "native"),
        "default": cell(nat, "default"),
        "winner": cell(win, "strict-ieee/default"),
        "cells": []
    }


def test_tsvc_render_tables_geomean_and_skipped(tmp_path):
    sd = tsvc_arena.ensure_seed_dir(tmp_path, 0)
    (sd / "sA.json").write_text(
        json.dumps({
            "key": "sA",
            "regime": "1d",
            "sizes": {
                "LEN_1D": 4
            },
            "rows": [_tsvc_row(10.0, 2.0)]
        }))
    (sd / "sB.json").write_text(
        json.dumps({
            "key": "sB",
            "regime": "1d",
            "sizes": {
                "LEN_1D": 4
            },
            "rows": [_tsvc_row(8.0, 4.0)]
        }))
    (sd / "sC.json").write_text(json.dumps({"key": "sC", "skipped": "no compute nest"}))
    rep = tsvc_arena.render_tables(tmp_path, 0)
    assert "2 kernels measured, 1 skipped" in rep
    assert "5.00x" in rep and "2.00x" in rep  # per-row speedup = native/best
    assert "3.162x" in rep  # geomean of {5, 2} = sqrt(10)
    assert "`sC` — no compute nest" in rep
    assert (sd / "tables.md").exists()


def test_crosslang_render_tables_fp_speedup(tmp_path):

    def cell(fp, t, ok=True):
        return {
            "language": "c",
            "compiler": "gcc",
            "fp_level": fp,
            "cost_model": "default",
            "ok": ok,
            "maxdiff": 0.0 if fp == "strict-ieee" else 1e-9,
            "time_us": t,
            "compile_us": 0.0,
            "error": None
        }

    (tmp_path / "tsvc2_sA.json").write_text(
        json.dumps({
            "key": "sA",
            "corpus": "tsvc2",
            "preset": "XL",
            "cells": [cell("strict-ieee", 10.0), cell("fast-math", 2.5)]
        }))
    (tmp_path / "tsvc2_sB.json").write_text(json.dumps({"key": "sB", "corpus": "tsvc2", "skipped": "no nest"}))
    rep = crosslang_xl.render_tables(tmp_path)
    assert "1 kernels measured, 1 skipped" in rep
    assert "fast-math/default" in rep and "4.00x" in rep  # fp speedup = strict/winner = 10/2.5
    assert "**c**: 1/1" in rep  # one validating (lang, compiler) pair
    assert "`sB` (tsvc2) — no nest" in rep


# --- key_seed determinism -----------------------------------------------------------------------------
def test_key_seed_is_stable_and_distinct():
    assert tsvc.key_seed("s000") == tsvc.key_seed("s000")  # process-independent (not salted hash)
    assert tsvc.key_seed("s000") != tsvc.key_seed("s112")  # different keys -> different offsets
    assert 0 <= tsvc.key_seed("anything") <= 0xFFFF


# --- fault isolation edge cases (run_isolated) --------------------------------------------------------
def test_run_isolated_malformed_result_is_error_not_crash():
    # a non-JSON-able return is caught in the child and comes back as an error sentinel; parent survives.
    res = run_isolated(lambda: {"bad": {1, 2, 3}})  # a set is not JSON-serializable
    assert "error" in res and "TypeError" in res["error"]


def test_run_isolated_passes_through_plain_dict():
    assert run_isolated(lambda: {"ok": True, "n": 7}) == {"ok": True, "n": 7}
