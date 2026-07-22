"""Compile-free unit tests for the perf/arena plumbing: signature parsing, FP-precision x cost-model flag
composition, winner selection, and the markdown reporters -- pure logic on synthetic inputs, so no compiler
needed (unlike the end-to-end ``test_tsvc_arena.py``, which compiles and skips without a toolchain).
"""
import ctypes
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from nestforge import tsvc
from nestforge.isolation import run_isolated
from nestforge.perf import crosslang_xl, flags, harness, tsvc_arena


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


# --- emitted-source signature order (harness.signature_order, re-exported via crosslang_xl) -----------
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
    assert harness.signature_order("void k_fp64(double *a, double* b, int64_t N) {", "k_fp64") == ["a", "b", "N"]


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


def test_veclib_flags_compose_and_gate_by_compatibility():
    # 'none'/empty -> no flags; incompatible (svml on gcc) or missing compiler -> rejected with a reason.
    # -L/-rpath is machine-dependent, so assert membership, not exact lists.
    assert flags.veclib_flags("g++", "none") == ([], None)
    assert flags.veclib_flags("clang++", None) == ([], None)
    fl, r = flags.veclib_flags("clang++", "sleef")  # x86: emit via libmvec token, link libsleefgnuabi
    assert r is None and "-fveclib=libmvec" in fl and any("-lsleefgnuabi" in a for a in fl)
    flg, rg = flags.veclib_flags("g++", "libmvec")  # glibc: no compile flag, -lmvec pinned at link
    assert rg is None and any("-lmvec" in a for a in flg) and not any("-fveclib" in a for a in flg)
    bad, reason = flags.veclib_flags("g++", "svml")  # gcc emits _ZGV*, never __svml_* -> unusable
    assert bad is None and "incompatible" in reason
    nocc, reason2 = flags.veclib_flags(None, "sleef")
    assert nocc is None and "without a compiler" in reason2
    assert set(flags.VECLIBS) == {"none", "sleef", "libmvec", "svml"}


def test_lane_flags_threads_veclib_and_rejects_incompatible():
    ok, r = flags.lane_flags("llvm", "default-fp", "default", "sequential", "c", 4, compiler="clang++", veclib="sleef")
    assert r is None and "-fveclib=libmvec" in ok and any("-lsleefgnuabi" in a for a in ok)
    bad, reason = flags.lane_flags("gnu", "default-fp", "default", "sequential", "c", 4, compiler="g++", veclib="svml")
    assert bad is None and "incompatible" in reason  # unsupported cell recorded, never silently emitted


def test_source_has_math_gates_the_veclib_axis():
    from nestforge.perf import tsvc_full
    assert tsvc_full.source_has_math("y[i] = sin(x[i]) + 1.0;")
    assert tsvc_full.source_has_math("z = pow(a, b);")
    assert not tsvc_full.source_has_math("c[i] = a[i] + b[i] * 2.0;")  # arithmetic-only nest -> no veclib


def test_veclibs_for_gates_on_math_and_compatibility():
    from nestforge.perf import tsvc_full
    # veclibs_for takes the PRECOMPUTED per-lang has_math flag (source scanned once at ctx-build time).
    assert tsvc_full.veclibs_for(True, ("none", "libmvec"), "gcc") == ("none", "libmvec")  # math + compatible
    assert tsvc_full.veclibs_for(True, ("none", "sleef"), "g++") == ("none", "sleef")  # gcc DOES sleef (gnuabi)
    assert tsvc_full.veclibs_for(True, ("none", "svml"), "g++") == ("none", )  # svml incompatible w/ gcc
    assert tsvc_full.veclibs_for(False, ("none", "libmvec"), "gcc") == ("none", )  # no math -> none only


def test_resolve_veclibs_spec_and_auto():
    from nestforge.perf import tsvc_full
    assert tsvc_full.resolve_veclibs(["none"]) == ("none", )
    assert tsvc_full.resolve_veclibs(["libmvec"]) == ("none", "libmvec")  # 'none' ensured present
    assert tsvc_full.resolve_veclibs(["sleef", "libmvec"])[0] == "none"
    auto = tsvc_full.resolve_veclibs(["auto"])  # none + characterized winner, or none if nothing installed
    assert auto[0] == "none" and 1 <= len(auto) <= 2


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not on PATH")
def test_enumerate_cells_gates_veclib_cells_by_nest_math(tmp_path):
    """The veclib axis fans lane-3 cells off the PRECOMPUTED per-lang ``has_math`` flag: a math nest gets
    both none and libmvec timing cells, a plain-arithmetic nest gets none only. Dummy paths -- no source I/O."""
    from nestforge.perf import tsvc_full
    from nestforge.perf.tsvc_arena import discover_toolchains
    tcs = discover_toolchains("gcc")
    axes = {
        "opt_mode": "simplify-parallel",
        "parallelism": ["sequential"],
        "cost_models": ["default"],
        "fp_modes": ["default-fp"],
        "gate": False,
        "matrix_preset": "lean",
        "veclibs": ("none", "libmvec")
    }
    pend, _ = tsvc_full.enumerate_cells(
        {
            "lang_src": {
                "c": (Path("m_fp64.c"), ["a", "b"], [None, None])
            },
            "has_math": {
                "c": True
            },
            "symbol": "m_fp64",
            "nest_idx": 0
        }, tcs, {}, axes, 4, flags.CXX_STD, tmp_path)
    assert {p.cell.veclib for p in pend if p.cell.role == "timing"} == {"none", "libmvec"}
    pend2, _ = tsvc_full.enumerate_cells(
        {
            "lang_src": {
                "c": (Path("p_fp64.c"), ["a", "b"], [None, None])
            },
            "has_math": {
                "c": False
            },
            "symbol": "p_fp64",
            "nest_idx": 0
        }, tcs, {}, axes, 4, flags.CXX_STD, tmp_path)
    assert {p.cell.veclib for p in pend2 if p.cell.role == "timing"} == {"none"}


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


# --- call_c output snapshotting (harness.call_c) -------------------------------------------------------
class CountingArray(np.ndarray):
    """An ndarray that counts its own .copy() calls, so a test can assert call_c did not snapshot."""
    copies = 0

    def copy(self, *a, **kw):
        self.copies += 1
        return super().copy(*a, **kw)


class FakeBoundary:

    def __init__(self, outputs):
        self.outputs = outputs


class FakeKernel:
    """Stands in for the ctypes entry: records calls, and accepts the argtypes/restype the binder sets."""

    def __init__(self):
        self.calls = 0
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        self.calls += 1


def call_c_on_stub(monkeypatch, reps, **kw):
    """Drive harness.call_c against a stubbed .so -- the ABI marshalling is real, only the compiled entry
    is faked, so no compiler/toolchain is needed."""
    fn = FakeKernel()
    monkeypatch.setattr(harness.ctypes, "CDLL", lambda path: {"k_fp64": fn})
    buf = np.zeros(4, dtype=np.float64).view(CountingArray)
    inputs = {"a": buf}
    argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int64]
    out, us = harness.call_c(Path("stub.so"), "k_fp64", ["a", "LEN_1D"], argtypes, FakeBoundary(["a"]), inputs,
                             {"LEN_1D": 4}, reps, **kw)
    return out, us, buf, fn


def test_call_c_skips_the_output_snapshot_when_not_requested(monkeypatch):
    # The timing path discards the outputs; at XL one output is GBs, so the snapshot must not be built.
    out, _, buf, fn = call_c_on_stub(monkeypatch, reps=3, copy_outputs=False)
    assert out is None
    assert buf.copies == 0
    assert fn.calls == 5  # correctness + warm + reps


def test_call_c_snapshots_outputs_by_default(monkeypatch):
    # The validate path still needs the post-correctness-run values, snapshotted before timing mutates them.
    out, _, buf, _ = call_c_on_stub(monkeypatch, reps=1)
    assert set(out) == {"a"} and buf.copies == 1
