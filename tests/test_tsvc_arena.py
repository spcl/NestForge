# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""TSVC compiler-arena driver + adapter, plus the emitter user-function rewrites it depends on."""
import ctypes
import ctypes.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import types

import pytest

import dace

from nestforge import tsvc
from nestforge.emit_numpy import EMITTED_BUILTINS, normalize_casts
from nestforge.extract import Boundary, extract_nest_to_sdfg, nest_defined_symbols, trip_count_symbols
from nestforge.isolation import run_isolated
from nestforge.multinest import extract_all_nests
from nestforge.perf import crosslang_xl, flags, harness, staticlib_overhead, tsvc_arena
from nestforge.strategies import get_strategy


# --- emitter: sympy user-function + qualified-math rewrites (the arena's translation depends on these) ---
def test_normalize_emits_floor_as_the_operator_and_ceil_as_a_call():
    # `int_floor` exists because sympy mis-simplifies a floor division in a DACE-side expression, not
    # because python needs a helper -- `//` is already floored for both signs. The EMITTED text is only
    # exec'd and handed to the translator, never re-parsed by dace, so the operator is safe there and is
    # strictly better: a translator reads `ast.FloorDiv` and lowers it with its own correct helper,
    # where a bare call is an unknown name. `int_ceil` has no operator and stays a call.
    assert normalize_casts("a[int_floor(LEN_1D, 2)]") == "a[((LEN_1D) // (2))]"
    assert normalize_casts("int_ceil(N, 4)") == "int_ceil(N, 4)"
    assert normalize_casts("int_floor(int_ceil(N, 2), 3)") == "((int_ceil(N, 2)) // (3))"


def test_emitted_builtins_match_python_semantics():
    # the exec-namespace definitions must agree with `//` on both signs, or the oracle drifts from C.
    for a, b in ((7, 2), (-7, 2), (7, -2), (-7, -2), (8, 4), (-8, 4)):
        assert EMITTED_BUILTINS["int_floor"](a, b) == a // b
        assert EMITTED_BUILTINS["int_ceil"](a, b) == -((-a) // b)


def test_normalize_rewrites_qualified_math_to_numpy():
    assert normalize_casts("dace.math.sin(x)") == "np.sin(x)"
    assert normalize_casts("math.cos(y)") == "np.cos(y)"
    assert normalize_casts("dace.math.asin(z)") == "np.arcsin(z)"  # name-mapped, not verbatim
    # a bare (unqualified) intrinsic still lowers, and a dtype cast still maps.
    assert normalize_casts("sqrt(w)") == "np.sqrt(w)"
    assert normalize_casts("dace.float64(v)") == "np.float64(v)"


def test_normalize_rewrites_bare_dtype_cast_in_subscript():
    # symbolic.symstr renders a DaCe typecast inside an array subscript as a BARE call (no dace./np.
    # prefix), so the emitted index would raise ``NameError: name 'int64' is not defined`` (xsbench).
    assert normalize_casts("g[(int64(mats_index)), k]") == "g[(np.int64(mats_index)), k]"
    assert normalize_casts("uint32(i) + int64(j)") == "np.uint32(i) + np.int64(j)"
    # every dtype in the map -- not just int64 -- is a bare cast the subscript path can render.
    assert normalize_casts("bool(m)") == "np.bool_(m)"  # bool -> np.bool_ (np.bool removed in NumPy 2)
    assert normalize_casts("complex128(z) + complex64(w)") == "np.complex128(z) + np.complex64(w)"
    assert normalize_casts("uint8(b)") == "np.uint8(b)"
    # an already-qualified cast is not double-prefixed; the word-boundary lookbehind leaves a dtype
    # name embedded in a longer identifier or attribute alone (no spurious rewrite of ``uint8`` in
    # ``__ruint8`` or the ``int8`` inside ``uint8`` / ``point8``).
    assert normalize_casts("np.int64(x)") == "np.int64(x)"
    assert normalize_casts("dace.int64(x)") == "np.int64(x)"
    assert normalize_casts("nuclide_grid_1_0_index") == "nuclide_grid_1_0_index"
    assert normalize_casts("point8(q)") == "point8(q)"  # 'int8' inside 'point8' must NOT match


# --- adapter ------------------------------------------------------------------------------------------
def test_iter_and_filter_kernels():
    only = tsvc.iter_tsvc_kernels(only=["s000", "s112"])
    assert {k.key for k in only} == {"s000", "s112"}
    s000 = only[0] if only[0].key == "s000" else only[1]
    assert s000.native_cpp is not None and s000.native_cpp.exists()  # foundation ships the native baseline
    assert s000.native_symbol == "s000_d"


def test_manifest_resolves_from_per_kernel_subfolder():
    # OptArena keeps each foundation kernel in its own ``foundation/<stem>/`` subfolder, so a flat
    # directory lookup silently returns None for every kernel but the first -- the manifest fill then
    # degenerates to all-zeros (see tests/test_index_fills.py). Resolve via the KERNELS registry instead.
    vag = tsvc.iter_tsvc_kernels(only=["vag"], corpus="tsvc2")[0]  # tsvc2: tsvc_2_<key> stem
    assert vag.yaml_path is not None and vag.yaml_path.exists()
    assert vag.bench_name == "tsvc_2_vag"
    assert vag.native_cpp is not None and vag.native_cpp.name == "tsvc_2_vag_native.cpp"
    reroll = tsvc.iter_tsvc_kernels(only=["reroll_gather"], corpus="tsvc2_5")[0]  # tsvc2_5: bare <key> stem
    assert reroll.yaml_path is not None and reroll.bench_name == "reroll_gather"


def test_sample_sizes_indices_zero_shapes_sized():
    k = tsvc.iter_tsvc_kernels(only=["s000"])[0]
    sdfg = tsvc.build_sdfg(k, "simplify-parallel")
    parent, node = get_strategy("skip-taskloops")(sdfg)[0]
    boundary = extract_nest_to_sdfg(parent, node, name="s000")
    sizes = tsvc.sample_sizes(k, boundary, seed=0, random_sizes=False)
    assert sizes["LEN_1D"] == tsvc._SYM_FIXED["LEN_1D"]  # a shape symbol is sized
    assert all(sizes[s] == 0 for s in boundary.symbols if s not in tsvc._SHAPE_SYMS)  # a leaked index is 0
    # a fixed seed is reproducible; random draws stay within the preset range and are seed-stable.
    rnd = tsvc.sample_sizes(k, boundary, seed=7, random_sizes=True)
    assert rnd == tsvc.sample_sizes(k, boundary, seed=7, random_sizes=True)
    assert tsvc._SYM_RANGE["LEN_1D"][0] <= rnd["LEN_1D"]


# --- flag matrix + discovery --------------------------------------------------------------------------
def test_flag_matrix_dedups_and_covers_levels():
    matrix = flags.flag_matrix("gnu")  # tsvc_arena + crosslang both sweep this shared matrix
    assert {lvl for lvl, _, _ in matrix} == set(flags.FP_LEVELS)  # every FP-precision level present
    assert len({tuple(f) for _, _, f in matrix}) == len(matrix)  # no duplicate flag sets
    assert len(matrix) == 12  # 4 levels x 3 cost-models, all distinct on gnu
    # clang's "cheap" collapses to "default", so its matrix is smaller than the raw 4x3.
    assert len(flags.flag_matrix("llvm")) == 8
    # nvidia's assume-finite collapses to contract-fma (no per-assumption flag), so 3 levels x {default,no-vec}.
    assert len(flags.flag_matrix("nvidia")) == 6


def test_discover_toolchains_present():
    tcs = tsvc_arena.discover_toolchains("gcc")
    assert tcs and tcs[0].name == "gcc" and tcs[0].cc  # gcc is on PATH in CI


def test_intel_is_its_own_fp_family():
    icx = tsvc_arena.Toolchain("intel", "icx", "icpx", (2026, 1), "path")
    assert icx.family == "llvm"  # OpenMP-runtime family: icx is clang-based
    assert icx.fp_family == "intel"  # but a distinct FP family (defaults to -fp-model=fast, needs explicit -fp-model)
    assert {lvl for lvl, _, _ in flags.flag_matrix("intel")} == set(flags.FP_LEVELS)
    assert flags.fp_flags("intel", "strict-ieee") == ["-fp-model=strict"]  # not clang's -ffp-contract=off


# --- multi-corpus adapter + preset sizing -------------------------------------------------------------
def test_tsvc2_5_corpus_loads():
    ks = tsvc.iter_tsvc_kernels(corpus="tsvc2_5")
    assert len(ks) > 50 and all(k.corpus == "tsvc2_5" for k in ks)
    only = tsvc.iter_tsvc_kernels(only=["ext_gather_load"], corpus="tsvc2_5")
    assert len(only) == 1 and only[0].key == "ext_gather_load"


def test_preset_sizes_scale():
    k = tsvc.iter_tsvc_kernels(only=["s000"])[0]
    sdfg = tsvc.build_sdfg(k, "simplify-parallel")
    parent, node = get_strategy("skip-taskloops")(sdfg)[0]
    b = extract_nest_to_sdfg(parent, node, name="s000")
    assert tsvc.sample_sizes(k, b, preset="XL")["LEN_1D"] == tsvc._PRESET["LEN_1D"]["XL"]
    assert tsvc.sample_sizes(k, b, preset="S")["LEN_1D"] == 512


def test_corpus_symbol_values_read_both_corpus_spellings():
    # The corpora bind their scalars differently -- tsvc2 a lone S_VALUE, tsvc2_5 a SIZES dict -- so the
    # value is inferred from whichever the corpus script declares, never assumed to be one of the two.
    tsvc2 = vars(tsvc.corpus_module("tsvc2"))
    tsvc2_5 = vars(tsvc.corpus_module("tsvc2_5"))
    assert tsvc.corpus_symbol_values("tsvc2")["S"] == tsvc2["S_VALUE"]
    assert tsvc.corpus_symbol_values("tsvc2_5")["S"] == tsvc2_5["SIZES"]["S"]
    # the corpus' own correctness-run shape sizes must not shadow the arena's preset/sample axis.
    assert not set(tsvc.corpus_symbol_values("tsvc2_5")) & set(tsvc._SHAPE_SYMS)


def test_tsvc2_5_scalar_symbols_resolve_from_corpus_sizes():
    # A tsvc2_5 kernel carrying S must size it from the corpus SIZES dict; reaching for a tsvc2-only
    # S_VALUE raised AttributeError, which the per-kernel except in tsvc_full swallowed into a silent skip.
    k = tsvc.iter_tsvc_kernels(only=["ext_tile_2d_sym"], corpus="tsvc2_5")[0]
    sdfg = tsvc.build_sdfg(k, "simplify-parallel")
    parent, node = get_strategy("skip-taskloops")(sdfg)[0]
    b = extract_nest_to_sdfg(parent, node, name="ext_tile_2d_sym")
    assert "S" in b.symbols  # guards the premise: this kernel is what exercises the S path
    assert tsvc.sample_sizes(k, b, preset="S")["S"] == vars(tsvc.corpus_module("tsvc2_5"))["SIZES"]["S"]
    # a corpus tile symbol likewise takes its declared value: 0 would be a degenerate (empty) tile.
    tiled = tsvc.iter_tsvc_kernels(only=["jacobi2d_tiled_sym"], corpus="tsvc2_5")[0]
    tp, tn = get_strategy("skip-taskloops")(tsvc.build_sdfg(tiled, "simplify-parallel"))[0]
    tb = extract_nest_to_sdfg(tp, tn, name="jacobi2d_tiled_sym")
    assert tsvc.sample_sizes(tiled, tb, preset="S")["T"] == vars(tsvc.corpus_module("tsvc2_5"))["SIZES"]["T"]


@pytest.mark.parametrize("opt_mode", list(tsvc.OPT_MODES))
def test_build_sdfg_propagates_a_derived_loop_bound_back_to_its_real_parameter(opt_mode):
    """s122 loops ``range(n1 - 1, LEN_1D, n3)``. The frontend names that bound with a fresh symbol and
    assigns it on an inter-state edge (``n1_minus_1 = n1 - 1``); the splitter leaves the edge outside the
    nest, so the nest takes ``n1_minus_1`` as a free argument -- which nothing can bind, because the kernel
    declares ``n1``. The kernel was then dropped from the sweep with a raise that looked principled.

    Every mode must propagate it away: it is a frontend artifact, not part of a variant's definition.
    ``canonicalize`` always did (its pipeline runs the pass); ``simplify-parallel`` and ``auto-opt`` did
    not, since DaCe's ``Simplify`` does not include it.
    """
    kernel = tsvc.iter_tsvc_kernels(only=["s122"])[0]
    assert "n1" in kernel.params, "premise: n1 is the parameter the corpus actually declares"
    parent, node = get_strategy("outer")(tsvc.build_sdfg(kernel, opt_mode))[0]
    boundary = extract_nest_to_sdfg(parent, node, name="s122_n0")
    # Only the LOOP BOUND's derived name is at issue. `__sym_LEN_1D_minus_k` is a different animal -- the
    # frontend's name for the `b[LEN_1D - k]` subscript, which is nest-assigned and sizes nothing.
    assert "n1_minus_1" not in boundary.symbols, f"derived bound leaked: {boundary.symbols}"
    sizes = tsvc.sample_sizes(kernel, boundary, preset="S")  # the raise this fixes
    if "n1" in boundary.symbols:  # folded to a constant under some modes; bound from params under others
        assert sizes["n1"] == kernel.params["n1"]


def synthetic_boundary(map_range: str, extra_symbols=(), subscript: str = "i"):
    """A one-map nest over ``a[LEN_1D]`` whose map range is ``map_range`` and whose write lands at
    ``a[subscript]``, as a bare :class:`Boundary`. Keeps the sizing contract testable without leaning on
    which nest DaCe's splitter happens to pick.

    ``map_range`` vs ``subscript`` is the discriminator the contract turns on: a symbol in the RANGE sizes
    the work (0 -> zero-trip -> vacuous), while one in the SUBSCRIPT only selects which element it lands on.
    """
    sdfg = dace.SDFG("sized_nest")
    sdfg.add_array("a", [dace.symbol("LEN_1D")], dace.float64)
    for s in extra_symbols:
        sdfg.add_symbol(s, dace.int64)  # arglist resolves a free symbol's dtype from sdfg.symbols
    state = sdfg.add_state(is_start_block=True)
    state.add_mapped_tasklet("t", {"i": map_range}, {},
                             "out = 1.0", {"out": dace.Memlet(f"a[{subscript}]")},
                             external_edges=True)
    symbols = ["LEN_1D", *extra_symbols]
    return Boundary(inputs=[], outputs=["a"], symbols=symbols, nsdfg_node=None, state=None, standalone_sdfg=sdfg)


def test_sample_sizes_zeroes_a_symbol_the_nest_never_takes_as_an_argument():
    # The PROVABLE zero: a value that is never passed cannot reach the computation.
    k = tsvc.TsvcKernel(key="synthetic", program=None, regime="1d", params={}, corpus="tsvc2")
    b = synthetic_boundary("0:LEN_1D", extra_symbols=["never_passed"])
    assert "never_passed" not in b.standalone_sdfg.arglist()
    sizes = tsvc.sample_sizes(k, b, preset="S")
    assert sizes["never_passed"] == 0 and sizes["LEN_1D"] == tsvc._PRESET["LEN_1D"]["S"]


def test_sample_sizes_zeroes_a_leaked_index_but_raises_on_a_bound_of_the_same_shape():
    """THE DISCRIMINATOR: both symbols are passed to the nest and neither is bindable, so only their USE
    separates them -- one sizes the work, the other only picks an element.

    Same nest, same symbol shape, opposite verdicts. A rule keyed on anything but the use (e.g. "the nest
    assigns it") cannot tell these two apart.
    """
    k = tsvc.TsvcKernel(key="synthetic", program=None, regime="1d", params={}, corpus="tsvc2")

    # in the map RANGE -> 0 makes the map zero-trip -> oracle and candidate agree on untouched memory.
    bound = synthetic_boundary("0:bound", extra_symbols=["bound"])
    assert "bound" in bound.standalone_sdfg.arglist()
    with pytest.raises(ValueError, match="bound"):
        tsvc.sample_sizes(k, bound, preset="S")

    # in the SUBSCRIPT only -> 0 still runs the full LEN_1D iteration space, just landing at a[0+i].
    off = synthetic_boundary("0:LEN_1D", extra_symbols=["off"], subscript="i + off")
    assert "off" in off.standalone_sdfg.arglist(), "premise: the nest is actually passed this symbol"
    assert tsvc.sample_sizes(k, off, preset="S")["off"] == 0


def test_sample_sizes_zeroes_the_leaked_outer_index_of_a_peeled_inner_nest():
    """REGRESSION (s1115, skip-taskloops): the inner nest is peeled with its outer index fixed, so ``i``
    enters as a free argument that the nest only CONSUMES (``aa[i, j]``, ``cc[j, i]``) and never assigns.

    A rule keyed on "the nest assigns it" raises here and takes the kernel out of the arena entirely. The
    work is a full row either way -- ``i`` picks WHICH row, not how many -- so 0 is sound.
    """
    real = tsvc.iter_tsvc_kernels(only=["s1115"])[0]
    parent, node = get_strategy("skip-taskloops")(tsvc.build_sdfg(real, "simplify-parallel"))[0]
    boundary = extract_nest_to_sdfg(parent, node, name="s1115_n0")
    sdfg = boundary.standalone_sdfg
    # the premises that make this the case the old nest-assigned allowance could not express
    assert "i" in sdfg.arglist(), "s1115's i is no longer passed -- it would take the never-passed proof"
    assert "i" not in nest_defined_symbols(sdfg), "s1115 now assigns i -- re-pick the kernel"
    assert "i" not in trip_count_symbols(sdfg), "s1115's i now sizes work -- it must raise, not zero"
    assert tsvc.sample_sizes(real, boundary, preset="S")["i"] == 0


def test_sample_sizes_zeroes_a_leaked_induction_start():
    """s123's ``j``: the corpus inits the counter before the loop (``j = -1``), the splitter leaves that
    init outside the nest, and the counter enters as a free argument whose only uses are ``a[j]`` and the
    ``j = j + 1`` inter-state ASSIGNMENT -- never a condition. So it cannot size the work either.

    0 is sound but SHIFTS the result (``a[0..]`` not ``a[-1..]``); oracle and candidate merely agree on the
    shift. Pinned to keep an assignment from being read as a trip-count use, which would raise here.
    """
    real = tsvc.iter_tsvc_kernels(only=["s123"])[0]
    nests = extract_all_nests(lambda: tsvc.build_sdfg(real, "simplify-parallel"), "outer", real.key)
    boundary = nests[0][3]
    sdfg = boundary.standalone_sdfg
    assert "j" in {str(s) for s in sdfg.free_symbols}, "s123's j is no longer free -- re-pick the kernel"
    assert "j" in sdfg.arglist(), "s123's j is no longer passed -- this would be the never-passed proof"
    assert "j" not in trip_count_symbols(sdfg), "s123's j now sizes work -- it must raise, not zero"
    assert tsvc.sample_sizes(real, boundary, preset="S")["j"] == 0


def test_opt_modes_produce_valid_splittable_sdfgs():
    # The pre-split axis is exactly {simplify-parallel, canonicalize, auto-opt}; each yields a validating
    # SDFG on which the loopnest-splitting pass still finds a compute nest.
    assert tsvc.OPT_MODES == ("simplify-parallel", "canonicalize", "auto-opt")
    k = tsvc.iter_tsvc_kernels(only=["s000"])[0]
    for mode in tsvc.OPT_MODES:
        sdfg = tsvc.build_sdfg(k, mode)
        sdfg.validate()
        refs = get_strategy("skip-taskloops")(sdfg)
        assert refs, f"loopnest split found no nest after opt_mode={mode!r}"
    with pytest.raises(ValueError):
        tsvc.build_sdfg(k, "simplify")  # dropped mode -> explicit error, not a silent bare SDFG


# --- cross-language job -------------------------------------------------------------------------------
def test_fortran_unmunge_reverses_leading_underscore():
    order = ["x_sym_out_i", "a", "b", "LEN_1D"]
    names = ["__sym_out_i", "a", "b", "LEN_1D"]  # the real SDFG/size names
    assert crosslang_xl.fortran_unmunge(order, names) == ["__sym_out_i", "a", "b", "LEN_1D"]


def test_validate_preset_caps_at_m():
    assert crosslang_xl.validate_preset("S") == "S"  # small stays small
    assert crosslang_xl.validate_preset("M") == "M"
    assert crosslang_xl.validate_preset("XL") == "M"  # never run the O(N) oracle at XL
    assert crosslang_xl.validate_preset("L") == "M"


def test_crosslang_run_kernel_c_and_fortran(tmp_path):
    tcs = tsvc_arena.discover_toolchains("gcc")
    compilers = crosslang_xl.lang_compilers(["c", "fortran"], tcs)
    if not compilers.get("fortran"):
        pytest.skip("no gfortran")
    k = tsvc.iter_tsvc_kernels(only=["s000"])[0]
    res = crosslang_xl.run_kernel(k, ["c", "fortran"], compilers, "skip-taskloops", "S", reps=2, workdir=tmp_path)
    assert "skipped" not in res, res.get("skipped")
    # the run sweeps the FP-precision-level x cost-model matrix; the strict-ieee rung of BOTH languages
    # must be bit-exact vs the same oracle (no FMA, no reassociation -> C and Fortran reproduce it exactly).
    for lang in ("c", "fortran"):
        strict = [
            c for c in res["cells"] if c["language"] == lang and c["fp_level"] == "strict-ieee" and c["compiler"] != "-"
        ]
        assert strict, f"no strict-ieee cell for {lang}"
        assert all(c["ok"] and c["maxdiff"] == 0.0 for c in strict), strict
    # every FP level was actually swept for C
    levels = {c["fp_level"] for c in res["cells"] if c["language"] == "c" and c["compiler"] != "-"}
    assert set(flags.FP_LEVELS) <= levels
    assert "validate" in res["sizes"] and "time" in res["sizes"]


def test_crosslang_2d_fortran_multiline_signature(tmp_path):
    """A 2D kernel whose Fortran signature wraps across lines with ``&`` continuations must still parse
    and validate -- both that the arg parser strips ``&`` and that numpyto's Fortran gives the SAME
    result as C on the SAME (C-contiguous) 2D buffer (numpyto reverses the index order for column-major)."""
    tcs = tsvc_arena.discover_toolchains("gcc")
    compilers = crosslang_xl.lang_compilers(["c", "fortran"], tcs)
    if not compilers.get("fortran"):
        pytest.skip("no gfortran")
    k = tsvc.iter_tsvc_kernels(only=["s1115"])[0]  # 2D, long (multi-line &-continued) Fortran signature
    res = crosslang_xl.run_kernel(k, ["c", "fortran"], compilers, "skip-taskloops", "S", reps=2, workdir=tmp_path)
    assert "skipped" not in res, res.get("skipped")
    ftn_strict = [
        c for c in res["cells"]
        if c["language"] == "fortran" and c["fp_level"] == "strict-ieee" and c["compiler"] != "-"
    ]
    assert ftn_strict and all(c["ok"] and c["maxdiff"] == 0.0 for c in ftn_strict), ftn_strict  # 2D col-major handled
    c_strict = [
        c for c in res["cells"] if c["language"] == "c" and c["fp_level"] == "strict-ieee" and c["compiler"] != "-"
    ]
    assert c_strict and all(c["ok"] for c in c_strict)


def test_rank_and_size_raises_on_rank_without_size(monkeypatch):
    for v in harness.RANK_VARS + harness.SIZE_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("PMIX_RANK", "2")  # a rank var but no recognized size var
    with pytest.raises(RuntimeError, match="no recognized size"):
        tsvc_arena.rank_and_size()


@pytest.mark.integration  # heavy multi-rank mpirun launch: slow + flaky on a 2-core CI runner
def test_mpirun_crosslang_distributed_preset_s(tmp_path):
    """Cross-language job under a real ``mpirun`` (2 ranks) at preset S: ranks self-partition the combined
    corpus, every kernel yields exactly one valid JSON, and no two ranks collide on a compile artifact."""
    mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
    if not mpirun:
        pytest.skip("no mpirun/mpiexec on PATH")
    keys = ["s000", "s111", "s112", "s113"]
    out = tmp_path / "res"
    repo = os.path.dirname(os.path.dirname(os.path.dirname(tsvc_arena.__file__)))
    env = {
        **os.environ, "PYTHONPATH": repo,
        "UCX_VFS_ENABLE": "n",
        "MPI4PY_RC_INITIALIZE": "0",
        "MPI4PY_RC_FINALIZE": "0",
        "OMPI_MCA_pml": "ob1",
        "OMPI_MCA_btl": "self,vader"
    }
    cmd = [mpirun, "-n", "2"]
    if "open-mpi" in subprocess.run([mpirun, "--version"], capture_output=True, text=True).stdout.lower():
        cmd.append("--oversubscribe")
    cmd += [
        sys.executable, "-m", "nestforge.perf.crosslang_xl", "--corpora", "tsvc2", "--languages", "c", "--preset", "S",
        "--compilers", "gcc", "--reps", "1", "--only", *keys, "--out",
        str(out)
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=400, cwd=repo, env=env)
    assert p.returncode == 0, p.stderr[-2000:]
    produced = {jf.stem for jf in (out).glob("*.json")}
    assert produced == {f"tsvc2_{k}" for k in keys}, f"{sorted(produced)}\n{p.stdout[-1500:]}"
    for jf in out.glob("*.json"):
        assert "cells" in json.loads(jf.read_text())  # complete, non-racy write
    assert "rank 0/2" in p.stdout and "rank 1/2" in p.stdout, p.stdout[-1500:]


def test_staticlib_overhead_run_kernel():
    k = tsvc.iter_tsvc_kernels(only=["s000"])[0]
    res = staticlib_overhead.run_kernel(k, "g++", reps=2, opt_mode="simplify-parallel", fast_libnodes=False)
    assert "skipped" not in res, res.get("skipped")
    assert res["monolithic_ms"] > 0 and res["external_ms"] > 0 and res["overhead_ratio"] > 0


# --- fault isolation: a crashing / hanging compiled kernel must not take down the rank ----------------
def test_run_isolated_returns_child_result():
    assert run_isolated(lambda: {"ok": True, "v": 3}) == {"ok": True, "v": 3}


def test_run_isolated_survives_segfault():
    # Contract: a crashing child yields an error sentinel and the PARENT survives (does not segfault).
    # In a single-threaded runner the error is "crashed (signal 11)"; under a multi-threaded test runner
    # (pytest-xdist) the fork can deadlock the child instead, in which case the timeout fires -- either
    # way the parent is intact and an error is returned, which is what the sweep rank relies on.
    def boom():
        ctypes.string_at(0)  # null deref -> SIGSEGV in the child
        return {"unreached": True}

    res = run_isolated(boom, timeout=15.0)
    assert "error" in res


def test_run_isolated_kills_runaway():

    def hang():
        while True:
            pass

    res = run_isolated(hang, timeout=1.0)
    assert "error" in res and "timeout" in res["error"].lower()


def test_abi_order_handles_empty_and_void_params():
    assert harness.signature_order("void k_fp64(void) {", "k_fp64") == []
    assert harness.signature_order("void k_fp64() {", "k_fp64") == []
    assert harness.signature_order("void k_fp64(double* a, int64_t N) {", "k_fp64") == ["a", "N"]


# --- distributed (multi-rank) self-partition ----------------------------------------------------------
def test_rank_and_size_reads_slurm_then_mpi(monkeypatch):
    for v in harness.RANK_VARS + harness.SIZE_VARS:
        monkeypatch.delenv(v, raising=False)
    assert tsvc_arena.rank_and_size() == (0, 1)  # plain single process
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "2")  # an mpirun (OpenMPI) launch
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    assert tsvc_arena.rank_and_size() == (2, 4)
    monkeypatch.setenv("SLURM_PROCID", "1")  # SLURM takes precedence (srun on daint)
    monkeypatch.setenv("SLURM_NTASKS", "8")
    assert tsvc_arena.rank_and_size() == (1, 8)


def test_my_slice_disjoint_and_covers():
    items = list(range(23))
    for n in (1, 2, 3, 4):
        slices = [tsvc_arena.my_slice(items, r, n) for r in range(n)]
        flat = [x for s in slices for x in s]
        assert sorted(flat) == items  # every item covered exactly once
        assert len({id(s) for s in slices}) == n  # each rank a distinct slice
        for i in range(n):
            for j in range(i + 1, n):
                assert not (set(slices[i]) & set(slices[j]))  # no overlap between ranks


@pytest.mark.integration  # heavy multi-rank mpirun launch: slow + flaky on a 2-core CI runner
def test_mpirun_distributed_no_compile_conflicts(tmp_path):
    """Launch the sweep under a real ``mpirun`` (2 ranks) and confirm the ranks self-partition, every
    kernel is measured exactly once, and no two ranks collide on a compile artifact (each kernel builds
    in its own mkdtemp, results are keyed per kernel). ``mpirun -n N python`` only sets the rank env
    vars -- the driver never calls MPI_Init -- so each rank is a plain, fork-safe process."""
    mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
    if not mpirun:
        pytest.skip("no mpirun/mpiexec on PATH")
    keys = ["s000", "s111", "s112", "s113"]  # 4 known-OK kernels -> 2 per rank
    out = tmp_path / "res"
    repo = os.path.dirname(os.path.dirname(os.path.dirname(tsvc_arena.__file__)))
    # dace transitively imports mpi4py, which auto-inits MPI under any launcher's PMI env -> abort/hang.
    # The driver never uses MPI (only the rank env vars), so disable mpi4py's auto-init entirely.
    env = {
        **os.environ, "PYTHONPATH": repo,
        "UCX_VFS_ENABLE": "n",
        "MPI4PY_RC_INITIALIZE": "0",
        "MPI4PY_RC_FINALIZE": "0",
        "OMPI_MCA_pml": "ob1",
        "OMPI_MCA_btl": "self,vader"
    }
    cmd = [mpirun, "-n", "2"]
    if "open-mpi" in subprocess.run([mpirun, "--version"], capture_output=True, text=True).stdout.lower():
        cmd.append("--oversubscribe")
    cmd += [
        sys.executable, "-m", "nestforge.perf.tsvc_arena", "--compilers", "gcc", "--reps", "1", "--seed", "0", "--only",
        *keys, "--out",
        str(out)
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=400, cwd=repo, env=env)
    assert p.returncode == 0, p.stderr[-2000:]
    # every requested kernel produced exactly one valid JSON (no lost/duplicated/corrupt writes)
    jsons = sorted((out / "seed0").glob("*.json"))
    produced = {p.stem for p in jsons}
    assert produced == set(keys), f"expected {keys}, got {sorted(produced)}\n{p.stdout[-1500:]}"
    for jf in jsons:
        d = json.loads(jf.read_text())  # parseable -> not a half-written/racy file
        assert "rows" in d or "skipped" in d
    # both ranks reported in (proof the partition actually spread across 2 processes)
    assert "rank 0/2" in p.stdout and "rank 1/2" in p.stdout, p.stdout[-1500:]


# --- driver end-to-end (all three columns) ------------------------------------------------------------
def test_run_kernel_three_columns(tmp_path):
    tcs = tsvc_arena.discover_toolchains("gcc")
    k = tsvc.iter_tsvc_kernels(only=["s000"])[0]
    res = tsvc_arena.run_kernel(k,
                                tcs,
                                "skip-taskloops",
                                "simplify-parallel",
                                seed=0,
                                reps=3,
                                random_sizes=False,
                                workdir=tmp_path)
    assert "skipped" not in res, res.get("skipped")
    row = res["rows"][0]
    # native baseline compiles + validates bit-exact; the flag-matrix finds a correct winner.
    assert row["native"] and row["native"]["ok"] and row["native"]["maxdiff"] == 0.0
    assert row["winner"] and row["winner"]["ok"]
    assert row["default"]["ok"]
    assert res["sizes"]["LEN_1D"] == tsvc._SYM_FIXED["LEN_1D"]


def test_native_work_reports_unchecked_when_nothing_was_compared():
    """A native lane whose outputs resolve to NO pointer arg compared nothing, so it must report
    unchecked/inf -- never ok/0.0. This cell is the speedup DENOMINATOR: a bogus pass would publish a
    "bit-exact" baseline that was never validated. Mirrors ``tsvc_full.native_validate_work``'s guard.

    Driven against a stock libc symbol (``abs``, a harmless scalar call) so the guard is exercised with no
    compile: the point is the empty-``outs`` verdict, not which native binary produced it.
    """
    libc = ctypes.util.find_library("c")
    if not libc:
        pytest.skip("no libc found")
    kernel = tsvc.iter_tsvc_kernels(only=["s000"])[0]
    boundary = types.SimpleNamespace(outputs=[])  # no output resolves to a pointer arg -> nothing to compare
    res = tsvc_arena.native_work(pathlib.Path(libc),
                                 "abs", [("n", "int", False)],
                                 kernel,
                                 boundary,
                                 inputs={},
                                 sizes={"n": 4},
                                 oracle={},
                                 reps=1)
    assert res["unchecked"] is True
    assert res["ok"] is False and res["maxdiff"] == float("inf")


def test_arena_sweeps_both_corpora_by_default(monkeypatch, tmp_path):
    """The driver must sweep BOTH corpora like crosslang_xl / tsvc_full / calloverhead: iterating only
    tsvc2 silently drops tsvc2_5's 65 kernels, and their absence is invisible in the output."""
    monkeypatch.setattr(tsvc_arena, "discover_toolchains",
                        lambda requested: [tsvc_arena.Toolchain("gcc", "gcc", "g++", (13, 0), "path")])
    seen = []

    def record(only=None, corpus="tsvc2"):
        seen.append(corpus)
        return []

    monkeypatch.setattr(tsvc, "iter_tsvc_kernels", record)
    assert tsvc_arena.main(["--out", str(tmp_path)]) == 0
    assert seen == ["tsvc2", "tsvc2_5"]  # both corpora, in the siblings' order


def test_tables_and_link_roundtrip(tmp_path):
    tcs = tsvc_arena.discover_toolchains("gcc")
    out = tmp_path / "results"
    seed_dir = tsvc_arena.ensure_seed_dir(out, 0)
    for key in ("s000", "s1112"):
        k = tsvc.iter_tsvc_kernels(only=[key])[0]
        wd = tmp_path / f"wd_{key}"
        wd.mkdir()
        res = tsvc_arena.run_kernel(k, tcs, "skip-taskloops", "simplify-parallel", 0, 3, False, wd)
        (seed_dir / f"{key}.json").write_text(json.dumps(res))

    report = tsvc_arena.render_tables(out, 0)
    assert "2 kernels measured" in report and "s000" in report
    assert (seed_dir / "tables.md").exists()

    link_report = tsvc_arena.link_whole_program(out, 0, tcs, "simplify-parallel", "skip-taskloops")
    assert "symbols verified present" in link_report
    assert (seed_dir / "link" / "libtsvc_all.so").exists()


# --- multi-nest kernels (s152 splits into 2 compute nests) --------------------------------------------
def test_extract_all_nests_single_and_multi():
    """The shared helper preserves the single-nest name/symbol (``<key>`` / ``<key>_fp64`` -- so the 148
    single-nest kernels are unchanged) and gives each nest of a multi-nest kernel a distinct ``_n<idx>``
    entry point, each extracted from its own FRESH SDFG (mutation-safe)."""
    single = extract_all_nests(lambda: tsvc.build_sdfg(tsvc.iter_tsvc_kernels(only=["s000"])[0], "simplify-parallel"),
                               "skip-taskloops", "s000")
    assert [(i, n, s) for i, n, s, _ in single] == [(0, "s000", "s000_fp64")]
    multi = extract_all_nests(lambda: tsvc.build_sdfg(tsvc.iter_tsvc_kernels(only=["s152"])[0], "simplify-parallel"),
                              "skip-taskloops", "s152")
    assert [(i, n, s) for i, n, s, _ in multi] == [(0, "s152_n0", "s152_n0_fp64"), (1, "s152_n1", "s152_n1_fp64")]


def test_arena_multinest_s152_aggregates_and_validates(tmp_path):
    """A multi-nest kernel is measured (not skipped): every cell aggregates the SUM over its nests, the
    schema is unchanged, and the default + winner cells validate (both nests bit-exact at strict flags)."""
    tcs = tsvc_arena.discover_toolchains("gcc")
    if not tcs:
        pytest.skip("no gcc")
    k = tsvc.iter_tsvc_kernels(only=["s152"])[0]
    res = tsvc_arena.run_kernel(k,
                                tcs,
                                "skip-taskloops",
                                "simplify-parallel",
                                seed=0,
                                reps=3,
                                random_sizes=False,
                                workdir=tmp_path)
    assert "skipped" not in res, res.get("skipped")
    # the roster records both nests with distinct entry points.
    assert [(n["name"], n["symbol"]) for n in res["nests"]] == [("s152_n0", "s152_n0_fp64"),
                                                                ("s152_n1", "s152_n1_fp64")]
    row = res["rows"][0]
    assert len(row["cells"]) == len(flags.flag_matrix("gnu"))  # one aggregated cell per flag combo, still 12
    assert row["default"]["ok"]  # both nests validate at the default flags
    assert row["winner"] and row["winner"]["ok"] and row["winner"]["time_us"] > 0.0  # a summed-over-nests winner


def test_crosslang_multinest_s152_aggregates_and_validates(tmp_path):
    """crosslang measures the multi-nest kernel (not skipped); each (lang, compiler, fp, cost) cell sums
    over both nests and the strict-ieee rung is bit-exact (both nests reproduce the oracle exactly)."""
    tcs = tsvc_arena.discover_toolchains("gcc")
    compilers = crosslang_xl.lang_compilers(["c"], tcs)
    if not compilers.get("c"):
        pytest.skip("no gcc C compiler")
    k = tsvc.iter_tsvc_kernels(only=["s152"])[0]
    res = crosslang_xl.run_kernel(k, ["c"], compilers, "skip-taskloops", "S", reps=2, workdir=tmp_path)
    assert "skipped" not in res, res.get("skipped")
    strict = [
        c for c in res["cells"] if c["language"] == "c" and c["fp_level"] == "strict-ieee" and c["compiler"] != "-"
    ]
    assert strict and all(c["ok"] and c["maxdiff"] == 0.0 for c in strict), strict


def test_link_multinest_s152_verifies_every_nest_symbol(tmp_path):
    """The whole-program link archives EVERY nest object of a multi-nest kernel into one ``lib<key>.a`` and
    verifies each nest symbol resolves in the linked ``.so`` (both ``s152_n0_fp64`` and ``s152_n1_fp64``)."""
    tcs = tsvc_arena.discover_toolchains("gcc")
    if not tcs:
        pytest.skip("no gcc")
    out = tmp_path / "results"
    seed_dir = tsvc_arena.ensure_seed_dir(out, 0)
    k = tsvc.iter_tsvc_kernels(only=["s152"])[0]
    wd = tmp_path / "wd_s152"
    wd.mkdir()
    (seed_dir / "s152.json").write_text(
        json.dumps(tsvc_arena.run_kernel(k, tcs, "skip-taskloops", "simplify-parallel", 0, 3, False, wd)))
    report = tsvc_arena.link_whole_program(out, 0, tcs, "simplify-parallel", "skip-taskloops")
    assert "2 symbols verified present, 0 missing" in report, report
    lib = ctypes.CDLL(str(seed_dir / "link" / "libtsvc_all.so"))
    for sym in ("s152_n0_fp64", "s152_n1_fp64"):
        assert lib[sym]  # each nest's distinct entry point is present in the whole-program library
