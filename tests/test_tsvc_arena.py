"""TSVC compiler-arena driver + adapter, plus the emitter user-function rewrites it depends on."""
import ctypes
import json
import os
import shutil
import subprocess
import sys

import pytest

from nestforge import tsvc
from nestforge.emit_numpy import normalize_casts
from nestforge.extract import extract_nest_to_sdfg
from nestforge.isolation import run_isolated
from nestforge.multinest import extract_all_nests
from nestforge.perf import crosslang_xl, flags, staticlib_overhead, tsvc_arena
from nestforge.strategies import get_strategy


# --- emitter: sympy user-function + qualified-math rewrites (the arena's translation depends on these) ---
def test_normalize_rewrites_int_floor_and_ceil():
    # int_floor / int_ceil are sympy user-functions (no operator); they must lower to python integer ops.
    assert normalize_casts("a[int_floor(LEN_1D, 2)]") == "a[((LEN_1D) // (2))]"
    assert normalize_casts("int_ceil(N, 4)") == "(-((-(N)) // (4)))"
    # nested user-functions resolve to a fixpoint.
    assert normalize_casts("int_floor(int_ceil(N, 2), 3)") == "(((-((-(N)) // (2)))) // (3))"


def test_normalize_rewrites_qualified_math_to_numpy():
    assert normalize_casts("dace.math.sin(x)") == "np.sin(x)"
    assert normalize_casts("math.cos(y)") == "np.cos(y)"
    assert normalize_casts("dace.math.asin(z)") == "np.arcsin(z)"  # name-mapped, not verbatim
    # a bare (unqualified) intrinsic still lowers, and a dtype cast still maps.
    assert normalize_casts("sqrt(w)") == "np.sqrt(w)"
    assert normalize_casts("dace.float64(v)") == "np.float64(v)"


# --- adapter ------------------------------------------------------------------------------------------
def test_iter_and_filter_kernels():
    only = tsvc.iter_tsvc_kernels(only=["s000", "s112"])
    assert {k.key for k in only} == {"s000", "s112"}
    s000 = only[0] if only[0].key == "s000" else only[1]
    assert s000.native_cpp is not None and s000.native_cpp.exists()  # foundation ships the native baseline
    assert s000.native_symbol == "s000_d"


def test_sample_sizes_indices_zero_shapes_sized():
    k = tsvc.iter_tsvc_kernels(only=["s000"])[0]
    sdfg = tsvc.build_sdfg(k, "baseline")
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


def test_single_openmp_runtime_drops_incompatible():
    T = tsvc_arena.Toolchain
    tcs = [
        T("gcc", "gcc", "g++", (15, 0), "path"),
        T("clang", "clang", "clang++", (21, 0), "path"),
        T("nvhpc", "nvc", "nvc++", (26, 0), "path")
    ]
    kept = {t.name for t in tsvc_arena.restrict_to_single_openmp_runtime(tcs, "libomp")}
    # gcc (gomp-compat) and clang (-fopenmp=libomp) can share libomp; nvc forces its native libnvomp.
    assert kept == {"gcc", "clang"}


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
    sdfg = tsvc.build_sdfg(k, "baseline")
    parent, node = get_strategy("skip-taskloops")(sdfg)[0]
    b = extract_nest_to_sdfg(parent, node, name="s000")
    assert tsvc.sample_sizes(k, b, preset="XL")["LEN_1D"] == tsvc._PRESET["LEN_1D"]["XL"]
    assert tsvc.sample_sizes(k, b, preset="S")["LEN_1D"] == 512


def test_opt_modes_produce_valid_splittable_sdfgs():
    # The pre-split axis is exactly {baseline, canonicalize}; each yields a validating SDFG on which
    # the loopnest-splitting pass still finds a compute nest.
    assert tsvc.OPT_MODES == ("baseline", "canonicalize")
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
    for v in tsvc_arena._RANK_VARS + tsvc_arena._SIZE_VARS:
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
    res = staticlib_overhead.run_kernel(k, "g++", reps=2, opt_mode="baseline", fast_libnodes=False)
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
    assert tsvc_arena.abi_order("void k_fp64(void) {", "k_fp64") == []
    assert tsvc_arena.abi_order("void k_fp64() {", "k_fp64") == []
    assert tsvc_arena.abi_order("void k_fp64(double* a, int64_t N) {", "k_fp64") == ["a", "N"]


# --- distributed (multi-rank) self-partition ----------------------------------------------------------
def test_rank_and_size_reads_slurm_then_mpi(monkeypatch):
    for v in tsvc_arena._RANK_VARS + tsvc_arena._SIZE_VARS:
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
        sys.executable, "-m", "nestforge.perf.tsvc_arena", "--select", "tsvc", "--compilers", "gcc", "--reps", "1",
        "--seed", "0", "--only", *keys, "--out",
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
                                "baseline",
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


def test_tables_and_link_roundtrip(tmp_path):
    tcs = tsvc_arena.discover_toolchains("gcc")
    out = tmp_path / "results"
    seed_dir = tsvc_arena.ensure_seed_dir(out, 0)
    for key in ("s000", "s1112"):
        k = tsvc.iter_tsvc_kernels(only=[key])[0]
        wd = tmp_path / f"wd_{key}"
        wd.mkdir()
        res = tsvc_arena.run_kernel(k, tcs, "skip-taskloops", "baseline", 0, 3, False, wd)
        (seed_dir / f"{key}.json").write_text(json.dumps(res))

    report = tsvc_arena.render_tables(out, 0)
    assert "2 kernels measured" in report and "s000" in report
    assert (seed_dir / "tables.md").exists()

    link_report = tsvc_arena.link_whole_program(out, 0, tcs, "baseline", "skip-taskloops")
    assert "symbols verified present" in link_report
    assert (seed_dir / "link" / "libtsvc_all.so").exists()


# --- multi-nest kernels (s152 splits into 2 compute nests) --------------------------------------------
def test_extract_all_nests_single_and_multi():
    """The shared helper preserves the single-nest name/symbol (``<key>`` / ``<key>_fp64`` -- so the 148
    single-nest kernels are unchanged) and gives each nest of a multi-nest kernel a distinct ``_n<idx>``
    entry point, each extracted from its own FRESH SDFG (mutation-safe)."""
    single = extract_all_nests(lambda: tsvc.build_sdfg(tsvc.iter_tsvc_kernels(only=["s000"])[0], "baseline"),
                               "skip-taskloops", "s000")
    assert [(i, n, s) for i, n, s, _ in single] == [(0, "s000", "s000_fp64")]
    multi = extract_all_nests(lambda: tsvc.build_sdfg(tsvc.iter_tsvc_kernels(only=["s152"])[0], "baseline"),
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
                                "baseline",
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
        json.dumps(tsvc_arena.run_kernel(k, tcs, "skip-taskloops", "baseline", 0, 3, False, wd)))
    report = tsvc_arena.link_whole_program(out, 0, tcs, "baseline", "skip-taskloops")
    assert "2 symbols verified present, 0 missing" in report, report
    lib = ctypes.CDLL(str(seed_dir / "link" / "libtsvc_all.so"))
    for sym in ("s152_n0_fp64", "s152_n1_fp64"):
        assert lib[sym]  # each nest's distinct entry point is present in the whole-program library
