"""The comprehensive TSVC full-matrix job: for every kernel of BOTH corpora (``tsvc2`` + ``tsvc2_5``),
measure three lanes and a large sweep, with MEDIAN-of-N timing at a memory-bound profiling size.

Lanes / columns per kernel
--------------------------
1. **native original.cpp** -- the ``<key>_original.cpp`` scalar reference loop at ``-O3 -march=native``
   (one C++ compiler). The classic "how well does the compiler auto-vectorize the reference" column.
2. **DaCe-cpp baseline** -- DaCe's OWN C++ codegen of the EXTRACTED-NEST standalone SDFG (owned
   direct-compile via ``build.build_sdfg`` -- NO cmake), a SINGLE C++ compiler, default cost model,
   ``-O3`` + **strict-ieee** FP (``-ffp-contract=off``). Fanned over the codegen-implementation axis:
   ``experimental`` (the readable constexpr-index-fn codegen, nest-forge's DEFAULT and the speedup
   denominator) plus ``legacy`` where the DaCe build has it (a measured variant). THIS default codegen is
   the baseline every nest-forge cell divides by; the standalone nest (not the whole kernel) does the SAME
   work the nest-forge lanes do, so the time is apples-to-apples even for kernels peeled to an inner nest.
   The median time is reported always; the strict cross-check bit-matches for most kernels
   (loop-carried-state recurrences are flagged in the tables -- see the lane-2 section comment).
3. **nest-forge external-nest** -- the extracted nest translated by numpyto and compiled, swept over the
   full axis matrix below.

The axis matrix (lane 3), per kernel
------------------------------------
  * **opt-mode**      : ``simplify-parallel`` | ``canonicalize`` | ``auto-opt`` (the pre-split SDFG
    optimization; changes the emitted source, so it is an emit-time axis, not just a flag).
  * **language**      : ``c`` | ``c++`` | ``fortran``. numpyto has NO C++ target, so **C++ = the emitted
    C source recompiled by a C++ frontend** (``-x c++`` + a ``restrict`` / ``__builtin_complex`` shim;
    see :func:`nestforge.perf.flags.cxx_source_flags`).
  * **parallelization**: ``sequential`` | ``auto-par`` (the compiler's plain-loop auto-parallelizer:
    gcc ``-ftree-parallelize-loops -fopenmp``, nvc ``-Mconcur``, icx ``-qopenmp -parallel``; clang/flang
    have none -> recorded unsupported, not dropped).
  * **compiler**      : every discovered family (gcc / clang / nvhpc / intel).
  * **cost-model**    : ``default`` | ``cheap`` | ``no-vec`` (the shared vectorizer cost axis).
  * **FP mode**       : a REDUCED two-rung axis ``default-fp`` | ``no-fast-errno`` for timing, PLUS a
    ``strict-ieee`` correctness GATE cell (sequential, default cost) that must be BIT-EXACT vs the numpy
    oracle. (See :data:`flags.REDUCED_FP_MODES`.)

Sizing
------
  * **validate** at a small preset (cap ``M``) -- the pure-Python O(N) oracle is slow, and the compiled
    ``.so`` is size-agnostic (LEN is a runtime arg), so validating small and timing large exercises the
    same code correctly and fast.
  * **profile/time** at the ``PROF`` preset -- sized so one fp64 array (>=128 MiB) clearly exceeds the
    GH200 Grace L3 (~114 MB/socket), i.e. the realistic memory-bound regime, but smaller than ``XL``
    (whose alloc/first-touch dominates). Pass ``--profile-preset XL`` (or the sbatch ``RUN_XL=1`` block)
    for the big confirmation run.

Timing method / speed
---------------------
  * **median of N** individually-timed reps (plus min / p25 / p75 / mean) -- robust to OS jitter.
  * Each cell is compiled ONCE and its ``.so`` reused for validate + all timing reps.
  * Identical flag sets are **deduped** to one compile; per-cell compiles run through a bounded thread
    pool (``--compile-jobs``) since compilation, not the timed run, is the bottleneck.
  * A fast **VALIDATE** pass (compile + one small run) precedes the **TIMING** pass; only cells that
    validate are timed, so broken cells never waste timing budget.
  * A language/kernel whose numpyto emit fails is skipped (no wasted compiles).

Every compiled-kernel execution runs in a forked child (:func:`nestforge.isolation.run_isolated`) so a
segfault / OOM / runaway in freshly-compiled code kills only the child, never the sweep rank. Kernels
self-partition across ranks (SLURM ``srun`` or MPI) via :func:`rank_and_size` + :func:`my_slice`.
``--tables-only`` merges the per-kernel JSON into markdown.

Usage::

    python -m nestforge.perf.tsvc_full --corpora tsvc2 tsvc2_5 --profile-preset PROF \\
        --languages c c++ fortran --parallelism both --reps 11 --compilers auto
    python -m nestforge.perf.tsvc_full --tables-only --out perf_results/tsvc_full
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import shutil
import socket
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import dace  # noqa: F401 -- ensure the real DaCe package is importable (not a cwd stub)

from nestforge import tsvc
from nestforge.arena import maxdiff, make_inputs, run_oracle
from nestforge.build import BuildOptions, codegen_impls_available, default_codegen_impl
from nestforge.build import build_sdfg as dace_build_sdfg
from nestforge.extract import extract_nest_to_sdfg
from nestforge.isolation import run_isolated
from nestforge.perf import flags
from nestforge.perf.crosslang_xl import fortran_unmunge, lang_compilers
from nestforge.perf.tsvc_arena import Toolchain, discover_toolchains
from nestforge.perf.harness import (RUN_TIMEOUT_S, c_argtypes, call_c, finite, fmt_us, jsonable, my_slice, native_setup,
                                    native_symbol, rank_and_size, run_compile, signature_order)
from nestforge.strategies import empty_strategy_reason, get_strategy, is_parallel_nest
from nestforge.translate import emit_sources, prepare

#: numpyto emit target + source suffix per language. C and C++ share the C target (numpyto has no C++
#: target); C++ just recompiles the emitted ``.c`` with a C++ frontend.
_EMIT = {"c": ("c", ".c"), "c++": ("c", ".c"), "fortran": ("fortran", ".f90")}
#: presets that are too large for the O(N) pure-Python oracle -> validate at ``M`` instead.
_VALIDATE_CAP = "M"


def validate_cap(profile_preset: str) -> str:
    """The preset to VALIDATE at for a given timing preset: the timing preset itself when it is already
    small (``S``/``M``), else ``M`` (``L``/``PROF``/``XL`` would make the pure-Python oracle take
    minutes). The ``.so`` is size-agnostic, so small-validate + large-time exercises the same code."""
    return profile_preset if profile_preset in ("S", _VALIDATE_CAP) else _VALIDATE_CAP


def default_threads() -> int:
    """Default auto-par thread count. ``OMP_NUM_THREADS`` may be a comma list (e.g. ``72,8`` for nested
    levels), which ``int()`` rejects -- fall back to the CPU count then rather than crashing at parse."""
    try:
        return int(os.environ.get("OMP_NUM_THREADS") or (os.cpu_count() or 4))
    except ValueError:
        return os.cpu_count() or 4


# --- median-of-N timing (pure functions: unit-tested without a compiler) -----------------------------
def summarize_times(samples: List[float]) -> Dict[str, float]:
    """Robust summary of per-rep microsecond samples: median (the headline), min (least-perturbed run),
    p25 / p75 (linear-interpolated quartiles) and mean. Empty input -> all ``inf``."""
    if not samples:
        return {
            "median_us": float("inf"),
            "min_us": float("inf"),
            "p25_us": float("inf"),
            "p75_us": float("inf"),
            "mean_us": float("inf")
        }
    s = sorted(samples)
    n = len(s)

    def percentile(p: float) -> float:
        if n == 1:
            return s[0]
        idx = p * (n - 1)
        lo = math.floor(idx)
        hi = math.ceil(idx)
        return s[lo] + (s[hi] - s[lo]) * (idx - lo)

    return {
        "median_us": float(statistics.median(s)),
        "min_us": float(s[0]),
        "p25_us": percentile(0.25),
        "p75_us": percentile(0.75),
        "mean_us": float(sum(s) / n),
    }


def collect_samples(fn, cargs, reps: int) -> List[float]:
    """Warm once, then time ``reps`` individual calls on the reused args -- one microsecond sample per
    rep (median-friendly), NOT one mean over the whole loop."""
    fn(*cargs)  # warm
    samples: List[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*cargs)
        samples.append((time.perf_counter() - t0) * 1e6)
    return samples


def c_call_args(order: List[str], argtypes: list, work: Dict[str, np.ndarray], sizes: Dict[str, int]) -> list:
    """ctypes arguments for the emitted kernel, in C-signature order: an array -> its buffer pointer, a
    size symbol -> an int64 by value."""
    return [work[a].ctypes.data_as(t) if a in work else ctypes.c_int64(int(sizes[a])) for a, t in zip(order, argtypes)]


# --- lane 3: nest cell validate / time (run inside a forked child) -----------------------------------
def nest_validate_work(so: Path, symbol: str, order: List[str], argtypes, boundary, validate_sizes, oracle,
                       atol: float) -> Dict:
    """Correctness at the SMALL preset: bind, run once, maxdiff vs the oracle. Fast (small buffers)."""
    vin = make_inputs(boundary, validate_sizes, seed=0)
    vout, _ = call_c(so, symbol, order, argtypes, boundary, vin, validate_sizes, reps=1)
    md = float(maxdiff(oracle, vout))
    return {"ok": bool(md <= atol), "maxdiff": md}


def nest_timing_work(so: Path, symbol: str, order: List[str], argtypes, time_inputs, time_sizes, reps: int) -> Dict:
    """Median-of-N timing at the PROFILING preset. ``time_inputs`` is the context's pre-allocated time-size
    buffer set, allocated ONCE per opt-mode and reused across every timing cell: the fork hands this child
    its own COW copy, so it times in place without disturbing the parent or siblings (timing does not
    validate output, so buffer freshness is not required)."""
    fn = ctypes.CDLL(str(so))[symbol]
    fn.argtypes, fn.restype = argtypes, None
    cargs = c_call_args(order, argtypes, time_inputs, time_sizes)
    return summarize_times(collect_samples(fn, cargs, reps))


def native_validate_work(so, symbol, sig, kernel, boundary, validate_sizes, oracle) -> Dict:
    buffers = make_inputs(boundary, validate_sizes, seed=0)  # fresh + correct (validation runs in place)
    fn, cargs, ptr_names = native_setup(so, symbol, sig, kernel, buffers, validate_sizes)
    fn(*cargs)
    outs = {o: buffers[o] for o in boundary.outputs if o in ptr_names}
    if not outs:  # nothing to compare -> UNCHECKED, never report 0.0/ok for an unvalidatable lane
        return {"ok": False, "maxdiff": float("inf"), "unchecked": True}
    md = float(maxdiff({k: oracle[k] for k in outs}, outs))
    return {"ok": bool(md <= 1e-6), "maxdiff": md}


def native_timing_work(so, symbol, sig, kernel, time_inputs, time_sizes, reps) -> Dict:
    fn, cargs, _ = native_setup(so, symbol, sig, kernel, time_inputs, time_sizes)  # COW copy in this fork
    return summarize_times(collect_samples(fn, cargs, reps))


def measure_native_lane(cxx: str, family: str, kernel, boundary, validate_sizes, time_inputs, time_sizes, oracle,
                        reps: int, cxx_std: str, workdir: Path) -> Optional[Dict]:
    """Lane 1: compile ``_original.cpp`` at ``-O3 -march=native``, validate@small + time@prof (median).
    ``None`` when the kernel ships no native source or the family has no C++ compiler."""
    cpp = kernel.native_cpp
    if cpp is None or cxx is None:
        return None
    nat_flags = flags.base_flags(family) + ["-std=" + cxx_std]
    text = cpp.read_text()
    try:
        symbol = native_symbol(text, kernel.native_symbol)
        sig = tsvc.native_signature(text, symbol)
    except LookupError as e:
        return {"ok": False, "error": str(e), **summarize_times([])}
    so = workdir / f"{kernel.key}_{family}_native.so"
    ok, compile_us, err = run_compile([cxx, *nat_flags, str(cpp), "-o", str(so)])
    if not ok:
        return {"ok": False, "error": err, "compile_us": compile_us, **summarize_times([])}
    vres = run_isolated(lambda: native_validate_work(so, symbol, sig, kernel, boundary, validate_sizes, oracle))
    if "error" in vres:
        return {"ok": False, "error": vres["error"], "compile_us": compile_us, **summarize_times([])}
    tres = run_isolated(lambda: native_timing_work(so, symbol, sig, kernel, time_inputs, time_sizes, reps),
                        timeout=RUN_TIMEOUT_S)
    stats = summarize_times([]) if "error" in tres else tres
    return {
        "compiler": family,
        "ok": vres["ok"],
        "maxdiff": vres["maxdiff"],
        "unchecked": vres.get("unchecked", False),
        "compile_us": compile_us,
        "error": tres.get("error"),
        **stats
    }


# --- lane 2: DaCe-cpp baseline (owned direct-compile of the EXTRACTED-NEST standalone SDFG) -----------
# The speedup baseline must do the SAME work as the nest-forge lanes, so it is DaCe's own codegen of the
# EXACT extracted nest (``boundary.standalone_sdfg``), not the whole-kernel SDFG. That matters for the
# many multi-level kernels the strategy peels to an INNER nest (a leaked outer index fixed to 0, e.g.
# s1115): the whole-kernel SDFG would compute ALL rows -- ~LEN more work -- and its time would inflate the
# speedup meaninglessly, whereas the standalone nest does the same one-row work the nest-forge lanes do.
# The median time is ALWAYS reported (same iteration space -> a fair baseline). Validation vs the numpy
# oracle is a recorded cross-check: it is bit-exact for most kernels, but DaCe's raw codegen and numpyto
# lower the boundary contract for PROMOTED loop-carried state (s111/s112 recurrences) differently, so
# those show a non-bit-exact baseline (flagged in the tables) while their timing stays representative.
def dace_run_work(built, boundary, validate_sizes, time_inputs, time_sizes, oracle, atol: float, reps: int) -> Dict:
    """Validate@small then time@prof DaCe's own codegen, in the forked child. ``built`` (its ``CDLL``) and
    ``time_inputs`` are shared copy-on-write across the fork, so a large time-size run OOM-kills only the
    child. Validation uses a fresh small buffer; timing reuses the context's pre-allocated time buffer."""
    vbuf = make_inputs(boundary, validate_sizes, seed=0)  # fresh + correct (validation runs in place)
    built.run(vbuf, validate_sizes)  # init -> program -> close
    outs = {o: vbuf[o] for o in boundary.outputs if o in vbuf}
    if outs:
        md = float(maxdiff({o: oracle[o] for o in outs}, outs))
        verdict = {"ok": bool(md <= atol), "maxdiff": md}
    else:  # nothing to compare against -> UNCHECKED, never report 0.0/ok for an unvalidatable lane
        verdict = {"ok": False, "maxdiff": float("inf"), "unchecked": True}
    tbuf = time_inputs
    built.init(time_sizes)
    try:
        # bind once, then time the bare ctypes call in the rep loop -- matches the native/nest bare-ctypes
        # timing (no per-rep numpy->ctypes marshaling).
        fn, cargs = built.bind_program(tbuf, time_sizes)
        fn(*cargs)  # warm
        samples: List[float] = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn(*cargs)
            samples.append((time.perf_counter() - t0) * 1e6)
    finally:
        built.close()
    return {**verdict, **summarize_times(samples)}


def measure_dace_cpp_lane(tc: Toolchain,
                          boundary,
                          validate_sizes,
                          time_inputs,
                          time_sizes,
                          oracle,
                          reps: int,
                          cxx_std: str,
                          workdir: Path,
                          codegen_impl: Optional[str] = None) -> Dict:
    """Lane 2: DaCe's own C++ codegen of the extracted-nest standalone SDFG, ``-O3`` + strict-ieee, one C++
    compiler. The owned build (``build.build_sdfg``) compiles DIRECTLY (no cmake) into ``workdir`` -- a
    per-kernel mkdtemp, so concurrent ranks never share a build dir (the cmake-deadlock trap does not
    apply). The median time is reported whether or not the run bit-matches (see the section comment).

    ``codegen_impl`` selects the DaCe CPU codegen (``experimental`` -- the default -- | ``legacy``); ``None``
    -> :func:`build.default_codegen_impl`. The new codegen is the speedup denominator; ``legacy`` is the
    measured variant. It is stamped into the returned dict so the reporter can group by codegen impl."""
    codegen_impl = codegen_impl or default_codegen_impl()
    if tc.cxx is None:
        return {
            "ok": False,
            "error": f"{tc.name}: no C++ compiler for the DaCe-cpp lane",
            "codegen_impl": codegen_impl,
            **summarize_times([])
        }
    fam = tc.fp_family
    dace_flags = flags.base_flags(fam) + ["-std=" + cxx_std] + flags.fp_flags(fam, "strict-ieee", "c")
    try:
        t0 = time.perf_counter()
        built = dace_build_sdfg(boundary.standalone_sdfg, workdir,
                                BuildOptions(compiler=tc.cxx, flags=dace_flags, codegen_impl=codegen_impl))
        compile_us = (time.perf_counter() - t0) * 1e6
    except Exception as e:  # codegen / compile failure must not crash the kernel
        return {
            "ok": False,
            "error": f"dace build: {type(e).__name__}: {str(e)[:200]}",
            "codegen_impl": codegen_impl,
            **summarize_times([])
        }
    atol = flags.FP_ATOL["strict-ieee"]
    try:
        res = run_isolated(
            lambda: dace_run_work(built, boundary, validate_sizes, time_inputs, time_sizes, oracle, atol, reps),
            timeout=RUN_TIMEOUT_S)
    finally:
        built.unload()  # parent side: free the dlopen mapping (the child ran in its own COW copy)
    if "error" in res:
        return {
            "ok": False,
            "error": res["error"],
            "compile_us": compile_us,
            "codegen_impl": codegen_impl,
            **summarize_times([])
        }
    return {
        "compiler": tc.name,
        "codegen_impl": codegen_impl,
        "ok": res["ok"],
        "maxdiff": res["maxdiff"],
        "unchecked": res.get("unchecked", False),
        "compile_us": compile_us,
        "error": None,
        "median_us": res["median_us"],
        "min_us": res["min_us"],
        "p25_us": res["p25_us"],
        "p75_us": res["p75_us"],
        "mean_us": res["mean_us"]
    }


def measure_dace_vectorized_lane(tc: Toolchain,
                                 boundary,
                                 validate_sizes,
                                 time_inputs,
                                 time_sizes,
                                 oracle,
                                 reps: int,
                                 cxx_std: str,
                                 workdir: Path,
                                 codegen_impl: Optional[str] = None,
                                 rounds: int = 2) -> Dict:
    """The DaCe lane WITH the multi-dim tile-op vectorizer: a coordinate-descent search over
    ``VectorizeConfig`` variants (``build_sdfg(vectorize=cfg)`` then a forked run) for the fastest config
    that still VALIDATES against the numpy oracle on this nest. Each candidate is validated on the
    ``contract-fma`` tolerance (FMA is a swept knob), so a mis-vectorization is caught, not timed. Returns
    the winning vectorized cell (``vectorized=True`` + the ``vec_variant`` name), or an error cell when no
    config validated (a non-vectorizable nest). The search's compiles are reused -- the winner's full timing
    is looked up from the descent, not re-measured."""
    if tc.cxx is None:
        return {
            "ok": False,
            "error": f"{tc.name}: no C++ compiler for the vectorized DaCe lane",
            "vectorized": True,
            **summarize_times([])
        }
    from nestforge import vectorize_variants as vv
    fam = tc.fp_family
    dace_flags = flags.base_flags(fam) + ["-std=" + cxx_std] + flags.fp_flags(fam, "contract-fma", "c")
    atol = flags.FP_ATOL["contract-fma"]
    results: Dict[tuple, Dict] = {}
    counter = {"i": 0}

    def measure(cfg) -> Optional[float]:
        counter["i"] += 1
        try:
            built = dace_build_sdfg(
                boundary.standalone_sdfg, workdir / f"vec{counter['i']}",
                BuildOptions(compiler=tc.cxx, flags=dace_flags, codegen_impl=codegen_impl, vectorize=cfg))
        except Exception:  # a config the vectorizer / codegen rejects is simply not a candidate
            return None
        try:
            res = run_isolated(
                lambda: dace_run_work(built, boundary, validate_sizes, time_inputs, time_sizes, oracle, atol, reps),
                timeout=RUN_TIMEOUT_S)
        finally:
            built.unload()
        if "error" in res or not res.get("ok") or not finite(res.get("median_us", float("inf"))):
            return None
        results[vv.resolved_key(cfg)] = {**res, "vec_variant": vv.variant_name(cfg)}
        return res["median_us"]

    best_cfg, best_t = vv.multistart_descent(vv.default_seeds(), vv.descent_axes(), measure, rounds)
    winner = results.get(vv.resolved_key(best_cfg)) if best_t is not None else None
    if winner is None:
        return {
            "ok": False,
            "error": "no vectorization config validated (non-vectorizable nest)",
            "vectorized": True,
            "compiler": tc.name,
            **summarize_times([])
        }
    return {
        "compiler": tc.name,
        "vectorized": True,
        "vec_variant": winner["vec_variant"],
        "ok": winner["ok"],
        "maxdiff": winner["maxdiff"],
        "error": None,
        "median_us": winner["median_us"],
        "min_us": winner["min_us"],
        "p25_us": winner["p25_us"],
        "p75_us": winner["p75_us"],
        "mean_us": winner["mean_us"]
    }


# --- lane 3 cell model -------------------------------------------------------------------------------
@dataclass
class Cell:
    """One lane-3 sweep point: the full axis coordinates + correctness + median timing, or an error."""
    opt_mode: str
    language: str
    compiler: str  # family label (gcc | clang | nvhpc | intel)
    parallel: str
    cost_model: str
    fp_mode: str
    role: str  # "timing" | "gate"
    nest: int = 0  # which extracted nest this cell belongs to (0-based); -1 = whole-kernel (native lane)
    veclib: str = "none"  # vector-math library axis: 'none' | the per-device characterized winner
    ok: bool = False
    maxdiff: float = float("inf")
    median_us: float = float("inf")
    min_us: float = float("inf")
    p25_us: float = float("inf")
    p75_us: float = float("inf")
    mean_us: float = float("inf")
    compile_us: float = float("inf")
    error: Optional[str] = None


@dataclass
class Pending:
    """A cell not yet compiled/run, plus the context it needs (kept off the serialized :class:`Cell`)."""
    cell: Cell
    compile_key: Optional[Tuple] = None  # (exe, flags-tuple, src) -> dedup identical compiles
    atol: float = 0.0
    symbol: str = ""
    order: List[str] = field(default_factory=list)
    argtypes: list = field(default_factory=list)
    opt_ctx_key: Tuple = ()  # (opt_mode, nest_idx, lang) -> the shared per-nest boundary/oracle/sizes


def compiler_for(lang: str, tc: Toolchain, fortran_by_family: Dict[str, str]) -> Optional[str]:
    """The compiler executable for a (language, family): C -> the C compiler, C++ -> the C++ frontend,
    Fortran -> the family's Fortran compiler (gfortran/flang/nvfortran/ifx). ``None`` if absent."""
    if lang == "c":
        return tc.cc
    if lang == "c++":
        return tc.cxx
    return fortran_by_family.get(tc.name)


def emit_lang_sources(prep,
                      boundary,
                      workdir: Path,
                      languages: List[str],
                      validate_sizes: Dict[str, int],
                      name: str,
                      parallel: bool = False) -> Dict[str, Tuple[Path, List[str], list]]:
    """Emit the numpyto sources for the requested languages and parse each signature.

    Returns ``{lang: (src_path, arg_order, argtypes)}``. ``name`` is the PER-NEST base name
    (``<key>_n<idx>``); the emitted symbol is ``<name>_fp64`` so every nest of a kernel binds a distinct
    entry point. C and C++ share one emitted ``.c`` (numpyto has no C++ target); the C++ lane compiles a
    tiny generated wrapper ``.cpp`` that ``#include``s the ``.c`` inside ``extern "C" {}`` -- without it the
    C++ frontend name-mangles ``<name>_fp64`` and ctypes cannot bind it. A language whose emit/parse fails
    is omitted (its cells are skipped, not compiled).

    ``parallel=True`` requests the OpenMP variant (numpyto ``c_omp`` / ``fortran_omp`` -- a drop-in
    ``<base>_omp.{c,f90}`` with the SAME symbol/signature, ``#pragma omp parallel for``). numpyto RAISES
    (nonzero exit -> :class:`RuntimeError`) for a nest with no sound parallel form, so the caller emits into
    a distinct ``omp`` subdir and treats a raise as "no OpenMP lane for this nest"."""
    symbol = f"{name}_fp64"
    names = list(boundary.standalone_sdfg.arrays) + list(validate_sizes)
    c_target = "c_omp" if parallel else "c"
    f_target = "fortran_omp" if parallel else "fortran"
    out: Dict[str, Tuple[Path, List[str], list]] = {}
    want_c = any(lg in ("c", "c++") for lg in languages)
    c_info: Optional[Tuple[Path, List[str], list]] = None
    if want_c:
        src = next(s for s in emit_sources(prep, workdir, target=c_target)
                   if s.suffix == ".c" and "pluto" not in s.name)
        order = signature_order(src.read_text(), symbol, "c")
        c_info = (src, order, c_argtypes(order, boundary))
    for lang in languages:
        if lang == "c":
            if c_info is not None:
                out[lang] = c_info
        elif lang == "c++":
            if c_info is not None:
                c_src, order, argtypes = c_info
                wrapper = workdir / f"{name}_cxxwrap.cpp"
                wrapper.write_text('extern "C" {\n#include "%s"\n}\n' % c_src.resolve())
                out[lang] = (wrapper, order, argtypes)
        else:  # fortran
            fsrc = next(s for s in emit_sources(prep, workdir, target=f_target)
                        if s.suffix == ".f90" and "pluto" not in s.name)
            forder = fortran_unmunge(signature_order(fsrc.read_text(), symbol, "fortran"), names)
            out[lang] = (fsrc, forder, c_argtypes(forder, boundary))
    return out


# --- per-kernel run ----------------------------------------------------------------------------------
def build_opt_context(kernel, opt_mode: str, strategy: str, profile_preset: str, languages: List[str],
                      workdir: Path) -> List[Dict]:
    """Everything a lane-3 sweep needs for ONE opt-mode: EVERY extracted nest, each with its own sizes,
    numpy oracle, and per-language emitted sources. Returns a LIST of per-nest context dicts (one per
    compute nest the strategy found). Raises on a skip condition (no nest, emit failure).

    MUTATION HAZARD: :func:`extract_nest_to_sdfg` mutates the parent SDFG in place, so a ``(parent, node)``
    ref captured up front goes stale after the first extraction. Rebuild a fresh SDFG + strategy refs per
    nest (both are deterministic, so ``refs_i`` aligns positionally with the initial ``refs``) and extract
    the idx-th nest from its own untouched copy."""
    sdfg = tsvc.build_sdfg(kernel, opt_mode=opt_mode)
    refs = get_strategy(strategy)(sdfg)
    if not refs:
        raise RuntimeError(empty_strategy_reason(sdfg))
    ctxs: List[Dict] = []
    for idx in range(len(refs)):
        sdfg_i = tsvc.build_sdfg(kernel, opt_mode=opt_mode)
        refs_i = get_strategy(strategy)(sdfg_i)
        parent, node = refs_i[idx]
        parallel = is_parallel_nest(node)  # read the schedule BEFORE extraction mutates the graph
        name = f"{kernel.key}_n{idx}"
        boundary = extract_nest_to_sdfg(parent, node, name=name)
        nest_dir = workdir / f"n{idx}"
        time_sizes = tsvc.sample_sizes(kernel, boundary, preset=profile_preset)
        validate_sizes = tsvc.sample_sizes(kernel, boundary, preset=validate_cap(profile_preset))
        prep = prepare(boundary, name, nest_dir, sizes=validate_sizes)
        oracle = run_oracle(prep, boundary, make_inputs(boundary, validate_sizes, seed=0), validate_sizes)
        lang_src = emit_lang_sources(prep, boundary, nest_dir, languages, validate_sizes, name)
        # A DaCe-parallel nest ALSO gets the OpenMP ``omp-emit`` sources (numpyto c_omp/fortran_omp). numpyto
        # refuses a nest it cannot soundly parallelize (colliding scatter) -> RuntimeError; that nest simply
        # runs no omp-emit lane (never fatal). Emitted into an ``omp`` subdir so its ``_omp.c`` never shadows
        # the sequential ``.c`` glob.
        omp_src: Dict[str, Tuple[Path, List[str], list]] = {}
        if parallel:
            try:
                omp_src = emit_lang_sources(prep,
                                            boundary,
                                            nest_dir / "omp",
                                            languages,
                                            validate_sizes,
                                            name,
                                            parallel=True)
            except (RuntimeError, StopIteration) as e:
                print(f"[tsvc-full] {name}: no OpenMP variant ({type(e).__name__}: {str(e)[:120]})")
        # Precompute the veclib gate per lang ONCE, here where the sources exist (enumeration stays I/O-free).
        # C++ compiles the SAME C source (its own path is just an ``extern "C"`` #include wrapper with no math
        # calls of its own), so it inherits C's math content rather than scanning the wrapper.
        has_math: Dict[str, bool] = {
            lang: source_has_math(src.read_text())
            for lang, (src, _o, _a) in lang_src.items() if lang != "c++"
        }
        if "c++" in lang_src:
            has_math["c++"] = has_math.get("c", False)
        ctxs.append({
            "nest_idx": idx,
            "name": name,
            "symbol": f"{name}_fp64",
            "boundary": boundary,
            "time_sizes": time_sizes,
            # allocated ONCE per nest; every timing cell reuses it via COW-fork isolation (see nest_timing_work).
            "time_inputs": make_inputs(boundary, time_sizes, seed=0),
            "validate_sizes": validate_sizes,
            "oracle": oracle,
            "lang_src": lang_src,
            "has_math": has_math,
            "omp_src": omp_src,
            "parallel": parallel,
        })
    return ctxs


def cost_models_for(parallel: str, cost_models: List[str], matrix_preset: str) -> List[str]:
    """The vectorizer cost models to sweep for a given ``parallel`` mode under a ``matrix_preset``.

    The ``cost`` axis (``default`` / ``cheap`` / ``no-vec``) is a **single-core vectorization** question:
    scalar floor vs the compiler's vectorizer. Once a lane is threaded (``auto-par`` / ``omp-emit``) the
    story is threading, not the scalar/vector floor, so the extra cost points there are low value and
    multiply the matrix. Presets:

      * ``full`` -- sweep every requested cost model on every parallel mode (the exhaustive matrix).
      * ``lean`` -- sweep the full cost axis ONLY on the ``sequential`` (single-core) lane -- exactly the
        data :mod:`perf.plot_vectorization` reads -- and collapse ``auto-par`` / ``omp-emit`` to the
        compiler default. This roughly halves the timing cells while losing NO single-core vec data.

    An unknown preset behaves like ``full`` (fail open -- never silently drop cells)."""
    if matrix_preset == "lean" and parallel != "sequential":
        return ["default"] if "default" in cost_models else list(cost_models[:1])
    return list(cost_models)


#: libm transcendentals a vector-math library (``-fveclib``) can accelerate in the emitted source. A nest
#: whose source calls none of these gets no veclib axis (the library is inert for it) -- the plan's prune.
_MATH_CALLS = ("sin", "cos", "exp", "log", "pow", "tan", "asin", "acos", "atan", "atan2", "hypot", "sinh", "cosh",
               "tanh", "cbrt", "expm1", "log1p")


def source_has_math(text: str) -> bool:
    """True when the emitted source calls a libm transcendental a veclib could vectorize (a plain ``name(``
    scan of the numpyto output). Cheap gate so a math-free nest skips the veclib axis entirely."""
    return any(f"{fn}(" in text for fn in _MATH_CALLS)


def veclibs_for(has_math: bool, veclibs: Sequence[str], compiler: Optional[str]) -> Tuple[str, ...]:
    """The veclib values to actually sweep for one (nest-lang, compiler): always ``none``; a candidate is
    added only when the nest HAS a libm call (``has_math``, precomputed per-lang in the ctx by
    :func:`source_has_math` -- see :func:`build_opt_context`) AND the veclib is compatible with the compiler
    (an incompatible one is dropped here, never turned into an error cell). Taking the precomputed flag (not
    the source text) keeps :func:`enumerate_cells` free of source I/O and lets the C++ lane inherit the C
    source's math content instead of scanning its ``#include``-only wrapper."""
    out = ["none"]
    if has_math:
        for vl in veclibs:
            if vl == "none":
                continue
            resolved, _ = flags.veclib_flags(compiler, vl)
            if resolved:  # compatible and contributes at least one flag
                out.append(vl)
    return tuple(out)


def enumerate_cells(opt_ctx: Dict, toolchains: List[Toolchain], fortran_by_family: Dict[str, str], axes: Dict,
                    nthreads: int, cxx_std: str, workdir: Path) -> Tuple[List[Pending], Dict[Tuple, Dict]]:
    """Build every lane-3 :class:`Pending` cell for one opt-mode + the deduped compile jobs.

    Compile-job key is ``(exe, flags-tuple, src)`` so two axis points with identical flags compile once.
    An unsupported combo (clang auto-par) becomes a finished error :class:`Cell` with no compile job."""
    pendings: List[Pending] = []
    jobs: Dict[Tuple, Dict] = {}
    opt_mode = axes["opt_mode"]
    nest_idx = opt_ctx.get("nest_idx", 0)  # dict.get (not getattr): a synthetic ctx may omit it -> nest 0
    # Dedup identical TIMING cells: two axis points whose (exe, flags, src) collapse to the same compile
    # (e.g. clang/icx/nvc `cheap` == `default`, nvc `assume-finite` == `contract-fma`) are the SAME
    # measurement -- previously they were timed once PER label (same .so, run N times) and produced N
    # duplicate rows. Keep the FIRST (canonical `default`-cost label wins, iteration order) and skip the
    # rest, so the family that has no such knob contributes one cell, not three. Gate cells are exempt
    # (different FP flags; never collide with a timing cell) so the bit-exact gate always runs.
    seen_timing: set = set()

    def add(cell: Cell, exe: str, cflags: List[str], src: Path, atol: float, symbol, order, argtypes, ctx_key):
        key = (exe, tuple(cflags), str(src))
        if cell.role == "timing":
            if key in seen_timing:
                return
            seen_timing.add(key)
        so = workdir / (f"{opt_mode}_n{cell.nest}_{cell.language.replace('+', 'x')}_{cell.compiler}_{cell.parallel}_"
                        f"{cell.cost_model}_{cell.fp_mode}_{cell.veclib}_{len(jobs)}.so")
        jobs.setdefault(key, {"exe": exe, "flags": cflags, "src": src, "so": so})
        pendings.append(
            Pending(cell=cell,
                    compile_key=key,
                    atol=atol,
                    symbol=symbol,
                    order=order,
                    argtypes=argtypes,
                    opt_ctx_key=ctx_key))

    for lang, (src, order, argtypes) in opt_ctx["lang_src"].items():
        symbol = opt_ctx["symbol"]
        ctx_key = (opt_mode, nest_idx, lang)
        # veclib gate is a PRECOMPUTED per-lang fact (build_opt_context read the source once); enumeration
        # itself does no source I/O, so it runs on synthetic ctxs with dummy paths. Absent -> math-free.
        has_math = opt_ctx.get("has_math", {}).get(lang, False)
        for tc in toolchains:
            exe = compiler_for(lang, tc, fortran_by_family)
            fam = tc.fp_family
            if exe is None:
                pendings.append(
                    Pending(cell=Cell(opt_mode,
                                      lang,
                                      tc.name,
                                      "sequential",
                                      "default",
                                      "-",
                                      "timing",
                                      nest=nest_idx,
                                      error=f"no {lang} compiler for family {tc.name}")))
                continue
            # timing cells: parallel x cost x reduced-FP
            for parallel in axes["parallelism"]:
                # omp-emit uses OUR ``#pragma omp`` source (numpyto c_omp/fortran_omp); it exists only for
                # a nest the DaCe schedule marked parallel AND numpyto could soundly parallelize.
                psrc = src
                if parallel == "omp-emit":
                    omp = opt_ctx.get("omp_src", {}).get(lang)
                    if omp is None:
                        continue
                    psrc = omp[0]
                for cost in cost_models_for(parallel, axes["cost_models"], axes.get("matrix_preset", "full")):
                    for fp in axes["fp_modes"]:
                        for veclib in veclibs_for(has_math, axes.get("veclibs", ("none", )), exe):
                            cflags, reason = flags.lane_flags(fam,
                                                              fp,
                                                              cost,
                                                              parallel,
                                                              lang,
                                                              nthreads,
                                                              cxx_std,
                                                              compiler=exe,
                                                              veclib=veclib)
                            cell = Cell(opt_mode,
                                        lang,
                                        tc.name,
                                        parallel,
                                        cost,
                                        fp,
                                        "timing",
                                        nest=nest_idx,
                                        veclib=veclib)
                            if cflags is None:
                                if veclib == "none":  # base combo unsupported (e.g. clang auto-par): record once
                                    cell.error = f"unsupported: {reason}"
                                    pendings.append(Pending(cell=cell))
                                continue
                            add(cell, exe, cflags, psrc, flags.REDUCED_FP_ATOL[fp], symbol, order, argtypes, ctx_key)
            # correctness GATE cell: strict-ieee, sequential, default cost (bit-exact vs the oracle)
            if axes["gate"]:
                cflags, _ = flags.lane_flags(fam,
                                             "strict-ieee",
                                             "default",
                                             "sequential",
                                             lang,
                                             nthreads,
                                             cxx_std,
                                             compiler=exe)
                gate = Cell(opt_mode, lang, tc.name, "sequential", "default", "strict-ieee", "gate", nest=nest_idx)
                add(gate, exe, cflags, src, flags.FP_ATOL["strict-ieee"], symbol, order, argtypes, ctx_key)
    return pendings, jobs


def run_kernel(kernel, toolchains: List[Toolchain], fortran_by_family: Dict[str, str], strategy: str, axes: Dict,
               reps: int, profile_preset: str, nthreads: int, cxx_std: str, compile_jobs: int, workdir: Path) -> Dict:
    """Run all three lanes + the full lane-3 sweep for one kernel; return the JSON-able result."""
    result = {
        "key": kernel.key,
        "corpus": kernel.corpus,
        "regime": kernel.regime,
        "host": socket.gethostname(),
        "profile_preset": profile_preset
    }

    # per-opt-mode context: EVERY extracted nest, as a LIST of per-nest ctxs. A failing opt-mode is noted,
    # not fatal.
    contexts: Dict[str, List[Dict]] = {}
    opt_notes: Dict[str, str] = {}
    for opt_mode in axes["opt_modes"]:
        try:
            contexts[opt_mode] = build_opt_context(kernel, opt_mode, strategy, profile_preset, axes["languages"],
                                                   workdir / opt_mode)
        except Exception as e:
            opt_notes[opt_mode] = f"{type(e).__name__}: {str(e)[:160]}"
    if not contexts:
        return {**result, "skipped": "; ".join(f"{m}: {n}" for m, n in opt_notes.items()) or "no opt-mode built"}

    base_key = "simplify-parallel" if "simplify-parallel" in contexts else next(iter(contexts))
    base_ctxs = contexts[base_key]  # the base opt-mode's per-nest ctxs (what lanes 1/2 measure against)
    result["baseline_opt"] = base_key  # which opt-mode's nests lanes 1/2 used (may not be 'simplify-parallel')
    # lane 1 (native): ONE whole-kernel measurement stamped nest=-1 -- it is compared against the SUM over
    # nests, not any single nest, so it borrows the base opt-mode's FIRST nest only for buffer sizing / the
    # oracle cross-check. lane 2 (DaCe-cpp): runs PER NEST (each cell carries its nest idx); the speedup
    # denominator is the SUM over nests. A single C++ toolchain (first with cxx) drives both.
    cxx_tc = next((tc for tc in toolchains if tc.cxx is not None), toolchains[0] if toolchains else None)
    if cxx_tc is not None:
        nat0 = base_ctxs[0]
        native = measure_native_lane(cxx_tc.cxx, cxx_tc.fp_family, kernel, nat0["boundary"], nat0["validate_sizes"],
                                     nat0["time_inputs"], nat0["time_sizes"], nat0["oracle"], reps, cxx_std, workdir)
        if native is not None:
            native["nest"] = -1  # whole-kernel sentinel: not a single-nest measurement
        result["native"] = native
        # The DaCe lane fans out over the codegen-implementation axis: 'legacy' (the speedup denominator)
        # plus 'experimental' when this DaCe build has it. Each cell carries its nest idx + codegen_impl;
        # the reporter sums only the legacy cells for the denominator and geomeans experimental against it.
        dace_cpp_cells: List[Dict] = []
        for nc in base_ctxs:
            for impl in codegen_impls_available():
                dcell = measure_dace_cpp_lane(cxx_tc,
                                              nc["boundary"],
                                              nc["validate_sizes"],
                                              nc["time_inputs"],
                                              nc["time_sizes"],
                                              nc["oracle"],
                                              reps,
                                              cxx_std,
                                              workdir / "dace_cpp" / f"n{nc['nest_idx']}_{impl}",
                                              codegen_impl=impl)
                dcell["nest"] = nc["nest_idx"]
                dace_cpp_cells.append(dcell)
        result["dace_cpp"] = dace_cpp_cells
        # Optional vectorized DaCe lane: the multi-dim tile-op vectorizer, coordinate-descent per nest for
        # the fastest validating VectorizeConfig. Opt-in (--vectorize) since it compiles ~15-30 cells/nest.
        if axes.get("vectorize"):
            dace_vec_cells: List[Dict] = []
            for nc in base_ctxs:
                vcell = measure_dace_vectorized_lane(cxx_tc,
                                                     nc["boundary"],
                                                     nc["validate_sizes"],
                                                     nc["time_inputs"],
                                                     nc["time_sizes"],
                                                     nc["oracle"],
                                                     reps,
                                                     cxx_std,
                                                     workdir / "dace_vec" / f"n{nc['nest_idx']}",
                                                     codegen_impl=default_codegen_impl())
                vcell["nest"] = nc["nest_idx"]
                dace_vec_cells.append(vcell)
            result["dace_cpp_vec"] = dace_vec_cells

    # lane 3: enumerate + dedup-compile + validate + time. Each opt-mode holds a LIST of per-nest ctxs; every
    # nest is enumerated independently (its own emitted sources + boundary/oracle/sizes + symbol).
    all_pending: List[Pending] = []
    all_jobs: Dict[Tuple, Dict] = {}
    ctx_by_key: Dict[Tuple, Dict] = {}
    for opt_mode, nest_ctxs in contexts.items():
        axes_om = {**axes, "opt_mode": opt_mode}
        for nc in nest_ctxs:
            pend, jobs = enumerate_cells(nc, toolchains, fortran_by_family, axes_om, nthreads, cxx_std,
                                         workdir / opt_mode)
            all_pending.extend(pend)
            all_jobs.update(jobs)
            for lang in nc["lang_src"]:
                ctx_by_key[(opt_mode, nc["nest_idx"], lang)] = nc

    # (a) parallel compile of the unique flag sets (compilation is the bottleneck; the timed runs are not).
    def do_compile(job: Dict) -> Tuple[bool, float, Optional[str]]:
        return run_compile([job["exe"], *job["flags"], str(job["src"]), "-o", str(job["so"])])

    if all_jobs:
        with ThreadPoolExecutor(max_workers=max(1, compile_jobs)) as pool:
            for key, res in zip(all_jobs, pool.map(do_compile, all_jobs.values())):
                all_jobs[key]["result"] = res

    # (b) validate every compiled cell (fast, small preset); (c) time only the ones that validate.
    cells_out: List[Dict] = []
    for p in all_pending:
        cell = p.cell
        if p.compile_key is None:  # unsupported combo or missing compiler -> already an error cell
            cells_out.append(asdict(cell))
            continue
        job = all_jobs[p.compile_key]
        ok, compile_us, cerr = job["result"]
        cell.compile_us = compile_us
        if not ok:
            cell.error = cerr
            cells_out.append(asdict(cell))
            continue
        ctx = ctx_by_key[p.opt_ctx_key]
        so = job["so"]
        vres = run_isolated(lambda: nest_validate_work(so, p.symbol, p.order, p.argtypes, ctx["boundary"], ctx[
            "validate_sizes"], ctx["oracle"], p.atol))
        if "error" in vres:
            cell.error = vres["error"]
            cells_out.append(asdict(cell))
            continue
        cell.ok, cell.maxdiff = vres["ok"], vres["maxdiff"]
        if cell.role == "timing" and cell.ok:  # only VALIDATED timing cells are (expensively) timed
            tres = run_isolated(lambda: nest_timing_work(so, p.symbol, p.order, p.argtypes, ctx["time_inputs"], ctx[
                "time_sizes"], reps),
                                timeout=RUN_TIMEOUT_S)
            if "error" in tres:
                cell.error = f"timing: {tres['error']}"
            else:
                cell.median_us, cell.min_us = tres["median_us"], tres["min_us"]
                cell.p25_us, cell.p75_us, cell.mean_us = tres["p25_us"], tres["p75_us"], tres["mean_us"]
        cells_out.append(asdict(cell))

    result["cells"] = cells_out
    # per-nest sizes of the base opt-mode, keyed by nest idx (as a str for JSON object keys).
    result["sizes"] = {
        str(nc["nest_idx"]): {
            "validate": {
                k: int(v)
                for k, v in nc["validate_sizes"].items()
            },
            "time": {
                k: int(v)
                for k, v in nc["time_sizes"].items()
            },
        }
        for nc in base_ctxs
    }
    if opt_notes:
        result["opt_notes"] = opt_notes
    return result


# --- tables ------------------------------------------------------------------------------------------
def kernel_winner(cells: List[Dict],
                  opt_mode: str,
                  lang: str,
                  compiler: str,
                  nest: Optional[int] = None) -> Optional[Dict]:
    """Fastest VALIDATED timing cell (min median) for one (opt, lang, compiler[, nest]) group, or None.

    ``nest`` restricts the search to one extracted nest; ``None`` (the default) matches any nest. A kernel's
    offloaded time is the SUM over nests of the per-nest winner, so :func:`render_tables` passes an explicit
    nest per call and sums the results. ``c.get("nest", 0)`` (dict.get, not getattr) tolerates a legacy or
    synthetic cell that predates the ``nest`` field -> treated as the single nest 0."""
    grp = [
        c for c in cells if c["role"] == "timing" and c["ok"] and c["opt_mode"] == opt_mode and c["language"] == lang
        and c["compiler"] == compiler and (nest is None or c.get("nest", 0) == nest) and finite(c["median_us"])
    ]
    return min(grp, key=lambda c: c["median_us"]) if grp else None


def render_tables(out: Path) -> str:
    files = sorted(p for p in out.glob("*.json") if p.name != "tables.md")
    kernels = [json.loads(p.read_text()) for p in files]
    done = [k for k in kernels if "cells" in k]
    skipped = [k for k in kernels if "skipped" in k]

    lines = [
        "# TSVC full-matrix job (native cpp / DaCe-cpp baseline / nest-forge sweep)",
        "",
        f"{len(done)} kernels measured, {len(skipped)} skipped. Timing is MEDIAN of N reps at the profiling "
        "preset; the speedup baseline is the DaCe-cpp (strict-ieee) lane.",
        "",
    ]

    # strict-ieee gate: the gate cell must VALIDATE within the strict tolerance vs the oracle. strict-ieee is
    # NOT atol-0 (reductions/transcendentals differ ~1e-15 from numpy pairwise sum), so the gate agrees with
    # the cell's own ``ok`` verdict (validated at FP_ATOL["strict-ieee"]) rather than demanding maxdiff == 0.
    gate_fail = []
    for k in done:
        for c in k["cells"]:
            if c["role"] == "gate" and c["error"] is None and not c["ok"]:
                gate_fail.append((k["key"], k["corpus"], c["opt_mode"], c["language"], c["compiler"], c["maxdiff"]))
    gate_status = "PASS" if not gate_fail else f"{len(gate_fail)} FAILURES"
    lines += [
        f"**strict-ieee gate:** {gate_status} "
        "(validates within the strict tolerance vs the numpy oracle).", ""
    ]
    if gate_fail:
        lines += ["| kernel | corpus | opt | lang | compiler | maxdiff |", "|" + "---|" * 6]
        for a, b, c, d, e, f in gate_fail:
            mds = "—" if f is None else f"{f:g}"
            lines.append(f"| {a} | {b} | {c} | {d} | {e} | {mds} |")
        lines.append("")

    lines += [
        "## per (kernel, opt-mode, language, compiler): best offloaded nests vs the DaCe-cpp baseline", "",
        "A kernel may split into several loop-nests; the offloaded time is the SUM over nests of each nest's "
        "own best (min validated median) cell, and the DaCe-cpp baseline is the SUM over nests of the "
        "per-nest DaCe-cpp lane -- an apples-to-apples denominator. The offloaded sum is blank unless EVERY "
        "nest has a validated cell for that (lang, compiler). A `†` marks a baseline where some nest did not "
        "bit-match the oracle (loop-carried-state kernel -- DaCe vs numpyto boundary-semantics divergence). "
        "``native`` is the single whole-kernel measurement (not per-nest).", "",
        "| kernel | corpus | opt | lang | compiler | native (us) | DaCe-cpp (us) | best nests (us) | best config "
        "| maxdiff | nest/DaCe |", "|" + "---|" * 11
    ]
    speedups: List[float] = []
    codegen_speedups: List[float] = []  # per-kernel legacy/experimental DaCe-codegen ratios (the codegen axis)
    vec_speedups: List[float] = []  # per-kernel plain-DaCe / vectorized-DaCe ratios (the vectorization axis)
    novalidate = 0
    for k in sorted(done, key=lambda x: (x["corpus"], x["key"])):
        # dace_cpp is a LIST of per-nest cells (older JSON: single-dict / empty tolerated), now also fanned
        # over the codegen-impl axis. The DENOMINATOR is the legacy cells only (a cell with no codegen_impl
        # predates the axis -> treated as legacy), summed over nests, defined only when every nest timed.
        dcpp_raw = k.get("dace_cpp")
        dcpp_list = dcpp_raw if isinstance(dcpp_raw, list) else ([dcpp_raw] if dcpp_raw else [])
        legacy_cells = [d for d in dcpp_list if d.get("codegen_impl", "legacy") == "legacy"]
        exp_cells = [d for d in dcpp_list if d.get("codegen_impl") == "experimental"]
        # Denominator = the NEW (experimental) codegen when the run produced it (nest-forge's default),
        # else legacy. A cell with no codegen_impl predates the axis -> counts as legacy.
        primary_cells = exp_cells if exp_cells else legacy_cells
        dace_meds = [d.get("median_us") for d in primary_cells]
        dace_us = sum(dace_meds) if (dace_meds and all(finite(m) for m in dace_meds)) else None
        dace_flag = "" if (dace_us is None or all(d.get("ok") for d in primary_cells)) else "†"
        if dace_flag:
            novalidate += 1
        # codegen axis: legacy total / experimental total for this kernel, when BOTH impls fully timed.
        legacy_meds = [d.get("median_us") for d in legacy_cells]
        exp_meds = [d.get("median_us") for d in exp_cells]
        legacy_us = sum(legacy_meds) if (legacy_meds and all(finite(m) for m in legacy_meds)) else None
        exp_us = sum(exp_meds) if (exp_meds and all(finite(m) for m in exp_meds)) else None
        if legacy_us and exp_us:
            codegen_speedups.append(legacy_us / exp_us)
        # vectorization axis: plain-DaCe (denominator) total / vectorized-DaCe total, when both fully timed.
        vec_raw = k.get("dace_cpp_vec")
        vec_list = vec_raw if isinstance(vec_raw, list) else ([vec_raw] if vec_raw else [])
        vec_meds = [d.get("median_us") for d in vec_list if d.get("ok")]
        vec_us = sum(vec_meds) if (vec_list and len(vec_meds) == len(vec_list) and all(finite(m)
                                                                                       for m in vec_meds)) else None
        if dace_us is not None and vec_us:
            vec_speedups.append(dace_us / vec_us)
        nat = k.get("native") or {}
        nat_med = nat.get("median_us")
        nat_us = nat_med if (nat.get("ok") and finite(nat_med)) else None
        groups = sorted({(c["opt_mode"], c["language"], c["compiler"]) for c in k["cells"] if c["role"] == "timing"})
        for opt_mode, lang, comp in groups:
            nests = sorted(
                {c.get("nest", 0)
                 for c in k["cells"] if c["role"] == "timing" and c["opt_mode"] == opt_mode})
            wins = [kernel_winner(k["cells"], opt_mode, lang, comp, n) for n in nests]
            if wins and all(w is not None for w in wins):
                win_us = sum(w["median_us"] for w in wins)
                cfg = "+".join(f"{w['parallel']}/{w['cost_model']}/{w['fp_mode']}"
                               f"{'/' + w['veclib'] if w.get('veclib', 'none') != 'none' else ''}" for w in wins)
                md = f"{max(w['maxdiff'] for w in wins):g}"
            else:
                win_us, cfg, md = None, "—", "—"
            sp = (dace_us / win_us) if (dace_us is not None and win_us) else None
            if sp is not None and math.isfinite(sp):
                speedups.append(sp)
            lines.append(f"| {k['key']} | {k['corpus']} | {opt_mode} | {lang} | {comp} | {fmt_us(nat_us)} "
                         f"| {fmt_us(dace_us)}{dace_flag} | {fmt_us(win_us)} | {cfg} | {md} | "
                         f"{'—' if sp is None else f'{sp:.2f}x'} |")

    if speedups:
        geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        lines += [
            "", f"**Geomean best-nest / DaCe-cpp speedup:** {geo:.3f}x over {len(speedups)} "
            "(kernel x opt x lang x compiler) rows where both timed."
        ]
    if codegen_speedups:
        cgeo = math.exp(sum(math.log(s) for s in codegen_speedups) / len(codegen_speedups))
        lines += [
            "", f"**Geomean DaCe legacy / experimental codegen speedup:** {cgeo:.3f}x over "
            f"{len(codegen_speedups)} kernels where both codegen impls timed "
            "(>1 = experimental faster; the codegen-implementation axis)."
        ]
    if vec_speedups:
        vgeo = math.exp(sum(math.log(s) for s in vec_speedups) / len(vec_speedups))
        lines += [
            "", f"**Geomean DaCe plain / tile-op-vectorized speedup:** {vgeo:.3f}x over "
            f"{len(vec_speedups)} kernels where the vectorized lane's best config timed "
            "(>1 = the multi-dim tile-op vectorizer faster; the vectorization axis)."
        ]
    if novalidate:
        lines += [
            "", f"`†` {novalidate} kernels' DaCe-cpp baseline did not bit-match the numpy oracle (DaCe vs "
            "numpyto boundary semantics for loop-carried state); their timing baseline is still "
            "representative (identical iteration space)."
        ]

    # unsupported / error summary (auto-par on clang, missing Fortran compilers, ...).
    reasons: Dict[str, int] = {}
    for k in done:
        for c in k["cells"]:
            if c["error"] and ("unsupported" in c["error"] or "no " in c["error"][:4]):
                reasons[c["error"][:80]] = reasons.get(c["error"][:80], 0) + 1
    if reasons:
        lines += ["", "## unsupported / skipped cells (recorded, not silently dropped)", ""]
        lines += [f"- {n}x — {r}" for r, n in sorted(reasons.items(), key=lambda kv: -kv[1])]

    # real compile / validation failures -- NOT the expected unsupported/missing-compiler cells above.
    failures: Dict[str, int] = {}
    for k in done:
        for c in k["cells"]:
            err = c["error"]
            if err and not ("unsupported" in err or "no " in err[:4]):
                failures[err[:60]] = failures.get(err[:60], 0) + 1
    if failures:
        lines += ["", "## compile / validation failures (recorded)", ""]
        lines += [f"- {n}x  {r}" for r, n in sorted(failures.items(), key=lambda kv: -kv[1])[:12]]
    if skipped:
        lines += ["", "## skipped kernels", ""]
        lines += [
            f"- `{k['key']}` ({k.get('corpus', '?')}) — {k['skipped']}" for k in sorted(skipped, key=lambda x: x["key"])
        ]
    report = "\n".join(lines) + "\n"
    (out / "tables.md").write_text(report)
    return report


# --- CLI ---------------------------------------------------------------------------------------------
def resolve_veclibs(spec: List[str], compiler: str = "gcc") -> Tuple[str, ...]:
    """Resolve the ``--veclibs`` spec to the axis values. ``'auto'`` -> ``('none',)`` plus the per-device
    characterized winner (``device_profile.rank_veclibs`` -- compiles tiny probes once); ``'none'`` ->
    ``('none',)``; an explicit list is used verbatim with ``none`` ensured present. A box with no installed
    veclib resolves to ``('none',)``, so the axis silently disappears rather than erroring."""
    if list(spec) == ["auto"]:
        from nestforge.device_profile import rank_veclibs
        ranked = [p for p in rank_veclibs(compiler) if p.ok]
        return ("none", ranked[0].name) if ranked else ("none", )
    out = list(dict.fromkeys(spec))
    if "none" not in out:
        out = ["none"] + out
    return tuple(out)


def resolved_axes(args) -> Dict:
    parallelism = list(flags.PARALLEL_MODES) if args.parallelism == "both" else [args.parallelism]
    return {
        "opt_modes": args.opt_modes,
        "languages": args.languages,
        "parallelism": parallelism,
        "cost_models": args.cost_models,
        "fp_modes": args.fp_modes,
        "gate": not args.no_gate,
        "matrix_preset": args.matrix_preset,
        "veclibs": resolve_veclibs(args.veclibs),
        "vectorize": args.vectorize,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="TSVC full-matrix job (native / DaCe-cpp / nest-forge sweep)")
    ap.add_argument("--corpora", nargs="*", default=["tsvc2", "tsvc2_5"], choices=["tsvc2", "tsvc2_5"])
    ap.add_argument("--languages", nargs="*", default=["c", "c++", "fortran"], choices=list(_EMIT))
    ap.add_argument("--opt-modes",
                    nargs="*",
                    default=list(tsvc.OPT_MODES),
                    choices=list(tsvc.OPT_MODES),
                    dest="opt_modes")
    ap.add_argument("--parallelism",
                    default="both",
                    choices=["sequential", "auto-par", "omp-emit", "both"],
                    help="'both' = all of sequential/auto-par/omp-emit; omp-emit uses OUR "
                    "``#pragma omp`` source (only for DaCe-parallel nests).")
    ap.add_argument("--cost-models",
                    nargs="*",
                    default=list(flags.COST_MODELS),
                    choices=list(flags.COST_MODELS),
                    dest="cost_models")
    ap.add_argument("--fp-modes",
                    nargs="*",
                    default=list(flags.REDUCED_FP_MODES),
                    choices=list(flags.REDUCED_FP_MODES),
                    dest="fp_modes")
    ap.add_argument("--matrix-preset",
                    default="full",
                    choices=["full", "lean"],
                    dest="matrix_preset",
                    help="'full' = sweep every cost model on every parallel mode; 'lean' = sweep the cost "
                    "(vectorizer) axis only on the sequential single-core lane and collapse auto-par/"
                    "omp-emit to the compiler default (roughly halves the timing cells, keeps all "
                    "single-core vectorization data). Identical-flag cost cells are always deduped.")
    ap.add_argument("--no-gate", action="store_true", help="skip the strict-ieee bit-exact correctness gate cells")
    ap.add_argument("--veclibs",
                    nargs="*",
                    default=["auto"],
                    help="vector-math library axis: 'auto' (none + the per-device characterized winner), "
                    "'none', or an explicit list from {none, sleef, libmvec, svml}. Only nests whose source "
                    "calls a libm transcendental get the non-none cells.")
    ap.add_argument("--vectorize",
                    action="store_true",
                    help="add the vectorized DaCe lane: coordinate-descent over the multi-dim tile-op "
                    "VectorizeConfig per nest (compiles ~15-30 cells/nest; off by default).")
    ap.add_argument("--profile-preset",
                    default="PROF",
                    choices=["S", "M", "L", "PROF", "XL"],
                    dest="profile_preset",
                    help="timing size (PROF = >L3 memory-bound; XL = the big confirmation run)")
    ap.add_argument("--compilers", default="auto", help="'auto' or a whitespace list (gcc clang nvc++ icx)")
    ap.add_argument("--strategy", default="skip-taskloops")
    ap.add_argument("--reps", type=int, default=11, help="median-of-N timing reps")
    ap.add_argument("--threads",
                    type=int,
                    default=default_threads(),
                    help="thread count baked into auto-par (gcc -ftree-parallelize-loops=N)")
    ap.add_argument("--compile-jobs",
                    type=int,
                    default=max(1, (os.cpu_count() or 8) // 2),
                    dest="compile_jobs",
                    help="bounded pool of concurrent per-cell compiles within this rank")
    ap.add_argument("--cxx-std", default=os.environ.get("DACE_PERF_CXX_STD", flags.CXX_STD), dest="cxx_std")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="perf_results/tsvc_full")
    ap.add_argument("--tables-only", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.tables_only:
        print(render_tables(out))
        return 0

    toolchains = discover_toolchains(args.compilers)
    if not toolchains:
        print("[tsvc-full] no toolchains discovered (checked PATH + spack); nothing to run")
        return 1
    fortran_by_family = lang_compilers(["fortran"], toolchains).get("fortran", {})
    axes = resolved_axes(args)
    print("[tsvc-full] toolchains: " +
          ", ".join(f"{t.name}(cc={Path(t.cc).name},cxx={Path(t.cxx).name if t.cxx else '-'},"
                    f"ftn={Path(fortran_by_family[t.name]).name if t.name in fortran_by_family else '-'})"
                    for t in toolchains))
    print(f"[tsvc-full] axes: opt={axes['opt_modes']} lang={axes['languages']} par={axes['parallelism']} "
          f"cost={axes['cost_models']} fp={axes['fp_modes']} gate={axes['gate']} preset={axes['matrix_preset']} "
          f"profile={args.profile_preset}")

    kernels = [k for corpus in args.corpora for k in tsvc.iter_tsvc_kernels(only=args.only, corpus=corpus)]
    procid, ntasks = rank_and_size()
    mine = my_slice(kernels, procid, ntasks)
    if args.limit:
        mine = mine[:args.limit]
    print(f"[tsvc-full] rank {procid}/{ntasks}: {len(mine)} of {len(kernels)} kernels -> {out}")

    for i, kernel in enumerate(mine):
        workdir = Path(tempfile.mkdtemp(prefix=f"nf_full_{kernel.corpus}_{kernel.key}_"))
        try:
            res = run_kernel(kernel, toolchains, fortran_by_family, args.strategy, {
                **axes, "opt_mode": None
            }, args.reps, args.profile_preset, args.threads, args.cxx_std, args.compile_jobs, workdir)
        except Exception as e:  # pragma: no cover - a kernel must never crash the whole rank
            res = {"key": kernel.key, "corpus": kernel.corpus, "skipped": f"crash: {type(e).__name__}: {str(e)[:160]}"}
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        (out / f"{kernel.corpus}_{kernel.key}.json").write_text(json.dumps(jsonable(res), indent=1))
        print(f"[tsvc-full] ({i + 1}/{len(mine)}) {kernel.corpus}/{kernel.key}: {res.get('skipped', 'ok')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
