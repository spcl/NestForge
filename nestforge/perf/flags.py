"""Shared compile-flag matrix for the arena: **FP-precision level** axis crossed with a **vectorizer
cost-model** axis, per compiler family, for C and Fortran (see ``docs/FP_PRECISION_LEVELS.md``).
``intel`` is split from ``llvm`` because icx/icpx/ifx default to ``-fp-model=fast``, so a bare
``-ffp-contract=off`` would leave reassociation/FTZ on.
"""
from __future__ import annotations

import functools
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from nestforge.build import (CXX_STD, LIBOMP, OpenMPRuntime, compiler_family, driver_lib_path, lib_linkable,
                             linkable_lib_dir)

#: The ONE OpenMP runtime every lane links unless a cell names another. libomp because both gcc and
#: clang can link it (it carries a GOMP_* compat layer), so their node libraries share one thread pool.
DEFAULT_OPENMP_RUNTIME = LIBOMP

#: FP-precision levels, strictest first; the index is the ladder rung.
FP_LEVELS: Tuple[str, ...] = ("strict-ieee", "contract-fma", "assume-finite", "fast-math")

#: Validation tolerance vs the numpy fp64 oracle, which isn't bit-reproducible itself (pairwise np.sum,
#: BLAS dot, non-correctly-rounded libm), so even ``strict-ieee`` isn't atol 0.
FP_ATOL: Dict[str, float] = {
    "strict-ieee": 1e-14,
    "contract-fma": 1e-13,
    "assume-finite": 1e-13,
    "fast-math": 1e-5,
}

#: FP-mode flags per (family, level) -- C spellings; Fortran deltas applied by :func:`fortran_fp_flags`.
_FP: Dict[str, Dict[str, List[str]]] = {
    "gnu": {
        "strict-ieee": ["-ffp-contract=off", "-fexcess-precision=standard"],
        "contract-fma": ["-ffp-contract=fast", "-fexcess-precision=standard"],
        "assume-finite": [
            "-ffp-contract=fast", "-fexcess-precision=standard", "-fno-math-errno", "-fno-trapping-math",
            "-ffinite-math-only", "-fno-signed-zeros"
        ],
        "fast-math": ["-ffast-math", "-mrecip"],
    },
    "llvm": {
        "strict-ieee": ["-ffp-contract=off"],
        "contract-fma": ["-ffp-contract=fast"],
        "assume-finite":
        ["-ffp-contract=fast", "-fno-math-errno", "-fno-trapping-math", "-ffinite-math-only", "-fno-signed-zeros"],
        "fast-math": ["-ffast-math", "-mrecip"],
    },
    "nvidia": {
        # nvc has only whole-model FP knobs: assume-finite collapses to contract-fma (deduped below)
        "strict-ieee": ["-Kieee", "-Mnofma"],
        "contract-fma": ["-Kieee", "-Mfma"],
        "assume-finite": ["-Kieee", "-Mfma"],
        "fast-math": ["-fast", "-Mfma", "-Mfprelaxed=div,sqrt,rsqrt,recip"],
    },
    "intel": {
        # icx/ifx default to -fp-model=fast, so each rung sets an explicit model to reset the baseline.
        "strict-ieee": ["-fp-model=strict"],
        "contract-fma": ["-fp-model=precise"],
        "assume-finite": ["-fp-model=precise", "-ffinite-math-only", "-fno-math-errno"],
        "fast-math": ["-fp-model=fast=2", "-ftz"],
    },
}

#: Native-tuning flag per family (nvc uses -tp=native, not -march=native).
_ARCH: Dict[str, str] = {
    "gnu": "-march=native",
    "llvm": "-march=native",
    "intel": "-march=native",
    "nvidia": "-tp=native"
}

#: Vectorizer cost-model axis. "default" = compiler's own model; "no-vec" = scalar floor; "cheap" =
#: fewer/safer vectorizations (only gcc has a direct knob).
COST_MODELS: Tuple[str, ...] = ("default", "cheap", "no-vec")

#: Vector-math-library axis DOMAIN; the arena sweeps only ``none`` + the accuracy-gated winner.
#: Per-family spelling lives in ``build.VectorMathLib``.
VECLIBS: Tuple[str, ...] = ("none", "sleef", "libmvec", "svml")


def base_flags(family: str) -> List[str]:
    """``-O3`` + native tuning + PIC/shared -- the common prefix every cell shares."""
    return ["-O3", _ARCH.get(family, "-march=native"), "-fPIC", "-shared"]


def fortran_fp_flags(family: str, level: str) -> List[str]:
    """FP-mode flags for a family's **Fortran** frontend. gfortran needs ``-fno-frontend-optimize`` (it
    reassociates at ``-O`` even under ``-ffp-contract=off``); drops flags gfortran/ifx reject
    (``f951: sorry, unimplemented``)."""
    drop = {"-fno-math-errno", "-fexcess-precision=standard"}  # C-family flags the Fortran frontends reject
    flags = [f for f in _FP[family][level] if f not in drop]
    if family == "gnu":
        if level != "fast-math":
            flags.append("-fno-frontend-optimize")
        else:
            flags.append("-fno-protect-parens")
    return flags


def fp_flags(family: str, level: str, lang: str = "c") -> List[str]:
    """FP-mode flags for a (family, level), adjusted for ``lang`` ("c" or "fortran")."""
    return fortran_fp_flags(family, level) if lang == "fortran" else list(_FP[family][level])


def cost_flags(family: str, model: str) -> List[str]:
    """Vectorizer cost-model flags for a family. Empty where the family has no equivalent knob."""
    if model == "no-vec":
        return {
            "gnu": ["-fno-tree-vectorize"],
            "llvm": ["-fno-vectorize", "-fno-slp-vectorize"],
            "intel": ["-fno-vectorize", "-fno-slp-vectorize"],
            "nvidia": ["-Mnovect"],
        }.get(family, [])
    if model == "cheap":
        return {"gnu": ["-fvect-cost-model=cheap"]}.get(family, [])
    return []


def flag_matrix(family: str, lang: str = "c") -> List[Tuple[str, str, List[str]]]:
    """``[(fp_level, cost_model, full_flags), ...]`` for a family/language, deduped so a collapse
    (nvidia assume-finite==contract-fma) produces one compile, not two."""
    matrix: List[Tuple[str, str, List[str]]] = []
    seen = set()
    base = base_flags(family)
    for level in FP_LEVELS:
        for model in COST_MODELS:
            flags = base + fp_flags(family, level, lang) + cost_flags(family, model)
            key = tuple(flags)
            if key in seen:
                continue
            seen.add(key)
            matrix.append((level, model, flags))
    return matrix


# --- the REDUCED FP axis for the full-matrix (tsvc_full) job -----------------------------------------
#: The two FP rungs the full-matrix job TIMES (``strict-ieee`` still runs as its bit-exact gate):
#:  * ``default-fp``    -- compiler's own default at ``-O3``, no flag (vendor-dependent).
#:  * ``no-fast-errno`` -- FMA contraction + ``-fno-math-errno``, no reassociation: "fast but ordered".
REDUCED_FP_MODES: Tuple[str, ...] = ("default-fp", "no-fast-errno")

#: FP-mode flags per (family, reduced-mode) -- C spellings; Fortran deltas via :func:`reduced_fp_flags`.
_REDUCED_FP: Dict[str, Dict[str, List[str]]] = {
    "gnu": {
        "default-fp": [],
        "no-fast-errno": ["-ffp-contract=fast", "-fno-math-errno"],
    },
    "llvm": {
        "default-fp": [],
        "no-fast-errno": ["-ffp-contract=fast", "-fno-math-errno"],
    },
    "nvidia": {
        "default-fp": [],
        "no-fast-errno": ["-Kieee", "-Mfma"],  # nvc has no -fno-math-errno; -Kieee -Mfma == contract-fma
    },
    "intel": {
        "default-fp": [],  # icx default is -fp-model=fast (already non-reproducible)
        "no-fast-errno": ["-fp-model=precise", "-fno-math-errno"],
    },
}

#: Validation tolerance per reduced rung; ``default-fp`` is loose (intel defaults to fast-math) but still
#: catches an O(1) wrong answer, ``no-fast-errno`` is near-bit-exact (only FMA differs).
REDUCED_FP_ATOL: Dict[str, float] = {"default-fp": 1e-6, "no-fast-errno": 1e-12}

#: The parallelization axis of the full-matrix job.
#:  * ``sequential`` -- the sequential emit, no parallel flags.
#:  * ``auto-par``   -- compiler's OWN auto-parallelizer, polyhedral by default; an absent back end is a
#:    recorded skip (:func:`autopar_flags`).
#:  * ``omp-emit``   -- OUR ``#pragma omp parallel for`` source + ``-fopenmp``; every family supports it,
#:    but only for nests DaCe marks parallel AND numpyto can soundly parallelize.
PARALLEL_MODES: Tuple[str, ...] = ("sequential", "auto-par", "omp-emit")


def reduced_fp_flags(family: str, mode: str, lang: str = "c") -> List[str]:
    """FP-mode flags for a REDUCED rung (:data:`REDUCED_FP_MODES`), adjusted for ``lang``: Fortran drops
    ``-fno-math-errno``, and gfortran's ``no-fast-errno`` gains ``-fno-frontend-optimize`` (it
    reassociates at ``-O`` otherwise)."""
    flags = [f for f in _REDUCED_FP[family][mode] if not (lang == "fortran" and f == "-fno-math-errno")]
    if lang == "fortran" and family == "gnu" and mode == "no-fast-errno":
        flags.append("-fno-frontend-optimize")
    return flags


@functools.lru_cache(maxsize=None)
def compiler_accepts(compiler: str, probe_flags: Tuple[str, ...]) -> bool:
    """True if ``compiler`` accepts ``probe_flags`` on a trivial COMPILE-ONLY invocation. Necessary but
    not sufficient -- a back end can accept a flag and do nothing (see :func:`autopar_fires`). ``-c``
    keeps a missing OpenMP runtime from confounding the probe."""
    src = "void f(double *a, int n){for (int i = 0; i < n; i++) a[i] *= 2.0;}\n"
    try:
        proc = subprocess.run([compiler, "-x", "c", "-", "-c", "-O3", "-o", os.devnull, *probe_flags],
                              input=src,
                              capture_output=True,
                              text=True,
                              timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


#: The OpenMP fork call an auto-parallelized loop must contain to have actually been parallelized:
#: ``GOMP_parallel`` (gcc's parloops) or ``__kmpc_fork_call`` (LLVM). Absence == the loop stayed serial.
_AUTOPAR_FORK_SYMS = ("GOMP_parallel", "kmpc_fork")


@functools.lru_cache(maxsize=None)
def autopar_fires(compiler: str, probe_flags: Tuple[str, ...]) -> bool:
    """True if ``probe_flags`` make ``compiler`` actually EMIT a parallel loop (``nm -u`` for the fork
    call), not merely accept it: Ubuntu clang 21 parses ``-mllvm -polly`` but schedules no Polly passes,
    and gcc's ``-floop-nest-optimize`` is similarly inert alone."""
    src = "void f(double *restrict a, const double *restrict b, int n){\n" \
          "  for (int i = 0; i < n; i++) a[i] = b[i] * 2.0 + 1.0;\n}\n"
    with tempfile.TemporaryDirectory() as d:
        obj = os.path.join(d, "probe.o")
        try:
            proc = subprocess.run([compiler, "-x", "c", "-", "-c", "-O3", "-o", obj, *probe_flags],
                                  input=src,
                                  capture_output=True,
                                  text=True,
                                  timeout=60)
            if proc.returncode != 0:
                return False
            syms = subprocess.run(["nm", "-u", obj], capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            return False
    return any(s in syms for s in _AUTOPAR_FORK_SYMS)


def autopar_flags(family: str,
                  nthreads: int,
                  compiler: Optional[str] = None) -> Tuple[Optional[List[str]], Optional[str]]:
    """Compiler AUTO-PARALLELIZER flags for a family, or ``(None, reason)`` when it has none or its
    polyhedral back end is absent. gcc needs ``-fgraphite-identity`` to force SCoP detection (it
    parallelizes no symbolic-bound loop without it). Nothing here forces a back end past its own cost
    model -- the arena's timing IS the final cost model. An absent back end is probed and returned as a
    recorded skip; ``compiler=None`` skips probing (pure composition, for tests/figures)."""
    if family == "gnu":
        ap = [
            "-ftree-parallelize-loops=%d" % nthreads, "-floop-parallelize-all", "-floop-nest-optimize",
            "-fgraphite-identity", "-fopenmp"
        ]
        absent = "gcc built without Graphite (isl unavailable: -floop-nest-optimize rejected)"
    elif family == "llvm":
        ap = ["-mllvm", "-polly", "-mllvm", "-polly-parallel", "-fopenmp"]
        absent = "clang built without Polly (-mllvm -polly rejected)"
    elif family == "nvidia":
        # must fall through to the same accepts+fires probes: -Mconcur is accepted even when it leaves the
        # loop serial, and an early return would time a sequential build under the 'auto-par' label.
        ap = ["-Mconcur"]
        absent = "nvc rejected -Mconcur (auto-parallelizer unavailable)"
    elif family == "intel":
        ap = ["-qopenmp", "-parallel"]
        absent = "icx/icc rejected -qopenmp -parallel (auto-parallelizer unavailable)"
    else:
        return None, f"no auto-parallelizer known for compiler family {family!r}"
    if compiler is not None:
        if not compiler_accepts(compiler, tuple(ap)):
            return None, absent
        # accepted isn't enough (Ubuntu clang 21 parses -polly and schedules nothing): require it to FIRE
        if not autopar_fires(compiler, tuple(ap)):
            return None, f"{Path(compiler).name} accepts the auto-par flags but emits no parallel loop (back end inert)"
    return ap, None


def omp_emit_flags(family: str) -> List[str]:
    """Switch that turns ON the OpenMP pragmas WE emit: ``-fopenmp`` (gcc/clang/icx) or ``-mp`` (nvc)."""
    return {"nvidia": ["-mp"]}.get(family, ["-fopenmp"])


#: icx auto-links libsvml/libimf/libirng/libintlc from its own off-path lib dir with NO RUNPATH; probing
#: this one locates the whole set (they share a directory). gcc/clang have no comparable set.
SUPPORT_LIB_PROBE = "svml"


@functools.lru_cache(maxsize=None)
def support_rpath_flags(compiler: str) -> Tuple[str, ...]:
    """``-Wl,-rpath`` for the compiler's own auto-linked support libraries, or ``()`` when it has none.
    NOT the OpenMP runtime's directory -- an icx cell rpathing only the libomp dir links but dies at
    ``dlopen``. The arena dlopens node libraries in-process, so ``LD_LIBRARY_PATH`` is not an option."""
    found = driver_lib_path(SUPPORT_LIB_PROBE, compiler)
    return ("-Wl,-rpath,%s" % found.parent, ) if found else ()


@functools.lru_cache(maxsize=None)
def runtime_dir(soname: str, compiler: str) -> Optional[str]:
    """Directory for BOTH ``-L`` and ``-Wl,-rpath``, or ``None`` when neither is needed. Asks the LINKING
    compiler first, widening only if it doesn't know: resolving availability and ``-L`` through different
    helpers once emitted a bare ``-lomp`` with no ``-L``."""
    found = driver_lib_path(soname, compiler)
    if found is not None:
        return str(found.parent)
    return linkable_lib_dir(soname, compiler)


def openmp_runtime_flags(compiler: Optional[str], family: str,
                         runtime: OpenMPRuntime) -> Tuple[Optional[List[str]], Optional[str]]:
    """Pin this cell to EXACTLY ONE OpenMP runtime, or ``(None, reason)`` when this compiler can't link
    it. Bare ``-fopenmp`` lets each family link its own default (gcc->libgomp, clang->libomp), putting two
    thread pools in one process once a sweep spans compilers. ``gnu`` has no select-by-name switch, so
    ``--push-state,--no-as-needed`` pins it NEEDED regardless of link position, else the driver's trailing
    ``-lgomp`` wins. ``-L``/``-rpath`` let the ``.so`` load without ``LD_LIBRARY_PATH``."""
    if not compiler:
        return [], None  # pure composition (tests / figures)
    if not runtime.compatible(compiler):
        return None, f"{Path(compiler).name} cannot link {runtime.name} (single-runtime contract)"
    # compatible() checks ABI only (gcc+libnvomp passes ABI but dies at link), so confirm -l<soname>
    # resolves. Runtime family != FP family: icx is 'intel' above but clang-based for the -l branch here.
    omp_family = compiler_family(compiler)
    if omp_family not in ("intel-classic", "nvidia") and not lib_linkable(runtime.soname, compiler):
        return None, f"{runtime.name} is not linkable by {Path(compiler).name} (runtime not installed for it)"
    lib_dir = runtime_dir(runtime.soname, compiler)
    search = ["-L%s" % lib_dir, "-Wl,-rpath,%s" % lib_dir] if lib_dir else []
    search += list(support_rpath_flags(compiler))
    if omp_family == "llvm":
        return [f"-fopenmp={runtime.name}", *search], None
    if omp_family in ("intel-classic", "nvidia"):
        return search, None  # -qopenmp / -mp already hard-link the only runtime they accept
    # gnu: pin NEEDED despite preceding the source, then restore --as-needed so -lgomp drops out
    return [*search, f"-Wl,--push-state,--no-as-needed,-l{runtime.soname},--pop-state"], None


def cxx_source_flags(family: str, cxx_std: str = CXX_STD) -> List[str]:
    """Flags to compile the numpyto-emitted **C** source as C++ (there is no distinct C++ target). C++
    lacks ``restrict``, and gnu also lacks ``__builtin_complex`` in C++ mode, so both are shimmed."""
    flags = ["-x", "c++", "-std=" + cxx_std, "-Drestrict=__restrict__"]
    if family == "gnu":
        flags.append("-D__builtin_complex(re,im)=((__complex__ double){re,im})")
    return flags


def veclib_flags(compiler: Optional[str], veclib: Optional[str]) -> Tuple[Optional[List[str]], Optional[str]]:
    """Compile+link flags for a vector-math library on an external lane, or ``(None, reason)`` when
    incompatible/unknown/requested without a compiler. Per-family spelling delegates to
    :class:`build.VectorMathLib`, imported lazily to keep this module dace-free."""
    if not veclib or veclib == "none":
        # icx auto-links libsvml/libimf, so even the scalar baseline needs the support rpath
        return (list(support_rpath_flags(compiler)) if compiler else []), None
    if not compiler:
        return None, f"veclib {veclib} requested without a compiler to resolve its family"
    from nestforge.build import VECTOR_LIBS
    vl = VECTOR_LIBS.get(veclib)
    if vl is None:
        return None, f"unknown veclib {veclib!r} (expected one of {tuple(VECTOR_LIBS)})"
    if not vl.compatible(compiler):
        return None, f"veclib {veclib} incompatible with {Path(compiler).name}"
    return vl.compile_flags(compiler) + vl.link_flags(compiler) + list(support_rpath_flags(compiler)), None


def lane_flags(family: str,
               fp_mode: str,
               cost_model: str,
               parallel: str,
               lang: str,
               nthreads: int,
               cxx_std: str = CXX_STD,
               compiler: Optional[str] = None,
               veclib: Optional[str] = None,
               openmp: Optional[OpenMPRuntime] = None) -> Tuple[Optional[List[str]], Optional[str]]:
    """Compose the full compile flags for ONE full-matrix (tsvc_full) sweep cell, or ``(None, reason)``
    when the axis combination is unsupported.

    :param fp_mode: a :data:`FP_LEVELS` rung or a :data:`REDUCED_FP_MODES` rung.
    :param lang: ``"c"``, ``"c++"`` or ``"fortran"``.
    :param openmp: the ONE runtime every parallel cell links (:data:`DEFAULT_OPENMP_RUNTIME` when unset);
        a compiler that can't link it declines rather than silently falling back."""
    fp_lang = "fortran" if lang == "fortran" else "c"
    out = base_flags(family)
    if lang == "c++":
        out = out + cxx_source_flags(family, cxx_std)
    # BOTH axes are accepted: routing every non-strict-ieee value to reduced_fp_flags made the other
    # FP_LEVELS rungs raise KeyError deep in _REDUCED_FP instead of composing.
    if fp_mode in FP_LEVELS:
        out = out + fp_flags(family, fp_mode, fp_lang)
    elif fp_mode in REDUCED_FP_MODES:
        out = out + reduced_fp_flags(family, fp_mode, fp_lang)
    else:
        return None, f"unknown fp_mode {fp_mode!r} (known: {FP_LEVELS + REDUCED_FP_MODES})"
    out = out + cost_flags(family, cost_model)
    vec, vreason = veclib_flags(compiler, veclib)
    if vec is None:
        return None, vreason
    out = out + vec
    if parallel in ("auto-par", "omp-emit"):
        if parallel == "auto-par":
            par, reason = autopar_flags(family, nthreads, compiler)
        else:
            par, reason = omp_emit_flags(family), None  # OUR pragmas are already in the source
        if par is None:
            return None, reason
        # pin the ONE mandated runtime, so gcc/clang cells share a thread pool
        rt, reason = openmp_runtime_flags(compiler, family, openmp or DEFAULT_OPENMP_RUNTIME)
        if rt is None:
            return None, reason
        out = out + par + rt
    return out, None
