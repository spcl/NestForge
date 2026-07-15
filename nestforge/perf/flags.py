"""Shared compile-flag matrix for the arena: an **FP-precision level** axis crossed with a
**vectorizer cost-model** axis, per compiler family, for both C and Fortran.

The FP-precision ladder (0 strictest -> 3 fastest) is documented in ``docs/FP_PRECISION_LEVELS.md``
and was derived + verified against the real gcc 15, clang 21, nvc 26.3 and icx/ifx 2026.1 on this
box. The per-level lists here are the **FP-mode component only**: :func:`base_flags` adds
``-O3``/arch/``-fPIC``/``-shared`` and :func:`cost_flags` adds the vectorizer axis, so
:func:`flag_matrix` composes the full command for a cell.

Family labels: ``gnu`` | ``llvm`` | ``nvidia`` | ``intel``. ``intel`` (icx/icpx/ifx) is split out
from ``llvm`` even though those compilers are clang-based, because they **default to
``-fp-model=fast``**: a bare ``-ffp-contract=off`` would leave reassociation/reciprocal/FTZ on, so
only an explicit ``-fp-model`` resets Intel to IEEE. The FP flags therefore differ from clang's.
"""
from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: FP-precision levels, strictest first; the index is the ladder rung.
FP_LEVELS: Tuple[str, ...] = ("strict-ieee", "contract-fma", "assume-finite", "fast-math")

#: Validation tolerance for a cell at each level vs the numpy fp64 oracle. The oracle itself is not
#: bit-reproducible (pairwise ``np.sum``, BLAS-backed dot, non-correctly-rounded libm), so even
#: ``strict-ieee`` is not atol 0; tolerances are relative-ish and cover O(N)*eps reduction growth.
FP_ATOL: Dict[str, float] = {
    "strict-ieee": 1e-14,
    "contract-fma": 1e-13,
    "assume-finite": 1e-13,
    "fast-math": 1e-5,
}

#: FP-mode flags per (family, level) -- the C spellings. Fortran deltas are applied by
#: :func:`fortran_fp_flags`. Verified to compile on the local gcc/clang/nvc/icx builds.
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
        # nvc exposes only whole-model FP knobs, so `assume-finite` collapses to `contract-fma`
        # numerically (there is no per-assumption flag); the matrix dedups the duplicate.
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

#: Vectorizer cost-model axis. "default" = the compiler's own model; "no-vec" = the scalar floor;
#: "cheap" = fewer/safer vectorizations (only gcc has a direct knob, so it collapses elsewhere).
COST_MODELS: Tuple[str, ...] = ("default", "cheap", "no-vec")

#: Vector-math-library axis DOMAIN (``-fveclib=`` / ``-lsleef`` etc.): ``none`` plus the three libraries.
#: The arena does NOT sweep all of these -- per-device characterization (``device_profile.rank_veclibs``)
#: collapses it to ``none`` + the accuracy-gated winner. This is the full domain the figure/CLI validate
#: against; the per-family spelling lives in ``build.VectorMathLib`` (delegated to by :func:`veclib_flags`).
VECLIBS: Tuple[str, ...] = ("none", "sleef", "libmvec", "svml")


def base_flags(family: str) -> List[str]:
    """``-O3`` + native tuning + PIC/shared -- the common prefix every cell shares."""
    return ["-O3", _ARCH.get(family, "-march=native"), "-fPIC", "-shared"]


def fortran_fp_flags(family: str, level: str) -> List[str]:
    """The FP-mode flags for a family's **Fortran** frontend (gfortran/flang/nvfortran/ifx).

    gfortran needs ``-fno-frontend-optimize`` (its front end reassociates source at ``-O`` even with
    ``-ffp-contract=off``) and ``-fno-protect-parens`` at the fast rung. Two gcc C-family flags are
    unsupported by gfortran and stripped: ``-fno-math-errno`` (a no-op -- Fortran intrinsics never set
    ``errno``) and ``-fexcess-precision=standard`` (``f951: sorry, unimplemented``). ``ifx`` likewise
    has no ``errno``. flang mirrors clang and nvfortran mirrors nvc, so those need no change.
    """
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
    """``[(fp_level, cost_model, full_flags), ...]`` for a family/language, deduped so two axis
    combinations that collapse to the same flag list (nvidia ``assume-finite`` == ``contract-fma``,
    clang ``cheap`` == ``default``) produce one compile, not two."""
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
# The full-matrix job does NOT sweep the 4-rung ladder above; it sweeps exactly TWO FP rungs for timing
# (keeping the axis small so the much larger opt-mode x parallelization x language cross-product stays
# tractable), and additionally runs ``strict-ieee`` (from the ladder above) as a bit-exact correctness
# GATE and as the DaCe-cpp speedup-baseline FP mode.

#: The two REDUCED FP rungs the full-matrix timing sweep uses:
#:  * ``default-fp``    -- the compiler's own default FP at ``-O3`` (no FP flag). On gcc/clang ``-O3``
#:    already allows within-statement FMA contraction; on intel the default is ``-fp-model=fast``. So
#:    this rung means "whatever the vendor ships by default", not a fixed numeric guarantee.
#:  * ``no-fast-errno`` -- FMA contraction allowed + ``-fno-math-errno``, but NO reassociation / no
#:    fast-math (the ``contract-fma`` rung with math-errno dropped). The clean "fast but still ordered"
#:    middle rung the plan asks for.
REDUCED_FP_MODES: Tuple[str, ...] = ("default-fp", "no-fast-errno")

#: FP-mode flags per (family, reduced-mode) -- C spellings; :func:`reduced_fp_flags` applies Fortran
#: deltas. ``default-fp`` is empty for every family (the vendor default). ``no-fast-errno`` allows FMA
#: (``-ffp-contract=fast`` / ``-Mfma`` / ``-fp-model=precise``) plus ``-fno-math-errno`` where the
#: frontend accepts it (nvc rejects ``-fno-math-errno``, so nvidia's rung is exactly ``contract-fma``).
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

#: Validation tolerance for each reduced rung vs the numpy fp64 oracle. ``default-fp`` is loose because
#: on intel it is fast-math and on nvidia it is relaxed-div; the tolerance still catches a genuinely
#: broken translation (an O(1) wrong answer), which is all validation is for. ``no-fast-errno`` keeps
#: IEEE ordering so it is near-bit-exact (only FMA single-rounding differs).
REDUCED_FP_ATOL: Dict[str, float] = {"default-fp": 1e-6, "no-fast-errno": 1e-12}

#: The parallelization axis of the full-matrix job.
#:  * ``sequential`` -- the sequential emit, no parallel flags.
#:  * ``auto-par``   -- the sequential emit + the compiler's OWN plain-loop auto-parallelizer
#:    (gcc ``-ftree-parallelize-loops``, nvc ``-Mconcur``, icx ``-parallel``; clang/flang have none).
#:  * ``omp-emit``   -- OUR ``#pragma omp parallel for`` source (numpyto ``c_omp``/``fortran_omp``) + a
#:    bare ``-fopenmp``. Works for EVERY family (clang/flang included, which auto-par cannot reach) and
#:    only for the nests the DaCe schedule marks parallel AND numpyto can soundly parallelize.
PARALLEL_MODES: Tuple[str, ...] = ("sequential", "auto-par", "omp-emit")

#: Default C++ standard for the C++ lane (numpyto emits no C++ target; the C source is recompiled as
#: C++). Overridable per call; the daint job passes ``DACE_PERF_CXX_STD``.
CXX_STD = "c++23"


def reduced_fp_flags(family: str, mode: str, lang: str = "c") -> List[str]:
    """FP-mode flags for a REDUCED rung (:data:`REDUCED_FP_MODES`), adjusted for ``lang``.

    Fortran frontends reject the C-only ``-fno-math-errno`` (Fortran intrinsics never set ``errno``), so
    it is stripped; gfortran additionally reassociates at ``-O`` unless told not to, so ``no-fast-errno``
    gains ``-fno-frontend-optimize`` to keep it genuinely un-reassociated (mirrors :func:`fortran_fp_flags`)."""
    flags = [f for f in _REDUCED_FP[family][mode] if not (lang == "fortran" and f == "-fno-math-errno")]
    if lang == "fortran" and family == "gnu" and mode == "no-fast-errno":
        flags.append("-fno-frontend-optimize")
    return flags


def autopar_flags(family: str, nthreads: int) -> Tuple[Optional[List[str]], Optional[str]]:
    """Compiler AUTO-PARALLELIZER flags for a family, or ``(None, reason)`` if the family has no
    plain-loop auto-parallelizer.

    Verified on the local gcc 15 / clang 21 / nvc 26.3:
      * ``gnu``     -- ``-ftree-parallelize-loops=N -fopenmp`` genuinely emits ``GOMP_parallel`` for
        loops the middle-end proves independent (confirmed via ``nm`` on the emitted ``.so``);
      * ``nvidia``  -- ``-Mconcur`` auto-concurrentizes; threads come from ``OMP_NUM_THREADS``/``NCPUS``;
      * ``intel``   -- ``-qopenmp -parallel`` (classic icc/ifort auto-par). icx (LLVM-based) may have
        dropped ``-parallel``; a rejecting compile is RECORDED as an error cell, never silently dropped,
        so this stays a best-effort attempt to be confirmed on the daint oneAPI;
      * ``llvm``    -- UNSUPPORTED. clang/flang have no plain-loop auto-parallelizer: the numpyto-emitted
        source carries NO OpenMP pragmas for ``-fopenmp`` to act on, and Polly is not reliably built into
        a distribution clang. Returned as an explicit unsupported reason (recorded, not dropped)."""
    if family == "gnu":
        return ["-ftree-parallelize-loops=%d" % nthreads, "-fopenmp"], None
    if family == "nvidia":
        return ["-Mconcur"], None
    if family == "intel":
        return ["-qopenmp", "-parallel"], None
    return None, ("clang/flang has no plain-loop auto-parallelizer (the emitted source has no OpenMP "
                  "pragmas for -fopenmp to act on; Polly is not guaranteed present)")


def omp_emit_flags(family: str) -> List[str]:
    """The switch that turns ON the OpenMP pragmas WE emit (``omp-emit`` parallel mode): a bare
    ``-fopenmp`` on gcc/clang/icx, ``-mp`` on nvc. Unlike :func:`autopar_flags` this needs no
    middle-end auto-parallelizer -- the parallelism is already in the source -- so it is supported by
    EVERY family (clang/flang included). The runtime thread count comes from ``OMP_NUM_THREADS``."""
    return {"nvidia": ["-mp"]}.get(family, ["-fopenmp"])


#: The soname of the OpenMP runtime a bare ``-fopenmp`` / ``-mp`` links per family: gcc pulls libgomp,
#: clang pulls LLVM's libomp, icx pulls Intel's libiomp5, nvc (``-mp``) its native libnvomp. Used to
#: LOCATE that runtime via the compiler driver so it can be rpath-baked into the linked node library.
_OMP_DEFAULT_SONAME: Dict[str, str] = {"gnu": "gomp", "llvm": "omp", "intel": "iomp5", "nvidia": "nvomp"}


@functools.lru_cache(maxsize=None)
def _driver_lib_dir(compiler: str, soname: str) -> Optional[str]:
    """The directory holding ``lib<soname>.so``, as reported by the COMPILER DRIVER itself
    (``<compiler> -print-file-name=lib<soname>.so``). Absolute directory, or ``None`` if the driver
    only echoes the bare name (does not know the file).

    This is how a spack/module OpenMP runtime that sits OFF the loader's default path is found -- e.g.
    LLVM's ``libomp.so`` lives in the clang install tree, on neither the ldconfig cache nor
    ``LD_LIBRARY_PATH``, so the loader / ``ctypes.util.find_library`` cannot see it, but the driver
    that will link it can. Cached: the answer is fixed per (compiler, soname) for the whole sweep."""
    for cand in (f"lib{soname}.so", f"lib{soname}.dylib"):
        try:
            out = subprocess.run([compiler, f"-print-file-name={cand}"], capture_output=True, text=True,
                                 timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            continue
        if out and os.path.isabs(out) and os.path.exists(out):
            return str(Path(out).resolve().parent)
    return None


def openmp_rpath_flags(compiler: Optional[str], family: str) -> List[str]:
    """Link flags so an OpenMP-enabled node library LOADS its runtime WITHOUT ``LD_LIBRARY_PATH``:
    ``-L`` (link search) plus ``-Wl,-rpath`` (bake the dir into the ``.so`` for the runtime loader).

    The runtime directory is detected from the compiler driver (:func:`_driver_lib_dir`) for the
    soname a bare ``-fopenmp`` / ``-mp`` links on this ``family`` (:data:`_OMP_DEFAULT_SONAME`).
    Returns ``[]`` when the compiler is unknown or the runtime cannot be located (then the default
    loader path must already carry it -- as gcc's libgomp does). This is what lets clang ``omp-emit``
    cells load at all: without it they die at dlopen with ``libomp.so: cannot open shared object
    file`` because LLVM's libomp is off the default loader path. Mirrors the rpath baking libnode.py
    already does for its own node-library dependency."""
    if not compiler:
        return []
    d = _driver_lib_dir(compiler, _OMP_DEFAULT_SONAME.get(family, "omp"))
    return ["-L%s" % d, "-Wl,-rpath,%s" % d] if d else []


def cxx_source_flags(family: str, cxx_std: str = CXX_STD) -> List[str]:
    """Flags to compile the numpyto-emitted **C** source as C++ (numpyto has no distinct C++ target).

    ``-x c++`` retargets the ``.c`` input to the C++ frontend; C++ has no ``restrict`` keyword, so
    ``-Drestrict=__restrict__`` supplies the equivalent (accepted by g++/clang++/nvc++/icpx). g++
    ADDITIONALLY lacks ``__builtin_complex`` in C++ mode (clang/nvc/icx have it), so a compound-literal
    shim producing the same ``__complex__ double`` is defined for the gnu family only. Both verified to
    build the emitted source on the local g++ 15 / clang++ 21 / nvc++ 26.3."""
    flags = ["-x", "c++", "-std=" + cxx_std, "-Drestrict=__restrict__"]
    if family == "gnu":
        flags.append("-D__builtin_complex(re,im)=((__complex__ double){re,im})")
    return flags


def veclib_flags(compiler: Optional[str], veclib: Optional[str]) -> Tuple[Optional[List[str]], Optional[str]]:
    """Compile+link flags for a vector-math library on an external lane, or ``(None, reason)`` when it is
    incompatible with ``compiler`` / unknown / requested without a compiler. ``None``/``"none"`` -> no
    flags. The per-family spelling is delegated to the :class:`build.VectorMathLib` registry, imported
    lazily so this module stays dace-free for the plot readers that never touch a veclib."""
    if not veclib or veclib == "none":
        return [], None
    if not compiler:
        return None, f"veclib {veclib} requested without a compiler to resolve its family"
    from nestforge.build import VECTOR_LIBS
    vl = VECTOR_LIBS.get(veclib)
    if vl is None:
        return None, f"unknown veclib {veclib!r} (expected one of {tuple(VECTOR_LIBS)})"
    if not vl.compatible(compiler):
        return None, f"veclib {veclib} incompatible with {Path(compiler).name}"
    return vl.compile_flags(compiler) + vl.link_flags(compiler), None


def lane_flags(family: str,
               fp_mode: str,
               cost_model: str,
               parallel: str,
               lang: str,
               nthreads: int,
               cxx_std: str = CXX_STD,
               compiler: Optional[str] = None,
               veclib: Optional[str] = None) -> Tuple[Optional[List[str]], Optional[str]]:
    """Compose the full compile flags for ONE full-matrix (tsvc_full) sweep cell, or ``(None, reason)``
    when the axis combination is unsupported (e.g. clang auto-par, or an incompatible veclib).

    ``fp_mode`` is either ``"strict-ieee"`` (the bit-exact gate rung, from :data:`FP_LEVELS`) or one of
    :data:`REDUCED_FP_MODES`. ``lang`` is ``"c"`` | ``"c++"`` | ``"fortran"``; the C++ lane recompiles the
    C source (so it uses the C FP spellings plus the C++ frontend flags), Fortran uses the Fortran FP
    spellings. ``veclib`` (``none`` | ``sleef`` | ``libmvec`` | ``svml``) adds the vector-math-library
    ``-fveclib=``/``-l`` flags. Every list starts from :func:`base_flags` (``-O3``/arch/PIC/shared)."""
    fp_lang = "fortran" if lang == "fortran" else "c"
    out = base_flags(family)
    if lang == "c++":
        out = out + cxx_source_flags(family, cxx_std)
    if fp_mode == "strict-ieee":
        out = out + fp_flags(family, "strict-ieee", fp_lang)
    else:
        out = out + reduced_fp_flags(family, fp_mode, fp_lang)
    out = out + cost_flags(family, cost_model)
    vec, vreason = veclib_flags(compiler, veclib)
    if vec is None:
        return None, vreason
    out = out + vec
    if parallel == "auto-par":
        ap, reason = autopar_flags(family, nthreads)
        if ap is None:
            return None, reason
        # -fopenmp/-qopenmp here also pulls an OpenMP runtime -> rpath it so the .so loads standalone.
        out = out + ap + openmp_rpath_flags(compiler, family)
    elif parallel == "omp-emit":
        # OUR pragmas are already in the source (numpyto c_omp/fortran_omp); just enable OpenMP. Bake an
        # rpath to the OpenMP runtime so the linked .so loads it without LD_LIBRARY_PATH (fixes clang
        # 'libomp.so: cannot open shared object file': LLVM's libomp is off the default loader path).
        out = out + omp_emit_flags(family) + openmp_rpath_flags(compiler, family)
    return out, None
