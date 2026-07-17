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

from nestforge.build import LIBOMP, OpenMPRuntime, compiler_family, driver_lib_path, lib_linkable, linkable_lib_dir

#: The ONE OpenMP runtime every lane links unless a cell names another. libomp because it is the only
#: runtime BOTH families can link -- LLVM selects it by name, and it carries a ``GOMP_*`` compat layer a
#: gcc object resolves against -- so gcc- and clang-built node libraries share one runtime and one thread
#: pool. (libiomp5 qualifies too, ABI-compatible with libomp, but ships only with Intel's toolchain.)
#: Same default and same class as the DaCe lane's :class:`~nestforge.build.OpenMPRuntime`, deliberately:
#: one runtime GLOBALLY means one choice, honoured by every lane.
DEFAULT_OPENMP_RUNTIME = LIBOMP

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
#:  * ``auto-par``   -- the sequential emit + the compiler's OWN auto-parallelizer, POLYHEDRAL by default
#:    per the plan (gcc ``Graphite``, clang ``Polly``, nvc ``-Mconcur``, icx ``-parallel``). Graphite and
#:    Polly are optional back ends: a compiler lacking them yields a recorded skip (see :func:`autopar_flags`).
#:  * ``omp-emit``   -- OUR ``#pragma omp parallel for`` source (numpyto ``c_omp``/``fortran_omp``) + a
#:    bare ``-fopenmp``. Works for EVERY family (clang/flang included) and only for the nests the DaCe
#:    schedule marks parallel AND numpyto can soundly parallelize.
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


@functools.lru_cache(maxsize=None)
def compiler_accepts(compiler: str, probe_flags: Tuple[str, ...]) -> bool:
    """True if ``compiler`` accepts ``probe_flags`` on a trivial COMPILE-ONLY invocation.

    Gates the two polyhedral auto-parallelizers, which ride an OPTIONAL compiler back end a given
    install may lack -- clang's Polly (``-mllvm -polly``) and gcc's isl/Graphite (``-floop-nest-optimize``).
    A missing back end makes the invocation exit non-zero (LLVM: ``Unknown command line argument '-polly'``;
    gcc: ``Graphite loop optimizations cannot be used (isl is not available)``), which is exactly what this
    detects. ``-c`` keeps it compile-only, so a missing OpenMP RUNTIME (a link-time concern the caller
    rpaths separately) does not confound the feature probe. Cached: fixed per (compiler, flags) for the run."""
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


def autopar_flags(family: str,
                  nthreads: int,
                  compiler: Optional[str] = None) -> Tuple[Optional[List[str]], Optional[str]]:
    """Compiler AUTO-PARALLELIZER flags for a family, or ``(None, reason)`` when the family has no
    auto-parallelizer OR its polyhedral back end is absent from ``compiler``.

    Per the plan the DEFAULT auto-parallelizer for the two open-source families is polyhedral:
      * ``gnu``     -- **Graphite**: ``-ftree-parallelize-loops=N -floop-parallelize-all
        -floop-nest-optimize -fopenmp`` (the isl loop-nest optimizer plus parallelize-all, which needs
        the ``-ftree-parallelize-loops`` machinery). ``-floop-nest-optimize`` requires gcc built with isl;
      * ``llvm``    -- **Polly**: ``-mllvm -polly -mllvm -polly-parallel -fopenmp`` (mirrors optarena
        ``POLLY_PAR``). Requires clang built with Polly -- distribution clang frequently is not;
      * ``nvidia``  -- ``-Mconcur`` auto-concurrentizes; threads come from ``OMP_NUM_THREADS``/``NCPUS``;
      * ``intel``   -- ``-qopenmp -parallel`` (classic icc/ifort auto-par); a rejecting icx compile is
        recorded as an error cell, never silently dropped.

    Both polyhedral back ends are OPTIONAL compiler features: when ``compiler`` is supplied it is probed
    (:func:`compiler_accepts`) and an absent back end yields ``(None, reason)`` -- a recorded skip, not a
    crash. With no ``compiler`` the intended flag list is returned unprobed (pure composition, for tests /
    figures). ``-fopenmp`` here also pulls an OpenMP runtime; :func:`lane_flags` rpaths it so the ``.so``
    loads standalone."""
    if family == "gnu":
        ap = ["-ftree-parallelize-loops=%d" % nthreads, "-floop-parallelize-all", "-floop-nest-optimize", "-fopenmp"]
        absent = "gcc built without Graphite (isl unavailable: -floop-nest-optimize rejected)"
    elif family == "llvm":
        ap = ["-mllvm", "-polly", "-mllvm", "-polly-parallel", "-fopenmp"]
        absent = "clang built without Polly (-mllvm -polly rejected)"
    elif family == "nvidia":
        return ["-Mconcur"], None
    elif family == "intel":
        return ["-qopenmp", "-parallel"], None
    else:
        return None, f"no auto-parallelizer known for compiler family {family!r}"
    if compiler is not None and not compiler_accepts(compiler, tuple(ap)):
        return None, absent
    return ap, None


def omp_emit_flags(family: str) -> List[str]:
    """The switch that turns ON the OpenMP pragmas WE emit (``omp-emit`` parallel mode): a bare
    ``-fopenmp`` on gcc/clang/icx, ``-mp`` on nvc. Unlike :func:`autopar_flags` this needs no
    middle-end auto-parallelizer -- the parallelism is already in the source -- so it is supported by
    EVERY family (clang/flang included). The runtime thread count comes from ``OMP_NUM_THREADS``."""
    return {"nvidia": ["-mp"]}.get(family, ["-fopenmp"])


#: A support library a compiler AUTO-LINKS into every object, over and above the OpenMP runtime. icx
#: pulls Intel's vector-math + libm set (libsvml/libimf/libirng/libintlc), which live in the compiler's
#: OWN lib dir -- off the loader's default path -- and icx bakes NO RUNPATH. Probing for one of them
#: locates the whole set: they share a directory. gcc/clang auto-link nothing comparable, so the probe
#: simply answers None for them and costs one cached driver call.
SUPPORT_LIB_PROBE = "svml"


@functools.lru_cache(maxsize=None)
def support_rpath_flags(compiler: str) -> Tuple[str, ...]:
    """``-Wl,-rpath`` for the compiler's own auto-linked support libraries, or ``()`` when it has none.

    Not the same directory as the OpenMP runtime, and assuming so is a real failure: an icx cell built
    against the DEFAULT libomp rpaths the SYSTEM libomp dir, which holds no libsvml, and the .so links
    cleanly and then dies at ``dlopen`` with ``libsvml.so: cannot open shared object file``. It works with
    libiomp5 only because Intel keeps its runtime and its support libs in ONE dir -- coincidence, not
    design, and it evaporates the moment the global runtime is libomp. Measured both ways.

    This is what lets an icx cell be discovered by path alone, with no ``setvars.sh``: the arena dlopens
    node libraries IN-PROCESS, so a runtime that needs ``LD_LIBRARY_PATH`` set before interpreter start is
    unusable to it. Baking the rpath is the same technique :func:`openmp_runtime_flags` already uses to
    make clang's off-path libomp loadable.
    """
    found = driver_lib_path(SUPPORT_LIB_PROBE, compiler)
    return ("-Wl,-rpath,%s" % found.parent, ) if found else ()


@functools.lru_cache(maxsize=None)
def runtime_dir(soname: str, compiler: str) -> Optional[str]:
    """The directory for BOTH ``-L`` (so ``-l<soname>`` resolves) and ``-Wl,-rpath`` (so the ``.so``
    LOADS without ``LD_LIBRARY_PATH``), or ``None`` when neither is needed.

    Two questions, two answers, and conflating them is a real bug rather than a nicety -- it is exactly
    how CI broke: availability was checked with :func:`~nestforge.build.lib_linkable` (which knows to ask
    a SIBLING driver, and answered "yes, clang-18 has it") while the ``-L`` came from
    :func:`~nestforge.build.driver_lib_path` (which asks gcc ITSELF, and answered None). The cell then
    emitted a bare ``-lomp`` with no ``-L`` and died with ``cannot find -lomp``. On a box where gcc finds
    libomp directly the two agree and the bug is invisible.

    Ask the compiler that will do the linking FIRST: it ships its own runtime and knows its libdir under
    any prefix -- clang -> its libomp, icx -> libiomp5 (whose directory also holds libsvml, which icx
    auto-links, so one rpath covers both and the .so loads standalone), nvc -> libnvomp. Only when that
    driver does not know the runtime does the wider search apply (explicit LD_LIBRARY_PATH / LIBRARY_PATH,
    then a sibling driver, then the distro's LLVM layouts).
    """
    found = driver_lib_path(soname, compiler)
    if found is not None:
        return str(found.parent)
    return linkable_lib_dir(soname, compiler)


def openmp_runtime_flags(compiler: Optional[str], family: str,
                         runtime: OpenMPRuntime) -> Tuple[Optional[List[str]], Optional[str]]:
    """Pin this cell to EXACTLY ONE OpenMP runtime -- ``runtime``, whichever compiler builds it -- or
    ``(None, reason)`` when this compiler cannot link it (a recorded skip, never a silent other runtime).

    PARALLEL.md mandates one runtime for every node library AND the driver, so that libraries built by
    different compilers share ONE runtime and ONE thread pool. Left to the bare ``-fopenmp`` each family
    links its OWN default (gcc->libgomp, clang->libomp, icx->libiomp5), which quietly puts TWO runtimes
    and two thread pools in one process the moment a sweep spans gcc and clang -- the dual-runtime
    oversubscription :class:`~nestforge.build.OpenMPRuntime` exists to prevent. The DaCe lane already
    routes through that class; this is the same contract for the external lanes, so a runtime chosen
    once holds across every lane and compiler.

    Per family, given ``-fopenmp`` is already on the line (from :func:`omp_emit_flags` /
    :func:`autopar_flags`, which is what compiles the pragmas):

    * ``llvm``  -- selects BY NAME: ``-fopenmp=<name>`` overrides the default; nothing else needed.
    * ``gnu``   -- has no ``-fopenmp=<lib>``, so the runtime is chosen at LINK, and ORDER decides it.
      These flags land BEFORE the source (``[exe, *flags, src, -o, so]``), where nothing is undefined
      yet, so a plain ``--as-needed -l<soname>`` is dropped as unused and the driver's implicit trailing
      ``-lgomp`` wins -- the exact opposite of the intent. ``--push-state,--no-as-needed`` pins the
      runtime NEEDED regardless of position, and ``--pop-state`` restores ``--as-needed`` so that
      trailing ``-lgomp`` is then dropped: its ``GOMP_*`` calls already resolve against the runtime
      already in the link. Bare ``--no-as-needed`` would link BOTH. Verified by ``readelf -d``.
    * ``intel-classic`` / ``nvidia`` -- ``-qopenmp`` / ``-mp`` hard-link their native runtime and
      accept no other, which :meth:`OpenMPRuntime.compatible` already encodes; only the search path
      is added.

    The ``-L`` + ``-Wl,-rpath`` pair makes the ``.so`` LOAD its runtime without ``LD_LIBRARY_PATH``
    (LLVM's libomp lives in the clang install tree, off the default loader path -- without this a clang
    omp-emit cell dies at dlopen with ``libomp.so: cannot open shared object file``). The directory
    comes from the compiler driver via :func:`~nestforge.build.driver_lib_path`, the one helper that
    knows to normalise lexically rather than ``resolve()`` (``libomp.so`` is a symlink to a
    ``libomp.so.5`` that can live in a DIFFERENT directory, one holding no ``libomp.so`` to link).
    """
    if not compiler:
        return [], None  # pure composition (tests / figures): no driver to ask, no cell to build
    if not runtime.compatible(compiler):
        return None, f"{Path(compiler).name} cannot link {runtime.name} (single-runtime contract)"
    # compatible() answers ABI ("gcc COULD link any gomp-ABI runtime"), not availability. libnvomp ships
    # only inside the NVIDIA HPC SDK, so gcc+libnvomp passes the ABI test and then dies at link with
    # `cannot find -lnvomp`. Ask whether -l<soname> actually resolves for THIS compiler before emitting a
    # cell that cannot build: an absent runtime is a recorded skip, never a hard failure.
    # The OpenMP family is NOT the caller's ``family``: that labels the FP-flag matrix, where icx/ifx are
    # split out as 'intel' because they default to -fp-model=fast. For RUNTIME selection they are
    # clang-based and select by name, so ask the authority -- compiler_family() -- or icx would take the
    # gnu branch and pin its runtime with -l instead of -fopenmp=<name>.
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
    # gnu: pin NEEDED despite preceding the source, then restore --as-needed so -lgomp drops out.
    return [*search, f"-Wl,--push-state,--no-as-needed,-l{runtime.soname},--pop-state"], None


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
               veclib: Optional[str] = None,
               openmp: Optional[OpenMPRuntime] = None) -> Tuple[Optional[List[str]], Optional[str]]:
    """Compose the full compile flags for ONE full-matrix (tsvc_full) sweep cell, or ``(None, reason)``
    when the axis combination is unsupported (e.g. clang auto-par, or an incompatible veclib).

    ``fp_mode`` is either ``"strict-ieee"`` (the bit-exact gate rung, from :data:`FP_LEVELS`) or one of
    :data:`REDUCED_FP_MODES`. ``lang`` is ``"c"`` | ``"c++"`` | ``"fortran"``; the C++ lane recompiles the
    C source (so it uses the C FP spellings plus the C++ frontend flags), Fortran uses the Fortran FP
    spellings. ``veclib`` (``none`` | ``sleef`` | ``libmvec`` | ``svml``) adds the vector-math-library
    ``-fveclib=``/``-l`` flags. Every list starts from :func:`base_flags` (``-O3``/arch/PIC/shared).

    ``openmp`` is the ONE runtime every parallel cell links, whatever compiler builds it
    (:data:`DEFAULT_OPENMP_RUNTIME` when unset) -- see :func:`openmp_runtime_flags`. It is a knob, not a
    constant: any of :data:`~nestforge.build.OPENMP_RUNTIMES` can be selected, and a compiler that cannot
    link the chosen one yields ``(None, reason)`` rather than quietly falling back to its own default.
    Note libgomp can only ever be a gnu-ONLY choice: clang emits ``__kmpc_*``, which libgomp does not
    implement, so it is not a viable cross-compiler runtime (:meth:`OpenMPRuntime.compatible`)."""
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
    if parallel in ("auto-par", "omp-emit"):
        if parallel == "auto-par":
            par, reason = autopar_flags(family, nthreads, compiler)
        else:
            # OUR pragmas are already in the source (numpyto c_omp/fortran_omp); just enable OpenMP.
            par, reason = omp_emit_flags(family), None
        if par is None:
            return None, reason
        # Both spellings enable OpenMP but leave the RUNTIME to the family default; pin the one mandated
        # runtime instead, so a gcc cell and a clang cell in one process share it (and one thread pool).
        rt, reason = openmp_runtime_flags(compiler, family, openmp or DEFAULT_OPENMP_RUNTIME)
        if rt is None:
            return None, reason
        out = out + par + rt
    return out, None
