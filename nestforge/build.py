"""nest-forge owns the DaCe build (see BUILD.md): generate DaCe's C++ ourselves, compile + link it with
one chosen compiler + flag set, and call it via ctypes with manual init / program / exit -- instead of
``dace.compile()`` (whose Python ``__call__`` re-marshals every argument, confounding timing, and whose
build system we do not control).

``sdfg.generate_code()`` yields a Frame ``.cpp`` + a CallHeader ``.h`` (+ a SampleMain we drop). The
generated code exposes three C-linkage entry points for an SDFG named ``N``:
  * ``N_state_t *__dace_init_N(<init-symbols>)``  -- allocate the SDFG state (persistent transients,
    library-node/BLAS handles); returns an opaque handle,
  * ``void __program_N(N_state_t *state, <args>)`` -- the kernel (timed per invocation),
  * ``int __dace_exit_N(N_state_t *state)``        -- free the state.
The ``.so`` does not auto-initialize; we call all three. Arrays pass as pointers, size symbols and
scalars pass by value (a DaCe ``Scalar`` is by value -- unlike nest-forge's C-style emission, which
treats it as a size-1 buffer).
"""
from __future__ import annotations

import ctypes
import ctypes.util
import functools
import re
import shutil
import subprocess
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import dace

#: ctypes type per C scalar type appearing in a DaCe entry-point signature.
_C_SCALAR = {
    "int32_t": ctypes.c_int32,
    "int64_t": ctypes.c_int64,
    "int": ctypes.c_int,
    "float": ctypes.c_float,
    "double": ctypes.c_double,
    "bool": ctypes.c_bool
}
_C_PTR = {"float": ctypes.c_float, "double": ctypes.c_double, "int32_t": ctypes.c_int32, "int64_t": ctypes.c_int64}

DEFAULT_COMPILER = "g++"
DEFAULT_FLAGS = ["-O3", "-march=native", "-std=c++14", "-fPIC", "-shared"]


def compiler_family(compiler: str) -> str:
    """The OpenMP-relevant family of a compiler executable: ``llvm`` (clang/clang++/flang/flang-new and
    the LLVM-based Intel icx/icpx/ifx -- all select the runtime by name), ``intel-classic`` (icc/icpc/
    ifort -> ``-qopenmp``, libiomp5), ``nvidia`` (nvc/nvc++/nvfortran -> ``-mp``), or ``gnu`` (gcc/g++/
    gfortran -> ``-fopenmp``, emits GOMP calls with the runtime chosen at link)."""
    b = Path(compiler).name.lower()
    if "clang" in b or "flang" in b or b.startswith(("icx", "icpx", "ifx")):
        return "llvm"
    if b.startswith(("icc", "icpc", "ifort")):
        return "intel-classic"
    if b.startswith(("nvc", "nvfortran", "pgcc", "pgfortran")):
        return "nvidia"
    return "gnu"


#: The OpenMP ABI a compiler family *emits* -- ``gomp`` (GCC's ``GOMP_*`` entry points) or ``kmpc`` (the
#: LLVM/Intel ``__kmpc_*`` entry points, which clang/flang/icx/icc AND nvc/nvc++ all emit). A runtime is
#: link-compatible with a compiler only if it implements the ABI the compiler emits.
_COMPILER_ABI = {"gnu": "gomp", "llvm": "kmpc", "intel-classic": "kmpc", "nvidia": "kmpc"}

#: The runtimes clang/flang/icx can select by NAME via ``-fopenmp=<name>`` -- the driver only understands
#: these three tokens, so a runtime outside this set (libnvomp, a custom name) is NOT reachable from an
#: LLVM compiler even if its ABI matches. gcc, by contrast, links any runtime explicitly with ``-l<soname>``.
_LLVM_SELECTABLE = frozenset({"libomp", "libgomp", "libiomp5"})


@dataclass
class OpenMPRuntime:
    """The single OpenMP runtime the whole program links against -- a SEPARATE, configurable flag axis,
    not folded into the base flags (PARALLEL.md mandates one runtime for every node library and the
    driver). ``libomp`` is the default because it is the most portable: LLVM/Clang/flang select it by
    name (``-fopenmp=libomp`` -- the user's example), it is ABI-compatible with Intel's libiomp5, AND it
    implements the ``GOMP_*`` ABI, so a GCC-compiled object (which emits ``GOMP_*`` calls) resolves
    against it too. That is what lets a set of node libraries built with DIFFERENT compilers all share
    ONE runtime and one thread pool."""
    name: str = "libomp"  # runtime selected by name on LLVM (``-fopenmp=<name>``)
    soname: str = "omp"  # ``-l<soname>`` for explicit linking (omp/gomp/iomp5)
    lib_dir: Optional[str] = None  # ``-L`` if the runtime is not on the default search path
    #: the OpenMP ABIs this runtime implements. libomp/libiomp5/libnvomp expose BOTH ``__kmpc_*`` and a
    #: ``GOMP_*`` compat layer; libgomp exposes only ``GOMP_*`` -- so a kmpc compiler (clang/flang/icx/
    #: nvc++) cannot use libgomp.
    provides: frozenset = frozenset({"kmpc", "gomp"})

    def compatible(self, compiler: str) -> bool:
        """True if ``compiler`` can actually LINK against THIS runtime -- which depends on how each family
        selects its runtime, not on ABI alone:

        * ``nvidia`` (nvc/nvc++/nvfortran): OpenMP only via ``-mp``, which hard-links the native
          ``libnvomp`` -- no ``-fopenmp=<lib>`` switch. Compatible with ``libnvomp`` ALONE.
        * ``intel-classic`` (icc/icpc/ifort): ``-qopenmp`` hard-links Intel's native ``libiomp5``.
          Compatible with ``libiomp5`` ALONE (which is ABI-identical to libomp, so mixing is still safe).
        * ``llvm`` (clang/flang/icx): selects the runtime BY NAME (``-fopenmp=<name>``), but the driver
          only knows :data:`_LLVM_SELECTABLE` -- so compatible with a runtime in that set whose ABI it
          emits (kmpc). That is ``libomp``/``libiomp5``; NOT ``libnvomp`` or a custom name (unreachable by
          name) and NOT ``libgomp`` (lacks ``__kmpc_*``).
        * ``gnu`` (gcc/gfortran): links any runtime explicitly with ``-l<soname>``, so compatible with any
          runtime implementing the ``gomp`` ABI it emits (all four, via the GOMP-compat layer).
        """
        fam = compiler_family(compiler)
        if fam == "nvidia":
            return self.name == "libnvomp"
        if fam == "intel-classic":
            return self.name == "libiomp5"
        if fam == "llvm":
            return self.name in _LLVM_SELECTABLE and _COMPILER_ABI["llvm"] in self.provides
        return _COMPILER_ABI["gnu"] in self.provides  # gnu

    def _check(self, compiler: str) -> None:
        if self.compatible(compiler):
            return
        fam = compiler_family(compiler)
        if fam == "nvidia":
            raise ValueError(f"{Path(compiler).name} (NVIDIA HPC) links OpenMP only through '-mp', which uses its "
                             f"native libnvomp; it cannot link {self.name}. Use the libnvomp runtime for nvc/nvc++, "
                             f"or drop the NVIDIA compiler from this runtime's sweep.")
        if fam == "intel-classic":
            raise ValueError(f"{Path(compiler).name} (classic Intel) links OpenMP through '-qopenmp', which uses its "
                             f"native libiomp5; it cannot link {self.name}. Use the libiomp5 runtime for icc/icpc, "
                             f"or drop the classic Intel compiler from this runtime's sweep.")
        if fam == "llvm":
            if _COMPILER_ABI["llvm"] not in self.provides:
                raise ValueError(f"{Path(compiler).name} emits the 'kmpc' OpenMP ABI, which {self.name} does not "
                                 f"implement (it provides {sorted(self.provides)}); libgomp is gomp-only. Use a "
                                 f"kmpc runtime (libomp/libiomp5).")
            raise ValueError(f"{Path(compiler).name} selects the OpenMP runtime by name and only knows "
                             f"{sorted(_LLVM_SELECTABLE)}; {self.name} is not name-selectable by an LLVM compiler. "
                             f"Use libomp/libiomp5, or build with gcc (which links {self.name} via -l{self.soname}).")
        raise ValueError(f"{Path(compiler).name} emits the 'gomp' OpenMP ABI, which {self.name} does not implement "
                         f"(it provides {sorted(self.provides)}). Use a gomp-capable runtime "
                         f"(libomp/libiomp5/libnvomp carry a GOMP-compat layer; libgomp is gomp-only).")

    def compile_flags(self, compiler: str) -> List[str]:
        """Flags to compile a translation unit with OpenMP against this runtime."""
        self._check(compiler)
        fam = compiler_family(compiler)
        if fam == "llvm":  # clang / clang++ / flang / icx: pick the runtime by name
            return [f"-fopenmp={self.name}"]
        if fam == "intel-classic":
            return ["-qopenmp"]
        if fam == "nvidia":  # nvc/nvc++/nvfortran: -mp links libnvomp (its native kmpc
            return ["-mp"]  # runtime); no -fopenmp=<lib> switch to force another one
        return ["-fopenmp"]  # gnu: emit GOMP calls; the runtime is fixed at link

    def link_flags(self, compiler: str) -> List[str]:
        """Flags to link a program against THIS runtime (and no other -- avoids the dual-runtime abort /
        oversubscription of mixing libgomp + libomp)."""
        self._check(compiler)
        fam = compiler_family(compiler)
        libdir = [f"-L{self.lib_dir}"] if self.lib_dir else []
        if fam == "llvm":
            return [f"-fopenmp={self.name}", *libdir]
        if fam == "intel-classic":
            return ["-qopenmp", *libdir]
        if fam == "nvidia":
            return ["-mp", *libdir]
        # gnu: link the mandated runtime EXPLICITLY rather than ``-fopenmp`` (which would pull libgomp).
        # libomp's GOMP-compat layer resolves the object's GOMP_* symbols, so a gcc lib joins the same
        # single runtime as the clang/flang libs.
        return [*libdir, f"-l{self.soname}"]


#: The popular OpenMP runtimes as ready knobs. libomp / libgomp / libiomp5 are mutually GOMP-ABI
#: compatible (libomp and libiomp5 both implement the ``GOMP_*`` entry points), so any of gcc / clang /
#: flang / icx can target any of the three -- LLVM compilers select by name (``-fopenmp=<name>``), gcc
#: emits GOMP calls and links the runtime explicitly. NVIDIA's HPC SDK ships its OWN runtime (libnvomp),
#: reachable only via nvc/nvfortran ``-mp`` and NOT interchangeable with the other three.
LIBOMP = OpenMPRuntime(name="libomp", soname="omp")  # LLVM (clang / flang) -- the default; kmpc+gomp
LIBGOMP = OpenMPRuntime(
    name="libgomp",
    soname="gomp",  # GNU (gcc / gfortran); GOMP-only -> a kmpc
    provides=frozenset({"gomp"}))  #   compiler (clang/flang/icx/nvc++) cannot use it
LIBIOMP5 = OpenMPRuntime(name="libiomp5", soname="iomp5")  # Intel (icx / icc); kmpc+gomp, ABI-compat with libomp
LIBNVOMP = OpenMPRuntime(name="libnvomp", soname="nvomp")  # NVIDIA HPC (nvc/nvc++ -mp); kmpc+gomp

#: name -> runtime, for a config/CLI knob.
OPENMP_RUNTIMES = {"libomp": LIBOMP, "libgomp": LIBGOMP, "libiomp5": LIBIOMP5, "libnvomp": LIBNVOMP}


def resolve_runtime(name: str) -> OpenMPRuntime:
    """A named runtime as an :class:`OpenMPRuntime`. Known names hit the registry; an unknown name is taken
    as ``lib<soname>`` with the default ABI set so it has a compat model. In practice a custom runtime is
    only reachable from gcc (which links it explicitly via ``-l<soname>``); LLVM compilers cannot
    name-select an unregistered runtime and icc/nvc++ use their native ones (see
    :meth:`OpenMPRuntime.compatible`)."""
    rt = OPENMP_RUNTIMES.get(name)
    if rt is not None:
        return rt
    soname = name[3:] if name.startswith("lib") else name
    return OpenMPRuntime(name=name, soname=soname)


def _lib_findable(soname: str, lib_dir: Optional[str]) -> bool:
    """True if ``lib<soname>`` can be found -- in a pinned ``lib_dir`` (any of ``.so`` / ``.a`` / ``.dylib``)
    or on the system loader search path (ldconfig cache / ``LD_LIBRARY_PATH`` / ``DYLD_*``). Shared by the
    OpenMP-runtime and vector-math-library installed-probes so the two never drift."""
    if lib_dir:
        d = Path(lib_dir)
        if any((d / f"lib{soname}{ext}").exists() for ext in (".so", ".a", ".dylib")):
            return True
    return ctypes.util.find_library(soname) is not None


def runtime_installed(rt: OpenMPRuntime) -> bool:
    """True if the runtime's shared object can be found -- on its ``lib_dir`` (if pinned) or the system
    loader search path (ldconfig cache / ``LD_LIBRARY_PATH``). NVIDIA's libnvomp lives off the default
    path, so without a ``lib_dir`` it reads as not-installed here -- which is why a config that names it
    for a non-nvhpc link gets pruned with a warning."""
    return _lib_findable(rt.soname, rt.lib_dir)


@dataclass
class ArenaConfig:
    """The DESIRED sweep: which compilers and which OpenMP runtimes to try. It is a wish list -- some
    entries may be uninstalled or ABI-incompatible with each other. :func:`prune_to_valid_combinations`
    turns it into the set of (compiler, runtime) pairs that actually work on this machine.

    Runtimes default to just ``libomp`` (PARALLEL.md mandates ONE runtime for the whole program, and
    libomp is the portable one -- name-selectable by clang/flang/icx, gomp-compat for gcc). NOTE: nvc++ and
    classic icc link only their native runtimes (libnvomp / libiomp5), so under the libomp-only default they
    are pruned with a warning -- add ``libnvomp`` / ``libiomp5`` to ``runtimes`` to sweep them. List more
    runtimes only to sweep runtime choices."""
    compilers: List[str] = field(default_factory=lambda: ["g++", "clang++", "nvc++", "icpx"])
    runtimes: List[str] = field(default_factory=lambda: ["libomp"])


@dataclass
class PrunedConfig:
    """The result of :func:`prune_to_valid_combinations`: the surviving compilers and runtimes, and the
    concrete ABI-valid, installed ``(compiler, runtime_name)`` pairs to actually build."""
    compilers: List[str]
    runtimes: List[str]
    combos: List[Tuple[str, str]]


def prune_to_valid_combinations(config: ArenaConfig,
                                *,
                                probe_compilers: bool = True,
                                probe_runtimes: bool = True) -> PrunedConfig:
    """Reduce a desired :class:`ArenaConfig` to the combinations that can actually be built here.

    Removal happens for three reasons, and EVERY removal raises a ``warnings.warn`` so a silently
    shrinking matrix is visible:

    1. a compiler not on ``PATH`` is dropped;
    2. a runtime whose library is not found on the system is dropped (see :func:`runtime_installed`);
    3. compatibility pruning to a fixpoint -- a runtime no remaining compiler can LINK is dropped (this is
       the "remove runtimes by default" step), then a compiler with no compatible remaining runtime is
       dropped. Removing one can orphan the other, so it iterates until stable. Concretely: select
       ``libgomp`` (gomp-only) and every LLVM compiler (clang++/flang/icx) plus nvc++ is discarded; give
       only ``nvc++`` and it forces its native ``libnvomp`` (never libomp/libiomp5/libgomp); give only
       classic ``icc`` and it forces ``libiomp5``.

    :param probe_compilers: check ``PATH`` (off for a pure-logic test on a machine missing the toolchains).
    :param probe_runtimes: check the filesystem for each runtime's ``.so`` (off to test ABI logic alone).
    """
    compilers = list(dict.fromkeys(config.compilers))  # de-dup, keep order
    runtimes = list(dict.fromkeys(config.runtimes))

    if probe_compilers:
        present = []
        for c in compilers:
            if shutil.which(c):
                present.append(c)
            else:
                warnings.warn(f"compiler {c!r} is not on PATH; removing it from the arena candidates "
                              f"(install it, or drop it from ArenaConfig.compilers)")
        compilers = present

    if probe_runtimes:
        found = []
        for r in runtimes:
            rt = resolve_runtime(r)
            if runtime_installed(rt):
                found.append(r)
            else:
                warnings.warn(f"OpenMP runtime {r!r} (lib{rt.soname}) is not installed on this system; removing "
                              f"it from the arena candidates (install it, or set OpenMPRuntime.lib_dir to its path)")
        runtimes = found

    # An unknown runtime name has no registered ABI. resolve_runtime assumes it is GOMP-capable so gcc can
    # link it explicitly (-l<soname>); no LLVM compiler can name-select a custom runtime and icc/nvc++ use
    # their native ones, so a custom runtime is only ever usable with gcc/gfortran. Warn once, here (after
    # the install-probe, so we do not reassure about a runtime that was just dropped as missing).
    for r in runtimes:
        if r not in OPENMP_RUNTIMES:
            rt = resolve_runtime(r)
            warnings.warn(f"OpenMP runtime {r!r} is not a known runtime; assuming lib{rt.soname} is GOMP-ABI so "
                          f"gcc/gfortran can link it via -l{rt.soname}. LLVM compilers cannot select a custom "
                          f"runtime by name and icc/nvc++ use their native runtimes, so {r!r} is only usable with "
                          f"gcc/gfortran -- verify it actually provides the GOMP_* symbols.")

    while True:  # fixpoint: drops shrink both lists monotonically, so this terminates
        keep_rt = [r for r in runtimes if any(resolve_runtime(r).compatible(c) for c in compilers)]
        for r in runtimes:
            if r not in keep_rt:
                warnings.warn(f"OpenMP runtime {r!r} is ABI-incompatible with every remaining compiler "
                              f"({compilers}); removing it. Add a compiler that emits its ABI, or a runtime "
                              f"those compilers can link.")
        keep_cc = [c for c in compilers if any(resolve_runtime(r).compatible(c) for r in keep_rt)]
        for c in compilers:
            if c not in keep_cc:
                extra = (" -- NVIDIA nvc/nvc++ needs the 'libnvomp' runtime (it links libnvomp via -mp)"
                         if compiler_family(c) == "nvidia" else "")
                warnings.warn(f"compiler {c!r} has no compatible OpenMP runtime among {keep_rt}; removing it from "
                              f"the arena candidates{extra}.")
        if keep_rt == runtimes and keep_cc == compilers:
            break
        runtimes, compilers = keep_rt, keep_cc

    combos = [(c, r) for c in compilers for r in runtimes if resolve_runtime(r).compatible(c)]
    return PrunedConfig(compilers=compilers, runtimes=runtimes, combos=combos)


#: -fveclib token (clang / flang / icx / icpx) per vector-math-library name.
_CLANG_VECLIB = {"sleef": "SLEEF", "libmvec": "libmvec", "svml": "SVML"}


@dataclass
class VectorMathLib:
    """A vectorized math library supplying SIMD implementations of elementary functions (exp/log/sin/pow/
    ...), so an autovectorized loop calls a packed routine instead of scalarizing the transcendental. A
    SEPARATE axis from the OpenMP runtime and the base flags. Support is per-compiler-family:

    * ``sleef``   (SLEEF, portable): clang/flang/icx via ``-fveclib=SLEEF`` (+ ``-lsleef``). gcc has no
      ``-fveclib`` and no ``-mveclibabi`` for SLEEF -> unsupported on gcc.
    * ``libmvec`` (glibc's vector math): clang via ``-fveclib=libmvec``; gcc uses it AUTOMATICALLY under
      ``-O3``/fast-math with an AVX ``-march`` (no compile flag), linking ``-lmvec``.
    * ``svml``    (Intel Short Vector Math Library): icx/clang via ``-fveclib=SVML``; gcc via
      ``-mveclibabi=svml``; classic icc emits SVML calls natively. Links ``-lsvml`` (Intel runtime).

    Note: the vectorizer only SUBSTITUTES these calls when the FP mode relaxes math semantics (errno /
    precision) -- that is the fast-math FP-mode axis, kept separate from this library selection.
    """
    name: str  # sleef | libmvec | svml
    soname: Optional[str]  # -l<soname> for the vector symbols (None: toolchain/glibc provides)
    lib_dir: Optional[str] = None  # -L if the library is not on the default search path

    def compatible(self, compiler: str) -> bool:
        fam = compiler_family(compiler)
        if fam == "llvm":
            return self.name in _CLANG_VECLIB
        if fam == "gnu":
            return self.name in ("libmvec", "svml")  # glibc libmvec (auto) or -mveclibabi=svml
        if fam == "intel-classic":
            return self.name == "svml"  # classic icc emits SVML natively
        return False  # nvidia: use its own -Mvect, not these

    def _check(self, compiler: str) -> None:
        if not self.compatible(compiler):
            raise ValueError(f"{Path(compiler).name} ({compiler_family(compiler)}) cannot use the {self.name} "
                             f"vector math library; try a compatible compiler or a different veclib.")

    def compile_flags(self, compiler: str) -> List[str]:
        self._check(compiler)
        fam = compiler_family(compiler)
        if fam == "llvm":
            return [f"-fveclib={_CLANG_VECLIB[self.name]}"]
        if fam == "gnu":
            return ["-mveclibabi=svml"] if self.name == "svml" else []  # libmvec is automatic on gcc
        return []  # intel-classic svml: native

    def link_flags(self, compiler: str) -> List[str]:
        self._check(compiler)
        if not self.soname:
            return []
        libdir = [f"-L{self.lib_dir}"] if self.lib_dir else []
        return [*libdir, f"-l{self.soname}"]


SLEEF = VectorMathLib(name="sleef", soname="sleef")
LIBMVEC = VectorMathLib(name="libmvec", soname="mvec")  # glibc's libmvec
SVML = VectorMathLib(name="svml", soname="svml")  # Intel SVML runtime

#: name -> vector-math library, for a config/CLI knob.
VECTOR_LIBS = {"sleef": SLEEF, "libmvec": LIBMVEC, "svml": SVML}


def vectorlib_installed(vl: VectorMathLib) -> bool:
    """True if the vector library's shared object is findable (shares :func:`_lib_findable` with
    :func:`runtime_installed`). A ``soname``-less entry (toolchain/glibc-provided) is always present."""
    if not vl.soname:
        return True
    return _lib_findable(vl.soname, vl.lib_dir)


# TODO(blas): add a BLAS/LAPACK library axis (openblas / mkl / blis / nvpl / accelerate) the same way --
# a linkable-library knob with per-library link flags + an installed-probe, feeding the same prune model.
# Discovery already exists (nestforge.arena.discover_blas_libraries); what's missing is threading a chosen
# BLAS into the owned build's link line (for library-node expansions that lower to gemm/gemv) and a
# compat/prune step. Kept as a TODO here so the vector-math axis lands first.


def dace_runtime_include() -> Path:
    """The ``-I`` directory holding DaCe's runtime headers (``dace/runtime/include``)."""
    inc = Path(dace.__file__).parent / "runtime" / "include"
    if not inc.is_dir():
        raise FileNotFoundError(f"DaCe runtime include not found at {inc}")
    return inc


@dataclass
class _Param:
    name: str
    ctype: object  # a ctypes type
    is_pointer: bool


def _parse_params(param_str: str) -> List[_Param]:
    """Parse a C parameter list into typed params. Skips the leading ``N_state_t *__state`` handle."""
    params: List[_Param] = []
    for raw in _split_params(param_str):
        tok = raw.replace("__restrict__", "").replace("const", "").strip()
        if not tok or tok.endswith("_state_t *__state") or tok.endswith("_state_t* __state"):
            continue
        is_ptr = "*" in tok
        name = re.split(r"[\s*]+", tok)[-1]
        base = tok[:tok.rfind(name)].replace("*", "").strip()
        if is_ptr:
            params.append(_Param(name, ctypes.POINTER(_C_PTR.get(base, ctypes.c_double)), True))
        else:
            params.append(_Param(name, _C_SCALAR.get(base, ctypes.c_int64), False))
    return params


def _split_params(param_str: str) -> List[str]:
    """Split a parameter list on top-level commas (none nest here, but be safe)."""
    out, depth, cur = [], 0, ""
    for ch in param_str:
        if ch in "(<":
            depth += 1
        elif ch in ")>":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def _signature(code: str, symbol: str) -> str:
    """The parameter list of ``symbol(...)`` in ``code`` (first declaration)."""
    m = re.search(rf"{symbol}\s*\((.*?)\)", code, re.S)
    if not m:
        raise LookupError(f"entry point {symbol} not found in generated code")
    return m.group(1)


@dataclass
class BuiltSDFG:
    """A nest-forge-built DaCe ``.so`` with its entry points bound and init/exit managed."""
    name: str
    so_path: Path
    _lib: ctypes.CDLL
    _init_params: List[_Param]
    _prog_params: List[_Param]
    #: wall time of the OPTIMIZATION phase (DaCe codegen: the optimizing passes + C++ emission +
    #: source-tree layout) -- distinct from the toolchain compile below.
    codegen_seconds: float = 0.0
    #: wall time of the post-optimization COMPILE (the compiler/linker subprocess turning the generated
    #: C++ into the ``.so``); reflects whether external linking was used (see ``link_external``).
    compile_seconds: float = 0.0
    _handle: Optional[ctypes.c_void_p] = field(default=None, repr=False)

    def _init(self, sizes: Dict[str, int]) -> None:
        fn = self._lib[f"__dace_init_{self.name}"]  # ctypes CDLL indexing (not getattr) to bind the entry point
        fn.restype = ctypes.c_void_p
        fn.argtypes = [p.ctype for p in self._init_params]
        # Use each parameter's OWN ctype -- DaCe types a size symbol as int / int64_t per its declared
        # dtype, so a hardcoded width mismatches (jacobi's ``int N`` vs gemm's ``int64_t NI``).
        self._handle = ctypes.c_void_p(fn(*[p.ctype(int(sizes[p.name])) for p in self._init_params]))

    def program(self, buffers: Dict[str, np.ndarray], sizes: Dict[str, int]) -> None:
        """Call ``__program_N(handle, args...)`` once, in place (init must have run)."""
        fn = self._lib[f"__program_{self.name}"]
        fn.restype = None
        fn.argtypes = [ctypes.c_void_p] + [p.ctype for p in self._prog_params]
        args = [self._handle]
        for p in self._prog_params:
            if p.is_pointer:
                args.append(buffers[p.name].ctypes.data_as(p.ctype))
            elif p.name in buffers:  # a DaCe Scalar passed by value
                args.append(p.ctype(buffers[p.name].item()))
            else:  # a size symbol
                args.append(p.ctype(int(sizes[p.name])))
        fn(*args)

    def close(self) -> None:
        if self._handle is not None:
            fn = self._lib[f"__dace_exit_{self.name}"]
            fn.restype = ctypes.c_int
            fn.argtypes = [ctypes.c_void_p]
            fn(self._handle)
            self._handle = None

    def run(self, buffers: Dict[str, np.ndarray], sizes: Dict[str, int]) -> None:
        """One-shot init -> program -> exit (for correctness; for timing, init once + loop program)."""
        self._init(sizes)
        try:
            self.program(buffers, sizes)
        finally:
            self.close()


def generate_program_folder(sdfg: dace.SDFG, out_dir: Path) -> Tuple[Path, str]:
    """Lay out DaCe's full compilable source tree (``src/cpu/<name>.cpp`` + ``include/`` with the
    generated headers) via DaCe's own ``generate_program_folder`` -- so the relative
    ``#include "../../include/hash.h"`` resolves -- WITHOUT letting DaCe compile it. We compile it.

    :returns: (the C++ Frame source path, sdfg name).
    """
    from dace.codegen import codegen, compiler as dace_compiler
    out_dir.mkdir(parents=True, exist_ok=True)
    code_objects = codegen.generate_code(sdfg)
    folder = Path(dace_compiler.generate_program_folder(sdfg, code_objects, str(out_dir)))
    frame = folder / "src" / "cpu" / f"{sdfg.name}.cpp"
    if not frame.exists():  # fall back to whatever CPU Frame the layout produced
        frame = next(folder.glob("src/cpu/*.cpp"))
    return frame, sdfg.name


def _include_flags(folder: Path) -> List[str]:
    """Header search paths: the generated ``include/`` (hash.h + copied dace headers) and DaCe's
    runtime include (angle-bracket ``<dace/...>``)."""
    return [f"-I{folder / 'include'}", f"-I{dace_runtime_include()}"]


def _clang_major_via_preprocessor(compiler: str) -> Optional[int]:
    """The underlying clang major of an LLVM compiler that hides it from ``--version`` (icx/icpx/ifx print
    an oneAPI banner, not ``clang version``), by asking the preprocessor for ``__clang_major__``. ``None``
    if it cannot be determined."""
    try:
        p = subprocess.run([compiler, "-dM", "-E", "-x", "c", "/dev/null"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"#define __clang_major__ (\d+)", p.stdout)
    return int(m.group(1)) if m else None


@functools.lru_cache(maxsize=None)
def _compiler_version(compiler: str) -> Tuple[int, int]:
    """The compiler's ``(major, minor)`` version, parsed from ``<compiler> --version`` (cached per
    invocation string). Used to gate features whose support depends on the toolchain version -- fast
    linkers and fat-LTO objects. Returns ``(0, 0)`` if the compiler cannot be run or the version cannot be
    determined: a CONSERVATIVE "assume old" that DISABLES the version-gated features (so we never emit a
    flag the compiler may reject) rather than optimistically enabling them."""
    try:
        p = subprocess.run([compiler, "--version"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return (0, 0)
    out = f"{p.stdout}\n{p.stderr}"
    fam = compiler_family(compiler)
    if fam in ("llvm", "intel-classic"):
        m = re.search(r"clang version (\d+)\.(\d+)", out)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        # icx/icpx/ifx (oneAPI) hide the clang version behind their own banner -- ask the preprocessor,
        # so the >=18 fat-LTO gate reflects the REAL clang base (an old oneAPI must not be assumed modern).
        cmaj = _clang_major_via_preprocessor(compiler)
        return (cmaj, 0) if cmaj is not None else (0, 0)
    if fam == "gnu":
        m = re.search(r"\bg(?:cc|\+\+)?[^\n]*?\b(\d+)\.(\d+)\.\d+\b", out) or re.search(r"\b(\d+)\.(\d+)\.\d+\b", out)
        if m:
            return (int(m.group(1)), int(m.group(2)))
    return (0, 0)


def _ar_for(compiler: str) -> str:
    """The ``ar`` that understands this compiler's object files -- the LTO-plugin-aware wrapper (``gcc-ar``
    for gcc, ``llvm-ar`` for clang/flang/icx) when present, so archiving an ``-flto`` object keeps it
    linkable; plain ``ar`` otherwise. Classic icc is not fat-LTO'd here (see :func:`_fat_lto_flags`), so its
    object is plain and plain ``ar`` suffices."""
    cand = {"gnu": "gcc-ar", "llvm": "llvm-ar"}.get(compiler_family(compiler), "ar")
    return cand if shutil.which(cand) else "ar"


#: Minimum compiler ``(major, minor)`` that accepts ``-fuse-ld=<linker>``, per family. Absent (family,
#: linker) pairs are treated as unsupported. mold needs gcc>=12.1 / clang>=12; lld is old on both; gold is
#: effectively always there via binutils. (icx/icpx report as modern LLVM, so they clear the clang gates.)
_LINKER_MIN: Dict[str, Dict[str, Tuple[int, int]]] = {
    "mold": {
        "gnu": (12, 1),
        "llvm": (12, 0)
    },
    "lld": {
        "gnu": (9, 0),
        "llvm": (3, 0),
        "intel-classic": (0, 0)
    },
    "gold": {
        "gnu": (0, 0),
        "llvm": (3, 0),
        "intel-classic": (0, 0)
    },
}


def _linker_supported(compiler: str, linker: str) -> bool:
    """True if ``compiler`` is new enough to accept ``-fuse-ld=<linker>``. Version-gated (see
    :data:`_LINKER_MIN`) so we never hand an old gcc/clang a ``-fuse-ld=mold`` it rejects."""
    fam = compiler_family(compiler)
    floor = _LINKER_MIN.get(linker, {}).get(fam)
    return floor is not None and _compiler_version(compiler) >= floor


def _fat_lto_flags(compiler: str) -> List[str]:
    """Flags to compile a FAT-LTO object (LTO bitcode + real machine code) for this compiler, or ``[]`` if
    it cannot -- in which case a warning is emitted and the node library is archived without LTO. gcc has
    fat LTO since ~4.8; clang only since 18; ``icx``/``icpx`` (modern LLVM) qualify; classic icc uses
    ``-ipo`` (a different, non-fat model) and NVIDIA has no fat-LTO, so both archive without it."""
    fam = compiler_family(compiler)
    if fam == "gnu":
        return ["-flto", "-ffat-lto-objects"]
    if fam == "llvm" and _compiler_version(compiler) >= (18, 0):
        return ["-flto", "-ffat-lto-objects"]
    reason = ("clang < 18 has no -ffat-lto-objects" if fam == "llvm" else
              "classic icc uses -ipo, not fat LTO" if fam == "intel-classic" else "no fat-LTO support")
    warnings.warn(f"{Path(compiler).name}: {reason}; archiving the node library without LTO "
                  f"(the .so still links from real machine code and runs correctly).")
    return []


def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd[:2])} ...\n{p.stderr[-2000:]}")


#: Fast alternative linkers, FASTEST FIRST. The default ``bfd`` ``ld`` is always present and is the
#: fallback, so it is not listed here. Install more with e.g. ``apt install mold`` (fastest) or
#: ``apt install binutils-gold``.
_FAST_LINKERS = ("mold", "lld", "gold")


def available_linkers() -> Dict[str, str]:
    """The fast alternative linkers installed on this system, fastest first: name (the ``-fuse-ld=<name>``
    token) -> backing binary path. The default ``bfd`` ld is not listed (it is always the fallback). Use
    it to report what is available and what could be installed for a faster link."""
    found: Dict[str, str] = {}
    for ld in _FAST_LINKERS:
        p = shutil.which(ld) or shutil.which(f"ld.{ld}")
        if p:
            found[ld] = p
    return found


def _fastest_linker(compiler: str) -> List[str]:
    """``-fuse-ld=<linker>`` for the fastest installed linker THIS compiler is new enough to accept
    (mold > lld > gold), or ``[]`` if none qualify / only the default ``ld`` is present. NVIDIA's
    nvc/nvc++ has no ``-fuse-ld`` switch, so it always keeps its default. The version gate (see
    :func:`_linker_supported`) is what stops an old gcc/clang being handed ``-fuse-ld=mold`` it rejects."""
    if compiler_family(compiler) == "nvidia":
        return []
    for ld in available_linkers():  # dict preserves the fastest-first order of _FAST_LINKERS
        if _linker_supported(compiler, ld):
            return [f"-fuse-ld={ld}"]
    return []


@dataclass
class BuildOptions:
    """Toolchain + optimization knobs for the owned build, grouped so :func:`build_sdfg` /
    :func:`compare_link_modes` take one options object instead of a long parameter list. Each axis is
    independent: the base ``flags``, the ``openmp`` runtime, the ``veclib``, and the link mode."""
    compiler: str = DEFAULT_COMPILER
    flags: Optional[List[str]] = None  # None -> DEFAULT_FLAGS
    expand_libnodes: bool = False  # expand library nodes to loops first (the "without libnodes" variant)
    openmp: Optional[OpenMPRuntime] = None  # the one mandated runtime to link (per-compiler flags)
    link_external: bool = False  # link the nest as a separate static .a (else a monolithic single TU)
    lto: bool = False  # enable LTO: -flto (monolithic) / fat-LTO object in the .a (external) -- applies to both
    veclib: Optional[VectorMathLib] = None  # SLEEF / libmvec / SVML, a separate axis from flags/openmp

    def resolved_flags(self) -> List[str]:
        return list(self.flags if self.flags is not None else DEFAULT_FLAGS)


def _compile(frame: Path, folder: Path, name: str, opts: BuildOptions) -> Tuple[Path, float]:
    """Compile the generated frame into ``lib<name>.so`` and return (path, wall_seconds -- the toolchain
    work ONLY: capability probes and PATH lookups are resolved before the clock starts, so the returned
    time is comparable across link modes).

    Two link modes -- the axis behind "compile time WITH vs WITHOUT external linking":

    * ``opts.link_external=False`` (monolithic): a single translation unit, so the compiler inlines freely.
      Without a mandated OpenMP runtime it is one compile+link command. WITH a runtime it is split into a
      ``-c`` compile then a ``-shared`` link, so EXACTLY that runtime is linked: a single ``g++ -fopenmp
      ... -lomp`` command would auto-link libgomp (from ``-fopenmp``) alongside the explicit ``-lomp`` and
      load two runtimes; compiling with ``-fopenmp`` then linking with ``-lomp`` only links one. ``opts.lto``
      adds ``-flto``.
    * ``opts.link_external=True``: the static-node-library path -- compile the frame to an object, archive
      it into ``lib<name>_nest.a``, then link the ``.so`` from that archive (``--whole-archive`` keeps every
      symbol) via the fastest available linker. ``opts.lto`` makes the archived object a FAT-LTO object
      (``-flto -ffat-lto-objects`` -- LTO bitcode AND real machine code, when the compiler supports it; see
      :func:`_fat_lto_flags`) so the ``.a`` is LTO-ready for a future driver link that spans node
      boundaries; without ``opts.lto`` the object is plain (the cheaper default, since there is no cross-TU
      consumer yet). Either way the ``.so`` is linked from the object's REAL code (never ``-flto`` at this
      link) with ``--export-dynamic``, so the extern-C entry points (``__dace_init_*`` etc.) survive.
    """
    compiler = opts.compiler
    flags = opts.resolved_flags()
    inc = _include_flags(folder)
    omp_c = opts.openmp.compile_flags(compiler) if opts.openmp else []
    omp_l = opts.openmp.link_flags(compiler) if opts.openmp else []
    vec_c = opts.veclib.compile_flags(compiler) if opts.veclib else []
    vec_l = opts.veclib.link_flags(compiler) if opts.veclib else []
    so = folder / f"lib{name}.so"
    cflags = [f for f in flags if f != "-shared"]  # -shared is a link-only flag; drop it for any -c step
    obj = folder / f"{name}.o"
    lto_f = ["-flto"] if opts.lto else []

    if not opts.link_external and not opts.openmp:
        # No mandated runtime: one compile+link command is safe (nothing for a second runtime to sneak in).
        cmd = [compiler, *flags, *lto_f, *vec_c, *vec_l, *inc, str(frame), "-o", str(so)]
        t0 = time.perf_counter()
        _run(cmd)
    elif not opts.link_external:
        # Monolithic but with a mandated runtime: split compile from link so ONLY that runtime is linked
        # (single TU, so inlining is unaffected). See the docstring for the gnu dual-runtime trap.
        compile_cmd = [compiler, *cflags, *lto_f, "-c", *omp_c, *vec_c, *inc, str(frame), "-o", str(obj)]
        link_cmd = [compiler, "-shared", *cflags, *lto_f, str(obj), *omp_l, *vec_l, "-o", str(so)]
        t0 = time.perf_counter()
        _run(compile_cmd)
        _run(link_cmd)
    else:
        # External static-node-library path. Resolve everything that is NOT toolchain work -- fat-LTO
        # capability probe (runs the compiler once, cached), archiver, fastest linker, stale-archive
        # cleanup -- BEFORE the clock starts, so compile_seconds is just the compile+archive+link work.
        lto_c = _fat_lto_flags(compiler) if opts.lto else []
        ar = _ar_for(compiler)
        ld = _fastest_linker(compiler)
        archive = folder / f"lib{name}_nest.a"
        if archive.exists():
            archive.unlink()  # ar r APPENDS; start clean so a rebuild doesn't stack stale members
        compile_cmd = [compiler, *cflags, *lto_c, "-c", *omp_c, *vec_c, *inc, str(frame), "-o", str(obj)]
        ar_cmd = [ar, "rcs", str(archive), str(obj)]
        # Link the .so from the object's REAL code (NOT -flto) so the entry points survive + export.
        link_cmd = [
            compiler, "-shared", *cflags, *ld, "-Wl,--export-dynamic", "-Wl,--whole-archive",
            str(archive), "-Wl,--no-whole-archive", *omp_l, *vec_l, "-o",
            str(so)
        ]
        t0 = time.perf_counter()
        _run(compile_cmd)
        _run(ar_cmd)
        _run(link_cmd)
    return so, time.perf_counter() - t0


def build_sdfg(sdfg: dace.SDFG, out_dir: Path, opts: Optional[BuildOptions] = None) -> BuiltSDFG:
    """Generate + compile + link an SDFG ourselves; return a :class:`BuiltSDFG` ready to call, carrying
    the ``codegen_seconds`` (optimization) and ``compile_seconds`` (post-optimization toolchain) times.

    :param opts: the toolchain + optimization knobs (:class:`BuildOptions`); ``None`` uses all defaults
        (g++, ``DEFAULT_FLAGS``, monolithic link, no OpenMP/veclib).
    """
    import copy
    opts = opts or BuildOptions()
    t_opt = time.perf_counter()
    sdfg = copy.deepcopy(sdfg)
    if opts.expand_libnodes:
        sdfg.expand_library_nodes()
    frame, name = generate_program_folder(sdfg, out_dir)
    folder = frame.parent.parent.parent  # <out>/src/cpu/x.cpp -> <out>
    codegen_seconds = time.perf_counter() - t_opt

    code = frame.read_text()
    init_params = _parse_params(_signature(code, f"__dace_init_{name}"))
    prog_params = _parse_params(_signature(code, f"__program_{name}"))

    so, compile_seconds = _compile(frame, folder, name, opts)
    return BuiltSDFG(name=name,
                     so_path=so,
                     _lib=ctypes.CDLL(str(so)),
                     _init_params=init_params,
                     _prog_params=prog_params,
                     codegen_seconds=codegen_seconds,
                     compile_seconds=compile_seconds)


@dataclass
class LinkTimings:
    """Optimization time and the two post-optimization compile times isolated on ONE codegen."""
    codegen_seconds: float  # the optimization (DaCe codegen) phase, run once
    compile_seconds_monolithic: float  # WITHOUT external linking (single TU)
    compile_seconds_external: float  # WITH external linking (static .a -> .so)


def compare_link_modes(sdfg: dace.SDFG, out_dir: Path, opts: Optional[BuildOptions] = None) -> LinkTimings:
    """Generate the code ONCE (one optimization pass), then compile that same frame both monolithically
    and via an external static library, so ``compile_seconds`` is the only thing that differs. Returns
    the optimization time plus both post-optimization compile times. ``opts``' link mode is overridden per
    build; its other axes (compiler / flags / openmp / veclib / expand_libnodes / lto) apply to both."""
    import copy
    opts = opts or BuildOptions()
    t_opt = time.perf_counter()
    sdfg = copy.deepcopy(sdfg)
    if opts.expand_libnodes:  # mirror build_sdfg: compare the SAME (expanded) SDFG the caller configured
        sdfg.expand_library_nodes()
    frame, name = generate_program_folder(sdfg, out_dir)
    folder = frame.parent.parent.parent
    codegen_seconds = time.perf_counter() - t_opt
    _, mono = _compile(frame, folder, name, replace(opts, link_external=False))
    _, ext = _compile(frame, folder, name, replace(opts, link_external=True))
    return LinkTimings(codegen_seconds=codegen_seconds, compile_seconds_monolithic=mono, compile_seconds_external=ext)
