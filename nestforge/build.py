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
import re
import shutil
import subprocess
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import dace

#: ctypes type per C scalar type appearing in a DaCe entry-point signature.
_C_SCALAR = {"int32_t": ctypes.c_int32, "int64_t": ctypes.c_int64, "int": ctypes.c_int,
             "float": ctypes.c_float, "double": ctypes.c_double, "bool": ctypes.c_bool}
_C_PTR = {"float": ctypes.c_float, "double": ctypes.c_double,
          "int32_t": ctypes.c_int32, "int64_t": ctypes.c_int64}

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


@dataclass
class OpenMPRuntime:
    """The single OpenMP runtime the whole program links against -- a SEPARATE, configurable flag axis,
    not folded into the base flags (PARALLEL.md mandates one runtime for every node library and the
    driver). ``libomp`` is the default because it is the most portable: LLVM/Clang/flang select it by
    name (``-fopenmp=libomp`` -- the user's example), it is ABI-compatible with Intel's libiomp5, AND it
    implements the ``GOMP_*`` ABI, so a GCC-compiled object (which emits ``GOMP_*`` calls) resolves
    against it too. That is what lets a set of node libraries built with DIFFERENT compilers all share
    ONE runtime and one thread pool."""
    name: str = "libomp"                     # runtime selected by name on LLVM (``-fopenmp=<name>``)
    soname: str = "omp"                       # ``-l<soname>`` for explicit linking (omp/gomp/iomp5)
    lib_dir: Optional[str] = None             # ``-L`` if the runtime is not on the default search path
    #: the OpenMP ABIs this runtime implements. libomp/libiomp5/libnvomp expose BOTH ``__kmpc_*`` and a
    #: ``GOMP_*`` compat layer; libgomp exposes only ``GOMP_*`` -- so a kmpc compiler (clang/flang/icx/
    #: nvc++) cannot use libgomp.
    provides: frozenset = frozenset({"kmpc", "gomp"})

    def compatible(self, compiler: str) -> bool:
        """True if this runtime implements the ABI ``compiler`` emits (else linking would leave the
        OpenMP entry points unresolved -- e.g. nvc++/clang emit ``__kmpc_*`` which libgomp lacks)."""
        return _COMPILER_ABI[compiler_family(compiler)] in self.provides

    def _check(self, compiler: str) -> None:
        if not self.compatible(compiler):
            abi = _COMPILER_ABI[compiler_family(compiler)]
            raise ValueError(
                f"{Path(compiler).name} emits the {abi!r} OpenMP ABI, which {self.name} does not "
                f"implement (it provides {sorted(self.provides)}). Use a {abi}-capable runtime "
                f"(libomp/libiomp5/libnvomp are kmpc+gomp; libgomp is gomp-only).")

    def compile_flags(self, compiler: str) -> List[str]:
        """Flags to compile a translation unit with OpenMP against this runtime."""
        self._check(compiler)
        fam = compiler_family(compiler)
        if fam == "llvm":                     # clang / clang++ / flang / icx: pick the runtime by name
            return [f"-fopenmp={self.name}"]
        if fam == "intel-classic":
            return ["-qopenmp"]
        if fam == "nvidia":                   # nvc/nvc++/nvfortran: -mp links libnvomp (its native kmpc
            return ["-mp"]                    # runtime); no -fopenmp=<lib> switch to force another one
        return ["-fopenmp"]                   # gnu: emit GOMP calls; the runtime is fixed at link

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
LIBOMP = OpenMPRuntime(name="libomp", soname="omp")         # LLVM (clang / flang) -- the default; kmpc+gomp
LIBGOMP = OpenMPRuntime(name="libgomp", soname="gomp",      # GNU (gcc / gfortran); GOMP-only -> a kmpc
                        provides=frozenset({"gomp"}))       #   compiler (clang/flang/icx/nvc++) cannot use it
LIBIOMP5 = OpenMPRuntime(name="libiomp5", soname="iomp5")   # Intel (icx / icc); kmpc+gomp, ABI-compat with libomp
LIBNVOMP = OpenMPRuntime(name="libnvomp", soname="nvomp")   # NVIDIA HPC (nvc/nvc++ -mp); kmpc+gomp

#: name -> runtime, for a config/CLI knob.
OPENMP_RUNTIMES = {"libomp": LIBOMP, "libgomp": LIBGOMP, "libiomp5": LIBIOMP5, "libnvomp": LIBNVOMP}


def resolve_runtime(name: str) -> OpenMPRuntime:
    """A named runtime as an :class:`OpenMPRuntime`. Known names hit the registry; an unknown name is
    taken as ``lib<soname>`` and assumed kmpc+gomp (the portable default) so a custom runtime still has
    a compat model."""
    rt = OPENMP_RUNTIMES.get(name)
    if rt is not None:
        return rt
    soname = name[3:] if name.startswith("lib") else name
    return OpenMPRuntime(name=name, soname=soname)


def runtime_installed(rt: OpenMPRuntime) -> bool:
    """True if the runtime's shared object can be found -- on its ``lib_dir`` (if pinned) or the system
    loader search path (ldconfig cache / ``LD_LIBRARY_PATH``). NVIDIA's libnvomp lives off the default
    path, so without a ``lib_dir`` it reads as not-installed here -- which is why a config that names it
    for a non-nvhpc link gets pruned with a warning."""
    if rt.lib_dir:
        d = Path(rt.lib_dir)
        if any((d / f"lib{rt.soname}{ext}").exists() for ext in (".so", ".a", ".dylib")):
            return True
    return ctypes.util.find_library(rt.soname) is not None


@dataclass
class ArenaConfig:
    """The DESIRED sweep: which compilers and which OpenMP runtimes to try. It is a wish list -- some
    entries may be uninstalled or ABI-incompatible with each other. :func:`prune_to_valid_combinations`
    turns it into the set of (compiler, runtime) pairs that actually work on this machine.

    Runtimes default to just ``libomp`` (PARALLEL.md mandates ONE runtime for the whole program, and
    libomp is the portable one -- kmpc for clang/flang/icx/nvc++, gomp-compat for gcc). List more only to
    sweep runtime choices."""
    compilers: List[str] = field(default_factory=lambda: ["g++", "clang++", "nvc++", "icpx"])
    runtimes: List[str] = field(default_factory=lambda: ["libomp"])


@dataclass
class PrunedConfig:
    """The result of :func:`prune_to_valid_combinations`: the surviving compilers and runtimes, and the
    concrete ABI-valid, installed ``(compiler, runtime_name)`` pairs to actually build."""
    compilers: List[str]
    runtimes: List[str]
    combos: List[Tuple[str, str]]


def prune_to_valid_combinations(config: ArenaConfig, *, probe_compilers: bool = True,
                                probe_runtimes: bool = True) -> PrunedConfig:
    """Reduce a desired :class:`ArenaConfig` to the combinations that can actually be built here.

    Removal happens for three reasons, and EVERY removal raises a ``warnings.warn`` so a silently
    shrinking matrix is visible:

    1. a compiler not on ``PATH`` is dropped;
    2. a runtime whose library is not found on the system is dropped (see :func:`runtime_installed`);
    3. ABI pruning to a fixpoint -- a runtime compatible with none of the remaining compilers is dropped
       (this is the "remove runtimes by default" step), then a compiler with no compatible remaining
       runtime is dropped. Removing one can orphan the other, so it iterates until stable. Concretely:
       select ``libgomp`` (gomp-only) and every kmpc compiler (clang++/flang/icx/**nvc++**) is discarded;
       keep ``nvc++`` and it forces a kmpc runtime (libomp/libiomp5/libnvomp), never libgomp.

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
                warnings.warn(f"compiler {c!r} is not on PATH; removing it from the arena candidates")
        compilers = present

    if probe_runtimes:
        found = []
        for r in runtimes:
            rt = resolve_runtime(r)
            if runtime_installed(rt):
                found.append(r)
            else:
                warnings.warn(f"OpenMP runtime {r!r} (lib{rt.soname}) is not installed on this system; "
                              f"removing it from the arena candidates")
        runtimes = found

    while True:  # fixpoint: drops shrink both lists monotonically, so this terminates
        keep_rt = [r for r in runtimes if any(resolve_runtime(r).compatible(c) for c in compilers)]
        for r in runtimes:
            if r not in keep_rt:
                warnings.warn(f"OpenMP runtime {r!r} is ABI-incompatible with every remaining compiler "
                              f"({compilers}); removing it")
        keep_cc = [c for c in compilers if any(resolve_runtime(r).compatible(c) for r in keep_rt)]
        for c in compilers:
            if c not in keep_cc:
                warnings.warn(f"compiler {c!r} has no compatible OpenMP runtime among {keep_rt}; "
                              f"removing it from the arena candidates")
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
    name: str                          # sleef | libmvec | svml
    soname: Optional[str]              # -l<soname> for the vector symbols (None: toolchain/glibc provides)
    lib_dir: Optional[str] = None      # -L if the library is not on the default search path

    def compatible(self, compiler: str) -> bool:
        fam = compiler_family(compiler)
        if fam == "llvm":
            return self.name in _CLANG_VECLIB
        if fam == "gnu":
            return self.name in ("libmvec", "svml")  # glibc libmvec (auto) or -mveclibabi=svml
        if fam == "intel-classic":
            return self.name == "svml"               # classic icc emits SVML natively
        return False                                 # nvidia: use its own -Mvect, not these

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
LIBMVEC = VectorMathLib(name="libmvec", soname="mvec")   # glibc's libmvec
SVML = VectorMathLib(name="svml", soname="svml")         # Intel SVML runtime

#: name -> vector-math library, for a config/CLI knob.
VECTOR_LIBS = {"sleef": SLEEF, "libmvec": LIBMVEC, "svml": SVML}


def vectorlib_installed(vl: VectorMathLib) -> bool:
    """True if the vector library's shared object is findable (mirrors :func:`runtime_installed`). A
    ``soname``-less entry (toolchain-provided) is always considered present."""
    if not vl.soname:
        return True
    if vl.lib_dir and any((Path(vl.lib_dir) / f"lib{vl.soname}{e}").exists() for e in (".so", ".a")):
        return True
    return ctypes.util.find_library(vl.soname) is not None


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
    _scalar_names: set
    #: wall time of the OPTIMIZATION phase (DaCe codegen: the optimizing passes + C++ emission +
    #: source-tree layout) -- distinct from the toolchain compile below.
    codegen_seconds: float = 0.0
    #: wall time of the post-optimization COMPILE (the compiler/linker subprocess turning the generated
    #: C++ into the ``.so``); reflects whether external linking was used (see ``link_external``).
    compile_seconds: float = 0.0
    _handle: Optional[ctypes.c_void_p] = field(default=None, repr=False)

    def _init(self, sizes: Dict[str, int]) -> None:
        fn = getattr(self._lib, f"__dace_init_{self.name}")
        fn.restype = ctypes.c_void_p
        fn.argtypes = [p.ctype for p in self._init_params]
        # Use each parameter's OWN ctype -- DaCe types a size symbol as int / int64_t per its declared
        # dtype, so a hardcoded width mismatches (jacobi's ``int N`` vs gemm's ``int64_t NI``).
        self._handle = ctypes.c_void_p(fn(*[p.ctype(int(sizes[p.name])) for p in self._init_params]))

    def program(self, buffers: Dict[str, np.ndarray], sizes: Dict[str, int]) -> None:
        """Call ``__program_N(handle, args...)`` once, in place (init must have run)."""
        fn = getattr(self._lib, f"__program_{self.name}")
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
            fn = getattr(self._lib, f"__dace_exit_{self.name}")
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


def _ar_for(compiler: str) -> str:
    """The ``ar`` that understands this compiler's object files -- the LTO-plugin-aware wrapper
    (``gcc-ar`` / ``llvm-ar``) when present, so archiving ``-flto`` objects keeps them linkable; plain
    ``ar`` otherwise."""
    fam = compiler_family(compiler)
    cand = "gcc-ar" if fam == "gnu" else "llvm-ar" if fam in ("llvm", "intel-classic") else "ar"
    return cand if shutil.which(cand) else "ar"


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
    """``-fuse-ld=<linker>`` for the fastest installed linker (mold > lld > gold), or ``[]`` if only the
    default ``ld`` is present. NVIDIA's nvc/nvc++ has no ``-fuse-ld`` switch, so it keeps its default."""
    if compiler_family(compiler) == "nvidia":
        return []
    for ld in available_linkers():  # dict preserves the fastest-first order of _FAST_LINKERS
        return [f"-fuse-ld={ld}"]
    return []


def _compile(frame: Path, folder: Path, name: str, compiler: str, flags: List[str],
             openmp: Optional[OpenMPRuntime], link_external: bool, lto: bool,
             veclib: Optional[VectorMathLib] = None) -> Tuple[Path, float]:
    """Compile the generated frame into ``lib<name>.so`` and return (path, wall_seconds).

    Two link modes -- the axis behind "compile time WITH vs WITHOUT external linking":

    * ``link_external=False`` (monolithic): one compiler invocation compiles + links the ``.so`` from a
      single translation unit. The compiler sees everything and inlines freely. ``lto`` optionally adds
      ``-flto`` here too.
    * ``link_external=True``: the static-node-library path -- compile the frame to an object, archive it
      into ``lib<name>_nest.a``, then link the ``.so`` from that archive (``--whole-archive`` keeps every
      symbol). Built to be as fast + as optimized as the toolchain allows: EVERY optimization flag is
      propagated, the object is compiled with FAT LTO (``-flto -ffat-lto-objects`` -- LTO bitcode AND real
      machine code) so the ``.a`` is LTO-ready for a future driver link that spans node boundaries, and
      the loadable ``.so`` links via the fastest available linker. The ``.so`` is linked from the object's
      REAL code (no ``-flto`` at this link) with ``--export-dynamic``: a single-object LTO link would DCE
      the unreferenced extern-C entry points (``__dace_init_*`` etc.) and leave them undefined -- the
      cross-TU LTO win belongs to the eventual driver link, not to this wrapper ``.so``.
    """
    inc = _include_flags(folder)
    omp_c = openmp.compile_flags(compiler) if openmp else []
    omp_l = openmp.link_flags(compiler) if openmp else []
    vec_c = veclib.compile_flags(compiler) if veclib else []
    vec_l = veclib.link_flags(compiler) if veclib else []
    so = folder / f"lib{name}.so"
    t0 = time.perf_counter()
    if not link_external:
        lto_f = ["-flto"] if lto else []
        _run([compiler, *flags, *lto_f, *omp_c, *omp_l, *vec_c, *vec_l, *inc, str(frame), "-o", str(so)])
    else:
        opt = [f for f in flags if f != "-shared"]  # every opt flag (-O3/-march/-std/-fPIC); no -shared with -c
        fam = compiler_family(compiler)
        if fam in ("gnu", "llvm", "intel-classic"):
            lto_c = ["-flto", "-ffat-lto-objects"]  # bitcode (for the driver link) + real code (for our .so)
        else:
            lto_c = []  # nvc/nvc++ has no -ffat-lto-objects; archive the node lib without LTO
            warnings.warn(f"{Path(compiler).name} has no fat-LTO support; archiving the node lib without LTO")
        obj = folder / f"{name}.o"
        _run([compiler, *opt, *lto_c, "-c", *omp_c, *vec_c, *inc, str(frame), "-o", str(obj)])
        archive = folder / f"lib{name}_nest.a"
        if archive.exists():
            archive.unlink()  # ar r APPENDS; start clean so a rebuild doesn't stack stale members
        _run([_ar_for(compiler), "rcs", str(archive), str(obj)])
        # Link the .so from the fat object's real code (NOT -flto) so the entry points survive + export.
        _run([compiler, "-shared", *opt, *_fastest_linker(compiler), "-Wl,--export-dynamic",
              "-Wl,--whole-archive", str(archive), "-Wl,--no-whole-archive", *omp_l, *vec_l, "-o", str(so)])
    return so, time.perf_counter() - t0


def build_sdfg(sdfg: dace.SDFG, out_dir: Path, compiler: str = DEFAULT_COMPILER,
               flags: Optional[List[str]] = None, expand_libnodes: bool = False,
               openmp: Optional[OpenMPRuntime] = None, link_external: bool = False,
               lto: bool = False, veclib: Optional[VectorMathLib] = None) -> BuiltSDFG:
    """Generate + compile + link an SDFG ourselves; return a :class:`BuiltSDFG` ready to call, carrying
    the ``codegen_seconds`` (optimization) and ``compile_seconds`` (post-optimization toolchain) times.

    :param expand_libnodes: expand library nodes to loops first (the "without libnodes" timing variant).
    :param openmp: when set, add this runtime's per-compiler OpenMP compile+link flags (a SEPARATE axis
        from ``flags``) so a parallel kernel links against the one mandated runtime -- and a set built
        with different compilers still shares it (e.g. flang ``-fopenmp=libomp``).
    :param link_external: link the nest as a separate static ``.a`` rather than a monolithic TU (see
        :func:`_compile`); this path is ALWAYS built maximally (LTO + fastest linker + all opt flags), so
        ``compile_seconds`` reflects the fully-optimized external link.
    :param lto: add ``-flto`` to the MONOLITHIC build too (external linking always uses LTO regardless).
    :param veclib: a vector-math library (SLEEF / libmvec / SVML) to compile + link against, so
        autovectorized transcendentals use packed routines -- a SEPARATE axis from ``flags``/``openmp``.
    """
    import copy
    t_opt = time.perf_counter()
    sdfg = copy.deepcopy(sdfg)
    if expand_libnodes:
        sdfg.expand_library_nodes()
    frame, name = generate_program_folder(sdfg, out_dir)
    folder = frame.parent.parent.parent  # <out>/src/cpu/x.cpp -> <out>
    codegen_seconds = time.perf_counter() - t_opt

    flags = list(flags if flags is not None else DEFAULT_FLAGS)
    code = frame.read_text()
    init_params = _parse_params(_signature(code, f"__dace_init_{name}"))
    prog_params = _parse_params(_signature(code, f"__program_{name}"))
    scalar_names = {a for a, d in sdfg.arrays.items() if isinstance(d, dace.data.Scalar)}

    so, compile_seconds = _compile(frame, folder, name, compiler, flags, openmp, link_external, lto, veclib)
    return BuiltSDFG(name=name, so_path=so, _lib=ctypes.CDLL(str(so)),
                     _init_params=init_params, _prog_params=prog_params, _scalar_names=scalar_names,
                     codegen_seconds=codegen_seconds, compile_seconds=compile_seconds)


@dataclass
class LinkTimings:
    """Optimization time and the two post-optimization compile times isolated on ONE codegen."""
    codegen_seconds: float                 # the optimization (DaCe codegen) phase, run once
    compile_seconds_monolithic: float      # WITHOUT external linking (single TU)
    compile_seconds_external: float        # WITH external linking (static .a -> .so)


def compare_link_modes(sdfg: dace.SDFG, out_dir: Path, compiler: str = DEFAULT_COMPILER,
                       flags: Optional[List[str]] = None, openmp: Optional[OpenMPRuntime] = None,
                       lto: bool = False, veclib: Optional[VectorMathLib] = None) -> LinkTimings:
    """Generate the code ONCE (one optimization pass), then compile that same frame both monolithically
    and via an external static library, so ``compile_seconds`` is the only thing that differs. Returns
    the optimization time plus both post-optimization compile times."""
    import copy
    t_opt = time.perf_counter()
    sdfg = copy.deepcopy(sdfg)
    frame, name = generate_program_folder(sdfg, out_dir)
    folder = frame.parent.parent.parent
    codegen_seconds = time.perf_counter() - t_opt
    flags = list(flags if flags is not None else DEFAULT_FLAGS)
    _, mono = _compile(frame, folder, name, compiler, flags, openmp, link_external=False, lto=lto, veclib=veclib)
    _, ext = _compile(frame, folder, name, compiler, flags, openmp, link_external=True, lto=lto, veclib=veclib)
    return LinkTimings(codegen_seconds=codegen_seconds, compile_seconds_monolithic=mono,
                       compile_seconds_external=ext)
