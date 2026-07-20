"""Owns the DaCe build (BUILD.md): codegen + compile/link with one compiler, call via ctypes
(manual init/program/exit) -- not ``dace.compile()``, whose ``__call__`` re-marshals args and
confounds timing. Entry points: ``__dace_init_N``/``__program_N``/``__dace_exit_N``.
"""
from __future__ import annotations

import contextlib
import copy
import ctypes
import ctypes.util
import functools
import os
from _ctypes import dlclose  # release a built .so mapping (BuiltSDFG.unload)
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
from dace.transformation.auto.auto_optimize import set_fast_implementations

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
DEFAULT_FLAGS = ["-O3", "-march=native", "-std=c++20", "-fPIC", "-shared"]


def compiler_family(compiler: str) -> str:
    """OpenMP-relevant compiler family: ``llvm`` (clang/flang, icx/icpx/ifx), ``intel-classic``
    (icc/icpc/ifort), ``nvidia`` (nvc/nvc++/nvfortran), or ``gnu`` (gcc/gfortran, default)."""
    b = Path(compiler).name.lower()
    if "clang" in b or "flang" in b or b.startswith(("icx", "icpx", "ifx")):
        return "llvm"
    if b.startswith(("icc", "icpc", "ifort")):
        return "intel-classic"
    if b.startswith(("nvc", "nvfortran", "pgcc", "pgfortran")):
        return "nvidia"
    return "gnu"


#: OpenMP ABI a family emits -- ``gomp`` (GCC ``GOMP_*``) or ``kmpc`` (LLVM/Intel ``__kmpc_*``, incl. nvc/nvc++).
_COMPILER_ABI = {"gnu": "gomp", "llvm": "kmpc", "intel-classic": "kmpc", "nvidia": "kmpc"}

#: Runtimes selectable by name via ``-fopenmp=<name>`` on clang/flang/icx; other names aren't reachable
#: this way even with a matching ABI (gcc links any runtime explicitly via ``-l<soname>``).
_LLVM_SELECTABLE = frozenset({"libomp", "libgomp", "libiomp5"})


@dataclass
class OpenMPRuntime:
    """One OpenMP runtime the whole program links (PARALLEL.md: one runtime for every node library + the
    driver). ``libomp`` is default: LLVM-selectable by name, ABI-compat with libiomp5, implements GOMP_*
    too, so gcc- and clang-built libraries share one runtime/thread pool."""
    name: str = "libomp"  # selected by name on LLVM (``-fopenmp=<name>``)
    soname: str = "omp"  # ``-l<soname>`` for explicit linking
    #: ``-L`` for the runtime; None -> DISCOVERED via :func:`linkable_lib_dir`. ``""`` forces bare ``-l<soname>``.
    lib_dir: Optional[str] = None
    #: ABIs this runtime implements. libomp/libiomp5/libnvomp expose both; libgomp is GOMP_*-only, so a
    #: kmpc compiler (clang/flang/icx/nvc++) can't use it.
    provides: frozenset = frozenset({"kmpc", "gomp"})

    def compatible(self, compiler: str) -> bool:
        """True if ``compiler`` can LINK this runtime: nvidia/intel-classic hard-link their own native
        runtime only; llvm selects by name from :data:`_LLVM_SELECTABLE`; gnu links any gomp-ABI runtime."""
        fam = compiler_family(compiler)
        if fam == "nvidia":
            return self.name == "libnvomp"
        if fam == "intel-classic":
            return self.name == "libiomp5"
        if fam == "llvm":
            return self.name in _LLVM_SELECTABLE and _COMPILER_ABI["llvm"] in self.provides
        return _COMPILER_ABI["gnu"] in self.provides  # gnu

    def check(self, compiler: str) -> None:
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
        self.check(compiler)
        fam = compiler_family(compiler)
        if fam == "llvm":  # pick the runtime by name
            return [f"-fopenmp={self.name}"]
        if fam == "intel-classic":
            return ["-qopenmp"]
        if fam == "nvidia":
            return ["-mp"]  # hard-links native libnvomp; no -fopenmp=<lib> switch
        return ["-fopenmp"]  # gnu: runtime fixed at link, not by this flag

    def link_flags(self, compiler: str) -> List[str]:
        """Flags to link a program against THIS runtime only (avoids dual-runtime oversubscription)."""
        self.check(compiler)
        fam = compiler_family(compiler)
        # explicit lib_dir wins (pin a spack/module runtime, "" forces bare -l<soname>); else discover it
        pinned = self.lib_dir if self.lib_dir is not None else linkable_lib_dir(self.soname, compiler)
        # -L alone only satisfies the LINKER: a runtime off the default loader path (the very case
        # linkable_lib_dir globs for, e.g. /usr/lib/llvm-18/lib) then leaves DT_NEEDED with no RUNPATH, and
        # the ctypes.CDLL right after the build dies with "libomp.so.5: cannot open shared object file".
        # veclib.link_flags pairs -L with -rpath for exactly this reason; match it.
        libdir = [f"-L{pinned}", f"-Wl,-rpath,{pinned}"] if pinned else []
        if fam == "llvm":
            return [f"-fopenmp={self.name}", *libdir]
        if fam == "intel-classic":
            return ["-qopenmp", *libdir]
        if fam == "nvidia":
            return ["-mp", *libdir]
        # gnu: link the runtime EXPLICITLY (bare -fopenmp would pull libgomp instead)
        return [*libdir, f"-l{self.soname}"]


#: Ready-made OpenMP runtimes. libomp/libgomp/libiomp5 are mutually GOMP-ABI compatible (libomp/libiomp5
#: also implement __kmpc_*). NVIDIA's libnvomp is reachable only via nvc/nvfortran -mp.
LIBOMP = OpenMPRuntime(name="libomp", soname="omp")  # LLVM default; kmpc+gomp
LIBGOMP = OpenMPRuntime(name="libgomp", soname="gomp",
                        provides=frozenset({"gomp"}))  # GOMP-only; unusable by a kmpc compiler
LIBIOMP5 = OpenMPRuntime(name="libiomp5", soname="iomp5")  # Intel; kmpc+gomp, ABI-compat with libomp
LIBNVOMP = OpenMPRuntime(name="libnvomp", soname="nvomp")  # NVIDIA HPC; kmpc+gomp

#: name -> runtime, for a config/CLI knob.
OPENMP_RUNTIMES = {"libomp": LIBOMP, "libgomp": LIBGOMP, "libiomp5": LIBIOMP5, "libnvomp": LIBNVOMP}


def resolve_runtime(name: str) -> OpenMPRuntime:
    """Named runtime as an :class:`OpenMPRuntime`; unknown names fall back to ``lib<soname>`` (gcc-only)."""
    rt = OPENMP_RUNTIMES.get(name)
    if rt is not None:
        return rt
    soname = name[3:] if name.startswith("lib") else name
    return OpenMPRuntime(name=name, soname=soname)


def env_library_dirs() -> List[str]:
    """Dirs from env vars (``LD_LIBRARY_PATH``/``LIBRARY_PATH``/``DYLD_*``) -- ``find_library`` only
    consults ldconfig, missing a spack/module runtime living here."""
    dirs: List[str] = []
    for var in ("LD_LIBRARY_PATH", "LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"):
        dirs += [d for d in os.environ.get(var, "").split(os.pathsep) if d]
    return dirs


#: Drivers to ask where a runtime lives when the target compiler can't find it (each ships its own).
#: clang-first since libomp is the mandated runtime.
_LIB_PROBE_DRIVERS = ("clang++", "clang", "g++", "gcc")


def driver_lib_path(soname: str, compiler: str) -> Optional[Path]:
    """Where ``compiler`` resolves ``lib<soname>.so``, or ``None``. ``-print-file-name`` asks what the
    driver resolves, a different question from what ldconfig/``find_library`` finds (see
    :func:`linkable_lib_dir`)."""
    try:
        out = subprocess.run([compiler, f"-print-file-name=lib{soname}.so"], capture_output=True,
                             text=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not out or out == f"lib{soname}.so":
        return None
    # normalize LEXICALLY, never resolve(): libomp.so is often a symlink into another dir
    path = Path(os.path.normpath(out))
    return path if path.exists() else None


def driver_search_dirs(compiler: str) -> List[str]:
    """Library directories ``compiler`` itself searches (``-print-search-dirs`` -> ``libraries: =a:b:c``).
    Asked of the driver rather than guessed from distro layout, which varies by distro AND by toolchain
    version. Complements :func:`driver_lib_path`: that asks about ONE library the driver already resolves,
    this enumerates where it would look for any."""
    try:
        out = subprocess.run([compiler, "-print-search-dirs"], capture_output=True, text=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    for line in out.splitlines():
        if line.startswith("libraries:"):
            raw = line.split(":", 1)[1].strip().lstrip("=")  # "libraries: =/a:/b"
            return [os.path.normpath(d) for d in raw.split(os.pathsep) if d]
    return []


#: ``ldconfig`` by name AND by full path: /usr/sbin is off the default non-root PATH on Debian-family
#: slim images, where a bare-name spawn would silently drop the whole loader-cache layer.
_LDCONFIG_EXES = ("ldconfig", "/usr/sbin/ldconfig", "/sbin/ldconfig")


def ldconfig_dirs(soname: str) -> List[str]:
    """Directories the LOADER cache lists for ``lib<soname>``, in cache order. The linker needs the
    ``-dev`` ``.so`` symlink, which ldconfig does not index -- but it sits in the same directory as the
    versioned ``.so.N`` that ldconfig DOES index, so this locates the directory to probe."""
    out = ""
    for exe in _LDCONFIG_EXES:  # sbin is off the non-root PATH on slim images, so try it by full path
        try:
            out = subprocess.run([exe, "-p"], capture_output=True, text=True).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        if out:
            break
    if not out:
        return []
    dirs: List[str] = []
    for line in out.splitlines():
        if f"lib{soname}.so" not in line or "=>" not in line:
            continue
        d = os.path.dirname(line.split("=>")[-1].strip())
        if d and d not in dirs:
            dirs.append(d)
    return dirs


#: Common install layouts, tried ONLY after every driver/loader query came up empty. Hints, not truth:
#: a dev package can ship the ``.so`` symlink somewhere no driver searches and no ldconfig indexes
#: (Ubuntu's libomp-N-dev under /usr/lib/llvm-N/lib, reachable only if clang-N is also installed).
_LIB_DIR_HINT_ROOTS = ("/usr/lib", "/usr/lib64")  # globbed for llvm-<N>/lib*
_LIB_DIR_HINTS = ("/usr/lib64", "/usr/local/lib64", "/usr/local/lib")


def llvm_version(path: Path) -> Tuple[int, ...]:
    """The version tuple of an ``llvm-N[.M]`` directory, or ``(-1,)``. Sorting these as STRINGS ranks
    llvm-9 above llvm-21, which would link an ancient libomp against a modern object; comparing the parts
    as INTEGERS also keeps llvm-18.1 above llvm-18 instead of tying them."""
    parts = path.parent.name.partition("llvm-")[2].split(".")
    if not parts or not parts[0].isdigit():
        return (-1, )
    return tuple(int(p) for p in parts if p.isdigit())


def hint_dirs() -> List[str]:
    """Guessed library directories, newest LLVM first (a box with llvm-9 and llvm-21 must get 21).

    Ranked ACROSS roots, not within each: /usr/lib and /usr/lib64 both hold llvm-* trees on mixed
    installs, and sorting per root then concatenating yields /usr/lib/llvm-14 ahead of /usr/lib64/llvm-18.
    The path is the sort tiebreaker so the order is total -- Path.glob returns raw directory order, which
    varies with inode layout, and an unstable hint order picks a different libomp on identical machines.
    """
    found = [p for root in _LIB_DIR_HINT_ROOTS for p in Path(root).glob("llvm-*/lib*")]
    ranked = sorted({str(p) for p in found}, key=lambda d: (llvm_version(Path(d)), d), reverse=True)
    return ranked + [d for d in _LIB_DIR_HINTS if d not in ranked]


def linker_finds(soname: str, compiler: str = DEFAULT_COMPILER) -> bool:
    """True if ``compiler`` already resolves ``-l<soname>`` with no ``-L``."""
    return driver_lib_path(soname, compiler) is not None


@functools.lru_cache(maxsize=None)
def linkable_lib_dir(soname: str, compiler: str = DEFAULT_COMPILER) -> Optional[str]:
    """The ``-L`` directory needed to LINK ``lib<soname>``, or ``None`` if the linker already finds it.
    Loader and linker search different paths (e.g. Ubuntu's libomp-dev symlink can land off the link
    path while ldconfig still reports it "installed"). Tries: env loader path, then drivers that ship
    these runtimes, then a layout guess."""
    if shutil.which(compiler) is None:
        return None  # no linker to ask; a guessed -L would be worse than none
    if linker_finds(soname, compiler):
        return None
    for d in env_library_dirs():  # explicit intent (spack/module) outranks anything inferred
        p = Path(d)
        if (p / f"lib{soname}.so").exists() or (p / f"lib{soname}.a").exists() or (p / f"lib{soname}.dylib").exists():
            return d
    for probe in _LIB_PROBE_DRIVERS:
        if probe != compiler and shutil.which(probe):
            found = driver_lib_path(soname, probe)
            if found is not None:
                return str(found.parent)
    # Nothing RESOLVED it, so enumerate where the drivers look and where the loader cache says it lives.
    # Both are host-derived: a hardcoded distro ladder (/usr/lib/llvm-*/lib, /usr/lib64, ...) goes stale
    # per distro and per toolchain version, and silently picks the wrong LLVM when several are installed.
    for probe in _LIB_PROBE_DRIVERS:
        if shutil.which(probe):
            for d in driver_search_dirs(probe):
                if (Path(d) / f"lib{soname}.so").exists():
                    return d
    # The loader cache lists dirs in ITS order, which is not version order -- so rank the candidates the
    # same way hint_dirs does before taking the first hit. Without this the version ranking below is
    # unreachable whenever ldconfig knows any llvm-* dir at all, and an llvm-14 libomp wins over llvm-18.
    for d in sorted(ldconfig_dirs(soname), key=lambda d: (llvm_version(Path(d)), d), reverse=True):
        if (Path(d) / f"lib{soname}.so").exists():
            return d
    for d in hint_dirs():  # last resort: common layouts, only for a runtime NO query above admitted to
        if (Path(d) / f"lib{soname}.so").exists():
            return d
    return None


def lib_linkable(soname: str, compiler: str = DEFAULT_COMPILER) -> bool:
    """True if ``-l<soname>`` resolves at link time. Not the same as ``find_library``, which a versioned
    ``.so.5`` satisfies even when the linker needs the ``-dev`` package's ``.so`` symlink."""
    return linker_finds(soname, compiler) or linkable_lib_dir(soname, compiler) is not None


def lib_findable(soname: str, lib_dir: Optional[str]) -> bool:
    """True if ``lib<soname>`` is in ``lib_dir``, an env loader path, or the system loader path (matches
    versioned ``.so.N`` too). Shared by the OpenMP-runtime and veclib installed-probes."""
    for d in ([lib_dir] if lib_dir else []) + env_library_dirs():
        p = Path(d)
        if (p / f"lib{soname}.a").exists() or (p / f"lib{soname}.dylib").exists() or any(p.glob(f"lib{soname}.so*")):
            return True
    return ctypes.util.find_library(soname) is not None


def runtime_installed(rt: OpenMPRuntime) -> bool:
    """True if the runtime's shared object can be found. libnvomp lives off the default path, so without
    a ``lib_dir`` it reads as not-installed (pruned with a warning for a non-nvhpc link)."""
    return lib_findable(rt.soname, rt.lib_dir)


#: clang/icx -fveclib token per veclib. x86 has no -fveclib=SLEEF, so sleef reuses libmvec's token
#: (differs only in linked lib, libsleefgnuabi); svml is __svml_*.
_CLANG_VECLIB = {"sleef": "libmvec", "libmvec": "libmvec", "svml": "SVML"}

#: Intel oneAPI roots holding libsvml (+ libintlc/libimf/libirng), off the default path; globbed for */lib.
_INTEL_ONEAPI_ROOTS = ("/opt/intel/oneapi/compiler", "/opt/intel/oneapi")


def veclib_lib_dir(soname: str, compiler: str) -> Optional[str]:
    """Directory holding ``lib<soname>`` for ``-L``/``-rpath``, or ``None`` if on the default path.
    Tries the driver, Intel oneAPI dirs, then a SLEEF prefix (env/``~/.local``/``/usr/local``)."""
    found = driver_lib_path(soname, compiler)
    if found is not None:
        return str(found.parent)
    dirs: List[str] = []
    for root in _INTEL_ONEAPI_ROOTS:
        dirs += sorted((str(p) for p in Path(root).glob("*/lib")), reverse=True)
    prefix = os.environ.get("NF_SLEEF_PREFIX")
    if prefix:
        dirs.append(str(Path(prefix) / "lib"))
    dirs += [str(Path.home() / ".local" / "lib"), "/usr/local/lib"]
    for d in dirs:
        if any(Path(d).glob(f"lib{soname}.so*")):
            return d
    return None


@dataclass
class VectorMathLib:
    """SIMD elementary-math library (sin/exp/...) an autovectorized loop calls instead of scalarizing.
    ``libmvec``/``sleef`` both emit ``_ZGV*`` (gcc fast-math, or clang/icx ``-fveclib=libmvec``; sleef
    just links ``libsleefgnuabi`` instead); ``svml`` is clang/icx-only, ``-fveclib=SVML`` -> ``__svml_*``."""
    name: str  # libmvec | sleef | svml
    soname: Optional[str]  # -l<soname> for the vector symbols (None: toolchain/glibc provides)
    lib_dir: Optional[str] = None  # explicit -L override; None resolves via veclib_lib_dir

    def compatible(self, compiler: str) -> bool:
        fam = compiler_family(compiler)
        if fam == "llvm":  # -fveclib=libmvec (also SLEEF's path) or -fveclib=SVML
            return self.name in ("libmvec", "sleef", "svml")
        if fam == "gnu":  # gcc emits _ZGV* under fast-math; libmvec/SLEEF satisfy it
            return self.name in ("libmvec", "sleef")  # NOT svml: gcc never emits __svml_*
        if fam == "intel-classic":
            return self.name == "svml"  # classic icc emits SVML natively
        return False  # nvidia: uses its own -Mvect

    def check(self, compiler: str) -> None:
        if not self.compatible(compiler):
            raise ValueError(f"{Path(compiler).name} ({compiler_family(compiler)}) cannot use the {self.name} "
                             f"vector math library; try a compatible compiler or a different veclib.")

    def compile_flags(self, compiler: str) -> List[str]:
        self.check(compiler)
        if compiler_family(compiler) == "llvm":  # SVML -> __svml_*, else glibc _ZGV*
            return [f"-fveclib={_CLANG_VECLIB[self.name]}"]
        return []  # gnu: -ffast-math autovec already emits _ZGV*; intel-classic: SVML native

    def link_flags(self, compiler: str) -> List[str]:
        self.check(compiler)
        if not self.soname:
            return []
        libdir = self.lib_dir or veclib_lib_dir(self.soname, compiler)
        search = [f"-L{libdir}", f"-Wl,-rpath,{libdir}"] if libdir else []
        if self.name == "svml":
            search.append("-Wl,--disable-new-dtags")  # transitive libintlc needs DT_RPATH, not RUNPATH
        # pin NEEDED regardless of link-line position (else a veclib -l before the object is dropped)
        return [*search, f"-Wl,--push-state,--no-as-needed,-l{self.soname},--pop-state"]


SLEEF = VectorMathLib(name="sleef", soname="sleefgnuabi")  # GNU-ABI lib, exports _ZGV* symbols
LIBMVEC = VectorMathLib(name="libmvec", soname="mvec")  # glibc's libmvec
SVML = VectorMathLib(name="svml", soname="svml")  # Intel SVML runtime

#: name -> vector-math library, for a config/CLI knob.
VECTOR_LIBS = {"sleef": SLEEF, "libmvec": LIBMVEC, "svml": SVML}


def vectorlib_installed(vl: VectorMathLib) -> bool:
    """True if the vector library is findable. A ``soname``-less entry is always present. libsvml/
    libsleefgnuabi live off the ldconfig cache, so fall back to :func:`veclib_lib_dir`."""
    if not vl.soname:
        return True
    return lib_findable(vl.soname, vl.lib_dir) or veclib_lib_dir(vl.soname, DEFAULT_COMPILER) is not None


# TODO(blas): a BLAS/LAPACK axis (openblas/mkl/blis/nvpl/accelerate) the same way -- discovery exists
# (arena.discover_blas_libraries); missing is threading a chosen BLAS into the link line + a prune step.


def dace_runtime_include() -> Path:
    """The ``-I`` directory holding DaCe's runtime headers (``dace/runtime/include``)."""
    inc = Path(dace.__file__).parent / "runtime" / "include"
    if not inc.is_dir():
        raise FileNotFoundError(f"DaCe runtime include not found at {inc}")
    return inc


@dataclass
class Param:
    name: str
    ctype: object  # a ctypes type
    is_pointer: bool


def parse_params(param_str: str) -> List[Param]:
    """Parse a C parameter list into typed params. Skips the leading ``N_state_t *__state`` handle."""
    params: List[Param] = []
    for raw in split_params(param_str):
        # strip qualifiers as whole WORDS: a substring strip would corrupt names like `const_term`
        tok = re.sub(r"\b(?:const|__restrict__)\b", "", raw).strip()
        if not tok or tok.endswith("_state_t *__state") or tok.endswith("_state_t* __state"):
            continue
        is_ptr = "*" in tok
        name = re.split(r"[\s*]+", tok)[-1]
        base = tok[:tok.rfind(name)].replace("*", "").strip()
        if is_ptr:
            params.append(Param(name, ctypes.POINTER(_C_PTR.get(base, ctypes.c_double)), True))
        else:
            # an unmapped type would guess a width silently -- an ABI bug ctypes can't catch -- so refuse
            ctype = _C_SCALAR.get(base)
            if ctype is None:
                raise ValueError(f"parameter {name!r} of entry point has C type {base!r}, which has no ctypes "
                                 f"mapping (known: {sorted(_C_SCALAR)}); add it to _C_SCALAR")
            params.append(Param(name, ctype, False))
    return params


def split_params(param_str: str) -> List[str]:
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


def signature(code: str, symbol: str) -> str:
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
    _init_params: List[Param]
    _prog_params: List[Param]
    #: wall time of the OPTIMIZATION phase (DaCe codegen + C++ emission), distinct from the compile below.
    codegen_seconds: float = 0.0
    #: wall time of the post-optimization COMPILE (compiler/linker turning C++ into the ``.so``).
    compile_seconds: float = 0.0
    _handle: Optional[ctypes.c_void_p] = field(default=None, repr=False)

    def init(self, sizes: Dict[str, int]) -> None:
        fn = self._lib[f"__dace_init_{self.name}"]  # ctypes CDLL indexing (not getattr) binds the entry point
        fn.restype = ctypes.c_void_p
        fn.argtypes = [p.ctype for p in self._init_params]
        # each param's OWN ctype: a hardcoded width would mismatch (jacobi's int N vs gemm's int64_t NI)
        self._handle = ctypes.c_void_p(fn(*[p.ctype(int(sizes[p.name])) for p in self._init_params]))

    def bind_program(self, buffers: Dict[str, np.ndarray], sizes: Dict[str, int]):
        """Bind ``__program_N`` and its ctypes args ONCE; return ``(fn, args)``, so a timed rep loop calls
        ``fn(*args)`` with no per-rep marshaling. ``init`` must have run; ``buffers`` must stay alive."""
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
        return fn, args

    def program(self, buffers: Dict[str, np.ndarray], sizes: Dict[str, int]) -> None:
        """Call ``__program_N(handle, args...)`` once, in place (init must have run)."""
        fn, args = self.bind_program(buffers, sizes)
        fn(*args)

    def unload(self) -> None:
        """Release the ``dlopen`` mapping (file may be deleted after). Prevents a long sweep from
        accumulating one live mapping per kernel."""
        if self._lib is not None:
            dlclose(self._lib._handle)
            self._lib = None

    def close(self) -> None:
        if self._handle is not None:
            fn = self._lib[f"__dace_exit_{self.name}"]
            fn.restype = ctypes.c_int
            fn.argtypes = [ctypes.c_void_p]
            fn(self._handle)
            self._handle = None

    def run(self, buffers: Dict[str, np.ndarray], sizes: Dict[str, int]) -> None:
        """One-shot init -> program -> exit (for correctness; for timing, init once + loop program)."""
        self.init(sizes)
        try:
            self.program(buffers, sizes)
        finally:
            self.close()


def config_has(*path) -> bool:
    """True when the running DaCe config schema DEFINES the key at ``path`` (``Config.get`` raises on an
    unknown key), so the codegen axis degrades gracefully instead of crashing."""
    try:
        dace.config.Config.get(*path)
        return True
    except (KeyError, TypeError):
        return False


#: Codegen-implementation axis (``compiler.cpu.implementation``): ``experimental`` is the readable
#: constexpr-index-fn codegen (nest-forge's DEFAULT); ``legacy`` is the classic connector-based codegen.
CODEGEN_IMPLS = ("experimental", "legacy")
#: nest-forge defaults to DaCe's NEW (human-readable) codegen when the running DaCe build supports it.
DEFAULT_CODEGEN_IMPL = "experimental"


def default_codegen_impl() -> str:
    """Codegen impl used when the caller specifies none: ``experimental`` if this DaCe build supports
    ``compiler.cpu.implementation``, else ``legacy``."""
    return DEFAULT_CODEGEN_IMPL if config_has("compiler", "cpu", "implementation") else "legacy"


def codegen_impls_available() -> Tuple[str, ...]:
    """Codegen-implementation values THIS DaCe build supports, default first: both when the schema has
    ``compiler.cpu.implementation``, else just ``('legacy',)``. The driver sweeps exactly this tuple."""
    return CODEGEN_IMPLS if config_has("compiler", "cpu", "implementation") else ("legacy", )


@contextlib.contextmanager
def codegen_config(codegen_impl: str):
    """Scope the DaCe codegen config for ONE ``generate_code`` call: pin ``emit_tree_reductions`` true and
    select the CPU codegen ``implementation``. Raises for ``experimental`` on a build lacking the key,
    rather than silently emitting legacy and mislabelling it."""
    with dace.config.temporary_config():
        dace.config.Config.set("compiler", "emit_tree_reductions", value=True)
        if config_has("compiler", "cpu", "implementation"):
            dace.config.Config.set("compiler", "cpu", "implementation", value=codegen_impl)
        elif codegen_impl != "legacy":
            raise ValueError(f"codegen_impl={codegen_impl!r} requested, but this DaCe build has no "
                             "'compiler.cpu.implementation' key (needs the experimental-codegen branch)")
        yield


def generate_program_folder(sdfg: dace.SDFG, out_dir: Path, codegen_impl: Optional[str] = None) -> Tuple[Path, str]:
    """Lay out DaCe's compilable source tree (``src/cpu/<name>.cpp`` + ``include/``) via DaCe's own
    ``generate_program_folder`` so relative includes resolve, but WITHOUT letting DaCe compile it.

    :param codegen_impl: ``experimental`` | ``legacy``; ``None`` -> :func:`default_codegen_impl`.
    :returns: (the C++ Frame source path, sdfg name).
    """
    from dace.codegen import codegen, compiler as dace_compiler
    out_dir.mkdir(parents=True, exist_ok=True)
    with codegen_config(codegen_impl or default_codegen_impl()):
        code_objects = codegen.generate_code(sdfg)
    folder = Path(dace_compiler.generate_program_folder(sdfg, code_objects, str(out_dir)))
    frame = folder / "src" / "cpu" / f"{sdfg.name}.cpp"
    if not frame.exists():  # fall back to whatever CPU Frame the layout produced
        frame = next(folder.glob("src/cpu/*.cpp"))
    return frame, sdfg.name


def include_flags(folder: Path) -> List[str]:
    """Header search paths: the generated ``include/`` and DaCe's runtime include."""
    return [f"-I{folder / 'include'}", f"-I{dace_runtime_include()}"]


def clang_major_via_preprocessor(compiler: str) -> Optional[int]:
    """Underlying clang major via the preprocessor's ``__clang_major__`` -- for icx/icpx/ifx, whose
    ``--version`` prints an oneAPI banner instead of ``clang version``. ``None`` if undetermined."""
    try:
        p = subprocess.run([compiler, "-dM", "-E", "-x", "c", "/dev/null"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"#define __clang_major__ (\d+)", p.stdout)
    return int(m.group(1)) if m else None


@functools.lru_cache(maxsize=None)
def compiler_version(compiler: str) -> Tuple[int, int]:
    """The compiler's ``(major, minor)`` version, parsed from ``--version``. Gates version-dependent
    features (fast linkers, fat-LTO). Returns ``(0, 0)`` (assume old) if unparseable, never a guess."""
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
        # icx/icpx/ifx hide the clang version behind their own banner; ask the preprocessor instead
        cmaj = clang_major_via_preprocessor(compiler)
        return (cmaj, 0) if cmaj is not None else (0, 0)
    if fam == "gnu":
        m = re.search(r"\bg(?:cc|\+\+)?[^\n]*?\b(\d+)\.(\d+)\.\d+\b", out) or re.search(r"\b(\d+)\.(\d+)\.\d+\b", out)
        if m:
            return (int(m.group(1)), int(m.group(2)))
    return (0, 0)


def ar_for(compiler: str) -> str:
    """The LTO-plugin-aware ``ar`` for this compiler (``gcc-ar``/``llvm-ar``) when present, so archiving
    an ``-flto`` object keeps it linkable; plain ``ar`` otherwise."""
    cand = {"gnu": "gcc-ar", "llvm": "llvm-ar"}.get(compiler_family(compiler), "ar")
    return cand if shutil.which(cand) else "ar"


#: Minimum compiler version accepting ``-fuse-ld=<linker>``, per family; absent pairs are unsupported.
#: mold needs gcc>=12.1/clang>=12; icx/icpx report as modern LLVM, clearing the clang gates.
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


def linker_supported(compiler: str, linker: str) -> bool:
    """True if ``compiler`` is new enough to accept ``-fuse-ld=<linker>``. Version-gated (see
    :data:`_LINKER_MIN`) so we never hand an old gcc/clang a ``-fuse-ld=mold`` it rejects."""
    fam = compiler_family(compiler)
    floor = _LINKER_MIN.get(linker, {}).get(fam)
    return floor is not None and compiler_version(compiler) >= floor


def fat_lto_flags(compiler: str) -> List[str]:
    """Flags for a FAT-LTO object (bitcode + real machine code), or ``[]`` if this compiler can't (warns,
    archives without LTO). gcc since ~4.8, clang since 18; classic icc/NVIDIA have no fat-LTO."""
    fam = compiler_family(compiler)
    if fam == "gnu":
        return ["-flto", "-ffat-lto-objects"]
    if fam == "llvm" and compiler_version(compiler) >= (18, 0):
        return ["-flto", "-ffat-lto-objects"]
    reason = ("clang < 18 has no -ffat-lto-objects" if fam == "llvm" else
              "classic icc uses -ipo, not fat LTO" if fam == "intel-classic" else "no fat-LTO support")
    warnings.warn(f"{Path(compiler).name}: {reason}; archiving the node library without LTO "
                  f"(the .so still links from real machine code and runs correctly).")
    return []


#: Wall-clock ceiling for a SINGLE compile/link/archive command. A stuck compile would otherwise freeze
#: the whole sweep rank (ThreadPoolExecutor shutdown waits for every worker). Override: NF_COMPILE_TIMEOUT.
COMPILE_TIMEOUT_S: float = float(os.environ.get("NF_COMPILE_TIMEOUT", "900"))


def run(cmd: List[str], timeout: Optional[float] = COMPILE_TIMEOUT_S) -> None:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        # child already SIGKILLed; surface as a normal build failure so the sweep moves on
        raise RuntimeError(f"command timed out after {timeout:.0f}s: {' '.join(cmd[:2])} ... "
                           f"(pathological compile/link; ceiling is NF_COMPILE_TIMEOUT)")
    if p.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd[:2])} ...\n{p.stderr[-2000:]}")


#: Fast alternative linkers, FASTEST FIRST. Default ``bfd`` ``ld`` is always the fallback (not listed).
_FAST_LINKERS = ("mold", "lld", "gold")


@functools.lru_cache(maxsize=None)
def ccache_available() -> bool:
    """Whether a compiler cache is installed and not disabled.

    ``ccache`` keys on PREPROCESSED SOURCE, not on the path, so the same generated kernel recompiled from a
    fresh temp directory (every arena cell, every sweep rung) is a cache hit. ``NF_NO_CCACHE=1`` forces it
    off. Probed once per process."""
    if os.environ.get("NF_NO_CCACHE"):
        return False
    return shutil.which("ccache") is not None


def ccache_prefix(use_ccache: Optional[bool]) -> List[str]:
    """Launcher prefix for a compiler invocation: ``["ccache"]`` or ``[]``.

    ``None`` means AUTO -- use the cache when it is installed. Pass ``False`` from any path that MEASURES
    compile time: a cache hit returns in ~0s, which would make the post-optimization toolchain cost
    (``Cell.compile_us``, :class:`LinkTimings`) meaningless rather than merely faster.
    """
    if use_ccache is False:
        return []
    return ["ccache"] if ccache_available() else []


def available_linkers() -> Dict[str, str]:
    """Fast alternative linkers installed, fastest first: name -> backing binary path. Default ``bfd``
    is not listed (always the fallback)."""
    found: Dict[str, str] = {}
    for ld in _FAST_LINKERS:
        p = shutil.which(ld) or shutil.which(f"ld.{ld}")
        if p:
            found[ld] = p
    return found


def fastest_linker(compiler: str) -> List[str]:
    """``-fuse-ld=<linker>`` for the fastest installed linker this compiler accepts (mold > lld > gold),
    or ``[]``. NVIDIA's nvc/nvc++ has no ``-fuse-ld`` switch, so it keeps its default."""
    if compiler_family(compiler) == "nvidia":
        return []
    for ld in available_linkers():  # dict preserves the fastest-first order of _FAST_LINKERS
        if linker_supported(compiler, ld):
            return [f"-fuse-ld={ld}"]
    return []


@dataclass
class BuildOptions:
    """Toolchain + optimization knobs for the owned build (:func:`build_sdfg` / :func:`compare_link_modes`
    take this instead of a long parameter list). Each axis is independent."""
    compiler: str = DEFAULT_COMPILER
    flags: Optional[List[str]] = None  # None -> DEFAULT_FLAGS
    expand_libnodes: bool = False  # expand library nodes to naive loops ("without libnodes" variant)
    fast_libnodes: bool = False  # instead of expanding, pick the fast library impl (OpenBLAS/MKL)
    blas_link: Optional[List[str]] = None  # link flags for the chosen BLAS (e.g. ['-lopenblas'])
    openmp: Optional[OpenMPRuntime] = None  # the one mandated runtime to link (per-compiler flags)
    link_external: bool = False  # link the nest as a separate static .a (else a monolithic single TU)
    lto: bool = False  # enable LTO: -flto (monolithic) / fat-LTO object in the .a (external)
    veclib: Optional[VectorMathLib] = None  # SLEEF / libmvec / SVML, a separate axis from flags/openmp
    # DaCe CPU codegen: 'experimental' (DEFAULT where available) | 'legacy'; downgrades on an older build.
    codegen_impl: str = field(default_factory=default_codegen_impl)
    # DaCe multi-dim tile-op vectorizer config applied before codegen; None = no vectorization. Typed as
    # object to keep the vectorizer import lazy.
    vectorize: Optional[object] = None
    # Extra link arguments appended AFTER the frame object -- the extern nest-variant libs (absolute .so
    # path + its -Wl,-rpath) for the differential swap path. nest-forge compiles the generated frame
    # directly (bypassing DaCe's CMake), so ``ExternLibEnv``'s libraries are NOT auto-linked; the caller
    # passes them here instead. Placed after the frame so ld resolves the extern-C entry symbols.
    extra_link: Optional[List[str]] = None
    # Compiler cache: None = AUTO (use ccache when installed), False = never. Set False for any build whose
    # compile_seconds is reported as a measurement -- a cache hit reports ~0s (see ccache_prefix).
    use_ccache: Optional[bool] = None

    def resolved_flags(self) -> List[str]:
        return list(self.flags if self.flags is not None else DEFAULT_FLAGS)


def set_fast_libnodes(sdfg: dace.SDFG) -> None:
    """Select the fastest AVAILABLE library-node implementation (OpenBLAS/MKL/LAPACK) for every library
    node, instead of lowering to naive loops. Link flags come via :attr:`BuildOptions.blas_link`.

    TODO(lib-axis): generalize into a per-node "try every backend" sweep, keeping the timed winner."""
    set_fast_implementations(sdfg, dace.dtypes.DeviceType.CPU)


def compile(frame: Path, folder: Path, name: str, opts: BuildOptions) -> Tuple[Path, float]:
    """Compile the generated frame into ``lib<name>.so``; return (path, toolchain wall_seconds only).
    Two link modes: ``link_external=False`` (monolithic, single TU) or ``=True`` (archive to a static
    ``.a`` then link via ``--whole-archive``); see the branches below."""
    compiler = opts.compiler
    flags = opts.resolved_flags()
    inc = include_flags(folder)
    omp_c = opts.openmp.compile_flags(compiler) if opts.openmp else []
    omp_l = opts.openmp.link_flags(compiler) if opts.openmp else []
    vec_c = opts.veclib.compile_flags(compiler) if opts.veclib else []
    vec_l = opts.veclib.link_flags(compiler) if opts.veclib else []
    blas_l = list(opts.blas_link or [])  # link the chosen BLAS (fast_libnodes)
    extra_l = list(opts.extra_link or [])  # extern nest-variant libs (differential swap), after the frame
    # AUTO-detected compiler cache. Only the COMPILE steps are cached (a link is not cacheable), and any
    # path that reports compile_seconds as a measurement passes use_ccache=False.
    cc = ccache_prefix(opts.use_ccache)
    so = folder / f"lib{name}.so"
    cflags = [f for f in flags if f != "-shared"]  # -shared is a link-only flag; drop it for any -c step
    obj = folder / f"{name}.o"
    lto_f = ["-flto"] if opts.lto else []

    if not opts.link_external and not opts.openmp:
        # no mandated runtime: one compile+link command is safe (nothing for a second runtime to sneak in)
        # libs go AFTER the source: ld resolves left-to-right, a -l before the object contributes nothing
        cmd = [*cc, compiler, *flags, *lto_f, *vec_c, *inc, str(frame), "-o", str(so), *vec_l, *blas_l, *extra_l]
        t0 = time.perf_counter()
        run(cmd)
    elif not opts.link_external:
        # mandated runtime: split compile from link so ONLY that runtime is linked (gnu dual-runtime trap)
        compile_cmd = [*cc, compiler, *cflags, *lto_f, "-c", *omp_c, *vec_c, *inc, str(frame), "-o", str(obj)]
        link_cmd = [compiler, "-shared", *cflags, *lto_f, str(obj), *omp_l, *vec_l, *blas_l, *extra_l, "-o", str(so)]
        t0 = time.perf_counter()
        run(compile_cmd)
        run(link_cmd)
    else:
        # external static-node-library path; resolve non-toolchain work (LTO probe, archiver, linker,
        # stale-archive cleanup) BEFORE the clock starts, so compile_seconds is compile+archive+link only
        lto_c = fat_lto_flags(compiler) if opts.lto else []
        ar = ar_for(compiler)
        ld = fastest_linker(compiler)
        archive = folder / f"lib{name}_nest.a"
        if archive.exists():
            archive.unlink()  # ar r APPENDS; start clean so a rebuild doesn't stack stale members
        compile_cmd = [*cc, compiler, *cflags, *lto_c, "-c", *omp_c, *vec_c, *inc, str(frame), "-o", str(obj)]
        ar_cmd = [ar, "rcs", str(archive), str(obj)]
        # link from the object's REAL code (NOT -flto) so the entry points survive + export
        link_cmd = [
            compiler, "-shared", *cflags, *ld, "-Wl,--export-dynamic", "-Wl,--whole-archive",
            str(archive), "-Wl,--no-whole-archive", *omp_l, *vec_l, *blas_l, *extra_l, "-o",
            str(so)
        ]
        t0 = time.perf_counter()
        run(compile_cmd)
        run(ar_cmd)
        run(link_cmd)
    return so, time.perf_counter() - t0


def apply_vectorizer(sdfg: dace.SDFG, config) -> None:
    """Apply the DaCe multi-dim tile-op CPU vectorizer to ``sdfg`` in place. Force-expands tile library
    nodes to tasklets regardless of ``config`` (no ``dace.compile`` here to lower them later). Lazy
    import: eager would close an import cycle."""
    import dataclasses
    from dace.transformation.passes.vectorization import VectorizeCPUMultiDim
    VectorizeCPUMultiDim(dataclasses.replace(config, expand_tile_nodes=True)).apply_pass(sdfg, {})


def build_sdfg(sdfg: dace.SDFG, out_dir: Path, opts: Optional[BuildOptions] = None) -> BuiltSDFG:
    """Generate + compile + link an SDFG ourselves; return a :class:`BuiltSDFG` carrying
    ``codegen_seconds``/``compile_seconds`` timing.

    :param opts: toolchain + optimization knobs; ``None`` uses all defaults (g++, monolithic, no OpenMP/veclib).
    """
    opts = opts or BuildOptions()
    t_opt = time.perf_counter()
    sdfg = copy.deepcopy(sdfg)
    if opts.expand_libnodes:
        sdfg.expand_library_nodes()
    elif opts.fast_libnodes:  # keep the library nodes, but pick the fast (OpenBLAS/MKL) implementation
        set_fast_libnodes(sdfg)
    if opts.vectorize is not None:
        apply_vectorizer(sdfg, opts.vectorize)
    frame, name = generate_program_folder(sdfg, out_dir, opts.codegen_impl)
    folder = frame.parent.parent.parent  # <out>/src/cpu/x.cpp -> <out>
    codegen_seconds = time.perf_counter() - t_opt

    code = frame.read_text()
    init_params = parse_params(signature(code, f"__dace_init_{name}"))
    prog_params = parse_params(signature(code, f"__program_{name}"))

    so, compile_seconds = compile(frame, folder, name, opts)
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
    """Generate the code ONCE, then compile that same frame both monolithically and externally, so
    ``compile_seconds`` is the only thing that differs. ``opts``' link mode is overridden per build; its
    other axes apply to both."""
    opts = opts or BuildOptions()
    t_opt = time.perf_counter()
    sdfg = copy.deepcopy(sdfg)
    if opts.expand_libnodes:  # mirror build_sdfg: compare the SAME (expanded) SDFG the caller configured
        sdfg.expand_library_nodes()
    elif opts.fast_libnodes:
        set_fast_libnodes(sdfg)
    frame, name = generate_program_folder(sdfg, out_dir, opts.codegen_impl)
    folder = frame.parent.parent.parent
    codegen_seconds = time.perf_counter() - t_opt
    # compile time IS the result here, so the cache is forced OFF: a hit would report ~0s and make the
    # monolithic-vs-external comparison meaningless.
    _, mono = compile(frame, folder, name, replace(opts, link_external=False, use_ccache=False))
    _, ext = compile(frame, folder, name, replace(opts, link_external=True, use_ccache=False))
    return LinkTimings(codegen_seconds=codegen_seconds, compile_seconds_monolithic=mono, compile_seconds_external=ext)
