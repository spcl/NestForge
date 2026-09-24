# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""What this machine's toolchains can do: compiler families, OpenMP runtimes, CUDA toolkits, and C-signature
parsing. Imports no DaCe; every answer is discovered, and subprocess probes are cached."""

from __future__ import annotations

import ctypes
import functools
import os
import re
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Iterator

import numpy as np

C_SCALAR = {
    "int32_t": ctypes.c_int32,
    "int64_t": ctypes.c_int64,
    "int": ctypes.c_int,
    "float": ctypes.c_float,
    "double": ctypes.c_double,
    "bool": ctypes.c_bool,
}

#: Pointer element types. A pointer argument passes only the buffer address, so a complex buffer binds as a
#: pointer to its component type.
C_PTR = {
    **C_SCALAR,
    "dace::complex64": ctypes.c_float,
    "dace::complex128": ctypes.c_double,
}

DEFAULT_COMPILER = "g++"

#: Records a shared library in DT_NEEDED only when something references it; every link passes it explicitly.
AS_NEEDED = "-Wl,--as-needed"

#: The C++ standard every build compiles against.
CXX_STD = "c++20"

DEFAULT_FLAGS = ["-O3", "-march=native", f"-std={CXX_STD}", "-fPIC", "-shared"]


@functools.lru_cache(maxsize=None, typed=True)
def compiler_family(compiler: str) -> str:
    """OpenMP-relevant compiler family: ``llvm`` (clang/flang, icx/icpx/ifx), or ``gnu``
    (gcc/gfortran, default)."""
    b = Path(compiler).name.lower()
    if "clang" in b or "flang" in b or b.startswith(("icx", "icpx", "ifx")):
        return "llvm"
    return "gnu"


#: OpenMP ABI a family emits -- ``gomp`` (GCC ``GOMP_*``) or ``kmpc`` (LLVM/oneAPI ``__kmpc_*``).
COMPILER_ABI = {"gnu": "gomp", "llvm": "kmpc"}

#: Runtimes selectable via -fopenmp=<name> on clang/flang/icx; gcc links any runtime explicitly via -l<soname>.
LLVM_SELECTABLE = frozenset({"libomp", "libgomp", "libiomp5"})


@dataclass(frozen=True, slots=True)
class OpenMPRuntime:
    """One OpenMP runtime the whole program links. ``libomp`` is default: LLVM-selectable by name and
    GOMP_*-compatible, so gcc- and clang-built libraries share one thread pool."""

    name: str = "libomp"  # selected by name on LLVM (``-fopenmp=<name>``)
    soname: str = "omp"  # ``-l<soname>`` for explicit linking
    #: ``-L`` for the runtime; None -> discovered via linkable_lib_dir. ``""`` forces bare ``-l<soname>``.
    lib_dir: str | None = None
    #: ABIs this runtime implements; libgomp is GOMP_*-only, unusable by a kmpc compiler (clang/nvc++).
    provides: frozenset[str] = frozenset({"kmpc", "gomp"})

    def compatible(self, compiler: str) -> bool:
        """Whether ``compiler`` can link this runtime: llvm selects it by name, gnu links any gomp-ABI runtime."""
        if compiler_family(compiler) == "llvm":
            return self.name in LLVM_SELECTABLE and COMPILER_ABI["llvm"] in self.provides
        return COMPILER_ABI["gnu"] in self.provides

    def check(self, compiler: str) -> None:
        exe = Path(compiler).name
        if compiler_family(compiler) == "gnu":
            if COMPILER_ABI["gnu"] not in self.provides:
                raise ValueError(
                    f"{exe} emits the 'gomp' OpenMP ABI, which {self.name} does not implement "
                    f"(it provides {sorted(self.provides)}). Use a gomp-capable runtime "
                    f"(libomp/libiomp5 carry a GOMP-compat layer; libgomp is gomp-only)."
                )
        elif COMPILER_ABI["llvm"] not in self.provides:
            raise ValueError(
                f"{exe} emits the 'kmpc' OpenMP ABI, which {self.name} does not implement "
                f"(it provides {sorted(self.provides)}); libgomp is gomp-only. Use a kmpc runtime (libomp/libiomp5)."
            )
        elif self.name not in LLVM_SELECTABLE:
            raise ValueError(
                f"{exe} selects the OpenMP runtime by name and only knows "
                f"{sorted(LLVM_SELECTABLE)}; {self.name} is not name-selectable by an LLVM compiler. "
                f"Use libomp/libiomp5, or build with gcc (which links {self.name} via -l{self.soname})."
            )

    def compile_flags(self, compiler: str) -> list[str]:
        """Flags to compile a translation unit with OpenMP against this runtime."""
        self.check(compiler)
        if compiler_family(compiler) == "llvm":  # pick the runtime by name
            return [f"-fopenmp={self.name}"]
        return ["-fopenmp"]  # gnu: runtime fixed at link, not by this flag

    def link_flags(self, compiler: str) -> list[str]:
        """Flags to link against this runtime only, so no second runtime starts its own thread pool. An explicit
        ``lib_dir`` wins (pin a spack/module runtime; ``""`` forces a bare ``-l<soname>``); otherwise the directory
        and library are discovered, see :func:`runtime_library`."""
        self.check(compiler)
        if self.lib_dir is not None:
            pinned, library = self.lib_dir, f"-l{self.soname}"
        else:
            pinned, library = linkable_lib_dir(self.soname, compiler), library_flag(self.soname, compiler)
        # -L alone leaves no RUNPATH; ctypes.CDLL fails to open the lib after build without -rpath too
        libdir = [f"-L{pinned}", f"-Wl,-rpath,{pinned}"] if pinned else []
        if compiler_family(compiler) == "llvm":
            return [f"-fopenmp={self.name}", *libdir]
        # gnu: a bare -fopenmp would pull in libgomp
        return [*libdir, library]


#: icx links libsvml/libimf/libirng/libintlc from off the loader path without a RUNPATH; this one finds them all.
SUPPORT_LIB_PROBE = "svml"


@functools.lru_cache(maxsize=None, typed=True)
def support_rpath_flags(compiler: str) -> tuple[str, ...]:
    """-Wl,-rpath for the compiler's own auto-linked support libs (icx svml/imf/irng/intlc), or () if none."""
    found = driver_lib_path(SUPPORT_LIB_PROBE, compiler)
    return (f"-Wl,-rpath,{found.parent}",) if found else ()


#: Ready-made OpenMP runtimes; libomp/libgomp/libiomp5 share the GOMP ABI.
LIBOMP = OpenMPRuntime(name="libomp", soname="omp")

#: GOMP-only, so a kmpc compiler cannot use it.
LIBGOMP = OpenMPRuntime(name="libgomp", soname="gomp", provides=frozenset({"gomp"}))

LIBIOMP5 = OpenMPRuntime(name="libiomp5", soname="iomp5")

#: name -> runtime, for a config/CLI knob.
OPENMP_RUNTIMES = {"libomp": LIBOMP, "libgomp": LIBGOMP, "libiomp5": LIBIOMP5}


def env_library_dirs() -> list[str]:
    """Dirs from LD_LIBRARY_PATH/LIBRARY_PATH; find_library only consults ldconfig, missing these."""
    dirs: list[str] = []
    for var in ("LD_LIBRARY_PATH", "LIBRARY_PATH"):
        dirs += [d for d in os.environ.get(var, "").split(os.pathsep) if d]
    return dirs


#: Ceiling on asking a driver or the loader something; an unbounded probe hangs the sweep.
PROBE_TIMEOUT_S: float = 15.0


def tool_stdout(cmd: list[str], timeout: float = PROBE_TIMEOUT_S) -> str | None:
    """stdout of ``cmd``, or ``None`` when it cannot run, times out or fails."""
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


@functools.lru_cache(maxsize=None, typed=True)
def driver_lib_path(soname: str, compiler: str) -> Path | None:
    """Where ``compiler`` resolves ``lib<soname>.so``, or ``None`` (a different question from what
    ldconfig/find_library find); cached since it sits on the hot flag-composition path."""
    out = (tool_stdout([compiler, f"-print-file-name=lib{soname}.so"]) or "").strip()
    if not out or out == f"lib{soname}.so":
        return None
    # normalize lexically, never resolve(): libomp.so is often a symlink into another directory
    path = Path(os.path.normpath(out))
    return path if path.exists() else None


@functools.lru_cache(maxsize=None, typed=True)
def driver_search_dirs(compiler: str) -> tuple[str, ...]:
    """Library directories ``compiler`` itself searches, via -print-search-dirs."""
    out = tool_stdout([compiler, "-print-search-dirs"]) or ""
    for line in out.splitlines():
        if line.startswith("libraries:"):
            raw = line.split(":", 1)[1].strip().lstrip("=")
            return tuple(os.path.normpath(d) for d in raw.split(os.pathsep) if d)
    return ()


#: ldconfig by name and by full path: /usr/sbin is off the non-root PATH on slim Debian images.
LDCONFIG_EXES = ("ldconfig", "/usr/sbin/ldconfig", "/sbin/ldconfig")


@functools.lru_cache(maxsize=None, typed=True)
def ldconfig_output() -> str:
    """``ldconfig -p`` output, or ``""``."""
    for exe in LDCONFIG_EXES:
        out = tool_stdout([exe, "-p"])
        if out:
            return out
    return ""


def ldconfig_dirs(soname: str) -> list[str]:
    """Directories the loader cache lists for lib<soname>. The linker needs the -dev .so symlink, which
    ldconfig does not index, but it shares a directory with the versioned .so.N ldconfig does index."""
    dirs: list[str] = []
    for line in ldconfig_output().splitlines():
        if f"lib{soname}.so" not in line or "=>" not in line:
            continue
        d = os.path.dirname(line.split("=>")[-1].strip())
        if d and d not in dirs:
            dirs.append(d)
    return dirs


#: Common install layouts, tried only after driver/loader queries come up empty; hints, not truth.
LIB_DIR_HINT_ROOTS = ("/usr/lib", "/usr/lib64")

LIB_DIR_HINTS = ("/usr/lib64", "/usr/local/lib64", "/usr/local/lib")


def llvm_version(path: Path) -> tuple[int, ...]:
    """Version tuple of an llvm-N[.M] dir, or (-1,); string sort ranks llvm-9 above llvm-21, so parsed as ints."""
    parts = path.parent.name.partition("llvm-")[2].split(".")
    if not parts or not parts[0].isdigit():
        return (-1,)
    return tuple(int(p) for p in parts if p.isdigit())


def hint_dirs() -> list[str]:
    """Guessed library dirs, newest LLVM first across all roots, ties broken by path for a stable order."""
    found = [p for root in LIB_DIR_HINT_ROOTS for p in Path(root).glob("llvm-*/lib*")]
    ranked = sorted({str(p) for p in found}, key=lambda d: (llvm_version(Path(d)), d), reverse=True)
    return ranked + [d for d in LIB_DIR_HINTS if d not in ranked]


def linker_finds(soname: str, compiler: str = DEFAULT_COMPILER) -> bool:
    return driver_lib_path(soname, compiler) is not None


#: LLVM drivers asked where an LLVM runtime lives: distributions install libomp under the LLVM prefix, off g++'s path.
LLVM_DRIVERS = ("clang++", "clang")

#: Drivers to ask where a runtime lives when the target compiler cannot find it; clang-first since libomp.
LIB_PROBE_DRIVERS = (*LLVM_DRIVERS, "g++", "gcc")


@functools.lru_cache(maxsize=None, typed=True)
def llvm_config_libdir() -> str | None:
    """``llvm-config --libdir`` of the LLVM on PATH, or ``None``."""
    if shutil.which("llvm-config") is None:
        return None
    return (tool_stdout(["llvm-config", "--libdir"]) or "").strip() or None


def shared_object_in(directory: str, soname: str) -> Path | None:
    """``lib<soname>.so`` in ``directory``, else its highest-versioned ``lib<soname>.so.N``."""
    unversioned = Path(directory) / f"lib{soname}.so"
    if unversioned.exists():
        return unversioned
    versioned = sorted(Path(directory).glob(f"lib{soname}.so.*"))
    return versioned[-1] if versioned else None


def library_search_dirs(soname: str) -> Iterator[str]:
    """Where a runtime may sit once no driver names it: the environment's library paths, each probe driver's
    own search dirs, the loader cache ranked by LLVM version, then common layouts."""
    yield from env_library_dirs()  # explicit intent (spack/module) outranks anything inferred
    for probe in LIB_PROBE_DRIVERS:
        if shutil.which(probe):
            yield from driver_search_dirs(probe)
    # ldconfig lists dirs in cache order, not version order; rank like hint_dirs or llvm-14 wins over 18
    yield from sorted(ldconfig_dirs(soname), key=lambda d: (llvm_version(Path(d)), d), reverse=True)
    yield from hint_dirs()


def runtime_library_candidates(soname: str, compiler: str) -> Iterator[Path | None]:
    yield driver_lib_path(soname, compiler)
    for driver in LLVM_DRIVERS:
        if driver != compiler and shutil.which(driver):
            yield driver_lib_path(soname, driver)
    libdir = llvm_config_libdir()
    if libdir is not None:
        yield shared_object_in(libdir, soname)
    for directory in library_search_dirs(soname):
        yield shared_object_in(directory, soname)


def runtime_library(soname: str, compiler: str) -> Path | None:
    """The shared object ``lib<soname>`` links from for ``compiler``: its own driver first, then an LLVM driver
    on PATH and ``llvm-config --libdir``, then the environment, loader and layout search. A versioned
    ``lib<soname>.so.N`` stands in for a missing ``lib<soname>.so``; ``None`` if nothing provides it."""
    return next((path for path in runtime_library_candidates(soname, compiler) if path is not None), None)


def library_flag(soname: str, compiler: str) -> str:
    """``-l<soname>``, or ``-l:<file>`` when only a versioned shared object provides the runtime."""
    found = runtime_library(soname, compiler)
    return f"-l:{found.name}" if found is not None and found.name != f"lib{soname}.so" else f"-l{soname}"


@functools.lru_cache(maxsize=None, typed=True)
def linkable_lib_dir(soname: str, compiler: str = DEFAULT_COMPILER) -> str | None:
    """The -L directory needed to link lib<soname>, or None if ``compiler`` finds it unaided or nothing provides
    it. Loader and linker search different paths, so the directory comes from :func:`runtime_library`."""
    if shutil.which(compiler) is None:
        return None  # no linker to ask; a guessed -L would be worse than none
    if linker_finds(soname, compiler):
        return None
    found = runtime_library(soname, compiler)
    return str(found.parent) if found is not None else None


def lib_linkable(soname: str, compiler: str = DEFAULT_COMPILER) -> bool:
    """True if -l<soname> resolves at link time; unlike find_library, which a versioned .so.5 satisfies too."""
    return linker_finds(soname, compiler) or linkable_lib_dir(soname, compiler) is not None


@functools.lru_cache(maxsize=None, typed=True)
def usable_openmp(compiler: str) -> OpenMPRuntime | None:
    """The OpenMP runtime ``compiler`` can link, libomp first, or ``None``. Never a bare -fopenmp: gcc and clang
    would each link their own default and a mixed-compiler program would run two thread pools."""
    for rt in OPENMP_RUNTIMES.values():  # deliberately libomp-first
        if rt.compatible(compiler) and lib_linkable(rt.soname, compiler):
            return rt
    return None


#: The two ctypes shapes a kernel parameter can take: a scalar type, or a pointer to one.
CType = type[ctypes._SimpleCData] | type[ctypes._Pointer]


@dataclass(slots=True)
class Param:
    name: str
    ctype: CType


#: ctypes' metaclass of every ``POINTER(T)``: tells a by-pointer parameter from a by-value one.
POINTER_TYPE = type(ctypes.POINTER(ctypes.c_double))


def bind_argument(arg: str, ctype: CType, buffers: dict[str, np.ndarray], sizes: dict[str, int]) -> object:
    """One ctypes argument: a buffer by pointer, a Scalar's one-element buffer by value, else a size by value."""
    if arg not in buffers:
        return ctype(int(sizes[arg]))
    if isinstance(ctype, POINTER_TYPE):
        return buffers[arg].ctypes.data_as(ctype)
    return ctype(buffers[arg].item())


def entry(so: str | Path, symbol: str, argtypes: list[CType]) -> Any:
    """The C function ``symbol`` of library ``so``, typed with ``argtypes`` and returning nothing."""
    fn = ctypes.CDLL(str(so))[symbol]  # CDLL indexing, not getattr: any symbol name binds
    fn.argtypes = argtypes
    fn.restype = None
    return fn


def parse_params(param_str: str) -> list[Param]:
    """Parse a C parameter list into typed params; skips the leading N_state_t *__state handle."""
    params: list[Param] = []
    for raw in split_params(param_str):
        # strip qualifiers as whole words: a substring strip would corrupt names like `const_term`
        tok = re.sub(r"\b(?:const|__restrict__)\b", "", raw).strip()
        if not tok or tok.endswith("_state_t *__state") or tok.endswith("_state_t* __state"):
            continue
        is_ptr = "*" in tok
        name = re.split(r"[\s*]+", tok)[-1]
        base = tok[: tok.rfind(name)].replace("*", "").strip()
        table, table_name = (C_PTR, "C_PTR") if is_ptr else (C_SCALAR, "C_SCALAR")
        # an unmapped base type would guess a width silently -- an ABI bug ctypes can't catch -- so refuse
        ctype = table.get(base)
        if ctype is None:
            raise ValueError(
                f"parameter {name!r} of entry point has {'pointer to ' if is_ptr else ''}C type {base!r}, which has "
                f"no ctypes mapping (known: {sorted(table)}); add it to {table_name}"
            )
        params.append(Param(name, ctypes.POINTER(ctype) if is_ptr else ctype))
    return params


def split_params(param_str: str) -> list[str]:
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


def raw_signature(text: str, symbol: str) -> str:
    """The parameter text of the kernel entry's definition, verbatim; :class:`LookupError` if it is absent.
    Anchored on ``void`` and the opening brace, since the bare name also matches a comment naming it."""
    # `[^)]*` not `(.*?)`: non-greedy backtracks across a preceding prototype, capturing a bogus span
    m = re.search(rf"void\s+{re.escape(symbol)}\s*\(([^)]*)\)\s*\{{", text, re.S)
    if not m:
        raise LookupError(f"entry {symbol} not found in the emitted source")
    return m.group(1)


def signature(code: str, symbol: str) -> str:
    """The parameter list of symbol(...) in code; unlike raw_signature, matches a non-void DaCe declaration."""
    m = re.search(rf"{re.escape(symbol)}\s*\((.*?)\)", code, re.S)
    if not m:
        raise LookupError(f"entry point {symbol} not found in generated code")
    return m.group(1)


@dataclass(slots=True)
class Toolchain:
    """One discovered toolchain family and its C++ compiler, which builds every variant."""

    name: str
    cxx: str

    @property
    def family(self) -> str:
        """OpenMP-runtime family of the compiler (icpx -> llvm)."""
        return compiler_family(self.cxx)

    @property
    def fp_family(self) -> str:
        """Flag-matrix FP family: Intel is its own (defaults to -fp-model=fast) despite being clang-based."""
        return "intel" if self.name == "intel" else self.family


#: family label -> its C++ compiler.
FAMILY_EXES = {"gcc": "g++", "clang": "clang++", "intel": "icpx"}
#: user tokens (compiler names/aliases) -> family label.
ALIASES = {
    "gcc": "gcc", "g++": "gcc", "gnu": "gcc",
    "clang": "clang", "clang++": "clang", "llvm": "clang",
    "icx": "intel", "icpx": "intel", "intel": "intel", "oneapi": "intel",
}  # fmt: skip


def discover_toolchains(requested: str = "auto") -> list[Toolchain]:
    """Discover toolchain families on PATH ("auto"/"all" -> gcc/clang/intel) whose C++ compiler is there."""
    tokens = list(FAMILY_EXES) if requested.strip() in ("", "auto", "all") else requested.split()
    families: list[str] = []
    for t in tokens:
        fam = ALIASES.get(t.strip())
        if fam is None:
            warnings.warn(f"unknown compiler token {t!r}; known: {sorted(ALIASES)}")
        elif fam not in families:
            families.append(fam)
    out: list[Toolchain] = []
    for fam in families:
        cxx = shutil.which(FAMILY_EXES[fam])
        if cxx is None:
            warnings.warn(f"{fam}: C++ compiler {FAMILY_EXES[fam]!r} not found on PATH; skipping this family")
            continue
        out.append(Toolchain(name=fam, cxx=cxx))
    return out


@dataclass(frozen=True, slots=True)
class CudaToolchain:
    """One nvcc found on PATH, with its CUDA release and the directory its ``libcudart`` lives in."""

    nvcc: str
    release: str
    cudart_dir: str

    @property
    def name(self) -> str:
        return f"nvcc-{self.release}"


def path_executables(exe: str) -> list[str]:
    """Every distinct ``exe`` on PATH, resolved through symlinks, in PATH order."""
    found: dict[str, None] = {}
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / exe
        if candidate.is_file() and os.access(candidate, os.X_OK):
            found.setdefault(str(candidate.resolve()), None)
    return list(found)


@functools.lru_cache(maxsize=None, typed=True)
def nvcc_release(nvcc: str) -> str:
    """``major.minor`` of the CUDA toolkit ``nvcc`` belongs to."""
    out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=60).stdout
    match = re.search(r"release (\d+\.\d+)", out)
    if match is None:
        raise LookupError(f"{nvcc} --version names no CUDA release: {out[:200]!r}")
    return match.group(1)


@functools.lru_cache(maxsize=None, typed=True)
def cudart_dir(nvcc: str) -> str:
    """The directory ``nvcc`` links ``libcudart`` from, read off its own verbose link of a probe library."""
    with tempfile.TemporaryDirectory(prefix="nf_cudart_") as scratch:
        source = Path(scratch) / "probe.cu"
        source.write_text("int nf_cudart_probe() { return 0; }\n")
        probe = str(Path(scratch) / "probe.so")
        link = [
            nvcc,
            "-v",
            "-shared",
            "-Xcompiler=-fPIC",
            nvcc_linker_flag(AS_NEEDED),
            str(source),
            "-o",
            probe,
            "-lcudart",
        ]
        proc = subprocess.run(link, capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S)
    for directory in re.findall(r"(?<=-L)\S+", proc.stdout + proc.stderr):
        candidate = Path(directory.strip('"'))
        if (candidate / "libcudart.so").exists():
            return str(candidate.resolve())
    raise LookupError(f"{nvcc} names no directory holding libcudart.so in its link line")


def nvcc_linker_flag(flag: str) -> str:
    """A ``-Wl,`` linker flag spelled for nvcc, which rejects ``-Wl,`` and forwards ``-Xlinker=`` instead."""
    return "-Xlinker=" + flag.removeprefix("-Wl,")


def discover_cuda_toolchains() -> list[CudaToolchain]:
    """Every nvcc on PATH, one toolchain per distinct compiler."""
    return [CudaToolchain(nvcc, nvcc_release(nvcc), cudart_dir(nvcc)) for nvcc in path_executables("nvcc")]


def cudart_link_flags(directory: str) -> list[str]:
    """Link ``libcudart`` from ``directory`` and find it there again at load time."""
    return [f"-L{directory}", "-lcudart", f"-Wl,-rpath,{directory}"]


def needed_libraries(shared: Path) -> list[str]:
    """The ``NEEDED`` sonames of a shared object, in ``readelf -d`` order."""
    out = subprocess.run(
        ["readelf", "-d", str(shared)], capture_output=True, text=True, check=True, timeout=PROBE_TIMEOUT_S
    ).stdout
    return re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", out)


#: The archiver; no build uses -flto, so no plugin-aware gcc-ar is needed.
AR = "ar"

#: Wall-clock ceiling for one compile, link or archive command.
COMPILE_TIMEOUT_S: float = float(os.environ.get("NF_COMPILE_TIMEOUT", "900"))

#: Distinct warning kinds reported per tool before the rest are only counted.
WARN_BUDGET: int = 5

#: tool -> the warning kinds reported, in order.
WARNED: dict[str, dict[str, None]] = {}


def warning_kinds(stderr: str) -> str:
    """The distinct [-Wflag] kinds in stderr, or its first line; keyed on kind, not text, to dedup across cells."""
    kinds = sorted({m.group(1) for m in re.finditer(r"\[-W([a-z0-9-]+)\]", stderr)})
    return ", ".join(kinds) if kinds else stderr.strip().splitlines()[0][:120]


def warn_once(tool: str, stderr: str) -> None:
    """Report a succeeding command's warnings, each kind once and at most :data:`WARN_BUDGET` kinds per tool, since
    a sweep compiles hundreds of cells."""
    kinds = warning_kinds(stderr)
    seen = WARNED.setdefault(tool, {})
    if kinds in seen or len(seen) >= WARN_BUDGET:
        return
    seen[kinds] = None
    warnings.warn(f"{tool} warnings [{kinds}]:\n{stderr[-2000:]}")


def run(cmd: list[str], timeout: float | None = COMPILE_TIMEOUT_S) -> None:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        # the child is already killed; a normal build failure lets the sweep move on
        raise RuntimeError(
            f"command timed out after {timeout:.0f}s: {' '.join(cmd[:2])} ... "
            f"(pathological compile/link; ceiling is NF_COMPILE_TIMEOUT)"
        )
    if p.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd[:2])} ...\n{p.stderr[-2000:]}")
    if p.stderr.strip():
        warn_once(Path(cmd[0]).name, p.stderr)
