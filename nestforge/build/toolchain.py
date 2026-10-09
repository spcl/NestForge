# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The compilers on PATH, the one OpenMP runtime every build links (LLVM libomp), CUDA's runtime, and the C
signature parsing that binds a kernel entry through ctypes. Imports no DaCe; subprocess probes are cached."""

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

import numpy as np

C_SCALAR = {
    "int32_t": ctypes.c_int32,
    "int64_t": ctypes.c_int64,
    "int": ctypes.c_int,
    "float": ctypes.c_float,
    "double": ctypes.c_double,
    "bool": ctypes.c_bool,
}

#: Pointer element types; a complex buffer binds as a pointer to its component type.
C_PTR = {**C_SCALAR, "dace::complex64": ctypes.c_float, "dace::complex128": ctypes.c_double}

DEFAULT_COMPILER = "g++"

#: Records a shared library in DT_NEEDED only when something references it; every link passes it explicitly.
AS_NEEDED = "-Wl,--as-needed"

CXX_STD = "c++20"

DEFAULT_FLAGS = ["-O3", "-march=native", f"-std={CXX_STD}", "-fPIC", "-shared"]

#: The archiver; no build uses -flto, so no plugin-aware gcc-ar is needed.
AR = "ar"

#: Wall-clock ceiling for one compile, link or archive command.
COMPILE_TIMEOUT_S: float = float(os.environ.get("NF_COMPILE_TIMEOUT", "900"))

#: Ceiling on asking a driver something; an unbounded probe hangs the sweep.
PROBE_TIMEOUT_S: float = 15.0

#: Characters of a tool's error output an exception or warning quotes.
ERROR_TAIL = 2000

#: Words of a failed command an error names: the tool and its first argument.
COMMAND_WORDS = 2

#: The OpenMP runtime every kernel and the program link, so gcc- and clang-built code share one thread pool:
#: libomp carries a GOMP compatibility layer for gcc's calls.
LIBOMP = "omp"


@functools.lru_cache(maxsize=None, typed=True)
def compiler_family(compiler: str) -> str:
    """``llvm`` (clang, flang, icx, icpx, ifx) or ``gnu`` (gcc, g++, gfortran, default)."""
    base = Path(compiler).name.lower()
    return "llvm" if "clang" in base or "flang" in base or base.startswith(("icx", "icpx", "ifx")) else "gnu"


def tool_stdout(cmd: list[str], timeout: float = PROBE_TIMEOUT_S) -> str | None:
    """stdout of ``cmd``, or ``None`` when it cannot run, times out or fails."""
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


@functools.lru_cache(maxsize=None, typed=True)
def driver_lib_path(soname: str, compiler: str) -> Path | None:
    """Where ``compiler`` itself resolves ``lib<soname>.so``, or ``None``."""
    out = (tool_stdout([compiler, f"-print-file-name=lib{soname}.so"]) or "").strip()
    if not out or out == f"lib{soname}.so":
        return None
    # lexically, never resolve(): libomp.so is often a symlink into another directory
    path = Path(os.path.normpath(out))
    return path if path.exists() else None


def shared_object_in(directory: str, soname: str) -> Path | None:
    """``lib<soname>.so`` in ``directory``, else its highest-versioned ``lib<soname>.so.N``."""
    unversioned = Path(directory) / f"lib{soname}.so"
    if unversioned.exists():
        return unversioned
    versioned = sorted(Path(directory).glob(f"lib{soname}.so.*"))
    return versioned[-1] if versioned else None


def search_dirs() -> list[str]:
    """``llvm-config --libdir``, then ``LD_LIBRARY_PATH`` and ``LIBRARY_PATH``: where a runtime sits that no
    driver names (an LLVM install off the default path, a spack or module runtime)."""
    dirs = [(tool_stdout(["llvm-config", "--libdir"]) or "").strip()] if shutil.which("llvm-config") else []
    for var in ("LD_LIBRARY_PATH", "LIBRARY_PATH"):
        dirs += os.environ.get(var, "").split(os.pathsep)
    return [d for d in dirs if d]


@functools.lru_cache(maxsize=None, typed=True)
def runtime_library(soname: str, compiler: str) -> Path | None:
    """The shared object ``lib<soname>`` links from for ``compiler``: its own driver first, then an LLVM driver on
    PATH, then :func:`search_dirs`; ``None`` if nothing provides it."""
    for driver in dict.fromkeys((compiler, "clang++", "clang")):
        found = driver_lib_path(soname, driver) if shutil.which(driver) else None
        if found is not None:
            return found
    return next((path for d in search_dirs() if (path := shared_object_in(d, soname)) is not None), None)


def lib_linkable(soname: str, compiler: str = DEFAULT_COMPILER) -> bool:
    """Whether ``-l<soname>`` resolves at link time for ``compiler``, with the flags :func:`library_flags` adds."""
    return runtime_library(soname, compiler) is not None


def library_flags(soname: str, compiler: str) -> list[str]:
    """Link ``lib<soname>`` and find it again at load time: ``-L`` and ``-rpath`` only when ``compiler`` does not
    find it unaided, and ``-l:<file>`` when only a versioned shared object provides it."""
    found = runtime_library(soname, compiler)
    if found is None:
        return [f"-l{soname}"]
    library = f"-l{soname}" if found.name == f"lib{soname}.so" else f"-l:{found.name}"
    if driver_lib_path(soname, compiler) is not None:
        return [library]
    return [f"-L{found.parent}", f"-Wl,-rpath,{found.parent}", library]


def openmp_compile_flags(compiler: str) -> list[str]:
    """Compile with OpenMP against libomp: an LLVM compiler selects it by name, gcc fixes it at link time."""
    return ["-fopenmp=libomp"] if compiler_family(compiler) == "llvm" else ["-fopenmp"]


def openmp_link_flags(compiler: str) -> list[str]:
    """Link libomp and no other runtime; a bare ``-fopenmp`` would pull in gcc's libgomp."""
    flags = library_flags(LIBOMP, compiler)
    if compiler_family(compiler) == "llvm":
        return ["-fopenmp=libomp", *[f for f in flags if not f.startswith("-l")]]
    return flags


@functools.lru_cache(maxsize=None, typed=True)
def support_rpath_flags(compiler: str) -> tuple[str, ...]:
    """``-rpath`` for the support libraries icx links from off the loader path (svml, imf, irng, intlc)."""
    found = driver_lib_path("svml", compiler)
    return (f"-Wl,-rpath,{found.parent}",) if found else ()


#: The two ctypes shapes a kernel parameter can take: a scalar type, or a pointer to one.
CType = type[ctypes._SimpleCData] | type[ctypes._Pointer]

#: ctypes' metaclass of every ``POINTER(T)``: tells a by-pointer parameter from a by-value one.
POINTER_TYPE = type(ctypes.POINTER(ctypes.c_double))


@dataclass(slots=True)
class Param:
    name: str
    ctype: CType


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
        # an unmapped base type would guess a width silently, an ABI bug ctypes can't catch
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
    """The parameter text of the kernel entry's definition; :class:`LookupError` if it is absent. Anchored on
    ``void`` and the opening brace, since the bare name also matches a comment naming it."""
    # `[^)]*` not `(.*?)`: non-greedy backtracks across a preceding prototype, capturing a bogus span
    m = re.search(rf"void\s+{re.escape(symbol)}\s*\(([^)]*)\)\s*\{{", text, re.DOTALL)
    if not m:
        raise LookupError(f"entry {symbol} not found in the emitted source")
    return m.group(1)


def signature(code: str, symbol: str) -> str:
    """The parameter list of symbol(...) in code; unlike raw_signature, matches a non-void DaCe declaration."""
    m = re.search(rf"{re.escape(symbol)}\s*\((.*?)\)", code, re.DOTALL)
    if not m:
        raise LookupError(f"entry point {symbol} not found in generated code")
    return m.group(1)


@dataclass(slots=True)
class Toolchain:
    """A C++ compiler on PATH and the family name reports use."""

    name: str
    cxx: str

    @property
    def fp_family(self) -> str:
        """Flag-matrix FP family: Intel is its own (defaults to -fp-model=fast) despite being clang-based."""
        return "intel" if self.name == "intel" else compiler_family(self.cxx)


#: family name -> its C++ compiler.
FAMILY_EXES = {"gcc": "g++", "clang": "clang++", "intel": "icpx"}


def discover_toolchains() -> list[Toolchain]:
    """Every family whose C++ compiler is on PATH."""
    return [Toolchain(fam, cxx) for fam, exe in FAMILY_EXES.items() if (cxx := shutil.which(exe)) is not None]


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
    out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=PROBE_TIMEOUT_S).stdout
    match = re.search(r"release (\d+\.\d+)", out)
    if match is None:
        raise LookupError(f"{nvcc} --version names no CUDA release: {out[:ERROR_TAIL]!r}")
    return match.group(1)


def nvcc_linker_flag(flag: str) -> str:
    """A ``-Wl,`` linker flag spelled for nvcc, which rejects ``-Wl,`` and forwards ``-Xlinker=`` instead."""
    return "-Xlinker=" + flag.removeprefix("-Wl,")


@functools.lru_cache(maxsize=None, typed=True)
def cudart_dir(nvcc: str) -> str:
    """The directory ``nvcc`` links ``libcudart`` from, read off its own verbose link of a probe library."""
    with tempfile.TemporaryDirectory(prefix="nf_cudart_") as scratch:
        source = Path(scratch) / "probe.cu"
        source.write_text("int nf_cudart_probe() { return 0; }\n")
        link = [nvcc, "-v", "-shared", "-Xcompiler=-fPIC", nvcc_linker_flag(AS_NEEDED), str(source)]
        proc = subprocess.run(
            [*link, "-o", str(Path(scratch) / "probe.so"), "-lcudart"],
            capture_output=True,
            text=True,
            timeout=COMPILE_TIMEOUT_S,
        )
    for directory in re.findall(r"(?<=-L)\S+", proc.stdout + proc.stderr):
        candidate = Path(directory.strip('"'))
        if (candidate / "libcudart.so").exists():
            return str(candidate.resolve())
    raise LookupError(f"{nvcc} names no directory holding libcudart.so in its link line")


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


#: Distinct warning kinds reported per tool before the rest are only counted.
WARN_BUDGET: int = 5

#: tool -> the warning kinds reported, in order.
WARNED: dict[str, dict[str, None]] = {}


def warning_kinds(stderr: str) -> str:
    """The distinct [-Wflag] kinds in stderr, or its first line; keyed on kind, not text, to dedup across cells."""
    kinds = sorted(dict.fromkeys(m.group(1) for m in re.finditer(r"\[-W([a-z0-9-]+)\]", stderr)))
    return ", ".join(kinds) if kinds else stderr.strip().splitlines()[0][:ERROR_TAIL]


def warn_once(tool: str, stderr: str) -> None:
    """Report a succeeding command's warnings, each kind once and at most :data:`WARN_BUDGET` kinds per tool, since
    a sweep compiles hundreds of cells."""
    kinds = warning_kinds(stderr)
    seen = WARNED.setdefault(tool, {})
    if kinds in seen or len(seen) >= WARN_BUDGET:
        return
    seen[kinds] = None
    warnings.warn(f"{tool} warnings [{kinds}]:\n{stderr[-ERROR_TAIL:]}")


def run(cmd: list[str], timeout: float | None = COMPILE_TIMEOUT_S) -> None:
    """Run a build command, reporting its warnings through :func:`warn_once`. A failure or timeout raises
    :class:`RuntimeError`, so a sweep moves on to its next cell."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"command timed out after {timeout:.0f}s: {' '.join(cmd[:COMMAND_WORDS])} ... (the ceiling is NF_COMPILE_TIMEOUT)"
        )
    if p.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd[:COMMAND_WORDS])} ...\n{p.stderr[-ERROR_TAIL:]}")
    if p.stderr.strip():
        warn_once(Path(cmd[0]).name, p.stderr)
