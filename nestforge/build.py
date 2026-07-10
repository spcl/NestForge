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
import re
import subprocess
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


def build_sdfg(sdfg: dace.SDFG, out_dir: Path, compiler: str = DEFAULT_COMPILER,
               flags: Optional[List[str]] = None, expand_libnodes: bool = False,
               openmp: Optional[OpenMPRuntime] = None) -> BuiltSDFG:
    """Generate + compile + link an SDFG ourselves; return a :class:`BuiltSDFG` ready to call.

    :param expand_libnodes: expand library nodes to loops first (the "without libnodes" timing variant).
    :param openmp: when set, add this runtime's per-compiler OpenMP compile+link flags (a SEPARATE axis
        from ``flags``) so a parallel kernel links against the one mandated runtime -- and a set built
        with different compilers still shares it (e.g. flang ``-fopenmp=libomp``).
    """
    import copy
    sdfg = copy.deepcopy(sdfg)
    if expand_libnodes:
        sdfg.expand_library_nodes()
    flags = list(flags if flags is not None else DEFAULT_FLAGS)
    omp = (openmp.compile_flags(compiler) + openmp.link_flags(compiler)) if openmp else []
    frame, name = generate_program_folder(sdfg, out_dir)
    folder = frame.parent.parent.parent  # <out>/src/cpu/x.cpp -> <out>
    code = frame.read_text()
    init_params = _parse_params(_signature(code, f"__dace_init_{name}"))
    prog_params = _parse_params(_signature(code, f"__program_{name}"))
    scalar_names = {a for a, d in sdfg.arrays.items() if isinstance(d, dace.data.Scalar)}

    so = out_dir / f"lib{name}.so"
    cmd = [compiler, *flags, *omp, *_include_flags(folder), str(frame), "-o", str(so)]
    comp = subprocess.run(cmd, capture_output=True, text=True)
    if comp.returncode != 0:
        raise RuntimeError(f"build failed for {name}:\n{comp.stderr[-2000:]}")
    return BuiltSDFG(name=name, so_path=so, _lib=ctypes.CDLL(str(so)),
                     _init_params=init_params, _prog_params=prog_params, _scalar_names=scalar_names)
