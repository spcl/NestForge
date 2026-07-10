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
               flags: Optional[List[str]] = None, expand_libnodes: bool = False) -> BuiltSDFG:
    """Generate + compile + link an SDFG ourselves; return a :class:`BuiltSDFG` ready to call.

    :param expand_libnodes: expand library nodes to loops first (the "without libnodes" timing variant).
    """
    import copy
    sdfg = copy.deepcopy(sdfg)
    if expand_libnodes:
        sdfg.expand_library_nodes()
    flags = list(flags if flags is not None else DEFAULT_FLAGS)
    frame, name = generate_program_folder(sdfg, out_dir)
    folder = frame.parent.parent.parent  # <out>/src/cpu/x.cpp -> <out>
    code = frame.read_text()
    init_params = _parse_params(_signature(code, f"__dace_init_{name}"))
    prog_params = _parse_params(_signature(code, f"__program_{name}"))
    scalar_names = {a for a, d in sdfg.arrays.items() if isinstance(d, dace.data.Scalar)}

    so = out_dir / f"lib{name}.so"
    cmd = [compiler, *flags, *_include_flags(folder), str(frame), "-o", str(so)]
    comp = subprocess.run(cmd, capture_output=True, text=True)
    if comp.returncode != 0:
        raise RuntimeError(f"build failed for {name}:\n{comp.stderr[-2000:]}")
    return BuiltSDFG(name=name, so_path=so, _lib=ctypes.CDLL(str(so)),
                     _init_params=init_params, _prog_params=prog_params, _scalar_names=scalar_names)
