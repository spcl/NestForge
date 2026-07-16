"""The arena: compile an extracted nest across a compiler x FP-mode matrix, validate each
against the numpy oracle, time it, and pick the winner per FP mode.

Scope: CPU, C target, compilers discovered from PATH (gcc/clang), three FP modes. Timing is
external wall-clock over repeats.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import dace
from dace import symbolic

from nestforge.extract import Boundary
from nestforge.isolation import run_isolated
from nestforge.translate import Prepared

# --- FP modes (the flag axis) -----------------------------------------------------------------
_BASE = ["-O3", "-march=native", "-fPIC", "-shared"]
FP_MODES: Dict[str, List[str]] = {
    # bit-exact vs numpy: no fast-math, no FMA contraction.
    "ieee-strict": _BASE + ["-ffp-contract=off"],
    # finite-math-only relaxations that preserve IEEE rounding; FMA left on (the accuracy pivot).
    "fast-but-ieee": _BASE + ["-fno-math-errno", "-fno-trapping-math", "-fno-signed-zeros"],
    # everything goes.
    "fast-math": _BASE + ["-ffast-math"],
}
MODE_ATOL = {"ieee-strict": 0.0, "fast-but-ieee": 1e-9, "fast-math": 1e-6}

_CANDIDATE_COMPILERS = {"gcc": "gcc", "clang": "clang"}
# numpy dtype name -> ctypes scalar for the emitted kernel's ABI. ``bool`` is here because a comparison
# in the kernel (e.g. s13110's ``aa[i, j] > maxv``) materialises a boolean buffer/transient, and DaCe
# lowers ``dace.bool`` to a 1-byte C ``bool`` (== ctypes.c_bool).
CTYPE = {
    "float64": ctypes.c_double,
    "float32": ctypes.c_float,
    "int64": ctypes.c_int64,
    "int32": ctypes.c_int32,
    "bool": ctypes.c_bool
}


def discover_compilers() -> Dict[str, str]:
    """Probe PATH for gcc/clang."""
    return {name: shutil.which(exe) for name, exe in _CANDIDATE_COMPILERS.items() if shutil.which(exe)}


# --- BLAS backends (a link axis for matmul-heavy kernels) -------------------------------------
@dataclass
class BlasBackend:
    """An installed BLAS implementation and the link flags that select it."""
    name: str
    link_flags: List[str]


#: BLAS backend -> candidate ``lib<soname>.so`` names, most specific first.
_BLAS_SONAMES = {
    "openblas": ["openblas"],
    "blis": ["blis"],
    "atlas": ["tatlas", "satlas"],
    "mkl": ["mkl_rt"],
    # A generic ``libblas.so`` -- could be reference/netlib OR an alternatives symlink to another
    # provider (OpenBLAS/ATLAS), so it is labelled by the soname, not assumed to be reference.
    "blas": ["blas"],
}


def ldconfig_sonames() -> set:
    """Base library names (``openblas`` from ``libopenblas.so.0``) known to the dynamic linker."""
    try:
        out = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    names = set()
    for line in out.splitlines():
        token = line.strip().split(" ", 1)[0]
        if token.startswith("lib") and ".so" in token:
            names.add(token[3:token.index(".so")])
    return names


def discover_blas_libraries() -> Dict[str, BlasBackend]:
    """Discover installed BLAS backends (OpenBLAS / MKL / BLIS / ATLAS / netlib) + their link flags.

    Backends become an extra link axis for kernels whose emitted numpy uses ``@``/``np.dot`` (a
    ``cblas``-calling variant links against each). Probes the dynamic linker cache plus ``MKLROOT``.
    """
    sonames = ldconfig_sonames()
    found: Dict[str, BlasBackend] = {}
    for name, candidates in _BLAS_SONAMES.items():
        for so in candidates:
            if so in sonames:
                found[name] = BlasBackend(name, [f"-l{so}"])
                break
    mklroot = os.environ.get("MKLROOT")
    if "mkl" not in found and mklroot:
        # oneAPI 2024+ put the libraries directly under lib/; older layouts use lib/intel64.
        for libdir in (Path(mklroot) / "lib", Path(mklroot) / "lib" / "intel64"):
            if (libdir / "libmkl_rt.so").exists():
                found["mkl"] = BlasBackend("mkl", [f"-L{libdir}", "-lmkl_rt"])
                break
    return found


# --- data generation from the manifest --------------------------------------------------------
def resolve_shape(shape, sizes: Dict[str, int]):
    env = {symbolic.symbol(k): v for k, v in sizes.items()}
    return tuple(int(symbolic.evaluate(d, env)) for d in shape)


def scratch_names(boundary: Boundary) -> List[str]:
    """Transient array buffers the C-style kernel expects the caller to pre-allocate."""
    sdfg = boundary.standalone_sdfg
    return sorted(name for name, desc in sdfg.arrays.items()
                  if desc.transient and not (isinstance(desc, dace.data.Scalar) or desc.total_size == 1))


#: Upper bound of the random-input range ``[0, INPUT_HIGH)``. Kept BELOW ``1/4`` so a squaring recurrence
#: ``x = x*x + b`` (TSVC s232, a triangular in-place nest) stays bounded: ``b < 1/4`` guarantees the map has
#: an attracting fixed point, so the chain converges instead of overflowing to ``inf`` (which then poisons
#: the maxdiff with a ``nan`` and spuriously fails validation). Magnitude is otherwise irrelevant to what the
#: arena measures -- validation is bit-exact at any scale and timing is magnitude-invariant -- and staying
#: non-negative keeps ``sqrt``/``log`` kernels real (a symmetric range would NaN them).
INPUT_HIGH = 0.25


def make_inputs(boundary: Boundary,
                sizes: Dict[str, int],
                seed: int = 0,
                given: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
    """Random arrays for inputs; zeros for outputs and scratch buffers (all caller-pre-allocated).

    Inputs are drawn from ``[0, INPUT_HIGH)`` -- see :data:`INPUT_HIGH` for why the range is conditioned.

    ``given`` supplies ready-made values for named input arrays, for the values a uniform float fill
    cannot express -- chiefly the manifest-declared index arrays of :func:`nestforge.tsvc.index_fills`,
    whose entries must be valid subscripts. Every array absent from ``given`` is filled as usual. A
    ``given`` array is checked against the resolved shape and dtype: it crosses the ABI as the kernel's
    own buffer, so a mismatch would corrupt memory instead of raising.
    """
    sdfg = boundary.standalone_sdfg
    rng = np.random.default_rng(seed)
    given = given or {}
    arrays: Dict[str, np.ndarray] = {}
    out_only = [o for o in boundary.outputs if o not in boundary.inputs]
    zero_filled = out_only + [s for s in scratch_names(boundary) if s not in boundary.inputs]
    for name in list(boundary.inputs) + zero_filled:
        desc = sdfg.arrays[name]
        shape = resolve_shape(desc.shape, sizes)
        dt = np.dtype(desc.dtype.type)
        if name in given:
            value = given[name]
            if value.shape != shape or value.dtype != dt:
                raise ValueError(f"given array {name!r} is {value.dtype}{value.shape}, but the nest declares "
                                 f"{dt}{shape}; it is passed straight across the ABI, so it must match exactly")
            arrays[name] = value.copy()
        else:
            arrays[name] = (np.zeros(shape, dt) if name in zero_filled else (rng.random(shape) * INPUT_HIGH).astype(dt))
    return arrays


def run_oracle(prep: Prepared, boundary: Boundary, inputs: Dict[str, np.ndarray],
               sizes: Dict[str, int]) -> Dict[str, np.ndarray]:
    """Run the emitted numpy kernel to get reference outputs."""
    missing = [s for s in boundary.symbols if s not in sizes]
    if missing:
        raise KeyError(f"no value for boundary symbol(s) {missing} (e.g. a loop index carried into an "
                       f"extracted nest); pass them in `sizes`")
    ns: Dict[str, object] = {"np": np}  # the emitter spells casts/intrinsics as ``np.*``; the exec scope must carry it
    exec(prep.numpy_source, ns)
    args = {k: v.copy() for k, v in inputs.items()}
    call = {**args, **{s: int(sizes[s]) for s in boundary.symbols}}
    ns[prep.name](**call)
    return {o: args[o] for o in boundary.outputs}


# --- compile + call ---------------------------------------------------------------------------
@dataclass
class Cell:
    compiler: str
    fp_mode: str
    ok: bool
    maxdiff: float
    time_us: float
    compile_us: float = 0.0  # wall time of THIS candidate's compile (the post-optimization toolchain cost)
    so_path: Optional[str] = None
    symbol: Optional[str] = None
    error: Optional[str] = None


def scalar_ctype(sdfg, name: str):
    """ctypes type of a by-value (non-array) kernel arg, matching the translator's signature.

    A FLOAT value scalar (a staged ``a_index = a[i]`` read that leaked into the boundary) is declared
    ``double`` by the translator (from ``init.scalars``) -> ``c_double``. EVERY integer sizing / index
    symbol is emitted ``int64_t`` by the translator regardless of the SDFG's own int width (a 32-bit dace
    ``int`` still crosses the ABI as ``int64_t``), so it must be ``c_int64`` here -- passing it as a 32-bit
    ``c_int`` would leave the upper half of the register garbage and blow the loop bound out of range."""
    if name in sdfg.symbols and np.dtype(sdfg.symbols[name].type).kind == "f":
        return ctypes.c_double
    return ctypes.c_int64


def resolve_argtypes(prep: Prepared, boundary: Boundary) -> list:
    sdfg = boundary.standalone_sdfg
    types = []
    for arg in prep.manifest["input_args"]:
        if arg in sdfg.arrays:
            dt = np.dtype(sdfg.arrays[arg].dtype.type).name
            types.append(ctypes.POINTER(CTYPE[dt]))
        else:
            types.append(scalar_ctype(sdfg, arg))
    return types


def call_native(so: Path, symbol: str, argtypes: list, prep: Prepared, boundary: Boundary,
                inputs: Dict[str, np.ndarray], sizes: Dict[str, int], reps: int):
    lib = ctypes.CDLL(str(so))
    fn = lib[symbol]  # ctypes CDLL indexing (not getattr) to bind the kernel symbol
    fn.argtypes = argtypes
    fn.restype = None
    work = {k: v.copy() for k, v in inputs.items()}

    def build_args():
        out = []
        for arg, at in zip(prep.manifest["input_args"], argtypes):
            if arg in work:
                out.append(work[arg].ctypes.data_as(at))
            else:
                out.append(at(sizes[arg]))  # at is the by-value ctype (c_int64 size / c_double value scalar)
        return out

    fn(*build_args())  # correctness run
    outputs = {o: work[o].copy() for o in boundary.outputs}
    t0 = time.perf_counter()  # timing runs
    for _ in range(reps):
        fn(*build_args())
    elapsed_us = (time.perf_counter() - t0) / reps * 1e6
    return outputs, elapsed_us


def maxdiff(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> float:
    return max((float(np.max(np.abs(a[k] - b[k]))) if a[k].size else 0.0) for k in a)


@dataclass
class ArenaResult:
    name: str
    cells: List[Cell] = field(default_factory=list)
    winners: Dict[str, Cell] = field(default_factory=dict)  # fp_mode -> best correct cell
    #: wall time of the whole sweep (all candidates: compile + validate + time) -- the search cost.
    optimization_seconds: float = 0.0


def run_arena(prep: Prepared,
              boundary: Boundary,
              c_source: Path,
              out_dir: Path,
              sizes: Dict[str, int],
              reps: int = 100,
              seed: int = 0,
              given: Optional[Dict[str, np.ndarray]] = None) -> ArenaResult:
    """Sweep discovered compilers x FP modes; validate + time each; pick a winner per FP mode.

    ``given`` is forwarded to :func:`make_inputs`: this layer is corpus-agnostic (it never sees a kernel),
    so a caller measuring a corpus kernel passes ``tsvc.index_fills(...)`` here -- without it a declared
    integer index array fills to all-zeros and the sweep times a degenerate gather while validating
    vacuously."""
    out_dir.mkdir(parents=True, exist_ok=True)
    compilers = discover_compilers()
    symbol = f"{prep.name}_fp64"
    argtypes = resolve_argtypes(prep, boundary)
    inputs = make_inputs(boundary, sizes, seed=seed, given=given)
    oracle = run_oracle(prep, boundary, inputs, sizes)

    result = ArenaResult(name=prep.name)
    t_sweep = time.perf_counter()
    for cname, cpath in compilers.items():
        for mode, flags in FP_MODES.items():
            so = out_dir / f"lib{prep.name}_{cname}_{mode}.so"
            cmd = [cpath, *flags, str(c_source), "-o", str(so)]
            t_c = time.perf_counter()
            comp = subprocess.run(cmd, capture_output=True, text=True)
            compile_us = (time.perf_counter() - t_c) * 1e6
            if comp.returncode != 0:
                result.cells.append(
                    Cell(cname,
                         mode,
                         False,
                         float("inf"),
                         float("inf"),
                         compile_us=compile_us,
                         error=comp.stderr[-400:]))
                continue

            # Validate + time in a forked child so a segfault/runaway in the fresh kernel kills only
            # the child. maxdiff is computed child-side (only the summary crosses the pipe). ``so``/
            # ``mode`` are default args to dodge late-binding-closure capture of the loop variables.
            def work(so=so, mode=mode):
                outs, us = call_native(so, symbol, argtypes, prep, boundary, inputs, sizes, reps)
                md = float(maxdiff(oracle, outs))
                return {"ok": bool(md <= MODE_ATOL[mode]), "maxdiff": md, "time_us": float(us)}

            res = run_isolated(work)
            if "error" in res:
                result.cells.append(
                    Cell(cname, mode, False, float("inf"), float("inf"), compile_us=compile_us, error=res["error"]))
            else:
                result.cells.append(
                    Cell(cname,
                         mode,
                         res["ok"],
                         res["maxdiff"],
                         res["time_us"],
                         compile_us=compile_us,
                         so_path=str(so),
                         symbol=symbol))

    result.optimization_seconds = time.perf_counter() - t_sweep
    for mode in FP_MODES:
        correct = [c for c in result.cells if c.fp_mode == mode and c.ok]
        if correct:
            result.winners[mode] = min(correct, key=lambda c: c.time_us)
    return result
