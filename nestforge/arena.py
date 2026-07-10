"""The arena: compile an extracted nest across a compiler x FP-mode matrix, validate each
against the numpy oracle, time it, and pick the winner per FP mode.

M0 scope: CPU, C target, compilers discovered from PATH (gcc/clang), three FP modes. Timing is
external wall-clock over repeats; OptArena's self-timed harness is an M1 upgrade.
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
from nestforge.translate import Prepared, emit_sources

# --- FP modes (the flag axis; see the plan's gramschmidt evidence) ----------------------------
_BASE = ["-O3", "-march=native", "-fPIC", "-shared"]
FP_MODES: Dict[str, List[str]] = {
    # bit-exact vs numpy: no fast-math, no FMA contraction.
    "ieee-strict": _BASE + ["-ffp-contract=off"],
    # finite-math-only relaxations that preserve IEEE rounding; FMA left on (the accuracy pivot).
    "fast-but-ieee": _BASE + ["-fno-math-errno", "-fno-trapping-math", "-fno-signed-zeros"],
    # everything goes.
    "fast-math": _BASE + ["-ffast-math"],
}
_MODE_ATOL = {"ieee-strict": 0.0, "fast-but-ieee": 1e-9, "fast-math": 1e-6}

_CANDIDATE_COMPILERS = {"gcc": "gcc", "clang": "clang"}
_CTYPE = {"float64": ctypes.c_double, "float32": ctypes.c_float, "int64": ctypes.c_int64, "int32": ctypes.c_int32}


def discover_compilers() -> Dict[str, str]:
    """M0: probe PATH for gcc/clang. (Spack discovery is M1.)"""
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


def _ldconfig_sonames() -> set:
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
    sonames = _ldconfig_sonames()
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
def _resolve_shape(shape, sizes: Dict[str, int]):
    env = {symbolic.symbol(k): v for k, v in sizes.items()}
    return tuple(int(symbolic.evaluate(d, env)) for d in shape)


def _scratch_names(boundary: Boundary) -> List[str]:
    """Transient array buffers the C-style kernel expects the caller to pre-allocate."""
    sdfg = boundary.standalone_sdfg
    return sorted(name for name, desc in sdfg.arrays.items()
                  if desc.transient and not (isinstance(desc, dace.data.Scalar) or desc.total_size == 1))


def make_inputs(boundary: Boundary, sizes: Dict[str, int], seed: int = 0) -> Dict[str, np.ndarray]:
    """Random arrays for inputs; zeros for outputs and scratch buffers (all caller-pre-allocated)."""
    sdfg = boundary.standalone_sdfg
    rng = np.random.default_rng(seed)
    arrays: Dict[str, np.ndarray] = {}
    out_only = [o for o in boundary.outputs if o not in boundary.inputs]
    zero_filled = out_only + [s for s in _scratch_names(boundary) if s not in boundary.inputs]
    for name in list(boundary.inputs) + zero_filled:
        desc = sdfg.arrays[name]
        shape = _resolve_shape(desc.shape, sizes)
        dt = np.dtype(desc.dtype.type)
        arrays[name] = (np.zeros(shape, dt) if name in zero_filled else rng.random(shape).astype(dt))
    return arrays


def run_oracle(prep: Prepared, boundary: Boundary, inputs: Dict[str, np.ndarray],
               sizes: Dict[str, int]) -> Dict[str, np.ndarray]:
    """Run the emitted numpy kernel to get reference outputs."""
    missing = [s for s in boundary.symbols if s not in sizes]
    if missing:
        raise KeyError(f"no value for boundary symbol(s) {missing} (e.g. a loop index carried into an "
                       f"extracted nest); pass them in `sizes`")
    ns: Dict[str, object] = {}
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


def _argtypes(prep: Prepared, boundary: Boundary) -> list:
    sdfg = boundary.standalone_sdfg
    types = []
    for arg in prep.manifest["input_args"]:
        if arg in sdfg.arrays:
            dt = np.dtype(sdfg.arrays[arg].dtype.type).name
            types.append(ctypes.POINTER(_CTYPE[dt]))
        else:
            types.append(ctypes.c_int64)  # size symbol
    return types


def _call_native(so: Path, symbol: str, argtypes: list, prep: Prepared, boundary: Boundary,
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
                out.append(ctypes.c_int64(int(sizes[arg])))
        return out

    fn(*build_args())  # correctness run
    outputs = {o: work[o].copy() for o in boundary.outputs}
    t0 = time.perf_counter()  # timing runs
    for _ in range(reps):
        fn(*build_args())
    elapsed_us = (time.perf_counter() - t0) / reps * 1e6
    return outputs, elapsed_us


def _maxdiff(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> float:
    return max((float(np.max(np.abs(a[k] - b[k]))) if a[k].size else 0.0) for k in a)


@dataclass
class ArenaResult:
    name: str
    cells: List[Cell] = field(default_factory=list)
    winners: Dict[str, Cell] = field(default_factory=dict)  # fp_mode -> best correct cell
    #: total wall time of the optimization sweep (all compiler x FP-mode candidates: compile + validate +
    #: time). This is the "total optimization time" -- the cost of searching for the winners.
    optimization_seconds: float = 0.0


def run_arena(prep: Prepared,
              boundary: Boundary,
              c_source: Path,
              out_dir: Path,
              sizes: Dict[str, int],
              reps: int = 100,
              seed: int = 0) -> ArenaResult:
    """Sweep discovered compilers x FP modes; validate + time each; pick a winner per FP mode."""
    out_dir.mkdir(parents=True, exist_ok=True)
    compilers = discover_compilers()
    symbol = f"{prep.name}_fp64"
    argtypes = _argtypes(prep, boundary)
    inputs = make_inputs(boundary, sizes, seed=seed)
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
            try:
                outs, us = _call_native(so, symbol, argtypes, prep, boundary, inputs, sizes, reps)
                md = _maxdiff(oracle, outs)
                ok = md <= _MODE_ATOL[mode]
                result.cells.append(Cell(cname, mode, ok, md, us, compile_us=compile_us, so_path=str(so),
                                         symbol=symbol))
            except Exception as e:  # pragma: no cover - defensive
                result.cells.append(
                    Cell(cname, mode, False, float("inf"), float("inf"), compile_us=compile_us, error=str(e)))

    result.optimization_seconds = time.perf_counter() - t_sweep
    for mode in FP_MODES:
        correct = [c for c in result.cells if c.fp_mode == mode and c.ok]
        if correct:
            result.winners[mode] = min(correct, key=lambda c: c.time_us)
    return result
