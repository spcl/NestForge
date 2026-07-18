"""Shared arena infrastructure: rank partitioning, bounded compiles, ABI binding, and result IO.

Every perf driver and every ``perf/plot_*.py`` script needs these same primitives. They used to live
inside ``tsvc_arena`` and get copy-pasted (10x ``geomean``, 7x ``finite``, 6x ``load_results``); this
module is that infra relocated to one home, functions unchanged.
"""
from __future__ import annotations

import ctypes
import json
import math
import os
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from nestforge.arena import CTYPE, scalar_ctype

#: Per-compile wall-clock ceiling (seconds). Compiles run in a ThreadPoolExecutor; an untimed compiler
#: spinning on a pathological kernel would hang the pool and freeze the whole sweep rank (the 19h "job
#: timed out" stall). On timeout we return a normal failed result. Override with ``NF_COMPILE_TIMEOUT``.
COMPILE_TIMEOUT_S: float = float(os.environ.get("NF_COMPILE_TIMEOUT", "900"))

#: Per-kernel *execution* wall-clock ceiling (seconds), distinct from the compile ceiling above. A
#: runaway kernel would otherwise hold the fork open for the whole job. Override with ``NF_RUN_TIMEOUT``.
RUN_TIMEOUT_S: float = float(os.environ.get("NF_RUN_TIMEOUT", "1800"))

#: base C-type name -> ctypes type, for binding a native baseline signature.
C_BASE = {"double": ctypes.c_double, "float": ctypes.c_float, "int64_t": ctypes.c_int64, "int": ctypes.c_int}

#: rank / size env vars, most specific launcher first (SLURM, then OpenMPI, then MPICH/Hydra), so a
#: driver self-partitions the same under any launcher.
RANK_VARS = ("SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK")
SIZE_VARS = ("SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE")


# --- rank self-partition ----------------------------------------------------------------------------
def my_slice(items: List, procid: int, ntasks: int) -> List:
    """This rank's disjoint stride of the kernel list (round-robin balances long/short kernels)."""
    return items[procid::ntasks] if ntasks > 1 else items


def rank_and_size() -> Tuple[int, int]:
    """``(rank, nranks)`` from the launcher env, defaulting to ``(0, 1)`` for a single process.

    Fails loud on an asymmetric environment (rank var set, no size var) -- silently defaulting size to 1
    would make every rank process the whole list, N-fold duplicated work with concurrent result writes."""
    rank_var = next((v for v in RANK_VARS if os.environ.get(v)), None)
    size_var = next((v for v in SIZE_VARS if os.environ.get(v)), None)
    if rank_var is not None and size_var is None:
        raise RuntimeError(f"launcher set a rank ({rank_var}={os.environ[rank_var]}) but no recognized size variable "
                           f"({list(SIZE_VARS)}); cannot partition safely -- every rank would run the whole list. Set "
                           "the matching size env var, or run without a launcher for a single process.")
    if size_var is not None and rank_var is None:
        # the quieter mirror asymmetry: defaulting rank to 0 with size N silently processes only 1/N of
        # the corpus. `sbatch --ntasks=4` without srun sets SLURM_NTASKS without SLURM_PROCID here.
        raise RuntimeError(f"launcher set a size ({size_var}={os.environ[size_var]}) but no recognized rank variable "
                           f"({list(RANK_VARS)}); cannot partition safely -- this process would measure only "
                           f"1/{os.environ[size_var]} of the list and report it as the whole. Set the matching rank "
                           "env var (srun/mpirun set both), or run without a launcher for a single process.")
    rank = int(os.environ[rank_var]) if rank_var else 0
    size = int(os.environ[size_var]) if size_var else 1
    return rank, max(size, 1)


# --- bounded compile --------------------------------------------------------------------------------
def run_compile(cmd: List[str]) -> Tuple[bool, float, Optional[str]]:
    """Run one compile under :data:`COMPILE_TIMEOUT_S`; ``(ok, microseconds, stderr_tail)``."""
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        dt = (time.perf_counter() - t0) * 1e6
        return False, dt, f"compile timed out after {COMPILE_TIMEOUT_S:.0f}s (NF_COMPILE_TIMEOUT)"
    dt = (time.perf_counter() - t0) * 1e6
    return (p.returncode == 0), dt, (None if p.returncode == 0 else p.stderr[-400:])


# --- ABI binding ------------------------------------------------------------------------------------
def signature_order(text: str, symbol: str, lang: str = "c") -> List[str]:
    """Parameter names of the kernel entry, in declaration order, for the language's syntax.

    numpyto's C parameter order (sorted arrays, then symbols) is NOT the manifest ``input_args`` order,
    so args must bind to the actual signature or a size lands in a pointer slot. An empty/``void`` list
    yields ``[]``, never an IndexError.

    Fortran's ``&`` line-continuation markers are stripped before splitting, or an arg name would carry
    a stray ``&``/newline."""
    if lang == "fortran":
        m = re.search(rf"subroutine\s+{re.escape(symbol)}\s*\((.*?)\)", text, re.S | re.I)
        if not m:
            raise LookupError(f"entry {symbol} not found in the emitted {lang} source")
        return [a.strip() for a in m.group(1).replace("&", " ").split(",") if a.strip()]
    m = re.search(rf"void\s+{re.escape(symbol)}\s*\((.*?)\)\s*\{{", text, re.S)
    if not m:
        raise LookupError(f"entry {symbol} not found in the emitted {lang} source")
    return [p.strip().split()[-1].lstrip("*") for p in m.group(1).split(",") if p.strip() and p.strip() != "void"]


def c_argtypes(order: List[str], boundary) -> list:
    """ctypes type per C parameter: array name -> pointer-to-dtype, size/index symbol -> int64, value
    scalar -> its SDFG dtype, matching the translator's signature."""
    sdfg = boundary.standalone_sdfg
    return [
        ctypes.POINTER(CTYPE[np.dtype(sdfg.arrays[a].dtype.type).name]) if a in sdfg.arrays else scalar_ctype(sdfg, a)
        for a in order
    ]


def call_c(so: Path,
           symbol: str,
           order: List[str],
           argtypes: list,
           boundary,
           inputs,
           sizes,
           reps: int,
           copy_outputs: bool = True):
    """Bind by the C signature order, run once for correctness (snapshotting outputs), then time ``reps``
    bare-ctypes calls on the same buffers, mutating ``inputs`` in place. Callers run this in a forked
    child, so the COW-inherited buffers mutate without touching the parent.

    ``copy_outputs=False`` skips the snapshot for a pure-timing caller -- at the profiling size an output
    can be GBs, and snapshotting it would double the child's peak RSS."""
    fn = ctypes.CDLL(str(so))[symbol]
    fn.argtypes, fn.restype = argtypes, None

    def build_args():
        return [
            inputs[a].ctypes.data_as(t) if a in inputs else t(sizes[a])  # t: c_int64 size / c_double value scalar
            for a, t in zip(order, argtypes)
        ]

    fn(*build_args())  # correctness run
    outputs = {o: inputs[o].copy() for o in boundary.outputs} if copy_outputs else None
    cargs = build_args()
    fn(*cargs)  # warm
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(*cargs)
    return outputs, (time.perf_counter() - t0) / reps * 1e6


def native_symbol(text: str, expected: str) -> str:
    """The first ``extern "C"`` kernel symbol in the baseline source (``expected`` is the ``<key>_d``
    convention, used only as the search hint / fallback)."""
    if re.search(rf"\b{re.escape(expected)}\s*\(", text):
        return expected
    m = re.search(r"\bvoid\s+(\w+)\s*\(", text)
    if not m:
        raise LookupError("no kernel function found in the native baseline source")
    return m.group(1)


def native_setup(so: Path, symbol: str, sig, kernel, buffers: Dict[str, np.ndarray], sizes: Dict[str, int]):
    """Bind ``_original.cpp`` on ``buffers``; return ``(fn, cargs, ptr_names)``. Native bounds are
    independent of nest buffers, so an OOB here is real -- run via :func:`nestforge.isolation.run_isolated`.
    Raises on an unresolved arg."""
    pool = {"iterations": 1, "vlen": 8}
    pool.update({s.lower(): int(v) for s, v in sizes.items()})
    pool.update({k.lower(): int(v) for k, v in kernel.params.items()})
    argtypes, ptr_names = [], []
    for name, base, is_ptr in sig:
        ct = C_BASE[base]
        if is_ptr:
            if name not in buffers:
                raise KeyError(f"native pointer arg {name!r} has no matching array buffer")
            argtypes.append(ctypes.POINTER(ct))
            ptr_names.append(name)
        else:
            if name.lower() not in pool:
                raise KeyError(f"native scalar arg {name!r} unresolved")
            argtypes.append(ct)
    fn = ctypes.CDLL(str(so))[symbol]
    fn.argtypes, fn.restype = argtypes, None
    cargs = [
        buffers[name].ctypes.data_as(ctypes.POINTER(C_BASE[base])) if is_ptr else C_BASE[base](pool[name.lower()])
        for name, base, is_ptr in sig
    ]
    return fn, cargs, ptr_names


# --- numbers + result IO ----------------------------------------------------------------------------
def finite(x) -> bool:
    """True only for a real, usable numeric time: a finite int/float. Rejects ``None``, ``inf``, ``nan``."""
    return isinstance(x, (int, float)) and math.isfinite(x)


def median(xs: List[float]) -> float:
    """Median of a non-empty sample (interpolates the two central values on an even-length list)."""
    return float(statistics.median(xs))


def geomean(xs: List[float]) -> Optional[float]:
    """Geometric mean of the finite positive values; ``None`` when there are none."""
    vals = [x for x in xs if finite(x) and x > 0.0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def fmt_us(x) -> str:
    """A microsecond time for a markdown cell; an em dash for missing / infinite."""
    return "—" if x is None or x == float("inf") else f"{x:.2f}"


def jsonable(obj):
    """Recursively map non-finite floats to ``None`` so ``json.dumps`` emits standard JSON. Dict/list
    structure otherwise preserved."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonable(v) for v in obj]
    return obj


def load_results(results_dir: Path) -> List[dict]:
    """Every per-kernel JSON in ``results_dir`` (``tables.md`` and unparseable files skipped)."""
    rows: List[dict] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "tables.md":
            continue
        try:
            rows.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError, ValueError):
            continue
    return rows
