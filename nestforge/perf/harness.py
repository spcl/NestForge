"""Shared arena infrastructure: rank partitioning, bounded compiles, ABI binding, and result IO.

Every perf driver (``tsvc_arena``, ``tsvc_full``, ``crosslang_xl``, ``calloverhead``,
``staticlib_overhead``) and every ``perf/plot_*.py`` script needs the same handful of primitives. They
used to live inside ``tsvc_arena`` -- a *driver* -- so the other four drivers imported it for infra and
the plot scripts carried their own byte-identical copies (10x ``geomean``, 7x ``finite``, 6x
``load_results``, 3x the ABI-order reader). This module is that infra, relocated: the functions are
unchanged, they simply have one home now.
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

#: Per-compile wall-clock ceiling (seconds). The phase-1 sweep runs compiles inside a
#: ThreadPoolExecutor; an untimed compiler spinning on a pathological kernel would hang the worker
#: thread, block ``pool.map``, freeze the whole sweep rank and leave srun waiting on it -- the 19h
#: "job timed out" stall. On timeout we return a normal (failed, err) result so only that one cell is
#: lost. Override with ``NF_COMPILE_TIMEOUT``.
COMPILE_TIMEOUT_S: float = float(os.environ.get("NF_COMPILE_TIMEOUT", "900"))

#: Per-kernel *execution* wall-clock ceiling (seconds), distinct from the compile ceiling above. A
#: validated timing cell runs the compiled kernel forked (``run_isolated``); a runaway kernel (a bad
#: unroll, a pathological size) would otherwise hold the fork open for the whole job. Override with
#: ``NF_RUN_TIMEOUT``. Default 1800 -- a full median-of-N run at the XL profiling size stays well under it.
RUN_TIMEOUT_S: float = float(os.environ.get("NF_RUN_TIMEOUT", "1800"))

#: base C-type name -> ctypes type, for binding a native baseline signature.
C_BASE = {"double": ctypes.c_double, "float": ctypes.c_float, "int64_t": ctypes.c_int64, "int": ctypes.c_int}

#: rank / size environment variables, most specific launcher first: SLURM (srun), then OpenMPI, then
#: MPICH/Hydra (mpirun). Whichever the job was launched with wins, so a driver self-partitions the same
#: under ``srun`` on daint and under a local ``mpirun`` in a test.
RANK_VARS = ("SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK")
SIZE_VARS = ("SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE")


# --- rank self-partition ----------------------------------------------------------------------------
def my_slice(items: List, procid: int, ntasks: int) -> List:
    """This rank's disjoint stride of the kernel list (round-robin balances long/short kernels)."""
    return items[procid::ntasks] if ntasks > 1 else items


def rank_and_size() -> Tuple[int, int]:
    """``(rank, nranks)`` from the launcher's environment (SLURM or an MPI runtime), defaulting to
    ``(0, 1)`` for a plain single-process run. Every rank sees the same total, so the round-robin
    partition is disjoint and covers every kernel exactly once.

    Fails loud on an asymmetric environment -- a launcher that sets a rank variable but no recognized
    size variable -- because silently defaulting the size to 1 would make EVERY rank process the whole
    list (N-fold duplicated work + concurrent writes to the same per-kernel result file)."""
    rank_var = next((v for v in RANK_VARS if os.environ.get(v)), None)
    size_var = next((v for v in SIZE_VARS if os.environ.get(v)), None)
    if rank_var is not None and size_var is None:
        raise RuntimeError(f"launcher set a rank ({rank_var}={os.environ[rank_var]}) but no recognized size variable "
                           f"({list(SIZE_VARS)}); cannot partition safely -- every rank would run the whole list. Set "
                           "the matching size env var, or run without a launcher for a single process.")
    if size_var is not None and rank_var is None:
        # The mirror asymmetry, and the quieter one: defaulting the rank to 0 keeps a size of N, so this
        # single process takes only slice 0 -- 1/N of the corpus -- and the run publishes a geomean over a
        # fraction of the kernels with nothing to notice. `sbatch --ntasks=4` running the module directly
        # (no srun) sets SLURM_NTASKS without SLURM_PROCID and lands exactly here.
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

    numpyto orders the C parameters canonically (sorted arrays, then symbols), which is NOT the manifest
    ``input_args`` order -- so args must be bound to the *actual* signature, or a size lands in a pointer
    slot (garbage index -> a runaway loop). An empty / ``void`` parameter list yields ``[]`` (a zero-arg
    kernel), never an IndexError.

    A long Fortran argument list wraps across lines with ``&`` free-form continuations
    (``arg1, &\\n  & arg2``); those markers are stripped before splitting, or an arg name would carry a
    stray ``&``/newline (``&\\n&aa_slice`` -> a bogus name -> KeyError at the call)."""
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
    """ctypes type per C parameter: an array name -> pointer-to-its-dtype, a size / index symbol -> int64,
    a value scalar (a staged ``a_index = a[i]`` read) -> its SDFG dtype (``c_double``), matching the
    translator's ``double`` signature so the value is neither truncated nor mis-passed."""
    sdfg = boundary.standalone_sdfg
    return [
        ctypes.POINTER(CTYPE[np.dtype(sdfg.arrays[a].dtype.type).name]) if a in sdfg.arrays else scalar_ctype(sdfg, a)
        for a in order
    ]


def call_c(so: Path, symbol: str, order: List[str], argtypes: list, boundary, inputs, sizes, reps: int):
    """Bind by the C signature order, run once for correctness (snapshotting outputs), then time ``reps``
    bare-ctypes calls on the same buffers. Runs the kernel IN PLACE on ``inputs``; every caller invokes it
    inside a forked child, so the COW-inherited buffers are mutated without touching the parent -- this
    avoids a full per-cell copy of the (time-size) input set. The one correctness run precedes any mutation
    from timing, so validation still sees fresh inputs."""
    fn = ctypes.CDLL(str(so))[symbol]
    fn.argtypes, fn.restype = argtypes, None

    def build_args():
        return [
            inputs[a].ctypes.data_as(t) if a in inputs else t(sizes[a])  # t: c_int64 size / c_double value scalar
            for a, t in zip(order, argtypes)
        ]

    fn(*build_args())  # correctness run
    outputs = {o: inputs[o].copy() for o in boundary.outputs}
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
    """Bind ``_original.cpp`` on ``buffers``; return ``(fn, cargs, ptr_names)``. The native bounds are
    independent of the nest buffers, so an OOB here is real -- run through
    :func:`nestforge.isolation.run_isolated`. Raises on an unresolved arg."""
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
    """True only for a real, usable numeric time: a finite int/float. Rejects ``None`` (a non-finite value
    mapped away by :func:`jsonable` on write) as well as ``inf`` / ``nan``."""
    return isinstance(x, (int, float)) and math.isfinite(x)


def median(xs: List[float]) -> float:
    """Median of a non-empty sample (``statistics.median``, so it interpolates the two central values on
    an even-length list)."""
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
    """Recursively map non-finite floats (``inf``/``nan``) to ``None`` so ``json.dumps`` emits standard JSON
    (which has no ``Infinity``/``NaN`` literals). Dict/list structure is otherwise preserved."""
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
