"""TSVC compiler-arena driver: run every TSVC kernel through the ``skip-taskloops`` strategy and, for
each kernel x each discovered compiler, report three runtime columns:

  1. **native baseline** -- the original ``_original.cpp`` loop at default flags (the compiler's own
     auto-vectorization of the reference),
  2. **default-flags**   -- the extracted nest translated to C, same compiler/flags (isolates translation
     overhead vs the baseline),
  3. **flag-matrix winner** -- the same nest swept over the flag matrix (FP-mode x vectorizer cost-model);
     the fastest cell that still validates against the numpy oracle.

Sizes are sampled with a fixed seed so every compiler sees identical data. Ranks self-partition the
kernel list via ``SLURM_PROCID`` / ``SLURM_NTASKS``; results land per kernel as JSON. ``--tables-only``
merges them into markdown; ``--link`` archives each winning cell and links them into one whole-TSVC
library for the aggregate whole-program comparison.

Usage::

    python -m nestforge.perf.tsvc_arena --select tsvc --strategy skip-taskloops \\
        --compilers auto --reps 100 --seed 0 --random-sizes
    python -m nestforge.perf.tsvc_arena --link --seed 0
    python -m nestforge.perf.tsvc_arena --tables-only --seed 0
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import dace  # noqa: F401 -- ensure the DaCe package (not a cwd-shadowing stub) is importable

from nestforge import tsvc
from nestforge.arena import CTYPE, maxdiff, make_inputs, run_oracle
from nestforge.build import compiler_family, compiler_version, resolve_runtime
from nestforge.perf import flags
from nestforge.isolation import run_isolated
from nestforge.multinest import extract_all_nests
from nestforge.translate import emit_sources, prepare


def default_flags(family: str) -> List[str]:
    """The default-flags column: the compiler's plain optimizing flags (``flags.base_flags``, family-aware
    so nvidia gets ``-tp=native``) with no FP-mode or cost-model tweak. The native ``.cpp`` baseline is
    compiled at the SAME flags, so the "same default flags" fairness the report claims cannot drift."""
    return flags.base_flags(family)


#: base C-type name -> ctypes type, for the native-baseline signature.
_C_BASE = {"double": ctypes.c_double, "float": ctypes.c_float, "int64_t": ctypes.c_int64, "int": ctypes.c_int}


# --- compiler discovery (item b: PATH + spack, gcc/clang/nvc++) --------------------------------------
@dataclass
class Toolchain:
    """One discovered toolchain family: its C compiler (for the translated nest) and C++ compiler (for
    the native ``.cpp`` baseline), plus a version and where it was found."""
    name: str  # family label: "gcc" | "clang" | "nvhpc" | "intel"
    cc: str  # C compiler path (gcc / clang / nvc / icx)
    cxx: Optional[str]  # C++ compiler path (g++ / clang++ / nvc++ / icpx); None -> no native column
    version: Tuple[int, int]
    source: str  # "path" | "spack"

    @property
    def family(self) -> str:
        """OpenMP-runtime family of the C compiler (icx -> llvm). Use :attr:`fp_family` for flag matrices."""
        return compiler_family(self.cc)

    @property
    def fp_family(self) -> str:
        """The flag-matrix FP family. Intel (icx/icpx) is its own FP family even though it is clang-based
        (``compiler_family`` calls it ``llvm``), because it defaults to ``-fp-model=fast`` and needs
        explicit ``-fp-model`` flags; gcc/clang/nvhpc coincide with :attr:`family`."""
        return "intel" if self.name == "intel" else self.family


#: family label -> (C compiler exe, C++ compiler exe).
_FAMILY_EXES = {
    "gcc": ("gcc", "g++"),
    "clang": ("clang", "clang++"),
    "nvhpc": ("nvc", "nvc++"),
    "intel": ("icx", "icpx")
}
#: user tokens (compiler names/aliases) -> family label.
_ALIASES = {
    "gcc": "gcc", "g++": "gcc", "gnu": "gcc",
    "clang": "clang", "clang++": "clang", "llvm": "clang",
    "nvc": "nvhpc", "nvc++": "nvhpc", "nvhpc": "nvhpc", "nvidia": "nvhpc",
    "icx": "intel", "icpx": "intel", "intel": "intel", "oneapi": "intel",
}  # yapf: disable


def spack_bin_dirs() -> List[Path]:
    """Best-effort: ``bin`` directories of spack-installed gcc/llvm/nvhpc, so a compiler that is
    installed but not ``spack load``ed onto PATH is still discoverable. Never fatal (spack absent, a
    slow/interactive shell, a prompt) -- returns whatever it can within a short timeout."""
    if not shutil.which("spack"):
        return []
    dirs: List[Path] = []
    try:
        out = subprocess.run(["spack", "find", "--paths", "--no-groups"], capture_output=True, text=True,
                             timeout=25).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].split("@")[0] in ("gcc", "llvm", "nvhpc"):
            bindir = Path(parts[-1]) / "bin"
            if bindir.is_dir():
                dirs.append(bindir)
    return dirs


def spack_compiler_bin_dirs() -> List[Path]:
    """``bin`` dirs of every compiler spack has REGISTERED (``spack compiler list`` + ``spack compiler
    info``) -- distinct from :func:`spack_bin_dirs`, which enumerates spack-INSTALLED packages. On a
    spack-default host (e.g. daint) a usable compiler is often registered but not ``spack load``ed onto
    PATH; its ``cc``/``cxx`` path from ``spack compiler info`` recovers its bin dir. Best-effort, never
    fatal, bounded (spack start-up is slow): capped specs + short per-call timeout."""
    if not shutil.which("spack"):
        return []
    try:
        listing = subprocess.run(["spack", "compiler", "list"], capture_output=True, text=True, timeout=25).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    specs = []
    for line in listing.splitlines():
        line = line.strip()
        if not line or line.startswith("==>") or line.startswith("--"):
            continue  # skip the banner and the per-family header rules
        specs += [tok for tok in line.split() if "@" in tok]
    dirs: List[Path] = []
    for spec in specs[:12]:
        try:
            info = subprocess.run(["spack", "compiler", "info", spec], capture_output=True, text=True,
                                  timeout=15).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        for line in info.splitlines():
            if "=" not in line:
                continue  # lines like "\t\tcc = /opt/.../bin/gcc"
            path = line.split("=", 1)[1].strip()
            if path and os.path.isabs(path):
                d = Path(path).parent
                if d.is_dir() and d not in dirs:
                    dirs.append(d)
    return dirs


def which_on_path(exe: str, extra_dirs: List[Path]) -> Optional[str]:
    """``exe`` on PATH, else under one of ``extra_dirs`` (the spack install bins)."""
    found = shutil.which(exe)
    if found:
        return found
    for d in extra_dirs:
        cand = d / exe
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def discover_toolchains(requested: str = "auto") -> List[Toolchain]:
    """Discover the requested toolchain families. ``"auto"`` (or ``"all"``) enumerates gcc / clang /
    nvhpc; otherwise ``requested`` is a whitespace list of compiler names/aliases. A family is kept when
    its C compiler is found (on PATH or in a spack install bin); its C++ compiler is optional (its
    absence just drops that family's native column)."""
    tokens = list(_FAMILY_EXES) if requested.strip() in ("", "auto", "all") else requested.split()
    families: List[str] = []
    for t in tokens:
        fam = _ALIASES.get(t.strip())
        if fam is None:
            warnings.warn(f"unknown compiler token {t!r}; known: {sorted(_ALIASES)}")
        elif fam not in families:
            families.append(fam)
    # PATH first (via which_on_path), then spack: installed packages AND registered compilers. On a
    # spack-default host the compiler may be in neither PATH nor a `spack find` prefix, only registered.
    spack_dirs = spack_bin_dirs()
    for d in spack_compiler_bin_dirs():
        if d not in spack_dirs:
            spack_dirs.append(d)
    out: List[Toolchain] = []
    for fam in families:
        cc_exe, cxx_exe = _FAMILY_EXES[fam]
        cc = which_on_path(cc_exe, spack_dirs)
        if cc is None:
            warnings.warn(f"{fam}: C compiler {cc_exe!r} not found (PATH or spack); skipping this family")
            continue
        cxx = which_on_path(cxx_exe, spack_dirs)
        if cxx is None:
            warnings.warn(f"{fam}: C++ compiler {cxx_exe!r} not found; native-baseline column disabled for {fam}")
        source = "path" if shutil.which(cc_exe) else "spack"
        out.append(Toolchain(name=fam, cc=cc, cxx=cxx, version=compiler_version(cc), source=source))
    return out


def restrict_to_single_openmp_runtime(toolchains: List[Toolchain], runtime_name: str = "libomp") -> List[Toolchain]:
    """Keep only toolchains that can link the ONE chosen OpenMP runtime.

    PARALLEL.md mandates a single runtime for the whole program: mixing e.g. nvc's native ``libnvomp``
    with gcc/clang's ``libomp`` double-loads the OpenMP runtime (the ``OMP Error #15`` abort /
    thread-pool oversubscription). Uses ``build.py``'s per-family compatibility model; drops -- with a
    warning -- any compiler that would be forced onto a different runtime. Apply this ONLY to a job that
    actually links OpenMP (a parallel-emitting or whole-program-linking job); the serial nest arena
    links no OpenMP, so it keeps every compiler."""
    rt = resolve_runtime(runtime_name)
    kept: List[Toolchain] = []
    for tc in toolchains:
        if rt.compatible(tc.cc) and (tc.cxx is None or rt.compatible(tc.cxx)):
            kept.append(tc)
        else:
            warnings.warn(f"{tc.name}: family {tc.family} cannot link the single OpenMP runtime "
                          f"{runtime_name!r} (it forces its own); dropping from the OpenMP sweep")
    return kept


# The flag matrix (FP-precision level x vectorizer cost-model, per family) is the shared
# ``nestforge.perf.flags.flag_matrix``; see ``docs/FP_PRECISION_LEVELS.md``. The crosslang job sweeps
# the same matrix, so the two arenas stay in lock-step on one FP-precision ladder.


# --- a single measured compile cell -----------------------------------------------------------------
@dataclass
class Cell:
    """One (compiler, flags) measurement: correctness + timing, or an error."""
    compiler: str  # family label
    label: str  # "native" | "default" | "<fp_mode>/<cost_model>"
    flags: List[str]
    ok: bool
    maxdiff: float
    time_us: float
    compile_us: float
    error: Optional[str] = None


@dataclass
class NestUnit:
    """One extracted nest of a kernel, with everything a cell needs to compile + validate + time it.

    A single-nest kernel has one :class:`NestUnit` whose ``name``/``symbol`` are the plain ``<key>`` /
    ``<key>_fp64`` (unchanged from the old path); a multi-nest kernel has one per nest with distinct
    ``<key>_n<idx>`` names, so each binds its own entry point."""
    idx: int
    name: str
    symbol: str
    boundary: object
    sizes: Dict[str, int]
    inputs: Dict[str, np.ndarray]
    oracle: Dict[str, np.ndarray]
    csrc: Path
    order: List[str]
    argtypes: list


def run_compile(cmd: List[str]) -> Tuple[bool, float, Optional[str]]:
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = (time.perf_counter() - t0) * 1e6
    return (p.returncode == 0), dt, (None if p.returncode == 0 else p.stderr[-400:])


def abi_order(csrc_text: str, symbol: str) -> List[str]:
    """The parameter names of ``void <symbol>(...)`` in declaration order, read from the emitted C.

    numpyto orders the C parameters canonically (sorted arrays, then symbols), which is NOT the manifest
    ``input_args`` order -- so args must be bound to the *actual* C signature, or a size lands in a
    pointer slot (garbage index -> a runaway loop). Mirrors ``scripts/overhead_baseline.abi_order``.
    An empty / ``void`` parameter list yields ``[]`` (a zero-arg kernel), never an IndexError.
    """
    m = re.search(rf"void\s+{re.escape(symbol)}\s*\((.*?)\)\s*\{{", csrc_text, re.S)
    if not m:
        raise LookupError(f"entry point {symbol} not found in the emitted C")
    params = [p.strip() for p in m.group(1).split(",") if p.strip() and p.strip() != "void"]
    return [p.split()[-1].lstrip("*") for p in params]


def c_argtypes(order: List[str], boundary) -> list:
    """ctypes type per C parameter: an array name -> pointer-to-its-dtype, a size symbol -> int64."""
    sdfg = boundary.standalone_sdfg
    return [
        ctypes.POINTER(CTYPE[np.dtype(sdfg.arrays[a].dtype.type).name]) if a in sdfg.arrays else ctypes.c_int64
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
            inputs[a].ctypes.data_as(t) if a in inputs else ctypes.c_int64(int(sizes[a]))
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


def measure_nest(cc: str, csrc: Path, flags: List[str], symbol: str, order: List[str], argtypes, boundary, inputs,
                 sizes, oracle, reps: int, atol: float, family: str, label: str, workdir: Path) -> Cell:
    """Compile the translated-nest C at ``flags``, then validate + time it in a forked child (so an OOB
    or runaway loop in the compiled kernel cannot take down the sweep rank)."""
    so = workdir / f"{symbol}_{family}_{label.replace('/', '_')}.so"
    ok, compile_us, err = run_compile([cc, *flags, str(csrc), "-o", str(so)])
    if not ok:
        return Cell(family, label, flags, False, float("inf"), float("inf"), compile_us, error=err)

    def work():
        outs, us = call_c(so, symbol, order, argtypes, boundary, inputs, sizes, reps)
        md = maxdiff(oracle, outs)
        return {"ok": bool(md <= atol), "maxdiff": float(md), "time_us": float(us)}

    res = run_isolated(work)
    if "error" in res:
        return Cell(family, label, flags, False, float("inf"), float("inf"), compile_us, error=res["error"])
    return Cell(family, label, flags, res["ok"], res["maxdiff"], res["time_us"], compile_us)


def measure_over_nests(cc: str, units: List[NestUnit], cflags: List[str], reps: int, atol: float, family: str,
                       label: str, workdir: Path) -> Cell:
    """One (compiler, flags) cell SUMMED over every nest of the kernel: compile + time each nest's own
    source at ``cflags`` (each nest a distinct symbol), then aggregate into a single :class:`Cell` whose
    ``time_us`` / ``compile_us`` are the sums, ``ok`` iff every nest validated, and ``maxdiff`` the max
    over nests. A single-nest kernel returns exactly the old single measurement (a sum of one), so the
    148 single-nest kernels' cells are byte-identical to before."""
    per = [
        measure_nest(cc, u.csrc, cflags, u.symbol, u.order, u.argtypes, u.boundary, u.inputs, u.sizes, u.oracle, reps,
                     atol, family, label, workdir) for u in units
    ]
    return Cell(family, label, cflags, all(c.ok for c in per), max(c.maxdiff for c in per),
                sum(c.time_us for c in per), sum(c.compile_us for c in per),
                error=next((c.error for c in per if c.error), None))


# --- native baseline (item e) -----------------------------------------------------------------------
def native_symbol(text: str, expected: str) -> str:
    """The first ``extern "C"`` kernel symbol in the baseline source (``expected`` is the ``<key>_d``
    convention, used only as the search hint / fallback)."""
    if re.search(rf"\b{re.escape(expected)}\s*\(", text):
        return expected
    m = re.search(r"\bvoid\s+(\w+)\s*\(", text)
    if not m:
        raise LookupError("no kernel function found in the native baseline source")
    return m.group(1)


def native_work(so: Path, symbol: str, sig, kernel, boundary, inputs, sizes, oracle, reps: int) -> Dict:
    """Bind + validate + time the native baseline; runs inside the forked child
    (:func:`nestforge.isolation.run_isolated`), so an out-of-bounds access in the original C (its bounds
    are independent of the nest-sized buffers) segfaults only the child. Raises on an unresolved arg."""
    pool = {"iterations": 1, "vlen": 8}
    pool.update({s.lower(): int(v) for s, v in sizes.items()})
    pool.update({k.lower(): int(v) for k, v in kernel.params.items()})
    argtypes, ptr_names = [], []
    for name, base, is_ptr in sig:
        ct = _C_BASE[base]
        if is_ptr:
            if name not in inputs:
                raise KeyError(f"native pointer arg {name!r} has no matching array buffer")
            argtypes.append(ctypes.POINTER(ct))
            ptr_names.append(name)
        else:
            if name.lower() not in pool:
                raise KeyError(f"native scalar arg {name!r} unresolved")
            argtypes.append(ct)

    fn = ctypes.CDLL(str(so))[symbol]
    fn.argtypes, fn.restype = argtypes, None

    def build_args():
        return [
            inputs[name].ctypes.data_as(ctypes.POINTER(_C_BASE[base])) if is_ptr else _C_BASE[base](pool[name.lower()])
            for name, base, is_ptr in sig
        ]

    fn(*build_args())  # correctness run
    outs = {o: inputs[o].copy() for o in boundary.outputs if o in ptr_names}
    md = maxdiff({k: oracle[k] for k in outs}, outs) if outs else 0.0
    cargs = build_args()
    fn(*cargs)  # warm
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(*cargs)
    us = (time.perf_counter() - t0) / reps * 1e6
    return {"ok": bool(md <= 1e-6), "maxdiff": float(md), "time_us": float(us)}


def measure_native(cxx: str, kernel: "tsvc.TsvcKernel", boundary, inputs, sizes, oracle, reps: int, family: str,
                   workdir: Path) -> Optional[Cell]:
    """Compile the ``_original.cpp`` baseline and time it (in a forked child) on the SAME inputs/sizes as
    the nest columns. Returns ``None`` when this kernel ships no native source or the family has no C++
    compiler."""
    cpp = kernel.native_cpp
    if cpp is None or cxx is None:
        return None
    nat = default_flags(family)  # native baseline uses the SAME default flags as the default column
    text = cpp.read_text()
    try:
        symbol = native_symbol(text, kernel.native_symbol)
        sig = tsvc.native_signature(text, symbol)
    except LookupError as e:
        return Cell(family, "native", nat, False, float("inf"), float("inf"), 0.0, error=str(e))

    so = workdir / f"{kernel.key}_{family}_native.so"
    ok, compile_us, err = run_compile([cxx, *nat, str(cpp), "-o", str(so)])
    if not ok:
        return Cell(family, "native", nat, False, float("inf"), float("inf"), compile_us, error=err)

    res = run_isolated(lambda: native_work(so, symbol, sig, kernel, boundary, inputs, sizes, oracle, reps))
    if "error" in res:
        return Cell(family, "native", nat, False, float("inf"), float("inf"), compile_us, error=res["error"])
    return Cell(family, "native", nat, res["ok"], res["maxdiff"], res["time_us"], compile_us)


# --- per-kernel run ---------------------------------------------------------------------------------
def select_c_source(sources: List[Path]) -> Path:
    return next(p for p in sources if p.suffix == ".c" and "pluto" not in p.name)


def run_kernel(kernel: "tsvc.TsvcKernel", toolchains: List[Toolchain], strategy: str, opt_mode: str, seed: int,
               reps: int, random_sizes: bool, workdir: Path) -> Dict:
    """Run one kernel through all three columns for every toolchain; return the JSON-able result dict.

    A kernel may split into several compute nests (its work is the SUM of its nests): every ``default`` /
    flag-matrix cell compiles + times its (source, flags) for EACH nest and sums the per-nest times, so a
    cell just aggregates its nests and the result/row schema is unchanged. The whole-kernel native
    ``.cpp`` baseline stays a single measurement (it already covers all the kernel's work); it borrows the
    first nest's buffers for sizing, mirroring ``tsvc_full.build_opt_context``."""
    result: Dict = {"key": kernel.key, "regime": kernel.regime, "seed": seed, "host": socket.gethostname()}
    try:
        nests = extract_all_nests(lambda: tsvc.build_sdfg(kernel, opt_mode=opt_mode), strategy, kernel.key)
        if not nests:
            return {**result, "skipped": "no compute nest (strategy returned nothing)"}
        units: List[NestUnit] = []
        for idx, name, symbol, boundary in nests:
            sizes = tsvc.sample_sizes(kernel, boundary, seed=seed, random_sizes=random_sizes)
            nest_dir = workdir / f"n{idx}"
            prep = prepare(boundary, name, nest_dir, sizes=sizes)
            inputs = make_inputs(boundary, sizes, seed=seed)
            oracle = run_oracle(prep, boundary, inputs, sizes)
            csrc = select_c_source(emit_sources(prep, nest_dir))
            order = abi_order(csrc.read_text(), symbol)
            units.append(
                NestUnit(idx, name, symbol, boundary, sizes, inputs, oracle, csrc, order, c_argtypes(order, boundary)))
    except Exception as e:
        return {**result, "skipped": f"{type(e).__name__}: {str(e)[:160]}"}

    # union of per-nest sizes (a shared shape symbol resolves to the same value in every nest; leaked
    # indices are 0). For a single-nest kernel this is exactly that nest's sizes (schema unchanged).
    merged: Dict[str, int] = {}
    for u in units:
        merged.update(u.sizes)
    result["sizes"] = {k: int(v) for k, v in merged.items()}
    # per-kernel nest roster (name/symbol), so the whole-program link can verify each nest's symbol.
    result["nests"] = [{"idx": u.idx, "name": u.name, "symbol": u.symbol} for u in units]
    rows: List[Dict] = []
    for tc in toolchains:
        fam = tc.fp_family  # flag-matrix / FP family (intel != llvm); also the cell's compiler label
        default = measure_over_nests(tc.cc, units, default_flags(fam), reps, 1e-6, fam, "default", workdir)
        cells = [
            measure_over_nests(tc.cc, units, cflags, reps, flags.FP_ATOL[level], fam, f"{level}/{model}", workdir)
            for level, model, cflags in flags.flag_matrix(fam)
        ]
        # native is the whole-kernel .cpp -> one measurement on the first nest's buffers (see docstring).
        native = measure_native(tc.cxx, kernel, units[0].boundary, units[0].inputs, units[0].sizes, units[0].oracle,
                                reps, fam, workdir)
        correct = [c for c in cells if c.ok]
        winner = min(correct, key=lambda c: c.time_us) if correct else None
        rows.append({
            "compiler": tc.name,
            "version": list(tc.version),
            "source": tc.source,
            "native": asdict(native) if native else None,
            "default": asdict(default),
            "winner": asdict(winner) if winner else None,
            "cells": [asdict(c) for c in cells],
        })
    result["rows"] = rows
    return result


# --- storage ----------------------------------------------------------------------------------------
def ensure_seed_dir(out: Path, seed: int) -> Path:
    d = out / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def my_slice(items: List, procid: int, ntasks: int) -> List:
    """This rank's disjoint stride of the kernel list (round-robin balances long/short kernels)."""
    return items[procid::ntasks] if ntasks > 1 else items


#: rank / size environment variables, most specific launcher first: SLURM (srun), then OpenMPI, then
#: MPICH/Hydra (mpirun). Whichever the job was launched with wins, so the driver self-partitions the
#: same under ``srun`` on daint and under a local ``mpirun`` in a test.
_RANK_VARS = ("SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK")
_SIZE_VARS = ("SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE")


def rank_and_size() -> Tuple[int, int]:
    """``(rank, nranks)`` from the launcher's environment (SLURM or an MPI runtime), defaulting to
    ``(0, 1)`` for a plain single-process run. Every rank sees the same total, so the round-robin
    partition is disjoint and covers every kernel exactly once.

    Fails loud on an asymmetric environment -- a launcher that sets a rank variable but no recognized
    size variable -- because silently defaulting the size to 1 would make EVERY rank process the whole
    list (N-fold duplicated work + concurrent writes to the same per-kernel result file)."""
    rank_var = next((v for v in _RANK_VARS if os.environ.get(v)), None)
    size_var = next((v for v in _SIZE_VARS if os.environ.get(v)), None)
    if rank_var is not None and size_var is None:
        raise RuntimeError(f"launcher set a rank ({rank_var}={os.environ[rank_var]}) but no recognized size variable "
                           f"({list(_SIZE_VARS)}); cannot partition safely -- every rank would run the whole list. Set "
                           "the matching size env var, or run without a launcher for a single process.")
    rank = int(os.environ[rank_var]) if rank_var else 0
    size = int(os.environ[size_var]) if size_var else 1
    return rank, max(size, 1)


# --- tables (item g) --------------------------------------------------------------------------------
def fmt_us(x) -> str:
    return "—" if x is None or x == float("inf") else f"{x:.2f}"


def render_tables(out: Path, seed: int) -> str:
    """Merge every per-kernel JSON under ``seed<seed>/`` into a markdown report."""
    seed_dir = ensure_seed_dir(out, seed)
    files = sorted(p for p in seed_dir.glob("*.json") if p.name not in ("tables.md", ))
    kernels = [json.loads(p.read_text()) for p in files]
    done = [k for k in kernels if "rows" in k]
    skipped = [k for k in kernels if "skipped" in k]

    lines = [f"# TSVC compiler-arena (seed {seed})", "", f"{len(done)} kernels measured, {len(skipped)} skipped.", ""]
    lines.append("| kernel | regime | compiler | sizes | native (us) | default (us) | best (us) | best flags "
                 "| maxdiff | speedup best/native |")
    lines.append("|" + "---|" * 10)
    speedups: List[float] = []
    for k in sorted(done, key=lambda x: x["key"]):
        sizes = ",".join(f"{s}={v}" for s, v in k.get("sizes", {}).items())
        for r in k["rows"]:
            nat = r["native"]
            dfl = r["default"]
            win = r["winner"]
            nat_us = nat["time_us"] if nat and nat["ok"] else None
            best_us = win["time_us"] if win else None
            best_lbl = win["label"] if win else "—"
            md = f"{win['maxdiff']:g}" if win else "—"
            # `nat_us is not None` (not truthiness) so a legitimately-measured 0.00us is not dropped;
            # `best_us` stays a truthiness guard to avoid a divide-by-zero.
            sp = (nat_us / best_us) if (nat_us is not None and best_us) else None
            if sp is not None and math.isfinite(sp):
                speedups.append(sp)
            lines.append(f"| {k['key']} | {k['regime']} | {r['compiler']} | {sizes} | {fmt_us(nat_us)} "
                         f"| {fmt_us(dfl['time_us'] if dfl['ok'] else None)} | {fmt_us(best_us)} | {best_lbl} "
                         f"| {md} | {'—' if sp is None else f'{sp:.2f}x'} |")

    if speedups:
        geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        lines += [
            "", f"**Geomean flag-matrix speedup vs native baseline:** {geo:.3f}x "
            f"(over {len(speedups)} kernel x compiler rows where both timed)."
        ]
    if skipped:
        lines += ["", "## skipped kernels", ""]
        for k in sorted(skipped, key=lambda x: x["key"]):
            lines.append(f"- `{k['key']}` — {k['skipped']}")
    report = "\n".join(lines) + "\n"
    (seed_dir / "tables.md").write_text(report)
    return report


# --- whole-program link (item f) --------------------------------------------------------------------
def global_winner(k: Dict) -> Optional[Dict]:
    """The fastest correct nest cell for a kernel across all toolchains (its winning cell to assemble)."""
    best = None
    for r in k["rows"]:
        w = r["winner"]
        if w and (best is None or w["time_us"] < best["time_us"]):
            best = {**w, "compiler": r["compiler"]}
    return best


def link_whole_program(out: Path, seed: int, toolchains: List[Toolchain], opt_mode: str, strategy: str) -> str:
    """Assemble each kernel's winning cell as a static ``.a``, link them all into one whole-TSVC ``.so``,
    verify each kernel symbol loads and computes correctly, and report the aggregate whole-program
    comparison (Sum winner vs Sum native vs Sum default across the independent kernels). Sizes are read
    back from the stored sweep JSON (not re-sampled), so the reconstructed nest matches what was timed."""
    seed_dir = ensure_seed_dir(out, seed)
    kernels = [json.loads(p.read_text()) for p in sorted(seed_dir.glob("*.json")) if p.name != "tables.md"]
    done = [k for k in kernels if "rows" in k]
    by_name = {tc.name: tc for tc in toolchains}
    link_dir = seed_dir / "link"
    link_dir.mkdir(exist_ok=True)

    archives: List[Tuple[Path, List[str]]] = []  # (archive, [nest symbol, ...]) -- multi-nest kernels list all
    verified, failed = 0, 0
    sum_win = sum_nat = sum_dfl = 0.0
    notes: List[str] = []
    for k in sorted(done, key=lambda x: x["key"]):
        win = global_winner(k)
        if win is None:
            notes.append(f"`{k['key']}` — no correct winner cell; excluded from the linked program")
            continue
        tc = by_name.get(win["compiler"]) or next(iter(toolchains), None)
        # A kernel may split into several nests; the winning flags apply to all of them (the winner cell
        # is one flag set summed over nests). Compile EACH nest's winning-flags object and archive them all
        # into one lib<key>.a; the whole-program verify then checks every nest symbol.
        try:
            kernel = tsvc.iter_tsvc_kernels(only=[k["key"]])[0]
            nests = extract_all_nests(lambda: tsvc.build_sdfg(kernel, opt_mode=opt_mode), strategy, kernel.key)
        except Exception as e:
            notes.append(f"`{k['key']}` — extract failed: {type(e).__name__}: {str(e)[:120]}")
            continue
        cflags = [f for f in win["flags"] if f != "-shared"]
        objs: List[Path] = []
        symbols: List[str] = []
        compile_note: Optional[str] = None
        for idx, name, symbol, boundary in nests:
            try:
                sizes = {s: k["sizes"][s] for s in boundary.symbols}
                prep = prepare(boundary, name, link_dir, sizes=sizes)
                csrc = select_c_source(emit_sources(prep, link_dir))
                obj = link_dir / f"{name}.o"
                if obj.exists():
                    obj.unlink()  # never let a stale object from an earlier --link run be archived silently
                cok, _, cerr = run_compile([tc.cc, *cflags, "-c", str(csrc), "-o", str(obj)])
                if not cok:
                    compile_note = f"nest {name}: {cerr}"
                    break
                objs.append(obj)
                symbols.append(symbol)
            except Exception as e:
                compile_note = f"nest {name}: {type(e).__name__}: {str(e)[:120]}"
                break
        if compile_note is not None:
            notes.append(f"`{k['key']}` — winner compile failed: {compile_note}")
            continue
        try:
            archive = link_dir / f"lib{kernel.key}.a"
            if archive.exists():
                archive.unlink()
            subprocess.run([shutil.which("ar") or "ar", "rcs", str(archive), *[str(o) for o in objs]], check=True,
                           capture_output=True)
            archives.append((archive, symbols))
        except Exception as e:
            notes.append(f"`{k['key']}` — assemble failed: {type(e).__name__}: {str(e)[:120]}")
            continue
        # aggregate the measured columns (independent kernels -> whole-program time is their sum)
        r0 = k["rows"][0]
        sum_win += win["time_us"]
        if r0["native"] and r0["native"]["ok"]:
            sum_nat += r0["native"]["time_us"]
        if r0["default"]["ok"]:
            sum_dfl += r0["default"]["time_us"]

    linker = next((tc.cc for tc in toolchains), "gcc")
    whole = link_dir / "libtsvc_all.so"
    if archives:
        cmd = [
            linker, "-shared", "-fPIC", "-o",
            str(whole), "-Wl,--whole-archive", *[str(a) for a, _ in archives], "-Wl,--no-whole-archive"
        ]
        ok, _, err = run_compile(cmd)
        if ok:
            lib = ctypes.CDLL(str(whole))
            for _, symbols in archives:
                for symbol in symbols:  # every nest of the kernel must resolve in the whole-program .so
                    try:
                        _ = lib[symbol]
                        verified += 1
                    except (AttributeError, ValueError):
                        failed += 1
        else:
            notes.append(f"whole-program link failed: {err}")

    lines = [
        f"# TSVC whole-program link (seed {seed})", "",
        f"Linked {len(archives)} kernel winner libraries into `{whole.name}`; "
        f"{verified} symbols verified present, {failed} missing.", "", "| metric | total (us) |", "|---|---|",
        f"| Sum flag-matrix winner | {sum_win:.2f} |", f"| Sum native baseline | {sum_nat:.2f} |",
        f"| Sum default-flags | {sum_dfl:.2f} |"
    ]
    if sum_nat > 0:
        lines.append("")
        lines.append(f"**Whole-program (aggregate) speedup, winner vs native:** {sum_nat / sum_win:.3f}x "
                     "(TSVC kernels are independent, so the whole-program time is the sum of per-kernel times).")
    if notes:
        lines += ["", "## notes", ""] + [f"- {n}" for n in notes]
    report = "\n".join(lines) + "\n"
    (seed_dir / "whole_program.md").write_text(report)
    return report


# --- CLI --------------------------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="TSVC compiler-arena driver")
    ap.add_argument("--select", default="tsvc", help="kernel set (only 'tsvc' is supported today)")
    ap.add_argument("--strategy", default="skip-taskloops", help="nest-detection strategy")
    ap.add_argument("--opt-mode",
                    default="baseline",
                    choices=list(tsvc.OPT_MODES),
                    help="pre-split optimization mode: baseline (simplify+LoopToMap+MapFusion) "
                    "or canonicalize (extended-branch canonicalization)")
    ap.add_argument("--compilers", default="auto", help="'auto' or a whitespace list (gcc clang nvc++)")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random-sizes", action="store_true", help="sample sizes from the OptArena preset range")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these kernel keys (e.g. s000 s112)")
    ap.add_argument("--limit", type=int, default=None, help="stop after this many kernels (this rank's slice)")
    ap.add_argument("--out", default="perf_results/tsvc", help="results directory")
    ap.add_argument("--link", action="store_true", help="assemble winners into one whole-TSVC library + compare")
    ap.add_argument("--tables-only", action="store_true", help="merge existing per-kernel JSON into markdown")
    args = ap.parse_args(argv)

    if args.select != "tsvc":
        ap.error(f"--select {args.select!r} unsupported; only 'tsvc' is available")
    out = Path(args.out)

    if args.tables_only:
        print(render_tables(out, args.seed))
        return 0

    toolchains = discover_toolchains(args.compilers)
    if args.link:
        if not toolchains:
            print("[tsvc-arena] no toolchains discovered; cannot link")
            return 1
        print(link_whole_program(out, args.seed, toolchains, args.opt_mode, args.strategy))
        return 0

    if not toolchains:
        print("[tsvc-arena] no toolchains discovered (checked PATH + spack); nothing to run")
        return 1
    print("[tsvc-arena] toolchains: " +
          ", ".join(f"{t.name}(cc={Path(t.cc).name},cxx={Path(t.cxx).name if t.cxx else '-'},v{t.version[0]})"
                    for t in toolchains))

    procid, ntasks = rank_and_size()
    kernels = tsvc.iter_tsvc_kernels(only=args.only)
    mine = my_slice(kernels, procid, ntasks)
    if args.limit:
        mine = mine[:args.limit]
    seed_dir = ensure_seed_dir(out, args.seed)
    print(f"[tsvc-arena] rank {procid}/{ntasks}: {len(mine)} of {len(kernels)} kernels -> {seed_dir}")

    for i, kernel in enumerate(mine):
        workdir = Path(tempfile.mkdtemp(prefix=f"nf_tsvc_{kernel.key}_"))
        try:
            res = run_kernel(kernel, toolchains, args.strategy, args.opt_mode, args.seed, args.reps, args.random_sizes,
                             workdir)
        except Exception as e:  # pragma: no cover - a kernel must never crash the whole rank
            res = {"key": kernel.key, "seed": args.seed, "skipped": f"crash: {type(e).__name__}: {str(e)[:160]}"}
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        (seed_dir / f"{kernel.key}.json").write_text(json.dumps(res, indent=1))
        tag = res.get("skipped", "ok")
        print(f"[tsvc-arena] ({i + 1}/{len(mine)}) {kernel.key}: {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
