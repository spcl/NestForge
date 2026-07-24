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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import dace
from dace import symbolic

from nestforge.build import COMPILE_TIMEOUT_S, ldconfig_output
from nestforge.emit_numpy import load_emitted, maxsize_loop_scratch, scratch_arrays
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
# numpy dtype name -> ctypes scalar for the emitted kernel's ABI. ``bool`` is needed because a comparison
# materialises a boolean transient, which DaCe lowers to a 1-byte C ``bool``.
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
@dataclass(slots=True)
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
    # generic ``libblas.so``: may be netlib OR an alternatives symlink, so labelled by soname only
    "blas": ["blas"],
}


def ldconfig_sonames() -> set:
    """Base library names (``openblas`` from ``libopenblas.so.0``) known to the dynamic linker."""
    out = ldconfig_output()
    names = set()
    for line in out.splitlines():
        token = line.strip().split(" ", 1)[0]
        if token.startswith("lib") and ".so" in token:
            names.add(token[3:token.index(".so")])
    return names


def discover_blas_libraries() -> Dict[str, BlasBackend]:
    """Discover installed BLAS backends (OpenBLAS / MKL / BLIS / ATLAS / netlib) + their link flags.

    An extra link axis for kernels whose emitted numpy uses ``@``/``np.dot``. Probes the dynamic linker
    cache plus ``MKLROOT``.
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
                # -rpath paired with -L: an MKLROOT install is off the loader path, so without it the
                # linked .so needs LD_LIBRARY_PATH to run
                found["mkl"] = BlasBackend("mkl", [f"-L{libdir}", f"-Wl,-rpath,{libdir}", "-lmkl_rt"])
                break
    return found


# --- data generation from the manifest --------------------------------------------------------
def resolve_shape(shape: Sequence[Any], sizes: Dict[str, int]) -> Tuple[int, ...]:
    env = {symbolic.symbol(k): v for k, v in sizes.items()}
    return tuple(int(symbolic.evaluate(d, env)) for d in shape)


def emitted_sdfg(boundary: Boundary) -> dace.SDFG:
    """The descriptors the EMITTED kernel is written against, not the raw nest's.

    The emitter widens a loop-sized scratch transient (``maxsize_loop_scratch``) before rendering, so
    caller-side allocation must use the SAME widened descriptor -- sizing from
    ``boundary.standalone_sdfg`` gives a smaller buffer than the kernel writes: a heap overflow.
    """
    return maxsize_loop_scratch(boundary.standalone_sdfg, boundary.symbols)


def scratch_names(boundary: Boundary) -> List[str]:
    """Transient array buffers the C-style kernel expects the caller to pre-allocate."""
    return scratch_arrays(emitted_sdfg(boundary))


#: Upper bound of the random-input range ``[0, INPUT_HIGH)``. Must stay <= 1/4 so a squaring recurrence
#: ``x = x*x + b`` (TSVC s232) has an attracting fixed point instead of overflowing to inf (nan maxdiff ->
#: spurious validation failure); non-negative keeps ``sqrt``/``log`` kernels real.
INPUT_HIGH = 0.25


def make_inputs(boundary: Boundary,
                sizes: Dict[str, int],
                seed: int = 0,
                given: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
    """Random arrays for inputs; zeros for outputs and scratch buffers (all caller-pre-allocated).

    Inputs are drawn from ``[0, INPUT_HIGH)`` -- see :data:`INPUT_HIGH` for why the range is conditioned.

    ``given`` supplies ready-made values a uniform float fill cannot express, chiefly the index arrays of
    :func:`nestforge.tsvc.index_fills`. It is checked against the resolved shape/dtype: it crosses the ABI
    as the kernel's own buffer, so a mismatch would corrupt memory instead of raising.
    """
    sdfg = emitted_sdfg(boundary)  # widened scratch: allocate what the kernel indexes, not the raw shape
    rng = np.random.default_rng(seed)
    given = given or {}
    arrays: Dict[str, np.ndarray] = {}
    out_only = [o for o in boundary.outputs if o not in boundary.inputs]
    zero_filled = out_only + [s for s in scratch_arrays(sdfg) if s not in boundary.inputs]
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
    module = load_emitted(prep.numpy_source, prep.name)
    args = {k: v.copy() for k, v in inputs.items()}
    call = {**args, **{s: int(sizes[s]) for s in boundary.symbols}}
    vars(module)[prep.name](**call)
    return {o: args[o] for o in boundary.outputs}


# --- compile + call ---------------------------------------------------------------------------
@dataclass(slots=True)
class Cell:
    compiler: str
    fp_mode: str
    ok: bool
    maxdiff: float
    time_us: float
    compile_us: float = 0.0  # wall time of THIS candidate's compile (the post-optimization toolchain cost)
    so_path: Optional[str] = None
    symbol: Optional[str] = None
    #: The EMITTED signature's parameter order (harness.signature_order); use it verbatim. numpyto orders
    #: by param_order() (arrays sorted, then scalars), not the manifest's role order -- re-deriving it
    #: elsewhere silently swaps pointers.
    abi_order: Optional[List[str]] = None
    error: Optional[str] = None


def scalar_ctype(sdfg: dace.SDFG, name: str) -> type[ctypes._SimpleCData]:
    """ctypes type of a by-value (non-array) kernel arg, matching the translator's signature.

    A float value scalar is ``double`` -> ``c_double``. EVERY integer symbol is emitted ``int64_t`` by the
    translator regardless of the SDFG's own int width, so it must be ``c_int64`` here -- a 32-bit
    ``c_int`` leaves the upper half of the register garbage and blows the loop bound out of range."""
    if name in sdfg.symbols and np.dtype(sdfg.symbols[name].type).kind == "f":
        return ctypes.c_double
    return ctypes.c_int64


def resolve_argtypes(order: List[str], boundary: Boundary) -> list:
    """ctypes argtypes for the emitted entry, in the order the EMITTED SIGNATURE declares.

    ``order`` must come from parsing the generated source (``harness.signature_order``), NOT the
    manifest's ``input_args``: numpyto emits ``param_order()`` (arrays sorted, then scalars), which
    coincides with the manifest's role order only by luck of the alphabet.
    """
    sdfg = boundary.standalone_sdfg
    types = []
    for arg in order:
        if arg in sdfg.arrays:
            dt = np.dtype(sdfg.arrays[arg].dtype.type).name
            types.append(ctypes.POINTER(CTYPE[dt]))
        else:
            types.append(scalar_ctype(sdfg, arg))
    return types


def call_native(so: Path, symbol: str, order: List[str], argtypes: list, boundary: Boundary,
                inputs: Dict[str, np.ndarray], sizes: Dict[str, int], reps: int) -> Tuple[Dict[str, np.ndarray], float]:
    """Bind + call the compiled entry. ``order`` is the EMITTED-signature parameter order (see
    :func:`resolve_argtypes`); binding by the manifest's role order instead puts each buffer in the wrong
    parameter slot, which same-typed arrays make completely silent."""
    lib = ctypes.CDLL(str(so))
    fn = lib[symbol]  # ctypes CDLL indexing (not getattr) to bind the kernel symbol
    fn.argtypes = argtypes
    fn.restype = None
    work = {k: v.copy() for k, v in inputs.items()}

    def build_args() -> list:
        out = []
        for arg, at in zip(order, argtypes):
            if arg in work:
                out.append(work[arg].ctypes.data_as(at))
            else:
                out.append(at(sizes[arg]))  # at is the by-value ctype (c_int64 size / c_double value scalar)
        return out

    # bind ONCE (every rep reuses these buffers): per-rep data_as would time Python marshaling
    args = build_args()
    fn(*args)  # correctness run
    outputs = {o: work[o].copy() for o in boundary.outputs}
    # Restore every written buffer before each rep (outside the timed region): an in-place nest otherwise
    # feeds on its own output, reaching denormals/Inf in a few reps and timing those instead.
    mutated = [o for o in boundary.outputs if o in work]
    total = 0.0
    for _ in range(reps):
        for name in mutated:
            work[name][...] = inputs[name]
        t0 = time.perf_counter()
        fn(*args)
        total += time.perf_counter() - t0
    elapsed_us = total / reps * 1e6
    return outputs, elapsed_us


def maxdiff(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> float:
    """Largest absolute elementwise difference; ``inf`` if any difference is non-finite.

    The non-finite mapping is load-bearing: builtin ``max`` DROPS a non-first NaN (``nan > x`` is False),
    so a NaN-poisoned kernel would report 0.0 and win the arena.
    """
    worst = 0.0
    for k in a:
        if not a[k].size:
            continue
        d = float(np.max(np.abs(a[k] - b[k])))
        if not np.isfinite(d):
            return float("inf")
        worst = max(worst, d)
    return worst


def diff_stats(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Tuple[float, float]:
    """``(worst_abs, worst_scaled)`` in ONE pass over the abs-difference, instead of :func:`maxdiff` and
    :func:`relative_maxdiff` each recomputing ``np.abs(a[k] - b[k])`` separately. Same semantics as calling
    both (NaN/Inf still map to ``inf`` in both slots); use this wherever a caller needs both numbers."""
    worst_abs, worst_rel = 0.0, 0.0
    for k in a:
        if not a[k].size:
            continue
        diff = np.abs(a[k] - b[k])
        d_abs = float(np.max(diff))
        if not np.isfinite(d_abs):
            return float("inf"), float("inf")
        scale = np.maximum(np.maximum(np.abs(a[k]), np.abs(b[k])), 1.0)
        with np.errstate(invalid="ignore"):  # inf/inf -> nan, which is a FAILURE, not a warning
            d_rel = float(np.max(diff / scale))
        if not np.isfinite(d_rel):
            return float("inf"), float("inf")
        worst_abs = max(worst_abs, d_abs)
        worst_rel = max(worst_rel, d_rel)
    return worst_abs, worst_rel


def relative_maxdiff(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> float:
    """Largest elementwise difference SCALED by the magnitude of the values it is between.

    An absolute gate is unreachable for a reduction: summing 32000 order-1 elements lands near 1.6e4,
    where one fp64 ULP (1.8e-12) already exceeds the 1e-14 default, so a correct vectorized reduce is
    recorded WRONG and the kernel silently vanishes from the corpus. The denominator floors at 1.0, so
    small values keep the absolute reading -- the gate is never loosened below what fp64 promises, and a
    real miscompile (far more than a few ULP) is still caught. NaN/Inf still fail.
    """
    worst = 0.0
    for k in a:
        if not a[k].size:
            continue
        scale = np.maximum(np.maximum(np.abs(a[k]), np.abs(b[k])), 1.0)
        with np.errstate(invalid="ignore"):  # inf/inf -> nan, which is a FAILURE, not a warning
            d = float(np.max(np.abs(a[k] - b[k]) / scale))
        # builtin max(0.0, nan) is 0.0, i.e. a PERFECT match for a NaN-poisoned result -- map to inf
        if not np.isfinite(d):
            return float("inf")
        worst = max(worst, d)
    return worst


@dataclass(slots=True)
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

    ``given`` is forwarded to :func:`make_inputs`; this layer is corpus-agnostic, so a caller measuring a
    corpus kernel must pass ``tsvc.index_fills(...)`` -- without it an integer index array fills to
    all-zeros and the sweep times a degenerate gather while validating vacuously."""
    out_dir.mkdir(parents=True, exist_ok=True)
    compilers = discover_compilers()
    symbol = f"{prep.name}_fp64"
    # bind order comes from the REAL emitted signature, never the manifest (see resolve_argtypes).
    # Imported here because harness imports arena: a top-level import would cycle.
    from nestforge.perf.harness import signature_order
    order = signature_order(c_source.read_text(), symbol)
    argtypes = resolve_argtypes(order, boundary)
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

            # Forked so a segfault/runaway kills only the child; only the summary crosses the pipe.
            # ``so``/``mode`` are default args to dodge late-binding capture of the loop variables.
            def work(so: Path = so, mode: str = mode) -> Dict[str, Any]:
                outs, us = call_native(so, symbol, order, argtypes, boundary, inputs, sizes, reps)
                # report the ABSOLUTE difference, gate on the scaled one (see relative_maxdiff); one pass
                # over the diff computes both instead of two.
                md, md_rel = diff_stats(oracle, outs)
                return {"ok": bool(md_rel <= MODE_ATOL[mode]), "maxdiff": float(md), "time_us": float(us)}

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
                         symbol=symbol,
                         abi_order=list(order)))

    result.optimization_seconds = time.perf_counter() - t_sweep
    for mode in FP_MODES:
        correct = [c for c in result.cells if c.fp_mode == mode and c.ok]
        if correct:
            result.winners[mode] = min(correct, key=lambda c: c.time_us)
    return result


def run_tool(cmd: List[str], what: str) -> None:
    """Run one toolchain command with a deadline and CAPTURED stderr, so a failure raises with the compiler's
    actual diagnostic instead of dumping it to the console and raising a bare CalledProcessError."""
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{what} exceeded {COMPILE_TIMEOUT_S}s: {' '.join(cmd)}") from None
    if done.returncode != 0:
        raise RuntimeError(f"{what} failed ({done.returncode}): {' '.join(cmd)}\n{done.stderr[-2000:]}")


def compile_object(cpath: str, fp_mode: str, c_source: Path, name: str, out_dir: Path) -> Path:
    """Compile one emitted C source to a ``.o`` with a chosen ``(compiler, fp-mode)`` -- shared by the
    winner-archive and the per-backend E1 variant. ``-fPIC`` (in every ``FP_MODES`` entry) makes the
    object linkable into the parent ``.so``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    obj = out_dir / f"{name}_nest.o"
    run_tool([cpath, *FP_MODES[fp_mode], "-c", str(c_source), "-o", str(obj)], f"compiling {name} with {cpath}")
    return obj


def archive_objects(objs: List[Path], name: str, out_dir: Path) -> Path:
    """Bundle objects into ``lib<name>_nest.a`` (single-object offload today: one winning nest per archive).

    WARNING: never put several nests' objects in one archive -- DaCe SORTS the parent's link flags and can
    place the archive before the parent objects, so ld pulls no member and later references stay
    unresolved (``undefined symbol`` at ``dlopen``). The sort also scrambles the
    ``--whole-archive``/``--no-whole-archive`` pair. Use :func:`link_shared` for a multi-nest swap."""
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"lib{name}_nest.a"
    if archive.exists():
        archive.unlink()  # ar r APPENDS; start clean so a rebuild doesn't stack stale members
    run_tool(["ar", "rcs", str(archive), *[str(o) for o in objs]], f"archiving {name}")
    return archive


def link_shared(objs: List[Path], name: str, out_dir: Path, cpath: str) -> Path:
    """Link objects into ``lib<name>_nest.so``. Resolved at ``dlopen`` time, hence order-independent, so it
    survives dace SORTING the parent's link flags (which leaves a static archive's members un-pulled). The
    parent links it with an rpath (see :meth:`ExternLibEnv.configure`)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    so = out_dir / f"lib{name}_nest.so"
    run_tool([cpath, "-shared", "-fPIC", *[str(o) for o in objs], "-o", str(so)], f"linking lib{name}_nest.so")
    return so


def build_winner_archive(win: Cell, c_source: Path, name: str, out_dir: Path) -> Path:
    """Materialize a winning cell as a static ``lib<name>_nest.a`` for STATIC offload into a parent SDFG.

    Recompiles the SAME source with the winner's ``(compiler, fp-mode)`` to an object and archives it. An
    archive carries objects only, no linked runtime, so the parent supplies the single libomp instead of
    every nest ``.so`` dragging its own."""
    obj = compile_object(discover_compilers()[win.compiler], win.fp_mode, c_source, name, out_dir)
    return archive_objects([obj], name, out_dir)
