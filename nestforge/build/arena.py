# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Validation and timing of a built kernel: seeded inputs, the NumPy oracle, the FP-mode gate, and the ctypes call
that binds its arguments once and times repeated calls."""

from __future__ import annotations

import ctypes
import mmap
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from dace import symbolic

from nestforge.build import flags
from nestforge.build.isolation import Failure, Result, run_isolated
from nestforge.build.toolchain import POINTER_TYPE, CType, bind_argument, entry, needed_libraries, parse_params
from nestforge.corpus.translate import Prepared
from nestforge.ir.emit_python import factored_inputs, load_emitted, oracle_sdfg, scratch_arrays
from nestforge.ir.extract import Boundary


def resolve_shape(shape: Sequence[Any], sizes: dict[str, int]) -> tuple[int, ...]:
    env: dict[symbolic.symbol | str, int | float] = {symbolic.symbol(k): v for k, v in sizes.items()}
    return tuple(int(symbolic.evaluate(d, env)) for d in shape)


#: Upper bound of random inputs [0, INPUT_HIGH); must stay <= 1/4 so TSVC s232's squaring recurrence converges.
INPUT_HIGH = 0.25


def make_inputs(boundary: Boundary, sizes: dict[str, int], seed: int = 0) -> dict[str, np.ndarray]:
    """Seeded random inputs, and zeroed outputs and scratch buffers, all allocated by the caller."""
    sdfg = oracle_sdfg(boundary)  # the raw nest's scratch is too small
    rng = np.random.default_rng(seed)
    arrays: dict[str, np.ndarray] = {}
    out_only = [o for o in boundary.outputs if o not in boundary.inputs]
    zero_filled = out_only + [s for s in scratch_arrays(sdfg) if s not in boundary.inputs]
    for name in list(boundary.inputs) + zero_filled:
        desc = sdfg.arrays[name]
        shape = resolve_shape(desc.shape, sizes)
        dt = np.dtype(desc.dtype.type)
        arrays[name] = np.zeros(shape, dt) if name in zero_filled else (rng.random(shape) * INPUT_HIGH).astype(dt)
    for name in factored_inputs(sdfg):
        if name in boundary.inputs:
            arrays[name] = positive_definite(arrays[name])
    return arrays


def positive_definite(array: np.ndarray) -> np.ndarray:
    """``array`` made a positive definite matrix: a factorization of random numbers fails, and both sides of the
    comparison would stop at a pivot of their own."""
    return (array @ array.T + array.shape[0] * np.eye(array.shape[0])).astype(array.dtype)


def run_oracle(
    prep: Prepared, boundary: Boundary, inputs: dict[str, np.ndarray], sizes: dict[str, int]
) -> dict[str, np.ndarray]:
    """The outputs of the kernel's NumPy oracle on copies of ``inputs``."""
    args = {k: v.copy() for k, v in inputs.items()}
    call_oracle(prep, boundary, args, sizes)
    return {o: args[o] for o in boundary.outputs}


def call_oracle(prep: Prepared, boundary: Boundary, args: dict[str, np.ndarray], sizes: dict[str, int]) -> float:
    """The kernel's NumPy oracle on ``args``, in place; returns the seconds the call took."""
    missing = [s for s in boundary.symbols if s not in sizes]
    if missing:
        raise KeyError(
            f"no value for boundary symbol(s) {missing} (e.g. a loop index carried into an "
            f"extracted nest); pass them in `sizes`"
        )
    module = load_emitted(prep.numpy_source, prep.name)
    start = time.perf_counter()
    vars(module)[prep.name](**{**args, **{s: int(sizes[s]) for s in boundary.symbols}})
    return time.perf_counter() - start


def shared_copy(array: np.ndarray) -> np.ndarray:
    """``array`` in anonymous shared memory, so a forked child's writes reach the parent."""
    memory = mmap.mmap(-1, max(array.nbytes, 1))
    shared = np.ndarray(array.shape, array.dtype, buffer=memory)
    shared[...] = array
    return shared


class OracleDone(Result):
    """The oracle's child finished in ``seconds``; its outputs are in the shared arrays."""

    seconds: float


def bounded_oracle(
    prep: Prepared, boundary: Boundary, inputs: dict[str, np.ndarray], sizes: dict[str, int], timeout_s: float
) -> tuple[dict[str, np.ndarray], float] | Failure:
    """:func:`run_oracle` in a forked child, which an interpreted nest cannot hold past ``timeout_s``, nor a crash take
    the caller down: its outputs and the seconds of its call, or a :class:`Failure` when it did not finish. An oracle's
    loops are interpreted Python: hours at a size whose compiled kernel takes a minute."""
    args = {k: shared_copy(v) for k, v in inputs.items()}

    def work() -> OracleDone:
        return OracleDone(seconds=call_oracle(prep, boundary, args, sizes))

    done = run_isolated(work, OracleDone, timeout_s)
    return done if isinstance(done, Failure) else ({o: args[o] for o in boundary.outputs}, done.seconds)


def accumulating_outputs(boundary: Boundary, buffers: dict[str, np.ndarray]) -> list[str]:
    """Outputs the kernel both reads and writes; a timed rep loop restores these so an unrestored in-place
    kernel does not decay into denormals within a few reps and time subnormal arithmetic instead."""
    return [o for o in boundary.outputs if o in boundary.inputs and o in buffers]


def rewind_snapshot(boundary: Boundary, buffers: dict[str, np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each accumulating buffer paired with a pristine copy, taken once before the warm call for :func:`rewind`."""
    return [(buffers[o], buffers[o].copy()) for o in accumulating_outputs(boundary, buffers)]


def rewind(snapshot: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """Restore the pristine contents of every accumulating buffer, outside the timed region."""
    for buf, pristine in snapshot:
        buf[...] = pristine


def call_native(
    so: Path,
    symbol: str,
    order: list[str],
    argtypes: list[CType],
    boundary: Boundary,
    inputs: dict[str, np.ndarray],
    sizes: dict[str, int],
    reps: int,
    copy_inputs: bool = True,
    copy_outputs: bool = True,
) -> tuple[dict[str, np.ndarray] | None, float]:
    """Call the compiled entry once and snapshot its outputs, then time ``reps`` calls on the same buffers.

    ``order`` is the compiled signature's order, not the oracle's: same-typed buffers in the wrong slot go
    unnoticed. A read-write output is restored before every timed call (see :func:`accumulating_outputs`).
    ``copy_inputs=False`` works on the caller's buffers; ``copy_outputs=False`` skips the snapshot.
    """
    fn = entry(so, symbol, argtypes)
    work = {k: v.copy() for k, v in inputs.items()} if copy_inputs else inputs

    # bound once: per-call data_as would time Python marshaling
    args = [bind_argument(arg, ctype, work, sizes) for arg, ctype in zip(order, argtypes)]
    snapshot = rewind_snapshot(boundary, work)
    fn(*args)  # correctness run
    outputs = {o: work[o].copy() for o in boundary.outputs} if copy_outputs else None
    total = 0.0
    rewind(snapshot)  # the warm call primes the caches from the same state a timed rep sees
    warm = time.perf_counter()
    fn(*args)
    reps = reps_within_budget(reps, time.perf_counter() - warm)
    for _ in range(reps):
        rewind(snapshot)
        t0 = time.perf_counter()
        fn(*args)
        total += time.perf_counter() - t0
    elapsed_us = total / reps * US_PER_S
    return outputs, elapsed_us


#: Microseconds per second.
US_PER_S = 1e6

#: Timed calls per measurement, after one warm-up call.
TIMED_REPS = 10

#: Seconds of timed calls one measurement aims for: a call too long for :data:`TIMED_REPS` of them gets fewer, never none.
TIMED_BUDGET_S = 60.0


def reps_within_budget(reps: int, call_s: float) -> int:
    """``reps``, cut so that calls of ``call_s`` seconds stay near :data:`TIMED_BUDGET_S`; a call is never left untimed."""
    return max(1, min(reps, int(TIMED_BUDGET_S / call_s))) if call_s > 0.0 else reps


#: ``cudaMemcpyKind`` / ``hipMemcpyKind`` values (the same in both runtimes).
HOST_TO_DEVICE = 1
DEVICE_TO_HOST = 2

#: Each device runtime's soname prefix -> the prefix of its API's names.
DEVICE_RUNTIMES: dict[str, str] = {"libcudart": "cuda", "libamdhip64": "hip"}


@dataclass(frozen=True, slots=True)
class DeviceRuntime:
    """The device runtime a kernel library links (``libcudart`` or ``libamdhip64``), called by its API prefix."""

    library: ctypes.CDLL
    api: str

    def call(self, function: str, *args: object) -> int:
        """``<api><function>(*args)``, e.g. ``cudaMalloc`` or ``hipMalloc``; returns its status."""
        return int(self.library[f"{self.api}{function}"](*args))

    def check(self, status: int, what: str) -> None:
        if status != 0:
            raise RuntimeError(f"{what} failed with {self.api} status {status}")


def loaded_device_runtime(shared: Path) -> DeviceRuntime:
    """The device runtime an already loaded kernel library links, bound without loading a second copy."""
    for soname in needed_libraries(shared):
        for prefix, api in DEVICE_RUNTIMES.items():
            if soname.startswith(prefix):
                return DeviceRuntime(ctypes.CDLL(soname, mode=os.RTLD_NOLOAD), api)
    raise LookupError(f"{shared} links no device runtime ({', '.join(DEVICE_RUNTIMES)}); a device kernel must name it")


@dataclass(slots=True)
class DeviceMemory:
    """Device copies of a kernel's pointer arguments, allocated and freed through one device runtime."""

    runtime: DeviceRuntime
    pointers: dict[str, ctypes.c_void_p]

    def upload(self, name: str, host: np.ndarray) -> None:
        source = host.ctypes.data_as(ctypes.c_void_p)
        size = ctypes.c_size_t(host.nbytes)
        status = self.runtime.call("Memcpy", self.pointers[name], source, size, HOST_TO_DEVICE)
        self.runtime.check(status, f"copy of {name} to the device")

    def download(self, name: str, host: np.ndarray) -> None:
        target = host.ctypes.data_as(ctypes.c_void_p)
        size = ctypes.c_size_t(host.nbytes)
        status = self.runtime.call("Memcpy", target, self.pointers[name], size, DEVICE_TO_HOST)
        self.runtime.check(status, f"copy of {name} to the host")

    def free(self) -> None:
        for pointer in self.pointers.values():
            self.runtime.call("Free", pointer)


def device_memory(runtime: DeviceRuntime, buffers: dict[str, np.ndarray], names: Sequence[str]) -> DeviceMemory:
    """A device buffer per name, holding the host contents."""
    memory = DeviceMemory(runtime, {})
    for name in names:
        pointer = ctypes.c_void_p()
        size = ctypes.c_size_t(buffers[name].nbytes)
        runtime.check(runtime.call("Malloc", ctypes.byref(pointer), size), f"allocation of {name}")
        memory.pointers[name] = pointer
        memory.upload(name, buffers[name])
    return memory


def time_device_reps(
    fn: Any,
    args: list,
    memory: DeviceMemory,
    output_names: Sequence[str],
    restore: Sequence[str],
    host: dict[str, np.ndarray],
    reps: int,
) -> tuple[dict[str, np.ndarray], float]:
    """One correctness call and its outputs, a warm call, then ``reps`` timed calls; an accumulating output in
    ``restore`` is uploaded again from its pristine host copy before every call, outside the timed region."""
    fn(*args)  # the CPF entry synchronizes before it returns
    outputs = {name: host[name].copy() for name in output_names if name in memory.pointers}
    for name, buffer in outputs.items():
        memory.download(name, buffer)
    for name in restore:
        memory.upload(name, host[name])
    warm = time.perf_counter()
    fn(*args)
    reps, total = reps_within_budget(reps, time.perf_counter() - warm), 0.0
    for _ in range(reps):
        for name in restore:
            memory.upload(name, host[name])
        t0 = time.perf_counter()
        fn(*args)
        total += time.perf_counter() - t0
    return outputs, total / reps * US_PER_S


@dataclass(frozen=True, slots=True)
class DeviceCall:
    """A device kernel call a freshly spawned interpreter can make: every field pickles, and the entry's C
    parameter list stands in for its ctypes argument types."""

    shared: str
    symbol: str
    order: list[str]
    parameters: str
    output_names: list[str]
    restore: list[str]
    inputs: dict[str, np.ndarray]
    sizes: dict[str, int]
    reps: int


class DeviceResult(Result):
    """A device measurement: the kernel's outputs, read back once, and its time per call."""

    outputs: dict[str, np.ndarray]
    time_us: float


def call_device(call: DeviceCall) -> DeviceResult:
    """The spawned child's side of a device measurement: the kernel's outputs and its time per call. Every
    pointer argument is a device buffer, copied down once and read back once; scalars and sizes go by value."""
    argtypes = [param.ctype for param in parse_params(call.parameters)]
    fn = entry(call.shared, call.symbol, argtypes)
    host = {k: v.copy() for k, v in call.inputs.items()}
    on_device = [arg for arg, ctype in zip(call.order, argtypes) if arg in host and isinstance(ctype, POINTER_TYPE)]
    memory = device_memory(loaded_device_runtime(Path(call.shared)), host, on_device)
    args = [
        ctypes.cast(memory.pointers[arg], cast(type[ctypes._Pointer], ctype))
        if arg in memory.pointers
        else bind_argument(arg, ctype, host, call.sizes)
        for arg, ctype in zip(call.order, argtypes)
    ]
    try:
        outputs, time_us = time_device_reps(fn, args, memory, call.output_names, call.restore, host, call.reps)
    finally:
        memory.free()
    return DeviceResult(outputs=outputs, time_us=time_us)


def dtype_floor(arrays: dict[str, np.ndarray]) -> float:
    """The loosest :data:`flags.DTYPE_RTOL` floor among ``arrays`` (one ULP of the narrowest format present)."""
    return max(
        (flags.DTYPE_RTOL[v.dtype.name] for v in arrays.values() if v.dtype.name in flags.DTYPE_RTOL), default=0.0
    )


def rung_rtol(mode: str, floor: float) -> float:
    """The relative gate at FP rung ``mode``, never tighter than the dtype ``floor`` the outputs allow."""
    return max(flags.FP_RTOL[mode], floor)


def diff_stats(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> tuple[float, float]:
    """``(worst_abs, worst_scaled)`` over every array: the largest elementwise difference, and the same scaled by
    the larger magnitude compared, floored at 1.0 so a reduction's ULP noise passes while small values stay
    absolute. Both are ``inf`` on a non-finite difference and on a comparison of zero elements."""
    worst_abs, worst_rel = 0.0, 0.0
    compared = False
    for k in a:
        if not a[k].size:
            continue
        compared = True
        # at least float64: bool cannot subtract and unsigned would wrap; complex keeps its magnitude
        diff = np.abs(np.subtract(a[k], b[k], dtype=np.result_type(a[k], b[k], np.float64)))
        d_abs = float(np.max(diff))
        if not np.isfinite(d_abs):
            return float("inf"), float("inf")
        scale = np.maximum(np.maximum(np.abs(a[k]), np.abs(b[k])), 1.0)
        with np.errstate(invalid="ignore"):  # inf/inf is nan, a failure caught below
            d_rel = float(np.max(diff / scale))
        if not np.isfinite(d_rel):
            return float("inf"), float("inf")
        worst_abs = max(worst_abs, d_abs)
        worst_rel = max(worst_rel, d_rel)
    if not compared:
        return float("inf"), float("inf")
    return worst_abs, worst_rel
