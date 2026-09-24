# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Validation and timing of a built kernel: seeded inputs, the NumPy oracle, the FP-mode gate, and the ctypes call
that binds its arguments once and times repeated calls."""

from __future__ import annotations

import ctypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from collections.abc import Sequence

import numpy as np

from dace import symbolic

from nestforge.build import flags
from nestforge.build.toolchain import POINTER_TYPE, CType, bind_argument, entry, needed_libraries, parse_params
from nestforge.ir.emit_numpy import load_emitted, maxsize_loop_scratch, scratch_arrays
from nestforge.ir.extract import Boundary
from nestforge.corpus.translate import Prepared


def resolve_shape(shape: Sequence[Any], sizes: dict[str, int]) -> tuple[int, ...]:
    env: dict[symbolic.symbol | str, int | float] = {symbolic.symbol(k): v for k, v in sizes.items()}
    return tuple(int(symbolic.evaluate(d, env)) for d in shape)


#: Upper bound of random inputs [0, INPUT_HIGH); must stay <= 1/4 so TSVC s232's squaring recurrence converges.
INPUT_HIGH = 0.25


def make_inputs(boundary: Boundary, sizes: dict[str, int], seed: int = 0) -> dict[str, np.ndarray]:
    """Seeded random inputs, and zeroed outputs and scratch buffers, all allocated by the caller."""
    sdfg = maxsize_loop_scratch(boundary.standalone_sdfg, boundary.symbols)  # the raw nest's scratch is too small
    rng = np.random.default_rng(seed)
    arrays: dict[str, np.ndarray] = {}
    out_only = [o for o in boundary.outputs if o not in boundary.inputs]
    zero_filled = out_only + [s for s in scratch_arrays(sdfg) if s not in boundary.inputs]
    for name in list(boundary.inputs) + zero_filled:
        desc = sdfg.arrays[name]
        shape = resolve_shape(desc.shape, sizes)
        dt = np.dtype(desc.dtype.type)
        arrays[name] = np.zeros(shape, dt) if name in zero_filled else (rng.random(shape) * INPUT_HIGH).astype(dt)
    return arrays


def run_oracle(
    prep: Prepared, boundary: Boundary, inputs: dict[str, np.ndarray], sizes: dict[str, int]
) -> dict[str, np.ndarray]:
    """The outputs of the kernel's NumPy oracle on copies of ``inputs``."""
    missing = [s for s in boundary.symbols if s not in sizes]
    if missing:
        raise KeyError(
            f"no value for boundary symbol(s) {missing} (e.g. a loop index carried into an "
            f"extracted nest); pass them in `sizes`"
        )
    module = load_emitted(prep.numpy_source, prep.name)
    args = {k: v.copy() for k, v in inputs.items()}
    call = {**args, **{s: int(sizes[s]) for s in boundary.symbols}}
    vars(module)[prep.name](**call)
    return {o: args[o] for o in boundary.outputs}


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

    ``order`` is the compiled signature's order, not the manifest's: same-typed buffers in the wrong slot go
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
    fn(*args)
    for _ in range(reps):
        rewind(snapshot)
        t0 = time.perf_counter()
        fn(*args)
        total += time.perf_counter() - t0
    elapsed_us = total / reps * 1e6
    return outputs, elapsed_us


#: ``cudaMemcpyKind`` values.
HOST_TO_DEVICE = 1
DEVICE_TO_HOST = 2


def cuda_check(status: int, what: str) -> None:
    if status != 0:
        raise RuntimeError(f"{what} failed with CUDA status {status}")


def loaded_cudart(shared: Path) -> ctypes.CDLL:
    """The ``libcudart`` an already loaded kernel library links, bound without loading a second copy."""
    soname = next((name for name in needed_libraries(shared) if name.startswith("libcudart")), None)
    if soname is None:
        raise LookupError(f"{shared} does not link libcudart; a device kernel must name its runtime")
    return ctypes.CDLL(soname, mode=os.RTLD_NOLOAD)


@dataclass(slots=True)
class DeviceMemory:
    """Device copies of a kernel's pointer arguments, allocated and freed through one ``libcudart``."""

    cudart: ctypes.CDLL
    pointers: dict[str, ctypes.c_void_p]

    def upload(self, name: str, host: np.ndarray) -> None:
        source = host.ctypes.data_as(ctypes.c_void_p)
        status = self.cudart.cudaMemcpy(self.pointers[name], source, ctypes.c_size_t(host.nbytes), HOST_TO_DEVICE)
        cuda_check(status, f"copy of {name} to the device")

    def download(self, name: str, host: np.ndarray) -> None:
        target = host.ctypes.data_as(ctypes.c_void_p)
        status = self.cudart.cudaMemcpy(target, self.pointers[name], ctypes.c_size_t(host.nbytes), DEVICE_TO_HOST)
        cuda_check(status, f"copy of {name} to the host")

    def free(self) -> None:
        for pointer in self.pointers.values():
            self.cudart.cudaFree(pointer)


def device_memory(cudart: ctypes.CDLL, buffers: dict[str, np.ndarray], names: Sequence[str]) -> DeviceMemory:
    """A device buffer per name, holding the host contents."""
    memory = DeviceMemory(cudart, {})
    for name in names:
        pointer = ctypes.c_void_p()
        cuda_check(
            cudart.cudaMalloc(ctypes.byref(pointer), ctypes.c_size_t(buffers[name].nbytes)), f"allocation of {name}"
        )
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
    total = 0.0
    for rep in range(reps + 1):
        for name in restore:
            memory.upload(name, host[name])
        t0 = time.perf_counter()
        fn(*args)
        total += (time.perf_counter() - t0) if rep > 0 else 0.0
    return outputs, total / reps * 1e6


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


def call_device(call: DeviceCall) -> dict[str, object]:
    """The spawned child's side of a device measurement: the kernel's outputs and its time per call. Every
    pointer argument is a device buffer, copied down once and read back once; scalars and sizes go by value."""
    argtypes = [param.ctype for param in parse_params(call.parameters)]
    fn = entry(call.shared, call.symbol, argtypes)
    host = {k: v.copy() for k, v in call.inputs.items()}
    on_device = [arg for arg, ctype in zip(call.order, argtypes) if arg in host and isinstance(ctype, POINTER_TYPE)]
    memory = device_memory(loaded_cudart(Path(call.shared)), host, on_device)
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
    return {"outputs": outputs, "time_us": time_us}


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
