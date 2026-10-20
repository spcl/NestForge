# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Performance feedback for every phase: the whole program's time, where each kernel runs with the transfers that
implies, and each kernel's own time. Stages not reached yet run with their defaults on a copy (:meth:`Session.trial`),
so asking never changes the program being shaped."""

from __future__ import annotations

import copy
import ctypes
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import dace
import numpy as np
from pydantic import BaseModel, ConfigDict

from nestforge.build.arena import US_PER_S
from nestforge.build.isolation import Failure, Result, run_isolated, run_spawned
from nestforge.build.sdfg import BuiltSDFG, new_gpu_codegen
from nestforge.build.toolchain import parse_params, signature
from nestforge.stages.feedback import Evidence
from nestforge.stages.placement import Transfer


class Report(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProgramReport(Report):
    """The whole program at the session's sizes; ``error`` set when it did not build or run."""

    time_us: float | None = None
    error: str = ""


class TransferReport(Report):
    direction: str
    arg: str
    producer: str
    consumer: str
    #: ``None`` when ``arg`` is no array of the program.
    bytes: int | None


class OffloadReport(Report):
    #: Kernel -> ``cpu`` or ``gpu``.
    devices: dict[str, str]
    #: ``None`` when the dependency analysis refuses the program.
    transfers: list[TransferReport] | None
    transfer_bytes: int


class KernelReport(Report):
    name: str
    device: str
    correct: bool
    time_us: float | None
    share: float | None
    error: str = ""


class Performance(Report):
    """Every phase's feedback: ``structured`` (whole program), ``offloading`` and ``kernels``, and ranked hints."""

    structured: ProgramReport
    offloading: OffloadReport
    kernels: list[KernelReport]
    hints: str


class ProgramTime(Result):
    time_us: float


@dataclass(frozen=True, slots=True)
class ProgramCall:
    """A built program and its arguments; plain data, so it pickles into a spawned child."""

    so_path: str
    source: str
    name: str
    buffers: dict[str, np.ndarray]
    sizes: dict[str, int]
    reps: int


def run_program(call: ProgramCall) -> ProgramTime:
    """Init once, one warm-up call, then the mean of ``reps`` timed calls."""
    built = BuiltSDFG(
        call.name,
        Path(call.so_path),
        ctypes.CDLL(call.so_path),
        parse_params(signature(call.source, f"__dace_init_{call.name}")),
        parse_params(signature(call.source, f"__program_{call.name}")),
    )
    built.init(call.sizes)
    try:
        built.program(call.buffers, call.sizes)
        start = time.perf_counter()
        for _ in range(call.reps):
            built.program(call.buffers, call.sizes)
        return ProgramTime(time_us=(time.perf_counter() - start) / call.reps * US_PER_S)
    finally:
        built.close()


def program_buffers(sdfg: dace.SDFG, inputs: dict[str, np.ndarray], sizes: dict[str, int]) -> dict[str, np.ndarray]:
    """Every non-transient array of ``sdfg``: from ``inputs`` where given, else zeros of its shape at ``sizes``."""
    buffers: dict[str, np.ndarray] = {}
    for name, desc in sdfg.arrays.items():
        if desc.transient:
            continue
        dtype = desc.dtype.as_numpy_dtype()
        if name in inputs:
            buffers[name] = np.array(inputs[name], dtype=dtype, order="C", copy=True).reshape(-1)
        else:
            shape = [int(dace.symbolic.evaluate(extent, sizes)) for extent in desc.shape]
            buffers[name] = np.zeros(int(np.prod(shape)), dtype=dtype)
    return buffers


def time_program(
    sdfg: dace.SDFG, out_dir: Path, inputs: dict[str, np.ndarray], sizes: dict[str, int], reps: int, gpu: bool
) -> ProgramReport:
    """Build ``sdfg`` with DaCe (its CMake links the kernels' libraries and CUDA) and time it in an isolated child:
    forked on the CPU, spawned once a GPU context is needed."""
    program = copy.deepcopy(sdfg)
    try:
        with new_gpu_codegen() if gpu else nullcontext():
            with dace.config.set_temporary("default_build_folder", value=str(out_dir)):
                so_path = program.compile().filename
                frame = Path(program.build_folder) / "src" / "cpu" / f"{program.name}.cpp"
    except Exception as err:  # noqa: BLE001 -- a refused build is feedback, not a crash
        return ProgramReport(error=f"{type(err).__name__}: {err}")
    call = ProgramCall(so_path, frame.read_text(), program.name, program_buffers(sdfg, inputs, sizes), sizes, reps)
    timed = run_spawned(run_program, call) if gpu else run_isolated(lambda: run_program(call), ProgramTime)
    if isinstance(timed, Failure):
        return ProgramReport(error=timed.error)
    return ProgramReport(time_us=timed.time_us)


def offload_report(
    sdfg: dace.SDFG, devices: dict[str, str], moves: list[Transfer] | None, sizes: dict[str, int]
) -> OffloadReport:
    """Where each kernel runs and what crosses memory spaces, with bytes at ``sizes``."""
    transfers = None
    if moves is not None:
        transfers = [
            TransferReport(
                direction=move.direction,
                arg=move.arg,
                producer=move.producer,
                consumer=move.consumer,
                bytes=array_bytes(sdfg, move.arg, sizes),
            )
            for move in moves
        ]
    total = sum(move.bytes or 0 for move in transfers or ())
    return OffloadReport(devices=devices, transfers=transfers, transfer_bytes=total)


def array_bytes(sdfg: dace.SDFG, name: str, sizes: dict[str, int]) -> int | None:
    desc = sdfg.arrays.get(name)
    if desc is None:
        return None
    return int(dace.symbolic.evaluate(desc.total_size, sizes)) * desc.dtype.bytes


def kernel_reports(evidence: list[Evidence], devices: dict[str, str]) -> list[KernelReport]:
    """Each kernel's verdict, with its share of the correct kernels' time."""
    total = sum(ev.verdict.time_us for ev in evidence if ev.verdict.ok)
    return [
        KernelReport(
            name=ev.kernel,
            device=devices.get(ev.kernel, "cpu"),
            correct=ev.verdict.ok,
            time_us=ev.verdict.time_us if ev.verdict.ok else None,
            share=ev.verdict.time_us / total if ev.verdict.ok and total > 0 else None,
            error=ev.verdict.error,
        )
        for ev in evidence
    ]
