# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Performance feedback for every phase: the whole program's time, where each kernel runs with the transfers that
implies, and each kernel's own time. Stages not reached yet run with their defaults on a copy (:meth:`Session.trial`),
so asking never changes the program being shaped."""

from __future__ import annotations

import copy
import ctypes
import time
from dataclasses import dataclass
from pathlib import Path

import dace
import numpy as np

from nestforge.build.arena import US_PER_S
from nestforge.build.isolation import Failure, Result, run_isolated, run_spawned
from nestforge.build.sdfg import BuiltSDFG, compile_linked_program
from nestforge.build.toolchain import parse_params, signature
from nestforge.records import Record
from nestforge.stages.feedback import Evidence
from nestforge.stages.placement import Transfer


class ProgramReport(Record):
    """The whole program at the session's sizes: its time, and whether its outputs match the reference (``None``
    without one; ``mismatched`` names the outputs that differ). ``error`` set when it did not build or run."""

    time_us: float | None = None
    correct: bool | None = None
    mismatched: list[str] = []
    error: str = ""


class TransferReport(Record):
    direction: str
    arg: str
    producer: str
    consumer: str
    #: ``None`` when ``arg`` is no array of the program.
    bytes: int | None


class OffloadReport(Record):
    #: Kernel -> ``cpu`` or ``gpu``.
    devices: dict[str, str]
    #: ``None`` when the dependency analysis refuses the program.
    transfers: list[TransferReport] | None
    transfer_bytes: int


class KernelReport(Record):
    name: str
    device: str
    correct: bool
    time_us: float | None
    share: float | None
    error: str = ""


class Performance(Record):
    """Every phase's feedback: ``structured`` (whole program), ``offloading`` and ``kernels``, and ranked hints."""

    structured: ProgramReport
    offloading: OffloadReport
    kernels: list[KernelReport]
    hints: str


class ProgramTime(Result):
    time_us: float
    mismatched: list[str] | None


@dataclass(frozen=True, slots=True)
class Reference:
    """The outputs the whole program must reproduce, and the tolerance it is graded at."""

    outputs: dict[str, np.ndarray]
    rtol: float
    atol: float

    def mismatched(self, buffers: dict[str, np.ndarray]) -> list[str]:
        """The outputs ``buffers`` gets wrong: exactly compared when both sides are integer, else ``allclose``."""
        wrong: list[str] = []
        for name, expected in self.outputs.items():
            got = buffers[name].reshape(np.shape(expected))
            if got.dtype.kind in "iub" and np.asarray(expected).dtype.kind in "iub":
                same = np.array_equal(got, expected)
            else:
                same = np.allclose(got, expected, rtol=self.rtol, atol=self.atol, equal_nan=True)
            if not same:
                wrong.append(name)
        return wrong


@dataclass(frozen=True, slots=True)
class ProgramCall:
    """A built program and its arguments; plain data, so it pickles into a spawned child."""

    so_path: str
    source: str
    name: str
    buffers: dict[str, np.ndarray]
    sizes: dict[str, int]
    reps: int
    reference: Reference | None


def run_program(call: ProgramCall) -> ProgramTime:
    """Init once, one call checked against the reference, then the mean of ``reps`` timed calls."""
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
        mismatched = call.reference.mismatched(call.buffers) if call.reference is not None else None
        start = time.perf_counter()
        for _ in range(call.reps):
            built.program(call.buffers, call.sizes)
        return ProgramTime(time_us=(time.perf_counter() - start) / call.reps * US_PER_S, mismatched=mismatched)
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
    sdfg: dace.SDFG,
    out_dir: Path,
    inputs: dict[str, np.ndarray],
    reference: Reference | None,
    sizes: dict[str, int],
    reps: int,
    gpu: bool,
) -> ProgramReport:
    """Build ``sdfg`` linking its kernels' libraries (:func:`compile_linked_program`) and time it in an isolated
    child: forked on the CPU, spawned once a GPU context is needed."""
    program = copy.deepcopy(sdfg)
    try:
        so_path = compile_linked_program(program, out_dir).filename
    except Exception as err:  # noqa: BLE001 -- a refused build is feedback, not a crash
        return ProgramReport(error=f"{type(err).__name__}: {err}")
    frame = out_dir / "src" / "cpu" / f"{program.name}.cpp"
    call = ProgramCall(
        so_path, frame.read_text(), program.name, program_buffers(sdfg, inputs, sizes), sizes, reps, reference
    )
    timed = run_spawned(run_program, call) if gpu else run_isolated(lambda: run_program(call), ProgramTime)
    if isinstance(timed, Failure):
        return ProgramReport(error=timed.error)
    if timed.mismatched is None:
        return ProgramReport(time_us=timed.time_us)
    return ProgramReport(time_us=timed.time_us, correct=not timed.mismatched, mismatched=timed.mismatched)


def offload_report(
    sdfg: dace.SDFG, devices: dict[str, str], moves: tuple[Transfer, ...] | None, sizes: dict[str, int]
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
