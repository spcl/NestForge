# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 4: the device of every kernel, the host/device copies DaCe's offloading inserts for it, and the
transfers the kernel DAG implies. Default: parallel kernels on the GPU when it is a target, the rest on the CPU."""

from __future__ import annotations

from dataclasses import dataclass

import dace
from dace import dtypes
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace.libraries.standard.nodes.external_call import ExternalCall, external_calls
from dace.sdfg import nodes
from dace.transformation.passes.offloading.offload_to_accelerator import OffloadToAccelerator

from nestforge.ir.depends import KernelGraph
from nestforge.stages.canonicalize import Targets
from nestforge.stages.scopes import is_parallel_kernel

DEVICES = ("cpu", "gpu")


class OffloadKernels(OffloadToAccelerator):
    """DaCe's offloading, with the kernels named in ``host_kernels`` kept on the host: the copy analysis then
    places their operands in host memory and copies around them."""

    __slots__ = ("host_kernels",)

    def __init__(self, host_kernels: list[str]) -> None:
        super().__init__()
        self.host_kernels = host_kernels

    def assign_schedules(self, sdfg: dace.SDFG, host_level: bool = True) -> None:
        super().assign_schedules(sdfg, host_level)
        for ext in external_calls(sdfg):
            if ext.name in self.host_kernels:
                ext.schedule = dtypes.ScheduleType.Default


@dataclass(frozen=True, slots=True)
class Placement:
    """Device per kernel name, and ``(source, destination)`` containers of every host/device copy."""

    devices: dict[str, str]
    copies: tuple[tuple[str, str], ...]


def kernel_device(ext: ExternalCall) -> str:
    return "gpu" if ext.schedule in dtypes.GPU_SCHEDULES else "cpu"


def default_devices(sdfg: dace.SDFG, targets: Targets) -> dict[str, str]:
    """Parallel kernels on the GPU when it is a target; every other kernel on the CPU."""
    return {ext.name: "gpu" if targets.gpu and is_parallel_kernel(ext) else "cpu" for ext in external_calls(sdfg)}


def on_device(state: dace.SDFGState, node: nodes.AccessNode) -> bool:
    return node.desc(state.sdfg).storage in GPU_RESIDENT_STORAGES


def device_copies(sdfg: dace.SDFG) -> list[tuple[str, str]]:
    """Access-to-access edges whose two ends live in different memory spaces, in state order."""
    return [
        (edge.src.data, edge.dst.data)
        for state in sdfg.states()
        for edge in state.edges()
        if isinstance(edge.src, nodes.AccessNode)
        and isinstance(edge.dst, nodes.AccessNode)
        and on_device(state, edge.src) != on_device(state, edge.dst)
    ]


def check_devices(sdfg: dace.SDFG, targets: Targets, devices: dict[str, str]) -> str | None:
    """Why ``devices`` is not a placement of every kernel of ``sdfg``, or ``None``."""
    names = [ext.name for ext in external_calls(sdfg)]
    unknown = [name for name in devices if name not in names]
    if unknown:
        return f"no kernel is named {', '.join(unknown)}; kernels are {', '.join(names)}."
    bad = [f"{name}={device}" for name, device in devices.items() if device not in DEVICES]
    if bad:
        return f"{', '.join(bad)}: a device is 'cpu' or 'gpu'."
    if not targets.gpu and "gpu" in devices.values():
        return "the GPU is not a target of this session; every kernel runs on the CPU."
    return None


def place(sdfg: dace.SDFG, targets: Targets, devices: dict[str, str]) -> Placement:
    """Run every kernel on its device of ``devices`` (checked by :func:`check_devices`), in place. With a kernel on
    the GPU, DaCe's ``OffloadToAccelerator`` schedules it and inserts the copies; with none nothing changes."""
    if "gpu" in devices.values():
        OffloadKernels([name for name, device in devices.items() if device == "cpu"]).apply_pass(sdfg, {})
    return Placement({ext.name: kernel_device(ext) for ext in external_calls(sdfg)}, tuple(device_copies(sdfg)))


@dataclass(frozen=True, slots=True)
class Transfer:
    """A value crossing memory spaces on one dependency edge: ``producer`` feeds ``consumer``'s ``arg``."""

    direction: str
    arg: str
    producer: str
    consumer: str


def crossings(arg: str, producers: list[str], consumer: str, device: str, devices: dict[str, str]) -> list[Transfer]:
    """The transfers of ``arg`` into ``consumer`` on ``device``; a producer that is no kernel is host memory."""
    direction = "to_gpu" if device == "gpu" else "to_host"
    return [
        Transfer(direction, arg, label, consumer)
        for label in producers
        if devices.get(label.partition(".")[0], "cpu") != device
    ]


def transfers(graph: KernelGraph, devices: dict[str, str]) -> list[Transfer]:
    """Every kernel input and program exit whose producer sits in the other memory space. Program inputs, host
    code and the program exit are host memory; ``devices`` maps each kernel to ``cpu`` or ``gpu``."""
    moves: list[Transfer] = []
    for edge in graph.edges:
        if edge.role == "input":
            moves += crossings(edge.arg, edge.labels(), edge.consumer, devices[edge.consumer], devices)
    for edge in graph.exits:
        moves += crossings(edge.arg, edge.labels(), "exit", "cpu", devices)
    return moves
