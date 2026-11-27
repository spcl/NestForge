# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 4: the device of every kernel, the host/device copies DaCe's offloading inserts for it, and the
transfers the kernel DAG implies. Default: parallel kernels on the GPU when it is a target, the rest on the CPU.

A GPU kernel is lowered inside its call: DaCe's ``OffloadToAccelerator`` runs on the kernel's body, its arguments in
the memory the call passes them in. :func:`place` passes host memory, so the body copies at its own boundary and the
program around the call is unchanged; :func:`insert_copies` offloads the program first, so the copies sit around the
calls, shared by adjacent GPU kernels, and the bodies receive device memory."""

from __future__ import annotations

from dataclasses import dataclass

import dace
from dace import dtypes
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace.libraries.standard.nodes.copy import CopyLibraryNode
from dace.libraries.standard.nodes.external_call import ExternalCall, call_site, external_calls
from dace.ordered import OrderedSet
from dace.sdfg import nodes
from dace.transformation.passes.offloading.host_maps import HostMapSpec
from dace.transformation.passes.offloading.offload_to_accelerator import OffloadToAccelerator

from tactics.ir.depends import KernelGraph
from tactics.stages.canonicalize import Targets
from tactics.stages.expand import is_device_implementation, is_host_implementation
from tactics.stages.scopes import is_parallel_kernel, kernel_runtime_sized

DEVICES = ("cpu", "gpu")


class OffloadKernels(OffloadToAccelerator):
    """DaCe's offloading, with the kernels named in ``host_kernels`` kept on the host: the copy analysis then
    places their operands in host memory and copies around them.

    The positional parameters are DaCe's, because DaCe places each host-level nested SDFG with a fresh
    ``type(self)(taskloop_overrides, host_maps)``. ``host_kernels`` defaults to none there: that body offloader only
    places copies; the schedules, the one place ``host_kernels`` is read, are assigned tree-wide by the outer pass."""

    __slots__ = ("host_kernels",)

    def __init__(
        self,
        taskloop_overrides: dict[str, bool] | None = None,
        host_maps: HostMapSpec = False,
        *,
        host_kernels: list[str] | None = None,
    ) -> None:
        super().__init__(taskloop_overrides, host_maps)
        self.host_kernels = list(host_kernels or ())

    def find_taskloops(self, sdfg: dace.SDFG) -> None:
        """DaCe's host maps, and every map outside the kernels pinned to the host: only a scope placed on the GPU
        runs there, so the program's own maps stay host code, as in its CPU form."""
        super().find_taskloops(sdfg)
        for state in sdfg.states():
            self._host_pinned |= OrderedSet(n for n in state.scope_children()[None] if isinstance(n, nodes.MapEntry))

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


def kernel_interfaces(sdfg: dace.SDFG) -> dict[str, tuple[object, ...]]:
    """Per kernel: its device and each argument's storage, type and passing; a build fits while they are unchanged."""
    out: dict[str, tuple[object, ...]] = {}
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, ExternalCall):
            site = call_site(node, state)
            args = [(conn, desc.storage, desc.dtype, conn in site.by_value) for conn, desc in site.descriptors.items()]
            out[node.name] = (kernel_device(node), *sorted(args, key=str))
    return out


def pinned_implementations(ext: ExternalCall) -> list[str]:
    """The implementations ``set-library-implementation`` pinned on the library calls inside kernel ``ext``."""
    body = ext.standalone_sdfg
    if body is None:
        return []
    return [
        n.implementation
        for n, _ in body.all_nodes_recursive()
        if isinstance(n, nodes.LibraryNode) and not n.auto_select_implementation and n.implementation
    ]


def default_device(ext: ExternalCall, targets: Targets) -> str:
    """A kernel holding a call pinned to a device library on the GPU, one pinned to a host library on the CPU;
    otherwise parallel kernels (doalls, library calls among them) on the GPU when it is a target, except one that
    sizes a temporary while it runs (:func:`kernel_runtime_sized`): a GPU kernel cannot allocate it."""
    pinned = pinned_implementations(ext)
    if targets.gpu and any(is_device_implementation(i) for i in pinned):
        return "gpu"
    if any(is_host_implementation(i) for i in pinned):
        return "cpu"
    return "gpu" if targets.gpu and is_parallel_kernel(ext) and not kernel_runtime_sized(ext) else "cpu"


def default_devices(sdfg: dace.SDFG, targets: Targets) -> dict[str, str]:
    """Every kernel on its :func:`default_device`."""
    return {ext.name: default_device(ext, targets) for ext in external_calls(sdfg)}


def on_device(state: dace.SDFGState, node: nodes.AccessNode) -> bool:
    return node.desc(state.sdfg).storage in GPU_RESIDENT_STORAGES


def edge_copies(state: dace.SDFGState) -> list[tuple[str, str]]:
    """Access-to-access edges of ``state`` whose two ends live in different memory spaces. An empty memlet only
    orders the two accesses and copies nothing, so it is no copy."""
    return [
        (edge.src.data, edge.dst.data)
        for edge in state.edges()
        if isinstance(edge.src, nodes.AccessNode)
        and isinstance(edge.dst, nodes.AccessNode)
        and not edge.data.is_empty()
        and on_device(state, edge.src) != on_device(state, edge.dst)
    ]


def copy_node_ends(state: dace.SDFGState, node: CopyLibraryNode) -> tuple[nodes.AccessNode, nodes.AccessNode] | None:
    """The access nodes ``node`` copies from and to, as DaCe's ``src_storage``/``dst_storage`` find them; ``None``
    when either side is unwired or no access node."""
    ins = [e for e in state.in_edges(node) if e.dst_conn == CopyLibraryNode.INPUT_CONNECTOR_NAME]
    outs = [e for e in state.out_edges(node) if e.src_conn == CopyLibraryNode.OUTPUT_CONNECTOR_NAME]
    if not ins or not outs:
        return None
    src, dst = state.memlet_path(ins[0])[0].src, state.memlet_path(outs[0])[-1].dst
    if not isinstance(src, nodes.AccessNode) or not isinstance(dst, nodes.AccessNode):
        return None
    return src, dst


def library_copies(state: dace.SDFGState) -> list[tuple[str, str]]:
    """The ``CopyLibraryNode`` copies of ``state`` between the two memory spaces."""
    ends = [copy_node_ends(state, n) for n in state.nodes() if isinstance(n, CopyLibraryNode)]
    return [
        (src.data, dst.data)
        for src, dst in (pair for pair in ends if pair is not None)
        if on_device(state, src) != on_device(state, dst)
    ]


def device_copies(sdfg: dace.SDFG) -> list[tuple[str, str]]:
    """``(source, destination)`` containers of every host/device copy of the program, in state order, nested SDFGs
    included: an access-to-access edge (:func:`edge_copies`) or a ``CopyLibraryNode`` (:func:`library_copies`). A
    kernel's body is no nested SDFG of the program, so a copy inside a call is not listed."""
    return [
        pair
        for nested in sdfg.all_sdfgs_recursive()
        for state in nested.states()
        for pair in edge_copies(state) + library_copies(state)
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
    return runtime_sized_on_gpu(sdfg, devices)


def runtime_sized_on_gpu(sdfg: dace.SDFG, devices: dict[str, str]) -> str | None:
    """Why a kernel ``devices`` puts on the GPU cannot run there: it sizes a temporary while it runs; or ``None``."""
    for ext in external_calls(sdfg):
        sized = kernel_runtime_sized(ext) if devices.get(ext.name) == "gpu" else ()
        if sized:
            return (
                f"{ext.name} sizes {', '.join(sized)} while it runs, which a GPU kernel cannot allocate; "
                f"place it on the cpu."
            )
    return None


def offload_kernel_bodies(sdfg: dace.SDFG) -> None:
    """DaCe's ``OffloadToAccelerator`` inside every GPU kernel's body, each argument in the call's memory for it."""
    for ext, state in sdfg.all_nodes_recursive():
        if not isinstance(ext, ExternalCall) or kernel_device(ext) != "gpu" or ext.standalone_sdfg is None:
            continue
        body = ext.standalone_sdfg
        bound = [(e.dst_conn, e.data.data) for e in state.in_edges(ext)]
        bound += [(e.src_conn, e.data.data) for e in state.out_edges(ext)]
        for conn, data in bound:
            if conn is None or data is None:
                continue
            storage, desc = state.sdfg.arrays[data].storage, body.arrays[conn]
            if isinstance(desc, dace.data.Scalar) and storage in GPU_RESIDENT_STORAGES:
                # DaCe's offloading sees host reads of a device length-1 array, never of a device Scalar (#93)
                body.arrays[conn] = dace.data.Array(desc.dtype, (1,), storage=storage)
            else:
                desc.storage = storage
        OffloadToAccelerator().apply_pass(body, {})


def place(sdfg: dace.SDFG, devices: dict[str, str]) -> Placement:
    """Run every kernel on its device of ``devices`` (checked by :func:`check_devices`), in place: a GPU kernel's body
    is offloaded inside its call, with its copies at its boundary; nothing outside the calls changes."""
    for ext in external_calls(sdfg):
        gpu = devices[ext.name] == "gpu"
        ext.schedule = dtypes.ScheduleType.GPU_Device if gpu else dtypes.ScheduleType.Default
    offload_kernel_bodies(sdfg)
    return Placement({ext.name: kernel_device(ext) for ext in external_calls(sdfg)}, tuple(device_copies(sdfg)))


def insert_copies(sdfg: dace.SDFG, devices: dict[str, str]) -> Placement:
    """Like :func:`place` on an unplaced ``sdfg``, but DaCe's ``OffloadToAccelerator`` offloads the program first: the
    GPU kernels' operands live in device memory and the copies sit at the kernels' boundaries in the program."""
    if "gpu" in devices.values():
        OffloadKernels(host_kernels=[name for name, device in devices.items() if device == "cpu"]).apply_pass(sdfg, {})
        offload_kernel_bodies(sdfg)
    return Placement({ext.name: kernel_device(ext) for ext in external_calls(sdfg)}, tuple(device_copies(sdfg)))


def has_device_memory(sdfg: dace.SDFG) -> bool:
    """Whether :func:`insert_copies` put any container of the program in device memory."""
    return any(desc.storage in GPU_RESIDENT_STORAGES for desc in sdfg.arrays.values())


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
