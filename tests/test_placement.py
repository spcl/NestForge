# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 4: parallel kernels go to the GPU target, sequential ones stay on the CPU, an agent may choose per kernel,
and the kernel DAG names the transfers each placement implies. A GPU kernel is offloaded inside its call; inserting the
copies moves them into the program."""

import dace
import numpy as np
import pytest
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace.libraries.standard.nodes.external_call import ExternalCall
from dace.sdfg.nodes import MapEntry

from nestforge.ir.depends import KernelGraph, kernel_dependencies
from nestforge.records import MoveStatus
from nestforge.session import Session
from nestforge.stages.canonicalize import Targets
from nestforge.stages.placement import OffloadKernels, Transfer, device_copies, transfers
from nestforge.stages.scopes import lower_nests_to_external_call

N = dace.symbol("N", dtype=dace.int64)


@dace.program
def scaled_sum(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = 2.0 * a[i] + b[i]


@dace.program
def chain(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    t = np.empty_like(a)
    for i in dace.map[0:N]:
        t[i] = a[i] + b[i]
    for i in dace.map[0:N]:
        c[i] = t[i] * 2.0


@dace.program
def scale_then_prefix(a: dace.float64[N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        a[i] = a[i] * 2.0
    for i in range(1, N):
        out[i] = out[i - 1] + a[i]


def lowered_graph(program) -> KernelGraph:
    sdfg = program.to_sdfg(simplify=True)
    lower_nests_to_external_call(sdfg)
    return kernel_dependencies(sdfg)


def kernel_session(gpu: bool, tmp_path, program=scaled_sum) -> Session:
    session = Session(program.to_sdfg(simplify=True), targets=Targets(gpu=gpu), work_dir=str(tmp_path))
    assert session.default_scopes()
    return session


def operand_storages(session: Session, name: str) -> list:
    ext = session.kernel(name)
    state = next(s for s in session.sdfg.states() if ext in s.nodes())
    operands = [e.src.data for e in state.in_edges(ext)] + [e.dst.data for e in state.out_edges(ext)]
    return [session.sdfg.arrays[operand].storage for operand in operands]


def test_cpu_target_keeps_the_kernel_on_the_host_and_leaves_the_graph_unchanged(tmp_path):
    session = kernel_session(False, tmp_path)
    before = session.sdfg.to_json()

    result = session.place()

    assert result.model_dump(mode="json") == {
        "status": "applied",
        "reason": "",
        "devices": {"extcall_doall0": "cpu"},
        "copies": [],
        "transfers": [],
    }
    assert session.sdfg.to_json() == before


def test_gpu_target_schedules_a_parallel_kernel_on_the_device_over_device_memory(tmp_path):
    session = kernel_session(True, tmp_path)

    devices = session.place().devices
    session.insert_copies()

    assert devices == {"extcall_doall0": "gpu"}
    assert session.kernel("extcall_doall0").schedule == dace.ScheduleType.GPU_Device
    storages = operand_storages(session, "extcall_doall0")
    assert len(storages) == 3 and all(storage in GPU_RESIDENT_STORAGES for storage in storages)
    session.sdfg.validate()


def test_inserted_copies_take_the_inputs_to_the_device_and_the_output_back(tmp_path):
    session = kernel_session(True, tmp_path)
    session.place()

    copies = session.insert_copies().copies

    arrays = session.sdfg.arrays
    to_device = sorted(src for src, dst in copies if arrays[dst].storage in GPU_RESIDENT_STORAGES)
    to_host = sorted(dst for src, dst in copies if arrays[src].storage in GPU_RESIDENT_STORAGES)
    assert to_device == ["a", "b"]
    assert to_host == ["c"]


def test_a_sequential_kernel_stays_on_the_cpu_beside_a_gpu_kernel_by_default(tmp_path):
    session = kernel_session(True, tmp_path, scale_then_prefix)  # the map, and the sequential loop as a kernel too
    parallel = {k.name: k.parallel for k in session.list_kernels()}
    assert sorted(parallel.values()) == [False, True], parallel

    devices = session.place().devices

    assert devices == {name: "gpu" if is_parallel else "cpu" for name, is_parallel in parallel.items()}
    session.sdfg.validate()


def test_an_agent_placement_keeps_a_parallel_kernel_on_the_host_and_its_operands_in_host_memory(tmp_path):
    session = kernel_session(True, tmp_path)

    result = session.place({"extcall_doall0": "cpu"})

    assert result.devices == {"extcall_doall0": "cpu"} and result.copies == ()
    assert not any(storage in GPU_RESIDENT_STORAGES for storage in operand_storages(session, "extcall_doall0"))


def test_placing_again_starts_from_the_program_before_the_first_placement(tmp_path):
    session = kernel_session(True, tmp_path)
    session.place()
    session.insert_copies()

    again = session.place({"extcall_doall0": "cpu"})

    assert again.status is MoveStatus.APPLIED and again.copies == ()
    assert session.kernel("extcall_doall0").schedule != dace.ScheduleType.GPU_Device


@pytest.mark.parametrize(
    ("gpu", "devices", "status", "reason"),
    [
        (True, {"extcall_9": "gpu"}, MoveStatus.ILLEGAL, "no kernel is named extcall_9"),
        (True, {"extcall_doall0": "tpu"}, MoveStatus.ILLEGAL, "'cpu' or 'gpu'"),
        (False, {"extcall_doall0": "gpu"}, MoveStatus.ILLEGAL, "GPU is not a target"),
    ],
)
def test_a_placement_that_cannot_apply_changes_nothing(tmp_path, gpu, devices, status, reason):
    session = kernel_session(gpu, tmp_path)
    before = session.sdfg.to_json()

    result = session.place(devices)

    assert result.status is status and reason in result.reason, result
    assert session.sdfg.to_json() == before


def test_default_scopes_after_gpu_placement_finds_no_further_scope(tmp_path):
    session = kernel_session(True, tmp_path)
    session.place()

    assert session.default_scopes() == []


def test_a_gpu_kernel_receives_its_program_inputs_and_returns_its_output_to_the_host():
    graph = lowered_graph(scaled_sum)

    moves = transfers(graph, {"extcall_0": "gpu"})

    assert moves == [
        Transfer("to_gpu", "a", "program", "extcall_0"),
        Transfer("to_gpu", "b", "program", "extcall_0"),
        Transfer("to_host", "c", "extcall_0.c", "exit"),
    ]


def test_a_host_kernel_feeding_a_gpu_kernel_moves_only_the_value_between_them():
    graph = lowered_graph(chain)
    assert graph.kernels == ("extcall_0", "extcall_1")

    moves = transfers(graph, {"extcall_0": "cpu", "extcall_1": "gpu"})

    assert moves == [
        Transfer("to_gpu", "t", "extcall_0.t", "extcall_1"),
        Transfer("to_host", "c", "extcall_1.c", "exit"),
    ]


def test_gpu_placement_transfers_name_the_containers_its_copies_move(tmp_path):
    session = kernel_session(True, tmp_path)
    session.place()

    result = session.insert_copies()

    arrays = session.sdfg.arrays
    to_device = sorted(src for src, dst in result.copies if arrays[dst].storage in GPU_RESIDENT_STORAGES)
    to_host = sorted(dst for src, dst in result.copies if arrays[src].storage in GPU_RESIDENT_STORAGES)
    assert sorted(move.arg for move in result.transfers if move.direction == "to_gpu") == to_device
    assert sorted(move.arg for move in result.transfers if move.direction == "to_host") == to_host


@pytest.mark.gpu
def test_placed_program_computes_on_the_gpu_what_numpy_computes(tmp_path):
    session = kernel_session(True, tmp_path)
    session.place()
    rng = np.random.default_rng(0)
    a, b, c = rng.random(256), rng.random(256), np.zeros(256)

    session.sdfg(a=a, b=b, c=c, N=256)

    np.testing.assert_allclose(c, 2.0 * a + b, rtol=1e-15, atol=0)


@dace.program
def doubled(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = 2.0 * a[i]


@dace.program
def calls_doubled(a: dace.float64[N], b: dace.float64[N]):
    doubled(a, b)


def test_gpu_offloading_takes_a_nested_sdfg_at_the_top_level_of_a_state_and_holds_its_maps_on_the_host():
    """DaCe places a host-level nested SDFG with a fresh ``type(self)(taskloop_overrides, host_maps)``, so
    ``OffloadKernels`` must build from DaCe's arguments alone (H1 `cloudsc_solver_carry`'s GPU reference raised
    a missing-argument ``TypeError`` there). Its maps are outside every kernel, so they stay host code."""
    sdfg = calls_doubled.to_sdfg(simplify=False)
    (nested,) = [
        node
        for state in sdfg.states()
        for node in state.nodes()
        if isinstance(node, dace.nodes.NestedSDFG) and state.entry_node(node) is None
    ]

    OffloadKernels(host_kernels=[]).apply_pass(sdfg, {})

    maps = [node for node, _ in nested.sdfg.all_nodes_recursive() if isinstance(node, dace.nodes.MapEntry)]
    assert maps and dace.ScheduleType.GPU_Device not in [entry.map.schedule for entry in maps]
    sdfg.validate()


@dace.program
def staged_scale(a: dace.float64[N], b: dace.float64[N]):
    t = np.empty_like(a)
    for i in dace.map[0:N]:
        t[i] = a[i] * 2.0
    for i in dace.map[0:N]:
        b[i] = t[N - 1 - i] + 1.0


def test_a_kernel_placed_on_the_gpu_keeps_its_own_arrays_off_the_host_stack(tmp_path):
    """Pinned to ``Register(force=True)`` when the scope was set, ``t`` stayed a host stack array the device kernels
    wrote through, corrupting the host stack."""
    session = Session(staged_scale.to_sdfg(simplify=True), targets=Targets(gpu=True), work_dir=str(tmp_path))
    session.describe()
    maps = [label for label, (obj, _) in session.row_index().items() if isinstance(obj, dace.nodes.MapEntry)]
    name = session.set_scope(maps).reason

    session.place({name: "gpu"})

    assert session.kernel(name).standalone_sdfg.arrays["t"].storage != dace.StorageType.Register


def scoped_session(tmp_path, program=staged_scale) -> tuple[Session, str]:
    """``program`` with its top-level doalls set as one scope by the agent, on a GPU target."""
    session = Session(program.to_sdfg(simplify=True), targets=Targets(gpu=True), work_dir=str(tmp_path))
    session.describe()
    maps = [label for label, (obj, _) in session.row_index().items() if isinstance(obj, dace.nodes.MapEntry)]
    return session, session.set_scope(maps).reason


def body_maps(ext) -> list:
    return [node.map.schedule for node, _ in ext.standalone_sdfg.all_nodes_recursive() if isinstance(node, MapEntry)]


def test_a_gpu_placed_scope_runs_device_maps_and_copies_its_arguments_inside_its_call(tmp_path):
    session, name = scoped_session(tmp_path)
    storages = {array: desc.storage for array, desc in session.sdfg.arrays.items()}

    result = session.place({name: "gpu"})

    ext = session.kernel(name)
    body = ext.standalone_sdfg
    assert isinstance(ext, ExternalCall) and ext.schedule == dace.ScheduleType.GPU_Device
    assert body_maps(ext) == [dace.ScheduleType.GPU_Device] * 2
    assert body.arrays["t"].storage in GPU_RESIDENT_STORAGES
    assert sorted(device_copies(body)) == [("_in_a", "_in_a_gpu"), ("_out_b_gpu", "_out_b")]
    assert result.copies == () and {a: d.storage for a, d in session.sdfg.arrays.items()} == storages
    session.sdfg.validate()


def test_a_cpu_scope_keeps_its_body_and_its_host_operands_beside_a_gpu_scope(tmp_path):
    session = kernel_session(True, tmp_path, chain)
    cpu, gpu = (kernel.name for kernel in session.list_kernels())
    before = session.kernel(cpu).standalone_sdfg.hash_sdfg()

    session.place({cpu: "cpu", gpu: "gpu"})
    session.insert_copies()

    assert session.kernel(cpu).schedule != dace.ScheduleType.GPU_Device
    assert session.kernel(cpu).standalone_sdfg.hash_sdfg() == before
    assert not any(storage in GPU_RESIDENT_STORAGES for storage in operand_storages(session, cpu))
    assert body_maps(session.kernel(gpu)) == [dace.ScheduleType.GPU_Device]


def test_inserted_copies_leave_the_gpu_body_over_device_arguments_with_no_copy_of_its_own(tmp_path):
    session, name = scoped_session(tmp_path)
    session.place({name: "gpu"})

    result = session.insert_copies()

    body = session.kernel(name).standalone_sdfg
    assert device_copies(body) == [] and body_maps(session.kernel(name)) == [dace.ScheduleType.GPU_Device] * 2
    assert sorted(src for src, _ in result.copies if src in ("a", "b")) == ["a"]
    assert all(storage in GPU_RESIDENT_STORAGES for storage in operand_storages(session, name))


def test_two_adjacent_gpu_kernels_pass_their_intermediate_on_the_device_once_copies_are_inserted(tmp_path):
    session = kernel_session(True, tmp_path, chain)
    session.place()

    copies = session.insert_copies().copies

    assert sorted(name for pair in copies for name in pair if not name.endswith("_gpu")) == ["a", "b", "c"]


def test_inserting_copies_before_any_placement_changes_nothing(tmp_path):
    session = kernel_session(True, tmp_path)
    before = session.sdfg.to_json()

    result = session.insert_copies()

    assert result.status is MoveStatus.ILLEGAL and "place first" in result.reason
    assert session.sdfg.to_json() == before


def test_default_placement_keeps_the_agents_devices_and_inserts_the_copies(tmp_path):
    session = kernel_session(True, tmp_path)
    session.place({"extcall_doall0": "cpu"})
    assert session.run_defaults("placement").status is MoveStatus.ILLEGAL  # placed, nothing on the GPU to copy

    session.place({"extcall_doall0": "gpu"})
    done = session.run_defaults("placement")

    assert done.devices == {"extcall_doall0": "gpu"} and done.copies
    assert all(storage in GPU_RESIDENT_STORAGES for storage in operand_storages(session, "extcall_doall0"))


@pytest.mark.gpu
@pytest.mark.parametrize("copies", [False, True], ids=["inside_the_call", "in_the_program"])
def test_a_gpu_scope_with_a_device_temporary_computes_what_numpy_computes(tmp_path, copies):
    session, name = scoped_session(tmp_path)
    session.place({name: "gpu"})
    if copies:
        session.insert_copies()
    a, b = np.random.default_rng(0).random(64), np.zeros(64)

    session.sdfg(a=a, b=b, N=64)

    np.testing.assert_allclose(b, a[::-1] * 2.0 + 1.0, rtol=1e-15, atol=0)
