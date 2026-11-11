# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Moves inside a scope: the stage-2 menu rewrites one kernel's body, addressed by the kernel's name and the body's
labels, and a GPU kernel's body can stage its reads through shared memory. Neither touches the program around the
kernel or another kernel."""

import dace
import numpy as np
import pytest
from dace.sdfg.nodes import MapEntry

from nestforge.records import MoveStatus
from nestforge.session import Session
from nestforge.stages.canonicalize import Targets
from nestforge.stages.placement import has_device_memory

N = dace.symbol("N", dtype=dace.int64)


@dace.program
def chain(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    t = np.empty_like(a)
    for i in dace.map[1 : N - 1]:
        t[i] = a[i - 1] + a[i] + a[i + 1] + b[i]
    for i in dace.map[2 : N - 2]:
        c[i] = (t[i - 1] + t[i + 1]) * 2.0


def chain_reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    t = a[:-2] + a[1:-1] + a[2:] + b[1:-1]
    return (t[:-2] + t[2:]) * 2.0


def two_kernels(tmp_path, gpu: bool) -> tuple[Session, str, str]:
    session = Session(chain.to_sdfg(simplify=True), targets=Targets(gpu=gpu), work_dir=str(tmp_path))
    first, second = (kernel.name for kernel in session.default_scopes())
    return session, first, second


def placed_on_gpu(tmp_path) -> tuple[Session, str, str]:
    session, first, second = two_kernels(tmp_path, gpu=True)
    session.place({first: "gpu", second: "gpu"})
    session.insert_copies()
    return session, first, second


def body_map_schedules(session: Session, name: str) -> list[dace.ScheduleType]:
    body = session.kernel(name).standalone_sdfg
    return [node.map.schedule for node, _ in body.all_nodes_recursive() if isinstance(node, MapEntry)]


def only_move(session: Session, kind: str, scope: str) -> tuple[str, ...]:
    (move,) = session.list_moves(kind, scope=scope)
    assert move.scope == scope
    return move.labels


def test_a_tile_inside_a_scope_rewrites_that_kernels_body_and_nothing_around_it(tmp_path):
    session, first, second = two_kernels(tmp_path, gpu=False)
    program, other = session.sdfg.to_json(), session.kernel(second).standalone_sdfg.hash_sdfg()

    done = session.apply_move("tile-doall", only_move(session, "tile-doall", first), tile_sizes=[32], scope=first)

    assert done.status is MoveStatus.APPLIED and done.scope == first
    assert len(body_map_schedules(session, first)) == 2  # the tile map and the map over one tile
    assert session.kernel(second).standalone_sdfg.hash_sdfg() == other
    assert session.sdfg.to_json() == program


def test_scoping_one_doall_keeps_the_other_doalls_program_moves_and_labels(tmp_path):
    session = Session(chain.to_sdfg(simplify=True), targets=Targets(gpu=False), work_dir=str(tmp_path))
    session.describe()
    first, second = (label for label, (obj, _) in session.row_index().items() if isinstance(obj, MapEntry))
    before = [move for move in session.list_moves() if set(move.labels) == {second}]

    name = session.set_scope([first]).reason

    assert before and all(move in session.list_moves() for move in before)
    assert "tile-doall" in {move.kind for move in session.list_moves(scope=name)}


def test_a_move_inside_a_placed_gpu_scope_is_placed_again_with_the_change(tmp_path):
    session, first, _ = placed_on_gpu(tmp_path)

    session.apply_move("tile-doall", only_move(session, "tile-doall", first), tile_sizes=[32], scope=first)

    schedules = sorted(body_map_schedules(session, first), key=lambda schedule: schedule.value)
    assert schedules == [dace.ScheduleType.Sequential, dace.ScheduleType.GPU_Device]  # one thread per tile
    assert has_device_memory(session.sdfg)  # the copies stay inserted


def test_staging_reused_loads_gives_only_that_gpu_kernel_a_shared_tile(tmp_path):
    session, first, second = placed_on_gpu(tmp_path)
    program, other = session.sdfg.to_json(), session.kernel(second).standalone_sdfg.hash_sdfg()

    done = session.apply_move(
        "stage-loads-shared", only_move(session, "stage-loads-shared", first), tile_sizes=(64,), scope=first
    )

    body = session.kernel(first).standalone_sdfg
    shared = [name for name, desc in body.arrays.items() if desc.storage == dace.StorageType.GPU_Shared]
    assert done.status is MoveStatus.APPLIED and len(shared) == 1  # the tile of a, which neighbours reread; not b
    assert dace.ScheduleType.GPU_ThreadBlock in body_map_schedules(session, first)
    kernels = [
        n.map
        for n, _ in body.all_nodes_recursive()
        if isinstance(n, MapEntry) and n.map.schedule == dace.ScheduleType.GPU_Device
    ]
    assert [k.gpu_block_size for k in kernels] == [[64, 1, 1]]  # the block the agent passed, not the config default
    assert session.kernel(second).standalone_sdfg.hash_sdfg() == other
    assert session.sdfg.to_json() == program
    session.sdfg.validate()


def test_staging_without_a_thread_block_is_refused_and_says_what_to_pass(tmp_path):
    session, first, _ = placed_on_gpu(tmp_path)
    before = session.kernel(first).standalone_sdfg.hash_sdfg()

    done = session.apply_move("stage-loads-shared", only_move(session, "stage-loads-shared", first), scope=first)

    assert done.status is not MoveStatus.APPLIED and "pass tile_sizes, the thread block" in done.reason
    assert session.kernel(first).standalone_sdfg.hash_sdfg() == before


def test_inserting_copies_leaves_a_doall_outside_every_scope_on_the_host(tmp_path):
    session, first, second = two_kernels(tmp_path, gpu=True)
    assert session.unset_scope(first).status is MoveStatus.APPLIED
    session.place({second: "gpu"})

    session.insert_copies()

    top = [
        n.map.schedule
        for n, st in session.sdfg.all_nodes_recursive()
        if isinstance(n, MapEntry) and st.sdfg is session.sdfg and st.entry_node(n) is None
    ]
    assert len(top) == 1 and top[0] != dace.ScheduleType.GPU_Device  # the unscoped doall
    assert session.devices() == {second: "gpu"}
    session.sdfg.validate()


def test_a_cpu_scope_offers_no_shared_memory_staging(tmp_path):
    session, first, second = two_kernels(tmp_path, gpu=True)
    session.place({first: "cpu", second: "gpu"})
    session.insert_copies()

    assert session.list_moves("stage-loads-shared", scope=first) == []
    assert [move.scope for move in session.list_moves("stage-loads-shared", scope=second)] == [second]


@pytest.mark.gpu
def test_a_stencil_staging_its_reused_loads_in_shared_memory_computes_what_numpy_computes(tmp_path):
    session, first, _ = placed_on_gpu(tmp_path)
    session.apply_move(
        "stage-loads-shared", only_move(session, "stage-loads-shared", first), tile_sizes=(64,), scope=first
    )
    rng = np.random.default_rng(0)
    a, b, c = rng.random(1000), rng.random(1000), np.zeros(1000)

    session.sdfg(a=a, b=b, c=c, N=1000)

    np.testing.assert_allclose(c[2:-2], chain_reference(a, b), rtol=1e-15, atol=0)
