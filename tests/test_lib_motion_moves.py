# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``lib-before`` / ``lib-after`` / ``lib-into``: a library call whose memlets are its whole effect (a ``lib_copy``
above all, which every copy between two arrays becomes), or a host ``Sync``, moves into a state of its own before or
after a block of an enclosing region, or joins a state there, judged by the read and write sets of what it crosses."""

import pathlib
import re

import dace
import numpy as np
from dace import dtypes
from dace.libraries.standard.nodes.code import CodeLibraryNode
from dace.libraries.standard.nodes.copy import CopyLibraryNode
from dace.libraries.standard.nodes.sync import Sync, SyncScope
from dace.sdfg.state import LoopRegion

from nestforge.records import MoveStatus
from nestforge.session import Session
from nestforge.stages.moves import plan_move

N = dace.symbol("N", dtype=dace.int64)
K = dace.symbol("K", dtype=dace.int64)
SIZE = 8
TRIPS = 40


@dace.program
def snapshot_loop(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for k in range(TRIPS):
        b[:] = a
        for i in dace.map[0:N]:
            c[i] = c[i] + b[i] * k


@dace.program
def snapshot_sized_loop(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for k in range(K):
        b[:] = a
        for i in dace.map[0:N]:
            c[i] = c[i] + b[i] * k


@dace.program
def snapshot_of_a_changing_array(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for k in range(TRIPS):
        b[:] = a
        for i in dace.map[0:N]:
            c[i] = c[i] + b[i] * k
        for i in dace.map[0:N]:
            a[i] = c[i] * 0.5


@dace.program
def copy_between_loops(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
    for i in range(1, N):
        a[i] = np.sin(a[i - 1]) + c[i]
    d[:] = c
    for i in range(1, N):
        b[i] = np.sin(b[i - 1]) + a[i]


@dace.program
def copy_of_the_first_loop(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
    for i in range(1, N):
        a[i] = np.sin(a[i - 1]) + c[i]
    d[:] = a
    for i in range(1, N):
        b[i] = np.sin(b[i - 1]) + d[i]


@dace.program
def guarded_copy(a: dace.float64[N], b: dace.float64[N], c: dace.float64[1]):
    if c[0] > 0.0:
        b[:] = a


@dace.program
def copied_chain(a: dace.float64[N], u: dace.float64[N], c: dace.float64[N]):
    t = np.empty_like(a)
    for i in dace.map[0:N]:
        t[i] = a[i] + 1.0
    u[:] = t
    for i in dace.map[0:N]:
        c[i] = u[i] * 2.0 + t[i]


@dace.program
def copy_after_a_loop(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
    for i in dace.map[0:N]:
        a[i] = c[i] * 2.0
    for i in range(1, N):
        b[i] = np.sin(b[i - 1]) + c[i]
    d[:] = a


def session(program, tmp_path: pathlib.Path) -> Session:
    sut = Session(program.to_sdfg(simplify=True), work_dir=str(tmp_path))
    sut.canonicalize()
    return sut


def copy_rows(sut: Session) -> list[str]:
    return [label for label, (obj, _) in sut.row_index().items() if isinstance(obj, CopyLibraryNode)]


def block_of(sut: Session, kind: type) -> list[str]:
    return [label for label, (obj, state) in sut.row_index().items() if state is None and isinstance(obj, kind)]


def copy_states(sdfg: dace.SDFG) -> list[dace.SDFGState]:
    return [state for state in sdfg.states() for n in state.nodes() if isinstance(n, CopyLibraryNode)]


def in_a_loop(block: dace.SDFGState) -> bool:
    parent = block.parent_graph
    while parent is not None and not isinstance(parent, dace.SDFG):
        if isinstance(parent, LoopRegion):
            return True
        parent = parent.parent_graph
    return False


def arrays(seed: int, count: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.random(SIZE) for _ in range(count)]


def run(sdfg: dace.SDFG, names: str, values: list[np.ndarray], **symbols: int) -> list[np.ndarray]:
    copies = [v.copy() for v in values]
    sdfg(**dict(zip(names, copies)), N=SIZE, **symbols)
    return copies


def test_a_copy_between_arrays_is_a_lib_copy_row_after_canonicalization(tmp_path):
    sut = session(copy_between_loops, tmp_path)

    (row,) = copy_rows(sut)

    assert re.fullmatch(r"lib_copy\d+", row), row
    assert f"{row}  LIBNODE Copy [" in sut.view(expand=[sut.row_index()[row][1].label])


def test_a_lib_copy_row_is_a_copy_edge_again_once_scopes_are_made_so_no_kernel_copies(tmp_path):
    sut = session(copied_chain, tmp_path)
    assert len(copy_rows(sut)) == 1

    sut.default_scopes()

    assert copy_rows(sut) == []
    kernels = {kernel.name: kernel.depends for kernel in sut.list_kernels()}
    assert len(kernels) == 2 and all("copy" not in name for name in kernels), kernels


def test_a_loop_invariant_copy_hoists_out_of_its_loop_and_runs_once(tmp_path):
    sut = session(snapshot_loop, tmp_path)
    expected = run(sut.sdfg, "abc", arrays(0, 3))
    (row,) = copy_rows(sut)
    (loop,) = block_of(sut, LoopRegion)

    result = sut.apply_move("lib-before", [row, loop])

    assert result.status is MoveStatus.APPLIED, result
    (state,) = copy_states(sut.sdfg)
    assert not in_a_loop(state)
    for got, want in zip(run(sut.sdfg, "abc", arrays(0, 3)), expected):
        np.testing.assert_allclose(got, want, rtol=1e-14)


def test_a_loop_whose_trip_count_is_not_proven_positive_keeps_its_copy(tmp_path):
    sut = session(snapshot_sized_loop, tmp_path)
    (row,) = copy_rows(sut)
    (loop,) = block_of(sut, LoopRegion)

    result = sut.apply_move("lib-before", [row, loop])

    assert result.status is MoveStatus.ILLEGAL and "not proven to run once" in result.reason, result
    assert in_a_loop(copy_states(sut.sdfg)[0])


def test_a_copy_of_an_array_the_loop_writes_is_not_hoisted(tmp_path):
    sut = session(snapshot_of_a_changing_array, tmp_path)
    (row,) = copy_rows(sut)
    (loop,) = block_of(sut, LoopRegion)

    result = sut.apply_move("lib-before", [row, loop])

    assert result.status is MoveStatus.ILLEGAL and "writes a" in result.reason, result
    assert in_a_loop(copy_states(sut.sdfg)[0])


def test_a_copy_read_in_its_loop_is_not_sunk_after_it(tmp_path):
    sut = session(snapshot_loop, tmp_path)
    (row,) = copy_rows(sut)
    (loop,) = block_of(sut, LoopRegion)

    result = sut.apply_move("lib-after", [row, loop])

    assert result.status is MoveStatus.ILLEGAL and "is read" in result.reason, result


def test_a_copy_between_two_loops_moves_ahead_of_them_and_leaves_them_adjacent(tmp_path):
    sut = session(copy_between_loops, tmp_path)
    expected = run(sut.sdfg, "abcd", arrays(1, 4))
    (row,) = copy_rows(sut)
    first, _ = block_of(sut, LoopRegion)

    result = sut.apply_move("lib-before", [row, first])

    assert result.status is MoveStatus.APPLIED, result
    loops = [b for b in sut.sdfg.nodes() if isinstance(b, LoopRegion)]
    (between,) = sut.sdfg.out_edges(loops[0])
    assert between.dst is loops[1]
    (state,) = copy_states(sut.sdfg)
    assert sut.sdfg.successors(state) == [loops[0]]
    for got, want in zip(run(sut.sdfg, "abcd", arrays(1, 4)), expected):
        np.testing.assert_allclose(got, want, rtol=1e-14)


def test_a_copy_of_what_a_loop_computes_is_not_moved_ahead_of_that_loop(tmp_path):
    sut = session(copy_of_the_first_loop, tmp_path)
    (row,) = copy_rows(sut)
    first, second = block_of(sut, LoopRegion)

    ahead = sut.apply_move("lib-before", [row, first])
    behind = sut.apply_move("lib-after", [row, second])

    assert ahead.status is MoveStatus.ILLEGAL and "writes a" in ahead.reason, ahead
    assert behind.status is MoveStatus.ILLEGAL and "reads d" in behind.reason, behind


def test_a_copy_never_moves_into_a_block(tmp_path):
    sut = session(copy_between_loops, tmp_path)
    (row,) = copy_rows(sut)
    first, _ = block_of(sut, LoopRegion)
    inside = sut.row_index()[first][0].start_block.label

    result = sut.apply_move("lib-after", [row, inside])

    assert result.status is MoveStatus.ILLEGAL and "never into a block" in result.reason, result


def test_a_copy_under_a_condition_does_not_leave_its_branch(tmp_path):
    sut = session(guarded_copy, tmp_path)
    (row,) = copy_rows(sut)
    outer = next(label for label, (obj, st) in sut.row_index().items() if st is None and obj.parent_graph is sut.sdfg)

    result = sut.apply_move("lib-before", [row, outer])

    assert result.status is MoveStatus.ILLEGAL and "under a condition" in result.reason, result


def test_a_call_named_with_its_own_state_is_refused(tmp_path):
    sut = session(copy_between_loops, tmp_path)
    (row,) = copy_rows(sut)
    own = sut.row_index()[row][1].label

    result = sut.apply_move("lib-after", [row, own])

    assert result.status is MoveStatus.ILLEGAL and "already" in result.reason, result


def test_a_call_with_side_effects_never_moves():
    sdfg = dace.SDFG("opaque")
    first = sdfg.add_state("first", is_start_block=True)
    second = sdfg.add_state_after(first, "second")
    call = CodeLibraryNode([], [], name="opaque")
    second.add_node(call)

    result = plan_move("lib-before", [(call, second), (first, None)])

    assert isinstance(result, str) and "side effects" in result, result


def test_a_copy_joins_an_earlier_state_and_reads_what_that_state_writes(tmp_path):
    sut = session(copy_after_a_loop, tmp_path)
    expected = run(sut.sdfg, "abcd", arrays(2, 4))
    (row,) = copy_rows(sut)
    producer = next(s for s in sut.sdfg.nodes() if isinstance(s, dace.SDFGState) and s is not sut.row_index()[row][1])

    result = sut.apply_move("lib-into", [row, producer.label])

    assert result.status is MoveStatus.APPLIED, result
    (state,) = copy_states(sut.sdfg)
    assert state is producer
    (copy_node,) = [n for n in state.nodes() if isinstance(n, CopyLibraryNode)]
    (source,) = [e.src for e in state.in_edges(copy_node)]
    assert source.data == "a" and any(not e.data.is_empty() for e in state.in_edges(source))
    for got, want in zip(run(sut.sdfg, "abcd", arrays(2, 4)), expected):
        np.testing.assert_allclose(got, want, rtol=1e-14)


def test_a_call_joins_only_a_state(tmp_path):
    sut = session(copy_after_a_loop, tmp_path)
    (row,) = copy_rows(sut)
    (loop,) = block_of(sut, LoopRegion)

    result = sut.apply_move("lib-into", [row, loop])

    assert result.status is MoveStatus.ILLEGAL and "is a LoopRegion" in result.reason, result


def copy_then_target(target_writes: int, wcr: bool) -> tuple[dace.SDFG, CopyLibraryNode, dace.SDFGState]:
    """``target`` writes ``x`` ``target_writes`` times (unordered, or once through a ``+`` accumulation), then a state
    copies ``x`` to ``y``."""
    sdfg = dace.SDFG("joined")
    for name in "xy":
        sdfg.add_array(name, [4], dace.float64)
    target = sdfg.add_state("target", is_start_block=True)
    for index in range(target_writes):
        tasklet = target.add_tasklet(f"w{index}", {}, {"out"}, "out = 1.0")
        target.add_edge(
            tasklet, "out", target.add_access("x"), None, dace.Memlet("x[0]", wcr="lambda a, b: a + b" if wcr else None)
        )
    copier = sdfg.add_state_after(target, "copier")
    node = CopyLibraryNode("copy")
    copier.add_node(node)
    copier.add_edge(copier.add_access("x"), None, node, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.Memlet("x[0:4]"))
    copier.add_edge(node, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, copier.add_access("y"), None, dace.Memlet("y[0:4]"))
    return sdfg, node, copier


def test_a_call_does_not_join_a_state_that_accumulates_into_what_it_touches():
    sdfg, node, copier = copy_then_target(1, wcr=True)

    result = plan_move("lib-into", [(node, copier), (sdfg.start_block, None)])

    assert isinstance(result, str) and "accumulates into x" in result, result


def test_a_call_does_not_join_a_state_with_two_unordered_writes_of_what_it_reads():
    sdfg, node, copier = copy_then_target(2, wcr=False)

    result = plan_move("lib-into", [(node, copier), (sdfg.start_block, None)])

    assert isinstance(result, str) and "no order between the writes" in result, result


def device_program(barrier_scope: SyncScope = SyncScope.DEVICE) -> tuple[dace.SDFG, Sync, list[dace.SDFGState]]:
    """``kernel`` (a GPU doall writing device array ``g``), ``host`` (host work on ``z``), ``wait`` (a ``Sync``)."""
    sdfg = dace.SDFG("device")
    sdfg.add_array("g", [4], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array("z", [4], dace.float64)
    kernel = sdfg.add_state("kernel", is_start_block=True)
    kernel.add_mapped_tasklet(
        "fill",
        {"i": "0:4"},
        {},
        "out = 1.0",
        {"out": dace.Memlet("g[i]")},
        dtypes.ScheduleType.GPU_Device,
        external_edges=True,
    )
    host = sdfg.add_state_after(kernel, "host")
    host.add_mapped_tasklet("bump", {"i": "0:4"}, {}, "out = 2.0", {"out": dace.Memlet("z[i]")}, external_edges=True)
    wait = sdfg.add_state_after(host, "wait")
    barrier = Sync("barrier", scope=barrier_scope)
    wait.add_node(barrier)
    return sdfg, barrier, [kernel, host, wait]


def test_a_host_barrier_moves_ahead_of_host_work_that_touches_no_device_data():
    sdfg, barrier, (kernel, host, wait) = device_program()

    plan = plan_move("lib-before", [(barrier, wait), (host, None)])

    assert not isinstance(plan, str), plan
    plan.commit()
    (moved,) = [s for s in sdfg.states() if barrier in s.nodes()]
    assert sdfg.successors(kernel) == [moved] and sdfg.successors(moved) == [host]


def test_a_host_barrier_never_crosses_device_work():
    sdfg, barrier, (kernel, host, wait) = device_program()

    result = plan_move("lib-before", [(barrier, wait), (kernel, None)])

    assert isinstance(result, str) and "kernel issues device work" in result, result


def test_a_host_barrier_joins_a_state_after_its_device_work():
    sdfg, barrier, (kernel, host, wait) = device_program()

    plan = plan_move("lib-into", [(barrier, wait), (kernel, None)])

    assert not isinstance(plan, str), plan
    plan.commit()
    assert barrier in kernel.nodes()
    (kernel_exit,) = [n for n in kernel.nodes() if isinstance(n, dace.nodes.MapExit)]
    assert kernel_exit in kernel.predecessors(barrier)


def test_a_host_barrier_never_joins_ahead_of_device_work():
    sdfg, barrier, (kernel, host, wait) = device_program()
    later = sdfg.add_state_after(wait, "later")
    later.add_mapped_tasklet(
        "use",
        {"i": "0:4"},
        {},
        "out = 3.0",
        {"out": dace.Memlet("g[i]")},
        dtypes.ScheduleType.GPU_Device,
        external_edges=True,
    )

    result = plan_move("lib-into", [(barrier, wait), (later, None)])

    assert isinstance(result, str) and "device work the barrier would come before" in result, result


def test_a_barrier_inside_a_kernel_never_moves():
    sdfg, barrier, (kernel, host, wait) = device_program(SyncScope.BLOCK)

    result = plan_move("lib-before", [(barrier, wait), (host, None)])

    assert isinstance(result, str) and "block barrier" in result, result


if __name__ == "__main__":
    import tempfile

    def scratch() -> pathlib.Path:
        return pathlib.Path(tempfile.mkdtemp())

    test_a_copy_between_arrays_is_a_lib_copy_row_after_canonicalization(scratch())
    test_a_lib_copy_row_is_a_copy_edge_again_once_scopes_are_made_so_no_kernel_copies(scratch())
    test_a_loop_invariant_copy_hoists_out_of_its_loop_and_runs_once(scratch())
    test_a_loop_whose_trip_count_is_not_proven_positive_keeps_its_copy(scratch())
    test_a_copy_of_an_array_the_loop_writes_is_not_hoisted(scratch())
    test_a_copy_read_in_its_loop_is_not_sunk_after_it(scratch())
    test_a_copy_between_two_loops_moves_ahead_of_them_and_leaves_them_adjacent(scratch())
    test_a_copy_of_what_a_loop_computes_is_not_moved_ahead_of_that_loop(scratch())
    test_a_copy_never_moves_into_a_block(scratch())
    test_a_copy_under_a_condition_does_not_leave_its_branch(scratch())
    test_a_call_named_with_its_own_state_is_refused(scratch())
    test_a_call_with_side_effects_never_moves()
    test_a_copy_joins_an_earlier_state_and_reads_what_that_state_writes(scratch())
    test_a_call_joins_only_a_state(scratch())
    test_a_call_does_not_join_a_state_that_accumulates_into_what_it_touches()
    test_a_call_does_not_join_a_state_with_two_unordered_writes_of_what_it_reads()
    test_a_host_barrier_moves_ahead_of_host_work_that_touches_no_device_data()
    test_a_host_barrier_never_crosses_device_work()
    test_a_host_barrier_joins_a_state_after_its_device_work()
    test_a_host_barrier_never_joins_ahead_of_device_work()
    test_a_barrier_inside_a_kernel_never_moves()
