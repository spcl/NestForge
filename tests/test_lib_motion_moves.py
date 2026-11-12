# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``lib-before`` / ``lib-after``: a library call whose memlets are its whole effect (a ``lib_copy`` above all, which
every copy between two arrays becomes) moves into a state of its own before or after a block of an enclosing region,
judged by the read and write sets of what it crosses."""

import pathlib
import re

import dace
import numpy as np
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
    sdfg = dace.SDFG("barrier")
    first = sdfg.add_state("first", is_start_block=True)
    second = sdfg.add_state_after(first, "second")
    barrier = Sync("barrier", scope=SyncScope.DEVICE)
    second.add_node(barrier)

    result = plan_move("lib-before", [(barrier, second), (first, None)])

    assert isinstance(result, str) and "side effects" in result, result


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
