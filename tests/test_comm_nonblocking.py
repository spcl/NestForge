# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Nonblocking communication patterns (``docs/design/comm-nodes.md``, sections 1 to 4), as the frontend builds them
and canonicalization leaves them: one ``Wait`` per request, one ``Waitall`` over every request or over a slice of them, two exchanges whose waits complete in the opposite order
to their starts, a halo exchange every time step, exchanges pre-posted one step ahead, a row-split Jacobi solver, and
three programs that break the rules of a request in flight. Each wait must complete exactly the starts whose requests
it reads, a wait moves only where nothing touches what is in flight, every move the session lists for a row applies,
and the in-flight check (I2) names a program that writes a sent buffer or reads a received one before its wait.
Nothing here runs MPI: ``tests/test_comm_ranks.py`` runs the same fixtures on ranks."""

import dace
import pytest
from comm_programs import overwrite_in_flight, read_before_wait, read_last_step_before_wait
from comm_rank_programs import (
    halo_split_waitall,
    halo_steps,
    halo_waitall,
    halo_waits,
    jacobi_rows,
    pipelined_rows,
    two_exchanges,
)
from dace.frontend.python.parser import DaceProgram
from dace.libraries.comm.nodes.node import CommNode
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion

from tactics.ir.comm import (
    CommClass,
    Skeleton,
    comm_class,
    in_flight,
    isolation_refusal,
    paths_between,
    skeleton,
    starts_of,
    written_requests,
)
from tactics.records import MoveStatus
from tactics.session import Session
from tactics.stages.lib_motion import LIB_AFTER, LIB_BEFORE, LIB_INTO, accessed, in_flight_refusal, plan_lib_motion

#: The nonblocking fixtures that follow the rules, by name.
FIXTURES: dict[str, DaceProgram] = {
    "halo_waits": halo_waits,
    "halo_waitall": halo_waitall,
    "halo_split_waitall": halo_split_waitall,
    "two_exchanges": two_exchanges,
    "halo_steps": halo_steps,
    "pipelined_rows": pipelined_rows,
    "jacobi_rows": jacobi_rows,
}


def canonical(program: DaceProgram) -> Session:
    session = Session(program.to_sdfg(simplify=True))
    session.canonicalize()
    return session


def program_order(sdfg: dace.SDFG) -> list[dace.SDFGState]:
    """The states of ``sdfg``'s top level, in the order they run."""
    return [b for b in sdfg.bfs_nodes(sdfg.start_block) if isinstance(b, dace.SDFGState)]


def waits_in_order(sdfg: dace.SDFG) -> list[tuple[nodes.Node, dace.SDFGState]]:
    """The ``WAIT`` rows of ``sdfg``'s top level, in the order they run."""
    return [(n, s) for s in program_order(sdfg) for n in s.nodes() if comm_class(n) is CommClass.WAIT]


def holds_row(state: dace.SDFGState) -> bool:
    return any(comm_class(n) is not None for n in state.nodes())


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_every_nonblocking_fixture_keeps_the_invariants(name: str) -> None:
    sdfg = canonical(FIXTURES[name]).sdfg

    assert skeleton(sdfg) is not None, f"{name} communicates"
    assert isolation_refusal(sdfg) is None
    assert in_flight_refusal(sdfg) is None


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_every_listed_move_of_a_nonblocking_row_applies(name: str) -> None:
    """A move the session lists for a communication row keeps the invariants, so applying it never answers illegal:
    a listing that offered a move the invariants then refused would cost the agent a turn for nothing."""
    session = canonical(FIXTURES[name])
    rows = session.row_index()
    moves = [
        move
        for kind in (LIB_BEFORE, LIB_AFTER)
        for move in session.list_moves(kind)
        if isinstance(rows[move.labels[0]][0], CommNode)
    ]
    assert moves, f"{name}: a communication row has somewhere to go"

    for move in moves:
        result = session.apply_move(move.kind, list(move.labels))
        assert result.status is MoveStatus.APPLIED, (move, result.reason)
        assert session.undo().status is MoveStatus.APPLIED


def test_each_wait_of_one_request_completes_only_its_own_start() -> None:
    sdfg = canonical(halo_waits).sdfg
    waits = waits_in_order(sdfg)
    assert len(waits) == 4, "one Wait per request"

    completed = [starts_of(sdfg, state, wait) for wait, state in waits]
    flights = [in_flight(sdfg, state, wait) for wait, state in waits]

    assert all(len(starts) == 1 for starts in completed), completed
    assert len({id(starts[0][1]) for starts in completed}) == 4, "no two waits complete the same start"
    assert flights[:2] == [(frozenset(), frozenset({"ghost"}))] * 2, "the receives complete first"
    assert flights[2:] == [(frozenset({"a"}), frozenset())] * 2, "the sends complete last"


def test_the_wait_of_a_receive_does_not_move_past_the_reader_of_its_ghost_cell() -> None:
    sdfg = canonical(halo_waits).sdfg
    (wait, state), *_ = waits_in_order(sdfg)
    order = program_order(sdfg)
    reader = next(s for s in order[order.index(state) + 1 :] if "ghost" in accessed(s)[0])

    refused = plan_lib_motion((wait, state), (reader, None), "after")

    assert isinstance(refused, str) and "ghost" in refused, refused


def test_the_wait_of_a_send_moves_earlier_past_compute_that_only_reads_what_it_sends() -> None:
    sdfg = canonical(halo_waits).sdfg
    wait, state = waits_in_order(sdfg)[2]
    order = program_order(sdfg)
    compute = next(s for s in reversed(order[: order.index(state)]) if not holds_row(s))
    reads, writes = accessed(compute)
    assert "a" in reads and "a" not in writes, "the compute reads the sent buffer and does not write it"

    move = plan_lib_motion((wait, state), (compute, None), "before")

    assert callable(move), move


def test_one_waitall_over_every_request_completes_all_four_starts() -> None:
    sdfg = canonical(halo_waitall).sdfg
    ((wait, state),) = waits_in_order(sdfg)

    assert len(starts_of(sdfg, state, wait)) == 4
    assert in_flight(sdfg, state, wait) == (frozenset({"a"}), frozenset({"ghost"}))


def test_the_waitall_of_every_request_does_not_move_past_the_reader_of_the_ghost_cells() -> None:
    sdfg = canonical(halo_waitall).sdfg
    ((wait, state),) = waits_in_order(sdfg)
    order = program_order(sdfg)
    reader = next(s for s in order[order.index(state) + 1 :] if "ghost" in accessed(s)[0])

    refused = plan_lib_motion((wait, state), (reader, None), "after")

    assert isinstance(refused, str) and "ghost" in refused, refused


def test_a_waitall_over_a_slice_completes_only_the_starts_of_its_slice() -> None:
    sdfg = canonical(halo_split_waitall).sdfg
    waits = waits_in_order(sdfg)
    assert len(waits) == 2

    completed = [starts_of(sdfg, state, wait) for wait, state in waits]
    flights = [in_flight(sdfg, state, wait) for wait, state in waits]

    assert [len(starts) for starts in completed] == [2, 2], completed
    assert not {id(s) for _, s in completed[0]} & {id(s) for _, s in completed[1]}, "no start completes twice"
    assert flights == [(frozenset(), frozenset({"ghost"})), (frozenset({"a"}), frozenset())]


def test_the_waitall_of_the_sends_moves_earlier_past_compute_that_only_reads_what_they_send() -> None:
    sdfg = canonical(halo_split_waitall).sdfg
    wait, state = waits_in_order(sdfg)[1]
    order = program_order(sdfg)
    compute = next(s for s in reversed(order[: order.index(state)]) if not holds_row(s))
    reads, writes = accessed(compute)
    assert "a" in reads and "a" not in writes, "the compute reads the sent buffer and does not write it"

    move = plan_lib_motion((wait, state), (compute, None), "before")

    assert callable(move), move


def test_waits_in_the_opposite_order_complete_the_exchange_their_requests_belong_to() -> None:
    sdfg = canonical(two_exchanges).sdfg
    waits = waits_in_order(sdfg)
    assert len(waits) == 2

    flights = [in_flight(sdfg, state, wait) for wait, state in waits]

    assert flights == [
        (frozenset({"x"}), frozenset({"ghost_x"})),
        (frozenset({"a"}), frozenset({"ghost_a"})),
    ]


def test_the_later_wait_moves_past_the_use_of_the_earlier_exchange_but_not_past_its_own() -> None:
    sdfg = canonical(two_exchanges).sdfg
    (first, first_state), (second, second_state) = waits_in_order(sdfg)
    order = program_order(sdfg)
    between = [s for s in order[order.index(first_state) + 1 : order.index(second_state)] if not holds_row(s)]
    use_x = next(s for s in between if "ghost_x" in accessed(s)[0])
    use_a = next(s for s in order[order.index(second_state) + 1 :] if "ghost_a" in accessed(s)[0])

    earlier = plan_lib_motion((second, second_state), (use_x, None), "before")
    past_own_use = plan_lib_motion((second, second_state), (use_a, None), "after")
    first_past_its_use = plan_lib_motion((first, first_state), (use_x, None), "after")

    assert callable(earlier), earlier
    assert isinstance(past_own_use, str) and "ghost_a" in past_own_use, past_own_use
    assert isinstance(first_past_its_use, str) and "ghost_x" in first_past_its_use, first_past_its_use


def test_a_halo_exchanged_every_step_keeps_its_starts_and_its_wait_in_the_loop() -> None:
    sdfg = canonical(halo_steps).sdfg

    found = skeleton(sdfg)

    assert found is not None
    (loop,) = [row for row in found.rows if isinstance(row, Skeleton)]
    assert loop.kind == "for" and not found.waits, "nothing communicates outside the time loop"
    assert sorted(str(row).split()[0] for row in loop.rows) == ["Recv", "Recv", "Send", "Send"]
    assert len(loop.waits) == 4, "the Waitall completes all four starts, each step"


def test_the_wait_of_a_step_does_not_move_past_the_stencil_that_reads_the_ghost_cells() -> None:
    sdfg = canonical(halo_steps).sdfg
    (wait, state) = next((n, s) for n, s in sdfg.all_nodes_recursive() if comm_class(n) is CommClass.WAIT)
    body = state.parent_graph
    after = [b for b in body.bfs_nodes(state) if isinstance(b, dace.SDFGState) and b is not state]
    stencil = next(s for s in after if "ap" in accessed(s)[0])

    refused = plan_lib_motion((wait, state), (stencil, None), "after")

    assert isinstance(refused, str) and "ap" in refused, refused


def test_a_pre_posted_wait_completes_the_starts_before_it_and_keeps_only_the_received_rows_in_flight() -> None:
    """The wait at the head of a step completes what the prologue or the previous step started: control flow is not
    followed, so it counts every start of its requests, and only ``b`` is received."""
    sdfg = canonical(pipelined_rows).sdfg
    waits = [(n, s) for n, s in sdfg.all_nodes_recursive() if comm_class(n) is CommClass.WAIT]
    assert len(waits) == 2, "one wait in the loop, one after it"

    for wait, state in waits:
        starts = starts_of(sdfg, state, wait)
        assert sorted(type(start).__name__ for _, start in starts) == ["Irecv", "Irecv", "Isend", "Isend"]
        assert in_flight(sdfg, state, wait) == (frozenset({"a"}), frozenset({"b"}))


@pytest.mark.parametrize(
    ("program", "detail"),
    [
        (overwrite_in_flight, "writes a"),
        (read_before_wait, "reads ghost"),
        (read_last_step_before_wait, "reads b"),
    ],
)
def test_a_program_that_touches_a_buffer_in_flight_is_named(program: DaceProgram, detail: str) -> None:
    """I2 on a program that breaks it from the start: the compute between the starts and the wait writes the sent
    buffer, or reads the received one, so no move may keep or create that order."""
    sdfg = canonical(program).sdfg

    refused = in_flight_refusal(sdfg)

    assert refused is not None and "is in flight" in refused and detail in refused, refused


def test_a_path_around_the_loop_compares_what_the_last_step_received_by_container() -> None:
    """From the receive of row ``i`` to the wait of the next step, the path steps the loop: the read of ``b[i - 1]``
    there is the row in flight, though the two subsets look disjoint, so the path marks it as stepped."""
    sdfg = canonical(read_last_step_before_wait).sdfg
    in_loop = [
        (n, s)
        for n, s in sdfg.all_nodes_recursive()
        if comm_class(n) is not None and isinstance(s.parent_graph, LoopRegion)
    ]
    receive, receive_state = next((n, s) for n, s in in_loop if type(n).__name__ == "Irecv")
    _, wait_state = next((n, s) for n, s in in_loop if comm_class(n) is CommClass.WAIT)

    between = paths_between(sdfg, receive_state, wait_state, written_requests(sdfg, receive_state, receive))

    assert between is not None, "the receive of a step reaches the wait of the next"
    readers = [stepped for state, stepped in between if "b" in accessed(state)[0]]
    assert readers == [True], between


def test_a_request_stored_in_the_state_of_its_wait_is_complete_there() -> None:
    """The frontend stores each request in ``req`` through a copy; ``lib-into`` may join the last copy to the state of
    the ``Waitall``, which then stores the request before it reads it. The wait still completes all four starts, and
    the boundary update after it is not in flight."""
    session = canonical(jacobi_rows)
    rows = session.row_index()
    (wait_state,) = {
        id(s): s for n, s in session.sdfg.all_nodes_recursive() if comm_class(n) is CommClass.WAIT
    }.values()
    joins = [
        move
        for move in session.list_moves(LIB_INTO)
        if rows[move.labels[-1]][0] is wait_state and not isinstance(rows[move.labels[0]][0], CommNode)
    ]
    assert joins, "a request copy may join the state of the wait"

    for move in joins:
        result = session.apply_move(move.kind, list(move.labels))
        assert result.status is MoveStatus.APPLIED, (move, result.reason)
        sdfg = session.sdfg
        ((wait, state),) = [(n, s) for n, s in sdfg.all_nodes_recursive() if comm_class(n) is CommClass.WAIT]
        assert len(starts_of(sdfg, state, wait)) == 4
        assert in_flight_refusal(sdfg) is None
        assert session.undo().status is MoveStatus.APPLIED
