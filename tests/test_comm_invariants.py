# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The four invariants of a program that communicates (``docs/design/comm-nodes.md``, section 4), checked after every
kept move: I1 the skeleton stays, I2 nothing between a start and its wait touches what is in flight, I3 each row keeps
a state of its own outside every doall, I4 no kernel holds a row. A move that breaks one is undone and answers
``illegal`` naming the row: without the check, a fusion or a tiling could change what the ranks exchange with no data
edge in DaCe to stop it."""

import dace
import pytest
from comm_programs import PROGRAMS, residual_in_time_loop, ring
from comm_rank_programs import halo_overlap
from dace.frontend.python.parser import DaceProgram
from dace.libraries.comm.nodes import Irecv, Isend, Waitall
from dace.sdfg import nodes

from tactics.ir.comm import isolation_refusal, skeleton, skeleton_difference
from tactics.ir.introspect import describe_graph
from tactics.records import MoveStatus
from tactics.session import Session
from tactics.stages.lib_motion import SPLIT_BLOCKING, in_flight_refusal

N = 16

FIXTURES: dict[str, DaceProgram] = {**PROGRAMS, "halo_overlap": halo_overlap}


def canonical(program: DaceProgram) -> Session:
    session = Session(program.to_sdfg(simplify=True))
    session.canonicalize()
    return session


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_every_canonical_fixture_keeps_the_invariants(name: str) -> None:
    sdfg = canonical(FIXTURES[name]).sdfg

    assert skeleton(sdfg) is not None, f"{name} communicates"
    assert isolation_refusal(sdfg) is None
    assert in_flight_refusal(sdfg) is None


def test_a_tiling_of_the_loop_around_a_reduction_is_undone() -> None:
    """I1: tiling the time loop puts the Allreduce in two loops where it was in one."""
    session = canonical(residual_in_time_loop)
    tree = describe_graph(session.sdfg)

    result = session.apply_move("tile-for", ["for0"], tile_sizes=(4,))

    assert result.status is MoveStatus.ILLEGAL, result
    assert "tile-for would change how lib_comm0 runs" in str(result.reason), result.reason
    assert describe_graph(session.sdfg) == tree, "the program is the one before the move"


def test_a_row_that_leaves_its_loop_changes_the_skeleton() -> None:
    sdfg = canonical(residual_in_time_loop).sdfg
    before = skeleton(sdfg)
    (loop,) = [b for b in sdfg.nodes() if isinstance(b, dace.sdfg.state.LoopRegion)]
    (state,) = [s for s in loop.nodes() if any(isinstance(n, nodes.LibraryNode) for n in s.nodes())]
    loop.remove_node(state)
    sdfg.add_node(state)
    sdfg.add_edge(loop, state, dace.InterstateEdge())

    assert skeleton_difference(before, skeleton(sdfg)) == (
        "change how lib_comm0 runs: inside top, not top > for 0, T - 1, 1"
    )


def scalar_input(state: dace.SDFGState, node: nodes.LibraryNode, connector: str, name: str) -> None:
    state.add_edge(state.add_read(name), None, node, connector, dace.Memlet(f"{name}[0]"))


def doall(state: dace.SDFGState, read: str, write: str) -> None:
    state.add_mapped_tasklet(
        f"from_{read}_to_{write}",
        {"i": f"0:{N}"},
        {"x": dace.Memlet(f"{read}[i]")},
        "y = x + 1.0",
        {"y": dace.Memlet(f"{write}[i]")},
        external_edges=True,
    )


def exchange(between: tuple[tuple[str, str], ...]) -> dace.SDFG:
    """An Isend of A and an Irecv into ghost, one state each, then a state per doall of ``between`` (read, write),
    then the Waitall of both."""
    sdfg = dace.SDFG("exchange")
    for name in ("A", "B", "C", "ghost"):
        sdfg.add_array(name, [N], dace.float64)
    for name in ("peer", "tag"):
        sdfg.add_array(name, [1], dace.int32)
    sdfg.add_array("req", [2], dace.dtypes.opaque("MPI_Request"), transient=True)
    send_state = sdfg.add_state("send")
    send = Isend("send_A")
    send_state.add_node(send)
    send_state.add_edge(send_state.add_read("A"), None, send, "_buffer", dace.Memlet(f"A[0:{N}]"))
    scalar_input(send_state, send, "_dest", "peer")
    scalar_input(send_state, send, "_tag", "tag")
    send_state.add_edge(send, "_request", send_state.add_write("req"), None, dace.Memlet("req[0]"))
    recv_state = sdfg.add_state_after(send_state, "recv")
    recv = Irecv("recv_ghost")
    recv_state.add_node(recv)
    scalar_input(recv_state, recv, "_src", "peer")
    scalar_input(recv_state, recv, "_tag", "tag")
    recv_state.add_edge(recv, "_buffer", recv_state.add_write("ghost"), None, dace.Memlet(f"ghost[0:{N}]"))
    recv_state.add_edge(recv, "_request", recv_state.add_write("req"), None, dace.Memlet("req[1]"))
    previous = recv_state
    for index, (read, write) in enumerate(between):
        previous = sdfg.add_state_after(previous, f"between{index}")
        doall(previous, read, write)
    wait_state = sdfg.add_state_after(previous, "wait")
    wait = Waitall("wait_all")
    wait_state.add_node(wait)
    wait_state.add_edge(wait_state.add_read("req"), None, wait, "_request", dace.Memlet("req[0:2]"))
    return sdfg


def test_compute_on_other_arrays_and_reads_of_the_sent_buffer_may_overlap_the_exchange() -> None:
    assert in_flight_refusal(exchange((("A", "B"), ("B", "C")))) is None


def test_a_read_of_a_received_buffer_before_the_wait_breaks_i2() -> None:
    refused = in_flight_refusal(exchange((("ghost", "C"),)))

    assert refused is not None and refused.startswith("let between0 run while recv_ghost is in flight"), refused
    assert "ghost" in refused


def test_a_write_of_a_sent_buffer_before_the_wait_breaks_i2() -> None:
    refused = in_flight_refusal(exchange((("B", "A"),)))

    assert refused is not None and "send_A" in refused and "A" in refused, refused


def test_a_row_that_shares_its_state_with_a_doall_breaks_i3() -> None:
    sdfg = exchange(())
    doall(next(s for s in sdfg.states() if s.label == "send"), "B", "C")

    assert isolation_refusal(sdfg) == "put send_A into one state with from_B_to_C_map"


def test_a_wait_counts_its_own_starts_out_of_what_runs_between() -> None:
    """Two receives of one wait both write ghost: neither breaks I2 for the other."""
    sdfg = exchange(())
    second = sdfg.add_state_before(next(s for s in sdfg.states() if s.label == "wait"), "recv2")
    recv = Irecv("recv_ghost_again")
    second.add_node(recv)
    scalar_input(second, recv, "_src", "peer")
    scalar_input(second, recv, "_tag", "tag")
    second.add_edge(recv, "_buffer", second.add_write("ghost"), None, dace.Memlet(f"ghost[0:{N}]"))
    second.add_edge(recv, "_request", second.add_write("req"), None, dace.Memlet("req[1]"))

    assert in_flight_refusal(sdfg) is None


@pytest.mark.parametrize("program", [ring, residual_in_time_loop])
def test_splitting_every_blocking_row_keeps_the_skeleton(program: DaceProgram) -> None:
    """A start and its wait, next to each other where the blocking row was, communicate as that row did (I1):
    ``split-blocking`` is always legal, and a skeleton that saw a new row in the start refused it."""
    session = canonical(program)
    before = skeleton(session.sdfg)

    while splits := session.list_moves(SPLIT_BLOCKING):
        result = session.apply_move(SPLIT_BLOCKING, list(splits[0].labels))
        assert result.status is MoveStatus.APPLIED, result

    assert session.history, "a blocking row was split"
    assert skeleton_difference(before, skeleton(session.sdfg)) is None
