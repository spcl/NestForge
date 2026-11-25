# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Communication rows move under their own rules (``docs/design/comm-nodes.md``, section 3): a fixed row stays (R4),
a row keeps its region (R2) and a state of its own (no ``lib-into``), a row that starts communication never overtakes
another (R1), and a wait moves only where no block touches what is in flight (R3). Nothing here runs MPI: each
answer is the plan, made on the program and checked before anything changes."""

import dace
from comm_rank_programs import halo_overlap
from dace.libraries.comm.nodes import Barrier, Irecv, Isend, Waitall
from dace.sdfg import nodes

from tactics.ir.comm import CommClass, comm_class, in_flight
from tactics.session import Session
from tactics.stages.lib_motion import accessed, plan_lib_motion

N = 16


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


def halo() -> tuple[dace.SDFG, dict[str, tuple[nodes.Node, dace.SDFGState]], dict[str, dace.SDFGState]]:
    """``send``: Isend of A; ``recv``: Irecv into ghost; ``compute``: B from A; ``wait``: Waitall of both;
    ``use``: C from ghost; ``barrier``: a Barrier at the end. One state each, in that order."""
    sdfg = dace.SDFG("halo")
    for name in ("A", "B", "C", "ghost"):
        sdfg.add_array(name, [N], dace.float64)
    for name in ("peer", "tag"):
        sdfg.add_array(name, [1], dace.int32)
    sdfg.add_array("req", [2], dace.dtypes.opaque("MPI_Request"), transient=True)
    labels = ("send", "recv", "compute", "wait", "use", "barrier")
    states: dict[str, dace.SDFGState] = {}
    previous = None
    for label in labels:
        states[label] = sdfg.add_state_after(previous, label) if previous is not None else sdfg.add_state(label)
        previous = states[label]
    rows: dict[str, tuple[nodes.Node, dace.SDFGState]] = {}

    send, state = Isend("send_A"), states["send"]
    state.add_node(send)
    state.add_edge(state.add_read("A"), None, send, "_buffer", dace.Memlet(f"A[0:{N}]"))
    scalar_input(state, send, "_dest", "peer")
    scalar_input(state, send, "_tag", "tag")
    state.add_edge(send, "_request", state.add_write("req"), None, dace.Memlet("req[0]"))
    rows["send"] = (send, state)

    recv, state = Irecv("recv_ghost"), states["recv"]
    state.add_node(recv)
    scalar_input(state, recv, "_src", "peer")
    scalar_input(state, recv, "_tag", "tag")
    state.add_edge(recv, "_buffer", state.add_write("ghost"), None, dace.Memlet(f"ghost[0:{N}]"))
    state.add_edge(recv, "_request", state.add_write("req"), None, dace.Memlet("req[1]"))
    rows["recv"] = (recv, state)

    doall(states["compute"], "A", "B")

    wait, state = Waitall("wait_all"), states["wait"]
    state.add_node(wait)
    state.add_edge(state.add_read("req"), None, wait, "_request", dace.Memlet("req[0:2]"))
    rows["wait"] = (wait, state)

    doall(states["use"], "ghost", "C")

    barrier = Barrier("barrier")
    states["barrier"].add_node(barrier)
    rows["barrier"] = (barrier, states["barrier"])
    return sdfg, rows, states


def test_every_row_of_the_halo_has_its_class() -> None:
    _, rows, _ = halo()

    classes = {label: comm_class(node) for label, (node, _) in rows.items()}

    assert classes == {
        "send": CommClass.START,
        "recv": CommClass.START,
        "wait": CommClass.WAIT,
        "barrier": CommClass.FIXED,
    }


def test_a_barrier_stays() -> None:
    _, rows, states = halo()

    refused = plan_lib_motion(rows["barrier"], (states["use"], None), "before")

    assert isinstance(refused, str) and "it stays" in refused


def test_a_receive_does_not_overtake_the_send_that_starts_before_it() -> None:
    _, rows, states = halo()

    refused = plan_lib_motion(rows["recv"], (states["send"], None), "before")

    assert isinstance(refused, str) and "would overtake send_A" in refused


def test_a_wait_moves_earlier_past_compute_that_touches_nothing_in_flight() -> None:
    _, rows, states = halo()

    move = plan_lib_motion(rows["wait"], (states["compute"], None), "before")

    assert callable(move), move


def test_a_wait_does_not_move_past_a_reader_of_what_it_receives() -> None:
    _, rows, states = halo()

    refused = plan_lib_motion(rows["wait"], (states["use"], None), "after")

    assert isinstance(refused, str) and "ghost" in refused


def test_a_communication_row_keeps_a_state_of_its_own() -> None:
    _, rows, states = halo()

    refused = plan_lib_motion(rows["wait"], (states["compute"], None), "into")

    assert isinstance(refused, str) and "a state of its own" in refused


def test_a_communication_row_in_a_loop_stays_in_the_loop() -> None:
    sdfg = dace.SDFG("halo_in_a_loop")
    sdfg.add_array("A", [N], dace.float64)
    for name in ("peer", "tag"):
        sdfg.add_array(name, [1], dace.int32)
    sdfg.add_array("req", [1], dace.dtypes.opaque("MPI_Request"), transient=True)
    before = sdfg.add_state("before", is_start_block=True)
    loop = dace.sdfg.state.LoopRegion("steps", "t < 4", "t", "t = 0", "t = t + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(before, loop, dace.InterstateEdge())
    body = loop.add_state("body", is_start_block=True)
    send = Isend("send_A")
    body.add_node(send)
    body.add_edge(body.add_read("A"), None, send, "_buffer", dace.Memlet(f"A[0:{N}]"))
    scalar_input(body, send, "_dest", "peer")
    scalar_input(body, send, "_tag", "tag")
    body.add_edge(send, "_request", body.add_write("req"), None, dace.Memlet("req[0]"))

    refused = plan_lib_motion((send, body), (before, None), "after")

    assert isinstance(refused, str) and "how often it communicates" in refused


def frontend_halo() -> tuple[dace.SDFG, tuple[nodes.Node, dace.SDFGState], list[dace.SDFGState]]:
    """``halo_overlap`` as the frontend builds it, canonical: its ``Waitall`` row and the states after it that read
    a ghost cell. The frontend binds every buffer and request through a view."""
    session = Session(halo_overlap.to_sdfg(simplify=True))
    session.canonicalize()
    (wait,) = [(n, s) for n, s in session.sdfg.all_nodes_recursive() if isinstance(n, Waitall)]
    order = list(session.sdfg.bfs_nodes(wait[1]))
    readers = [s for s in order[1:] if isinstance(s, dace.SDFGState) and "ghost" in accessed(s)[0]]
    return session.sdfg, wait, readers


def test_a_wait_completes_the_starts_that_write_its_requests_through_views() -> None:
    sdfg, (wait, state), _ = frontend_halo()

    sent, received = in_flight(sdfg, state, wait)

    assert "a" in sent and received == frozenset({"ghost"})


def test_a_wait_does_not_move_past_a_reader_of_a_ghost_cell_received_through_a_view() -> None:
    _, wait, readers = frontend_halo()
    assert readers, "a state after the wait reads ghost"

    refused = plan_lib_motion(wait, (readers[0], None), "after")

    assert isinstance(refused, str) and "ghost" in refused


def test_a_wait_completes_starts_whose_requests_an_assignment_forwards() -> None:
    """Before canonicalization the frontend stores each request through a tasklet ``__out = __inp`` into the array
    the wait reads; a wait that found no start there would let R3 move it past a reader of a ghost cell."""
    sdfg = halo_overlap.to_sdfg(simplify=True)
    (wait, state) = next((n, s) for n, s in sdfg.all_nodes_recursive() if isinstance(n, Waitall))

    sent, received = in_flight(sdfg, state, wait)

    assert "a" in sent and received == frozenset({"ghost"})
