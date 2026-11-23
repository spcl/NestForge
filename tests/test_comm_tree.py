# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Communication rows in the structure tree (``docs/design/comm-nodes.md``, section 2): a row prints ``COMM``, the
buffers it sends and receives, the request a start writes, the starts a wait completes, the peer and tag of a
point-to-point row, and the operator of a reduction. A collapsed state counts its communication rows apart from its
doalls. An agent reads the exchange from these fields; without them a halo exchange is five opaque library calls."""

from comm_programs import halo_exchange, residual_in_time_loop, ring
from dace.frontend.python.parser import DaceProgram

from tactics.ir.introspect import describe_graph
from tactics.session import Session


def rows(program: DaceProgram) -> dict[str, str]:
    """The communication rows of ``program``'s canonical tree, by label, the tree drawing stripped."""
    session = Session(program.to_sdfg(simplify=True))
    session.canonicalize()
    tree = describe_graph(session.sdfg)
    found: dict[str, str] = {}
    for line in tree.splitlines():
        text = line.lstrip("|`- ")
        if "  COMM " in text:
            found[text.split()[0]] = text
    return found


def test_the_halo_rows_name_their_buffers_requests_peers_and_tags() -> None:
    found = rows(halo_exchange)
    assert found == {
        "lib_comm0": "lib_comm0  COMM Isend [auto]  reads=['a']  request=req  to=left  tag=0",
        "lib_comm1": "lib_comm1  COMM Isend [auto]  reads=['a']  request=req  to=right  tag=1",
        "lib_comm2": "lib_comm2  COMM Irecv [auto]  writes=['ghost']  request=req  from=left  tag=1",
        "lib_comm3": "lib_comm3  COMM Irecv [auto]  writes=['ghost']  request=req  from=right  tag=0",
        "lib_comm4": "lib_comm4  COMM Waitall [auto]  completes=[lib_comm0, lib_comm1, lib_comm2, lib_comm3]",
    }, "\n".join(found.values())


def test_a_reduction_prints_its_operator_and_blocking_rows_their_peers() -> None:
    assert rows(residual_in_time_loop) == {
        "lib_comm0": "lib_comm0  COMM Allreduce [auto]  reads=['r']  writes=['r']  op=MPI_SUM"
    }
    assert rows(ring) == {
        "lib_comm0": "lib_comm0  COMM Send [auto]  reads=['a']  to=right  tag=0",
        "lib_comm1": "lib_comm1  COMM Recv [auto]  writes=['b']  from=left  tag=0",
    }


def test_a_collapsed_state_counts_communication_rows_apart_from_doalls() -> None:
    session = Session(halo_exchange.to_sdfg(simplify=True))
    session.canonicalize()
    tree = describe_graph(session.sdfg, expanded=())
    assert "comm=1" in tree and "doalls=" not in tree, tree
    assert "COMM" not in tree, "a collapsed state shows no rows"
