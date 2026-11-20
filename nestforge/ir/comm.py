# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Communication rows (``docs/design/comm-nodes.md``, sections 1 and 3): the class of a communication node, the
buffers an operation has in flight, and why a move of one would change what the program communicates.

Every rule reads the class, never the node type. A ``QUERY`` moves like any library call; the others move under rules
R1 to R4 (R5, views, is the rule of every library call)."""

import enum

import dace
from dace.libraries.comm.nodes import (
    Abort,
    Barrier,
    CommF2c,
    CommFree,
    CommRank,
    CommSize,
    CommSplit,
    Dummy,
    Irecv,
    Isend,
    Recv,
    Send,
    Wait,
    Waitall,
)
from dace.libraries.comm.nodes.node import CommNode, NonblockingCommNode
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowBlock, SDFGState

from nestforge.records import Refusal, Refused

#: The connector through which an operation writes, and a wait reads, its request.
REQUEST = "_request"


class CommClass(enum.Enum):
    """How a communication row may move (design, section 1)."""

    QUERY = "query"  # reads the communicator, sends nothing
    START = "start"  # starts an operation and writes a request
    WAIT = "wait"  # completes the operations whose requests it reads
    BLOCKING = "blocking"  # starts and completes in one call
    FIXED = "fixed"  # changes the communicator, synchronizes every rank, or holds state of the MPI runtime


QUERIES = (CommRank, CommSize, CommF2c)
WAITS = (Wait, Waitall)
FIXED_NODES = (Barrier, Abort, CommSplit, CommFree, Dummy)
POINT_TO_POINT_STARTS = (Isend, Irecv)
POINT_TO_POINT_BLOCKING = (Send, Recv)


def comm_class(node: nodes.Node) -> CommClass | None:
    """The class of ``node``; ``None`` when it is no communication node."""
    if not isinstance(node, CommNode):
        return None
    if isinstance(node, QUERIES):
        return CommClass.QUERY
    if isinstance(node, WAITS):
        return CommClass.WAIT
    if isinstance(node, FIXED_NODES):
        return CommClass.FIXED
    if isinstance(node, POINT_TO_POINT_STARTS) or (isinstance(node, NonblockingCommNode) and node.nonblocking):
        return CommClass.START
    return CommClass.BLOCKING


#: The classes R1 keeps in order: the rows that start communication.
STARTING = frozenset({CommClass.START, CommClass.BLOCKING})


def buffers(state: SDFGState, node: nodes.Node) -> tuple[frozenset[str], frozenset[str]]:
    """What ``node`` sends and receives: the containers it reads and writes, its request left out."""
    sent = frozenset(str(e.data.data) for e in state.in_edges(node) if not e.data.is_empty() and e.dst_conn != REQUEST)
    received = frozenset(
        str(e.data.data) for e in state.out_edges(node) if not e.data.is_empty() and e.src_conn != REQUEST
    )
    return sent, received


def requests_of(state: SDFGState, node: nodes.Node, written: bool) -> frozenset[str]:
    """The request containers ``node`` writes (a start) or reads (a wait)."""
    edges = state.out_edges(node) if written else state.in_edges(node)
    return frozenset(
        str(e.data.data) for e in edges if (e.src_conn if written else e.dst_conn) == REQUEST and not e.data.is_empty()
    )


def starts_of(sdfg: dace.SDFG, state: SDFGState, wait: nodes.Node) -> list[tuple[SDFGState, nodes.Node]]:
    """The ``START`` rows whose requests ``wait`` reads: every start of the program writing one of them. Without the
    control flow between them this may hold too many rows, which keeps more buffers in flight and is safe."""
    requests = requests_of(state, wait, written=False)
    return [
        (other, node)
        for other in sdfg.states()
        for node in other.nodes()
        if comm_class(node) is CommClass.START and requests_of(other, node, written=True) & requests
    ]


def in_flight(sdfg: dace.SDFG, state: SDFGState, wait: nodes.Node) -> tuple[frozenset[str], frozenset[str]]:
    """What the starts ``wait`` completes send and receive: a wait reads the one and writes the other (R3)."""
    sent: set[str] = set()
    received: set[str] = set()
    for other, start in starts_of(sdfg, state, wait):
        out, into = buffers(other, start)
        sent |= out
        received |= into
    return frozenset(sent), frozenset(received)


def starting_rows(block: ControlFlowBlock) -> list[str]:
    """The labels of the rows in ``block`` that start communication (R1), at any depth."""
    return [n.label for n, _ in block.all_nodes_recursive() if comm_class(n) in STARTING]


def motion_refusal(node: nodes.Node, state: SDFGState, target: ControlFlowBlock, place: str) -> Refused | None:
    """Why moving the communication row ``node`` next to ``target`` (``before``, ``after``, ``into``) changes what the
    program communicates, before the crossed blocks are known: R4 (a fixed row stays), R2 (a row keeps its region) and
    ``lib-into`` (a row keeps a state of its own). ``None`` for a ``QUERY``, which moves like any call."""
    kind = comm_class(node)
    if kind in (None, CommClass.QUERY):
        return None
    if kind is CommClass.FIXED:
        return Refused(
            f"{node.label} ({type(node).__name__}) synchronizes every rank or changes the communicator; it stays.",
            Refusal.EFFECTS,
        )
    if place == "into":
        return Refused(f"{node.label} communicates: a communication row keeps a state of its own.", Refusal.EFFECTS)
    if target.parent_graph is not state.parent_graph:
        return Refused(
            f"{node.label} runs once per pass of {state.parent_graph.label}; moving it to another region changes how "
            "often it communicates.",
            Refusal.EFFECTS,
        )
    return None


def crossing_refusal(node: nodes.Node, block: ControlFlowBlock) -> Refused | None:
    """R1: a row that starts communication never crosses another one."""
    if comm_class(node) not in STARTING:
        return None
    others = starting_rows(block)
    if not others:
        return None
    return Refused(
        f"{node.label} would overtake {others[0]}; operations that start communication keep their order.",
        Refusal.EFFECTS,
    )
