# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Communication rows (``docs/design/comm-nodes.md``, sections 1 and 3): the class of a communication node, the
buffers an operation has in flight, and why a move of one would change what the program communicates.

Every rule reads the class, never the node type. A ``QUERY`` moves like any library call; the others move under rules
R1 to R4 (R5, views, is the rule of every library call)."""

import enum
from typing import cast

import dace
from dace import subsets
from dace.libraries.comm.nodes import (
    Abort,
    Allreduce,
    Barrier,
    CommF2c,
    CommFree,
    CommRank,
    CommSize,
    CommSplit,
    Dummy,
    Irecv,
    Isend,
    Reduce,
    ReduceScatter,
    Wait,
    Waitall,
)
from dace.libraries.comm.nodes.node import CommNode, NonblockingCommNode
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import ControlFlowBlock, ControlFlowRegion, SDFGState

from tactics.records import Refusal, Refused

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


def container(state: SDFGState, node: nodes.Node, name: str) -> str:
    """The container ``name`` stands for at ``node``: what a view there views, else ``name`` itself."""
    if isinstance(node, nodes.AccessNode) and isinstance(node.desc(state.sdfg), dace.data.View):
        root = sdutil.get_last_view_node(state, node)
        if root is not None:
            return str(root.data)
    return name


def with_viewed(block: ControlFlowBlock, names: set[str]) -> set[str]:
    """``names`` and the containers that the views among them view in ``block``."""
    states = [block] if isinstance(block, SDFGState) else list(cast(ControlFlowRegion, block).states())
    viewed = {
        container(state, node, node.data) for state in states for node in state.data_nodes() if node.data in names
    }
    return names | viewed


def buffers(state: SDFGState, node: nodes.Node) -> tuple[frozenset[str], frozenset[str]]:
    """What ``node`` sends and receives: the containers it reads and writes, its request left out."""
    sent = frozenset(
        container(state, e.src, str(e.data.data))
        for e in state.in_edges(node)
        if not e.data.is_empty() and e.dst_conn != REQUEST
    )
    received = frozenset(
        container(state, e.dst, str(e.data.data))
        for e in state.out_edges(node)
        if not e.data.is_empty() and e.src_conn != REQUEST
    )
    return sent, received


def requests_of(state: SDFGState, node: nodes.Node, written: bool) -> frozenset[str]:
    """The request containers ``node`` writes (a start) or reads (a wait)."""
    if written:
        edges = [(e.dst, e) for e in state.out_edges(node) if e.src_conn == REQUEST]
    else:
        edges = [(e.src, e) for e in state.in_edges(node) if e.dst_conn == REQUEST]
    return frozenset(container(state, end, str(e.data.data)) for end, e in edges if not e.data.is_empty())


def forwarded(sdfg: dace.SDFG, names: frozenset[str]) -> frozenset[str]:
    """``names`` and every container a code node forwards one of them into, transitively: the frontend writes a
    request into a scalar that a copy, or an assignment tasklet (``__out = __inp``), then stores into the request
    array the wait reads. Any code node that reads one counts, which may add containers and is safe: more buffers
    stay in flight."""
    copies = [
        (state, node)
        for state in sdfg.states()
        for node in state.nodes()
        if isinstance(node, nodes.CodeNode) and comm_class(node) is None
    ]
    found = set(names)
    grew = True
    while grew:
        grew = False
        for state, node in copies:
            sources = {container(state, e.src, str(e.data.data)) for e in state.in_edges(node) if not e.data.is_empty()}
            if not sources & found:
                continue
            targets = {
                container(state, e.dst, str(e.data.data)) for e in state.out_edges(node) if not e.data.is_empty()
            }
            grew = grew or not targets <= found
            found |= targets
    return frozenset(found)


def starts_of(sdfg: dace.SDFG, state: SDFGState, wait: nodes.Node) -> list[tuple[SDFGState, nodes.Node]]:
    """The ``START`` rows whose requests ``wait`` reads, directly or through copies: every start of the program
    writing one of them. Without the control flow between them this may hold too many rows, which keeps more buffers
    in flight and is safe."""
    requests = requests_of(state, wait, written=False)
    return [
        (other, node)
        for other in sdfg.states()
        for node in other.nodes()
        if comm_class(node) is CommClass.START and forwarded(sdfg, requests_of(other, node, written=True)) & requests
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


#: The connectors that name a point-to-point row's peer and tag, and the field each prints as (design, section 2).
PEER_FIELDS = {"_dest": "to", "_src": "from", "_tag": "tag"}

#: The collectives that reduce, and so print their operator.
REDUCTIONS = (Allreduce, Reduce, ReduceScatter)


def literal(sdfg: dace.SDFG, name: str) -> str | None:
    """The constant that the one write of transient ``name`` stores, when that write is a tasklet without inputs
    (the frontend's ``__out = 0`` for a literal tag, in any state once canonicalization has split them), else
    ``None``."""
    desc = sdfg.arrays.get(name)
    if desc is None or not desc.transient:
        return None
    writes = [
        edge
        for state in sdfg.states()
        for node in state.data_nodes()
        if node.data == name
        for edge in state.in_edges(node)
    ]
    if len(writes) != 1 or not isinstance(writes[0].src, nodes.Tasklet) or writes[0].src.in_connectors:
        return None
    _, assigned, value = writes[0].src.code.as_string.partition("=")
    return value.strip() if assigned else None


def operand(state: SDFGState, edge: MultiConnectorEdge) -> str:
    """What a peer or tag connector is passed: a literal, else its container, with the element when the container
    holds more than one."""
    value = literal(state.sdfg, str(edge.data.data))
    if value is not None:
        return value
    name = container(state, edge.src, str(edge.data.data))
    if state.sdfg.arrays[str(edge.data.data)].total_size == 1:
        return name
    return f"{name}[{edge.data.subset}]"


def shown_requests(sdfg: dace.SDFG, state: SDFGState, node: nodes.Node) -> list[str]:
    """The requests a ``START`` row prints: the containers a copy forwards its request into (the frontend writes it
    into a scalar first), else the one it writes."""
    direct = requests_of(state, node, written=True)
    return sorted((forwarded(sdfg, direct) - direct) or direct)


def comm_fields(state: SDFGState, node: nodes.Node) -> list[str]:
    """The fields a communication row adds after its reads and writes (design, section 2): ``request=`` on a start,
    ``completes=`` on a wait, the peer and tag of a point-to-point row, the operator of a reduction. A ``QUERY``
    row has none."""
    kind = comm_class(node)
    if kind in (None, CommClass.QUERY):
        return []
    fields = []
    if kind is CommClass.START:
        fields.append(f"request={','.join(shown_requests(state.sdfg, state, node))}")
    if kind is CommClass.WAIT:
        completed = sorted(str(start.label) for _, start in starts_of(state.sdfg, state, node))
        fields.append(f"completes=[{', '.join(completed)}]")
    peers = {e.dst_conn: e for e in state.in_edges(node) if e.dst_conn in PEER_FIELDS and not e.data.is_empty()}
    fields += [f"{field}={operand(state, peers[conn])}" for conn, field in PEER_FIELDS.items() if conn in peers]
    if isinstance(node, REDUCTIONS):
        fields.append(f"op={node.op}")
    return fields


def comm_line(state: SDFGState, node: nodes.LibraryNode) -> str:
    """A communication row: ``COMM`` in place of ``LIBNODE``, the buffers it sends (``reads``) and receives
    (``writes``) without its request, peer and tag, then :func:`comm_fields`; empty lists are left out."""
    kind = type(node).__name__
    if comm_class(node) is CommClass.QUERY:
        return f"{node.label}  COMM {kind} [{node.implementation or 'auto'}]"
    skipped = {REQUEST, *PEER_FIELDS}
    sent = sorted(
        {
            container(state, e.src, str(e.data.data))
            for e in state.in_edges(node)
            if not e.data.is_empty() and e.dst_conn not in skipped
        }
    )
    received = sorted(
        {
            container(state, e.dst, str(e.data.data))
            for e in state.out_edges(node)
            if not e.data.is_empty() and e.src_conn not in skipped
        }
    )
    parts = [f"{node.label}  COMM {kind} [{node.implementation or 'auto'}]"]
    parts += [f"reads={sent}"] if sent else []
    parts += [f"writes={received}"] if received else []
    return "  ".join(parts + comm_fields(state, node))


#: The parts of a container an access touches; ``None``: all of it (a view, or no subset to tell).
Extent = list[subsets.Subset] | None


def merged(extents: dict[str, Extent], name: str, extent: Extent) -> None:
    """Add ``extent`` of ``name`` to ``extents``; ``None`` once either is the whole container."""
    known = extents.get(name, [])
    extents[name] = None if known is None or extent is None else [*known, *extent]


def end_extent(state: SDFGState, edge: MultiConnectorEdge, end: nodes.Node) -> Extent:
    """The part of the container at ``end`` (an access node) that ``edge`` reads or writes there."""
    if isinstance(end, nodes.AccessNode) and isinstance(end.desc(state.sdfg), dace.data.View):
        return None
    subset = edge.data.get_src_subset(edge, state) if end is edge.src else edge.data.get_dst_subset(edge, state)
    return None if subset is None else [subset]


def row_extents(state: SDFGState, node: nodes.Node, writes: bool) -> dict[str, Extent]:
    """What the row ``node`` writes (``writes``) or reads, per container, as its memlets' subsets."""
    extents: dict[str, Extent] = {}
    for edge in state.out_edges(node) if writes else state.in_edges(node):
        if edge.data.is_empty():
            continue
        end = edge.dst if writes else edge.src
        merged(extents, container(state, end, str(edge.data.data)), end_extent(state, edge, end))
    return extents


def in_flight_extents(sdfg: dace.SDFG, state: SDFGState, wait: nodes.Node, writes: bool) -> dict[str, Extent]:
    """What the starts ``wait`` completes receive (``writes``) or send, per container, their requests left out."""
    extents: dict[str, Extent] = {}
    for other, start in starts_of(sdfg, state, wait):
        requests = requests_of(other, start, written=True)
        for name, extent in row_extents(other, start, writes).items():
            if name not in requests:
                merged(extents, name, extent)
    return extents


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
