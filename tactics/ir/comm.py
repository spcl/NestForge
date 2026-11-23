# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Communication rows (``docs/design/comm-nodes.md``, sections 1 and 3): the class of a communication node, the
buffers an operation has in flight, and why a move of one would change what the program communicates.

Every rule reads the class, never the node type. A ``QUERY`` moves like any library call; the others move under rules
R1 to R4 (R5, views, is the rule of every library call)."""

import enum
from dataclasses import dataclass, field
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
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation.passes.analysis import loop_analysis

from tactics.ir.names import in_order
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
    """What ``node`` sends and receives: the containers it reads and writes, its request, peer and tag left out. MPI
    reads a peer and a tag when the call is made, so only the buffers stay in flight until the wait."""
    sent = frozenset(
        container(state, e.src, str(e.data.data))
        for e in state.in_edges(node)
        if not e.data.is_empty() and e.dst_conn != REQUEST and e.dst_conn not in PEER_FIELDS
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


#: An element range of a request container, or ``None`` where it cannot be told (through a view).
RequestRange = tuple[str, subsets.Subset | None]


def request_range(state: SDFGState, end: nodes.Node, edge: MultiConnectorEdge) -> RequestRange:
    """The container ``edge`` moves a request through at ``end``, and the elements, unless a view hides them."""
    name = str(edge.data.data)
    viewed = isinstance(end, nodes.AccessNode) and isinstance(end.desc(state.sdfg), dace.data.View)
    return container(state, end, name), None if viewed else edge.data.subset


def written_requests(sdfg: dace.SDFG, state: SDFGState, start: nodes.Node) -> list[RequestRange]:
    """Where ``start``'s request ends up: what it writes through ``_request``, and every element a code node or a
    copy then stores it into, transitively, as :func:`forwarded` follows it by container."""
    reached = forwarded(sdfg, requests_of(state, start, written=True))
    ranges = [request_range(state, e.dst, e) for e in state.out_edges(start) if e.src_conn == REQUEST]
    for other in sdfg.states():
        for edge in other.edges():
            if edge.data.is_empty() or not isinstance(edge.dst, nodes.AccessNode) or comm_class(edge.src) is not None:
                continue
            source = edge.src.data if isinstance(edge.src, nodes.AccessNode) else None
            sources = (
                {container(other, edge.src, source)}
                if source is not None
                else {
                    container(other, e.src, str(e.data.data)) for e in other.in_edges(edge.src) if not e.data.is_empty()
                }
            )
            if sources & reached and container(other, edge.dst, str(edge.dst.data)) in reached:
                name, elements = request_range(other, edge.dst, edge)
                if source is not None and edge.data.data == source:
                    elements = edge.data.dst_subset  # a copy names the source; the destination's elements differ
                ranges.append((name, elements))
    return ranges


def same_request(sdfg: dace.SDFG, written: RequestRange, read: RequestRange) -> bool:
    """Whether a start that writes ``written`` may be the one a wait reading ``read`` completes: same container, and
    elements that overlap or cannot be told apart."""
    if written[0] != read[0]:
        return False
    if written[1] is None or read[1] is None:
        return True
    return subsets.intersects(written[1], read[1], sdfg.facts()) is not dace.symbolic.Truth.FALSE


def starts_of(sdfg: dace.SDFG, state: SDFGState, wait: nodes.Node) -> list[tuple[SDFGState, nodes.Node]]:
    """The ``START`` rows whose requests ``wait`` reads, directly or through copies: every start of the program
    that writes an element of a request array the wait reads (``Wait(req[2])`` completes only the start stored in
    ``req[2]``). Where an element cannot be told, through a view or an unknown index, every start that writes the
    array counts, and without the control flow between them the answer may hold too many rows: both keep more
    buffers in flight, which is safe."""
    reads = [
        request_range(state, e.src, e) for e in state.in_edges(wait) if e.dst_conn == REQUEST and not e.data.is_empty()
    ]
    return [
        (other, node)
        for other in sdfg.states()
        for node in other.nodes()
        if comm_class(node) is CommClass.START
        and any(same_request(sdfg, w, r) for w in written_requests(sdfg, other, node) for r in reads)
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


def exchanged(state: SDFGState, node: nodes.Node) -> tuple[list[str], list[str]]:
    """The containers a communication row sends and receives, sorted: its request, peer and tag left out."""
    skipped = {REQUEST, *PEER_FIELDS}
    sent = {
        container(state, e.src, str(e.data.data))
        for e in state.in_edges(node)
        if not e.data.is_empty() and e.dst_conn not in skipped
    }
    received = {
        container(state, e.dst, str(e.data.data))
        for e in state.out_edges(node)
        if not e.data.is_empty() and e.src_conn not in skipped
    }
    return sorted(sent), sorted(received)


def comm_line(state: SDFGState, node: nodes.LibraryNode) -> str:
    """A communication row: ``COMM`` in place of ``LIBNODE``, the buffers it sends (``reads``) and receives
    (``writes``) without its request, peer and tag, then :func:`comm_fields`; empty lists are left out."""
    kind = type(node).__name__
    if comm_class(node) is CommClass.QUERY:
        return f"{node.label}  COMM {kind} [{node.implementation or 'auto'}]"
    sent, received = exchanged(state, node)
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


@dataclass(frozen=True, slots=True)
class Skeleton:
    """The communication skeleton of one region (design, section 4): its kind (``top``, ``for``, ``while``, ``if``,
    ``branch``, ``region``), its iteration range or condition with the loop variable renamed away, its rows other than
    ``WAIT`` in program order (sub-regions holding rows in their place), and the completions of its blocking rows and
    ``WAIT`` rows (:func:`completions`), sorted, since a wait may move past another. A row is its
    :func:`row_signature`, since a move may give it a new label; ``labels`` names the signatures for a reason and is
    not compared. Two programs communicate alike when their skeletons are equal."""

    kind: str
    extent: tuple[str, ...]
    rows: tuple["str | Skeleton", ...]
    waits: tuple[str, ...]
    labels: tuple[tuple[str, str], ...] = field(default=(), compare=False)


#: The blocking operation a point-to-point start begins: ``split-blocking`` turns one into the other in place.
BLOCKING_FORM = {"Isend": "Send", "Irecv": "Recv"}


def row_signature(state: SDFGState, node: nodes.Node) -> str:
    """What a row does, without its label: its operation, what it sends and receives, its peer, tag and operator. A
    start and the blocking row it splits from sign alike (``Isend`` as ``Send``, a collective whether or not it is
    nonblocking), so ``split-blocking`` keeps the skeleton."""
    sent, received = exchanged(state, node)
    fields = [f for f in comm_fields(state, node) if not f.startswith(("request=", "completes="))]
    name = type(node).__name__
    return " ".join([BLOCKING_FORM.get(name, name), f"{sent}->{received}", *fields])


def completions(state: SDFGState, node: nodes.Node, found: CommClass) -> list[str]:
    """The operations ``node`` completes, one skeleton wait each: a blocking row completes itself, a ``WAIT`` the
    starts it waits for."""
    if found is CommClass.BLOCKING:
        return [f"complete {row_signature(state, node)}"]
    if found is CommClass.WAIT:
        return sorted(f"complete {row_signature(s, start)}" for s, start in starts_of(state.sdfg, state, node))
    return []


#: The name a loop variable takes in a skeleton: a fusion may rename it, and the name is not compared.
LOOP_VARIABLE = "__loop"


def loop_extent(loop: LoopRegion) -> tuple[str, tuple[str, ...]]:
    """A loop's kind and its range (start, end, stride), or its condition when it is not counted, the variable
    renamed to :data:`LOOP_VARIABLE`."""
    rename = {loop.loop_variable: LOOP_VARIABLE} if loop.loop_variable else {}
    bounds = [
        loop_analysis.get_init_assignment(loop),
        loop_analysis.get_loop_end(loop),
        loop_analysis.get_loop_stride(loop),
    ]
    if loop.loop_variable and all(bound is not None for bound in bounds):
        return "for", tuple(str(dace.symbolic.pystr_to_symbolic(str(b)).subs(rename)) for b in bounds)
    condition = loop.loop_condition.as_string if loop.loop_condition is not None else ""
    return "while", (str(dace.symbolic.pystr_to_symbolic(condition).subs(rename)) if condition else "",)


def region_skeleton(region: ControlFlowRegion, kind: str, extent: tuple[str, ...]) -> Skeleton | None:
    """The skeleton of ``region``; ``None`` when it holds no row other than ``QUERY``."""
    rows: list[str | Skeleton] = []
    waits: list[str] = []
    labels: list[tuple[str, str]] = []
    for block in in_order(region):
        if isinstance(block, SDFGState):
            for node in in_order(block):
                found = comm_class(node)
                if found in (None, CommClass.QUERY):
                    continue
                if found is not CommClass.WAIT:
                    signature = row_signature(block, node)
                    labels.append((signature, str(node.label)))
                    rows.append(signature)
                for completion in completions(block, node, found):
                    labels.append((completion, str(node.label)))
                    waits.append(completion)
            continue
        inner = block_skeleton(block)
        if inner is not None:
            rows.append(inner)
    # where a completion sits among the others is free (I2 guards it): the waits compare as a sorted list
    return Skeleton(kind, extent, tuple(rows), tuple(sorted(waits)), tuple(labels)) if rows or waits else None


def block_skeleton(block: ControlFlowBlock) -> Skeleton | None:
    """The skeleton of a region that is not a state: a loop, a branch, or a plain region."""
    if isinstance(block, LoopRegion):
        kind, extent = loop_extent(block)
        return region_skeleton(block, kind, extent)
    if isinstance(block, ConditionalBlock):
        branches = [
            region_skeleton(body, "branch", (condition.as_string if condition is not None else "else",))
            for condition, body in block.branches
        ]
        if not any(branches):
            return None
        return Skeleton("if", (), tuple(b for b in branches if b is not None), ())
    if isinstance(block, ControlFlowRegion):
        return region_skeleton(block, "region", ())
    return None


def skeleton(sdfg: dace.SDFG) -> Skeleton | None:
    """The communication skeleton of ``sdfg``; ``None`` when it does not communicate."""
    return region_skeleton(sdfg, "top", ())


def skeleton_rows(found: Skeleton | None) -> dict[str, tuple[tuple[str, ...], str]]:
    """Each row of ``found``, by signature: the regions around it (kind and extent), outermost first, and its label."""
    where: dict[str, tuple[tuple[str, ...], str]] = {}

    def walk(part: Skeleton, around: tuple[str, ...]) -> None:
        here = (*around, f"{part.kind} {', '.join(part.extent)}".strip())
        named = dict(part.labels)
        for row in (*part.rows, *part.waits):
            if isinstance(row, Skeleton):
                walk(row, here)
            else:
                where[row] = (here, named[row])

    if found is not None:
        walk(found, ())
    return where


def skeleton_difference(before: Skeleton | None, after: Skeleton | None) -> str | None:
    """How ``after`` communicates differently from ``before`` (I1), as what the change would do, naming the first row
    whose place changed; ``None`` when they are equal."""
    if before == after:
        return None
    was, now = skeleton_rows(before), skeleton_rows(after)
    for row in sorted(was.keys() | now.keys()):
        if row not in now:
            return f"remove {was[row][1]}"
        if row not in was:
            return f"add {now[row][1]}"
        if was[row][0] != now[row][0]:
            regions = f"inside {' > '.join(now[row][0])}, not {' > '.join(was[row][0])}"
            return f"change how {now[row][1]} runs: {regions}"
    return "change the order of the rows that start communication"


def isolation_refusal(sdfg: dace.SDFG) -> str | None:
    """I3: every row other than ``QUERY`` sits alone in a state of the program itself, outside every doall."""
    for node, parent in sdfg.all_nodes_recursive():
        if comm_class(node) in (None, CommClass.QUERY) or not isinstance(parent, SDFGState):
            continue
        if parent.sdfg is not sdfg or parent.entry_node(node) is not None:
            return f"put {node.label} into a doall"
        others = [n for n in parent.nodes() if isinstance(n, (nodes.LibraryNode, nodes.MapEntry)) and n is not node]
        if others:
            other = others[0].map.label if isinstance(others[0], nodes.MapEntry) else others[0].label
            return f"put {node.label} into one state with {other}"
    return None
