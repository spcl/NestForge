# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``lib-before`` / ``lib-after`` / ``lib-into``: a library call whose memlets are its whole effect (a ``lib_copy``, a
reduce, a BLAS call), or a host ``Sync`` whose effect is the device work it waits for, leaves its state for a state of
its own before or after a block of the same or an enclosing region, or joins an existing state there, when no block it
crosses touches what it reads and writes."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, cast

import dace
import sympy
from dace import data, dtypes, subsets, symbolic_facts
from dace.libraries.comm import utils as comm_utils
from dace.libraries.comm.nodes import Allreduce, Irecv, Isend, Recv, ReduceScatter, Send, Wait
from dace.libraries.comm.nodes.node import NonblockingCommNode, disjoint, in_place
from dace.libraries.standard.nodes.external_call import ExternalCall
from dace.libraries.standard.nodes.sync import Sync, SyncScope
from dace.sdfg import nodes
from dace.sdfg.state import (
    BreakBlock,
    ContinueBlock,
    ControlFlowBlock,
    ControlFlowRegion,
    LoopRegion,
    ReturnBlock,
    SDFGState,
)
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import is_gpu_relevant_node

from nestforge.ir.comm import (
    CommClass,
    Extent,
    comm_class,
    container,
    crossing_refusal,
    end_extent,
    in_flight,
    in_flight_extents,
    merged,
    motion_refusal,
    row_extents,
    with_viewed,
)
from nestforge.ir.introspect import Row
from nestforge.ir.names import is_call
from nestforge.records import Refusal, Refused

LIB_BEFORE = "lib-before"
LIB_AFTER = "lib-after"
LIB_INTO = "lib-into"
#: Turns a blocking communication row into a start and a wait in the state after it.
SPLIT_BLOCKING = "split-blocking"
#: The barriers that wait on the host, outside any kernel, for the device work issued before them.
HOST_SYNCS = frozenset({SyncScope.DEVICE, SyncScope.STREAM})


@dataclass(frozen=True, slots=True)
class Effect:
    """What a library call reads and writes (container names), and the symbols its memlets and properties read."""

    reads: frozenset[str]
    writes: frozenset[str]
    symbols: frozenset[str]
    #: A host barrier: it may not cross, nor join ahead of, a block that issues device work.
    device: bool = False
    #: For a communication row, the parts of each container it reads and writes (12.1); a crossing then conflicts
    #: only where both sides touch a common element. ``None``: every container whole, compared by name.
    read_extents: Mapping[str, Extent] | None = None
    write_extents: Mapping[str, Extent] | None = None

    @property
    def touched(self) -> frozenset[str]:
        return self.reads | self.writes


def effect_refusal(state: SDFGState, node: nodes.CodeNode) -> str | None:
    """Why ``node``'s memlets are not its whole effect, or ``None`` when they are."""
    sdfg = state.sdfg
    if state.entry_node(node) is not None:
        return Refused(
            f"{node.label} runs inside a doall; only a call at the top of a state moves.", Refusal.WRONG_ROW_KIND
        )
    if isinstance(node, ExternalCall):
        return Refused(
            f"{node.label} is a kernel calling external code, whose effects its memlets do not show.", Refusal.EFFECTS
        )
    if isinstance(node, Sync):
        if node.scope not in HOST_SYNCS or any(not e.data.is_empty() for e in state.all_edges(node)):
            return Refused(
                f"{node.label} is a {node.scope.value} barrier tied to a kernel or a named stream; it stays.",
                Refusal.EFFECTS,
            )
        return None
    if node.has_side_effects(sdfg) and comm_class(node) is None:  # a communication row moves under its own rules
        return Refused(
            f"{node.label} ({type(node).__name__}) has side effects its memlets do not show (a barrier, a "
            "communication, a stateful call); it stays where it is.",
            Refusal.EFFECTS,
        )
    for edge in state.all_edges(node):
        other = edge.src if edge.dst is node else edge.dst
        if edge.data.is_empty():
            continue
        if not isinstance(other, nodes.AccessNode):
            return Refused(
                f"{node.label} is wired to a {type(other).__name__}, not to arrays alone.", Refusal.WRONG_ROW_KIND
            )
        desc = sdfg.arrays[other.data]
        if isinstance(desc, data.View) or (isinstance(desc, data.Array) and desc.may_alias):
            return Refused(
                f"{node.label} touches {other.data}, a view or an array that may alias another.", Refusal.EFFECTS
            )
        if edge.data.wcr is not None:
            return Refused(
                f"{node.label} accumulates into {other.data}; an accumulation is not moved.", Refusal.EFFECTS
            )
    return None


def device_work(state: SDFGState) -> list[nodes.Node]:
    """The nodes of ``state`` that issue or touch device work: kernel launches, GPU copies and fills, GPU arrays (DaCe's
    ``is_gpu_relevant_node``), and calls scheduled on the GPU."""
    return [
        n
        for n in state.nodes()
        if is_gpu_relevant_node(n, state.sdfg, state)
        or (isinstance(n, nodes.LibraryNode) and n.schedule in dtypes.GPU_SCHEDULES)
        or (isinstance(n, nodes.MapEntry) and n.map.schedule in dtypes.GPU_SCHEDULES)
    ]


def block_device_work(block: ControlFlowBlock) -> bool:
    states = [block] if isinstance(block, SDFGState) else list(cast(ControlFlowRegion, block).states())
    return any(device_work(state) for state in states)


def device_containers(sdfg: dace.SDFG) -> frozenset[str]:
    """The containers device work reads or writes anywhere in ``sdfg``: what a host ``Sync`` makes safe to touch."""
    names: set[str] = set()
    for state in sdfg.states():
        for node in device_work(state):
            ends = [node, state.exit_node(node)] if isinstance(node, nodes.EntryNode) else [node]
            for end in ends:
                for edge in state.all_edges(end):
                    names |= {n.data for n in (edge.src, edge.dst) if isinstance(n, nodes.AccessNode)}
    return frozenset(names)


def effect_of(state: SDFGState, node: nodes.CodeNode) -> Effect:
    if isinstance(node, Sync):  # it reads and writes, in effect, what the device work it waits for touches
        touched = device_containers(state.sdfg)
        return Effect(touched, touched, frozenset(), device=True)
    reads = frozenset(container(state, e.src, str(e.data.data)) for e in state.in_edges(node) if not e.data.is_empty())
    writes = frozenset(
        container(state, e.dst, str(e.data.data)) for e in state.out_edges(node) if not e.data.is_empty()
    )
    kind = comm_class(node)
    if kind is CommClass.WAIT:  # R3: it reads what its starts send and writes what they receive
        sent, received = in_flight(state.sdfg, state, node)
        reads, writes = reads | sent, writes | received
    symbols = set(node.free_symbols)
    for edge in state.all_edges(node):
        symbols |= {str(s) for s in edge.data.free_symbols}
    symbols -= state.sdfg.arrays.keys()
    if kind is None:
        return Effect(reads, writes, frozenset(symbols))
    extents = [row_extents(state, node, writes=w) for w in (False, True)]
    if kind is CommClass.WAIT:
        for w, found in ((False, extents[0]), (True, extents[1])):
            for name, extent in in_flight_extents(state.sdfg, state, node, writes=w).items():
                merged(found, name, extent)
    return Effect(reads, writes, frozenset(symbols), read_extents=extents[0], write_extents=extents[1])


def reaching(graph: Any, start: Any, forward: bool) -> set[Any]:
    """The nodes reachable from ``start`` (exclusive) along the graph's edges, or against them."""
    seen: set[Any] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        nxt = graph.successors(node) if forward else graph.predecessors(node)
        for other in nxt:
            if other not in seen:
                seen.add(other)
                stack.append(other)
    return seen


def in_state_refusal(state: SDFGState, node: nodes.CodeNode, effect: Effect, earlier: bool) -> str | None:
    """Whether ``node`` may leave ``state`` toward an earlier (``earlier``) or a later point: nothing it waits for, or
    nothing waiting for it, stays behind, and every other access that conflicts with it stays on the same side."""
    ordered = state.in_edges(node) if earlier else state.out_edges(node)
    if any(e.data.is_empty() for e in ordered):
        order = "after" if earlier else "before"
        return Refused(f"{node.label} is ordered {order} other nodes of {state.label} by empty edges.", Refusal.EFFECTS)
    inputs = [e.src for e in state.in_edges(node) if not e.data.is_empty()]
    outputs = [e.dst for e in state.out_edges(node) if not e.data.is_empty()]
    if any(state.in_degree(n) != 1 for n in outputs):
        return Refused(f"something else in {state.label} writes the access node {node.label} writes.", Refusal.EFFECTS)
    if earlier and any(state.in_degree(n) for n in inputs):
        return Refused(f"what {node.label} reads is computed (or ordered) in {state.label} first.", Refusal.EFFECTS)
    if not earlier and any(state.out_degree(n) for n in outputs):
        return Refused(f"what {node.label} writes is read (or ordered) in {state.label} after it.", Refusal.EFFECTS)
    return out_of_order(state, node, effect, {node, *inputs, *outputs} | reaching(state, node, forward=earlier))


def out_of_order(state: SDFGState, node: nodes.CodeNode, effect: Effect, kept: set[nodes.Node]) -> str | None:
    """Why an access of ``state`` outside ``kept`` (the call, its own access nodes, and the side it leaves toward)
    conflicts with the call: it writes what the call touches or reads what the call writes."""
    for other in state.data_nodes():
        if other in kept:
            continue
        if (written(state, other) and other.data in effect.touched) or (
            read(state, other) and other.data in effect.writes
        ):
            return Refused(f"{state.label} accesses {other.data} out of order with {node.label}.", Refusal.EFFECTS)
    return None


def accessed_beside(state: SDFGState, node: nodes.CodeNode, names: frozenset[str], writes: bool) -> str | None:
    """A container of ``names`` that a node of ``state`` other than ``node``'s outputs writes (``writes``) or reads."""
    outputs = {e.dst for e in state.out_edges(node)}
    for other in state.data_nodes():
        edges = state.in_edges(other) if writes else state.out_edges(other)
        if other.data in names and other not in outputs and any(not e.data.is_empty() for e in edges):
            return str(other.data)
    return None


def loop_state_refusal(
    state: SDFGState, node: nodes.CodeNode, effect: Effect, loop: LoopRegion, earlier: bool
) -> str | None:
    """What the call's own state does every iteration of ``loop``: it may not change what the call touches, nor, when
    the call sinks after ``loop``, read what it writes."""
    written = accessed_beside(state, node, effect.touched, writes=True)
    if written is not None:
        return Refused(f"{state.label} in {loop.label} writes {written}, which the call touches.", Refusal.EFFECTS)
    read = None if earlier else accessed_beside(state, node, effect.writes, writes=False)
    if read is not None:
        return Refused(
            f"{state.label} reads {read} every iteration; after {loop.label}, those reads would see the old value.",
            Refusal.EFFECTS,
        )
    return None


def interstate_refusal(edge_data: dace.InterstateEdge, effect: Effect, where: str) -> str | None:
    read = edge_data.read_symbols()
    if read & effect.writes:
        return Refused(
            f"a transition in {where} reads {min(read & effect.writes)}, which the call writes.", Refusal.EFFECTS
        )
    assigned = sorted(set(edge_data.assignments) & effect.symbols)
    if assigned:
        return Refused(f"a transition in {where} assigns {assigned[0]}, which the call reads.", Refusal.EFFECTS)
    return None


def assigned_symbols(block: ControlFlowBlock) -> set[str]:
    """The symbols ``block`` assigns: on its transitions and as loop variables, nested regions included."""
    if not isinstance(block, ControlFlowRegion):
        return set()
    names = {str(k) for e in block.all_interstate_edges() for k in e.data.assignments}
    loops = [block, *block.all_control_flow_blocks(recursive=True)]
    return names | {b.loop_variable for b in loops if isinstance(b, LoopRegion) and b.loop_variable}


def accessed(block: ControlFlowBlock) -> tuple[set[str], set[str]]:
    """The containers ``block`` reads and writes, through views too."""
    reads, writes = block.read_and_write_sets()
    return with_viewed(block, set(reads)), with_viewed(block, set(writes))


#: A part of a container a block touches, and the relations that hold whenever it is touched: the ranges of the doall
#: whose memlet it is are nonempty (an empty doall touches nothing, so a proof may assume it runs).
Guarded = tuple[subsets.Subset, tuple[symbolic_facts.Relation, ...]]


def runs(state: SDFGState, end: nodes.Node) -> tuple[symbolic_facts.Relation, ...]:
    """What holds whenever the doall at ``end`` (its entry or exit) runs: every range is nonempty. A ``Min`` last
    iteration or a ``Max`` first one (``split-doall``'s clamps) holds for each of its arguments, so the relations stay
    linear."""
    if not isinstance(end, (nodes.MapEntry, nodes.MapExit)):
        return ()
    return tuple(
        dace.symbolic.predicate_relation(dace.symbolic.Predicate.NONNEGATIVE, last - first)
        for lo, hi, _ in end.map.range
        for last in (hi.args if isinstance(hi, sympy.Min) else (hi,))
        for first in (lo.args if isinstance(lo, sympy.Max) else (lo,))
    )


def block_extents(block: ControlFlowBlock, names: set[str], writes: bool) -> dict[str, list[Guarded] | None]:
    """The elements of each container of ``names`` that ``block`` writes (``writes``) or reads, from the memlets at
    its access nodes, each with what holds whenever it is touched (:func:`runs`). A container touched elsewhere (a
    transition), or through a subset naming a symbol a region assigns, counts whole."""
    states = [block] if isinstance(block, SDFGState) else list(cast(ControlFlowRegion, block).states())
    changing = set() if isinstance(block, SDFGState) else assigned_symbols(block)
    extents: dict[str, list[Guarded] | None] = {}
    for state in states:
        for node in state.data_nodes():
            name = container(state, node, node.data)
            if name not in names:
                continue
            for edge in state.in_edges(node) if writes else state.out_edges(node):
                if edge.data.is_empty():
                    continue
                extent = end_extent(state, edge, node)
                if extent is not None and any({str(s) for s in part.free_symbols} & changing for part in extent):
                    extent = None
                other = edge.src if writes else edge.dst
                guard = runs(state, other) if state.entry_node(other) is None else ()
                known = extents.get(name, [])
                extents[name] = (
                    None if known is None or extent is None else [*known, *((part, guard) for part in extent)]
                )
    return {name: extents.get(name) for name in names}


def overlapping(
    block: ControlFlowBlock, names: set[str], theirs_written: bool, ours: list[Mapping[str, Extent] | None]
) -> list[str]:
    """The containers of ``names`` where what ``block`` writes (``theirs_written``) or reads shares an element with the
    call's accesses ``ours`` (its read and write extents, or ``None`` to compare by name: 12.1 holds for communication
    rows only); each named with both subsets once they are known."""
    if any(extents is None for extents in ours):
        return sorted(names)
    facts = block.sdfg.facts()
    found = []
    for name, theirs in block_extents(block, names, theirs_written).items():
        mine = [extents.get(name, []) for extents in ours if extents is not None]
        if theirs is None or any(part is None for part in mine):
            found.append(name)
            continue
        parts = [subset for part in mine if part is not None for subset in part]
        clash = [
            (a, b)
            for a, guard in theirs
            for b in parts
            if not disjoint(a, b, facts.with_relations(guard) if guard else facts)
        ]
        if clash:
            found.append(f"{name}[{clash[0][0]}] (the call: {name}[{clash[0][1]}])")
    return sorted(found)


def crossed_refusal(block: ControlFlowBlock, effect: Effect) -> str | None:
    if effect.device and block_device_work(block):
        return Refused(
            f"{block.label} issues device work, which the barrier would no longer wait for (or wait for too early).",
            Refusal.EFFECTS,
        )
    reads, writes = accessed(block)
    written = overlapping(block, writes & effect.touched, True, [effect.read_extents, effect.write_extents])
    if written:
        return Refused(f"{block.label} writes {written[0]}, which the call touches.", Refusal.EFFECTS)
    read = overlapping(block, reads & effect.writes, False, [effect.write_extents])
    if read:
        return Refused(f"{block.label} reads {read[0]}, which the call writes.", Refusal.EFFECTS)
    if assigned_symbols(block) & effect.symbols:
        return Refused(
            f"{block.label} assigns {min(assigned_symbols(block) & effect.symbols)}, which the call reads.",
            Refusal.EFFECTS,
        )
    return None


def loop_refusal(loop: LoopRegion, child: ControlFlowBlock, effect: Effect, earlier: bool) -> str | None:
    """Whether the call, in ``child`` of ``loop``, runs once before (``earlier``) or after ``loop`` with the effect of
    every iteration: it reads nothing the loop changes and nothing reads its writes ahead of it."""
    if not loop_analysis.provably_runs(loop, 1, loop.sdfg.facts()):
        return Refused(
            f"{loop.label} is not proven to run once (its trip count may be zero); the call would run when it did not.",
            Refusal.EFFECTS,
        )
    if effect.reads & effect.writes:
        return Refused(
            "the call reads what it writes, so running it once is not running it every iteration.", Refusal.EFFECTS
        )
    if any(isinstance(b, (BreakBlock, ContinueBlock, ReturnBlock)) for b in loop.all_control_flow_blocks(True)):
        return Refused(
            f"{loop.label} breaks, continues or returns, so the call does not run every iteration.", Refusal.EFFECTS
        )
    if effect.symbols & ({loop.loop_variable} | assigned_symbols(loop)):
        return Refused(f"the call reads a symbol {loop.label} changes.", Refusal.EFFECTS)
    meta = set().union(*(c.get_free_symbols() for c in loop.get_meta_codeblocks()))
    if meta & effect.writes:
        return Refused(f"{loop.label}'s bounds read what the call writes.", Refusal.EFFECTS)
    return body_refusal(loop, child, effect, earlier)


def body_refusal(loop: LoopRegion, child: ControlFlowBlock, effect: Effect, earlier: bool) -> str | None:
    """Whether a block or transition of ``loop`` besides ``child`` (the call's) writes what the call touches, or
    reads what it writes ahead of it."""
    after = reaching(loop, child, forward=True)
    for block in loop.nodes():
        if block is child:
            continue
        reads, writes = accessed(block)
        if writes & effect.touched:
            return Refused(
                f"{block.label} in {loop.label} writes {min(writes & effect.touched)}, which the call touches.",
                Refusal.EFFECTS,
            )
        if reads & effect.writes and not (earlier and block in after):
            return Refused(
                f"{block.label} in {loop.label} reads {min(reads & effect.writes)}, which the call writes.",
                Refusal.EFFECTS,
            )
    for edge in loop.edges():
        if (edge.data.read_symbols() & effect.writes) and not (earlier and (edge.src is child or edge.src in after)):
            return Refused(f"a transition in {loop.label} reads what the call writes.", Refusal.EFFECTS)
    return None


def exit_chain(state: SDFGState, target: ControlFlowBlock) -> tuple[list[LoopRegion], ControlFlowBlock] | str:
    """The loops the call leaves to reach ``target``'s region, innermost first, and the block of that region that
    holds it; or why ``target``'s region does not enclose the call."""
    if target.sdfg is not state.sdfg:
        return Refused(f"{target.label} is in another SDFG than the call.", Refusal.WRONG_ROW_KIND)
    region, anchor, loops = target.parent_graph, state, []
    while anchor.parent_graph is not region:
        parent = anchor.parent_graph
        if parent is None or isinstance(parent, dace.SDFG):
            return Refused(
                f"{target.label} is not in a region enclosing the call; a call moves only out, never into a block.",
                Refusal.WRONG_ROW_KIND,
            )
        if not isinstance(parent, LoopRegion):
            return Refused(
                f"the call would leave {parent.label}, which runs under a condition; it would run speculatively.",
                Refusal.EFFECTS,
            )
        loops.append(parent)
        anchor = parent
    return loops, anchor


def plan_lib_motion(row: Row, target_row: Row, place: str) -> Callable[[], object] | str:
    """The move of the library call ``row`` into a fresh state before or after (``place``) ``target_row``'s block,
    or ``into`` that state, or why it is illegal; nothing is mutated."""
    node, state = row
    target = target_row[0]
    assert isinstance(state, SDFGState) and is_call(state, node), "a library call row"
    why = effect_refusal(state, node) or motion_refusal(node, state, target, place)
    if why is not None:
        return why
    if target is state:
        return Refused(f"{node.label} is in {state.label} already; name another block.", Refusal.WRONG_ROW_KIND)
    if place == "into" and not isinstance(target, SDFGState):
        return Refused(f"{target.label} is not a state; a call joins only a state.", Refusal.WRONG_ROW_KIND)
    chain = exit_chain(state, target)
    if isinstance(chain, str):
        return chain
    loops, anchor = chain
    region = target.parent_graph
    assert region is not None, "exit_chain found the target's region"
    earlier = target in reaching(region, anchor, forward=False) or (target is anchor and place == "before")
    if not earlier and target is not anchor and target not in reaching(region, anchor, forward=True):
        return Refused(f"{target.label} is neither before nor after the call on any path.", Refusal.WRONG_ROW_KIND)
    effect = effect_of(state, node)
    why = path_refusal(state, node, effect, loops, earlier) or span_refusal(
        node, effect, region, anchor, target, earlier, place
    )
    if place != "into":
        return why or (lambda: move_call(state, node, target, place == "after"))
    joint = why or joint_of(cast(SDFGState, target), node, effect, earlier)
    if isinstance(joint, str):
        return joint
    return lambda: join_call(state, node, joint)


def span_refusal(
    node: nodes.CodeNode,
    effect: Effect,
    region: ControlFlowRegion,
    anchor: ControlFlowBlock,
    target: ControlFlowBlock,
    earlier: bool,
    place: str,
) -> str | None:
    """Whether the call may cross every block and interstate edge of ``region`` between ``anchor`` (the block it
    leaves) and ``target``."""
    first, last = (target, anchor) if earlier else (anchor, target)
    span = (reaching(region, first, forward=True) & reaching(region, last, forward=False)) | {first, last}
    # the target is crossed when the call lands on its far side
    crossed = span - {anchor} - ({target} if place == "into" or earlier == (place == "after") else set())
    for block in crossed:
        why = crossing_refusal(node, block) or crossed_refusal(block, effect)
        if why is not None:
            return why
    for edge in region.edges():
        if edge.src in span and edge.dst in span:
            why = interstate_refusal(edge.data, effect, region.label)
            if why is not None:
                return why
    return None


def path_refusal(
    state: SDFGState, node: nodes.CodeNode, effect: Effect, loops: list[LoopRegion], earlier: bool
) -> str | None:
    """Whether the call may leave its state and the ``loops`` around it toward an earlier or a later point."""
    why = in_state_refusal(state, node, effect, earlier)
    if why is None and loops:
        why = loop_state_refusal(state, node, effect, loops[0], earlier)
    child: ControlFlowBlock = state
    for loop in loops:
        why = why or loop_refusal(loop, child, effect, earlier)
        child = loop
    return why


@dataclass(frozen=True, slots=True)
class Joint:
    """Where a call joins a state: the access nodes it reads from and writes to there, by container (new ones for the
    rest), and the nodes ordered before and after it by empty edges."""

    state: SDFGState
    sources: dict[str, nodes.AccessNode]
    sinks: dict[str, nodes.AccessNode]
    before: list[nodes.Node]
    after: list[nodes.Node]


def written(state: SDFGState, access: nodes.AccessNode) -> bool:
    return any(not e.data.is_empty() for e in state.in_edges(access))


def read(state: SDFGState, access: nodes.AccessNode) -> bool:
    return any(not e.data.is_empty() for e in state.out_edges(access))


def last_writer(state: SDFGState, name: str) -> nodes.AccessNode | None | str:
    """The access node of ``name`` that every other writer of it in ``state`` comes before; ``None`` when nothing
    writes it; why not, when two writers are unordered."""
    writers = [a for a in state.data_nodes() if a.data == name and written(state, a)]
    last = [w for w in writers if not reaching(state, w, forward=True) & set(writers)]
    if len(last) > 1 or any(w is not last[0] and last[0] not in reaching(state, w, True) for w in writers):
        return Refused(f"{state.label} writes {name} twice with no order between the writes.", Refusal.EFFECTS)
    return last[0] if last else None


def producers(state: SDFGState, access: nodes.AccessNode) -> list[nodes.Node]:
    """The nodes that write ``access``: a map is ordered by its entry."""
    return [
        state.entry_node(e.src) if isinstance(e.src, nodes.ExitNode) else e.src
        for e in state.in_edges(access)
        if not e.data.is_empty()
    ]


def joint_of(target: SDFGState, node: nodes.CodeNode, effect: Effect, earlier: bool) -> Joint | str:
    """How the call joins ``target``: after every access there that conflicts with it, when it comes from a later
    point (``earlier``), else before every one, so no cycle can form; refused for an accumulation or unordered
    writers."""
    clashing = [
        a
        for a in target.data_nodes()
        if (written(target, a) and a.data in effect.touched) or (read(target, a) and a.data in effect.writes)
    ]
    why = join_refusal(target, clashing, effect, earlier)
    if why is not None:
        return why
    return earlier_joint(target, clashing, effect) if earlier else later_joint(target, clashing, effect)


def join_refusal(target: SDFGState, clashing: list[nodes.AccessNode], effect: Effect, earlier: bool) -> str | None:
    """Why the call cannot join ``target``, whose ``clashing`` accesses conflict with it: one accumulates, two of
    them write unordered, or a barrier would come before device work."""
    for access in clashing:
        if any(e.data.wcr is not None for e in target.in_edges(access)):
            return Refused(
                f"{target.label} accumulates into {access.data}; the call does not join an accumulation.",
                Refusal.EFFECTS,
            )
    for name in sorted({a.data for a in clashing if written(target, a)}):
        last = last_writer(target, name)
        if isinstance(last, str):
            return last
    if effect.device and not earlier and device_work(target):
        return Refused(
            f"{target.label} issues device work the barrier would come before; it joins only after such work.",
            Refusal.EFFECTS,
        )
    return None


def earlier_joint(target: SDFGState, clashing: list[nodes.AccessNode], effect: Effect) -> Joint:
    """The call, from a later point, joins ``target`` after every ``clashing`` access, reading each input from its
    last writer there."""
    sources = {}
    for name in () if effect.device else effect.reads:  # a barrier has no input to wire
        last = last_writer(target, name)
        if isinstance(last, nodes.AccessNode):
            sources[name] = last
    work = [target.exit_node(n) if isinstance(n, nodes.EntryNode) else n for n in device_work(target)]
    before = [a for a in clashing if a not in sources.values()] + (work if effect.device else [])
    return Joint(target, sources, {}, before, [])


def later_joint(target: SDFGState, clashing: list[nodes.AccessNode], effect: Effect) -> Joint:
    """The call, from an earlier point, joins ``target`` before every ``clashing`` access, writing into the access
    nodes there that nothing writes yet."""
    sinks = {}
    for name in effect.writes:
        fresh = [a for a in target.data_nodes() if a.data == name and target.in_degree(a) == 0]
        if fresh:
            sinks[name] = fresh[0]
    after: list[nodes.Node] = []
    for access in clashing:
        if access in sinks.values():
            continue
        after += producers(target, access) if written(target, access) else [access]
    return Joint(target, {}, sinks, [], after)


def join_call(state: SDFGState, node: nodes.CodeNode, joint: Joint) -> SDFGState:
    """Take ``node`` out of ``state`` into ``joint.state``, wired and ordered as ``joint`` says."""
    target = joint.state
    edges = detach(state, node)
    target.add_node(node)
    for edge in edges:
        if edge.dst is node:
            source = joint.sources.get(edge.src.data) or target.add_access(edge.src.data)
            target.add_edge(source, edge.src_conn, node, edge.dst_conn, copy.deepcopy(edge.data))
        else:
            sink = joint.sinks.get(edge.dst.data) or target.add_access(edge.dst.data)
            target.add_edge(node, edge.src_conn, sink, edge.dst_conn, copy.deepcopy(edge.data))
    for other in dict.fromkeys(joint.before):
        target.add_nedge(other, node, dace.Memlet())
    for other in dict.fromkeys(joint.after):
        target.add_nedge(node, other, dace.Memlet())
    return target


def detach(state: SDFGState, node: nodes.CodeNode) -> list[Any]:
    """Remove ``node`` from ``state`` with the access nodes only it used; returns its data edges."""
    edges = list(state.all_edges(node))
    state.remove_node(node)
    for edge in edges:
        for end in (edge.src, edge.dst):
            if (
                end is not node
                and end in state.nodes()
                and state.degree(end) == 0
                and isinstance(end, nodes.AccessNode)
            ):
                state.remove_node(end)
    return [e for e in edges if not e.data.is_empty()]


def move_call(state: SDFGState, node: nodes.CodeNode, target: ControlFlowBlock, after: bool) -> SDFGState:
    """Take ``node`` and its access nodes out of ``state`` into a new state next to ``target``."""
    region = target.parent_graph
    label = f"{node.label}_moved"
    if after:
        moved = region.add_state_after(target, label)
    else:
        moved = region.add_state_before(target, label, is_start_block=region.start_block is target)
    return join_call(state, node, Joint(moved, {}, {}, [], []))


def movable_rows(sdfg: dace.SDFG, states_only: bool = False) -> Iterator[tuple[Row, Row]]:
    """Every top-level library call paired with every other block (or state) of its region and of each loop
    around it."""
    for state in sdfg.states():
        calls = [n for n in state.nodes() if is_call(state, n) and state.entry_node(n) is None]
        if not calls:
            continue
        regions, block = [], state
        while isinstance(block.parent_graph, ControlFlowRegion):
            regions.append(block.parent_graph)
            if not isinstance(block.parent_graph, LoopRegion):
                break
            block = block.parent_graph
        for call in calls:
            for region in regions:
                blocks = [b for b in region.nodes() if b is not state]
                if states_only:
                    blocks = [b for b in blocks if isinstance(b, SDFGState)]
                yield from (((call, state), (b, None)) for b in blocks)
                if comm_class(call) is not None:  # a doall that shares its state: the call lands next to it
                    yield from (((call, state), (entry, b)) for b in blocks for entry in shared_doalls(b))


def shared_doalls(block: ControlFlowBlock) -> list[nodes.MapEntry]:
    """The top-level doalls of ``block``, a state, that share it with another top-level doall."""
    if not isinstance(block, SDFGState):
        return []
    doalls = [n for n in block.nodes() if isinstance(n, nodes.MapEntry) and block.entry_node(n) is None]
    return doalls if len(doalls) > 1 else []


def split_part(state: SDFGState, entry: nodes.MapEntry, after: bool) -> set[nodes.Node]:
    """The nodes of ``state`` that go first when it splits at the top-level doall ``entry``: the doall and what it
    needs (``after``, so a call can follow it), or what neither is nor needs the doall (so a call can precede it)."""
    scope = set(state.scope_subgraph(entry, include_entry=True, include_exit=True).nodes())
    if after:
        return scope | reaching(state, entry, forward=False)
    return set(state.nodes()) - scope - reaching(state, state.exit_node(entry), forward=True)


def split_refusal(state: SDFGState, entry: nodes.MapEntry, after: bool) -> str | None:
    """Why ``state`` cannot split at ``entry``: an edge out of the first part that no access node carries."""
    first = split_part(state, entry, after)
    for edge in state.edges():
        if edge.src in first and edge.dst not in first:
            if not isinstance(edge.src, nodes.AccessNode) and not isinstance(edge.dst, nodes.AccessNode):
                return Refused(
                    f"{state.label} cannot split at {entry.map.label}: {edge.src} and {edge.dst} are ordered with "
                    "no container between them.",
                    Refusal.WRONG_ROW_KIND,
                )
    return None


def split_at(state: SDFGState, entry: nodes.MapEntry, after: bool) -> SDFGState:
    """Split ``state`` in two at the top-level doall ``entry`` (:func:`split_part` goes first, into a new state
    before it); returns the state that holds the doall. An access node on the cut is in both states, so each state
    keeps its own reads and writes of the container. Legal whenever :func:`split_refusal` is ``None``: the first part
    needs nothing of the second."""
    first = split_part(state, entry, after)
    head = state.parent_graph.add_state_before(state, label=f"{state.label}_head")
    inside = [e for e in state.edges() if e.src in first and e.dst in first]
    cut = [e for e in state.edges() if e.src in first and e.dst not in first]
    twins: dict[nodes.Node, nodes.AccessNode] = {}
    for node in first:
        state.remove_node(node)
        head.add_node(node)
    for edge in inside:
        head.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, edge.data)
    for edge in cut:
        if isinstance(edge.src, nodes.AccessNode):  # the second state reads the container again
            twin = twins.setdefault(edge.src, state.add_access(edge.src.data))
            state.add_edge(twin, edge.src_conn, edge.dst, edge.dst_conn, edge.data)
        else:  # the first state writes the container into an access node of its own
            twin = twins.setdefault(edge.dst, head.add_access(edge.dst.data))
            head.add_edge(edge.src, edge.src_conn, twin, edge.dst_conn, edge.data)
    for part in (head, state):  # an access node whose edges all went to the other part
        for node in [n for n in part.nodes() if isinstance(n, nodes.AccessNode) and part.degree(n) == 0]:
            part.remove_node(node)
    return head if after else state


def split_blocking_refusal(state: SDFGState, node: nodes.Node) -> str | None:
    """Why the blocking row ``node`` has no start-and-wait form here; ``None`` when ``split-blocking`` applies."""
    label = node.label
    if comm_class(node) is not CommClass.BLOCKING:
        return Refused(f"{label} is no blocking communication row; split-blocking takes one.", Refusal.WRONG_ROW_KIND)
    if state.entry_node(node) is not None:
        return Refused(f"{label} is inside a doall; a communication row stays out of doalls.", Refusal.COMMUNICATION)
    if isinstance(node, (Send, Recv)):
        return None
    if not isinstance(node, NonblockingCommNode):
        return Refused(f"{label} ({type(node).__name__}) has no nonblocking form to split into.", Refusal.COMMUNICATION)
    if isinstance(node, (Allreduce, ReduceScatter)) and comm_utils.is_average(node.op):
        return Refused(
            f"{label} averages, which MPI cannot start without blocking: reduce the sum and divide after.",
            Refusal.COMMUNICATION,
        )
    buffers = [e for e in state.all_edges(node) if (e.dst_conn if e.dst is node else e.src_conn) in BUFFERS]
    if any(not in_place(state.sdfg, state, e, node) for e in buffers):
        return Refused(
            f"{label} has a strided buffer; a started collective needs contiguous ones.", Refusal.COMMUNICATION
        )
    return None


#: The buffer connectors of a collective.
BUFFERS = ("_inbuffer", "_outbuffer")


def plan_split_blocking(row: Row) -> Callable[[], object] | str:
    """``Send`` -> ``Isend``, ``Recv`` -> ``Irecv``, a collective -> its ``nonblocking`` form, each writing a fresh
    request that a ``Wait`` in a new state right after reads. Nothing runs between the two, so the program is the same
    until motion separates them."""
    node, state = row
    assert isinstance(state, SDFGState), "a communication row"
    why = split_blocking_refusal(state, node)
    if why is not None:
        return why

    def commit() -> None:
        sdfg = state.sdfg
        request, _ = sdfg.add_array(
            "split_request", [1], dtypes.opaque("MPI_Request"), transient=True, find_new_name=True
        )
        start = started(state, node)
        state.add_edge(start, "_request", state.add_write(request), None, dace.Memlet(f"{request}[0]"))
        waits = state.parent_graph.add_state_after(state, f"{start.label}_wait")
        wait = Wait(f"{start.label}_wait")
        for status in ("_stat_tag", "_stat_source"):  # the receive's status: optional, and nothing reads it
            wait.remove_out_connector(status)
        waits.add_edge(waits.add_read(request), None, wait, "_request", dace.Memlet(f"{request}[0]"))

    return commit


def started(state: SDFGState, node: nodes.Node) -> nodes.LibraryNode:
    """The row that starts what ``node`` does: ``node`` itself made nonblocking, or an ``Isend`` (``Irecv``) wired as
    the ``Send`` (``Recv``) was, in its place."""
    if isinstance(node, NonblockingCommNode):
        node.nonblocking = True
        node.add_out_connector("_request")
        return node
    assert isinstance(node, (Send, Recv)), "split_blocking_refusal let only these through"
    start: nodes.LibraryNode = (Isend if isinstance(node, Send) else Irecv)(node.label)
    start.implementation = node.implementation
    state.add_node(start)
    for edge in state.in_edges(node):
        state.add_edge(edge.src, edge.src_conn, start, edge.dst_conn, copy.deepcopy(edge.data))
    for edge in state.out_edges(node):
        state.add_edge(start, edge.src_conn, edge.dst, edge.dst_conn, copy.deepcopy(edge.data))
    state.remove_node(node)
    return start
