# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The accumulation length ``l`` of each output of a kernel: how many terms the kernel folds into one element, the
``l`` of HPCAgent-Bench's reassociation floor ``eps_acc * sqrt(l)``. The bench derives it per program output from
the BenchSpec (``harness.grading.contracted_extent``); a kernel's outputs are often transients no spec names, so here
it comes from the kernel's SDFG: the executions of everything that writes an output, over the output's size.

Counted at the check sizes. A bound that reads an enclosing iterator (a triangular loop) takes its larger end, so a
count is an upper bound, at most a small factor high. An output takes the longest ``l`` of the transients its writers
read, so a sum kept in scratch and copied out keeps its length. A writer that cannot be counted (a library node, a
loop with no counted trips, a size with no value) gives its outputs the size of the largest input, the bench's own
fallback (``largest_input_extent``)."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import dace
import numpy as np
import sympy
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation.passes.analysis import loop_analysis

from nestforge.build.toolchain import SymbolValue

#: The most free symbols an expression is evaluated over: each takes both ends of its interval, 2**n evaluations.
MAX_CORNER_SYMBOLS = 8

#: The smallest and largest value a symbol takes in the scope being counted.
type Interval = tuple[int, int]


def corners(expr: object, env: Mapping[str, Interval]) -> Interval | None:
    """The least and greatest value of ``expr`` over the ends of the intervals of its free symbols; exact for a bound
    linear in each symbol. ``None`` when a symbol has no interval or the value is not a number."""
    symbolic = dace.symbolic.pystr_to_symbolic(expr) if isinstance(expr, str) else sympy.sympify(expr)
    names = sorted({str(sym) for sym in free_symbols(symbolic)})
    if any(name not in env for name in names) or len(names) > MAX_CORNER_SYMBOLS:
        return None
    values: list[int] = []
    for choice in itertools.product(*(env[name] for name in names)):
        try:
            value = dace.symbolic.evaluate(symbolic, dict(zip(names, choice, strict=True)))
            values.append(math.ceil(float(value)))
        except (TypeError, ValueError, OverflowError):
            return None
    return min(values), max(values)


def free_symbols(expr: object) -> set[sympy.Basic]:
    """The free symbols of a sympy expression; none for a plain number."""
    return set(expr.free_symbols) if isinstance(expr, sympy.Basic) else set()


def largest(expr: object, env: Mapping[str, Interval]) -> int | None:
    span = corners(expr, env)
    return None if span is None else span[1]


@dataclass(frozen=True, slots=True)
class Frame:
    """Where a node sits: how often one execution of it repeats (``None``: not countable), the interval of every
    symbol in scope, and the key each container of the current SDFG has in the kernel's top SDFG."""

    count: int | None
    env: Mapping[str, Interval]
    names: Mapping[str, str]
    prefix: str

    def key(self, name: str) -> str:
        """The container's key: its top-level name, or a name of its own for a nested SDFG's local container."""
        return self.names.get(name, f"{self.prefix}{name}")

    def times(self, factor: int | None, bound: Mapping[str, Interval]) -> Frame:
        """This frame repeated ``factor`` times, with ``bound``'s iterators in scope."""
        count = None if self.count is None or factor is None else self.count * factor
        return Frame(count, {**self.env, **bound}, self.names, self.prefix)


@dataclass(frozen=True, slots=True)
class Write:
    """One writer of a container: how often it runs (``None``: not countable), what it reads, and whether its memlet
    accumulates (a WCR). A library node is taken as accumulating: most are reductions, and it is not countable."""

    count: int | None
    reads: tuple[str, ...]
    accumulates: bool


@dataclass(slots=True)
class Ledger:
    """Per container: its writers, its size, and whether it is a transient (scratch the kernel owns, whose length an
    output reading it inherits)."""

    writes: dict[str, list[Write]] = field(default_factory=dict)
    sizes: dict[str, int | None] = field(default_factory=dict)
    transients: dict[str, None] = field(default_factory=dict)

    def write(self, key: str, write: Write) -> None:
        self.writes.setdefault(key, []).append(write)

    def reads(self, key: str) -> dict[str, None]:
        """Every container the writers of ``key`` read."""
        return {name: None for write in self.writes.get(key, []) for name in write.reads}

    def reach(self, names: Sequence[str], through: str) -> dict[str, None]:
        """Every container ``names`` are computed from: themselves, and what the writers of each transient among them
        (and of ``through``) read, transitively."""
        seen: dict[str, None] = {}
        stack = list(names)
        while stack:
            name = stack.pop()
            if name in seen:
                continue
            seen[name] = None
            if name in self.transients or name == through:
                stack.extend(self.reads(name))
        return seen

    def feeds(self, write: Write, key: str) -> bool:
        """Whether ``write`` reads ``key`` itself, directly or through the transients it reads: a read-modify-write."""
        return key in self.reach(write.reads, key)

    def cycle(self, key: str) -> dict[str, None]:
        """``key`` and the transients its value goes round through: each is computed from ``key`` and feeds it."""
        members = {key: None}
        for name in self.reach(list(self.reads(key)), key):
            if name in self.transients and key in self.reach(list(self.reads(name)), key):
                members[name] = None
        return members

    def starts(self, key: str) -> int:
        """How many chains through ``key`` begin afresh: writes into its cycle that read nothing the cycle holds (an
        initialization), however many of its members they write."""
        members = self.cycle(key)
        return sum(
            write.count or 0
            for name in members
            for write in self.writes.get(name, [])
            if not write.accumulates and not members.keys() & self.reach(write.reads, key).keys()
        )


def map_frame(entry: nodes.MapEntry, frame: Frame) -> Frame:
    """``frame`` inside the map of ``entry``: repeated by its iteration count, its parameters in scope."""
    bound: dict[str, Interval] = {}
    factor: int | None = 1
    env = dict(frame.env)
    for param, (begin, end, step) in zip(entry.map.params, entry.map.range, strict=True):
        trips = largest(dace.symbolic.int_floor(end - begin, step) + 1, env)
        first, last = corners(begin, env), corners(end, env)
        factor = None if factor is None or trips is None else factor * max(trips, 0)
        if first is not None and last is not None:
            bound[param] = env[param] = (min(first[0], last[0]), max(first[1], last[1]))
    return frame.times(factor, bound)


def loop_frame(loop: LoopRegion, frame: Frame) -> Frame:
    """``frame`` inside ``loop``: repeated by its largest trip count, its iterator in scope; not countable when the
    loop has no counted bounds (a while loop)."""
    start, end = loop_analysis.get_init_assignment(loop), loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if not loop.loop_variable or start is None or end is None or stride is None:
        return frame.times(None, {})
    step = corners(stride, frame.env)
    if step is None or step[0] != step[1] or step[0] == 0:
        return frame.times(None, {})
    trips = largest(dace.symbolic.int_floor(end - start, step[0]) + 1, frame.env)
    first, last = corners(start, frame.env), corners(end, frame.env)
    bound = {} if first is None or last is None else {loop.loop_variable: (min(*first, *last), max(*first, *last))}
    return frame.times(None if trips is None else max(trips, 0), bound)


def scope_frame(state: SDFGState, node: nodes.Node, frame: Frame, scopes: dict[nodes.Node, nodes.Node | None]) -> Frame:
    """``frame`` inside every map around ``node`` in ``state``, outermost first."""
    chain: list[nodes.MapEntry] = []
    parent = scopes.get(node)
    while parent is not None:
        if isinstance(parent, nodes.MapEntry):
            chain.append(parent)
        parent = scopes.get(parent)
    for entry in reversed(chain):
        frame = map_frame(entry, frame)
    return frame


def source_key(state: SDFGState, edge: dace.sdfg.graph.MultiConnectorEdge, frame: Frame) -> str | None:
    first = state.memlet_path(edge)[0].src
    return frame.key(first.data) if isinstance(first, nodes.AccessNode) else None


def sink_key(state: SDFGState, edge: dace.sdfg.graph.MultiConnectorEdge, frame: Frame) -> str | None:
    last = state.memlet_path(edge)[-1].dst
    return frame.key(last.data) if isinstance(last, nodes.AccessNode) else None


def node_reads(state: SDFGState, node: nodes.Node, frame: Frame) -> list[str]:
    keys = (source_key(state, edge, frame) for edge in state.in_edges(node))
    return [key for key in keys if key is not None]


def count_state(state: SDFGState, frame: Frame, ledger: Ledger) -> None:
    """Record every writer in ``state``: tasklets, library nodes (uncountable), copies, and nested SDFGs."""
    scopes = state.scope_dict()
    for node in state.nodes():
        if isinstance(node, (nodes.Tasklet, nodes.LibraryNode)):
            here = scope_frame(state, node, frame, scopes)
            library = isinstance(node, nodes.LibraryNode)
            read = tuple(node_reads(state, node, frame))
            for edge in state.out_edges(node):
                key = sink_key(state, edge, frame)
                wcr = any(e.data.wcr is not None for e in state.memlet_path(edge))
                if key is not None:
                    ledger.write(key, Write(None if library else here.count, read, library or wcr))
        elif isinstance(node, nodes.NestedSDFG):
            count_nested(state, node, scope_frame(state, node, frame, scopes), ledger)
        elif isinstance(node, nodes.AccessNode):
            count_copies(state, node, scope_frame(state, node, frame, scopes), ledger)


def count_copies(state: SDFGState, node: nodes.AccessNode, frame: Frame, ledger: Ledger) -> None:
    """A direct copy into another container writes each element it moves once, per execution of its scope."""
    for edge in state.out_edges(node):
        if not isinstance(edge.dst, nodes.AccessNode):
            continue
        volume = largest(edge.data.volume, frame.env)
        count = None if frame.count is None or volume is None else frame.count * volume
        ledger.write(frame.key(edge.dst.data), Write(count, (frame.key(node.data),), edge.data.wcr is not None))


def count_nested(state: SDFGState, node: nodes.NestedSDFG, frame: Frame, ledger: Ledger) -> None:
    """Descend into a nested SDFG: its connectors take the keys of the containers they bind, its symbols the
    intervals of the expressions they are bound to."""
    names: dict[str, str] = {}
    for edge in state.in_edges(node):
        key = source_key(state, edge, frame)
        if key is not None and edge.dst_conn:
            names[edge.dst_conn] = key
    for edge in state.out_edges(node):
        key = sink_key(state, edge, frame)
        if key is not None and edge.src_conn:
            names[edge.src_conn] = key
    env: dict[str, Interval] = {}
    for inner, outer in node.symbol_mapping.items():
        span = corners(outer, frame.env)
        if span is not None:
            env[str(inner)] = span
    count_sdfg(node.sdfg, Frame(frame.count, env, names, f"{frame.prefix}{node.label}/"), ledger)


def count_region(region: ControlFlowRegion, frame: Frame, ledger: Ledger) -> None:
    for block in region.nodes():
        if isinstance(block, SDFGState):
            count_state(block, frame, ledger)
        elif isinstance(block, LoopRegion):
            count_region(block, loop_frame(block, frame), ledger)
        elif isinstance(block, ConditionalBlock):
            for _, body in block.branches:
                count_region(body, frame, ledger)
        elif isinstance(block, ControlFlowRegion):
            count_region(block, frame, ledger)


def count_sdfg(sdfg: dace.SDFG, frame: Frame, ledger: Ledger) -> None:
    """Record the size of every container ``sdfg`` owns, then every writer in it."""
    for name, desc in sdfg.arrays.items():
        key = frame.key(name)
        if key in ledger.sizes:
            continue
        ledger.sizes[key] = largest(desc.total_size, frame.env)
        if desc.transient:
            ledger.transients[key] = None
    count_region(sdfg, frame, ledger)


def own_length(ledger: Ledger, key: str, fallback: int) -> int:
    """``l`` of ``key`` from its own writers: the accumulating writes (a WCR, or a write that reads ``key``) over the
    chains they extend, at least one per element and one per initialization of the cycle the value goes round
    (:meth:`Ledger.starts`); at least 1."""
    writes = ledger.writes.get(key, [])
    chained = [write for write in writes if write.accumulates or ledger.feeds(write, key)]
    if not chained:
        return 1
    if any(write.count is None for write in chained):
        return fallback
    chains = max(ledger.sizes.get(key) or 0, ledger.starts(key), 1)
    return max(1, math.ceil(sum(write.count or 0 for write in chained) / chains))


def accumulation_lengths(
    sdfg: dace.SDFG, sizes: Mapping[str, SymbolValue], arrays: Mapping[str, np.ndarray], outputs: Sequence[str]
) -> dict[str, int]:
    """``l`` for each of ``outputs`` of the kernel ``sdfg`` at ``sizes``.

    :param arrays: The kernel's arrays at ``sizes`` (inputs and outputs), by top-level name; their sizes stand in for
        the declared ones, and the largest is the fallback length."""
    env = {name: (int(value), int(value)) for name, value in sizes.items() if isinstance(value, (int, np.integer))}
    ledger = Ledger(sizes={name: int(array.size) for name, array in arrays.items()})
    count_sdfg(sdfg, Frame(1, env, {name: name for name in sdfg.arrays}, ""), ledger)
    fallback = max((int(array.size) for array in arrays.values()), default=1)
    lengths = {key: own_length(ledger, key, fallback) for key in ledger.writes}
    # a container inherits the length of the scratch it is computed from; each round reaches one step further
    for _ in range(len(lengths)):
        changed = False
        for key in lengths:
            read = ledger.reads(key)
            inherited = max((lengths[r] for r in read if r in ledger.transients and r in lengths), default=1)
            if inherited > lengths[key]:
                lengths[key], changed = inherited, True
        if not changed:
            break
    return {name: lengths.get(name, 1) for name in outputs}
