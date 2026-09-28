# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The kernel DAG: for every ``ExternalCall`` argument, the producers whose value can reach it (the program, a
kernel output, or a host state) and the loops it is carried around. Whole containers only: a read reads all of
it, a write replaces all of it."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Literal

import dace
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import (
    BreakBlock,
    ConditionalBlock,
    ContinueBlock,
    ControlFlowBlock,
    ControlFlowRegion,
    LoopRegion,
    ReturnBlock,
    SDFGState,
)
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.analysis.analysis import names_read_by_text

from nestforge.ir.libnode import ExternalCall, in_conn, out_conn
from nestforge.ir.names import in_order

INPUT_PREFIX = in_conn("")
OUTPUT_PREFIX = out_conn("")

#: Why a kernel reads an argument: a data input or a symbol.
Role = Literal["input", "symbol"]


class UnsupportedProgram(Exception):
    """The program holds a construct whose producers cannot be named at container level."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class Producer:
    """Who wrote a value: ``program`` (never written), ``kernel`` (an ``ExternalCall`` output) or ``host``."""

    kind: Literal["program", "kernel", "host"]
    name: str = ""
    arg: str = ""

    def label(self) -> str:
        """``program``, ``<kernel>.<arg>`` or ``host:<state>``."""
        return {"kernel": f"{self.name}.{self.arg}", "host": f"host:{self.name}"}.get(self.kind, self.kind)


@dataclass(frozen=True, slots=True)
class Fact:
    """The producers reaching a name, and the loops whose back edge one of them crossed."""

    producers: tuple[Producer, ...] = ()
    carried: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ArgEdge:
    """The producers reaching one argument of one consumer (a kernel, or ``exit``)."""

    consumer: str
    arg: str
    role: Role
    producers: tuple[Producer, ...]
    carried: tuple[str, ...] = ()

    def text(self) -> str:
        """``arg <- p0 | p1``, with ``[carried: <loops>]`` when a value crossed a loop's back edge."""
        carried = f" [carried: {' > '.join(self.carried)}]" if self.carried else ""
        return f"{self.arg} <- {' | '.join(self.labels()) or 'none'}{carried}"

    def labels(self) -> list[str]:
        """The producer labels, in order."""
        return [producer.label() for producer in self.producers]

    def loops(self) -> list[str]:
        """The loops whose back edge a reaching value crossed."""
        return list(self.carried)

    def to_json(self) -> dict[str, object]:
        """A plain, key-ordered dictionary."""
        return {
            "consumer": self.consumer,
            "arg": self.arg,
            "role": self.role,
            "producers": self.labels(),
            "carried": self.loops(),
        }


@dataclass(frozen=True, slots=True)
class KernelGraph:
    """Kernels in program order, their argument edges, and the producers of each program output at exit."""

    kernels: tuple[str, ...]
    edges: tuple[ArgEdge, ...]
    exits: tuple[ArgEdge, ...]

    def consumers_of(self, kernel: str) -> tuple[ArgEdge, ...]:
        """Every kernel argument edge that ``kernel`` can produce."""
        return tuple(e for e in self.edges if any(p.kind == "kernel" and p.name == kernel for p in e.producers))

    def arguments(self, kernel: str) -> tuple[ArgEdge, ...]:
        """The argument edges of ``kernel``, inputs then symbols."""
        return tuple(edge for edge in self.edges if edge.consumer == kernel)

    def line(self, kernel: str) -> str:
        """``kernel: arg <- producers, ...``."""
        return f"{kernel}: " + ", ".join(edge.text() for edge in self.arguments(kernel))

    def lines(self) -> list[str]:
        """One :meth:`line` per kernel, then one ``exit:`` line."""
        exits = ["exit: " + ", ".join(edge.text() for edge in self.exits)] if self.exits else []
        return [self.line(kernel) for kernel in self.kernels] + exits

    def to_json(self) -> dict[str, object]:
        """A plain dictionary in the graph's order."""
        return {
            "kernels": list(self.kernels),
            "edges": [edge.to_json() for edge in self.edges],
            "exits": [edge.to_json() for edge in self.exits],
        }


#: What reaches each container or symbol name at one program point; an absent name has no producer.
Env = dict[str, Fact]
PROGRAM = Fact((Producer("program"),))


def merge(*facts: Fact) -> Fact:
    producers = dict.fromkeys(p for fact in facts for p in fact.producers)
    carried = dict.fromkeys(loop for fact in facts for loop in fact.carried)
    return Fact(tuple(sorted(producers, key=Producer.label)), tuple(carried))


def join(*envs: Env | None) -> Env | None:
    present = [env for env in envs if env is not None]
    if not present:
        return None
    return functools.reduce(lambda a, b: {n: merge(a.get(n, Fact()), b.get(n, Fact())) for n in {**a, **b}}, present)


@dataclass(frozen=True, slots=True)
class Outcome:
    """The environments leaving a block normally and through a break, a continue or a return."""

    normal: Env | None = None
    breaks: Env | None = None
    continues: Env | None = None
    returns: Env | None = None


@dataclass(slots=True)
class Tracker:
    """Kernels in visit order and each ``(consumer, role, arg)`` edge; a loop's last visit is its fixpoint."""

    kernels: dict[str, None]
    edges: dict[tuple[str, str, str], ArgEdge]
    written: dict[str, None]


def assign(env: Env, assignments: dict[str, str]) -> Env:
    env = dict(env)
    for name, text in assignments.items():  # in order: a later assignment reads an earlier one's target
        env[name] = merge(*(env[read] for read in sorted(names_read_by_text(text)) if read in env))
    return env


def kernel_symbols(node: ExternalCall) -> list[str]:
    """The symbol arguments of a kernel: its manifest's non-array inputs."""
    manifest = node.config  # pyright: ignore[reportAttributeAccessIssue]  # a DaCe property
    if not manifest:
        raise UnsupportedProgram(f"ExternalCall {node.label!r} has no manifest, so its symbol arguments are unknown")
    return sorted(arg for arg in manifest["input_args"] if arg not in manifest["array_args"])


def record(tracker: Tracker, consumer: str, arg: str, role: Role, fact: Fact) -> None:
    tracker.edges[(consumer, role, arg)] = ArgEdge(consumer, arg, role, fact.producers, fact.carried)


def state_flow(state: SDFGState, env: Env, tracker: Tracker) -> Env:
    """Writes of one state: a kernel output, a copy (which forwards its source), or any other host writer."""
    env = dict(env)
    facts: dict[int, Fact] = {}
    bindings: dict[int, None] = {}  # edges binding a view to what it views; they move no data
    roots: dict[int, str] = {}
    for node in state.data_nodes():
        if isinstance(node.desc(state.sdfg), dace.data.View):
            binding, root = sdutil.get_view_edge(state, node), sdutil.get_last_view_node(state, node)
            if binding is None or root is None:
                raise UnsupportedProgram(f"view {node.data!r} in state {state.label!r} binds no container")
            bindings[id(binding)] = None
            roots[id(node)] = root.data

    def source(edge: MultiConnectorEdge) -> Fact:
        if isinstance(edge.src, ExternalCall) and edge.src_conn and edge.src_conn.startswith(OUTPUT_PREFIX):
            return Fact((Producer("kernel", edge.src.label, edge.src_conn.removeprefix(OUTPUT_PREFIX)),))
        if isinstance(edge.src, nodes.AccessNode):
            return facts[id(edge.src)]
        return Fact((Producer("host", state.label),))

    for node in in_order(state):
        if isinstance(node, nodes.AccessNode):
            container = roots.get(id(node), node.data)
            writes = [e for e in state.in_edges(node) if not e.data.is_empty() and id(e) not in bindings]
            facts[id(node)] = merge(*map(source, writes)) if writes else env.get(container, Fact())
            if writes:
                env[container] = facts[id(node)]
                tracker.written[container] = None
        elif isinstance(node, ExternalCall):
            tracker.kernels[node.label] = None
            for edge in state.in_edges(node):
                if edge.dst_conn and edge.dst_conn.startswith(INPUT_PREFIX):
                    record(tracker, node.label, edge.dst_conn.removeprefix(INPUT_PREFIX), "input", source(edge))
            for name in kernel_symbols(node):
                record(tracker, node.label, name, "symbol", env.get(name, Fact()))
    return env


def region_flow(region: ControlFlowRegion, entry: Env, tracker: Tracker) -> Outcome:
    """Blocks in order, each entered with the join of the edges reaching it; a structured region has no cycle."""
    blocks = in_order(region)
    rank = {id(block): index for index, block in enumerate(blocks)}
    if any(rank[id(e.dst)] <= rank[id(e.src)] for e in region.edges()):
        raise UnsupportedProgram(f"region {region.label!r} has a cycle outside a loop region")
    leaving: dict[int, Env | None] = {}
    outcomes: list[Outcome] = []
    sinks: list[Env | None] = []
    for block in blocks:
        start = entry if block is region.start_block else None
        env = join(start, *(leaving.get(id(e)) for e in region.in_edges(block)))
        if env is None:
            continue
        outcome = block_flow(block, env, tracker)
        outcomes.append(outcome)
        for edge in region.out_edges(block):
            leaving[id(edge)] = None if outcome.normal is None else assign(outcome.normal, edge.data.assignments)
        if region.out_degree(block) == 0:
            sinks.append(outcome.normal)
    if not blocks:
        return Outcome(normal=entry)
    return Outcome(
        join(*sinks),
        join(*(o.breaks for o in outcomes)),
        join(*(o.continues for o in outcomes)),
        join(*(o.returns for o in outcomes)),
    )


def loop_flow(loop: LoopRegion, env: Env, tracker: Tracker) -> Outcome:
    """Iterate the body until the head stops growing; a producer arriving fresh over the back edge is carried."""
    init = loop_analysis.assignment_text(loop.init_statement, loop.loop_variable)
    update = loop_analysis.assignment_text(loop.update_statement, loop.loop_variable)
    incoming = env if init is None else assign(env, {loop.loop_variable: init})
    head = incoming
    while True:
        body = region_flow(loop, head, tracker)
        latch = join(body.normal, body.continues)
        if latch is not None and update is not None:
            latch = assign(latch, {loop.loop_variable: update})
        back: Env = {}
        for name, fact in (latch or {}).items():
            fresh = tuple(p for p in fact.producers if p not in incoming.get(name, Fact()).producers)
            if fresh:
                back[name] = Fact(fresh, (*fact.carried, loop.label))
        grown = join(incoming, back) or incoming
        if grown == head:
            break
        head = grown
    leaving = join(latch, body.breaks)
    if not loop_analysis.loop_provably_at_least_one_iteration(loop):
        leaving = join(leaving, incoming)
    return Outcome(normal=leaving, returns=body.returns)


def block_flow(block: ControlFlowBlock, env: Env, tracker: Tracker) -> Outcome:
    if isinstance(block, SDFGState):
        return Outcome(normal=state_flow(block, env, tracker))
    if isinstance(block, BreakBlock):
        return Outcome(breaks=env)
    if isinstance(block, ContinueBlock):
        return Outcome(continues=env)
    if isinstance(block, ReturnBlock):
        return Outcome(returns=env)
    if isinstance(block, LoopRegion):
        return loop_flow(block, env, tracker)
    if isinstance(block, ConditionalBlock):
        outcomes = [region_flow(branch, env, tracker) for _, branch in block.branches]
        has_else = any(condition is None for condition, _ in block.branches)
        return Outcome(
            join(*(o.normal for o in outcomes), None if has_else else env),
            join(*(o.breaks for o in outcomes)),
            join(*(o.continues for o in outcomes)),
            join(*(o.returns for o in outcomes)),
        )
    if isinstance(block, ControlFlowRegion):
        return region_flow(block, env, tracker)
    raise UnsupportedProgram(f"control-flow block {block.label!r} ({type(block).__name__}) is not modeled")


def kernel_dependencies(sdfg: dace.SDFG) -> KernelGraph:
    """The producers reaching every ``ExternalCall`` argument of ``sdfg``, and every program output at exit.

    :param sdfg: The top-level SDFG holding the kernels; it is not modified.
    :returns: The kernel graph, in kernel program order, then role, then argument name.
    :raises UnsupportedProgram: On a ``Reference`` container, a view bound to no container, an ``ExternalCall``
        inside a nested SDFG or without a manifest, or a control-flow cycle outside a loop region.
    """
    for sd in sdfg.all_sdfgs_recursive():
        for name, desc in sd.arrays.items():
            if isinstance(desc, dace.data.Reference):
                raise UnsupportedProgram(f"container {name!r} of SDFG {sd.name!r} is a Reference, bound at run time")
        kernels = [n for state in sd.all_states() for n in state.nodes() if isinstance(n, ExternalCall)]
        if sd is not sdfg and kernels:
            raise UnsupportedProgram(f"ExternalCall {kernels[0].label!r} sits inside nested SDFG {sd.name!r}")
    tracker = Tracker({}, {}, {})
    names = [name for name, desc in sdfg.arrays.items() if not desc.transient] + sorted(sdfg.free_symbols)
    outcome = region_flow(sdfg, dict.fromkeys(names, PROGRAM), tracker)
    final = join(outcome.normal, outcome.returns) or {}
    order = {kernel: index for index, kernel in enumerate(tracker.kernels)}
    edges = sorted(tracker.edges.values(), key=lambda e: (order[e.consumer], e.role != "input", e.arg))
    exits = [
        ArgEdge("exit", name, "input", final.get(name, Fact()).producers, final.get(name, Fact()).carried)
        for name in sorted(tracker.written)
        if not sdfg.arrays[name].transient
    ]
    return KernelGraph(tuple(tracker.kernels), tuple(edges), tuple(exits))
