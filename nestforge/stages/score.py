# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 2: a static score that ranks moves without building them.

``S = lines / p + c * E``, lower is better, evaluated on the SDFG at concrete sizes:

- ``E``: parallel-region entries, every top-level map times the trips of the loops around it (one fork or launch);
- ``lines``: cache lines the tasklets touch along their innermost iterator, ``volume * min(1, |stride| * bytes / 64)``
  per memlet, times the executions; an array of at most 1 MiB is fetched once (its size in lines);
- ``p``: workers that share the traffic, ``c``: the price of one entry in cache lines. Only ``k = c * p`` decides
  the sign of a change, and the frozen ``k`` is large, so ``E`` decides whenever it changes and ``lines`` breaks ties.

The weights were fitted on measured runtimes of TSVC and ``column_scan`` kernels and frozen before a test on new
move sequences; how, and where it fails: ``docs/score-design.md``.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from dataclasses import dataclass, field

import dace
import sympy
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.subsets import Range
from dace.transformation.passes.analysis import loop_analysis

from nestforge.ir.introspect import tree_rows
from nestforge.stages.moves import Rewrite, legal_moves, plan_move

#: Bytes in one cache line.
LINE_BYTES = 64.0
#: An array at most this large stays cached: its lines are fetched once, however often it is swept.
CACHE_BYTES = 2.0**20


@dataclass(frozen=True, slots=True)
class ScoreWeights:
    """``p`` (workers sharing the traffic) and ``c`` (one region entry, in cache lines) of one build."""

    workers: float
    entry_cost: float


#: Frozen weights per build: the largest ``k = c * p`` in the tie set of the fit.
WEIGHTS: dict[str, ScoreWeights] = {
    "cpu-serial": ScoreWeights(1.0, 1.0e4),
    "cpu-omp": ScoreWeights(32.0, 1.0e4),
    "gpu": ScoreWeights(2048.0, 1.0e4),
}


@dataclass(frozen=True, slots=True)
class ProgramTerms:
    """The two terms of one program; ``unresolved`` counts expressions left with a free symbol (counted as 1)."""

    entries: float
    lines: float
    unresolved: int

    def score(self, weights: ScoreWeights) -> float:
        return self.lines / weights.workers + weights.entry_cost * self.entries


def evaluate(expr: object, sizes: dict[str, float]) -> float | None:
    """``expr`` at ``sizes``, or ``None`` when a free symbol is left."""
    value = sympy.sympify(expr)
    value = value.subs({s: sizes[str(s)] for s in value.free_symbols if str(s) in sizes})
    return None if value.free_symbols else float(value)


def loop_trips(loop: LoopRegion, sizes: dict[str, float]) -> float | None:
    """Iterations of ``loop``, or ``None`` when a bound is unknown (also when it depends on an outer iterator)."""
    start, end = loop_analysis.get_init_assignment(loop), loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or stride is None:
        return None
    first, last, step = evaluate(start, sizes), evaluate(end, sizes), evaluate(stride, sizes)
    if first is None or last is None or step is None or step == 0:
        return None
    return max(0.0, (last - first) / step + 1.0)


# region entries


@dataclass(slots=True)
class Tally:
    value: float = 0.0
    unresolved: int = 0


def state_entries(state: SDFGState, mult: float, sizes: dict[str, float], tally: Tally) -> None:
    for node in state.nodes():
        if state.entry_node(node) is not None:
            continue
        if isinstance(node, nodes.MapEntry):
            tally.value += mult
        elif isinstance(node, nodes.NestedSDFG):
            region_entries(node.sdfg, mult, sizes, tally)


def region_entries(cfg: ControlFlowRegion, mult: float, sizes: dict[str, float], tally: Tally) -> None:
    """Add the region entries of ``cfg``, run ``mult`` times; a conditional counts its costliest branch."""
    for block in cfg.nodes():
        if isinstance(block, SDFGState):
            state_entries(block, mult, sizes, tally)
        elif isinstance(block, LoopRegion):
            trips = loop_trips(block, sizes)
            if trips is None:
                tally.unresolved += 1
            region_entries(block, mult * (trips if trips is not None else 1.0), sizes, tally)
        elif isinstance(block, ConditionalBlock):
            branches = []
            for _, region in block.branches:
                branch = Tally()
                region_entries(region, mult, sizes, branch)
                branches.append(branch)
            if branches:
                worst = max(branches, key=lambda b: b.value)
                tally.value += worst.value
                tally.unresolved += worst.unresolved
        elif isinstance(block, ControlFlowRegion):
            region_entries(block, mult, sizes, tally)


# cache lines


def maps_around(state: SDFGState, node: nodes.Node) -> list[nodes.MapEntry]:
    """Map entries enclosing ``node`` in its state, innermost first."""
    scopes = state.scope_dict()
    found: list[nodes.MapEntry] = []
    current = scopes.get(node)
    while current is not None:
        assert isinstance(current, nodes.MapEntry), "a scope is opened by a map entry"
        found.append(current)
        current = scopes.get(current)
    return found


def map_iterations(entries: list[nodes.MapEntry], sizes: dict[str, float]) -> tuple[float, int]:
    total, unresolved = 1.0, 0
    for entry in entries:
        n = evaluate(entry.map.range.num_elements(), sizes)
        if n is None:
            unresolved += 1
        else:
            total *= n
    return total, unresolved


def outer_iterations(sdfg: dace.SDFG, sizes: dict[str, float]) -> tuple[float, int]:
    """Product of the map ranges around the nested SDFG nodes that hold ``sdfg``, up to the program."""
    total, unresolved = 1.0, 0
    while sdfg.parent_nsdfg_node is not None and sdfg.parent is not None:
        state = sdfg.parent
        n, missing = map_iterations(maps_around(state, sdfg.parent_nsdfg_node), sizes)
        total, unresolved = total * n, unresolved + missing
        sdfg = state.sdfg
    return total, unresolved


def enclosing_trips(state: SDFGState, sizes: dict[str, float]) -> tuple[float, int]:
    """Product of the trips of the loops around ``state``, through nested SDFGs."""
    mult, unresolved = 1.0, 0
    graph: ControlFlowRegion | None = state.parent_graph
    while graph is not None:
        if isinstance(graph, LoopRegion):
            trips = loop_trips(graph, sizes)
            if trips is None:
                unresolved += 1
            else:
                mult *= trips
        if isinstance(graph, dace.SDFG):  # leave a nested SDFG through the state that holds it
            graph = graph.parent.parent_graph if graph.parent is not None else None
        else:
            graph = graph.parent_graph
    return mult, unresolved


def nearest_loop(state: SDFGState) -> LoopRegion | None:
    graph: ControlFlowRegion | None = state.parent_graph
    while graph is not None and not isinstance(graph, dace.SDFG):
        if isinstance(graph, LoopRegion) and graph.loop_variable:
            return graph
        graph = graph.parent_graph
    return None


@dataclass(frozen=True, slots=True)
class TaskletIterator:
    """The innermost iterator of a tasklet and the symbol mappings that lift an expression to its scope."""

    name: str
    lift: tuple[dict[str, str], ...]


def inherited_iterator(sdfg: dace.SDFG) -> TaskletIterator | None:
    """The iterator of the scope a nested SDFG sits in: the innermost map around its node, else the nearest loop,
    going up through the program; ``None`` when nothing iterates around it."""
    chain: list[dict[str, str]] = []
    while sdfg.parent_nsdfg_node is not None and sdfg.parent is not None:
        node, state = sdfg.parent_nsdfg_node, sdfg.parent
        chain.append({str(k): str(v) for k, v in node.symbol_mapping.items()})
        around = maps_around(state, node)
        if around:
            return TaskletIterator(str(around[0].map.params[-1]), tuple(chain))
        loop = nearest_loop(state)
        if loop is not None:
            return TaskletIterator(str(loop.loop_variable), tuple(chain))
        sdfg = state.sdfg
    return None


def tasklet_iterator(sdfg: dace.SDFG, state: SDFGState, around: list[nodes.MapEntry]) -> TaskletIterator | None:
    if around:
        return TaskletIterator(str(around[0].map.params[-1]), ())
    loop = nearest_loop(state)
    if loop is not None:
        return TaskletIterator(str(loop.loop_variable), ())
    return inherited_iterator(sdfg)


def lifted(expr: sympy.Basic, lift: tuple[dict[str, str], ...]) -> sympy.Basic:
    for mapping in lift:
        expr = expr.subs(
            {s: dace.symbolic.pystr_to_symbolic(mapping[str(s)]) for s in expr.free_symbols if str(s) in mapping}
        )
    return expr


def stride_elements(
    memlet: dace.Memlet, desc: dace.data.Data, it: TaskletIterator, sizes: dict[str, float]
) -> float | None:
    """Elements between the accesses of two consecutive iterations of ``it``."""
    if not isinstance(memlet.subset, Range):
        return None
    stride: sympy.Expr = sympy.Integer(0)
    for (begin, _, _), step in zip(memlet.subset.ranges, desc.strides):
        expr = lifted(sympy.sympify(begin), it.lift)
        hits = [s for s in expr.free_symbols if str(s) == it.name]  # DaCe symbols carry a dtype: match by name
        if hits:
            stride += sympy.diff(expr, hits[0]) * step
    return evaluate(stride, sizes)


@dataclass(slots=True)
class LineTally:
    streamed: float = 0.0
    resident: dict[tuple[dace.SDFG, str], float] = field(default_factory=dict)
    unresolved: int = 0


def tasklet_lines(
    sdfg: dace.SDFG, state: SDFGState, tasklet: nodes.Tasklet, runs: float, sizes: dict[str, float], tally: LineTally
) -> None:
    around = maps_around(state, tasklet)
    it = tasklet_iterator(sdfg, state, around)
    if it is None:
        return  # nothing iterates around it: a one-off scalar statement
    n, missing = map_iterations(around, sizes)
    runs, tally.unresolved = runs * n, tally.unresolved + missing
    for edge in [*state.in_edges(tasklet), *state.out_edges(tasklet)]:
        memlet = edge.data
        if memlet.data is None or memlet.subset is None:
            continue
        desc = sdfg.arrays[memlet.data]
        if desc.total_size == 1:
            continue
        stride = stride_elements(memlet, desc, it, sizes)
        if stride is None:
            tally.unresolved += 1
            continue
        nbytes = (evaluate(desc.total_size, sizes) or 0.0) * desc.dtype.bytes
        if 0 < nbytes <= CACHE_BYTES:
            tally.resident[sdfg, memlet.data] = nbytes / LINE_BYTES
        else:
            volume = evaluate(memlet.volume, sizes) or 1.0
            tally.streamed += runs * volume * min(1.0, abs(stride) * desc.dtype.bytes / LINE_BYTES)


def all_sdfgs(sdfg: dace.SDFG) -> Iterator[dace.SDFG]:
    yield sdfg
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.NestedSDFG):
            yield node.sdfg


def program_lines(sdfg: dace.SDFG, sizes: dict[str, float]) -> tuple[float, int]:
    tally = LineTally()
    for inner in all_sdfgs(sdfg):
        outer, missing = outer_iterations(inner, sizes)
        tally.unresolved += missing
        for state in (b for b in inner.all_control_flow_blocks() if isinstance(b, SDFGState)):
            trips, missing = enclosing_trips(state, sizes)
            tally.unresolved += missing
            for node in state.nodes():
                if isinstance(node, nodes.Tasklet):
                    tasklet_lines(inner, state, node, trips * outer, sizes, tally)
    return tally.streamed + sum(tally.resident.values()), tally.unresolved


def program_terms(sdfg: dace.SDFG, sizes: dict[str, int]) -> ProgramTerms:
    """``E`` and ``lines`` of ``sdfg`` at ``sizes`` (a value for every symbol of the loop bounds and shapes)."""
    values = {k: float(v) for k, v in sizes.items()}
    entries = Tally()
    region_entries(sdfg, 1.0, values, entries)
    lines, unresolved = program_lines(sdfg, values)
    return ProgramTerms(entries.value, lines, entries.unresolved + unresolved)


@dataclass(frozen=True, slots=True)
class ScoredMove:
    """One legal move and the change it makes to each term and to the score (negative: predicted faster)."""

    kind: str
    labels: tuple[str, ...]
    d_entries: float
    d_lines: float
    d_score: float


def build_name(gpu: bool, threads: int) -> str:
    """The :data:`WEIGHTS` key for a GPU target, or a CPU run on ``threads`` threads."""
    return "gpu" if gpu else ("cpu-omp" if threads > 1 else "cpu-serial")


def score_move(
    sdfg: dace.SDFG,
    kind: str,
    labels: tuple[str, ...],
    sizes: dict[str, int],
    base: ProgramTerms,
    weights: ScoreWeights,
) -> ScoredMove | None:
    """Apply the move to a copy of ``sdfg`` and score the copy against ``base``; ``None`` when it does not apply."""
    twin = copy.deepcopy(sdfg)
    rows = tree_rows(twin)
    if any(label not in rows for label in labels):
        return None
    plan = plan_move(kind, [rows[label] for label in labels])
    if not isinstance(plan, Rewrite):
        return None
    plan.commit()
    after = program_terms(twin, sizes)
    return ScoredMove(
        kind,
        labels,
        after.entries - base.entries,
        after.lines - base.lines,
        after.score(weights) - base.score(weights),
    )


def rank_moves(sdfg: dace.SDFG, sizes: dict[str, int], weights: ScoreWeights) -> list[ScoredMove]:
    """Every legal move of ``sdfg`` with its score change, best (most negative) first; no move is applied."""
    base = program_terms(sdfg, sizes)
    scored = [score_move(sdfg, kind, labels, sizes, base, weights) for kind, labels in legal_moves(sdfg)]
    return sorted((m for m in scored if m is not None), key=lambda m: m.d_score)
