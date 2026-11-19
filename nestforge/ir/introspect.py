# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only views of a program: the structure tree (:func:`describe_graph`), a kernel as Python, and the arrays a
nest reads and writes."""

from __future__ import annotations

import ast
import contextlib
import copy
import functools
import re
import signal
import threading
from collections import Counter
from collections.abc import Callable, Collection, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import dace
import sympy
from dace import dtypes
from dace.frontend.operations import detect_reduction_type
from dace.frontend.python import astutils
from dace.libraries.standard.nodes.external_call import ExternalCall
from dace.sdfg import nodes
from dace.sdfg.graph import NodeNotFoundError
from dace.sdfg.performance_evaluation import work_depth
from dace.sdfg.performance_evaluation.helpers import get_uuid
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation.passes.analysis import loop_analysis

from nestforge.ir.dace_types import strings
from nestforge.ir.emit_c import body_c
from nestforge.ir.emit_python import UnsupportedNest, lower, map_python, module
from nestforge.ir.names import in_order, is_call, is_copy_tasklet, ordered_scope_children

#: Tree drawing: the guide under a node that has siblings below it, and the one under the last child.
TEE, ELBOW, PIPE, BLANK = "|- ", "`- ", "|  ", "   "

#: Marks a Python body line, so a statement is never mistaken for a tree row.
BODY = ": "
#: The languages a leaf body is printed in: Python (the default) or C, which shows C++ tasklets (intrinsics, inline
#: assembly) as written.
Language = Literal["python", "c"]
PYTHON: Language = "python"
C: Language = "c"
LANGUAGES: tuple[Language, ...] = (PYTHON, C)

#: Entries of each cache of parsed index and bound texts; a program repeats a few hundred.
CACHE_SIZE = 4096

#: What a ``Handle`` is asked to name: kind ``region`` for a control-flow block, ``nest`` for a map or library node.
Handle = Callable[[str, object], str]

#: Work and depth per block and node, keyed as DaCe's work-depth analysis keys them (``helpers.get_uuid``).
WorkDepth = dict[str, tuple[sympy.Expr, sympy.Expr]]

#: The line to print under a library node's row, or ``None`` for none.
Notes = Callable[[nodes.LibraryNode], str | None]

#: The view state of the tree: a state or map label -> ``True`` opened (with everything below it), ``False`` closed
#: although a row above it is open. The nearest labeled row decides; with none, a row is closed.
View = Mapping[str, bool]


def interstate_definitions(sdfg: dace.SDFG) -> dict[str, str]:
    """``name -> expression`` for every interstate assignment in the SDFG; a name assigned more than
    one distinct expression is dropped (which one reaches a block depends on the path taken)."""
    assigned: dict[str, set[str]] = {}
    # this SDFG's loops and branches only: a nested SDFG has its own symbol namespace
    for edge in sdfg.all_interstate_edges():
        for name, expression in edge.data.assignments.items():
            assigned.setdefault(name, set()).add(expression)
    return {name: exprs.pop() for name, exprs in assigned.items() if len(exprs) == 1}


def resolve_scalars(expression: str, definitions: dict[str, str]) -> str:
    """Fold scalar definitions into ``expression`` until only arrays, non-transients and free symbols
    are left -- ``A_index > 0.0`` becomes ``A[i + 1] > 0.0``. Each name is substituted at most once,
    so a cyclic definition (``i = i + 1`` on a back edge) terminates rather than expanding forever."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:  # a condition the frontend wrote in something other than python
        return expression
    remaining = dict(definitions)
    while remaining:
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} & set(remaining)
        if not used:
            break
        tree = astutils.ASTFindReplace({name: remaining.pop(name) for name in used}).visit(tree)
    # ast.unparse: this string is read, never parsed again
    return ast.unparse(simplify_indices(tree)).strip()


@functools.lru_cache(maxsize=CACHE_SIZE, typed=True)
def simplified_index(text: str) -> str:
    """Cached sympy round-trip for one subscript's unparsed slice text."""
    return str(dace.symbolic.simplify(dace.symbolic.pystr_to_symbolic(text)))


def simplify_indices(tree: ast.AST) -> ast.AST:
    """Rewrite every subscript index through sympy, so a hoisted read prints ``A[i + 1]`` rather than
    the ``A[(1 + (1 * i))]`` the frontend builds it as."""
    for node in [node for node in ast.walk(tree) if isinstance(node, ast.Subscript)]:
        try:
            node.slice = ast.parse(simplified_index(astutils.unparse(node.slice)), mode="eval").body
        except (SyntaxError, TypeError, AttributeError):
            continue  # an index sympy will not take is still perfectly printable as it stands
    return ast.fix_missing_locations(tree)


#: A live node's id -> its copy in a twin lowered for emission (:func:`lowered_twin`).
Twin = dict[int, Any]


def lowered_twin(sdfg: dace.SDFG) -> Twin:
    """A copy of ``sdfg`` lowered for Python emission, found by the ids of the live states and nodes; emission never
    touches the live program."""
    twin = copy.deepcopy(sdfg)
    pairs: Twin = {id(a): b for a, b in zip(sdfg.states(), twin.states())}
    pairs |= {id(pair[0]): copied[0] for pair, copied in zip(sdfg.all_nodes_recursive(), twin.all_nodes_recursive())}
    lower(twin, expand=False)
    return pairs


#: A body line that only renames a symbol (``_loop_it_1 = _loop_it_0``, what a fusion leaves), in Python or C.
RENAME = re.compile(r"^(?P<name>[A-Za-z_]\w*) = (?P<value>[A-Za-z_]\w*);?$")


def container_names(sdfg: dace.SDFG) -> frozenset[str]:
    """Every container declared in ``sdfg``'s whole program, nested ones included."""
    return frozenset(name for sd in sdfg.root_sdfg.all_sdfgs_recursive() for name in sd.arrays)


def program_names(sdfg: dace.SDFG) -> frozenset[str]:
    """Every name the whole program declares: containers, symbols, map parameters and loop variables."""
    root = sdfg.root_sdfg
    names = set(container_names(sdfg))
    for sd in root.all_sdfgs_recursive():
        names |= sd.symbols.keys()
    for node, _ in root.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry):
            names.update(strings(node.map.params))
    for block in root.all_control_flow_regions(recursive=True):
        if isinstance(block, LoopRegion) and block.loop_variable:
            names.add(block.loop_variable)
    return frozenset(names)


def folded_renames(lines: list[str], containers: frozenset[str]) -> list[str]:
    """``lines`` without the top-level lines that only give a symbol a second name: each later use reads the first
    name instead. A container copy (``a = b`` of data) stays, and so does a name assigned more than once."""
    assigned = Counter(
        match["name"]
        for line in lines
        for match in re.finditer(r"(?:^|\bfor )\s*(?P<name>[A-Za-z_]\w*)\s*(?:=[^=]| in )", line.strip())
    )
    renames: dict[str, str] = {}
    kept: list[str] = []
    for line in lines:
        for name, value in renames.items():
            line = re.sub(rf"\b{name}\b", value, line)
        match = RENAME.match(line)
        if (
            match is not None
            and assigned[match["name"]] == 1
            and not {match["name"], match["value"]} & containers
            and match["name"] != match["value"]
        ):
            renames[match["name"]] = match["value"]
            continue
        kept.append(line)
    return kept


#: An identifier in a printed body.
IDENTIFIER = re.compile(r"\b[A-Za-z_]\w*\b")
#: A statement's own local name that DaCe made up: a tasklet connector such as ``B_slice`` or ``__in1``.
GENERATED_LOCAL = re.compile(r"^_|__|_slice")
#: The frontend's local for a read of an index array: ``<program>_<array>_index``.
INDEX_LOCAL = re.compile(r"^\w+?_(?P<array>[A-Za-z][A-Za-z0-9]*)_index$")
#: The name a body's generated locals take, numbered per body: ``v<n>``.
LOCAL = "v"


def shortened_locals(lines: list[str], declared: frozenset[str]) -> list[str]:
    """``lines`` with each local name DaCe made up (never a name the program declares) shortened: an index read to
    ``<array>_index`` unless another name holds that, any other to ``v<n>``, numbered in order of first use within the
    body."""
    used = {name for line in lines for name in IDENTIFIER.findall(line)}
    renames: dict[str, str] = {}
    count = 0
    for line in lines:
        for name in IDENTIFIER.findall(line):
            if name in renames or name in declared:
                continue
            index = INDEX_LOCAL.match(name)
            if index is not None and f"{index['array']}_index" not in used | declared:
                renames[name] = f"{index['array']}_index"
            elif index is not None or GENERATED_LOCAL.search(name):
                while f"{LOCAL}{count}" in used | declared:
                    count += 1
                renames[name] = f"{LOCAL}{count}"
                count += 1
            else:
                continue
            used.add(renames[name])
    if not renames:
        return lines
    return [IDENTIFIER.sub(lambda m: renames.get(m.group(0), m.group(0)), line) for line in lines]


def body_lines(twin: Twin, state: SDFGState, entry: nodes.MapEntry) -> list[str]:
    """What one iteration of ``entry`` computes, as Python; an emitter refusal becomes the line's text."""
    try:
        return map_python(twin[id(state)], twin[id(entry)], body_only=True)
    except UnsupportedNest as exc:
        return [f"<not emitted: {exc}>"]


def kernel_source(state: SDFGState, sdfg: dace.SDFG, entry: nodes.MapEntry) -> str:
    """One kernel as a runnable Python module: a ``def`` over the arrays it touches, then every symbol its scope
    reads, each sorted."""
    reads, writes = nest_reads_writes(state, entry)
    arrays = sorted(set(reads) | set(writes))
    symbols = sorted({str(sym) for sym in state.scope_subgraph(entry).free_symbols} - set(arrays))
    twin = lowered_twin(sdfg)
    return module(entry.map.label, arrays + symbols, map_python(twin[id(state)], twin[id(entry)]))


#: ``ReductionType`` -> how the tree spells it; anything absent renders its lowercased enum name.
REDUCTION_SPELLING = {dtypes.ReductionType.Sum: "+", dtypes.ReductionType.Product: "*"}


def kernel_reductions(state: SDFGState, entry: nodes.MapEntry) -> list[str]:
    """Every accumulation leaving this map, as ``<target> <op>= over <axes>``: the reduced axes are the parameters
    the written subset does not mention. With none (each iteration updates its own element) it is ``<target> <op>=``
    alone."""
    exit_node = state.exit_node(entry)
    out: list[str] = []
    # normalization puts every WCR on an AccessNode -> MapExit edge
    for edge in state.in_edges(exit_node):
        if edge.data.wcr is None:
            continue
        kind = detect_reduction_type(edge.data.wcr)
        op = "?" if kind is None else REDUCTION_SPELLING.get(kind, kind.name.lower())
        subset = edge.data.subset
        written = subset.free_symbols if isinstance(subset, dace.subsets.Subset) else set()
        collapsed = [p for p in strings(entry.map.params) if p not in written]
        over = f" over {', '.join(collapsed)}" if collapsed else ""
        out.append(f"{edge.data.data} {op}={over}")
    return out


def nest_reads_writes(container: SDFGState | dace.SDFG, node: object) -> tuple[list[str], list[str]]:
    """Arrays a nest or library node reads and writes, without outlining it; ``container`` is the state of a
    ``MapEntry``, ``LibraryNode`` or one-element copy's tasklet."""
    if isinstance(node, (nodes.MapEntry, nodes.LibraryNode)) or is_copy_tasklet(node):
        assert isinstance(container, SDFGState), "a map or library node lives in a state"
        last = container.exit_node(node) if isinstance(node, nodes.MapEntry) else node
        reads = sorted({e.data.data for e in container.in_edges(node) if e.data.data})
        writes = sorted({e.data.data for e in container.out_edges(last) if e.data.data})
        return reads, writes
    if isinstance(node, (LoopRegion, SDFGState)):
        reads, writes = node.read_and_write_sets()
        return sorted(reads), sorted(writes)
    raise TypeError(f"not a nest node: {type(node).__name__}")


def map_domain(entry: nodes.MapEntry) -> str:
    """A map's iteration domain, ``i=0:N, j=0:M``."""
    return ", ".join(f"{p}={render_range(r)}" for p, r in zip(entry.map.params, entry.map.range))


def loop_domain(loop: LoopRegion, defs: dict[str, str]) -> str:
    """A loop's iteration domain, map-shaped, or its resolved condition for an uncounted ``while``."""
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if loop.loop_variable and start is not None and end is not None:
        return f"{loop.loop_variable}={render_range((start, end, stride if stride is not None else 1))}"
    return resolve_scalars(loop.loop_condition.as_string, defs) if loop.loop_condition is not None else ""


@functools.lru_cache(maxsize=CACHE_SIZE, typed=True)
def exclusive_end(end_text: str, delta: int) -> str:
    """``end + delta``, simplified; the same bound recurs across a program's kernels."""
    return str(dace.symbolic.simplify(cast(sympy.Expr, dace.symbolic.pystr_to_symbolic(end_text)) + delta))


def render_range(rng: tuple[Any, Any, Any]) -> str:
    """``begin:end:step`` as ``range`` reads it: an end one past the last value in the direction of travel, and
    a unit step left off."""
    begin, end, step = rng
    delta = -1 if dace.symbolic.pystr_to_symbolic(step).is_negative is True else 1
    text = f"{begin}:{exclusive_end(str(end), delta)}"
    return text if step == 1 else f"{text}:{step}"


#: A tree row: the block or node a label names, and the state holding it (``None`` for a block).
Row = tuple[Any, SDFGState | None]


def enclosing_rows(obj: object, state: SDFGState | None) -> list[str]:
    """The labels of the states and maps around a tree row, nearest first, across nested SDFGs."""
    found: list[str] = []
    while True:
        if state is not None:
            entry = state.entry_node(obj) if isinstance(obj, nodes.Node) else None
            while isinstance(entry, nodes.MapEntry):
                found.append(entry.map.label)
                entry = state.entry_node(entry)
            found.append(state.label)
            obj, state = state, None
        region = obj.parent_graph if isinstance(obj, ControlFlowBlock) else None
        while region is not None and not isinstance(region, dace.SDFG):
            region = region.parent_graph
        if region is None or region.parent_nsdfg_node is None:
            return found
        obj, state = region.parent_nsdfg_node, region.parent


def tree_rows(sdfg: dace.SDFG) -> dict[str, Row]:
    """Every label a tree row can print -> ``(block or node, its state)``, the state ``None`` for a block. Covers
    conditional branches, maps and library nodes, nested SDFGs included."""
    rows: dict[str, Row] = {}
    # a ConditionalBlock's nodes() are its branches, so the recursive walk reaches them
    for cfg in sdfg.all_control_flow_regions(recursive=True):
        for block in cfg.nodes():
            rows[block.label] = (block, None)
            if isinstance(block, SDFGState):
                kernels = (n for n in block.nodes() if isinstance(n, nodes.MapEntry) or is_call(block, n))
                rows.update((n.map.label if isinstance(n, nodes.MapEntry) else n.label, (n, block)) for n in kernels)
    return rows


#: Wall-clock seconds DaCe's work-depth pass may take before the tree prints without costs.
WORK_DEPTH_BUDGET_S = 60
#: What the work-depth pass raises on a program it cannot cost: SymPy recursion on large nests, an unorderable NaN; an
#: unbounded loop below the top of an SDFG (looked up among the top-level blocks), a loop inside an uncounted one (no
#: range recorded for it), a WCR memlet whose subset is its other subset, SymPy's ``simplify`` on some sums of a tiled loop.
WORK_DEPTH_FAILURES = (
    RecursionError,
    ValueError,
    TypeError,
    ArithmeticError,
    NodeNotFoundError,
    KeyError,
    AttributeError,
    StopIteration,
)
#: Longest exception message kept as the reason costs were skipped.
SKIP_REASON_CHARS = 160


class WorkDepthTimeout(Exception):
    """The work-depth pass outlasted :data:`WORK_DEPTH_BUDGET_S`."""


@contextlib.contextmanager
def time_budget(seconds: int) -> Iterator[None]:
    """Raise :class:`WorkDepthTimeout` in the body after ``seconds``; only the main thread can be interrupted by a
    signal, so elsewhere the body runs unbounded."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def expire(signum: int, frame: object) -> None:
        raise WorkDepthTimeout

    previous = signal.signal(signal.SIGALRM, expire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


#: What sympy raises on an expression it cannot simplify or evaluate; the cost then prints as it is.
SYMPY_FAILURES = (TypeError, ValueError, AttributeError, NotImplementedError, RecursionError, ZeroDivisionError)

#: Significant digits of a cost evaluated at the session's sizes.
COST_DIGITS = 3


def cost_text(expr: object, facts: dace.symbolic.Facts, sizes: Mapping[str, object]) -> str:
    """``expr`` (a work or depth) simplified under ``facts``, then evaluated at ``sizes`` for the symbols it has
    values for, to :data:`COST_DIGITS` digits; symbols without a value (an argument such as an iteration count) stay."""
    cost = sympy.sympify(expr)
    with contextlib.suppress(*SYMPY_FAILURES):
        cost = dace.symbolic.simplify(cost, facts)
    values = {sym: sizes[sym.name] for sym in cost.free_symbols if isinstance(sym, sympy.Symbol) and sym.name in sizes}
    if not values:
        return str(cost)
    try:
        value = sympy.N(cost.subs(values).doit(), COST_DIGITS)
    except SYMPY_FAILURES:
        return str(cost)
    if value.is_number and value.is_finite and abs(float(value)) < 10**COST_DIGITS and float(value).is_integer():
        return str(int(value))  # a small whole count prints as one, never as 11.0 beside 2
    return str(value)


def guarded_work_depth(sdfg: dace.SDFG) -> tuple[WorkDepth, str | None]:
    """:func:`program_work_depth`, or no costs and why, when the pass raises or outlasts :data:`WORK_DEPTH_BUDGET_S`.
    Costs only annotate the tree, so a program DaCe cannot cost still gets one."""
    try:
        with time_budget(WORK_DEPTH_BUDGET_S):
            return program_work_depth(sdfg), None
    except WorkDepthTimeout:
        return {}, f"work-depth analysis took over {WORK_DEPTH_BUDGET_S} s"
    except WORK_DEPTH_FAILURES as err:
        return {}, f"work-depth analysis failed: {type(err).__name__}: {str(err)[:SKIP_REASON_CHARS]}"


def program_work_depth(sdfg: dace.SDFG) -> WorkDepth:
    """Work and depth of every loop, state, map and nested SDFG of ``sdfg``, by one pass of DaCe's analysis (which
    works on a copy)."""
    found: WorkDepth = {}
    # analyze_sdfg annotates its map as str -> Expr, but stores (work, depth) pairs
    work_depth.analyze_sdfg(sdfg, cast(dict[str, sympy.Expr], found), work_depth.get_tasklet_work_depth, [], False)
    return found


def describe_graph(
    sdfg: dace.SDFG,
    handle: Handle | None = None,
    bodies: bool = False,
    notes: Notes | None = None,
    expanded: View | Collection[str] | None = None,
    costs: WorkDepth | None = None,
    languages: Mapping[str, Language] | None = None,
) -> str:
    """The program as a text tree, one line per block or kernel; loop and map rows carry their work and depth.

    :param handle: Returns the id to print on a row.
    :param bodies: Also print what each leaf kernel of a shown state computes, as Python.
    :param notes: Returns a line to print under a library node's row.
    :param expanded: The :data:`View`: an opened state shows its kernels and maps as rows; an opened map shows
        everything below it (nested maps and states, leaf-kernel bodies) but what is closed in it, and the maps of an
        open state stay closed until opened themselves. A closed state is one row naming how many kernels it holds, a
        closed map one row naming its nested states. Bare labels are opened ones; ``None`` shows every content.
    :param costs: :func:`program_work_depth` of ``sdfg``, when the caller holds it already.
    :param languages: The language leaf bodies under a state or map row are printed in (:data:`LANGUAGES`): its own
        entry, else that of the nearest row around it, else Python.
    """
    header = f"SDFG '{sdfg.label}'"
    opened = (
        None if expanded is None else dict(expanded) if isinstance(expanded, Mapping) else dict.fromkeys(expanded, True)
    )
    twin = lowered_twin(sdfg) if bodies or any(opened.values() if opened else ()) else None
    known = program_work_depth(sdfg) if costs is None else costs
    tree = Tree(
        [header],
        handle,
        interstate_definitions(sdfg),
        twin,
        notes,
        opened,
        bodies,
        known,
        state_counts(sdfg),
        dict(languages or {}),
    )
    walk_regions(tree, sdfg, "")
    return "\n".join(tree.lines)


@dataclass(frozen=True, slots=True)
class Tree:
    """The lines rendered so far and what :func:`describe_graph` was asked to print on them."""

    lines: list[str]
    handle: Handle | None
    defs: dict[str, str]
    twin: Twin | None
    notes: Notes | None
    expanded: dict[str, bool] | None
    bodies: bool
    costs: WorkDepth
    #: How many states access each container, keyed by its SDFG's id and name.
    states_using: Counter[tuple[int, str]]
    #: Row label -> the language leaf bodies below it are printed in.
    languages: dict[str, Language]

    def stamp(self, text: str, kind: str, obj: object) -> str:
        """Prefix a line's body with its session id, when there is one to prefix."""
        return f"[{self.handle(kind, obj)}] {text}" if self.handle is not None else text

    def opens(self, label: str, inside: bool) -> bool:
        """Whether the row ``label`` shows its content: its own view entry, else that of the open row it is in."""
        return self.expanded is None or self.expanded.get(label, inside)

    def with_bodies(self, shown: bool) -> bool:
        """Leaf-kernel bodies are printed when asked for, and always under an open row."""
        return self.twin is not None and (self.bodies or (self.expanded is not None and shown))

    def cost(self, key: str) -> str:
        found = self.costs.get(key)
        return "" if found is None else f"  work={found[0]} depth={found[1]}"


def walk_regions(
    tree: Tree, cfg: dace.SDFG | ControlFlowRegion, prefix: str, inside: bool = False, language: Language = PYTHON
) -> None:
    """Render one CFG's blocks under ``prefix``, recursing; ``inside``: an open row holds them; ``language``: the one
    the row around them prints bodies in."""
    blocks = in_order(cfg)
    for index, block in enumerate(blocks):
        spoken = tree.languages.get(block.label, language)
        last = index == len(blocks) - 1
        shown = isinstance(block, SDFGState) and tree.opens(block.label, inside)
        tree.lines.append(
            prefix + (ELBOW if last else TEE) + tree.stamp(block_row(tree, block, shown), "region", block)
        )
        below = prefix + (BLANK if last else PIPE)
        if isinstance(block, SDFGState):
            if shown:
                walk_state(tree, block, below, spoken)
        elif isinstance(block, ConditionalBlock):
            walk_branches(tree, block, below, inside, spoken)
        elif isinstance(block, ControlFlowRegion):
            walk_regions(tree, block, below, inside, spoken)


def walk_branches(tree: Tree, block: ConditionalBlock, prefix: str, inside: bool, language: Language) -> None:
    """A conditional's branches, in stored order (the first matching one wins, so that is execution order)."""
    for index, (condition, branch) in enumerate(block.branches):
        last = index == len(block.branches) - 1
        tag = "else" if condition is None else f"when {resolve_scalars(condition.as_string, tree.defs)}"
        tree.lines.append(prefix + (ELBOW if last else TEE) + tree.stamp(f"{branch.label}  {tag}", "region", branch))
        walk_regions(tree, branch, prefix + (BLANK if last else PIPE), inside, language)


def walk_state(tree: Tree, state: SDFGState, prefix: str, language: Language) -> None:
    """A state's kernels: every map nest plus any library node, nested scopes recursed into."""
    children = ordered_scope_children(state)
    if not any(isinstance(n, nodes.MapEntry) or is_call(state, n) for n in children[None]):
        return

    def descend(scope: nodes.MapEntry | None, pad: str, inside: bool, spoken: Language) -> None:
        kernels = [n for n in children[scope] if isinstance(n, nodes.MapEntry) or is_call(state, n)]
        for index, node in enumerate(kernels):
            last = index == len(kernels) - 1
            below = pad + (BLANK if last else PIPE)
            text = kernel_line(state, node)
            if isinstance(node, nodes.MapEntry):
                # the nests directly in this map: one inside a map in it is printed under that map
                nested = [child for child in children[node] if isinstance(child, nodes.NestedSDFG)]
                text += f"  num_states={sum(state_count(n.sdfg) for n in nested)}" if nested else ""
                text += tree.cost(get_uuid(node, state))
            tree.lines.append(pad + (ELBOW if last else TEE) + tree.stamp(text, "nest", node))
            tree.lines.extend(note_lines(node, below, tree.notes))
            if isinstance(node, nodes.MapEntry):
                opened = tree.opens(node.map.label, inside)
                own = tree.languages.get(node.map.label, spoken)
                leaf = not any(isinstance(child, nodes.MapEntry) for child in children[node])
                if leaf and tree.with_bodies(opened):
                    assert tree.twin is not None, "with_bodies holds only with a lowered twin"
                    body = body_c(state, node) if own == C else body_lines(tree.twin, state, node)
                    containers = container_names(state.sdfg)
                    body = shortened_locals(folded_renames(body, containers), program_names(state.sdfg))
                    tree.lines.extend(below + BODY + line for line in body)
                descend(node, below, opened, own)
                if nested and tree.expanded is not None and opened:
                    for inner in nested:
                        walk_regions(tree, inner.sdfg, below, inside=True, language=own)

    # an open state shows its doall rows, not their bodies: a doall shows its statements and nested states only once
    # it is opened itself (a doall nested in it opens with it)
    descend(None, prefix, False, language)


def note_lines(node: nodes.Node, below: str, notes: Notes | None) -> list[str]:
    """The line ``notes`` gives a library node, under its row; nothing for any other node."""
    if notes is None or not isinstance(node, nodes.LibraryNode):
        return []
    note = notes(node)
    return [] if note is None else [below + note]


def state_count(region: ControlFlowRegion) -> int:
    """The states anywhere in ``region``'s control flow, nested SDFGs included."""
    return sum(isinstance(b, SDFGState) for b in region.all_control_flow_blocks(recursive=True))


def innermost(loop: LoopRegion) -> bool:
    """Whether ``loop`` holds no other loop in its control flow."""
    return not any(isinstance(b, LoopRegion) for b in loop.all_control_flow_blocks(recursive=True))


def doall_count(state: SDFGState) -> int:
    """The top-level doalls of ``state``: maps, and library calls, each a black-box doall."""
    return sum(
        (isinstance(n, nodes.MapEntry) or is_call(state, n)) and state.entry_node(n) is None for n in state.nodes()
    )


def block_row(tree: Tree, block: ControlFlowBlock, shown: bool) -> str:
    """A block's row: :func:`block_line`, then its size when it is collapsed or innermost, then its cost; a state's
    also the arrays it reads and writes, as a doall's row."""
    text = block_line(block, tree.defs)
    if isinstance(block, SDFGState):
        count = doall_count(block)
        reads, writes = (
            [name for name in names if outlives(block, name, tree.states_using)]
            for names in nest_reads_writes(block, block)
        )
        if not (reads or writes):
            return text
        size = f"  doalls={count}" if count and not shown else ""
        return f"{text}{size}  reads={reads} writes={writes}" + tree.cost(get_uuid(block))
    if isinstance(block, LoopRegion):
        if innermost(block):
            text += f"  num_states={state_count(block)}"
        return text + tree.cost(get_uuid(block))
    return text


def state_counts(sdfg: dace.SDFG) -> Counter[tuple[int, str]]:
    """How many states access each container of ``sdfg`` and its nested SDFGs."""
    return Counter(
        (id(state.sdfg), name)
        for state in sdfg.all_control_flow_blocks(recursive=True)
        if isinstance(state, SDFGState)
        for name in {n.data for n in state.data_nodes()}
    )


def outlives(state: SDFGState, name: str, states_using: Counter[tuple[int, str]]) -> bool:
    """Whether ``name`` is more than a temporary of ``state``: an argument, or a transient another state uses."""
    desc = state.sdfg.arrays.get(name)
    return desc is None or not desc.transient or states_using[(id(state.sdfg), name)] > 1


def block_line(block: ControlFlowBlock, defs: dict[str, str]) -> str:
    """One control-flow block's line: its canonical label, plus its domain or condition."""
    if isinstance(block, LoopRegion):
        domain = loop_domain(block, defs)
        return f"{block.label}  {domain}" if domain else block.label
    return block.label


def kernel_line(state: SDFGState, node: nodes.MapEntry | nodes.CodeNode) -> str:
    """One kernel's line: label, iteration domain, and the arrays it reads and writes."""
    reads, writes = nest_reads_writes(state, node)
    if is_copy_tasklet(node):  # a one-element copy: the Copy node as its automatic expansion makes it
        return f"{node.label}  LIBNODE Copy [tasklet]  reads={reads} writes={writes}"
    if isinstance(node, nodes.LibraryNode):
        kind = type(node).__name__.removesuffix("LibraryNode")  # CopyLibraryNode reads as Copy, like its label
        chosen = "" if isinstance(node, ExternalCall) else f" [{node.implementation or 'auto'}]"
        return f"{node.label}  LIBNODE {kind}{chosen}  reads={reads} writes={writes}"
    reductions = kernel_reductions(state, node)
    folds = f"  accum=({'; '.join(reductions)})" if reductions else ""
    return f"{node.map.label}  [{map_domain(node)}]{folds}  reads={reads} writes={writes}"
