# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only views of a program: the structure tree (:func:`describe_graph`), a kernel as NumPy, and the arrays a
nest reads and writes."""

from __future__ import annotations

import ast
import functools
from dataclasses import dataclass
from typing import Any, cast
from collections.abc import Callable

import dace
import sympy
from dace import dtypes
from dace.frontend.operations import detect_reduction_type
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.frontend.python import astutils
from dace.transformation.passes.analysis import loop_analysis

from nestforge.ir.dace_types import bounds, memlet_subset, strings
from nestforge.ir.emit_libnode import UnsupportedLibraryNode
from nestforge.ir.emit_numpy import UnsupportedNest, map_body_lines, map_lines, standalone_source
from nestforge.ir.names import ScopeChildren, in_order, ordered_scope_children

#: Tree drawing: the guide under a node that has siblings below it, and the one under the last child.
TEE, ELBOW, PIPE, BLANK = "|- ", "`- ", "|  ", "   "

#: Marks a numpy body line, so a statement is never mistaken for a tree row.
BODY = ": "

#: What a ``Handle`` is asked to name: kind ``region`` for a control-flow block, ``nest`` for a map or library node.
Handle = Callable[[str, object], str]

#: The suffix appended to a top-level map's kernel line, when metrics are asked for.
Metrics = Callable[[nodes.MapEntry], str]

#: The line to print under a library node's row, or ``None`` for none.
Notes = Callable[[nodes.LibraryNode], str | None]


class Substitute(ast.NodeTransformer):
    """Replace each ``Name`` that has a definition with that definition's expression."""

    __slots__ = ("definitions",)

    def __init__(self, definitions: dict[str, str]) -> None:
        self.definitions = definitions

    def visit_Name(self, node: ast.Name) -> ast.AST:
        expression = self.definitions.get(node.id)
        return ast.parse(expression, mode="eval").body if expression is not None else node


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
        tree = Substitute({name: remaining.pop(name) for name in used}).visit(tree)
    # ast.unparse: this string is read, never parsed again
    return ast.unparse(simplify_indices(tree)).strip()


@functools.lru_cache(maxsize=4096, typed=True)
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


def kernel_body(state: SDFGState, sdfg: dace.SDFG, entry: nodes.MapEntry, children: ScopeChildren) -> list[str]:
    """The NumPy statements a leaf kernel computes; an emitter refusal becomes the line's text. ``children`` is the
    state's ``scope_children()``, built once by the caller."""
    if any(isinstance(node, nodes.MapEntry) for node in children[entry]):
        return []
    try:
        return map_body_lines(state, sdfg, entry)
    except (UnsupportedNest, UnsupportedLibraryNode) as exc:
        return [f"<not emitted: {exc}>"]


def kernel_source(state: SDFGState, sdfg: dace.SDFG, entry: nodes.MapEntry) -> str:
    """One kernel as a runnable NumPy module: a ``def`` over the arrays it touches, then every symbol its scope
    reads, each sorted, and the preamble its body calls."""
    reads, writes = nest_reads_writes(state, entry)
    arrays = sorted(set(reads) | set(writes))
    symbols = sorted({str(sym) for sym in state.scope_subgraph(entry).free_symbols} - set(arrays))
    return standalone_source(entry.map.label, arrays + symbols, map_lines(state, sdfg, entry))


#: ``ReductionType`` -> how the tree spells it; anything absent renders its lowercased enum name.
REDUCTION_SPELLING = {
    dtypes.ReductionType.Sum: "+",
    dtypes.ReductionType.Product: "*",
    dtypes.ReductionType.Min: "min",
    dtypes.ReductionType.Max: "max",
    dtypes.ReductionType.Sub: "-",
    dtypes.ReductionType.Div: "/",
    dtypes.ReductionType.Logical_And: "and",
    dtypes.ReductionType.Logical_Or: "or",
    dtypes.ReductionType.Logical_Xor: "xor",
    dtypes.ReductionType.Bitwise_And: "&",
    dtypes.ReductionType.Bitwise_Or: "|",
    dtypes.ReductionType.Bitwise_Xor: "^",
}


def kernel_reductions(state: SDFGState, entry: nodes.MapEntry) -> list[str]:
    """Every reduction leaving this map, as ``<op> over <axes> -> <target>``; the reduced axes are the parameters
    the written subset does not mention."""
    exit_node = state.exit_node(entry)
    out: list[str] = []
    # normalization puts every WCR on an AccessNode -> MapExit edge
    for edge in state.in_edges(exit_node):
        if edge.data.wcr is None:
            continue
        kind = detect_reduction_type(edge.data.wcr)
        op = "?" if kind is None else REDUCTION_SPELLING.get(kind, kind.name.lower())
        subset = memlet_subset(edge.data)
        written = {str(s) for r in (bounds(subset) if subset else []) for b in r for s in b.free_symbols}
        collapsed = [p for p in strings(entry.map.params) if p not in written]
        over = ", ".join(collapsed) if collapsed else "-"
        out.append(f"{op} over {over} -> {edge.data.data}")
    return out


def nest_reads_writes(container: SDFGState | dace.SDFG, node: object) -> tuple[list[str], list[str]]:
    """Arrays a nest or library node reads and writes, without outlining it; ``container`` is the state of a
    ``MapEntry`` or ``LibraryNode``."""
    if isinstance(node, (nodes.MapEntry, nodes.LibraryNode)):
        assert isinstance(container, SDFGState), "a map or library node lives in a state"
        last = container.exit_node(node) if isinstance(node, nodes.MapEntry) else node
        reads = sorted({e.data.data for e in container.in_edges(node) if e.data.data})
        writes = sorted({e.data.data for e in container.out_edges(last) if e.data.data})
        return reads, writes
    if isinstance(node, LoopRegion):
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


@functools.lru_cache(maxsize=4096, typed=True)
def exclusive_end(end_text: str, delta: int) -> str:
    """``end + delta``, simplified; the same bound recurs across a program's kernels."""
    return str(dace.symbolic.simplify(cast(sympy.Expr, dace.symbolic.pystr_to_symbolic(end_text)) + delta))


def render_range(rng: tuple[Any, Any, Any]) -> str:
    """``begin:end:step`` as ``range`` reads it: an end one past the last value in the direction of travel, and
    a unit step left off."""
    begin, end, step = rng
    delta = -1 if sympy.sympify(step).is_negative is True else 1
    text = f"{begin}:{exclusive_end(str(end), delta)}"
    return text if step == 1 else f"{text}:{step}"


#: A tree row: the block or node a label names, and the state holding it (``None`` for a block).
Row = tuple[Any, SDFGState | None]


def tree_rows(sdfg: dace.SDFG) -> dict[str, Row]:
    """Every label a tree row can print -> ``(block or node, its state)``, the state ``None`` for a block. Covers
    conditional branches, maps and library nodes, nested SDFGs included."""
    rows: dict[str, Row] = {}
    # a ConditionalBlock's nodes() are its branches, so the recursive walk reaches them
    for cfg in sdfg.all_control_flow_regions(recursive=True):
        for block in cfg.nodes():
            rows[block.label] = (block, None)
            if isinstance(block, SDFGState):
                kernels = (n for n in block.nodes() if isinstance(n, (nodes.MapEntry, nodes.LibraryNode)))
                rows.update((n.map.label if isinstance(n, nodes.MapEntry) else n.label, (n, block)) for n in kernels)
    return rows


def describe_graph(
    sdfg: dace.SDFG,
    handle: Handle | None = None,
    bodies: bool = False,
    metrics: Metrics | None = None,
    notes: Notes | None = None,
    epoch: int | None = None,
) -> str:
    """The program as a text tree, one line per block or kernel.

    :param handle: Returns the id to print on a row.
    :param bodies: Also print what each leaf kernel computes, as NumPy.
    :param metrics: Returns the suffix of a top-level map's row.
    :param notes: Returns a line to print under a library node's row.
    :param epoch: Printed on the first line.
    """
    header = f"SDFG '{sdfg.label}'" if epoch is None else f"SDFG '{sdfg.label}'  epoch={epoch}"
    tree = Tree([header], handle, interstate_definitions(sdfg), bodies, metrics, notes)
    walk_regions(tree, sdfg, "")
    return "\n".join(tree.lines)


@dataclass(frozen=True, slots=True)
class Tree:
    """The lines rendered so far and what :func:`describe_graph` was asked to print on them."""

    lines: list[str]
    handle: Handle | None
    defs: dict[str, str]
    bodies: bool
    metrics: Metrics | None
    notes: Notes | None

    def stamp(self, text: str, kind: str, obj: object) -> str:
        """Prefix a line's body with its session id, when there is one to prefix."""
        return f"[{self.handle(kind, obj)}] {text}" if self.handle is not None else text


def walk_regions(tree: Tree, cfg: dace.SDFG | ControlFlowRegion, prefix: str) -> None:
    """Render one CFG's blocks under ``prefix``, recursing."""
    blocks = in_order(cfg)
    for index, block in enumerate(blocks):
        last = index == len(blocks) - 1
        tree.lines.append(prefix + (ELBOW if last else TEE) + tree.stamp(block_line(block, tree.defs), "region", block))
        below = prefix + (BLANK if last else PIPE)
        if isinstance(block, SDFGState):
            walk_state(tree, block, below)
        elif isinstance(block, ConditionalBlock):
            walk_branches(tree, block, below)
        elif isinstance(block, ControlFlowRegion):
            walk_regions(tree, block, below)


def walk_branches(tree: Tree, block: ConditionalBlock, prefix: str) -> None:
    """A conditional's branches, in stored order (the first matching one wins, so that is execution order)."""
    for index, (condition, branch) in enumerate(block.branches):
        last = index == len(block.branches) - 1
        tag = "else" if condition is None else f"when {resolve_scalars(condition.as_string, tree.defs)}"
        tree.lines.append(prefix + (ELBOW if last else TEE) + tree.stamp(f"{branch.label}  {tag}", "region", branch))
        walk_regions(tree, branch, prefix + (BLANK if last else PIPE))


def walk_state(tree: Tree, state: SDFGState, prefix: str) -> None:
    """A state's kernels: every map nest plus any library node, nested scopes recursed into."""
    children = ordered_scope_children(state)
    if not any(isinstance(n, (nodes.MapEntry, nodes.LibraryNode)) for n in children[None]):
        return

    def descend(scope: nodes.MapEntry | None, pad: str) -> None:
        kernels = [n for n in children[scope] if isinstance(n, (nodes.MapEntry, nodes.LibraryNode))]
        for index, node in enumerate(kernels):
            last = index == len(kernels) - 1
            below = pad + (BLANK if last else PIPE)
            text = kernel_line(state, node)
            if tree.metrics is not None and scope is None and isinstance(node, nodes.MapEntry):
                text = f"{text}  {tree.metrics(node)}"
            tree.lines.append(pad + (ELBOW if last else TEE) + tree.stamp(text, "nest", node))
            tree.lines.extend(note_lines(node, below, tree.notes))
            if isinstance(node, nodes.MapEntry):
                if tree.bodies:
                    tree.lines.extend(below + BODY + line for line in kernel_body(state, state.sdfg, node, children))
                descend(node, below)

    descend(None, prefix)


def note_lines(node: nodes.Node, below: str, notes: Notes | None) -> list[str]:
    """The line ``notes`` gives a library node, under its row; nothing for any other node."""
    if notes is None or not isinstance(node, nodes.LibraryNode):
        return []
    note = notes(node)
    return [] if note is None else [below + note]


def block_line(block: ControlFlowBlock, defs: dict[str, str]) -> str:
    """One control-flow block's line: its canonical label, plus its domain or condition."""
    if isinstance(block, LoopRegion):
        domain = loop_domain(block, defs)
        return f"{block.label}  {domain}" if domain else block.label
    return block.label


def kernel_line(state: SDFGState, node: nodes.MapEntry | nodes.LibraryNode) -> str:
    """One kernel's line: label, iteration domain, and the arrays it reads and writes."""
    reads, writes = nest_reads_writes(state, node)
    if isinstance(node, nodes.LibraryNode):
        return f"{node.label}  LIBNODE  reads={reads} writes={writes}"
    reductions = kernel_reductions(state, node)
    folds = f"  reduce=({'; '.join(reductions)})" if reductions else ""
    return f"{node.map.label}  [{map_domain(node)}]{folds}  reads={reads} writes={writes}"
