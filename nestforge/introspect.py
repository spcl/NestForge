"""Read-only structure inspection for the agent (and the deterministic path).

Two views, both non-mutating -- safe to call at any point in a session:
  * :func:`describe_graph` -- the SDFG as an ASCII TREE: nested regions/loops/conditionals, each State
    (a fusion barrier), each map-nest with its normalized iteration domain and read/write arrays.
  * :func:`nest_reads_writes` -- the arrays a single nest reads and writes, without extracting it.

The tree is the agent's whole view of the program, so it is projected from the normal form
(:mod:`nestforge.normalize`): every line names a block or kernel by its canonical
``<kind><level>_<index>`` label, and every loop and map shows a ``0:trip:1`` domain. Pass ``handle``
to also stamp each actionable line with a session id, so READING the tree and ACTING on it use one
vocabulary rather than two views the agent has to join by eyeballing labels.

Sibling States are a control-flow dependency: maps in different States cannot fuse (see
:func:`nestforge.fusion_arms.can_fuse`). The tree makes that structure visible so the agent knows which
nests are even fusion candidates before it asks.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import dace
from dace.sdfg import nodes
from dace.sdfg.state import (BreakBlock, ConditionalBlock, ContinueBlock, ControlFlowRegion, LoopRegion, ReturnBlock,
                             SDFGState)
from dace.transformation.passes.analysis import loop_analysis

from nestforge.normalize import in_order
from nestforge.strategies import is_parallel_nest

#: Tree drawing: the guide under a node that has siblings below it, and the one under the last child.
TEE, ELBOW, PIPE, BLANK = "|- ", "`- ", "|  ", "   "

#: What a ``Handle`` is asked to name. ``region`` covers every control-flow block, ``nest`` every map.
Handle = Callable[[str, object], str]


def nest_reads_writes(container: SDFGState, node: nodes.Node) -> Tuple[List[str], List[str]]:
    """Arrays a nest reads and writes (the interface arrays), without outlining it. ``container`` is the
    ``SDFGState`` holding a ``MapEntry``; ignored for a ``LoopRegion`` (which carries its own states)."""
    if isinstance(node, nodes.MapEntry):
        exit_node = container.exit_node(node)
        reads = sorted({e.data.data for e in container.in_edges(node) if e.data is not None and e.data.data})
        writes = sorted({e.data.data for e in container.out_edges(exit_node) if e.data is not None and e.data.data})
        return reads, writes
    if isinstance(node, LoopRegion):
        reads, writes = node.read_and_write_sets()
        return sorted(reads), sorted(writes)
    raise TypeError(f"not a nest node: {type(node).__name__}")


def map_domain(entry: nodes.MapEntry) -> str:
    """A map's iteration domain, ``i=0:N, j=0:M``. Normalized maps are zero-based and unit-stride, so a
    step only ever shows up when the caller skipped normalization -- and then it should show."""
    return ", ".join(f"{p}={render_range(r)}" for p, r in zip(entry.map.params, entry.map.range))


def loop_domain(loop: LoopRegion) -> str:
    """A loop's iteration domain in the same shape as a map's, or its raw condition when the loop is not
    a counted one (a ``while`` has no start/end to show)."""
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if loop.loop_variable and start is not None and end is not None:
        return f"{loop.loop_variable}={render_range((start, end, stride if stride is not None else 1))}"
    return loop.loop_condition.as_string if loop.loop_condition is not None else ""


def render_range(rng) -> str:
    """``begin:end:step`` with the two redundant parts dropped -- an inclusive end is rendered as the
    exclusive bound a reader expects, and a unit step is left off."""
    begin, end, step = rng
    text = f"{begin}:{dace.symbolic.simplify(end + 1)}"
    return text if step == 1 else f"{text}:{step}"


def describe_graph(sdfg: dace.SDFG, handle: Optional[Handle] = None) -> str:
    """The SDFG as an ASCII tree for the agent. Each line is one block or kernel; the guides show
    nesting. ``handle(kind, obj)``, when given, returns the session id to stamp on that line."""
    lines: List[str] = [f"SDFG '{sdfg.label}'"]
    walk_regions(sdfg, "", lines, handle)
    return "\n".join(lines)


def stamp(text: str, handle: Optional[Handle], kind: str, obj: object) -> str:
    """Prefix a line's body with its session id, when there is one to prefix."""
    return f"[{handle(kind, obj)}] {text}" if handle is not None else text


def walk_regions(cfg, prefix: str, lines: List[str], handle: Optional[Handle]) -> None:
    """Render one CFG's blocks under ``prefix``, recursing. ``prefix`` carries the guides of every
    ancestor, so a child knows whether to draw a pipe or a blank beneath each of them."""
    blocks = in_order(cfg)
    for index, block in enumerate(blocks):
        last = index == len(blocks) - 1
        lines.append(prefix + (ELBOW if last else TEE) + stamp(block_line(block), handle, "region", block))
        below = prefix + (BLANK if last else PIPE)
        if isinstance(block, SDFGState):
            walk_state(block, below, lines, handle)
        elif isinstance(block, ConditionalBlock):
            walk_branches(block, below, lines, handle)
        elif isinstance(block, ControlFlowRegion):
            walk_regions(block, below, lines, handle)


def walk_branches(block: ConditionalBlock, prefix: str, lines: List[str], handle: Optional[Handle]) -> None:
    """A conditional's branches. They are held in ``branches``, not as graph nodes, and the FIRST
    matching one wins -- so they are rendered in stored order, which is execution order."""
    for index, (condition, branch) in enumerate(block.branches):
        last = index == len(block.branches) - 1
        tag = "else" if condition is None else f"when {condition.as_string}"
        body = stamp(f"{branch.label}  {tag}", handle, "region", branch)
        lines.append(prefix + (ELBOW if last else TEE) + body)
        walk_regions(branch, prefix + (BLANK if last else PIPE), lines, handle)


def walk_state(state: SDFGState, prefix: str, lines: List[str], handle: Optional[Handle]) -> None:
    """A state's kernels: every map nest, plus any library node (which is a kernel that never became a
    map). Nested scopes recurse, so an inner map is shown under the map that encloses it."""
    children = state.scope_children()
    rank = {id(n): i for i, n in enumerate(in_order(state))}

    def descend(scope, pad: str) -> None:
        kernels = [
            n for n in sorted(children[scope], key=lambda n: rank.get(id(n), 0))
            if isinstance(n, (nodes.MapEntry, nodes.LibraryNode))
        ]
        for index, node in enumerate(kernels):
            last = index == len(kernels) - 1
            lines.append(pad + (ELBOW if last else TEE) + stamp(kernel_line(state, node), handle, "nest", node))
            if isinstance(node, nodes.MapEntry):
                descend(node, pad + (BLANK if last else PIPE))

    descend(None, prefix)


def block_line(block) -> str:
    """One control-flow block's line: its canonical label, then whatever the agent needs to act on it."""
    if isinstance(block, LoopRegion):
        domain = loop_domain(block)
        return f"{block.label}  {domain}" if domain else block.label
    if isinstance(block, SDFGState):
        return f"{block.label}  [fusion barrier]"
    if isinstance(block, ConditionalBlock):
        return f"{block.label}  [selector stays in core SDFG]"
    if isinstance(block, (BreakBlock, ContinueBlock, ReturnBlock)):
        return block.label
    return block.label


def kernel_line(state: SDFGState, node: nodes.Node) -> str:
    """One kernel's line: label, schedule, iteration domain, and the arrays it reads and writes."""
    if isinstance(node, nodes.LibraryNode):
        reads = sorted({e.data.data for e in state.in_edges(node) if e.data is not None and e.data.data})
        writes = sorted({e.data.data for e in state.out_edges(node) if e.data is not None and e.data.data})
        return f"{node.label}  LIBNODE  reads={reads} writes={writes}"
    reads, writes = nest_reads_writes(state, node)
    kind = "PARALLEL" if is_parallel_nest(node) else "SEQUENTIAL"
    return f"{node.map.label}  {kind}  [{map_domain(node)}]  reads={reads} writes={writes}"
