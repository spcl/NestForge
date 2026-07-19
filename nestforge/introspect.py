"""Read-only structure inspection for the agent (and the deterministic path).

Two views, both non-mutating -- safe to call at any point in a session:
  * :func:`describe_graph` -- a control-flow-region tree of the SDFG: nested regions/loops/conditionals,
    each State (a fusion barrier), each top-level map-nest with its parallel flag + read/write arrays.
  * :func:`nest_reads_writes` -- the arrays a single nest reads and writes, without extracting it.

Sibling States are a control-flow dependency: maps in different States cannot fuse (see
:func:`nestforge.fusion_arms.can_fuse`). The tree makes that structure visible so the agent knows which
nests are even fusion candidates before it asks.
"""
from __future__ import annotations

from typing import List, Tuple

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowRegion, LoopRegion, SDFGState

from nestforge.offload import label_nest
from nestforge.strategies import is_parallel_nest, top_level_map_entries

try:
    from dace.sdfg.state import ConditionalBlock
except ImportError:  # older DaCe without first-class conditional regions
    ConditionalBlock = None


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


def describe_graph(sdfg: dace.SDFG) -> str:
    """A control-flow-region tree of ``sdfg`` as text for the agent. Each line is one block; indentation is
    nesting. States are marked ``[fusion barrier]``; a conditional's selector is marked as staying in the
    core SDFG (its branches are separate regions)."""
    lines: List[str] = [f"SDFG '{sdfg.label}'"]
    walk_regions(sdfg, 1, lines)
    return "\n".join(lines)


def walk_regions(cfg, depth: int, lines: List[str]) -> None:
    pad = "  " * depth
    for block in cfg.nodes():
        if isinstance(block, SDFGState):
            lines.append(f"{pad}state '{block.label}'  [fusion barrier]")
            for entry in top_level_map_entries(block):
                emit_map_line(block, entry, depth + 1, lines)
        elif isinstance(block, LoopRegion):
            lines.append(f"{pad}loop '{block.label}'  SEQUENTIAL")
            walk_regions(block, depth + 1, lines)
        elif ConditionalBlock is not None and isinstance(block, ConditionalBlock):
            lines.append(f"{pad}if '{block.label}'  [selector stays in core SDFG]")
            for cond, branch in block.branches:
                tag = "else" if cond is None else f"when {cond.as_string}"
                lines.append(f"{pad}  region [{tag}]")
                walk_regions(branch, depth + 2, lines)
        elif isinstance(block, ControlFlowRegion):
            lines.append(f"{pad}region '{block.label}'")
            walk_regions(block, depth + 1, lines)
        else:
            lines.append(f"{pad}{type(block).__name__} '{block.label}'")


def emit_map_line(state: SDFGState, entry: nodes.MapEntry, depth: int, lines: List[str]) -> None:
    pad = "  " * depth
    reads, writes = nest_reads_writes(state, entry)
    kind = "PARALLEL" if is_parallel_nest(entry) else "SEQUENTIAL"
    lines.append(f"{pad}map {label_nest(entry)}  {kind}  reads={reads} writes={writes}")
