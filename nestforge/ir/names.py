# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The labels the structure tree prints, ``<kind><level>_<index>`` and unique across the program, and the inlining
of top-level nested SDFGs that canonicalization runs first."""

from __future__ import annotations

import heapq
from typing import Any

import dace
from dace.libraries.standard.nodes.external_call import ExternalCall
from dace.sdfg import nodes
from dace.sdfg.state import (
    AbstractControlFlowRegion,
    BreakBlock,
    ConditionalBlock,
    ContinueBlock,
    ControlFlowBlock,
    LoopRegion,
    ReturnBlock,
    SDFGState,
)
from dace.transformation.interstate.expand_nested_sdfg_inputs import ExpandNestedSDFGInputs
from dace.transformation.interstate.multistate_inline import InlineMultistateSDFG
from dace.utils import find_new_name


def in_order(graph: AbstractControlFlowRegion | SDFGState) -> list[Any]:
    """A graph's nodes in topological order, ties broken by insertion order, so labels are deterministic."""
    all_nodes: list[Any] = list(graph.nodes())  # blocks of a region or nodes of a state
    rank = {id(n): i for i, n in enumerate(all_nodes)}
    indegree = {id(n): 0 for n in all_nodes}
    for edge in graph.edges():
        indegree[id(edge.dst)] += 1
    ready = [rank[id(n)] for n in all_nodes if indegree[id(n)] == 0]
    heapq.heapify(ready)
    ordered: list[Any] = []
    while ready:
        node = all_nodes[heapq.heappop(ready)]
        ordered.append(node)
        for edge in graph.out_edges(node):
            indegree[id(edge.dst)] -= 1
            if indegree[id(edge.dst)] == 0:
                heapq.heappush(ready, rank[id(edge.dst)])
    seen = {id(n) for n in ordered}
    return ordered + [n for n in all_nodes if id(n) not in seen]


# no top-level nested SDFG


def top_level_map_entries(state: SDFGState) -> list[nodes.MapEntry]:
    """The maps of ``state`` not nested in another map, in ``scope_children`` order (breadth first from the
    sources), which kernel names follow."""
    return [n for n in state.scope_children()[None] if isinstance(n, nodes.MapEntry)]


def top_level_nsdfgs(sdfg: dace.SDFG) -> list[tuple[SDFGState, nodes.NestedSDFG]]:
    """Every ``NestedSDFG`` outside all map scopes; one inside a map is a kernel body."""
    return [
        (state, node)
        for state in sdfg.states()
        for node in state.scope_children()[None]
        if isinstance(node, nodes.NestedSDFG)
    ]


def inline_top_level_nsdfgs(sdfg: dace.SDFG) -> int:
    """Widen and inline every top-level nested SDFG; returns how many transformations applied."""
    if not top_level_nsdfgs(sdfg):
        return 0
    applied = sdfg.apply_transformations_repeated(
        ExpandNestedSDFGInputs, options={"top_level_only": True}, validate=False
    )
    return applied + sdfg.apply_transformations_repeated(InlineMultistateSDFG, validate=False)


# canonical labels


def block_kind(block: ControlFlowBlock) -> str:
    """The tree keyword of a block: every sequential loop (for or while) is a ``for``; a parallel map is a ``doall``
    (:func:`relabel_state`), a loop proven to have independent iterations. A kernel is a scope, not a row."""
    if isinstance(block, LoopRegion):
        return "for"
    if isinstance(block, ConditionalBlock):
        return "if"
    if isinstance(block, ContinueBlock):
        return "continue"
    if isinstance(block, BreakBlock):
        return "break"
    if isinstance(block, ReturnBlock):
        return "return"
    if isinstance(block, SDFGState):
        return "state"
    return "block"


def normalize_labels(sdfg: dace.SDFG) -> None:
    """Rename every block and map to ``<kind><level>_<index>`` and every library call to
    ``lib_<family><level>_<index>``, counted per kind and level over the whole program. A kernel NestForge built
    (``ExternalCall``) keeps its label unless an earlier one holds it."""
    relabel_cfg(sdfg, 0, {})
    unique_library_labels(sdfg)


def unique_library_labels(sdfg: dace.SDFG) -> None:
    """Rename each library node whose label an earlier one holds, nested SDFGs included."""
    taken: dict[str, None] = {}
    for node, _ in sdfg.all_nodes_recursive():
        if not isinstance(node, nodes.LibraryNode):
            continue
        if node.label in taken:
            # LibraryNode sets name and label alike; keep them equal. DaCe's Node stub declares label read-only,
            # though LibraryNode's label is a writable Property
            node.name = node.label = find_new_name(node.label, taken)  # pyright: ignore[reportAttributeAccessIssue]
        taken[node.label] = None


def library_family(node: nodes.LibraryNode) -> str:
    """The DaCe library ``node`` comes from (``blas``, ``fft``, ``linalg``, ...); a standard node is named by its own
    kind (``reduce``, ``scan``, ``fill``)."""
    parts = type(node).__module__.split(".")
    package = parts[2] if parts[:2] == ["dace", "libraries"] and len(parts) > 2 else "lib"
    return type(node).__name__.lower() if package == "standard" else package


def next_label(kind: str, level: int, counters: dict[tuple[str, int], int]) -> str:
    """The next free ``<kind><level>_<index>``, advancing that kind's counter at that level."""
    index = counters.get((kind, level), 0)
    counters[(kind, level)] = index + 1
    return f"{kind}{level}_{index}"


def relabel_cfg(cfg: AbstractControlFlowRegion, level: int, counters: dict[tuple[str, int], int]) -> None:
    """Relabel one CFG's blocks at ``level``, recursing into the regions and states among them."""
    for block in in_order(cfg):
        block.label = next_label(block_kind(block), level, counters)
        if isinstance(block, SDFGState):
            relabel_state(block, level + 1, counters)
        elif isinstance(block, AbstractControlFlowRegion):  # a ConditionalBlock's nodes are its branches
            relabel_cfg(block, level + 1, counters)


#: Each scope's child nodes, keyed by its entry node, the top level by ``None``.
ScopeChildren = dict[nodes.Node | SDFGState | None, list[nodes.Node]]


def ordered_scope_children(state: SDFGState) -> ScopeChildren:
    """Each scope's children in execution order (:func:`in_order`), the order labels are given and printed in."""
    rank = {id(n): i for i, n in enumerate(in_order(state))}
    return {scope: sorted(kids, key=lambda n: rank[id(n)]) for scope, kids in state.scope_children().items()}


def relabel_state(state: SDFGState, level: int, counters: dict[tuple[str, int], int]) -> None:
    """Name every map of ``state`` ``kernel<level>_<index>``, one level deeper per enclosing map, including maps
    inside nested SDFGs."""
    children = ordered_scope_children(state)

    def descend(scope: nodes.MapEntry | None, depth: int) -> None:
        for node in children[scope]:
            if isinstance(node, nodes.MapEntry):
                node.map.label = next_label("doall", depth, counters)
                descend(node, depth + 1)
            elif isinstance(node, nodes.NestedSDFG):
                relabel_cfg(node.sdfg, depth, counters)
            elif isinstance(node, nodes.LibraryNode) and not isinstance(node, ExternalCall):
                # LibraryNode sets name and label alike; keep them equal (see unique_library_labels)
                node.name = node.label = next_label(f"lib_{library_family(node)}", depth, counters)  # pyright: ignore[reportAttributeAccessIssue]

    descend(None, level)
