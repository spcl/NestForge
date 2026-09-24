# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The labels the structure tree prints, ``<kind><level>_<index>`` and unique across the program, and the tree's
normal form: no top-level nested SDFG, ``0:trip:1`` domains, every computation inside a map, and canonical names
for transients and map parameters."""

from __future__ import annotations

import copy
import heapq
import re
from typing import Any

import dace
from dace import data as dt
from dace.sdfg import nodes
from dace.sdfg.replace import replace_dict
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
from dace.transformation.passes.canonicalize.normalize_loops_and_maps import NormalizeLoopsAndMaps
from dace.transformation.passes.normalize_wcr import NormalizeWCR
from dace.transformation.passes.normalize_wcr_source import NormalizeWCRSource
from dace.utils import find_new_name

#: Iteration variable of a wrap map; shared by every wrap map since none of them ever read it.
WRAP_PARAM = "__nf_wrap"

#: A transient name that is already canonical: ``t<n>`` for an array, ``s<n>`` for a scalar.
CANONICAL_DATA = re.compile(r"[ts]\d+")


def in_order(graph: AbstractControlFlowRegion | SDFGState) -> list[Any]:
    """A graph's nodes in topological order, ties broken by insertion order, so labels are deterministic."""
    all_nodes: list[Any] = list(graph.nodes())  # blocks of a region or nodes of a state
    if not all_nodes:
        return []
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


# 1. no top-level nested SDFG


def top_level_nsdfgs(sdfg: dace.SDFG) -> list[tuple[SDFGState, nodes.NestedSDFG]]:
    """Every ``NestedSDFG`` outside all map scopes; one inside a map is a kernel body."""
    out: list[tuple[SDFGState, nodes.NestedSDFG]] = []
    for state in sdfg.all_states():
        sd = state.scope_dict()
        out += [(state, node) for node in state.nodes() if isinstance(node, nodes.NestedSDFG) and sd[node] is None]
    return out


def inline_top_level_nsdfgs(sdfg: dace.SDFG) -> int:
    """Widen and inline every top-level nested SDFG; returns how many transformations applied."""
    if not top_level_nsdfgs(sdfg):
        return 0
    applied = sdfg.apply_transformations_repeated(
        ExpandNestedSDFGInputs, options={"top_level_only": True}, validate=False
    )
    return applied + sdfg.apply_transformations_repeated(InlineMultistateSDFG, validate=False)


# 3. every computation inside a map


def free_tasklets(state: SDFGState) -> list[nodes.Tasklet]:
    """Tasklets of ``state`` outside every map scope; a library node already is a kernel."""
    sd = state.scope_dict()
    return [n for n in state.nodes() if isinstance(n, nodes.Tasklet) and sd[n] is None]


def wrap_groups(state: SDFGState) -> list[list[nodes.Tasklet]]:
    """The free tasklets of ``state`` in the fewest groups that can each become one map: tasklets share a group
    only when neither reaches the other, found by levelling each by its longest chain of free tasklets."""
    free = {id(t) for t in free_tasklets(state)}
    if not free:
        return []
    depth: dict[int, int] = {}
    groups: dict[int, list[nodes.Tasklet]] = {}
    for node in in_order(state):
        reaching = max((depth[id(e.src)] for e in state.in_edges(node) if id(e.src) in depth), default=-1)
        if id(node) in free:
            depth[id(node)] = reaching + 1
            groups.setdefault(reaching + 1, []).append(node)
        else:
            depth[id(node)] = reaching
    return [groups[level] for level in sorted(groups)]


def wrap_group(state: SDFGState, group: list[nodes.Tasklet], name: str) -> None:
    """Enclose ``group`` in one single-iteration map."""
    entry, exit_node = state.add_map(name, {WRAP_PARAM: "0:1"}, schedule=dace.ScheduleType.Sequential)
    for tasklet in group:
        in_edges = list(state.in_edges(tasklet))
        out_edges = list(state.out_edges(tasklet))
        for edge in in_edges:
            state.remove_edge(edge)
            state.add_memlet_path(
                edge.src,
                entry,
                tasklet,
                memlet=copy.deepcopy(edge.data),
                src_conn=edge.src_conn,
                dst_conn=edge.dst_conn,
            )
        for edge in out_edges:
            state.remove_edge(edge)
            state.add_memlet_path(
                tasklet,
                exit_node,
                edge.dst,
                memlet=copy.deepcopy(edge.data),
                src_conn=edge.src_conn,
                dst_conn=edge.dst_conn,
            )
        # a tasklet with no data on one side still needs holding, or it floats out of the map
        if not in_edges:
            state.add_nedge(entry, tasklet, dace.Memlet())
        if not out_edges:
            state.add_nedge(tasklet, exit_node, dace.Memlet())


def wrap_free_tasklets(sdfg: dace.SDFG) -> int:
    """Wrap every free tasklet in a map; returns how many maps that took. Their names are placeholders."""
    added = 0
    for state in sdfg.all_states():
        for group in wrap_groups(state):
            wrap_group(state, group, f"wrap_{added}")
            added += 1
    return added


# 4. canonical labels


def block_kind(block: ControlFlowBlock) -> str:
    """The tree keyword of a block; a loop with both an init and an update statement is a ``for``."""
    if isinstance(block, LoopRegion):
        return "for" if block.init_statement is not None and block.update_statement is not None else "while"
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
    """Rename every block and map to ``<kind><level>_<index>``, counted per kind and level over the whole program.
    A library node keeps its label unless an earlier one holds it."""
    relabel_cfg(sdfg, 0, {})
    unique_library_labels(sdfg)


def unique_library_labels(sdfg: dace.SDFG) -> None:
    """Rename each library node whose label an earlier one holds, nested SDFGs included."""
    taken: dict[str, None] = {}
    for node, _ in sdfg.all_nodes_recursive():
        if not isinstance(node, nodes.LibraryNode):
            continue
        if node.label in taken:
            # LibraryNode sets name and label alike; keep them equal
            node.name = node.label = find_new_name(node.label, taken)
        taken[node.label] = None


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


def rename_transient_data(sdfg: dace.SDFG) -> dict[str, str]:
    """Rename transients to ``t<n>`` (arrays) and ``s<n>`` (scalars); returns the renames. A canonical name keeps
    its index, so a label an agent holds keeps naming the same array."""
    targets = {n: ("s" if isinstance(desc, dt.Scalar) else "t") for n, desc in sdfg.arrays.items() if desc.transient}
    settled = {n for n, prefix in targets.items() if CANONICAL_DATA.fullmatch(n) and n[0] == prefix}
    taken = {prefix: {int(n[1:]) for n in settled if n[0] == prefix} for prefix in ("t", "s")}
    survivors = {n for n in sdfg.arrays if n not in targets} | set(sdfg.symbols)
    # a mis-prefixed canonical name (a scalar "t0") holds its name until its own rename
    held = {n for n in targets if n not in settled}
    renames: dict[str, str] = {}
    for old, prefix in targets.items():
        if old in settled:
            continue
        index = 0
        while index in taken[prefix] or f"{prefix}{index}" in (survivors | held):
            index += 1
        taken[prefix].add(index)
        renames[old] = f"{prefix}{index}"
    if not renames:
        return {}
    sdfg.replace_dict(renames)
    return renames


def enclosing_param_count(node: nodes.MapEntry, scope: dict) -> int:
    """How many map parameters the maps enclosing ``node`` own."""
    count, parent = 0, scope[node]
    while parent is not None:
        count += len(parent.map.params)
        parent = scope[parent]
    return count


def rename_map_params(sdfg: dace.SDFG) -> None:
    """Rename map parameters to ``i0, i1, ...`` down each nesting chain; reusing an ancestor's name would alias
    it. Two passes through fresh names, since renaming in place collides in either order."""
    for state in sdfg.all_states():
        scope = state.scope_dict()
        entries = [n for n in state.nodes() if isinstance(n, nodes.MapEntry) and WRAP_PARAM not in n.map.params]
        targets: dict[nodes.MapEntry, list[str]] = {}
        for node in entries:
            base = enclosing_param_count(node, scope)
            wanted = [f"i{base + axis}" for axis in range(len(node.map.params))]
            if node.map.params != wanted:
                targets[node] = wanted
        if not targets:
            continue
        # first to names nothing in the state holds
        temps: dict[nodes.MapEntry, list[str]] = {}
        for index, (node, wanted) in enumerate(targets.items()):
            temp = [f"__nf_param{index}_{axis}" for axis in range(len(wanted))]
            # one simultaneous substitution per scope
            replace_dict(state.scope_subgraph(node), dict(zip(node.map.params, temp)))
            node.map.params = temp
            temps[node] = temp
        # then to the final names
        for node, wanted in targets.items():
            replace_dict(state.scope_subgraph(node), dict(zip(temps[node], wanted)))
            node.map.params = wanted


def relabel_state(state: SDFGState, level: int, counters: dict[tuple[str, int], int]) -> None:
    """Name every map of ``state`` ``kernel<level>_<index>``, one level deeper per enclosing map, including maps
    inside nested SDFGs."""
    children = state.scope_children()
    rank = {id(n): i for i, n in enumerate(in_order(state))}

    def descend(scope: nodes.MapEntry | None, depth: int) -> None:
        for node in sorted(children[scope], key=lambda n: rank.get(id(n), 0)):
            if isinstance(node, nodes.MapEntry):
                node.map.label = next_label("kernel", depth, counters)
                descend(node, depth + 1)
            elif isinstance(node, nodes.NestedSDFG):
                relabel_cfg(node.sdfg, depth, counters)

    descend(None, level)


def normalize_reductions(sdfg: dace.SDFG) -> None:
    """Put every reduction in one shape: accumulate on a body-local transient, fold with a WCR on an
    ``AccessNode -> MapExit`` edge."""
    NormalizeWCR().apply_pass(sdfg, {})
    NormalizeWCRSource().apply_pass(sdfg, {})


# the pipeline


def normalize_for_tree(sdfg: dace.SDFG) -> None:
    """Put ``sdfg`` in the tree's normal form, in place; idempotent."""
    inline_top_level_nsdfgs(sdfg)
    normalize_reductions(sdfg)
    NormalizeLoopsAndMaps().apply_pass(sdfg, {})
    wrap_free_tasklets(sdfg)
    # names last: wrap maps and inlined transients are numbered with the rest
    rename_transient_data(sdfg)
    rename_map_params(sdfg)
    normalize_labels(sdfg)
