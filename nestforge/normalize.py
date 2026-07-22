"""Normalize an SDFG into the form the agent's text tree is projected from.

The tree (:mod:`nestforge.introspect`) is the agent's whole view of the program, so every name it
prints has to be an identifier the agent can hand back, and every construct has to render one way
regardless of which frontend (Fortran, numpy, C, C++) produced it. Four properties give that, and
:func:`normalize_for_tree` establishes them in this order:

1. **No top-level nested SDFG.** A ``NestedSDFG`` outside every map scope is an opaque box in the
   middle of the tree: its states are real control flow the agent cannot see or fuse across. Widening
   its boundary memlets (``ExpandNestedSDFGInputs``) then inlining it (``InlineMultistateSDFG``)
   lifts that control flow into the tree. A nested SDFG INSIDE a map is left alone -- it is the body
   of a kernel, not structure.
2. **Canonical iteration domains.** Every map range and loop counter becomes ``0:trip:1`` -- dace's
   ``NormalizeLoopsAndMaps``. Steps are then positive and unit by construction, so the tree never has
   to render a descending or strided domain and the emitters never have to reason about step sign.
3. **Every computation sits inside a map.** A ``Tasklet`` outside all map scopes is enclosed in a
   single-iteration map, so "one map is one kernel" holds with no exceptions for the tree, the
   granularity lattice, or offloading to special-case. A ``LibraryNode`` is deliberately NOT wrapped:
   it already denotes a kernel, and burying it under a map scope would hide that.
4. **Canonical labels.** Every control-flow block and every map is renamed ``<kind><level>_<index>``,
   globally unique, so two runs over the same input produce the same names and a tree line names
   something the agent can refer back to. Maps are ``kernel`` -- one map is one kernel, including the
   wrap maps step 3 just added.

Steps 1-3 act on this SDFG and its control-flow regions only; a ``NestedSDFG`` that survives step 1
is a kernel body, and the agent fuses kernels, not their insides. Step 4 is the exception and
descends everywhere, because a nameless map is a hole in the vocabulary wherever it sits.
"""
from __future__ import annotations

import copy
import heapq
from typing import Dict, List, Tuple

import dace
from dace.sdfg import nodes
from dace.sdfg.state import (BreakBlock, ConditionalBlock, ContinueBlock, ControlFlowRegion, LoopRegion, ReturnBlock,
                             SDFGState)
from dace.transformation.interstate.expand_nested_sdfg_inputs import ExpandNestedSDFGInputs
from dace.transformation.interstate.multistate_inline import InlineMultistateSDFG
from dace.transformation.passes.canonicalize.normalize_loops_and_maps import NormalizeLoopsAndMaps

#: Iteration variable of a wrap map. One name for every wrap map in the tree: they are all ``0:1``, so
#: the variable is never read, and a shared name keeps the rendered domain identical everywhere.
WRAP_PARAM = "__nf_wrap"


def in_order(graph) -> List:
    """A graph's nodes in topological order, ties broken by insertion order -- used for both a CFG
    (blocks) and a state (dataflow nodes).

    Kahn with the ready set kept in insertion order, rather than ``sdutil.dfs_topological_sort``,
    because the tie-break is the point: a topological order alone is not unique, and two orders give
    two label assignments for one program. That sort is deterministic too (it follows edge insertion
    order) but it is depth-first, so at a branch it numbers one arm to the bottom before starting the
    other -- with structured control flow, where a CFG is usually a chain, the two agree; where they
    differ, the earliest-ready node is the one the source put first. Nodes left over (a cycle has no
    source) keep insertion order too -- an unreachable or cyclic block still needs a name.
    """
    all_nodes = list(graph.nodes())
    if not all_nodes:
        return []
    rank = {id(n): i for i, n in enumerate(all_nodes)}
    indegree = {id(n): 0 for n in all_nodes}
    for edge in graph.edges():
        indegree[id(edge.dst)] += 1
    ready = [rank[id(n)] for n in all_nodes if indegree[id(n)] == 0]
    heapq.heapify(ready)
    ordered: List = []
    while ready:
        node = all_nodes[heapq.heappop(ready)]
        ordered.append(node)
        for edge in graph.out_edges(node):
            indegree[id(edge.dst)] -= 1
            if indegree[id(edge.dst)] == 0:
                heapq.heappush(ready, rank[id(edge.dst)])
    seen = {id(n) for n in ordered}
    return ordered + [n for n in all_nodes if id(n) not in seen]


# --- 1. no top-level nested SDFG -------------------------------------------------------------------


def top_level_nsdfgs(sdfg: dace.SDFG) -> List[Tuple[SDFGState, nodes.NestedSDFG]]:
    """Every ``NestedSDFG`` that sits outside all map scopes. One inside a map is a kernel body and is
    left alone."""
    return [(state, node) for state in sdfg.all_states() for node in state.nodes()
            if isinstance(node, nodes.NestedSDFG) and state.entry_node(node) is None]


def inline_top_level_nsdfgs(sdfg: dace.SDFG) -> int:
    """Widen and inline every top-level nested SDFG, returning how many transformations that took.

    Returns 0 without touching anything when there is no top-level nested SDFG -- the common case once
    the agent has normalized once, and worth a cheap scan: both transformations otherwise pattern-match
    across every state before concluding the same thing.
    """
    if not top_level_nsdfgs(sdfg):
        return 0
    applied = sdfg.apply_transformations_repeated(ExpandNestedSDFGInputs,
                                                  options={"top_level_only": True},
                                                  validate=False)
    return applied + sdfg.apply_transformations_repeated(InlineMultistateSDFG, validate=False)


# --- 3. every computation inside a map -------------------------------------------------------------


def free_tasklets(state: SDFGState) -> List[nodes.Tasklet]:
    """Tasklets in ``state`` that sit outside every map scope. A ``LibraryNode`` is not a ``Tasklet``
    and so is never here -- by design: it already is a kernel."""
    return [n for n in state.nodes() if isinstance(n, nodes.Tasklet) and state.entry_node(n) is None]


def wrap_groups(state: SDFGState) -> List[List[nodes.Tasklet]]:
    """The free tasklets of ``state``, partitioned into the FEWEST groups each of which can become one
    map.

    Two free tasklets may share a map only if neither can reach the other: contracting a reachable
    pair into a single scope closes a cycle through whatever sits between them. So a group must be an
    antichain of the reachability order. Levelling each tasklet by the longest chain of free tasklets
    ending at it produces antichains, and by Mirsky's theorem the number of levels equals the longest
    chain -- which is a lower bound on any such partition, so this is minimal.

    The level is computed by one forward pass in topological order: every node carries the length of
    the longest free-tasklet chain reaching it, and a free tasklet extends it by one. Paths through
    map scopes and library nodes are counted like any other, which is what keeps a tasklet before a
    map and a tasklet after it in different groups.
    """
    free = {id(t) for t in free_tasklets(state)}
    if not free:
        return []
    depth: Dict[int, int] = {}
    groups: Dict[int, List[nodes.Tasklet]] = {}
    for node in in_order(state):
        reaching = max((depth[id(e.src)] for e in state.in_edges(node) if id(e.src) in depth), default=-1)
        if id(node) in free:
            depth[id(node)] = reaching + 1
            groups.setdefault(reaching + 1, []).append(node)
        else:
            depth[id(node)] = reaching
    return [groups[level] for level in sorted(groups)]


def wrap_group(state: SDFGState, group: List[nodes.Tasklet], name: str) -> None:
    """Enclose ``group`` in one single-iteration map. The map is ``Sequential``: it runs once, and
    calling a wrapped scalar statement parallel would advertise a kernel there is no work to spread."""
    entry, exit_node = state.add_map(name, {WRAP_PARAM: "0:1"}, schedule=dace.ScheduleType.Sequential)
    for tasklet in group:
        in_edges = list(state.in_edges(tasklet))
        out_edges = list(state.out_edges(tasklet))
        for edge in in_edges:
            state.remove_edge(edge)
            state.add_memlet_path(edge.src,
                                  entry,
                                  tasklet,
                                  memlet=copy.deepcopy(edge.data),
                                  src_conn=edge.src_conn,
                                  dst_conn=edge.dst_conn)
        for edge in out_edges:
            state.remove_edge(edge)
            state.add_memlet_path(tasklet,
                                  exit_node,
                                  edge.dst,
                                  memlet=copy.deepcopy(edge.data),
                                  src_conn=edge.src_conn,
                                  dst_conn=edge.dst_conn)
        # A tasklet with no data on one side still has to be held in the scope, or it floats out of the
        # map and the wrap achieves nothing.
        if not in_edges:
            state.add_nedge(entry, tasklet, dace.Memlet())
        if not out_edges:
            state.add_nedge(tasklet, exit_node, dace.Memlet())


def wrap_free_tasklets(sdfg: dace.SDFG) -> int:
    """Wrap every free tasklet in the SDFG, returning how many maps that took (0 when there were none,
    in which case nothing is touched). The names given here are placeholders --
    :func:`normalize_labels` renumbers every map, these included."""
    added = 0
    for state in sdfg.all_states():
        for group in wrap_groups(state):
            wrap_group(state, group, f"wrap_{added}")
            added += 1
    return added


# --- 4. canonical labels ---------------------------------------------------------------------------


def block_kind(block) -> str:
    """The tree keyword for a control-flow block -- the ``<kind>`` half of its canonical label.

    A ``LoopRegion`` splits by shape rather than by class: one carrying both an init and an update
    statement is a counted ``for``, anything else is a ``while``. That distinction is what the agent
    reads to know whether a trip count exists, and it is invisible in the class name.
    """
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
    """Rename every control-flow block and every map to ``<kind><level>_<index>``, globally unique.

    ``level`` is nesting depth: the root SDFG's own blocks are level 0, a state's top-level maps are
    one deeper than the state, and a map inside a map is deeper again -- so the level matches the
    indentation the tree prints the thing at.

    ``index`` runs per ``(kind, level)`` across the WHOLE SDFG, so five kernels at depth 3 are
    ``kernel3_0 .. kernel3_4`` and the numbering of one kind never depends on how many of another
    happen to sit beside it. Counting per level rather than per CFG is what makes the names unique:
    two sibling loops each hold a first state, and numbering within the CFG would call both
    ``state1_0``.
    """
    relabel_cfg(sdfg, 0, {})


def next_label(kind: str, level: int, counters: Dict[tuple, int]) -> str:
    """The next free ``<kind><level>_<index>``, advancing that kind's counter at that level."""
    index = counters.get((kind, level), 0)
    counters[(kind, level)] = index + 1
    return f"{kind}{level}_{index}"


def relabel_cfg(cfg, level: int, counters: Dict[tuple, int]) -> None:
    """Relabel one CFG's blocks at ``level``, recursing into the regions and states among them.
    ``counters`` is the per-(kind, level) running index, shared across the whole traversal."""
    for block in in_order(cfg):
        block.label = next_label(block_kind(block), level, counters)
        if isinstance(block, SDFGState):
            relabel_state(block, level + 1, counters)
        elif isinstance(block, ConditionalBlock):
            # Branches live in ``_branches``, not in the graph, so the loop above never reaches them.
            for _, branch in block.branches:
                branch.label = next_label("block", level + 1, counters)
                relabel_cfg(branch, level + 2, counters)
        elif isinstance(block, ControlFlowRegion):
            relabel_cfg(block, level + 1, counters)


def relabel_state(state: SDFGState, level: int, counters: Dict[tuple, int]) -> None:
    """Name every map in ``state`` ``kernel<level>_<index>``, outermost first and one level deeper per
    enclosing map, and descend through any ``NestedSDFG`` into its blocks.

    The descent is what stops a map inside a kernel body from keeping a frontend name -- dace-fortran
    and the Python frontend both label inner maps after the SOURCE LINE they came from (``inner_9_4``),
    so an edit above them renames a kernel that did not change. Naming is cheap and harmless at any
    depth; wrapping (:func:`wrap_free_tasklets`) deliberately is not applied down here, where a trivial
    map around every statement would bloat the body the agent never fuses inside anyway.

    Setting ``Map.label`` renames the entry and the exit together -- they share the ``Map`` object,
    which is the thing being named.
    """
    children = state.scope_children()
    rank = {id(n): i for i, n in enumerate(in_order(state))}

    def descend(scope, depth: int) -> None:
        for node in sorted(children[scope], key=lambda n: rank.get(id(n), 0)):
            if isinstance(node, nodes.MapEntry):
                node.map.label = next_label("kernel", depth, counters)
                descend(node, depth + 1)
            elif isinstance(node, nodes.NestedSDFG):
                relabel_cfg(node.sdfg, depth, counters)

    descend(None, level)


# --- the pipeline ----------------------------------------------------------------------------------


def normalize_for_tree(sdfg: dace.SDFG) -> None:
    """Put ``sdfg`` in the tree's normal form, in place: no top-level nested SDFG, canonical domains,
    every computation in a map, canonical labels. Idempotent -- running it twice is running it once,
    which matters because the agent re-normalizes after each fusion move."""
    inline_top_level_nsdfgs(sdfg)
    NormalizeLoopsAndMaps().apply_pass(sdfg, {})
    wrap_free_tasklets(sdfg)
    # Labels LAST: a wrap map is a kernel like any other and has to be numbered with the rest.
    normalize_labels(sdfg)
