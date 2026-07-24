# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Isolate library nodes the numpy emitter cannot externalize (MPI / pblas communication, sparse, and the
other :data:`~nestforge.emit_libnode.REFUSED_LIBRARY_NODES`) so the whole-program / externalize lane can
SPLIT AROUND them.

An unsupported node is never offloaded: it is fissioned into its own state, with its producers in a
preceding state and everything downstream in a following one, so the lane externalizes the compute before
and after it while the node stays native. The general form of the MPI policy -- a non-emittable library
node is a hard boundary the surrounding computation splits around.

The primitive is :func:`~dace.transformation.helpers.state_fission`, applied twice (move the node with its
ancestors into a new top state, then peel the ancestors off).
"""
from __future__ import annotations

import copy
import re
from typing import List, Set, Tuple

import dace
from dace.sdfg import nodes
from dace.sdfg.graph import SubgraphView
from dace.transformation.helpers import state_fission

from nestforge.emit_libnode import is_emittable_library_node
from nestforge.emit_numpy import UnsupportedNest


def unsupported_library_nodes(state: dace.SDFGState) -> List[nodes.LibraryNode]:
    """Library nodes in ``state`` the numpy emitter will not emit -- the ones the whole-program lane must
    split around. Uses the SAME predicate as the emitter, so a name collision (an MPI ``Reduce`` vs the
    standard ``Reduce``) cannot make this pass keep a node the emitter would refuse."""
    return [n for n in state.nodes() if isinstance(n, nodes.LibraryNode) and not is_emittable_library_node(n)]


def upstream_nodes(state: dace.SDFGState, node: nodes.Node) -> Set[nodes.Node]:
    """Every node ``node`` transitively depends on within ``state`` (its data-flow ancestors)."""
    seen: Set[nodes.Node] = set()
    stack = [node]
    while stack:
        cur = stack.pop()
        for e in state.in_edges(cur):
            if e.src not in seen:
                seen.add(e.src)
                stack.append(e.src)
    return seen


def mixed_with_other_compute(state: dace.SDFGState, node: nodes.Node) -> bool:
    """True if ``state`` holds compute beyond ``node`` and access nodes -- i.e. there is something to split
    off. A state already holding only ``node`` and its in/out access nodes is left untouched."""
    return any(n is not node and not isinstance(n, nodes.AccessNode) for n in state.nodes())


def isolate_into_own_state(sdfg: dace.SDFG, state: dace.SDFGState, node: nodes.Node) -> None:
    """Fission ``state`` so ``node`` ends up alone (with its in/out access nodes) between a producers state
    and a consumers state. A node with no ancestors needs only the first fission.

    ``allow_isolated_nodes=False``: a source feeding only ``node`` would otherwise be left dataflow-dead on
    the wrong side of the split, breaking independent extraction of that state."""
    top = state_fission(SubgraphView(state, list(upstream_nodes(state, node) | {node})), allow_isolated_nodes=False)
    upstream_in_top = upstream_nodes(top, node)
    if upstream_in_top:
        state_fission(SubgraphView(top, list(upstream_in_top)), allow_isolated_nodes=False)


def isolate_unsupported_library_nodes(sdfg: dace.SDFG) -> int:
    """Split every state that mixes an unsupported library node with other computation, so each such node
    lands alone in its own state. Returns the number of nodes isolated. Idempotent: a re-run is a no-op
    because every unsupported node is already alone.

    Only top-level nodes are handled; an unsupported node inside a map scope is left in place (state
    fission cannot cut a scope) and surfaces later at emission as an :class:`UnsupportedNest`."""
    isolated = 0
    # bounded well above any real count: each fission isolates one node permanently, so exceeding it
    # means a fission failed to separate
    for _ in range(1000):
        target = None
        for state in sdfg.states():
            for node in unsupported_library_nodes(state):
                if mixed_with_other_compute(state, node):
                    target = (state, node)
                    break
            if target is not None:
                break
        if target is None:
            return isolated
        isolate_into_own_state(sdfg, target[0], target[1])
        isolated += 1
    raise RuntimeError("isolate_unsupported_library_nodes did not converge; a state_fission failed to "
                       "separate an unsupported node from surrounding compute")


def whole_program_regions(sdfg: dace.SDFG) -> Tuple[List[List[dace.SDFGState]], List[dace.SDFGState]]:
    """Isolate every unsupported node (in place), then partition ``sdfg``'s states into externalizable
    REGIONS and native ISLANDS -- the split-around-unsupported view of the whole program.

    Returns ``(regions, islands)``. An *island* is one state holding an unsupported node, left native. A
    *region* is a connected component of the state graph with the islands removed -- one externalizable
    unit, CFG-general (branches/loops of pure states stay in one region).

    Only a FLAT program is partitioned: edges reaching a control-flow region run to the REGION, not to its
    states, so the union-find below would never join them and would report too many, too small regions.
    Mutates ``sdfg`` (isolation fissions states) -- pass a detached copy to preserve the original.
    """
    nested = [s for s in sdfg.states() if s.parent_graph is not sdfg]
    if nested:
        raise UnsupportedNest(f"whole-program region partition needs a flat state graph, but "
                              f"{len(nested)} state(s) live inside a control-flow region "
                              f"(e.g. {nested[0].label!r} in {type(nested[0].parent_graph).__name__} "
                              f"{nested[0].parent_graph.label!r}); nested regions are not yet partitioned")
    isolate_unsupported_library_nodes(sdfg)
    islands = [s for s in sdfg.states() if unsupported_library_nodes(s)]
    island_set = set(islands)

    # union-find over the pure states; each resulting set is one externalizable region
    parent = {s: s for s in sdfg.states() if s not in island_set}

    # A non-state endpoint (control-flow region, bare Return/Break/Continue block) carries connectivity the
    # union-find cannot see, so its neighbours would silently be reported as independent regions. Not
    # covered by the nested-state check: a branch region holding no SDFGState leaves `nested` empty.
    known = set(parent) | island_set
    for edge in sdfg.all_interstate_edges():
        for end in (edge.src, edge.dst):
            if end not in known:
                raise UnsupportedNest(
                    f"whole-program region partition needs every interstate edge to run between states, but "
                    f"{edge.src.label!r} -> {edge.dst.label!r} has the endpoint {end.label!r} of type "
                    f"{type(end).__name__}, which is not a partitioned state; its connectivity would be "
                    f"dropped and the neighbours reported as independent regions")

    def find(s: dace.SDFGState) -> dace.SDFGState:
        while parent[s] is not s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    for edge in sdfg.all_interstate_edges():
        src, dst = edge.src, edge.dst
        if src in parent and dst in parent:
            parent[find(src)] = find(dst)

    groups: dict = {}
    for s in parent:
        groups.setdefault(find(s), []).append(s)
    regions = list(groups.values())
    return regions, islands


def region_to_standalone(sdfg: dace.SDFG, region_states: List[dace.SDFGState], name: str) -> dace.SDFG:
    """Copy one whole-program region (a set of connected pure-compute states) into a fresh, independently
    compilable SDFG that :func:`~nestforge.emit_numpy.sdfg_to_numpy` can emit on its own.

    A transient also used OUTSIDE the region crosses the boundary and is promoted to non-transient (a
    region input/output); a whole-program transient stays internal. Arrays touched only outside are
    dropped, and the region's single entry state becomes the start. ``sdfg`` is not mutated."""
    region_labels = {s.label for s in region_states}
    outside_read: Set[str] = set()
    outside_write: Set[str] = set()
    for state in sdfg.states():
        if state.label in region_labels:
            continue
        read, write = state.read_and_write_sets()
        outside_read |= read
        outside_write |= write

    work = copy.deepcopy(sdfg)
    work.name = name
    for state in list(work.states()):
        if state.label not in region_labels:
            work.remove_node(state)

    region_used: Set[str] = set()
    for state in work.states():
        read, write = state.read_and_write_sets()
        region_used |= read | write
    # An array may be referenced ONLY on an inter-state edge (`idx = offsets[k]`), which the dataflow-only
    # read_and_write_sets never reports -- dropping it leaves the kernel naming an undefined array.
    for edge in work.all_interstate_edges():
        expressions = list(edge.data.assignments.values()) + [str(edge.data.condition.as_string)]
        for expr in expressions:
            region_used |= set(re.findall(r"[A-Za-z_]\w*", expr)) & work.arrays.keys()
    for aname in list(work.arrays):
        if aname not in region_used:
            del work.arrays[aname]  # touched only outside the region (after orphan drop, no node references it)
        elif work.arrays[aname].transient and (aname in outside_read or aname in outside_write):
            work.arrays[aname].transient = False  # crosses the region boundary -> a region input / output

    # Entry decided on the ORIGINAL graph. in_degree == 0 on the carved copy instead refuses a valid
    # single-entry region whose header carries a back-edge (a flat pure-state loop).
    entry_labels = [
        s.label for s in sdfg.states()
        if s.label in region_labels and (sdfg.in_degree(s) == 0 or any(p.label not in region_labels
                                                                       for p in sdfg.predecessors(s)))
    ]
    entries = [s for s in work.states() if s.label in entry_labels]
    if len(entries) != 1:
        raise ValueError(f"region {name} has {len(entries)} entry states; only a single-entry region is "
                         "externalizable (a multi-entry region needs a synthetic join -- not yet handled)")
    work.start_block = work.node_id(entries[0])
    work.reset_cfg_list()
    return work
