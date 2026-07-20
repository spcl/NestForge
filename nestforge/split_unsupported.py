"""Isolate library nodes the numpy emitter cannot externalize (MPI / pblas communication, sparse, and the
other :data:`~nestforge.emit_libnode.REFUSED_LIBRARY_NODES`) so the whole-program / externalize lane can
SPLIT AROUND them.

An unsupported node is never offloaded. Instead each one is fissioned into its own state, with the pure
computation it depends on in a preceding state and everything downstream in a following state -- so the
lane can externalize the compute *before* and *after* the node independently while the node itself stays
native. This is the general form of the MPI policy: an MPI collective (or any non-emittable library node)
is a hard boundary the surrounding computation splits around.

The primitive is DaCe's :func:`~dace.transformation.helpers.state_fission`, applied twice: first move the
node together with its data-flow ancestors into a new top state, then peel the ancestors off so the node
stands alone. A nested SDFG lowers to a function call, so once a pure-compute state is on its own it is
directly externalizable via the same nesting the extractor already uses (:mod:`nestforge.extract`).
"""
from __future__ import annotations

import copy
import re
from typing import List, Set

import dace
from dace.sdfg import nodes
from dace.sdfg.graph import SubgraphView
from dace.transformation.helpers import state_fission

from nestforge.emit_libnode import is_emittable_library_node
from nestforge.emit_numpy import UnsupportedNest


def unsupported_library_nodes(state: dace.SDFGState) -> List[nodes.LibraryNode]:
    """Library nodes in ``state`` the numpy emitter will not emit -- MPI/pblas communication, sparse, an
    explicitly refused node, or one no emitter is registered for. These are exactly the nodes the
    whole-program lane must split around instead of emit (an emittable node stays in place).

    Uses :func:`~nestforge.emit_libnode.is_emittable_library_node` -- the SAME predicate the emitter uses --
    so a name collision (an MPI ``Reduce`` shares its class name with the registered standard ``Reduce``)
    cannot make this pass leave in place a node the emitter would refuse."""
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
    (the nodes it depends on) and a consumers state (everything downstream / independent).

    Two :func:`state_fission` steps: (1) move ``node`` and its ancestors into a new top state, leaving the
    rest below; (2) peel the ancestors off that top state so ``node`` stands alone. A node with no
    ancestors (a pure source) needs only the first step -- it lands at the top with the rest below.

    ``allow_isolated_nodes=False``: a source that fed only ``node`` (e.g. a scalar ``root`` read only by an
    MPI collective) would otherwise be left as an isolated, dataflow-dead access node on the wrong side of
    the split -- dropping it keeps each resulting state a clean, independently-extractable region."""
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
    # Bound the loop well above any real kernel's unsupported-node count: each fission isolates one node
    # permanently, so a run that exceeds this signals a fission that failed to separate (surface it).
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


def whole_program_regions(sdfg: dace.SDFG):
    """Isolate every unsupported node (in place), then partition ``sdfg``'s states into externalizable
    REGIONS and native ISLANDS -- the split-around-unsupported view of the whole program.

    Returns ``(regions, islands)``. Each *island* is one state holding an unsupported (non-emittable) node,
    left native. Each *region* is a maximal set of connected pure-compute states -- one externalizable unit
    the whole-program lane hands to an external tool (nest -> function call -> ExternalCall). Regions are
    connected components of the state graph with the islands removed, so the partition is CFG-general
    (branches and loops of pure states stay in one region); an interstate edge between two regions always
    runs through an island, which is the boundary the compute splits around.

    Only a FLAT program is partitioned: every state must sit directly in ``sdfg``. ``sdfg.states()`` also
    yields states nested in a control-flow region (a ``LoopRegion``, a ``ConditionalBlock``), but the edges
    reaching such a region run to the REGION, not to its states -- the union-find below would never join
    them to their neighbours and would report more (and smaller) regions than really exist, i.e. a wrong
    extraction. ``region_to_standalone`` likewise addresses states as top-level nodes of ``sdfg``. Refuse.

    Mutates ``sdfg`` (isolation fissions states) -- pass a detached copy if the original must be preserved.
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

    # Union-find over the pure states: union the endpoints of every interstate edge whose BOTH ends are
    # pure. Each resulting set is one externalizable region.
    parent = {s: s for s in sdfg.states() if s not in island_set}

    # Every interstate edge must run between states we partition. An endpoint that is NOT a state -- a
    # control-flow region, or a bare Return/Break/Continue block -- carries connectivity the union-find
    # cannot see, so its neighbours would be reported as separate regions with no island between them: a
    # wrong extraction, silently. The nested-state check above does not cover this: a branch region holding
    # no SDFGState at all (the ordinary ``if (cond) return;`` shape) leaves `nested` empty while its edges
    # still bypass the partition. Refuse on the CAUSE -- a skipped edge -- not on a proxy for it.
    known = set(parent) | island_set
    for edge in sdfg.all_interstate_edges():
        for end in (edge.src, edge.dst):
            if end not in known:
                raise UnsupportedNest(
                    f"whole-program region partition needs every interstate edge to run between states, but "
                    f"{edge.src.label!r} -> {edge.dst.label!r} has the endpoint {end.label!r} of type "
                    f"{type(end).__name__}, which is not a partitioned state; its connectivity would be "
                    f"dropped and the neighbours reported as independent regions")

    def find(s):
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

    The boundary is the crux: a **transient** that is written in the region and used OUTSIDE it (or read in
    the region and written outside) crosses the region boundary and is promoted to non-transient so it
    becomes a region input / output -- while a whole-program transient stays internal. Arrays touched only
    outside the region are dropped. The region's single entry state (no in-region predecessor) becomes the
    start.

    ``region_states`` are states of ``sdfg`` (as returned by :func:`whole_program_regions`); ``sdfg`` is not
    mutated (the work is done on a deep copy)."""
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
    # An array can be referenced ONLY on an inter-state edge (a data-dependent index hoisted onto the edge,
    # e.g. `idx = offsets[k]`); read_and_write_sets is dataflow-only and never reports it, so it would be
    # deleted below while the edge still assigns from it. Keeping an extra array is harmless; dropping a
    # referenced one leaves the emitted kernel naming an undefined array.
    for edge in work.all_interstate_edges():
        expressions = list(edge.data.assignments.values()) + [str(edge.data.condition.as_string)]
        for expr in expressions:
            region_used |= {a for a in work.arrays if re.search(rf"(?<![\w.]){re.escape(a)}\b", expr)}
    for aname in list(work.arrays):
        if aname not in region_used:
            del work.arrays[aname]  # touched only outside the region (after orphan drop, no node references it)
        elif work.arrays[aname].transient and (aname in outside_read or aname in outside_write):
            work.arrays[aname].transient = False  # crosses the region boundary -> a region input / output

    # The entry is decided on the ORIGINAL graph: a state entered from outside the region, or the program
    # start. Using in_degree == 0 on the carved copy instead misses a legitimate single-entry region whose
    # header carries a BACK-EDGE (a flat state loop of pure-compute states, which whole_program_regions
    # keeps in one piece) -- its in_degree is >= 1, so no state qualified and a valid region was refused.
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
