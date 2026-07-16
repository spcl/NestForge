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

from typing import List, Set

import dace
from dace.sdfg import nodes
from dace.sdfg.graph import SubgraphView
from dace.transformation.helpers import state_fission

from nestforge.emit_libnode import is_emittable_library_node


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
    ancestors (a pure source) needs only the first step -- it lands at the top with the rest below."""
    top = state_fission(SubgraphView(state, list(upstream_nodes(state, node) | {node})))
    upstream_in_top = upstream_nodes(top, node)
    if upstream_in_top:
        state_fission(SubgraphView(top, list(upstream_in_top)))


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

    Mutates ``sdfg`` (isolation fissions states) -- pass a detached copy if the original must be preserved.
    """
    isolate_unsupported_library_nodes(sdfg)
    islands = [s for s in sdfg.states() if unsupported_library_nodes(s)]
    island_set = set(islands)

    # Union-find over the pure states: union the endpoints of every interstate edge whose BOTH ends are
    # pure. Each resulting set is one externalizable region.
    parent = {s: s for s in sdfg.states() if s not in island_set}

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
