# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 1: DaCe canonicalization to the canonical parallel form, split at its ``fuse`` stage so stage 2 owns
the fusion decisions."""

from __future__ import annotations

from dataclasses import dataclass

import dace
from dace.libraries.standard.nodes.copy import CopyLibraryNode
from dace.sdfg import nodes
from dace.sdfg.graph import SubgraphView
from dace.sdfg.propagation import propagate_memlets_nested_sdfg, propagate_memlets_state
from dace.sdfg.state import SDFGState
from dace.transformation.helpers import nest_state_subgraph
from dace.transformation.passes.canonicalize import canonicalize as dace_canonicalize
from dace.transformation.passes.canonicalize import stage_labels
from dace.transformation.passes.canonicalize.normalize_map_body import NormalizeMapBody
from dace.transformation.passes.canonicalize.pipeline import StructuralCleanup, after_fusion_units, prepared
from dace.transformation.passes.insert_explicit_copies import InsertExplicitCopies
from dace.transformation.passes.normalize_wcr import NormalizeWCR
from dace.transformation.passes.symbol_propagation import SymbolPropagation

from nestforge.ir.names import inline_top_level_nsdfgs

FUSE_STAGE = "fuse"


@dataclass(frozen=True, slots=True)
class Targets:
    """Devices the program may run on (the CPU always, the GPU opt-in), and the largest default scope: a parallel
    map whose body holds more statements (tasklets) is split before it becomes a kernel."""

    gpu: bool = False
    max_statements: int = 200

    @property
    def canon_target(self) -> str:
        return "gpu" if self.gpu else "cpu"


def canonicalize(sdfg: dace.SDFG, targets: Targets) -> dace.SDFG:
    """Canonicalize ``sdfg`` in place up to, excluding, the fusion stage, then run DaCe's structural cleanup."""
    # the frontend binds derived loop bounds to fresh interstate symbols that extraction cannot pass in
    SymbolPropagation().apply_pass(sdfg, {})
    labels = stage_labels(targets.canon_target)
    dace_canonicalize(sdfg, target=targets.canon_target, stages=labels[: labels.index(FUSE_STAGE)])
    # the moves and the tree reach no nested SDFG at the top of a state: what canonicalization left there is inlined
    inline_top_level_nsdfgs(sdfg)
    structural_cleanup(sdfg)  # no stage after fusion runs, so nothing specializes
    narrow_nested_memlets(sdfg)
    # after the cleanup, which folds the copies it can: the ones left are rows
    explicit_copies(sdfg)
    return sdfg


def structural_cleanup(sdfg: dace.SDFG) -> None:
    """DaCe's ``StructuralCleanup`` (states fused to a fixpoint, empty and dead states and unused symbols dropped,
    redundant copies folded), then every map body made canonical (see ``normalize_map_bodies``). Runs after
    canonicalization and after every move."""
    StructuralCleanup().apply_pass(sdfg, {})
    normalize_map_bodies(sdfg)


def explicit_copies(sdfg: dace.SDFG) -> None:
    """Every copy between two access nodes outside a map, in ``sdfg`` itself, becomes a ``CopyLibraryNode``: a
    ``lib_copy`` row the moves take (codegen's ``InsertExplicitCopies``, run early); :func:`implicit_copies` is the
    inverse."""
    InsertExplicitCopies(only=lambda state, edge: state.sdfg is sdfg and state.entry_node(edge.src) is None).apply_pass(
        sdfg, {}
    )


def implicit_copies(sdfg: dace.SDFG) -> int:
    """Every ``CopyLibraryNode`` of ``sdfg`` itself outside a map, between access nodes of two arrays, becomes a copy
    edge again, its ordering edges kept on the access nodes, so canonicalization's copy folding sees it and a scope
    does not make it a kernel; returns how many."""
    count = 0
    for state in sdfg.states():
        for node in [n for n in state.nodes() if isinstance(n, CopyLibraryNode) and state.entry_node(n) is None]:
            ins = [e for e in state.in_edges(node) if not e.data.is_empty()]
            outs = [e for e in state.out_edges(node) if not e.data.is_empty()]
            if len(ins) != 1 or len(outs) != 1:
                continue
            src, dst = ins[0].src, outs[0].dst
            if not isinstance(src, nodes.AccessNode) or not isinstance(dst, nodes.AccessNode) or src.data == dst.data:
                continue
            for edge in state.in_edges(node):
                if edge.data.is_empty():
                    state.add_nedge(edge.src, dst, dace.Memlet())
            for edge in state.out_edges(node):
                if edge.data.is_empty():
                    state.add_nedge(src, edge.dst, dace.Memlet())
            memlet = dace.Memlet(data=src.data, subset=ins[0].data.subset, other_subset=outs[0].data.subset)
            state.remove_node(node)
            state.add_edge(src, None, dst, None, memlet)
            count += 1
    return count


def innermost_bodies(state: SDFGState) -> list[list[nodes.Node]]:
    """The bodies of ``state``'s innermost maps that are not one nested SDFG alone."""
    bodies = []
    for entry, children in state.scope_children().items():
        if not isinstance(entry, nodes.MapEntry):
            continue
        exit_node = state.exit_node(entry)
        body = [n for n in children if n is not exit_node and not reduces_into(state, n, exit_node)]
        if not body or any(isinstance(n, nodes.EntryNode) for n in body):  # only the reduction's access node
            continue
        if [type(n) for n in body] != [nodes.NestedSDFG]:
            bodies.append(body)
    return bodies


def reduces_into(state: SDFGState, node: nodes.Node, exit_node: nodes.MapExit) -> bool:
    """Whether ``node`` is the access node a reduction leaves the map body through."""
    return isinstance(node, nodes.AccessNode) and any(e.dst is exit_node and e.data.wcr for e in state.out_edges(node))


def normalize_map_bodies(sdfg: dace.SDFG) -> None:
    """Every innermost map body becomes one nested SDFG (its states are where ``subgraph-fission`` cuts), and a
    reduction leaves it through an access node: ``map { nested SDFG } -> access node -wcr->``. Siblings a fusion left
    under one map merge into one body; a map whose body holds maps keeps them, so map-map interchange still sees
    the nest."""
    NormalizeWCR().apply_pass(sdfg, {})  # since DaCe dc0e4003c it also sources each reduction from an access node
    NormalizeMapBody().apply_pass(sdfg, {})
    for nsdfg in [sdfg, *(n.sdfg for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG))]:
        for state in nsdfg.states():
            for body in innermost_bodies(state):
                nest_state_subgraph(nsdfg, state, SubgraphView(state, body))


def narrow_nested_memlets(sdfg: dace.SDFG) -> None:
    """A nest in a map takes whole arrays, and canonicalization leaves its outer memlets whole; narrowed to what one
    iteration touches, a fusion across the map sees the per-iteration access instead of the whole array."""
    for state in sdfg.states():
        nests = [n for n in state.nodes() if isinstance(n, nodes.NestedSDFG) and state.entry_node(n) is not None]
        for nest in nests:
            propagate_memlets_nested_sdfg(state.sdfg, state, nest)
        if nests:
            propagate_memlets_state(state.sdfg, state)


def after_fusion(targets: Targets) -> list[str]:
    labels = stage_labels(targets.canon_target)
    return labels[labels.index(FUSE_STAGE) + 1 :]


def fuse_and_finish(sdfg: dace.SDFG, targets: Targets) -> dace.SDFG:
    """Stage 2 default: canonicalization's own fusion stage, then the stages after it."""
    # map fusion never descends into a nested SDFG, and a re-inlined kernel arrives nested
    inline_top_level_nsdfgs(sdfg)
    implicit_copies(sdfg)
    dace_canonicalize(sdfg, target=targets.canon_target, stages=[FUSE_STAGE, *after_fusion(targets)])
    explicit_copies(sdfg)
    return sdfg


def finish(sdfg: dace.SDFG, targets: Targets) -> dace.SDFG:
    """The lifts and cleanup that close each fusion round, then the stages after fusion, once moves chose the
    granularity by hand."""
    implicit_copies(sdfg)
    for unit in prepared(after_fusion_units(lift=True, lift_copy=True, target=targets.canon_target)):
        unit.apply_pass(sdfg, {})
    dace_canonicalize(sdfg, target=targets.canon_target, stages=after_fusion(targets))
    explicit_copies(sdfg)
    return sdfg
