"""The Phase-2 agent's fission lever: explode a max-fused program to statement granularity by REUSING the
existing DaCe canon passes (no new transformation).

Phase 1 fuses maximally. To reach a different (finer) granularity the agent fissions, then fuses back up
with :mod:`nestforge.fusion_arms`. Fission is done by the existing pipeline:

  * ``SplitStatements`` -- prepares the body: replicates a fission-blocking NestedSDFG per independent
    output group (statements inside an ``if`` / gather-scatter index symbols) and snapshot-renames
    forward-read anti-dependences, so the distributors below can separate the statements.
  * ``LoopFission`` -- distributes a sequential loop into one loop per independent statement group.
  * ``MapFission`` -- the map-side distributor for a map whose body is a nested SDFG of independent groups.

Every step is a semantics-preserving DaCe pass; the composition is value-preserving by construction, and
the tests assert it bit-exact. ``MapFission`` is additionally a single-pair transformation, exposed here so
the agent can fission ONE map at a time when it wants fine control.
"""
from __future__ import annotations

from typing import List

import dace
from dace.sdfg import nodes
from dace.transformation.dataflow.map_fission import MapFission


def fission_to_statements(sdfg: dace.SDFG) -> int:
    """Explode ``sdfg`` to statement granularity in place -- ``SplitStatements`` then ``LoopFission`` then
    ``MapFission``, so each independent output statement lands in its own loop/map nest. Returns the number
    of fission steps applied. The inverse of Phase-1's max-fuse; the agent then fuses back up
    (:mod:`nestforge.fusion_arms`) to the chosen granularity."""
    from dace.transformation.passes.canonicalize.split_statements import SplitStatements
    from dace.transformation.passes.loop_fission import LoopFission

    applied = 0
    applied += SplitStatements().apply_pass(sdfg, {}) or 0
    applied += LoopFission().apply_pass(sdfg, {}) or 0
    applied += sdfg.apply_transformations_repeated(MapFission) or 0
    return applied


def map_fission_moves(sdfg: dace.SDFG) -> List[nodes.MapEntry]:
    """Map entries ``MapFission`` can split (a map whose nested-SDFG body has independent output groups) --
    the single-pair fission move for fine agent control. Each entry ``me`` is applied with
    ``MapFission.apply_to(sdfg, map_entry=me, nested_sdfg=<its body>)``."""
    entries: List[nodes.MapEntry] = []
    for state in sdfg.all_states():
        for node in state.nodes():
            if not isinstance(node, nodes.MapEntry):
                continue
            body = [e.dst for e in state.out_edges(node) if isinstance(e.dst, nodes.NestedSDFG)]
            for nsdfg in body:
                if MapFission.can_be_applied_to(sdfg, map_entry=node, nested_sdfg=nsdfg):
                    entries.append(node)
                    break
    return entries
