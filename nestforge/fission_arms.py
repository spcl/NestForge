"""The Phase-2 agent's fission lever: explode a max-fused program to statement granularity by REUSING the
existing DaCe canon passes (no new transformation).

Phase 1 fuses maximally. To reach a different (finer) granularity the agent fissions, then fuses back up
with :mod:`nestforge.fusion_arms`. Fission is done by the existing pipeline:

  * ``SplitStatements`` -- prepares the body: replicates a fission-blocking NestedSDFG per independent
    output group (statements inside an ``if`` / gather-scatter index symbols) and snapshot-renames
    forward-read anti-dependences, so the distributors below can separate the statements.
  * ``LoopFission`` -- distributes a sequential loop into one loop per independent statement group.
  * ``MapFission`` -- the map-side distributor for a map whose body is a nested SDFG of independent groups.

Every step is an existing DaCe pass, and the tests assert the composition bit-exact. ``MapFission`` is
additionally a single-pair transformation, exposed here so the agent can fission ONE map at a time when it
wants fine control.

Every step is value-preserving, but that was not free: fuzzing this composition found a real
``LoopFission`` miscompile (a body chained through a scalar had its per-iteration write-before-read
ordering dropped by a speculative rewrite the pass then failed to fission, wrong ~1/3 of runs). Fixed in
DaCe -- the pass now decides on a copy and leaves an un-fissionable loop untouched. Worth remembering that
the arm layer inherits whatever the passes it composes get wrong, so the bit-exact fuzz over this module
is the thing standing between the agent and a silently wrong program.
"""
from __future__ import annotations

from typing import List, Tuple

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


def map_fission_moves(sdfg: dace.SDFG) -> List[Tuple[nodes.MapEntry, nodes.NestedSDFG]]:
    """``(map_entry, nested_sdfg)`` pairs ``MapFission`` can split (a map whose nested-SDFG body has
    independent output groups) -- the single-pair fission move for fine agent control. Each pair is applied
    with ``MapFission.apply_to(sdfg, expr_index=1, map_entry=me, nested_sdfg=nsdfg)``; the validated body is
    carried in the move so the caller applies the same pair that was checked."""
    moves: List[Tuple[nodes.MapEntry, nodes.NestedSDFG]] = []
    for state in sdfg.all_states():
        for node in state.nodes():
            if not isinstance(node, nodes.MapEntry):
                continue
            # A map entry reaches its body over one edge per connector, so dedup before checking.
            body = dict.fromkeys(e.dst for e in state.out_edges(node) if isinstance(e.dst, nodes.NestedSDFG))
            for nsdfg in body:
                # expr_index=1 is MapFission's map-with-nested-SDFG pattern. The default 0 is the
                # map-with-subgraph pattern, whose match drops `nested_sdfg` and rejects a lone
                # NestedSDFG body as a single component -- i.e. exactly the maps enumerated here.
                if MapFission.can_be_applied_to(sdfg, expr_index=1, map_entry=node, nested_sdfg=nsdfg):
                    moves.append((node, nsdfg))
    return moves
