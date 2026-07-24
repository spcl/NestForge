# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
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
    """Explode ``sdfg`` to STATEMENT granularity in place, where a statement is one GLOBAL output written
    from N global inputs, with local temps recomputed (never materialized to a buffer). Returns the number
    of fission steps applied. The inverse of Phase-1's max-fuse; the agent then fuses back up
    (:mod:`nestforge.fusion_arms`) to the chosen granularity.

    Steps: ``SplitStatements(split_maps=True)`` (a straight-line map with several global outputs -> one
    flat map per output, shared local recomputed; a fission-blocking NestedSDFG replicated per output),
    then ``LoopFission`` (sequential loops), then :func:`fission_multi_output_maps` (the remaining
    NestedSDFG-bodied maps -- dependent / indirection).

    NOT ``apply_transformations_repeated(MapFission)``: that splits a map per TASKLET and materializes the
    local temps to size-N arrays (``{t=x*2; A=t+1}`` -> two maps + a buffer ``t``). Statement granularity
    is one map per global output precisely because that is the finest split that keeps a local a scalar.
    """
    from dace.transformation.passes.canonicalize.split_statements import SplitStatements
    from dace.transformation.passes.loop_fission import LoopFission

    applied = 0
    applied += SplitStatements(split_maps=True).apply_pass(sdfg, {}) or 0
    applied += LoopFission().apply_pass(sdfg, {}) or 0
    applied += fission_multi_output_maps(sdfg)
    return applied


def fission_multi_output_maps(sdfg: dace.SDFG) -> int:
    """Fission only the maps NOT yet at statement granularity: a top-level map still writing >=2 distinct
    global outputs (the NestedSDFG-bodied dependent / indirection maps ``SplitStatements`` left). A flat
    single-output map is already a statement and is left ALONE -- MapFission would split its tasklet chain
    and materialize the locals. Returns the number of MapFission applications."""
    applied = 0
    while True:
        target = None
        for state in sdfg.all_states():
            sd = state.scope_dict()
            for entry in [n for n in state.nodes() if isinstance(n, nodes.MapEntry) and sd[n] is None]:
                exit_node = state.exit_node(entry)
                global_outs = {e.data.data for e in state.in_edges(exit_node) if e.data is not None and e.data.data}
                if len(global_outs) < 2:
                    continue
                # expr_index=1 wants a single NestedSDFG body; expr_index=0 the multi-component form.
                bodies = [
                    n for n in state.scope_subgraph(entry, False, False).nodes() if isinstance(n, nodes.NestedSDFG)
                ]
                if len(bodies) == 1 and MapFission.can_be_applied_to(
                        sdfg, expr_index=1, map_entry=entry, nested_sdfg=bodies[0]):
                    target = (1, {"map_entry": entry, "nested_sdfg": bodies[0]})
                elif MapFission.can_be_applied_to(sdfg, expr_index=0, map_entry=entry):
                    target = (0, {"map_entry": entry})
                if target is not None:
                    break
            if target is not None:
                break
        if target is None:
            break
        expr_index, kwargs = target
        MapFission.apply_to(sdfg, expr_index=expr_index, **kwargs)
        applied += 1
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
