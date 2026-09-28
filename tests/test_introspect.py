# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Structure inspection and fusion diagnosis: describe_graph (CFG tree + per-nest read/write sets) and the
``map-fusion`` plan of a pair ("yes" or a reason). Listing uses the same plan, so a "yes" here is a listed move
and a reason marks a pair that is never listed.
"""

import numpy as np
import dace

from nestforge.ir.introspect import describe_graph, nest_reads_writes
from dace.sdfg.state import LoopRegion

from nestforge.stages.moves import Rewrite, plan_move
from helpers import fusion_moves
from nestforge.stages.scopes import top_level_map_entries

N = dace.symbol("N")


def can_fuse(first, second) -> str:
    """``"yes"`` when ``map-fusion`` plans a move for the two ``(node, state)`` rows, else its reason."""
    plan = plan_move("map-fusion", [first, second])
    return "yes" if isinstance(plan, Rewrite) else plan


@dace.program
def vertical_pair(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    T = np.empty_like(A)  # transient intermediate -> vertical fusion is legal
    for i in dace.map[0:N]:
        T[i] = A[i] + B[i]
    for i in dace.map[0:N]:
        C[i] = T[i] * 2.0


@dace.program
def live_output_pair(A: dace.float64[N], B: dace.float64[N], T: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:  # T is a parameter -> a live (non-transient) output
        T[i] = A[i] + B[i]
    for i in dace.map[0:N]:
        C[i] = T[i] * 2.0


@dace.program
def independent_pair(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], D: dace.float64[N]):
    for i in dace.map[0:N]:  # no shared data -> horizontal fusion
        C[i] = A[i] * 2.0
    for i in dace.map[0:N]:
        D[i] = B[i] * 3.0


def map_entries(sdfg):
    return [(st, me) for st in sdfg.states() for me in top_level_map_entries(st)]


def test_describe_graph_lists_nests_with_read_write_sets():
    sdfg = vertical_pair.to_sdfg(simplify=True)
    text = describe_graph(sdfg)
    # This SDFG is not normalized, so the blocks keep their frontend labels -- describe_graph renders
    # whatever it is given, and normalize_for_tree is what makes those labels canonical.
    assert "`- MapState" in text  # the state the two nests share, as a tree row
    assert "[i=0:N]" in text  # the iteration domain
    assert "reads=['A', 'B'] writes=['T']" in text  # producer nest
    assert "reads=['T'] writes=['C']" in text  # consumer nest


def test_nest_reads_writes_matches_the_tree():
    sdfg = vertical_pair.to_sdfg(simplify=True)
    reads_writes = [nest_reads_writes(st, me) for st, me in map_entries(sdfg)]

    assert reads_writes == [(["A", "B"], ["T"]), (["T"], ["C"])]
    text = describe_graph(sdfg)
    assert all(f"reads={r} writes={w}" in text for r, w in reads_writes), text


def test_can_fuse_yes_for_vertical_transient():
    sdfg = vertical_pair.to_sdfg(simplify=True)
    (s1, m1), (s2, m2) = map_entries(sdfg)
    assert can_fuse((m1, s1), (m2, s2)) == "yes"
    assert can_fuse((m2, s2), (m1, s1)) == "yes"  # direction-independent
    assert fusion_moves(sdfg), "a legal move must be listed when the pair plans one"


def test_can_fuse_yes_for_independent_horizontal():
    sdfg = independent_pair.to_sdfg(simplify=True)
    entries = map_entries(sdfg)
    (s1, m1), (s2, m2) = entries[0], entries[1]
    assert s1 is s2, "independent maps land in one state after simplify"
    assert can_fuse((m1, s1), (m2, s2)) == "yes"
    assert fusion_moves(sdfg)


def test_can_fuse_refuses_live_output_with_reason():
    sdfg = live_output_pair.to_sdfg(simplify=True)
    (s1, m1), (s2, m2) = map_entries(sdfg)
    reason = can_fuse((m1, s1), (m2, s2))
    assert reason != "yes"
    assert "live output" in reason and "non-transient" in reason
    assert not fusion_moves(sdfg), "a refused pair must not be a listed move"


def test_can_fuse_reports_state_barrier_across_states():
    sdfg = independent_pair.to_sdfg(simplify=False)  # each map keeps its own state
    entries = map_entries(sdfg)
    (s1, m1), (s2, m2) = entries[0], entries[1]
    assert s1 is not s2
    reason = can_fuse((m1, s1), (m2, s2))
    assert "different states" in reason
    assert "control-flow dependency" in reason


def test_a_map_and_a_loop_are_no_map_fusion_pair():
    sdfg = vertical_pair.to_sdfg(simplify=True)
    (s1, m1), _ = map_entries(sdfg)
    reason = can_fuse((m1, s1), (LoopRegion("for9_0"), None))
    assert "is a LoopRegion" in reason
