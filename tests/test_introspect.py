# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Structure inspection and fusion diagnosis: describe_graph (CFG tree + per-nest read/write sets) and the
``map-fusion`` plan of a pair ("yes" or a reason). Listing uses the same plan, so a "yes" here is a listed move
and a reason marks a pair that is never listed.
"""

import copy

import dace
import numpy as np
from dace.sdfg.state import LoopRegion
from helpers import fusion_moves

from nestforge.ir.introspect import describe_graph, nest_reads_writes
from nestforge.stages.moves import Rewrite, plan_move
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
    # whatever it is given, and normalize_labels is what makes those labels canonical.
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


def test_a_live_output_between_two_maps_fuses_and_keeps_its_values():
    """NestForge refused a non-transient intermediate before asking DaCe, which fuses it and still writes it (#27)."""
    sdfg = live_output_pair.to_sdfg(simplify=True)
    reference = copy.deepcopy(sdfg)
    reference.name = "live_output_pair_reference"
    (s1, m1), (s2, m2) = map_entries(sdfg)

    plan = plan_move("map-fusion", [(m1, s1), (m2, s2)])

    assert isinstance(plan, Rewrite), plan
    plan.commit()
    assert len(map_entries(sdfg)) == 1
    rng = np.random.default_rng(0)
    want = {name: rng.random(7) for name in "ABTC"}
    got = copy.deepcopy(want)
    reference(**want, N=7)
    sdfg(**got, N=7)
    for name in "TC":
        np.testing.assert_array_equal(got[name], want[name])


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
    reason = can_fuse((m1, s1), (LoopRegion("loop9_0"), None))
    assert "is a LoopRegion" in reason


@dace.program
def row_recurrences(p0: dace.float64[N], q: dace.float64[N, N]):
    p = np.empty((N, N), dtype=np.float64)
    for i in dace.map[0:N]:
        p[i, 0] = p0[i]
        for j in range(1, N):
            p[i, j] = 0.5 * p[i, j - 1] + 1.0
    for i in dace.map[0:N]:
        for j in range(1, N):
            q[i, j] = q[i, j - 1] + p[i, j - 1]


def test_two_row_recurrences_fuse_and_keep_their_values():
    """#27: iteration ``i`` of both maps touches only row ``i`` of ``p``, so the recurrences fuse into one map."""
    sdfg = row_recurrences.to_sdfg(simplify=True)
    reference = copy.deepcopy(sdfg)
    reference.name = "row_recurrences_reference"
    (s1, m1), (s2, m2) = map_entries(sdfg)

    plan = plan_move("map-fusion", [(m1, s1), (m2, s2)])

    assert isinstance(plan, Rewrite), plan
    plan.commit()
    assert len(map_entries(sdfg)) == 1
    rng = np.random.default_rng(0)
    want = {"p0": rng.random(9), "q": rng.random((9, 9))}
    got = copy.deepcopy(want)
    reference(**want, N=9)
    sdfg(**got, N=9)
    np.testing.assert_allclose(got["q"], want["q"], rtol=1e-14)
