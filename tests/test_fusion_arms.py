# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The stage 2 fusion moves (:mod:`nestforge.stages.moves`): list legal fusion moves and apply them, with the correctness net that any sequence of applied moves preserves the program's value bit-for-bit
against the un-fused reference. Exercises all three arms -- loop, vertical map, horizontal map -- and the
agent's real pattern of applying a random legal sequence.
"""

import dace
import numpy as np
import pytest
from dace.transformation.interstate.state_fusion import StateFusion
from helpers import apply_move, fusion_moves, random_vectors, run

from nestforge.ir.introspect import tree_rows
from nestforge.stages.moves import Rewrite, plan_move, tree_label
from nestforge.stages.scopes import top_level_map_entries

N = dace.symbol("N")
f64 = dace.float64


def apply_to_fixpoint(sdfg, order="greedy", seed=0):
    """Apply legal fusion moves until none remain; return the count. ``order='random'`` picks a random
    legal move each round (a seeded stand-in for the agent's choice)."""
    rng = np.random.default_rng(seed)
    applied = 0
    while True:
        moves = fusion_moves(sdfg)
        if not moves:
            return applied
        move = moves[0] if order == "greedy" else moves[int(rng.integers(len(moves)))]
        apply_move(sdfg, move)
        applied += 1


# programs exercising each arm (co-locate maps into one state so horizontal siblings can match)


@dace.program
def two_recurrences(a: f64[N], b: f64[N], c: f64[N]):
    for i in range(1, N):
        b[i] = b[i - 1] + a[i]
    for i in range(1, N):
        c[i] = c[i - 1] + b[i]


@dace.program
def producer_consumer_maps(a: f64[N], b: f64[N]):
    tmp = np.empty_like(a)
    for i in dace.map[0:N]:
        tmp[i] = a[i] * 2.0
    for i in dace.map[0:N]:
        b[i] = tmp[i] + 1.0


@dace.program
def sibling_maps(a: f64[N], b: f64[N], c: f64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i] * 2.0
    for i in dace.map[0:N]:
        c[i] = a[i] + 1.0


def co_located(prog):
    """Build the SDFG and StateFusion sequential states together, so producer/consumer and sibling maps
    land in one state where the map-fusion arms can match them (mirrors the fusion-ready canonical form)."""
    sdfg = prog.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(StateFusion)
    return sdfg


# enumeration finds the right arm


def the_only_fusion_applies(sdfg) -> str:
    (move,) = fusion_moves(sdfg)
    return apply_move(sdfg, move)


def test_two_recurrences_offer_a_loop_fusion():
    assert the_only_fusion_applies(two_recurrences.to_sdfg(simplify=True)) == "LoopFusion"


def test_a_producer_and_its_consumer_offer_a_vertical_map_fusion():
    assert the_only_fusion_applies(co_located(producer_consumer_maps)) == "MapFusionVertical"


def test_siblings_offer_a_horizontal_map_fusion():
    assert the_only_fusion_applies(co_located(sibling_maps)) == "MapFusionHorizontal"


# applying moves preserves value bit-for-bit


@pytest.mark.parametrize(
    "prog,names,colocate",
    [
        (two_recurrences, ("a", "b", "c"), False),
        (producer_consumer_maps, ("a", "b"), True),
        (sibling_maps, ("a", "b", "c"), True),
    ],
)
def test_apply_all_fusions_is_value_preserving(prog, names, colocate):
    inputs = random_vectors(names=names)
    ref = run(prog.to_sdfg(simplify=True), inputs, 48)
    sdfg = co_located(prog) if colocate else prog.to_sdfg(simplify=True)
    applied = apply_to_fixpoint(sdfg, order="greedy")
    got = run(sdfg, inputs, 48)
    assert applied >= 1
    for k in inputs:
        np.testing.assert_array_equal(got[k], ref[k], err_msg=k)


@dace.program
def all_three_arms(a: f64[N], b: f64[N], c: f64[N], d: f64[N]):
    tmp = np.empty_like(a)
    for i in dace.map[0:N]:  # producer
        tmp[i] = a[i] * 2.0
    for i in dace.map[0:N]:  # vertical consumer of tmp
        b[i] = tmp[i] + 1.0
    for i in dace.map[0:N]:  # sibling of the above over a
        c[i] = a[i] - 1.0
    for i in range(1, N):  # sequential recurrence pair
        d[i] = d[i - 1] + a[i]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_random_fusion_sequence_is_value_preserving(seed):
    inputs = random_vectors(names=("a", "b", "c", "d"))
    ref = run(all_three_arms.to_sdfg(simplify=True), inputs, 48)
    sdfg = co_located(all_three_arms)
    apply_to_fixpoint(sdfg, order="random", seed=seed)
    got = run(sdfg, inputs, 48)
    for k in inputs:
        np.testing.assert_array_equal(got[k], ref[k], err_msg=f"seed {seed}: diverged from reference")


def test_no_moves_on_a_single_map():

    @dace.program
    def one_map(a: f64[N], b: f64[N]):
        for i in dace.map[0:N]:
            b[i] = a[i] * 2.0

    assert fusion_moves(one_map.to_sdfg(simplify=True)) == []


@dace.program
def live_and_transient(A: dace.float64[N], B: dace.float64[N], live_out: dace.float64[N], C: dace.float64[N]):
    T = np.empty_like(A)  # transient intermediate
    for i in dace.map[0:N]:
        T[i] = A[i] + B[i]
        live_out[i] = A[i] * 3.0  # a non-transient result of the same producer map
    for i in dace.map[0:N]:
        C[i] = T[i] * 2.0 + live_out[i]  # consumer reads both intermediates


def map_pairs(sdfg):
    for state in sdfg.states():
        entries = top_level_map_entries(state)
        for first in entries:
            for second in entries:
                if first is not second:
                    yield first, second


def test_a_map_pair_plans_a_fusion_exactly_when_it_is_listed():
    """A live output beside a fusable transient must not hide the listed move, and every other pair gets a reason."""
    sdfg = live_and_transient.to_sdfg(simplify=True)
    offered = {frozenset(labels) for _, labels in fusion_moves(sdfg)}
    assert offered, "the fixture must offer a map fusion, else it tests nothing"
    rows = tree_rows(sdfg)
    for first, second in map_pairs(sdfg):
        plan = plan_move("map-fusion", [rows[tree_label(first)], rows[tree_label(second)]])
        listed = frozenset({tree_label(first), tree_label(second)}) in offered
        assert isinstance(plan, Rewrite) == listed, (first, second, plan)
        assert isinstance(plan, Rewrite) or plan
