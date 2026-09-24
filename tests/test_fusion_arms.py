# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The phase 1 fusion tool surface (:mod:`nestforge.phases.schedule`): enumerate legal fusion moves and apply
them, with the correctness net that any sequence of applied moves preserves the program's value bit-for-bit
against the un-fused reference. Exercises all three arms -- loop, vertical map, horizontal map -- and the
agent's real pattern of applying a random legal sequence.
"""

import numpy as np
import pytest

import dace
from dace.transformation.interstate.state_fusion import StateFusion

from nestforge.phases.scopes import top_level_map_entries
from nestforge.phases.schedule import (
    FusionMove,
    apply_fusion,
    can_fuse,
    enumerate_fusions,
    horizontal_map_moves,
    loop_fusion_moves,
    vertical_map_moves,
)
from helpers import random_vectors, run

N = dace.symbol("N")
f64 = dace.float64


def apply_to_fixpoint(sdfg, order="greedy", seed=0):
    """Apply legal fusion moves until none remain; return the count. ``order='random'`` picks a random
    legal move each round (a seeded stand-in for the agent's choice)."""
    rng = np.random.default_rng(seed)
    applied = 0
    while True:
        moves = enumerate_fusions(sdfg)
        if not moves:
            return applied
        move = moves[0] if order == "greedy" else moves[int(rng.integers(len(moves)))]
        apply_fusion(move)
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
    land in one state where the map-fusion arms can match them (mirrors the phase 1 fusion-ready canon)."""
    sdfg = prog.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(StateFusion)
    return sdfg


# enumeration finds the right arm


def test_enumerate_finds_loop_fusion():
    sdfg = two_recurrences.to_sdfg(simplify=True)
    moves = list(loop_fusion_moves(sdfg))
    assert any(m.kind == "fuse-loops" for m in moves)


def test_enumerate_finds_vertical_map_fusion():
    sdfg = co_located(producer_consumer_maps)
    moves = list(vertical_map_moves(sdfg))
    assert any(m.kind == "fuse-map-vertical" for m in moves)


def test_enumerate_finds_horizontal_map_fusion():
    sdfg = co_located(sibling_maps)
    moves = list(horizontal_map_moves(sdfg))
    assert any(m.kind == "fuse-map-horizontal" for m in moves)


def test_enumerated_moves_carry_apply_kwargs():
    sdfg = co_located(producer_consumer_maps)
    for m in enumerate_fusions(sdfg):
        assert isinstance(m, FusionMove) and m.where and m.xform is not None


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

    assert enumerate_fusions(one_map.to_sdfg(simplify=True)) == []


@dace.program
def live_and_transient(A: dace.float64[N], B: dace.float64[N], live_out: dace.float64[N], C: dace.float64[N]):
    T = np.empty_like(A)  # transient intermediate
    for i in dace.map[0:N]:
        T[i] = A[i] + B[i]
        live_out[i] = A[i] * 3.0  # a non-transient result of the same producer map
    for i in dace.map[0:N]:
        C[i] = T[i] * 2.0 + live_out[i]  # consumer reads both intermediates


def map_pairs(sdfg):
    for state in sdfg.all_states():
        entries = top_level_map_entries(state)
        for first in entries:
            for second in entries:
                if first is not second:
                    yield first, second


def offered_pairs(sdfg):
    """The map-entry pairs ``enumerate_fusions`` lists, unordered."""
    pairs = set()
    for move in enumerate_fusions(sdfg):
        if move.kind == "fuse-map-vertical":
            exit_node = move.where["first_map_exit"]
            state = next(s for s in sdfg.all_states() if exit_node in s.nodes())
            pairs.add(frozenset({state.entry_node(exit_node), move.where["second_map_entry"]}))
        elif move.kind == "fuse-map-horizontal":
            pairs.add(frozenset({move.where["first_parallel_map_entry"], move.where["second_parallel_map_entry"]}))
    return pairs


def test_can_fuse_agrees_with_enumerate_fusions():
    """``can_fuse`` says yes for exactly the pairs ``enumerate_fusions`` lists, and otherwise gives a reason: a live
    output beside a fusable transient must not hide the listed move."""
    sdfg = live_and_transient.to_sdfg(simplify=True)
    offered = offered_pairs(sdfg)
    assert offered, "the fixture must offer a map fusion, else it tests nothing"
    for first, second in map_pairs(sdfg):
        verdict = can_fuse(sdfg, first, second)
        assert (verdict == "yes") == (frozenset({first, second}) in offered), (first, second, verdict)
        assert isinstance(verdict, str) and verdict


def test_live_output_does_not_mask_a_transient_fusion():
    # the specific shape: if any move is offered for the producer/consumer pair, can_fuse must not report the
    # live output as the blocker.
    sdfg = live_and_transient.to_sdfg(simplify=True)
    assert enumerate_fusions(sdfg), "fixture must produce a fusable pair, else it tests nothing"
    verdicts = [can_fuse(sdfg, a, b) for a, b in map_pairs(sdfg)]
    assert any(v == "yes" for v in verdicts), f"a move is offered but no pair says yes: {verdicts}"
