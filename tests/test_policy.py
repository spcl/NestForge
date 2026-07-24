# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Granularity search policies (C4): exhaustive vs the scoped agentic hill-climb. Unit set, no compile --
a synthetic cost table stands in for the differential harness, so the search logic and its COST are tested
without building anything."""
from nestforge.policy import exhaustive_search, hillclimb_search

# a near-unimodal granularity->time curve: fastest in the middle (fuse-3), as fusion trades locality for
# vectorization headroom. atoms->maximal along the ladder.
LADDER = ["atoms", "fuse-1", "fuse-2", "fuse-3", "fuse-4", "maximal"]
COST = {"atoms": 10.0, "fuse-1": 8.0, "fuse-2": 5.0, "fuse-3": 4.0, "fuse-4": 6.0, "maximal": 9.0}


def measure(label):
    return COST[label]


def test_exhaustive_measures_every_rung_and_finds_the_min():
    res = exhaustive_search(LADDER, measure)
    assert res.best == "fuse-3" and res.best_us == 4.0
    assert res.ledger.measurements == len(LADDER)  # the traditional cost: measure everything


def test_hillclimb_finds_the_same_min_for_fewer_measurements():
    oracle = exhaustive_search(LADDER, measure)
    agent = hillclimb_search(LADDER, measure)
    assert agent.best == oracle.best  # same winner on a unimodal cost
    assert agent.ledger.measurements < oracle.ledger.measurements  # the C4 payoff: cheaper


def test_hillclimb_never_re_measures_a_rung():
    agent = hillclimb_search(LADDER, measure)
    assert len(agent.ledger.seen) == len(set(agent.ledger.seen))  # memoized: each rung timed at most once


def test_hillclimb_from_the_far_end_still_descends():
    agent = hillclimb_search(LADDER, measure, start=len(LADDER) - 1)  # start at 'maximal'
    assert agent.best == "fuse-3"  # walks back down to the optimum
