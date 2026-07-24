# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Predictive mode (:mod:`nestforge.predictive`): rank optimizers without building them. The hardcoded
strategy encodes one fixed policy -- no FP error, -O3, cheap vectorizer cost model -- and must be
deterministic, prefer a strict-ieee + cheap cell, and push a declining optimizer to the bottom.
"""
from nestforge.build import BuildOptions
from nestforge.optimizers import DaceOptimizer, NoOpAgent, Optimizer, Proposal
from nestforge.predictive import HardcodedStrategy, Prediction, Predictor


class DeclineAll(Optimizer):
    """An optimizer that declines every nest -- to test the -inf ranking of a non-proposer."""
    name = "declines"

    def propose(self, nest=None):
        return None


class FixedProposal(Optimizer):
    """An optimizer returning a chosen proposal verbatim -- to score exact (fp, cost) cells."""

    def __init__(self, name, fp_mode, cost_model):
        self.name = name
        self._p = Proposal(name,
                           "external",
                           fp_mode=fp_mode,
                           cost_model=cost_model,
                           language="c",
                           compiler="gcc",
                           flags=("-O3", ))

    def propose(self, nest=None):
        return self._p


def test_hardcoded_strategy_is_a_predictor():
    assert isinstance(HardcodedStrategy(), Predictor)


def test_prefers_no_fp_error_then_cheap_cost():
    strict_cheap = FixedProposal("strict-cheap", "strict-ieee", "cheap")  # +2 +1 = 3
    strict_default = FixedProposal("strict-default", "strict-ieee", "default")  # +2
    fast_cheap = FixedProposal("fast-cheap", "fast-math", "cheap")  # +1
    ranking = HardcodedStrategy().rank(None, [fast_cheap, strict_default, strict_cheap])
    assert [p.optimizer for p in ranking] == ["strict-cheap", "strict-default", "fast-cheap"]
    assert ranking[0].score == 3.0 and "no-fp-error" in ranking[0].reason and "cheap-vec-cost" in ranking[0].reason


def test_a_declining_optimizer_ranks_last_with_minus_inf():
    good = FixedProposal("good", "strict-ieee", "cheap")
    ranking = HardcodedStrategy().rank(None, [DeclineAll(), good])
    assert ranking[0].optimizer == "good"
    assert ranking[-1].optimizer == "declines"
    assert ranking[-1].score == float("-inf") and ranking[-1].proposal is None


def test_choose_returns_the_top_proposer():
    strict_cheap = FixedProposal("strict-cheap", "strict-ieee", "cheap")
    fast_default = FixedProposal("fast-default", "fast-math", "default")
    winner = HardcodedStrategy().choose(None, [fast_default, strict_cheap])
    assert winner is strict_cheap


def test_choose_returns_none_when_all_decline():
    assert HardcodedStrategy().choose(None, [DeclineAll()]) is None


def test_ranking_is_deterministic_with_a_stable_tie_break():
    # two equally-scored cells (both strict+cheap) must order by name, every time.
    a = FixedProposal("aaa", "strict-ieee", "cheap")
    b = FixedProposal("bbb", "strict-ieee", "cheap")
    r1 = [p.optimizer for p in HardcodedStrategy().rank(None, [b, a])]
    r2 = [p.optimizer for p in HardcodedStrategy().rank(None, [a, b])]
    assert r1 == r2 == ["aaa", "bbb"]


def test_noop_agent_scores_the_fp_clause():
    # the DaCe baseline is strict-ieee by construction, so the no-op agent earns the no-fp-error clause.
    ranking = HardcodedStrategy().rank(None, [NoOpAgent()])
    assert ranking[0].score == HardcodedStrategy.STRICT_FP and "no-fp-error" in ranking[0].reason


def test_prediction_carries_the_proposal():
    ranking = HardcodedStrategy().rank(None, [DaceOptimizer("canonicalize", BuildOptions())])
    assert isinstance(ranking[0], Prediction)
    assert ranking[0].proposal is not None and ranking[0].proposal.opt_mode == "canonicalize"
