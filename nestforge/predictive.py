"""Predictive mode: pick the optimizer that will win WITHOUT building them all.

The arena is exhaustive -- compile every variant, validate, time, take the best. Predictive mode replaces
the sweep with an estimate: rank the :class:`~nestforge.optimizers.Optimizer` set for a nest and build only
the top pick (the arena still validates it, so a wrong prediction costs time, never correctness).

For now the predictor is a HARDCODED STRATEGY -- no learned model, no compiling. It encodes one fixed
policy: **no FP error** (``strict-ieee``), **-O3** (every variant carries it), and a **cheap vectorizer
cost model** globally. That is a deliberate floor: the safest correct-and-fast cell, chosen without
measurement. The richer cost model (numerical-stability analysis, opt-report diffing, per-hardware
instruction scanning) is planned in ``docs/predictive/README.md`` and will implement the same
:class:`Predictor` contract, so swapping it in changes no caller.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import List, Optional, Sequence

from nestforge.optimizers import Optimizer, Proposal


@dataclass(frozen=True, slots=True)
class Prediction:
    """A predicted score for one optimizer. Higher ``score`` = predicted faster/preferred. ``reason`` is a
    short human string (the hardcoded strategy fills it with which policy clauses matched); ``proposal`` is
    the optimizer's recipe, or ``None`` if it declined this nest."""
    optimizer: str
    score: float
    reason: str
    proposal: Optional[Proposal]


class Predictor(abc.ABC):
    """Rank optimizers for a nest without building them. Deterministic: same nest + same optimizers ->
    same ranking. The real cost model will subclass this; every caller goes through :meth:`rank` /
    :meth:`choose`, so the policy is swappable behind one contract."""

    __slots__ = ()

    @abc.abstractmethod
    def rank(self, nest: Optional[object], optimizers: Sequence[Optimizer]) -> List[Prediction]:
        """Every optimizer scored, most-preferred first. An optimizer that declines the nest still appears,
        with its proposal ``None`` and a score below any that proposed."""

    def choose(self, nest: Optional[object], optimizers: Sequence[Optimizer]) -> Optional[Optimizer]:
        """The single predicted winner (the top of :meth:`rank` that actually proposed), or ``None`` when
        every optimizer declined."""
        by_name = {o.name: o for o in optimizers}
        for prediction in self.rank(nest, optimizers):
            if prediction.proposal is not None:
                return by_name[prediction.optimizer]
        return None


class HardcodedStrategy(Predictor):
    """The fixed policy: no FP error, -O3, cheap vectorization cost model -- scored, not measured.

    Scoring (deterministic; ties break on the optimizer name so the order is stable):

      * ``+2`` a strict-ieee variant -- the no-FP-error clause, the load-bearing one;
      * ``+1`` a cheap vectorizer cost model -- the global cheap-cost clause;
      * ``-inf`` an optimizer that declines the nest (``propose`` -> ``None``) -- unbuildable here.

    -O3 needs no clause: every variant's flags start from ``base_flags`` (``-O3``), so it is a constant.
    A DaCe-lane variant is strict-ieee by construction, so it scores the FP clause too; among equals the
    cheap-cost external cell wins, matching the stated "cheap cost model for vectorization globally".
    """

    __slots__ = ()

    STRICT_FP = 2.0
    CHEAP_COST = 1.0

    def rank(self, nest: Optional[object], optimizers: Sequence[Optimizer]) -> List[Prediction]:
        predictions: List[Prediction] = []
        for opt in optimizers:
            proposal = opt.propose(nest)
            if proposal is None:
                predictions.append(Prediction(opt.name, float("-inf"), "declined this nest", None))
                continue
            score = 0.0
            clauses: List[str] = []
            if proposal.fp_mode == "strict-ieee":
                score += self.STRICT_FP
                clauses.append("no-fp-error")
            if proposal.cost_model == "cheap":
                score += self.CHEAP_COST
                clauses.append("cheap-vec-cost")
            reason = "+".join(clauses) if clauses else "no strategy clause matched"
            predictions.append(Prediction(opt.name, score, reason, proposal))
        predictions.sort(key=lambda p: (-p.score, p.optimizer))
        return predictions
