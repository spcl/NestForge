"""Granularity search policies (paper C4): traditional exhaustive vs a scoped agentic hill-climb over the
fusion/offload lattice.

The search space is a 1-D ladder of labels (the fusion-granularity ladder atoms->maximal from
:mod:`nestforge.granularity`, at a fixed offloading unit; a 2-D sweep is the product over units). Each
policy drives measurements through a ``measure(label) -> microseconds`` callable and records its cost in a
:class:`~nestforge.sweep.MeasureLedger`, so the two are directly comparable: does an agent reach the same
winner as an exhaustive sweep, and with how many fewer measurements?

  * :func:`exhaustive_search` -- the traditional baseline: measure every rung, take the min.
  * :func:`hillclimb_search` -- the SCOPED agent: start at one rung, step toward the improving neighbor,
    stop at a local optimum. On a near-unimodal cost (the common granularity/time shape) it finds the same
    winner measuring far fewer rungs. It is a SCRIPTED stand-in for a learned policy -- no model inference
    runs on the dev box (see ``docs/agentic_optimizer``); the paper's agent section swaps a real policy in
    behind the same interface. Its blind spot (a non-unimodal cost can trap it in a local min) is real and
    mitigated by multi-start; the exhaustive sweep is the oracle it is measured against.

Measurements are memoized per policy run, so a re-visited rung is not re-timed -- the ledger counts only
the real measurements, the honest cost.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from nestforge.sweep import MeasureLedger


@dataclass
class SearchResult:
    """The winning rung, its measured time, and the ledger of what the search spent to find it."""
    best: str
    best_us: float
    ledger: MeasureLedger


def exhaustive_search(labels: Sequence[str],
                      measure: Callable[[str], float],
                      ledger: Optional[MeasureLedger] = None) -> SearchResult:
    """Measure every rung and return the fastest -- the traditional oracle. Cost = ``len(labels)``
    measurements, always."""
    ledger = ledger or MeasureLedger()
    costs = {label: ledger.measure(label, lambda lab=label: measure(lab)) for label in labels}
    best = min(costs, key=costs.get)
    return SearchResult(best, costs[best], ledger)


def hillclimb_search(labels: Sequence[str],
                     measure: Callable[[str], float],
                     ledger: Optional[MeasureLedger] = None,
                     start: int = 0) -> SearchResult:
    """Step along the ladder toward the improving neighbor until no neighbor is faster (the scoped agent).
    Memoized, so each rung is measured at most once; the ledger counts the real measurements."""
    ledger = ledger or MeasureLedger()
    memo = {}

    def cost(i: int) -> float:
        label = labels[i]
        if label not in memo:
            memo[label] = ledger.measure(label, lambda lab=label: measure(lab))
        return memo[label]

    i = start
    cost(i)
    while True:
        improving = [j for j in (i - 1, i + 1) if 0 <= j < len(labels) and cost(j) < cost(i)]
        if not improving:
            break
        i = min(improving, key=cost)
    best_label = min(memo, key=memo.get)
    return SearchResult(best_label, memo[best_label], ledger)
