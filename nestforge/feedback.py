# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 4 of the 4-phase optimizer: change granularity from MEASUREMENTS, and loop.

Phase 1 fixed a starting granularity, Phase 2 externalized it, Phase 3 tuned each nest. Phase 4 reads
the Phase-3 results and requests a DIFFERENT fuse/fission -- then re-runs the earlier phases on the
changed granularity. Bounded rounds; stop when a round yields no improvement (``docs/agentic_optimizer``:
"the fuse/fission decision is driven by measurement, not estimate ... stop when a round yields no
improvement"). This closes the P4->P1 loop.

The feedback is an :class:`~nestforge.optimizers.Outcome` per round -- did the granularity build, was it
bit-exact, how fast. The two rules that drive the loop:

  * :func:`improved` -- did this round beat every prior bit-exact outcome? A round that does not
    improve ends the loop.
  * :func:`best_outcome` -- the fastest bit-exact outcome so far. The selection rule: a candidate that
    lost the correctness gate never wins on speed.

:func:`run_feedback_loop` drives them over granularity states, one re-enumerated fusion move per round.
Every move is value-preserving (legality-gated + fuzzed bit-exact) and ``measure`` re-validates, so the
loop changes only speed, never correctness. The per-nest INNER loop (Phase 3, propose->measure->observe)
is :func:`nestforge.optimizers.run_agent_loop`, re-exported here.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, List, Optional

import dace
from nestforge.fusion_arms import apply_fusion, first_fusion
from nestforge.optimizers import AgenticOptimizer, Outcome, run_agent_loop

#: A granularity move: mutate the SDFG in place, return True if a move applied, False at the fixed point.
GranularityStep = Callable[[dace.SDFG], bool]

#: What one measured round yields: ``measure(sdfg) -> Outcome`` (build + validate bit-exact + time).
Measure = Callable[[dace.SDFG], Outcome]


def default_fuse_step(sdfg: dace.SDFG) -> bool:
    """Apply ONE re-enumerated fusion move (the Phase-1 arm) in place; True if a move applied, False at
    the fusion fixed point.

    The default Phase-4 lever: fuse back up from a fine granularity one move at a time, re-enumerating
    after each (applying one stales the rest). Swap in a fission step for the other direction.
    """
    move = first_fusion(sdfg)
    if move is None:
        return False
    apply_fusion(sdfg, move)
    return True


def best_outcome(outcomes: List[Outcome]) -> Optional[Outcome]:
    """The fastest bit-exact (``ok``) outcome, or ``None`` if none validated. The loop's selection rule --
    a candidate that lost the correctness gate never competes on speed."""
    valid = [o for o in outcomes if o.ok]
    return min(valid, key=lambda o: o.median_us) if valid else None


def improved(prior: List[Outcome], candidate: Outcome) -> bool:
    """Did ``candidate`` beat every prior bit-exact outcome? The stop rule -- a round that does not
    improve ends the loop. A non-bit-exact candidate never improves (it never competes on speed)."""
    if not candidate.ok:
        return False
    best = best_outcome(prior)
    return best is None or candidate.median_us < best.median_us


@dataclass(slots=True)
class FeedbackResult:
    """The loop's record: every round's outcome, the winner, the rounds run, and the granularity SDFG the
    winner was measured at (snapshotted whenever a round set a new best, so ``sdfg`` IS the best
    granularity, not merely the last one examined)."""
    outcomes: List[Outcome]
    best: Optional[Outcome]
    rounds: int
    sdfg: dace.SDFG


def run_feedback_loop(sdfg: dace.SDFG,
                      measure: Measure,
                      apply_move: GranularityStep = default_fuse_step,
                      max_rounds: int = 8) -> FeedbackResult:
    """Phase 4: adjust granularity from measured feedback until it stops paying off.

    Measure the current granularity, then each round apply one granularity move (default: re-enumerate +
    fuse), re-measure, and STOP when the round yields no improvement or no move remains. ``measure(sdfg)
    -> Outcome`` is the caller's build + validate + time step, so CI drives the whole loop with a fake
    measure and no compiler. ``max_rounds`` is the hard bound: a loop that never stalls must still
    terminate.
    """
    if max_rounds < 1:
        raise ValueError(f"max_rounds must be >= 1, got {max_rounds}")
    outcomes: List[Outcome] = [measure(sdfg)]
    best_sdfg = copy.deepcopy(sdfg)  # round-0 granularity is the best so far
    rounds = 0
    for _ in range(max_rounds):
        if not apply_move(sdfg):
            break  # granularity fixed point: no move left
        rounds += 1
        candidate = measure(sdfg)
        better = improved(outcomes, candidate)
        outcomes.append(candidate)
        if better:
            best_sdfg = copy.deepcopy(sdfg)  # new global best -- snapshot this granularity
        else:
            break  # this round did not help -- stop
    return FeedbackResult(outcomes=outcomes, best=best_outcome(outcomes), rounds=rounds, sdfg=best_sdfg)


__all__ = [
    "GranularityStep",
    "Measure",
    "default_fuse_step",
    "best_outcome",
    "improved",
    "FeedbackResult",
    "run_feedback_loop",
    # per-nest inner loop (from nestforge.optimizers)
    "Outcome",
    "AgenticOptimizer",
    "run_agent_loop",
]
