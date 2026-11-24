# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Random moves at random places of the canonicalized bench cloudsc_monolith either apply and keep the whole program
within the bench's grading tolerance, or are refused and leave the program untouched.

The program is the bench's Python cloudsc_monolith, canonicalized once up to fusion (stage 1) as an agent session
starts it. Each seed walks a copy of it. A move draws its kind first, since doall-fusion is almost every legal
candidate, then its place: either a move ``list_moves`` offers or any rows of the kind's shape. Applied moves are
checked in batches, as one codegen and build costs over a minute while a run costs milliseconds. A failure names the
seed and the moves since the last passing check. The same walk then rolls the applied moves back, one at a time, and
gets back each program before them, the canonical one at the end.
"""

from collections.abc import Iterator

import numpy as np
import pytest
from helpers import program_mismatch

from tactics import session as session_module
from tactics.records import MoveStatus
from tactics.session import Session
from tactics.stages.moves import MOVE_SHAPES, ROLLBACK, RUN_ON_KINDS
from tactics.stages.performance import Reference

pytestmark = pytest.mark.cloudsc

NSEEDS = 3
ATTEMPTS = 16
CHECK_EVERY = 4

Program = tuple[Session, dict[str, np.ndarray], Reference]


@pytest.fixture(scope="module")
def canonical(cloudsc_monolith: Program) -> Program:
    assert not (why := program_mismatch(*cloudsc_monolith)), f"the canonicalized program is already wrong: {why}"
    return cloudsc_monolith


def random_place(session: Session, kind: str, rng: np.random.Generator) -> tuple[str, ...] | None:
    """Labels of one place for ``kind``: half the time a move ``list_moves`` offers, which must apply and keep the
    program correct, else any rows of the kind's shape, which probe the refusals."""
    if rng.random() < 0.5:
        listed = session.list_moves(kind)
        if listed:
            return listed[int(rng.integers(len(listed)))].labels
    labels = []
    for wanted in MOVE_SHAPES[kind]:
        pool = [label for label, (obj, _) in session.row_index().items() if isinstance(obj, wanted)]
        if not pool:
            return None
        labels.append(pool[int(rng.integers(len(pool)))])
    return tuple(labels)


def moves(session: Session, rng: np.random.Generator) -> Iterator[tuple[str, tuple[str, ...]]]:
    """``ATTEMPTS`` random ``(kind, labels)``, skipping a kind that has no place in the program; the moves that place
    kernels belong to the scopes phase and take kernels, not rows (``test_run_on``)."""
    kinds = [kind for kind in MOVE_SHAPES if kind not in RUN_ON_KINDS]
    for _ in range(ATTEMPTS):
        kind = kinds[int(rng.integers(len(kinds)))]
        labels = random_place(session, kind, rng)
        if labels is not None:
            yield kind, labels


def check(session: Session, inputs: dict[str, np.ndarray], reference: Reference, seed: int, pending: list[str]) -> None:
    why = program_mismatch(session, inputs, reference)
    assert not why, f"seed={seed}: wrong after\n  " + "\n  ".join(pending) + f"\n{why}"
    pending.clear()


@pytest.mark.parametrize("seed", range(NSEEDS))
def test_random_moves_keep_the_program_correct_or_are_refused_and_roll_back_to_each_earlier_program(
    canonical: Program, seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One random walk per seed checks both facts, as a walk costs minutes: the moves keep the program correct or are
    refused untouched, and rolling the applied ones back gives back each program before them."""
    base, inputs, reference = canonical
    monkeypatch.setattr(session_module, "UNDO_DEPTH", ATTEMPTS)  # every applied move stays in reach
    session = base.trial()
    rng = np.random.default_rng(seed)
    pending: list[str] = []
    before: list[tuple[str, str]] = []
    applied = refused = 0
    for kind, labels in moves(session, rng):
        state = (session.sdfg.hash_sdfg(), session.describe())
        result = session.apply_move(kind, labels)
        step = f"{kind} {' '.join(labels)}"
        if result.status is MoveStatus.ILLEGAL:
            refused += 1
            assert session.sdfg.hash_sdfg() == state[0], (
                f"seed={seed}: refused {step} ({result.reason}) still changed the program"
            )
            continue
        assert result.status is MoveStatus.APPLIED, f"seed={seed}: {step} is {result.status.name}: {result.reason}"
        session.sdfg.validate()
        applied += 1
        if kind == ROLLBACK:  # an undo pops the change it undid; it is no change of its own to roll back
            if before:
                before.pop()
        else:
            before.append(state)
        pending.append(f"{step} -> {result.reason}")
        if len(pending) == CHECK_EVERY:
            check(session, inputs, reference, seed, pending)
    if pending:
        check(session, inputs, reference, seed, pending)
    print(f"seed={seed}: {applied} applied, {refused} refused")
    assert applied > 0, f"seed={seed}: no move applied ({refused} refused), so nothing was checked"

    assert before, f"seed={seed}: every applied move was a rollback, so nothing was rolled back"
    back = []
    for _ in before:
        session.apply_move(ROLLBACK, [])
        back.append((session.sdfg.hash_sdfg(), session.describe()))

    assert back == before[::-1], f"seed={seed}: a rollback did not give back the program before its move"
    assert not (why := program_mismatch(session, inputs, reference)), f"seed={seed}: wrong after rollback: {why}"
