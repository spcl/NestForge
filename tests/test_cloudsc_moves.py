# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Random moves at random places of the canonicalized bench cloudsc_monolith either apply and keep the whole program
within the bench's grading tolerance, or are refused and leave the program untouched.

The program is the bench's Python cloudsc_monolith, canonicalized once up to fusion (stage 1) as an agent session
starts it. Each seed walks a copy of it. A move draws its kind first, since doall-fusion is almost every legal
candidate, then its place: either a move ``list_moves`` offers or any rows of the kind's shape. Applied moves are
checked in batches, as one codegen and build costs over a minute while a run costs milliseconds. A failure names the
seed and the moves since the last passing check.
"""

from collections.abc import Iterator

import numpy as np
import pytest
from helpers import program_mismatch

from nestforge.records import MoveStatus
from nestforge.session import Session
from nestforge.stages.moves import MOVE_SHAPES
from nestforge.stages.performance import Reference

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
    """``ATTEMPTS`` random ``(kind, labels)``, skipping a kind that has no place in the program."""
    kinds = list(MOVE_SHAPES)
    for _ in range(ATTEMPTS):
        kind = kinds[int(rng.integers(len(kinds)))]
        labels = random_place(session, kind, rng)
        if labels is not None:
            yield kind, labels


@pytest.mark.parametrize("seed", range(NSEEDS))
def test_a_random_move_keeps_the_program_correct_or_is_refused(canonical: Program, seed: int) -> None:
    base, inputs, reference = canonical
    session = base.trial()
    rng = np.random.default_rng(seed)
    pending: list[str] = []
    applied = refused = 0
    for kind, labels in moves(session, rng):
        before, epoch = session.sdfg.hash_sdfg(), session.epoch
        result = session.apply_move(kind, labels, epoch)
        step = f"{kind} {' '.join(labels)}"
        if result.status is MoveStatus.ILLEGAL:
            refused += 1
            assert session.epoch == epoch and session.sdfg.hash_sdfg() == before, (
                f"seed={seed}: refused {step} ({result.reason}) still changed the program"
            )
            continue
        assert result.status is MoveStatus.APPLIED, f"seed={seed}: {step} is {result.status.name}: {result.reason}"
        session.sdfg.validate()
        applied += 1
        pending.append(f"{step} -> {result.reason}")
        if len(pending) == CHECK_EVERY:
            why = program_mismatch(session, inputs, reference)
            assert not why, f"seed={seed}: wrong after\n  " + "\n  ".join(pending) + f"\n{why}"
            pending = []
    if pending:
        why = program_mismatch(session, inputs, reference)
        assert not why, f"seed={seed}: wrong after\n  " + "\n  ".join(pending) + f"\n{why}"
    print(f"seed={seed}: {applied} applied, {refused} refused")
