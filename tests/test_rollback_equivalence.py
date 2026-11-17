# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``rollback`` after any move gives back the program before it: the same SDFG, the same row labels and the same tree
text. The skill page tells the agent to undo a move with ``rollback``, never with its inverse move, so this must hold
for every move kind. Each listed move of every fixture program of the session tests is applied and rolled back."""

import hashlib
import inspect
import json
from collections.abc import Callable

import dace
import pytest
import test_doall_tiling
import test_expand_library
import test_hoist_invariant_moves as hoist_fixtures
import test_lib_motion_moves
import test_move_soundness
import test_session_moves
import test_sink_invariant_code
import test_skew_wavefront
import test_unroll_config_moves
from dace.frontend.python.parser import DaceProgram

from nestforge.records import MoveStatus
from nestforge.session import Move, Session
from nestforge.stages.moves import MOVE_SHAPES, ROLLBACK

#: Kinds no fixture program here offers, each with why; every other kind must be applied and rolled back at least
#: once, so a new kind without a fixture fails the coverage test.
NOT_COVERED_HERE = {
    ROLLBACK: "the undo itself",
    "propagate-config": "listed without values; its variants are written and checked in test_unroll_config_moves",
    "set-schedule-cpu": "a scope move, applied inside a placed kernel",
    "set-schedule-gpu": "a scope move, applied inside a placed GPU kernel",
    "stage-loads-shared": "a device move, applied inside a placed GPU kernel",
    "auto-vectorize-scope": "judged when applied, on a copy, by a C++ build",
}

#: Fields of a listed move that are notes for the agent, not arguments of apply_move.
NOT_ARGUMENTS = frozenset({"kind", "labels", "current"})


#: The test modules whose module-level programs are the fixtures here.
FIXTURE_MODULES = (
    test_session_moves,
    test_lib_motion_moves,
    test_skew_wavefront,
    test_unroll_config_moves,
    test_expand_library,
    test_doall_tiling,
    test_move_soundness,
    test_sink_invariant_code,
)


def fixture_programs() -> list[tuple[str, dace.SDFG]]:
    """Every module-level program of the move tests, by module and name, and the hoisting tests' built SDFGs."""
    programs = [
        (f"{module.__name__}.{name}", value.to_sdfg(simplify=True))
        for module in FIXTURE_MODULES
        for name, value in inspect.getmembers(module)
        if isinstance(value, DaceProgram)
    ]
    programs += [
        ("product_in_a_loop", hoist_fixtures.product_in_a_loop(rewrites_a=False)),
        ("product_in_a_doall", hoist_fixtures.product_in_a_doall()),
    ]
    return programs


def without_guids(value: object) -> object:
    """``value`` (an SDFG's JSON) without its ``guid`` fields: a node's identity, which every copy draws anew."""
    if isinstance(value, dict):
        return {key: without_guids(item) for key, item in value.items() if key != "guid"}
    if isinstance(value, list):
        return [without_guids(item) for item in value]
    return value


def state_of(session: Session) -> tuple[str, list[str], str]:
    """The program as the agent sees it: the SDFG (its content), the row labels, and the tree text."""
    content = json.dumps(without_guids(json.loads(json.dumps(session.sdfg.to_json(), default=str))), sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest(), sorted(session.row_index()), session.describe()


#: One applied move and rolled back: ``(program, kind, labels, what differed)``.
Outcome = tuple[str, str, tuple[str, ...], str]


def applied_then_rolled_back(
    session: Session, move: Move, inside: Callable[[], None] = lambda: None
) -> tuple[bool, str]:
    """Apply ``move``, run ``inside`` on the changed program, and roll ``move`` back; whether it applied, and what
    differs afterwards (empty: nothing)."""
    before = state_of(session)
    arguments = {k: v for k, v in move.model_dump().items() if k not in NOT_ARGUMENTS}
    result = session.apply_move(move.kind, move.labels, **arguments)
    if result.status is not MoveStatus.APPLIED:
        return False, ""
    inside()
    session.apply_move(ROLLBACK, [])
    after = state_of(session)
    parts = ("SDFG", "row labels", "tree text")
    return True, ", ".join(part for part, a, b in zip(parts, before, after, strict=True) if a != b)


def uncovered_moves_after(session: Session, name: str, found: list[Outcome]) -> None:
    """Apply and roll back one listed move of each kind not yet covered, on the program as it is now: a kind only a
    move makes possible (``untile-for`` after ``tile-for``) is covered one move deep."""
    covered = {kind for _, kind, _, _ in found}
    for move in session.list_moves():
        if move.kind in NOT_COVERED_HERE or move.kind in covered:
            continue
        applied, differs = applied_then_rolled_back(session, move)
        if applied:
            found.append((name, move.kind, move.labels, differs))
            covered.add(move.kind)


@pytest.fixture(scope="module")
def outcomes(tmp_path_factory: pytest.TempPathFactory) -> list[Outcome]:
    """Every listed move that applied, over every fixture program, and the uncovered kinds one move deep."""
    found: list[Outcome] = []
    for name, sdfg in fixture_programs():
        session = Session(sdfg, work_dir=str(tmp_path_factory.mktemp(name.replace(".", "_"))))
        for move in session.list_moves():
            if move.kind in NOT_COVERED_HERE:
                continue
            applied, differs = applied_then_rolled_back(
                session, move, lambda: uncovered_moves_after(session, name, found)
            )
            if applied:
                found.append((name, move.kind, move.labels, differs))
    return found


def test_rollback_after_any_move_gives_back_the_same_program_labels_and_tree(outcomes):
    broken = [(program, kind, labels, differs) for program, kind, labels, differs in outcomes if differs]

    assert outcomes, "no move applied"
    assert not broken, broken


def test_every_move_kind_is_applied_and_rolled_back_by_some_fixture(outcomes):
    covered = {kind for _, kind, _, _ in outcomes}

    missing = set(MOVE_SHAPES) - covered - NOT_COVERED_HERE.keys()

    assert not missing, f"no fixture program here applies {sorted(missing)}; add one"
