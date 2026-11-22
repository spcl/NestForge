# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``rollback`` after any move gives back the program before it: the same SDFG, the same row labels and the same tree
text. The skill page tells the agent to undo a move with ``rollback``, never with its inverse move, so this must hold
for every move kind. Each listed move of every fixture program of the session tests is applied and rolled back."""

import hashlib
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import comm_rank_programs
import dace
import pytest
import test_doall_tiling
import test_hoist_invariant_moves as hoist_fixtures
import test_inline_library
import test_lib_motion_moves
import test_move_soundness
import test_persistent_kernel
import test_pipeline_depth
import test_run_on
import test_scope_moves as scope_fixtures
import test_session_moves
import test_sink_invariant_code
import test_skew_wavefront
import test_split_doall
import test_unroll_config_moves
import test_vectorize_doall_and_c_view as vectorize_fixtures
from dace.frontend.python.parser import DaceProgram

from nestforge.records import MoveStatus
from nestforge.session import UNDONE_BY, Move, MoveResult, Session
from nestforge.stages.moves import MOVE_SHAPES, ROLLBACK, config_names
from nestforge.stages.vectorize import VECTORIZE_DOALL

#: Kinds the fixture programs cannot apply as listed (they need values, a scope or a device); each has its own case
#: below, so every kind is applied and rolled back somewhere in this file.
OWN_CASES = {
    ROLLBACK: "test_rollback_twice_takes_back_two_moves_and_with_nothing_kept_changes_nothing",
    "propagate-config": "test_rollback_after_propagate_config",
    "set-schedule-cpu": "test_rollback_after_a_scope_move",
    "set-schedule-gpu": "test_rollback_after_a_scope_move",
    "stage-loads-shared": "test_rollback_after_a_scope_move",
    "vectorize-doall": "test_rollback_after_vectorizing_a_doall",
    "persistent-kernel": "test_rollback_after_a_persistent_kernel",
    "set-fp-mode": "test_rollback_after_set_fp_mode",
    "copy-to-tma": "test_rollback_after_copy_to_tma",
    "set-pipeline-depth": "test_rollback_after_set_pipeline_depth",
    "split-doall": "test_rollback_after_split_doall",
    "split-blocking": "test_rollback_after_split_blocking",
    "run-on-cpu": "test_rollback_after_a_placement_move",
    "run-on-gpu": "test_rollback_after_a_placement_move",
    "run-on-cpu-persistent": "test_rollback_after_a_placement_move",
    "run-on-gpu-persistent": "test_rollback_after_a_placement_move",
    "untile-doall": "test_the_inverse_of_an_earlier_move_is_applied_as_a_move_and_rolled_back",
    "untile-for": "test_the_inverse_of_an_earlier_move_is_applied_as_a_move_and_rolled_back",
}

#: Fields of a listed move that are notes for the agent, not arguments of apply_move.
NOT_ARGUMENTS = frozenset({"kind", "labels", "current"})


#: The test modules whose module-level programs are the fixtures here.
FIXTURE_MODULES = (
    test_session_moves,
    test_lib_motion_moves,
    test_skew_wavefront,
    test_unroll_config_moves,
    test_inline_library,
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


def content_of(sdfg: dace.SDFG) -> str:
    """A digest of ``sdfg``'s content, node GUIDs aside."""
    content = json.dumps(without_guids(json.loads(json.dumps(sdfg.to_json(), default=str))), sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


def state_of(session: Session) -> tuple[str, list[str], str, list[tuple[str, str, list[tuple[str, ...]]]]]:
    """The program as the agent sees it: the SDFG (its content), the row labels, the tree text, and each kernel's
    body as placed with the schedules its scope's moves report."""
    kernels = [
        (
            info.name,
            content_of(session.kernel(info.name).standalone_sdfg),
            sorted((move.kind, *move.labels, str(move.current)) for move in session.list_moves(scope=info.name)),
        )
        for info in session.list_kernels()
    ]
    return content_of(session.sdfg), sorted(session.row_index()), session.describe(), kernels


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
    parts = ("SDFG", "row labels", "tree text", "kernels")
    return True, ", ".join(part for part, a, b in zip(parts, before, after, strict=True) if a != b)


def uncovered_moves_after(session: Session, name: str, found: list[Outcome], outer: Move) -> None:
    """Apply and roll back one listed move of each kind not yet covered, on the program as it is now, after ``outer``:
    a kind only a move makes possible is covered one move deep. The exact inverse of ``outer`` is left out: the
    session answers it with ``rollback`` itself (its own test below)."""
    covered = {kind for _, kind, _, _ in found}
    for move in session.list_moves():
        if move.kind in OWN_CASES or move.kind in covered or move.kind == UNDONE_BY.get(outer.kind):
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
            if move.kind in OWN_CASES:
                continue
            applied, differs = applied_then_rolled_back(
                session, move, lambda: uncovered_moves_after(session, name, found, move)
            )
            if applied:
                found.append((name, move.kind, move.labels, differs))
    return found


def test_rollback_after_any_move_gives_back_the_same_program_labels_and_tree(outcomes: list[Outcome]) -> None:
    broken = [(program, kind, labels, differs) for program, kind, labels, differs in outcomes if differs]

    assert outcomes, "no move applied"
    assert not broken, broken


def test_every_move_kind_is_applied_and_rolled_back_by_some_fixture(outcomes: list[Outcome]) -> None:
    covered = {kind for _, kind, _, _ in outcomes}

    missing = set(MOVE_SHAPES) - covered - OWN_CASES.keys()

    assert not missing, f"no fixture program here applies {sorted(missing)}; add one"
    assert set(OWN_CASES.values()) <= set(globals()), "a kind's own case is missing"


def differs_after_rollback(session: Session, apply: Callable[[], MoveResult]) -> str:
    """What differs once ``apply``'s change is rolled back (empty: nothing); it must have applied."""
    before = state_of(session)
    result = apply()
    assert result.status is MoveStatus.APPLIED, result
    session.apply_move(ROLLBACK, [])
    parts = ("SDFG", "row labels", "tree text", "kernels")
    return ", ".join(part for part, a, b in zip(parts, before, state_of(session), strict=True) if a != b)


def test_rollback_after_a_persistent_kernel(tmp_path: Path) -> None:
    session, kernel = test_persistent_kernel.scoped(tmp_path)
    ((loop,),) = test_persistent_kernel.offered(session, kernel)

    assert differs_after_rollback(session, lambda: session.apply_move("persistent-kernel", [loop], scope=kernel)) == ""


@pytest.mark.parametrize("kind", ["run-on-cpu", "run-on-gpu", "run-on-cpu-persistent", "run-on-gpu-persistent"])
def test_rollback_after_a_placement_move(tmp_path: Path, kind: str) -> None:
    session, kernels = test_run_on.scoped(test_run_on.two_time_loops, tmp_path, gpu=True)

    assert differs_after_rollback(session, lambda: session.apply_move(kind, kernels)) == ""


def test_rollback_twice_takes_back_two_moves_and_with_nothing_kept_changes_nothing(tmp_path: Path) -> None:
    session = Session(test_session_moves.guarded_map.to_sdfg(simplify=True), work_dir=str(tmp_path))
    start = state_of(session)
    nothing = session.apply_move(ROLLBACK, [])
    for move in session.list_moves()[:2]:
        session.apply_move(
            move.kind, move.labels, **{k: v for k, v in move.model_dump().items() if k not in NOT_ARGUMENTS}
        )

    session.apply_move(ROLLBACK, [])
    session.apply_move(ROLLBACK, [])

    assert nothing.status is MoveStatus.APPLIED and "nothing to roll back" in nothing.reason
    assert state_of(session) == start


def test_rollback_after_propagate_config(tmp_path: Path) -> None:
    session = Session(test_unroll_config_moves.repeat.to_sdfg(simplify=True), work_dir=str(tmp_path))
    values = {name: 3 for name in config_names(session.sdfg) if name == "K"}

    differs = differs_after_rollback(session, lambda: session.apply_move("propagate-config", [], config_values=values))

    assert not differs, differs


@pytest.mark.parametrize(
    ("kind", "device", "arguments"),
    [
        ("set-schedule-cpu", "cpu", {"schedule": "dynamic", "chunk": 16, "threads": 4}),
        ("set-schedule-gpu", "gpu", {"block": [128]}),
        ("stage-loads-shared", "gpu", {"block": [64]}),
    ],
)
def test_rollback_after_a_scope_move(tmp_path: Path, kind: str, device: str, arguments: dict[str, Any]) -> None:
    session, first, second = scope_fixtures.two_kernels(tmp_path, gpu=device == "gpu")
    session.place({first: device, second: device})
    if device == "gpu":
        session.insert_copies()
    (move,) = session.list_moves(kind, scope=first)

    differs = differs_after_rollback(session, lambda: session.apply_move(kind, move.labels, scope=first, **arguments))

    assert not differs, differs


def test_rollback_after_set_fp_mode(tmp_path: Path) -> None:
    session, first, _ = scope_fixtures.two_kernels(tmp_path, gpu=False)

    differs = differs_after_rollback(
        session, lambda: session.apply_move("set-fp-mode", [], fp_mode="fast-math", scope=first)
    )

    assert not differs, differs


def test_rollback_after_vectorizing_a_doall(tmp_path: Path) -> None:
    session = vectorize_fixtures.session(tmp_path)
    move = session.list_moves(VECTORIZE_DOALL)[0]

    differs = differs_after_rollback(
        session, lambda: session.apply_move(VECTORIZE_DOALL, move.labels, lanes=[4], isa="scalar")
    )

    assert not differs, differs


def listed_move(session: Session, kind: str) -> Move:
    return session.list_moves(kind)[0]


def apply_listed(session: Session, kind: str) -> MoveResult:
    move = listed_move(session, kind)
    return session.apply_move(
        kind, move.labels, **{k: v for k, v in move.model_dump().items() if k not in NOT_ARGUMENTS}
    )


def test_the_inverse_of_the_last_move_on_the_rows_it_made_is_answered_by_rollback(tmp_path: Path) -> None:
    session = Session(scope_fixtures.chain.to_sdfg(simplify=True), work_dir=str(tmp_path))
    before = state_of(session)
    assert apply_listed(session, "tile-doall").status is MoveStatus.APPLIED

    undone = apply_listed(session, "untile-doall")

    assert undone.status is MoveStatus.APPLIED and undone.reason.startswith("rollback:"), undone
    assert state_of(session) == before


@pytest.mark.parametrize(
    ("program", "tile", "untile"),
    [
        (scope_fixtures.chain, "tile-doall", "untile-doall"),
        (test_session_moves.calls_then_loops, "tile-for", "untile-for"),
    ],
)
def test_the_inverse_of_an_earlier_move_is_applied_as_a_move_and_rolled_back(
    tmp_path: Path, program: DaceProgram, tile: str, untile: str
) -> None:
    session = Session(program.to_sdfg(simplify=True), work_dir=str(tmp_path))
    first, second = (move.labels for move in session.list_moves(tile)[:2])
    session.apply_move(tile, first, tile_sizes=[8])
    tiled_first = set(session.row_index())
    session.apply_move(tile, second, tile_sizes=[8])
    (pair,) = [move.labels for move in session.list_moves(untile) if set(move.labels) <= tiled_first]

    differs = differs_after_rollback(session, lambda: session.apply_move(untile, pair))

    assert session.applied[-1].kind == tile, "the untile was a move of its own, and only it was taken back"
    assert not differs, differs


def test_rollback_after_copy_to_tma(tmp_path: Path) -> None:
    session, first, second = scope_fixtures.two_kernels(tmp_path, gpu=True)
    session.place({first: "gpu", second: "gpu"})
    session.insert_copies()
    (staging,) = session.list_moves("stage-loads-shared", scope=first)
    session.apply_move("stage-loads-shared", staging.labels, scope=first, block=[64])

    with dace.config.set_temporary("compiler", "cuda", "cuda_arch", value="90"):
        (move,) = session.list_moves("copy-to-tma", scope=first)
        differs = differs_after_rollback(session, lambda: session.apply_move("copy-to-tma", move.labels, scope=first))

    assert not differs, differs


def test_rollback_after_set_pipeline_depth(tmp_path: Path) -> None:
    session, kernel = test_pipeline_depth.kernel_on(tmp_path, gpu=True)
    (move,) = session.list_moves("set-pipeline-depth", scope=kernel)

    differs = differs_after_rollback(
        session, lambda: session.apply_move("set-pipeline-depth", move.labels, scope=kernel, depth=3)
    )

    assert not differs, differs


def test_rollback_after_split_doall(tmp_path: Path) -> None:
    session = test_split_doall.session(test_split_doall.stencil, tmp_path)
    label, i = test_split_doall.only_doall(session)

    differs = differs_after_rollback(session, lambda: session.apply_move("split-doall", [label], at={i: "2"}))

    assert not differs, differs


def test_rollback_after_split_blocking(tmp_path: Path) -> None:
    session = Session(comm_rank_programs.ring.to_sdfg(simplify=True), work_dir=str(tmp_path))
    session.canonicalize()
    split = session.list_moves("split-blocking")

    assert len(split) == 2, split  # the ring's Send and Recv
    differs = differs_after_rollback(session, lambda: session.apply_move("split-blocking", split[0].labels))

    assert not differs, differs
