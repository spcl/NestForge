# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""What keeps the moves open, what closes them and what reopens them: a new round, the scopes that fix the kernels,
and a finalize that leaves the session where it was."""

from pathlib import Path

import dace
import numpy as np
import pytest
from dace.libraries.standard.nodes.external_call import external_calls
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion

from nestforge.records import MoveStatus
from nestforge.server import CALLS, Setup, start
from nestforge.session import Session

N = dace.symbol("N", dtype=dace.int64)


@dace.program
def three_stages(A: dace.float64[N], C: dace.float64[N]):
    T = np.empty_like(A)
    U = np.empty_like(A)
    for i in dace.map[0:N]:
        T[i] = A[i] + 1.0
    for i in dace.map[0:N]:
        U[i] = T[i] * 2.0
    for i in dace.map[0:N]:
        C[i] = U[i] - 3.0


@dace.program
def sweeps(A: dace.float64[N]):
    for _ in range(3):
        for i in dace.map[0:N]:
            A[i] = A[i] + 1.0
        A[0] = A[0] * 2.0


def fresh() -> Session:
    return Session(three_stages.to_sdfg(simplify=True))


def doalls(session: Session) -> list[str]:
    session.describe()
    return [label for label, (obj, _) in session.row_index().items() if isinstance(obj, nodes.MapEntry)]


def test_a_new_round_gives_the_moves_back_on_the_same_rows():
    session = fresh()
    moves, tree = session.list_moves(), session.describe()
    session.run_defaults("scopes")
    session.run_defaults("placement")
    assert external_calls(session.sdfg) and session.list_moves() == []

    session.run_defaults("round")

    assert not external_calls(session.sdfg)
    assert session.list_moves() == moves and session.describe() == tree


def test_a_new_round_keeps_the_moves_made_before_it():
    session = fresh()
    move = session.list_moves()[0]
    session.apply_move(move.kind, move.labels)
    tree = session.describe()
    session.run_defaults("placement")

    session.run_defaults("round")

    assert session.describe() == tree and [done.kind for done in session.applied] == [move.kind]


def test_a_new_round_without_kernels_changes_nothing():
    session = fresh()
    tree = session.describe()

    assert session.run_defaults("round") == tree


def test_placement_gives_what_no_scope_covers_its_default_scope():
    session = fresh()
    first, *rest = doalls(session)
    chosen = session.set_scope([first]).reason

    session.run_defaults("placement")

    names = [ext.name for ext in external_calls(session.sdfg)]
    assert chosen in names and len(names) == 1 + len(rest)


def test_a_loop_holding_a_kernel_stays_in_the_program():
    session = Session(sweeps.to_sdfg(simplify=True))
    (doall,) = doalls(session)
    chosen = session.set_scope([doall]).reason

    made = session.default_scopes()

    assert made == [] and [ext.name for ext in external_calls(session.sdfg)] == [chosen]
    assert any(isinstance(block, LoopRegion) for block in session.sdfg.all_control_flow_regions())


def test_moves_close_with_the_first_kernel_and_their_defaults_with_them():
    session = fresh()
    move = session.list_moves()[0]
    first = doalls(session)[0]
    CALLS["set_scope"](session, labels=[first])

    with pytest.raises(ValueError, match="moves are closed"):
        CALLS["apply_move"](session, kind=move.kind, labels=move.labels)
    with pytest.raises(ValueError, match="moves are closed"):
        CALLS["run_defaults"](session, stage="moves")


def test_placement_fixes_the_scopes_until_a_new_round_reopens_them():
    session = fresh()
    labels = doalls(session)
    CALLS["run_defaults"](session, stage="placement")
    name = external_calls(session.sdfg)[0].name

    with pytest.raises(ValueError, match="scopes are fixed"):
        CALLS["set_scope"](session, labels=labels[:1])
    with pytest.raises(ValueError, match="scopes are fixed"):
        CALLS["unset_scope"](session, name=name)

    CALLS["run_defaults"](session, stage="round")

    assert CALLS["set_scope"](session, labels=labels[:1]).status is MoveStatus.APPLIED


def test_default_kernels_keep_their_row_names_through_placement_and_a_new_round():
    session = fresh()
    chosen, *others = doalls(session)
    session.set_scope([chosen])
    (scoped,) = [ext.name for ext in external_calls(session.sdfg)]

    session.run_defaults("placement")
    placed = [ext.name for ext in external_calls(session.sdfg)]
    session.run_defaults("round")
    assert doalls(session) == [chosen, *others]
    session.run_defaults("placement")

    assert set(placed) == {scoped, *(f"extcall_{label}" for label in others)}
    assert {ext.name for ext in external_calls(session.sdfg)} == {f"extcall_{label}" for label in [chosen, *others]}


def test_each_default_that_changed_the_program_is_on_applied_until_a_round_drops_it():
    session = fresh()
    session.run_defaults("scopes")
    session.run_defaults("placement")
    assert [move.kind for move in session.applied] == ["defaults-scopes", "defaults-placement"]

    session.run_defaults("round")

    assert session.applied == [], "the round's program holds neither"


@pytest.mark.parametrize("stage", ["placement", "round"])
def test_a_default_that_fails_halfway_leaves_the_session_as_it_was(stage: str, monkeypatch: pytest.MonkeyPatch) -> None:
    session = fresh()
    if stage == "round":
        session.run_defaults("placement")
    kernels = [ext.name for ext in external_calls(session.sdfg)]
    placed, applied = session.unplaced is not None, list(session.applied)

    def fail(*_: object) -> None:
        raise RuntimeError("failed halfway")

    monkeypatch.setattr(Session, "place" if stage == "placement" else "describe", fail)
    with pytest.raises(RuntimeError, match="failed halfway"):
        session.run_defaults(stage)
    monkeypatch.undo()

    assert [ext.name for ext in external_calls(session.sdfg)] == kernels
    assert (session.unplaced is not None, session.applied) == (placed, applied)
    session.run_defaults("round")
    assert not external_calls(session.sdfg), "the kept program survives the failure"


@pytest.mark.integration
def test_finalizing_leaves_the_moves_open_and_the_program_unscoped(tmp_path: Path) -> None:
    session, _, _ = start(Setup("fuse_diamond", "S", "float64", False, str(tmp_path)), codegen="first")
    moves = session.list_moves()

    final = session.finalize(tmp_path / "out")

    assert final.library.is_file()
    assert session.list_moves() == moves and not external_calls(session.sdfg)
