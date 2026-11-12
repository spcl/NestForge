# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A call that fails leaves the session as it was: the program, the row labels and the record of applied moves."""

import json

import dace
import numpy as np
import pytest
from test_rounds import doalls, fresh

from nestforge import server
from nestforge import session as session_module
from nestforge.records import MoveStatus
from nestforge.server import answer
from nestforge.session import Session
from nestforge.stages import scopes

N = dace.symbol("N", dtype=dace.int64)
SIZE = 16
NO_DATA: tuple[dict[str, np.ndarray], None] = ({}, None)


@dace.program
def nest_in_loop(a: dace.float64[N, N], b: dace.float64[N, N]):
    for j in range(1, N):
        for i in dace.map[0:N]:
            a[j, i] = a[j - 1, i] + b[j, i]


def session(tmp_path) -> Session:
    sut = Session(nest_in_loop.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": SIZE})
    sut.canonicalize()
    return sut


def fail(*args: object, **kwargs: object) -> None:
    raise KeyError("injected")


def half_then_fail(sdfg: dace.SDFG, *args: object) -> None:
    sdfg.add_state("half_a_change")
    fail()


def without_guids(value: object) -> object:
    """A deep copy of an SDFG gives its memlets new guids; everything else must match."""
    if isinstance(value, dict):
        return {key: without_guids(item) for key, item in value.items() if key != "guid"}
    return [without_guids(item) for item in value] if isinstance(value, list) else value


def program(sut: Session) -> str:
    return json.dumps(without_guids(sut.sdfg.to_json()), sort_keys=True, default=str)


def first_move(sut: Session, kind: str) -> tuple[str, ...]:
    (move, *_) = sut.list_moves(kind)
    return move.labels


@pytest.mark.parametrize("kind, sizes", [("tile-for", (8,)), ("tile-doall", (8,))])
def test_a_move_that_fails_after_its_rewrite_is_undone(tmp_path, monkeypatch, kind, sizes):
    sut = session(tmp_path)
    labels, before, tree = first_move(sut, kind), program(sut), sut.describe()

    with monkeypatch.context() as patch:
        patch.setattr(session_module, "structural_cleanup", half_then_fail)
        with pytest.raises(KeyError, match="injected"):
            sut.apply_move(kind, labels, tile_sizes=sizes)

    assert program(sut) == before
    assert sut.describe() == tree
    assert sut.applied == []
    assert sut.apply_move(kind, labels, tile_sizes=sizes).status is MoveStatus.APPLIED


def test_default_scopes_that_fail_partway_leave_no_half_outlined_nest(tmp_path, monkeypatch):
    sut = session(tmp_path)
    before, tree = program(sut), sut.describe()

    with monkeypatch.context() as patch:
        patch.setattr(scopes, "prepare", fail)
        with pytest.raises(KeyError, match="injected"):
            sut.run_defaults("scopes")

    assert program(sut) == before
    assert sut.describe() == tree
    assert [scope.name for scope in sut.run_defaults("scopes")]


@pytest.mark.parametrize("stage", ["moves", "finish"])
def test_a_default_stage_that_fails_is_undone(tmp_path, monkeypatch, stage):
    sut = session(tmp_path)
    before = program(sut)
    monkeypatch.setattr(session_module, "fuse_and_finish", half_then_fail)
    monkeypatch.setattr(session_module, "finish", half_then_fail)

    with pytest.raises(KeyError, match="injected"):
        sut.run_defaults(stage)

    assert program(sut) == before


def test_a_kernel_whose_program_cannot_be_measured_is_undone_and_says_so(tmp_path, monkeypatch):
    sut = session(tmp_path)
    sut.default_scopes()
    (kernel, *_) = sut.list_kernels()
    source = sut.kernel_source(kernel.name)
    before = dict(sut.builds)
    monkeypatch.setattr(Session, "performance", fail)

    reply = answer(sut, "write_kernel", {"name": kernel.name, "source": source, "language": "cpp"}, NO_DATA)

    assert (
        reply.error == "KeyError: 'injected' (the program is unchanged; the session is intact, other calls still work)"
    )
    assert sut.builds == before


def test_a_move_that_crashes_is_named_rolled_back_and_the_next_move_applies(tmp_path, monkeypatch):
    sut = session(tmp_path)
    labels, before = first_move(sut, "tile-for"), program(sut)
    move = {"kind": "tile-for", "labels": labels, "tile_sizes": (8,)}

    with monkeypatch.context() as patch:
        patch.setattr(session_module, "structural_cleanup", half_then_fail)
        reply = answer(sut, "apply_move", move, None)

    assert reply.error == (
        f"transformation tile-for on {', '.join(labels)} crashed: KeyError: 'injected'. It was rolled back; "
        "the session is intact, try another move."
    )
    assert program(sut) == before
    assert answer(sut, "apply_move", move, None).result.status is MoveStatus.APPLIED


def test_a_failed_read_claims_nothing_about_the_program(tmp_path, monkeypatch):
    sut = session(tmp_path)
    monkeypatch.setitem(server.CALLS, "get_cpf", fail)

    assert answer(sut, "get_cpf", {}, None).error == "KeyError: 'injected'"


def test_a_call_that_succeeds_keeps_its_change(tmp_path):
    sut = session(tmp_path)
    labels = first_move(sut, "tile-for")

    reply = answer(sut, "apply_move", {"kind": "tile-for", "labels": labels, "tile_sizes": (8,)}, None)

    assert not reply.error
    assert reply.result.status is MoveStatus.APPLIED
    assert [move.kind for move in sut.applied] == ["tile-for"]


def test_every_timed_change_is_guarded():
    assert server.TIMED - {"get_performance"} <= server.CHANGING <= set(server.CALLS)


def test_a_refused_scope_leaves_the_program_a_new_round_returns_to():
    """A refused call is not journaled, so a session that repeats the others must reach the same program."""
    refused, repeated = fresh(), fresh()
    for sut in (refused, repeated):
        (first, *_) = doalls(sut)
        sut.unset_scope(sut.set_scope([first]).reason)
    first, _, last = doalls(refused)
    assert refused.set_scope([first, last]).status is MoveStatus.ILLEGAL

    assert refused.run_defaults("round") == repeated.run_defaults("round")
    assert refused.applied == repeated.applied


if __name__ == "__main__":
    import pathlib
    import tempfile

    with pytest.MonkeyPatch.context() as patch:
        for kind, sizes in (("tile-for", (8,)), ("tile-doall", (8,))):
            test_a_move_that_fails_after_its_rewrite_is_undone(pathlib.Path(tempfile.mkdtemp()), patch, kind, sizes)
        test_default_scopes_that_fail_partway_leave_no_half_outlined_nest(pathlib.Path(tempfile.mkdtemp()), patch)
    for stage in ("moves", "finish"):
        with pytest.MonkeyPatch.context() as patch:
            test_a_default_stage_that_fails_is_undone(pathlib.Path(tempfile.mkdtemp()), patch, stage)
    with pytest.MonkeyPatch.context() as patch:
        test_a_kernel_whose_program_cannot_be_measured_is_undone_and_says_so(pathlib.Path(tempfile.mkdtemp()), patch)
    with pytest.MonkeyPatch.context() as patch:
        test_a_move_that_crashes_is_named_rolled_back_and_the_next_move_applies(pathlib.Path(tempfile.mkdtemp()), patch)
    with pytest.MonkeyPatch.context() as patch:
        test_a_failed_read_claims_nothing_about_the_program(pathlib.Path(tempfile.mkdtemp()), patch)
    test_a_call_that_succeeds_keeps_its_change(pathlib.Path(tempfile.mkdtemp()))
    test_every_timed_change_is_guarded()
    test_a_refused_scope_leaves_the_program_a_new_round_returns_to()
