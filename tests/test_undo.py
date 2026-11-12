# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A move can be taken back, a move that builds an invalid program is refused, and a move whose check raises is not
listed while every other move is."""

import dace
import pytest
from dace.sdfg import InvalidSDFGError
from test_failed_calls import first_move, program, session

from nestforge import session as session_module
from nestforge.records import MoveStatus
from nestforge.server import CALLS, CHANGING, answer
from nestforge.stages import moves


def fresh_rows(sut, before: set[str]) -> set[str]:
    return set(sut.row_index()) - before


def test_an_undone_move_leaves_the_program_its_labels_and_its_record_as_before(tmp_path):
    sut = session(tmp_path)
    labels, before, tree, rows = first_move(sut, "tile-for"), program(sut), sut.describe(), set(sut.row_index())
    sut.apply_move("tile-for", labels, tile_sizes=(8,))
    made = fresh_rows(sut, rows)

    undone = sut.undo()

    assert (undone.status, undone.kind, undone.labels, undone.reason) == (
        MoveStatus.APPLIED,
        "undo",
        labels,
        "undid tile-for",
    )
    assert program(sut) == before
    assert sut.describe() == tree
    assert sut.applied == []
    # the rows are back under their labels, and the labels the undone move made are never handed out again
    assert sut.apply_move("tile-for", labels, tile_sizes=(8,)).status is MoveStatus.APPLIED
    assert made and not made & fresh_rows(sut, rows)


def test_each_undo_takes_back_one_move_and_none_is_left_after_the_first_move(tmp_path):
    sut = session(tmp_path)
    start = program(sut)
    sut.apply_move("tile-for", first_move(sut, "tile-for"), tile_sizes=(8,))
    tiled = program(sut)
    sut.apply_move("tile-doall", first_move(sut, "tile-doall"), tile_sizes=(4,))

    sut.undo()
    assert program(sut) == tiled
    assert [move.kind for move in sut.applied] == ["tile-for"]
    sut.undo()
    assert program(sut) == start

    refused = sut.undo()
    assert refused.status is MoveStatus.ILLEGAL
    assert refused.reason.startswith("no move to undo")
    assert program(sut) == start


def test_an_undo_is_a_change_the_journal_repeats(tmp_path):
    sut = session(tmp_path)
    before = program(sut)
    labels = first_move(sut, "tile-for")
    answer(sut, "apply_move", {"kind": "tile-for", "labels": labels, "tile_sizes": (8,)}, None)

    reply = answer(sut, "undo_move", {}, None)

    assert "undo_move" in CHANGING and "undo_move" in CALLS
    assert not reply.error and reply.result.status is MoveStatus.APPLIED
    assert program(sut) == before


def dangling_read(sdfg: dace.SDFG, *args: object) -> None:
    """Leave an access node of an array the program does not have: a program DaCe's validation refuses."""
    next(iter(sdfg.states())).add_access("no_such_array")


def test_a_move_that_builds_an_invalid_program_is_refused_and_undone(tmp_path, monkeypatch):
    sut = session(tmp_path)
    labels, before = first_move(sut, "tile-for"), program(sut)
    monkeypatch.setattr(session_module, "structural_cleanup", dangling_read)

    with pytest.raises(InvalidSDFGError):
        sut.apply_move("tile-for", labels, tile_sizes=(8,))

    assert program(sut) == before
    assert sut.applied == []
    assert sut.undo().status is MoveStatus.ILLEGAL


def raising(*rows: object, **options: object) -> None:
    raise KeyError("injected")


def test_a_move_whose_check_raises_is_not_listed_and_every_other_move_is(tmp_path, monkeypatch, capsys):
    sut = session(tmp_path)
    listed = {(move.kind, move.labels) for move in sut.list_moves()}
    monkeypatch.setitem(moves.PLANNERS, "tile-for", raising)

    left = {(move.kind, move.labels) for move in sut.list_moves()}

    assert {kind for kind, _ in listed - left} == {"tile-for"}
    assert left == {move for move in listed if move[0] != "tile-for"} and left
    assert "tile-for" in capsys.readouterr().err


if __name__ == "__main__":
    import pathlib
    import tempfile

    test_an_undone_move_leaves_the_program_its_labels_and_its_record_as_before(pathlib.Path(tempfile.mkdtemp()))
    test_each_undo_takes_back_one_move_and_none_is_left_after_the_first_move(pathlib.Path(tempfile.mkdtemp()))
    test_an_undo_is_a_change_the_journal_repeats(pathlib.Path(tempfile.mkdtemp()))
    with pytest.MonkeyPatch.context() as patch:
        test_a_move_that_builds_an_invalid_program_is_refused_and_undone(pathlib.Path(tempfile.mkdtemp()), patch)
