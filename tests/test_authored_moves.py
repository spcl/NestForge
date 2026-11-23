# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Authored moves: a catalogue folder of DaCe pattern transformations that sessions load as the family ``authored``.
A session offers an authored move where DaCe's matcher finds it, applies it, and rolls it back like any move."""

from pathlib import Path

import dace
import pytest

from tactics.records import MoveStatus, Refusal
from tactics.server import CALLS
from tactics.session import Session
from tactics.stages.authored import CatalogueError, load_catalogue

N = dace.symbol("N", dtype=dace.int64)
SIZE = 16
EXPAND = """
from dace.transformation.dataflow import MapExpansion

TRANSFORMATION = MapExpansion
ROWS = ("map_entry",)
TEXT = "a doall over several dimensions becomes one doall per dimension, nested"
"""


@dace.program
def grid(a: dace.float64[N, N], b: dace.float64[N, N], c: dace.float64[N]) -> None:
    for i, j in dace.map[0:N, 0:N]:
        b[i, j] = a[i, j] * 2.0
    for i in dace.map[0:N]:
        c[i] = a[i, 0] + 1.0


def catalogue(tmp_path: Path, entries: dict[str, str]) -> Path:
    folder = tmp_path / "catalogue"
    for kind, source in entries.items():
        (folder / kind).mkdir(parents=True)
        (folder / kind / "move.py").write_text(source)
    return folder


def session(tmp_path: Path, disabled: tuple[str, ...] = ()) -> Session:
    moves = load_catalogue(catalogue(tmp_path, {"expand-doall": EXPAND}))
    sut = Session(
        grid.to_sdfg(simplify=True),
        work_dir=str(tmp_path),
        sizes={"N": SIZE},
        catalogue=moves,
        disabled_families=disabled,
    )
    sut.canonicalize()
    return sut


def doalls(sut: Session) -> int:
    return sum(isinstance(n, dace.nodes.MapEntry) for n, _ in sut.sdfg.all_nodes_recursive())


def test_an_authored_move_is_offered_where_it_matches_applies_and_rolls_back(tmp_path: Path) -> None:
    sut = session(tmp_path)
    tree, before = sut.describe(), doalls(sut)

    (move,) = sut.list_moves("expand-doall")
    done = sut.apply_move("expand-doall", list(move.labels))

    assert done.status is MoveStatus.APPLIED
    assert doalls(sut) == before + 1, "the two-dimensional doall became two nested doalls"
    sut.apply_move("rollback", [])
    assert sut.describe() == tree and doalls(sut) == before


def test_the_kinds_listing_names_an_authored_move_and_its_row(tmp_path: Path) -> None:
    sut = session(tmp_path)

    listed = CALLS["list_move_kinds"](sut)
    assert isinstance(listed, list)
    kinds = {kind.kind: kind for kind in listed}

    assert kinds["expand-doall"].rows == ("doall",) and kinds["expand-doall"].runs == "MapExpansion"


def test_a_row_the_transformation_rejects_and_a_switched_off_family_are_refused(tmp_path: Path) -> None:
    sut = session(tmp_path)
    (offered,) = sut.list_moves("expand-doall")
    one_dimensional = next(
        label
        for label, (obj, _) in sut.row_index().items()
        if isinstance(obj, dace.nodes.MapEntry) and label != offered.labels[0]
    )
    off = session(tmp_path / "off", disabled=("authored",))

    rejected = sut.apply_move("expand-doall", [one_dimensional])
    switched = off.apply_move("expand-doall", list(offered.labels))

    assert (rejected.status, rejected.refusal) == (MoveStatus.ILLEGAL, Refusal.WRONG_ROW_KIND)
    assert (switched.status, switched.refusal) == (MoveStatus.ILLEGAL, Refusal.FAMILY_OFF)
    assert off.list_moves("expand-doall") == []


def test_a_catalogue_entry_without_its_fields_does_not_load(tmp_path: Path) -> None:
    folder = catalogue(tmp_path, {"broken": "TRANSFORMATION = None\n"})

    with pytest.raises(CatalogueError, match="broken"):
        load_catalogue(folder)
