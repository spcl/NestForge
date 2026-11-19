# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Move families: every move kind is in exactly one, and a run that switches a family off is offered none of its
moves, is refused each with the family's name, and runs none of them by default."""

from typing import cast

import dace
import numpy as np
import pytest
from dace.sdfg import nodes

from nestforge.records import MoveStatus, Refusal
from nestforge.server import CALLS
from nestforge.session import Session
from nestforge.stages.families import ALL, FAMILIES, FAMILY_FREE, checked_families, family_of
from nestforge.stages.moves import MOVE_SHAPES, ROLLBACK, MoveKind, move_kinds

N = dace.symbol("N", dtype=dace.int64)


@dace.program
def vertical_pair(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    tmp = np.empty_like(A)
    for i in dace.map[0:N]:
        tmp[i] = A[i] + B[i]
    for i in dace.map[0:N]:
        C[i] = tmp[i] * 2.0


def canonical(disabled: tuple[str, ...] = ()) -> Session:
    sut = Session(vertical_pair.to_sdfg(simplify=True), disabled_families=disabled)
    sut.canonicalize()
    return sut


def doalls(sut: Session) -> int:
    assert sut.sdfg is not None, "a session holds its program"
    return sum(isinstance(node, nodes.MapEntry) for node, _ in sut.sdfg.all_nodes_recursive())


def test_every_move_kind_but_the_family_free_ones_is_in_exactly_one_family() -> None:
    placed = {kind: [name for name, kinds in FAMILIES.items() if kind in kinds] for kind in MOVE_SHAPES}

    assert {kind: names for kind, names in placed.items() if len(names) != (kind not in FAMILY_FREE)} == {}
    assert set().union(*FAMILIES.values()) <= set(MOVE_SHAPES), "a family names a kind that does not exist"
    assert FAMILY_FREE <= set(MOVE_SHAPES) and family_of(ROLLBACK) is None


def test_all_stands_for_every_family_and_an_unknown_name_refuses_the_session() -> None:
    assert checked_families([ALL]) == frozenset(FAMILIES)

    with pytest.raises(ValueError, match="no move family"):
        Session(vertical_pair.to_sdfg(simplify=True), disabled_families=["fusions"])


def test_a_run_without_fusion_lists_none_refuses_one_by_name_and_fuses_nothing_by_default() -> None:
    sut = canonical(("fusion",))
    pair = canonical().list_moves("doall-fusion")
    assert pair, "the fixture offers a fusion with the family on"

    listed = sut.list_moves()
    refused = sut.apply_move("doall-fusion", list(pair[0].labels))
    sut.run_defaults("moves")

    assert not [move for move in listed if family_of(move.kind) == "fusion"]
    assert (refused.status, refused.refusal) == (MoveStatus.ILLEGAL, Refusal.FAMILY_OFF)
    assert "the family fusion" in refused.reason
    assert doalls(sut) == 2, "the default stage fused nothing"


def test_the_default_moves_fuse_the_pair_when_fusion_is_on() -> None:
    sut = canonical()

    sut.run_defaults("moves")

    assert doalls(sut) == 1


def test_a_trial_copy_keeps_the_families_switched_off() -> None:
    sut = canonical(("fusion", "tiling"))

    assert sut.trial().disabled_families == frozenset({"fusion", "tiling"})


def test_with_every_family_off_only_the_family_free_kinds_stay_listed() -> None:
    sut = canonical((ALL,))

    assert {move.kind for move in sut.list_moves()} <= FAMILY_FREE
    assert {kind.kind for kind in cast(list[MoveKind], CALLS["list_move_kinds"](sut))} == FAMILY_FREE
    assert {kind.kind for kind in move_kinds() if family_of(kind.kind) is None} == FAMILY_FREE
