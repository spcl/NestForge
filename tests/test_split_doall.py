# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``split-doall`` cuts a doall's range in two states: the parts partition the range, the program computes the same,
and a cut outside the range leaves one empty part."""

from pathlib import Path

import dace
import numpy as np
import pytest
from dace.sdfg import nodes

from tactics.records import MoveStatus, Refusal
from tactics.session import Session
from tactics.stages.moves import SPLIT_DOALL

N = dace.symbol("N", dace.int64, positive=True)
M = dace.symbol("M", dace.int64, positive=True)
SIZE = 12


@dace.program
def stencil(a: dace.float64[N], b: dace.float64[N]) -> None:
    for i in dace.map[1 : N - 1]:
        b[i] = a[i - 1] + 2.0 * a[i] + a[i + 1]


@dace.program
def plane(a: dace.float64[N, M], b: dace.float64[N, M]) -> None:
    for i, j in dace.map[0:N, 0:M]:
        b[i, j] = a[i, j] + 1.0


def session(program: dace.frontend.python.parser.DaceProgram, tmp_path: Path) -> Session:
    sut = Session(program.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": SIZE, "M": 5})
    sut.canonicalize()
    return sut


def doalls(sut: Session) -> dict[str, nodes.MapEntry]:
    """The top-level doall rows, by label."""
    return {
        label: row[0]
        for label, row in sut.row_index().items()
        if isinstance(row[0], nodes.MapEntry) and row[1] is not None and row[1].entry_node(row[0]) is None
    }


def ranges(sut: Session) -> list[tuple[tuple[int, int], ...]]:
    """Each top-level doall's ranges at N = 12 and M = 5, as (first, last) per dimension."""
    sizes = {N: SIZE, M: 5}
    return sorted(
        tuple(
            (int(dace.symbolic.evaluate(lo, sizes)), int(dace.symbolic.evaluate(hi, sizes)))
            for lo, hi, _ in entry.map.range
        )
        for entry in doalls(sut).values()
    )


def stencil_result(sut: Session) -> np.ndarray:
    a = np.arange(SIZE, dtype=np.float64) ** 2
    b = np.full(SIZE, -1.0)
    sut.sdfg(a=a, b=b, N=SIZE)
    return b


def only_doall(sut: Session) -> tuple[str, str]:
    """The label of the program's one top-level doall and its first parameter, as canonicalization named it."""
    ((label, entry),) = doalls(sut).items()
    return label, entry.map.params[0]


def test_two_splits_give_the_first_row_the_interior_and_the_last_row(tmp_path: Path) -> None:
    sut = session(stencil, tmp_path)
    expected = stencil_result(session(stencil, tmp_path / "base"))
    label, i = only_doall(sut)

    assert sut.apply_move(SPLIT_DOALL, [label], at={i: "2"}).status is MoveStatus.APPLIED
    second = next(
        lab for lab, entry in doalls(sut).items() if dace.symbolic.evaluate(entry.map.range[0][0], {N: SIZE}) == 2
    )
    i = doalls(sut)[second].map.params[0]
    assert sut.apply_move(SPLIT_DOALL, [second], at={i: "N - 2"}).status is MoveStatus.APPLIED

    assert ranges(sut) == [((1, 1),), ((2, 9),), ((10, 10),)]
    np.testing.assert_array_equal(stencil_result(sut), expected)


def test_a_cut_outside_the_range_leaves_one_empty_part_and_the_same_output(tmp_path: Path) -> None:
    sut = session(stencil, tmp_path)
    expected = stencil_result(session(stencil, tmp_path / "base"))
    label, i = only_doall(sut)

    assert sut.apply_move(SPLIT_DOALL, [label], at={i: "N + 5"}).status is MoveStatus.APPLIED

    assert ranges(sut) == [((1, 10),), ((11, 10),)]
    np.testing.assert_array_equal(stencil_result(sut), expected)


def test_a_two_dimensional_doall_is_cut_along_the_named_parameter_only(tmp_path: Path) -> None:
    sut = session(plane, tmp_path)
    ((label, entry),) = doalls(sut).items()

    assert sut.apply_move(SPLIT_DOALL, [label], at={entry.map.params[1]: "3"}).status is MoveStatus.APPLIED

    assert ranges(sut) == [((0, 11), (0, 2)), ((0, 11), (3, 4))]
    a, b = np.arange(SIZE * 5, dtype=np.float64).reshape(SIZE, 5).copy(), np.zeros((SIZE, 5))
    sut.sdfg(a=a, b=b, N=SIZE, M=5)
    np.testing.assert_array_equal(b, a + 1.0)


@pytest.mark.parametrize(
    "at, reason",
    [({}, "pass at"), ({"k": "2"}, "k is not a parameter"), ({"{i}": "K + 1"}, "not defined")],
)
def test_split_doall_refuses_a_cut_it_cannot_make(at: dict[str, str], reason: str, tmp_path: Path) -> None:
    sut = session(stencil, tmp_path)
    label, i = only_doall(sut)
    before = ranges(sut)

    result = sut.apply_move(SPLIT_DOALL, [label], at={key.format(i=i): value for key, value in at.items()})

    assert result.status is not MoveStatus.APPLIED and reason in str(result.reason), result
    assert result.refusal is Refusal.ARGUMENT
    assert ranges(sut) == before


def test_split_doall_refuses_a_doall_with_a_step(tmp_path: Path) -> None:
    """Canonicalization leaves every doall at step 1; one stepped otherwise (by a later move) is refused."""
    sut = session(stencil, tmp_path)
    label, i = only_doall(sut)
    entry = doalls(sut)[label]
    lo, hi, _ = entry.map.range[0]
    entry.map.range[0] = (lo, hi, 2)

    result = sut.apply_move(SPLIT_DOALL, [label], at={i: "4"})

    assert result.refusal is Refusal.ARGUMENT and "step 1" in str(result.reason), result
