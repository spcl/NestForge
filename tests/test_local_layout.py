# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``relayout`` on arrays the program makes itself: laid out in place, at any point, without a bench request; ``zip``
and ``unzip`` (array of structs, struct of arrays) too, since no argument list changes."""

from pathlib import Path

import dace
import numpy as np
from dace.frontend.python.parser import DaceProgram

from nestforge.records import MoveStatus
from nestforge.session import Session

N = dace.symbol("N", dtype=dace.int64)
SIZE = 16


@dace.program
def transposed(a: dace.float64[N, N], b: dace.float64[N, N]) -> None:
    t = np.empty_like(a)
    for i, j in dace.map[0:N, 0:N]:
        t[i, j] = a[i, j] * 2.0
    for i, j in dace.map[0:N, 0:N]:
        b[i, j] = t[j, i] + 1.0


@dace.program
def paired(a: dace.float64[N], b: dace.float64[N]) -> None:
    t = np.empty_like(a)
    u = np.empty_like(a)
    for i in dace.map[0:N]:
        t[i] = a[i] * 2.0
    for i in dace.map[0:N]:
        u[i] = a[i] + 1.0
    for i in dace.map[0:N]:
        b[i] = t[i] * u[i]


def session(program: DaceProgram, tmp_path: Path) -> Session:
    sut = Session(program.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": SIZE})
    sut.canonicalize()
    return sut


def locals_of(sut: Session) -> list[str]:
    return sorted(n for n, d in sut.sdfg.arrays.items() if d.transient and isinstance(d, dace.data.Array))


def test_an_internal_array_is_permuted_in_place_and_the_program_computes_the_same(tmp_path: Path) -> None:
    sut = session(transposed, tmp_path)
    (t,) = locals_of(sut)
    a = np.arange(SIZE * SIZE, dtype=np.float64).reshape(SIZE, SIZE).copy()

    done = sut.apply_move("relayout", [], operation={"action": "permute", "array": t, "order": ["i1", "i0"]})

    assert done.status is MoveStatus.APPLIED and done.reason.endswith("(internal)")
    assert sut.layout == (), "an internal array sends the bench no request"
    b = np.zeros((SIZE, SIZE))
    sut.sdfg.compile()(a=a, b=b, N=SIZE)
    assert np.array_equal(b, a.T * 2.0 + 1.0)


def test_two_internal_arrays_zip_into_an_array_of_structs_and_rollback_takes_it_back(tmp_path: Path) -> None:
    sut = session(paired, tmp_path)
    t, u = locals_of(sut)
    before = sut.describe()
    a = np.linspace(0.5, 2.0, SIZE)

    done = sut.apply_move("relayout", [], operation={"action": "zip", "arrays": [t, u], "into": "tu"})

    assert done.status is MoveStatus.APPLIED
    assert "tu" in sut.sdfg.arrays and t not in sut.sdfg.arrays
    b = np.zeros(SIZE)
    sut.sdfg.compile()(a=a, b=b, N=SIZE)
    assert np.allclose(b, (a * 2.0) * (a + 1.0))
    sut.apply_move("rollback", [])
    assert sut.describe() == before


def test_an_argument_layout_after_an_internal_one_is_refused(tmp_path: Path) -> None:
    sut = session(transposed, tmp_path)
    (t,) = locals_of(sut)
    sut.apply_move("relayout", [], operation={"action": "permute", "array": t, "order": ["i1", "i0"]})

    refused = sut.apply_move("relayout", [], operation={"action": "permute", "array": "a", "order": ["i1", "i0"]})

    assert refused.status is MoveStatus.ILLEGAL and "before any other change" in refused.reason
