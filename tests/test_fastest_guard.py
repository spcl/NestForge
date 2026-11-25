# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``get_performance`` never loses a measured program to a slower one: a correct program more than
:data:`GUARD_MARGIN` slower than the fastest version measured is rolled back to that version."""

from pathlib import Path

import dace

from tactics.session import GUARD_MARGIN, Restored, Session
from tactics.stages.performance import ProgramReport

N = dace.symbol("N", dtype=dace.int64)


@dace.program
def doubled(a: dace.float64[N], b: dace.float64[N]) -> None:
    for i in dace.map[0:N]:
        b[i] = 2.0 * a[i] + 1.0


def session(tmp_path: Path) -> Session:
    sut = Session(doubled.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": 64})
    sut.canonicalize()
    return sut


def measured(time_us: float, correct: bool | None = True) -> ProgramReport:
    return ProgramReport(time_us=time_us, correct=correct)


def tiled(sut: Session) -> None:
    move = sut.list_moves("tile-doall")[0]
    sut.apply_move("tile-doall", move.labels, tile_sizes=[8])


def test_a_slower_program_is_rolled_back_to_the_fastest_version_with_its_labels(tmp_path: Path) -> None:
    sut = session(tmp_path)
    before = sut.describe()
    sut.keep_fastest(measured(10.0))
    tiled(sut)
    tiled(sut)

    restored = sut.keep_fastest(measured(30.0))

    assert restored == Restored(fastest_us=10.0, now_us=30.0, rolled_back=2)
    assert sut.describe() == before


def test_a_faster_program_becomes_the_fastest_version_and_stays(tmp_path: Path) -> None:
    sut = session(tmp_path)
    sut.keep_fastest(measured(30.0))
    tiled(sut)
    after = sut.describe()

    assert sut.keep_fastest(measured(10.0)) is None
    assert sut.describe() == after and sut.fastest is not None and sut.fastest[0] == 10.0


def test_a_tie_within_the_margin_or_a_wrong_result_changes_nothing(tmp_path: Path) -> None:
    sut = session(tmp_path)
    sut.keep_fastest(measured(10.0))
    tiled(sut)
    after = sut.describe()

    assert sut.keep_fastest(measured(10.0 * GUARD_MARGIN)) is None
    assert sut.keep_fastest(measured(99.0, correct=False)) is None
    assert sut.describe() == after
