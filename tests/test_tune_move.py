# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``tune_move``: a tiling move tried at several sizes in one call, each measured and rolled back, so the agent picks
a size without tiling, untiling and tiling again."""

import dace
import numpy as np

from nestforge.session import MAX_TUNED_VALUES, Session
from nestforge.stages.performance import Reference

N = dace.symbol("N", dtype=dace.int64)
SIZE = 64


@dace.program
def doubled(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = 2.0 * a[i] + 1.0


def session(tmp_path) -> Session:
    sut = Session(doubled.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": SIZE})
    sut.canonicalize()
    return sut


def data() -> tuple[dict[str, np.ndarray], Reference]:
    a = np.arange(1.0, SIZE + 1)
    return {"a": a, "b": np.zeros(SIZE)}, Reference({"b": 2.0 * a + 1.0}, 0.0, 0.0)


def test_each_size_is_measured_against_the_program_now_and_the_program_stays(tmp_path):
    sut = session(tmp_path)
    (move,) = sut.list_moves("tile-doall")
    tree, applied = sut.describe(), len(sut.applied)

    result = sut.tune_move("tile-doall", move.labels, [[4], [16], [64]], *data())

    assert [(v.tile_sizes, v.status, v.correct) for v in result.values] == [
        ((), "now", True),
        ((4,), "measured", True),
        ((16,), "measured", True),
        ((64,), "measured", True),
    ]
    assert all(v.time_us is not None and v.time_us > 0 for v in result.values)
    assert sut.describe() == tree and len(sut.applied) == applied, "every try was rolled back"


def test_a_refused_size_is_reported_and_the_others_still_run(tmp_path):
    sut = session(tmp_path)
    (move,) = sut.list_moves("tile-doall")

    result = sut.tune_move("tile-doall", move.labels, [[0], [8]], *data())

    assert [v.status for v in result.values] == ["now", "illegal", "measured"]
    assert result.values[1].reason


def test_another_kind_or_too_many_sizes_is_refused_without_measuring(tmp_path):
    sut = session(tmp_path)
    (move,) = sut.list_moves("tile-doall")
    many = [[2**k] for k in range(MAX_TUNED_VALUES + 1)]

    other = sut.tune_move("doall-to-for", move.labels, [[8]], *data())
    too_many = sut.tune_move("tile-doall", move.labels, many, *data())

    assert [v.status for v in other.values] == ["illegal"] and [v.status for v in too_many.values] == ["illegal"]
