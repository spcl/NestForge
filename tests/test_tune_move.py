# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``tune_move``: a tiling move tried at several sizes in one call, each measured; the fastest size that beats the
program is kept, and with none the program stays, so the agent never tiles, untiles and tiles again."""

import dace
import numpy as np
import pytest
from dace.sdfg import nodes

from nestforge.session import MAX_TUNED_VALUES, Session, TunedValue, fastest_beating
from nestforge.stages.performance import Performance, Reference

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


def tile_size(sdfg: dace.SDFG) -> int:
    """The step of the program's tiled map, 0 when no map is tiled."""
    steps = [
        int(step)
        for node, _ in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.MapEntry)
        for _, _, step in node.map.range
        if step != 1
    ]
    return max(steps, default=0)


def timed_by_tile_size(monkeypatch: pytest.MonkeyPatch, times_us: dict[int, float]) -> None:
    """Every measurement runs for real, and its time is replaced by ``times_us`` of the tile size the tree shows
    (0: untiled), read from the program's map steps, so which size wins does not depend on this machine."""
    measured = Session.performance

    def performance(self: Session, *args, **kwargs) -> Performance:
        report = measured(self, *args, **kwargs)
        time_us = times_us[tile_size(self.sdfg)]
        return report.model_copy(update={"structured": report.structured.model_copy(update={"time_us": time_us})})

    monkeypatch.setattr(Session, "performance", performance)


def test_without_a_faster_size_every_try_is_measured_and_the_program_stays(tmp_path, monkeypatch):
    sut = session(tmp_path)
    (move,) = sut.list_moves("tile-doall")
    tree, applied = sut.describe(), len(sut.applied)
    timed_by_tile_size(monkeypatch, {0: 10.0, 4: 40.0, 16: 20.0, 64: 10.0})

    result = sut.tune_move("tile-doall", move.labels, [[4], [16], [64]], *data())

    assert [(v.sizes, v.status, v.correct, v.time_us) for v in result.values] == [
        ((), "now", True, 10.0),
        ((4,), "measured", True, 40.0),
        ((16,), "measured", True, 20.0),
        ((64,), "measured", True, 10.0),
    ]
    assert sut.describe() == tree and len(sut.applied) == applied, "a tie keeps the program"


def test_the_fastest_size_that_beats_the_program_is_kept_and_rollback_takes_it_back(tmp_path, monkeypatch):
    sut = session(tmp_path)
    (move,) = sut.list_moves("tile-doall")
    tree = sut.describe()
    timed_by_tile_size(monkeypatch, {0: 30.0, 4: 40.0, 16: 10.0, 64: 20.0})

    result = sut.tune_move("tile-doall", move.labels, [[4], [16], [64]], *data())

    assert [v.status for v in result.values] == ["now", "measured", "kept", "measured"]
    assert tile_size(sut.sdfg) == 16, "the program holds the kept size"
    sut.apply_move("rollback", [])
    assert sut.describe() == tree


def test_only_a_correct_measured_size_faster_than_a_correct_program_wins():
    now = TunedValue(status="now", time_us=10.0, correct=True)

    assert fastest_beating([now, TunedValue(sizes=(8,), status="measured", time_us=5.0, correct=True)]) == 1
    assert fastest_beating([now, TunedValue(sizes=(8,), status="measured", time_us=5.0, correct=False)]) is None
    assert fastest_beating([now, TunedValue(sizes=(8,), status="illegal")]) is None
    assert (
        fastest_beating(
            [
                TunedValue(status="now", correct=False, time_us=10.0),
                TunedValue(sizes=(8,), status="measured", time_us=5.0, correct=True),
            ]
        )
        is None
    )


def test_a_refused_size_is_reported_and_the_others_still_run(tmp_path, monkeypatch):
    sut = session(tmp_path)
    (move,) = sut.list_moves("tile-doall")
    timed_by_tile_size(monkeypatch, {0: 10.0, 8: 20.0})

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


def test_lanes_of_vectorize_doall_are_tried_in_the_argument_that_move_takes(tmp_path, monkeypatch):
    sut = session(tmp_path)
    (move,) = sut.list_moves("vectorize-doall")
    timed_by_tile_size(monkeypatch, {0: 10.0, 4: 20.0, 8: 20.0})  # a vector body steps by its lanes

    result = sut.tune_move("vectorize-doall", move.labels, [[4], [8]], *data())

    assert [(v.sizes, v.status) for v in result.values] == [((), "now"), ((4,), "measured"), ((8,), "measured")]
    assert all(v.correct for v in result.values)
