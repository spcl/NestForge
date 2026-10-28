# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Doall tiling moves: tile a doall at the agent's sizes and untile it back, collapse nested doalls into one
parallel space and expand it again; each leaves a program that computes the same."""

import re

import dace
import numpy as np
from dace.sdfg import nodes

from nestforge.records import MoveStatus
from nestforge.session import Session

N = dace.symbol("N", dtype=dace.int64)
SIZE = 40  # not a multiple of the tile, so the last tile is partial


@dace.program
def scale_add(a: dace.float64[N, N], b: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        b[i, j] = 2.0 * a[i, j] + 1.0


def session(tmp_path) -> Session:
    return Session(scale_add.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": SIZE})


def doall_labels(sut: Session) -> list[str]:
    states = re.findall(r"\b(state\d+_\d+)\b", sut.describe())
    sut.expand(states, sut.epoch)
    return re.findall(r"\b(doall\d+_\d+)\b", sut.describe())


def maps(sut: Session) -> list[list[str]]:
    return [n.map.params for n, _ in sut.sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]


def computes_scale_add(sut: Session) -> None:
    rng = np.random.default_rng(5)
    a, b = rng.random((SIZE, SIZE)), np.zeros((SIZE, SIZE))
    sut.sdfg(a=a, b=b, N=SIZE)
    np.testing.assert_allclose(b, 2.0 * a + 1.0, rtol=0, atol=0)


def test_a_doall_offers_tiling_and_expansion(tmp_path):
    sut = session(tmp_path)
    (doall,) = doall_labels(sut)

    kinds = {move.kind for move in sut.list_moves() if move.labels[0] == doall}

    assert {"tile-doall", "expand-doall"} <= kinds


def test_a_tiled_doall_steps_over_tiles_and_untiling_restores_one_doall(tmp_path):
    sut = session(tmp_path)
    (doall,) = doall_labels(sut)

    tiled = sut.apply_move("tile-doall", [doall], sut.epoch, tile_sizes=[8, 16])

    assert tiled.status is MoveStatus.APPLIED, tiled
    assert len(maps(sut)) == 2, maps(sut)
    computes_scale_add(sut)
    (pair,) = sut.list_moves("untile-doall")
    untiled = sut.apply_move("untile-doall", list(pair.labels), sut.epoch)
    assert untiled.status is MoveStatus.APPLIED, untiled
    assert len(maps(sut)) == 1, maps(sut)
    computes_scale_add(sut)


def test_tile_sizes_must_name_every_dimension(tmp_path):
    sut = session(tmp_path)
    (doall,) = doall_labels(sut)

    result = sut.apply_move("tile-doall", [doall], sut.epoch, tile_sizes=[8])

    assert result.status is MoveStatus.ILLEGAL
    assert "one positive size each" in result.reason


def test_tile_sizes_of_one_everywhere_tile_nothing_and_are_refused(tmp_path):
    sut = session(tmp_path)
    (doall,) = doall_labels(sut)

    result = sut.apply_move("tile-doall", [doall], sut.epoch, tile_sizes=[1, 1])

    assert result.status is MoveStatus.ILLEGAL
    assert len(maps(sut)) == 1


def test_a_listed_tiling_applies_as_listed(tmp_path):
    sut = session(tmp_path)
    (doall,) = doall_labels(sut)
    (move,) = [m for m in sut.list_moves("tile-doall") if m.labels == (doall,)]

    result = sut.apply_move(move.kind, move.labels, move.epoch, move.tile_sizes)

    assert move.tile_sizes == (32, 32)
    assert result.status is MoveStatus.APPLIED, result
    assert len(maps(sut)) == 2
    computes_scale_add(sut)


def test_an_expanded_doall_collapses_back_into_one_parallel_space(tmp_path):
    sut = session(tmp_path)
    (doall,) = doall_labels(sut)

    expanded = sut.apply_move("expand-doall", [doall], sut.epoch)

    assert expanded.status is MoveStatus.APPLIED, expanded
    assert sorted(len(params) for params in maps(sut)) == [1, 1]
    computes_scale_add(sut)
    (pair,) = sut.list_moves("collapse-doalls")
    collapsed = sut.apply_move("collapse-doalls", list(pair.labels), sut.epoch)
    assert collapsed.status is MoveStatus.APPLIED, collapsed
    assert [len(params) for params in maps(sut)] == [2]
    computes_scale_add(sut)


if __name__ == "__main__":
    import pathlib
    import tempfile

    for test in (
        test_a_doall_offers_tiling_and_expansion,
        test_a_tiled_doall_steps_over_tiles_and_untiling_restores_one_doall,
        test_tile_sizes_must_name_every_dimension,
        test_tile_sizes_of_one_everywhere_tile_nothing_and_are_refused,
        test_a_listed_tiling_applies_as_listed,
        test_an_expanded_doall_collapses_back_into_one_parallel_space,
    ):
        test(pathlib.Path(tempfile.mkdtemp()))
