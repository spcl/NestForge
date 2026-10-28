# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Kernel tiling moves: tile a kernel at the agent's sizes and untile it back, collapse nested kernels into one
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


def kernel_labels(sut: Session) -> list[str]:
    states = re.findall(r"\b(state\d+_\d+)\b", sut.describe())
    sut.expand(states, sut.epoch)
    return re.findall(r"\b(kernel\d+_\d+)\b", sut.describe())


def maps(sut: Session) -> list[list[str]]:
    return [n.map.params for n, _ in sut.sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]


def computes_scale_add(sut: Session) -> None:
    rng = np.random.default_rng(5)
    a, b = rng.random((SIZE, SIZE)), np.zeros((SIZE, SIZE))
    sut.sdfg(a=a, b=b, N=SIZE)
    np.testing.assert_allclose(b, 2.0 * a + 1.0, rtol=0, atol=0)


def test_a_kernel_offers_tiling_and_expansion(tmp_path):
    sut = session(tmp_path)
    (kernel,) = kernel_labels(sut)

    kinds = {move.kind for move in sut.list_moves() if move.labels[0] == kernel}

    assert {"tile-kernel", "expand-kernel"} <= kinds


def test_a_tiled_kernel_steps_over_tiles_and_untiling_restores_one_kernel(tmp_path):
    sut = session(tmp_path)
    (kernel,) = kernel_labels(sut)

    tiled = sut.apply_move("tile-kernel", [kernel], sut.epoch, tile_sizes=[8, 16])

    assert tiled.status is MoveStatus.APPLIED, tiled
    assert len(maps(sut)) == 2, maps(sut)
    computes_scale_add(sut)
    (pair,) = sut.list_moves("untile-kernel")
    untiled = sut.apply_move("untile-kernel", list(pair.labels), sut.epoch)
    assert untiled.status is MoveStatus.APPLIED, untiled
    assert len(maps(sut)) == 1, maps(sut)
    computes_scale_add(sut)


def test_tile_sizes_must_name_every_dimension(tmp_path):
    sut = session(tmp_path)
    (kernel,) = kernel_labels(sut)

    result = sut.apply_move("tile-kernel", [kernel], sut.epoch, tile_sizes=[8])

    assert result.status is MoveStatus.ILLEGAL
    assert "one positive size each" in result.reason


def test_an_expanded_kernel_collapses_back_into_one_parallel_space(tmp_path):
    sut = session(tmp_path)
    (kernel,) = kernel_labels(sut)

    expanded = sut.apply_move("expand-kernel", [kernel], sut.epoch)

    assert expanded.status is MoveStatus.APPLIED, expanded
    assert sorted(len(params) for params in maps(sut)) == [1, 1]
    computes_scale_add(sut)
    (pair,) = sut.list_moves("collapse-kernels")
    collapsed = sut.apply_move("collapse-kernels", list(pair.labels), sut.epoch)
    assert collapsed.status is MoveStatus.APPLIED, collapsed
    assert [len(params) for params in maps(sut)] == [2]
    computes_scale_add(sut)


if __name__ == "__main__":
    import pathlib
    import tempfile

    for test in (
        test_a_kernel_offers_tiling_and_expansion,
        test_a_tiled_kernel_steps_over_tiles_and_untiling_restores_one_kernel,
        test_tile_sizes_must_name_every_dimension,
        test_an_expanded_kernel_collapses_back_into_one_parallel_space,
    ):
        test(pathlib.Path(tempfile.mkdtemp()))
