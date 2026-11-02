# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""For-loop tiling moves: tile a sequential loop at the agent's size and untile it back; each leaves a program that
computes the same."""

import re

import dace
import numpy as np
from dace.libraries.standard.nodes.external_call import external_calls
from dace.sdfg.state import LoopRegion

from nestforge.records import MoveStatus
from nestforge.session import Session

N = dace.symbol("N", dtype=dace.int64)
SIZE = 21  # not a multiple of the tile, so the last tile is partial


@dace.program
def running_sum(a: dace.float64[N], b: dace.float64[N]):
    for i in range(1, N):
        a[i] = a[i - 1] + b[i]


def session(tmp_path) -> Session:
    return Session(running_sum.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": SIZE})


def loop_labels(sut: Session) -> list[str]:
    return re.findall(r"\b(for\d+)\b", sut.describe())


def loop_count(sut: Session) -> int:
    return sum(isinstance(r, LoopRegion) for r in sut.sdfg.all_control_flow_regions(recursive=True))


def computes_running_sum(sut: Session) -> None:
    a, b = np.zeros(SIZE), np.arange(SIZE, dtype=np.float64)
    sut.sdfg(a=a, b=b, N=SIZE)
    expected = np.concatenate([[0.0], np.cumsum(b[1:])])
    np.testing.assert_allclose(a, expected, rtol=0, atol=0)


def test_a_tiled_for_loop_holds_a_loop_over_one_tile_and_untiling_gives_one_loop_back(tmp_path):
    sut = session(tmp_path)
    (loop,) = loop_labels(sut)

    tiled = sut.apply_move("tile-for", [loop], tile_sizes=[8])

    assert tiled.status is MoveStatus.APPLIED, tiled
    assert loop_count(sut) == 2
    computes_running_sum(sut)
    (move,) = sut.list_moves("untile-for")
    untiled = sut.apply_move("untile-for", list(move.labels))
    assert untiled.status is MoveStatus.APPLIED, untiled
    assert loop_count(sut) == 1
    computes_running_sum(sut)


@dace.program
def column_steps(a: dace.float64[N, N], b: dace.float64[N, N]):
    for j in range(1, N):
        for i in dace.map[0:N]:
            a[j, i] = a[j - 1, i] + b[j, i]


def test_untiling_around_a_kernel_renames_the_iterator_inside_it_and_the_kernel_still_builds(tmp_path):
    sut = Session(column_steps.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": SIZE})
    sut.canonicalize()
    sut.default_scopes()
    (loop,) = loop_labels(sut)
    (old,) = [r.loop_variable for r in sut.sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]

    assert sut.apply_move("tile-for", [loop], tile_sizes=[8]).status is MoveStatus.APPLIED
    (move,) = sut.list_moves("untile-for")
    assert sut.apply_move("untile-for", list(move.labels)).status is MoveStatus.APPLIED

    (ext,) = external_calls(sut.sdfg)
    (new,) = [r.loop_variable for r in sut.sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]
    assert new != old and new in ext.abi_order and old not in ext.abi_order
    assert new in ext.standalone_sdfg.symbols and old not in ext.standalone_sdfg.symbols
    sut.prepare_kernel(ext.name)
    sut.default_build(ext.name)
    assert ext.name in sut.builds


def test_a_for_tile_is_one_size_of_at_least_two(tmp_path):
    sut = session(tmp_path)
    (loop,) = loop_labels(sut)

    result = sut.apply_move("tile-for", [loop], tile_sizes=[8, 8])

    assert result.status is MoveStatus.ILLEGAL
    assert "one size of at least 2" in result.reason


def test_an_untiled_for_loop_offers_no_untiling(tmp_path):
    sut = session(tmp_path)

    assert sut.list_moves("untile-for") == []


if __name__ == "__main__":
    import pathlib
    import tempfile

    for test in (
        test_a_tiled_for_loop_holds_a_loop_over_one_tile_and_untiling_gives_one_loop_back,
        test_untiling_around_a_kernel_renames_the_iterator_inside_it_and_the_kernel_still_builds,
        test_a_for_tile_is_one_size_of_at_least_two,
        test_an_untiled_for_loop_offers_no_untiling,
    ):
        test(pathlib.Path(tempfile.mkdtemp()))
