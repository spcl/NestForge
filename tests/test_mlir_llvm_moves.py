# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The MLIR and LLVM agent (docs/design/mlir-llvm-agent.md): compiler passes applied one at a time as named moves,
with the loops after each move; a move that changes nothing or fails is refused, and rollback takes a move back."""

import re
from pathlib import Path

import dace
import numpy as np
import pytest

from tactics.build.pass_moves import PassMove, made_move, mlir_outline
from tactics.records import MoveStatus, Refusal
from tactics.session import Session
from tactics.stages.performance import Reference

N = dace.symbol("N", dtype=dace.int64)
SIZE = 32

#: The affine MLIR cgeist writes for a 2D loop nest after a 2D tiling, cut to its loops.
TILED_MLIR = """func.func @kernel(%arg0: memref<?xf64>) {
  affine.for %arg1 = 0 to 32 step 8 {
    affine.for %arg2 = 0 to 32 step 8 {
      affine.for %arg3 = #map(%arg1) to #map1(%arg1) {
        affine.for %arg4 = #map(%arg2) to #map1(%arg2) {
          %0 = affine.load %arg0[%arg3 * 32 + %arg4] : memref<?xf64>
        }
      }
    }
  }
  scf.parallel (%i) = (%c0) to (%c4) step (%c1) {
    scf.yield
  }
  return
}
"""


@dace.program
def smooth(a: dace.float64[N, N], b: dace.float64[N, N]) -> None:
    for i, j in dace.map[1 : N - 1, 1 : N - 1]:
        b[i, j] = 0.25 * (a[i - 1, j] + a[i + 1, j] + a[i, j - 1] + a[i, j + 1])


def session(tmp_path: Path) -> Session:
    sut = Session(smooth.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": SIZE})
    sut.canonicalize()
    return sut


def data() -> tuple[dict[str, np.ndarray], Reference]:
    a = np.arange(SIZE * SIZE, dtype=np.float64).reshape(SIZE, SIZE) % 7.0
    b = np.zeros((SIZE, SIZE))
    expected = b.copy()
    expected[1:-1, 1:-1] = 0.25 * (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:])
    return {"a": a, "b": b}, Reference({"b": expected}, 0.0, 0.0)


def rows(outline: str) -> list[str]:
    """The loop labels of an outline view_ir printed."""
    return re.findall(r"^\s*((?:for|sfor|par|loop)\d+):", outline, re.MULTILINE)


def test_the_mlir_outline_lists_every_loop_in_preorder_with_its_depth_and_bounds() -> None:
    outline = mlir_outline(TILED_MLIR)

    assert [(label, depth) for label, depth, _ in outline] == [
        ("for0", 0),
        ("for1", 1),
        ("for2", 2),
        ("for3", 3),
        ("par0", 0),
    ]
    assert outline[0][2] == "%arg1 = 0 to 32 step 8"


def test_an_argument_the_move_does_not_take_is_refused_with_the_ones_it_takes() -> None:
    refused = made_move("mlir", "tile", {"factor": 4})

    assert refused == "tile takes ['tile_sizes']; not ['factor']."


def test_a_list_argument_fills_its_pass_option_and_a_knob_its_opt_flag() -> None:
    tile = made_move("mlir", "tile", {"tile_sizes": [8, 16]})
    vectorize = made_move("llvm", "vectorize", {"width": 4, "interleave": 2})
    distribute = made_move("llvm", "distribute", {})

    assert isinstance(tile, PassMove) and tile.text() == "affine-loop-tile{tile-sizes=8,16}"
    assert isinstance(vectorize, PassMove) and vectorize.text() == "loop-vectorize"
    assert vectorize.knobs() == ["-force-vector-width=4", "-force-vector-interleave=2"]
    assert isinstance(distribute, PassMove) and distribute.knobs() == ["-enable-loop-distribute"]


@pytest.mark.vendor  # Polygeist (cgeist, polygeist-opt) is in the AMD judge image, not on CI
def test_a_tiling_gives_four_loops_and_the_program_stays_correct(tmp_path: Path) -> None:
    sut = session(tmp_path)
    assert rows(sut.view_ir()) == ["for0", "for1"]

    done = sut.apply_pass_move("tile", {"tile_sizes": [8, 8]})

    assert done.status is MoveStatus.APPLIED
    assert rows(sut.view_ir()) == ["for0", "for1", "for2", "for3"]
    assert sut.performance(*data()).structured.correct is True


@pytest.mark.vendor  # Polygeist (cgeist, polygeist-opt) is in the AMD judge image, not on CI
def test_a_move_that_changes_no_kernel_is_refused_and_records_nothing(tmp_path: Path) -> None:
    sut = session(tmp_path)
    sut.apply_pass_move("tile", {"tile_sizes": [8, 8]})
    before = sut.pipeline

    refused = sut.apply_pass_move("peel")  # peels scf.for loops; cgeist raised every loop to affine.for

    assert (refused.status, refused.refusal) == (MoveStatus.ILLEGAL, Refusal.ARGUMENT)
    assert refused.reason.startswith("no change")
    assert sut.pipeline == before


@pytest.mark.vendor  # Polygeist (cgeist, polygeist-opt) is in the AMD judge image, not on CI
def test_rollback_after_two_moves_gives_back_the_loops_after_the_first(tmp_path: Path) -> None:
    sut = session(tmp_path)
    sut.apply_pass_move("tile", {"tile_sizes": [8, 8]})
    after_first = sut.view_ir()
    second = sut.apply_pass_move("parallelize")  # the stencil's loops are parallel, so this always changes them

    sut.apply_pass_move("rollback")

    assert second.status is MoveStatus.APPLIED, second.reason
    assert sut.view_ir() == after_first


@pytest.mark.vendor  # Polygeist (cgeist, polygeist-opt) is in the AMD judge image, not on CI
def test_an_llvm_vectorization_of_width_4_shows_in_the_outline_and_the_program_stays_correct(tmp_path: Path) -> None:
    sut = session(tmp_path)
    assert sut.advance_pass_phase().status is MoveStatus.APPLIED
    sut.apply_pass_move("cleanup")

    done = sut.apply_pass_move("vectorize", {"width": 4, "interleave": 1})

    assert done.status is MoveStatus.APPLIED
    assert "vec=4" in sut.view_ir()
    assert sut.performance(*data()).structured.correct is True


@pytest.mark.vendor  # Polygeist (cgeist, polygeist-opt) is in the AMD judge image, not on CI
def test_a_pass_its_tool_rejects_is_refused_with_the_tools_message_and_changes_nothing(tmp_path: Path) -> None:
    sut = session(tmp_path)

    failed = sut.apply_pass_move("tile", {"tile_sizes": "x"})

    assert failed.status is MoveStatus.ILLEGAL and failed.reason.startswith("failed:")
    assert "tile-sizes" in failed.reason
    assert sut.pipeline is None


@pytest.mark.vendor  # Polygeist (cgeist, polygeist-opt) is in the AMD judge image, not on CI
def test_the_mlir_moves_close_once_the_agent_advances_to_llvm(tmp_path: Path) -> None:
    sut = session(tmp_path)
    sut.advance_pass_phase()

    refused = sut.apply_pass_move("tile", {"tile_sizes": [8, 8]})

    assert refused.status is MoveStatus.ILLEGAL
    assert "tile is no llvm move" in refused.reason
    assert "fuse" in sut.list_pass_moves()
