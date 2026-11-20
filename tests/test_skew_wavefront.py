# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The ``skew-wavefront`` move. A session starts from the canonical form without wavefronts. The move skews a nest
whose diagonals DaCe proves independent, refuses one that is no wavefront with DaCe's sentence, and is undone by
``rollback``. The default moves skew every wavefront first, so ``nestforge_default`` keeps today's program."""

import ctypes

import dace
import numpy as np
import pytest
from dace.frontend.python.parser import DaceProgram
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from test_failed_calls import program

from nestforge.records import MoveStatus, Refusal
from nestforge.session import Session
from nestforge.stages.canonicalize import Targets
from nestforge.stages.moves import SKEW_WAVEFRONT

N = dace.symbol("N")
T = dace.symbol("T")
#: The extent the numeric checks run at.
SIZE = 40
#: A tile extent that puts several tiles on a diagonal at :data:`SIZE`.
TILE = 8
#: The threads the tiled numeric check runs on.
THREADS = 4
#: A GPU tile extent that differs from DaCe's default and from the CPU's.
GPU_TILE = 48


@dace.program
def wavefront_2d(aa: dace.float64[N, N]) -> None:
    """TSVC s2111: each element reads its left and its upper neighbor."""
    for i in range(1, N):
        for j in range(1, N):
            aa[i, j] = (aa[i, j - 1] + aa[i - 1, j]) / 1.9


@dace.program
def outer_carried(aa: dace.float64[N, N], bb: dace.float64[N, N]) -> None:
    """TSVC s1119: only the outer axis carries, so the inner one is parallel already."""
    for i in range(1, N):
        for j in range(N):
            aa[i, j] = aa[i - 1, j] + bb[i, j]


@dace.program
def time_steps(aa: dace.float64[N]) -> None:
    """A time loop whose iterator indexes nothing: every step re-applies the same update."""
    for t in range(T):
        for i in range(1, N):
            aa[i] = aa[i] + aa[i - 1]


def canonical(prog: dace.frontend.python.parser.DaceProgram, wavefront: bool = False, gpu: bool = False) -> Session:
    sut = Session(prog.to_sdfg(simplify=True), Targets(gpu=gpu, wavefront=wavefront))
    sut.canonicalize()
    return sut


def loops_and_maps(sut: Session) -> tuple[int, int]:
    assert sut.sdfg is not None, "a session holds its program"
    loops = [r for r in sut.sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]
    maps = [node for node, parent in sut.sdfg.all_nodes_recursive() if isinstance(node, nodes.MapEntry)]
    return len(loops), len(maps)


def skew_labels(sut: Session) -> list[tuple[str, ...]]:
    return [move.labels for move in sut.list_moves(SKEW_WAVEFRONT)]


def reference(aa: np.ndarray) -> np.ndarray:
    out = aa.copy()
    for i in range(1, out.shape[0]):
        for j in range(1, out.shape[1]):
            out[i, j] = (out[i, j - 1] + out[i - 1, j]) / 1.9
    return out


def test_a_session_starts_without_wavefronts_and_the_move_skews_the_nest_it_lists() -> None:
    sut = canonical(wavefront_2d)
    assert loops_and_maps(sut) == (2, 0), "the canonical form of a session leaves the wavefront to the moves"
    (labels,) = skew_labels(sut)

    done = sut.apply_move(SKEW_WAVEFRONT, list(labels))

    assert (done.status, done.reason) == (MoveStatus.APPLIED, "WavefrontSkew")
    assert loops_and_maps(sut)[1] == 1, "one doall runs each diagonal"
    assert labels[0] not in sut.row_index(), "the skewed loop's label retires"
    diagonal = [row for row, (obj, _) in sut.row_index().items() if isinstance(obj, LoopRegion)]
    assert diagonal and all(row.startswith("for") for row in diagonal)


@pytest.mark.e2e
def test_the_skewed_program_computes_the_reference() -> None:
    sut = canonical(wavefront_2d)
    (labels,) = skew_labels(sut)
    sut.apply_move(SKEW_WAVEFRONT, list(labels))
    aa = np.random.default_rng(2111).standard_normal((SIZE, SIZE))
    got = aa.copy()

    sut.sdfg(aa=got, N=SIZE)

    assert np.allclose(got, reference(aa))


@pytest.mark.e2e
def test_a_diagonal_of_many_tiles_run_by_several_threads_computes_the_reference() -> None:
    sut = canonical(wavefront_2d)
    (labels,) = skew_labels(sut)
    # tiles of 8 at an extent of 40 put up to five tiles on a diagonal, where the default tile holds the whole array
    sut.apply_move(SKEW_WAVEFRONT, list(labels), tile_sizes=[TILE, TILE])
    aa = np.random.default_rng(2111).standard_normal((SIZE, SIZE))
    got = aa.copy()
    compiled = sut.sdfg.compile()
    openmp = ctypes.CDLL("libgomp.so.1")
    openmp.omp_set_num_threads(THREADS)
    assert openmp.omp_get_max_threads() == THREADS, "the tiles of a diagonal run on several threads"

    compiled(aa=got, N=SIZE)

    assert np.allclose(got, reference(aa))


def test_on_the_gpu_target_tile_sizes_set_the_extents_of_the_gpu_tiles() -> None:
    sut = canonical(wavefront_2d, gpu=True)
    (labels,) = skew_labels(sut)

    done = sut.apply_move(SKEW_WAVEFRONT, list(labels), tile_sizes=[GPU_TILE, GPU_TILE])

    assert (done.status, done.reason) == (MoveStatus.APPLIED, "WavefrontSkew")
    assert sut.sdfg is not None
    steps = {str(loop.loop_condition.as_string) for loop in sut.sdfg.all_control_flow_regions(recursive=True)
             if isinstance(loop, LoopRegion)}  # fmt: skip
    assert any(str(GPU_TILE) in step for step in steps), steps


def test_tile_sizes_set_the_extents_of_the_tiles() -> None:
    sut = canonical(wavefront_2d)
    (labels,) = skew_labels(sut)

    sut.apply_move(SKEW_WAVEFRONT, list(labels), tile_sizes=[32, 32])

    assert sut.sdfg is not None
    steps = {str(loop.loop_condition.as_string) for loop in sut.sdfg.all_control_flow_regions(recursive=True)
             if isinstance(loop, LoopRegion)}  # fmt: skip
    assert any("32" in step for step in steps), steps


def test_tile_sizes_other_than_two_positive_extents_are_refused() -> None:
    sut = canonical(wavefront_2d)
    (labels,) = skew_labels(sut)

    refused = sut.apply_move(SKEW_WAVEFRONT, list(labels), tile_sizes=[32])

    assert (refused.status, refused.refusal) == (MoveStatus.ILLEGAL, Refusal.ARGUMENT)


@pytest.mark.parametrize(
    "prog, sentence",
    [
        # canonical form made the parallel inner axis a doall already, so the loop holds no nest to skew
        (outer_carried, "its body is not a nest of loops and maps"),
        (time_steps, "a time-step loop"),
    ],
)
def test_a_loop_that_is_no_wavefront_is_refused_with_daces_sentence_and_the_program_stays(
    prog: DaceProgram, sentence: str
) -> None:
    sut = canonical(prog)
    loops = [row for row, (obj, _) in sut.row_index().items() if isinstance(obj, LoopRegion)]
    assert loops, "the fixture keeps a loop"
    before = program(sut)

    refused = [sut.apply_move(SKEW_WAVEFRONT, [row]) for row in loops]

    assert any(sentence in result.reason for result in refused), [result.reason for result in refused]
    assert all(result.refusal is Refusal.DEPENDENCE for result in refused)
    assert program(sut) == before


def test_rollback_after_the_skew_restores_the_program_and_its_labels() -> None:
    sut = canonical(wavefront_2d)
    (labels,) = skew_labels(sut)
    before, rows = program(sut), set(sut.row_index())
    sut.apply_move(SKEW_WAVEFRONT, list(labels))

    sut.undo()

    assert program(sut) == before and set(sut.row_index()) == rows


def test_the_default_moves_skew_as_canonicalization_with_wavefronts_does() -> None:
    today = canonical(wavefront_2d, wavefront=True)
    sut = canonical(wavefront_2d)

    sut.run_defaults("moves")
    today.run_defaults("moves")

    assert loops_and_maps(sut) == loops_and_maps(today) == (3, 1)
    assert program(sut) == program(today), "the defaults reach the program canonicalization with wavefronts reaches"


def test_with_the_wavefront_family_off_the_defaults_leave_the_nest_sequential() -> None:
    sut = Session(wavefront_2d.to_sdfg(simplify=True), disabled_families=["wavefront"])
    sut.canonicalize()

    sut.run_defaults("moves")

    assert loops_and_maps(sut) == (2, 0)
    assert skew_labels(sut) == []
