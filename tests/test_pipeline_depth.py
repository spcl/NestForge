# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``set-pipeline-depth`` pipelines a loop whose iterations first copy a tile into a buffer: the move sets the loop's
depth, and code generation lowers it as DaCe's ``LowerMultipleBuffering``."""

import dace
from dace.sdfg.state import LoopRegion

from nestforge.records import Refused
from nestforge.stages.moves import SET_PIPELINE_DEPTH, Rewrite, legal_moves, plan_move

N = dace.symbol("N")
TILE = 4


@dace.program
def tiled(A: dace.float64[N], S: dace.float64[TILE]):
    """A recurrence over tiles: each iteration copies its tile into ``buf`` and folds it into ``S``, so the loop stays
    a loop."""
    buf = dace.define_local([TILE], dace.float64, storage=dace.StorageType.Register)
    for t in range(0, N, TILE):
        buf[:] = A[t : t + TILE]
        S[:] = 0.5 * S + buf


@dace.program
def unstaged(A: dace.float64[N], S: dace.float64[TILE]):
    for t in range(0, N, TILE):
        S[:] = S + A[t : t + TILE]


def only_loop(sdfg: dace.SDFG) -> LoopRegion:
    (loop,) = [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]
    return loop


def test_a_loop_staging_a_tile_is_offered_and_pipelined_at_the_depth_passed():
    sdfg = tiled.to_sdfg(simplify=True)
    loop = only_loop(sdfg)

    plan = plan_move(SET_PIPELINE_DEPTH, [(loop, None)], depth=3)
    assert isinstance(plan, Rewrite)
    plan.commit()

    assert [kind for kind, _ in legal_moves(sdfg, SET_PIPELINE_DEPTH)] == [SET_PIPELINE_DEPTH]
    assert loop.pipeline_depth == 3
    assert ", 3)" in sdfg.generate_code()[0].clean_code  # the slot the lowered loop loads into, py_mod(..., 3)


def test_a_loop_without_a_staged_tile_or_a_depth_below_two_is_refused():
    sdfg = unstaged.to_sdfg(simplify=True)
    staged = only_loop(tiled.to_sdfg(simplify=True))

    no_tile = plan_move(SET_PIPELINE_DEPTH, [(only_loop(sdfg), None)], depth=2)
    too_shallow = plan_move(SET_PIPELINE_DEPTH, [(staged, None)], depth=1)

    assert isinstance(no_tile, Refused) and "stages no buffer to pipeline" in no_tile
    assert isinstance(too_shallow, Refused) and "2 or more" in too_shallow
    assert legal_moves(sdfg, SET_PIPELINE_DEPTH) == []
