# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``set-pipeline-depth`` pipelines a loop inside a GPU kernel whose iterations first copy a tile into a buffer: the
move sets the loop's depth, and code generation lowers it as DaCe's ``LowerMultipleBuffering``. A loop on the host is
refused."""

from pathlib import Path

import dace
from dace.sdfg.state import LoopRegion

from tactics.records import MoveStatus, Refused
from tactics.session import Session
from tactics.stages.canonicalize import Targets
from tactics.stages.moves import SET_PIPELINE_DEPTH, plan_move

M, N = dace.symbol("M"), dace.symbol("N")
TILE = 4


@dace.program
def rows(A: dace.float64[M, N], S: dace.float64[M, TILE]) -> None:
    """Per row, a recurrence over tiles: each iteration copies its tile into ``buf`` and folds it into ``S``."""
    for i in dace.map[0:M]:
        buf = dace.define_local([TILE], dace.float64, storage=dace.StorageType.Register)
        for t in range(0, N, TILE):
            buf[:] = A[i, t : t + TILE]
            S[i, :] = 0.5 * S[i, :] + buf


def kernel_on(tmp_path: Path, gpu: bool) -> tuple[Session, str]:
    session = Session(rows.to_sdfg(simplify=True), Targets(gpu=gpu), work_dir=str(tmp_path), sizes={"M": 8, "N": 16})
    session.canonicalize()
    (kernel,) = (k.name for k in session.default_scopes())
    if gpu:
        session.place({kernel: "gpu"})
        session.insert_copies()
    return session, kernel


def staged_loop(session: Session, kernel: str) -> LoopRegion:
    body = session.kernel(kernel).standalone_sdfg
    (loop,) = [r for r in body.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]
    return loop


def test_a_gpu_kernel_loop_staging_a_tile_is_pipelined_at_the_depth_passed(tmp_path: Path) -> None:
    session, kernel = kernel_on(tmp_path, gpu=True)
    (move,) = session.list_moves(SET_PIPELINE_DEPTH, scope=kernel)

    done = session.apply_move(SET_PIPELINE_DEPTH, move.labels, scope=kernel, depth=3)

    assert done.status is MoveStatus.APPLIED, done.reason
    assert staged_loop(session, kernel).pipeline_depth == 3


def test_a_host_loop_or_a_depth_below_two_is_refused(tmp_path: Path) -> None:
    host, kernel = kernel_on(tmp_path, gpu=False)
    gpu, gpu_kernel = kernel_on(tmp_path / "gpu", gpu=True)

    on_host = plan_move(SET_PIPELINE_DEPTH, [(staged_loop(host, kernel), None)], depth=2)
    too_shallow = plan_move(SET_PIPELINE_DEPTH, [(staged_loop(gpu, gpu_kernel), None)], depth=1)

    assert isinstance(on_host, Refused) and "runs on the host" in on_host
    assert isinstance(too_shallow, Refused) and "2 or more" in too_shallow
    assert host.list_moves(SET_PIPELINE_DEPTH, scope=kernel) == []
