# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""map-fusion of two row recurrences after interchange-loop-map (spcl/NestForge#27): MapFusionVertical refuses the
pair, the in-place fusion makes it, and the fused program still computes the recurrences."""

import dace
import numpy as np

from nestforge.ir.names import top_level_map_entries
from nestforge.records import MoveStatus
from nestforge.session import Session
from nestforge.stages.canonicalize import Targets

N = dace.symbol("N")
#: Rows and columns of the test arrays.
SIZE = 24


@dace.program
def two_row_recurrences(p0: dace.float64[N], q: dace.float64[N, N]):
    p = np.empty((N, N), dtype=np.float64)
    for i in range(N):
        p[i, 0] = p0[i]
    for j in range(1, N):
        for i in range(N):
            p[i, j] = 0.5 * p[i, j - 1] + 1.0
    for j in range(1, N):
        for i in range(N):
            q[i, j] = q[i, j - 1] + p[i, j - 1]


def reference(p0: np.ndarray, q: np.ndarray) -> None:
    p = np.empty((len(p0), len(p0)))
    p[:, 0] = p0
    for j in range(1, len(p0)):
        p[:, j] = 0.5 * p[:, j - 1] + 1.0
    for j in range(1, len(p0)):
        q[:, j] = q[:, j - 1] + p[:, j - 1]


def writes(state: dace.SDFGState, entry: dace.nodes.MapEntry) -> set[str]:
    return {e.data.data for e in state.out_edges(state.exit_node(entry)) if e.data.data}


def interchange_sweep(s: Session, array: str) -> None:
    rows = s.row_index()
    for move in s.list_moves("interchange-loop-map"):
        entry, state = rows[move.labels[1]]
        if writes(state, entry) == {array}:
            assert s.apply_move(move.kind, move.labels, move.epoch).status is MoveStatus.APPLIED
            return
    raise AssertionError(f"no interchange offered for the sweep writing {array}")


def top_level_maps(sdfg: dace.SDFG) -> int:
    return sum(len(top_level_map_entries(state)) for state in sdfg.states())


def test_row_recurrences_fuse_in_place_after_the_interchanges():
    s = Session(two_row_recurrences.to_sdfg(simplify=True), targets=Targets(gpu=False), sizes={"N": SIZE})
    s.canonicalize()
    interchange_sweep(s, "p")
    interchange_sweep(s, "q")
    s.finish_moves()
    before = top_level_maps(s.sdfg)
    moves = s.list_moves("map-fusion")
    assert moves, "the two row maps must be offered as a map-fusion"
    result = s.apply_move(moves[0].kind, moves[0].labels, moves[0].epoch)
    assert (result.status, result.reason) == (MoveStatus.APPLIED, "MapFusionInPlace")
    assert top_level_maps(s.sdfg) == before - 1
    s.sdfg.validate()

    rng = np.random.default_rng(0)
    p0, q = rng.random(SIZE), rng.random((SIZE, SIZE))
    expected = q.copy()
    reference(p0, expected)
    s.sdfg(p0=p0, q=q, N=SIZE)
    assert np.allclose(q, expected)
