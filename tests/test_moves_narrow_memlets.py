# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""interchange-loop-map builds a nest inside a map; its outer memlets must name what one iteration touches, not the
whole array (#24): fusions and DaCe's schedule inference read them."""

import dace
import numpy as np

from nestforge.records import MoveStatus
from nestforge.session import Session
from nestforge.stages.canonicalize import Targets

N = dace.symbol("N")
#: Rows and columns of the test arrays.
SIZE = 16


@dace.program
def row_recurrence(p0: dace.float64[N], p: dace.float64[N, N]):
    for i in range(N):
        p[i, 0] = p0[i]
    for j in range(1, N):
        for i in range(N):
            p[i, j] = 0.5 * p[i, j - 1] + 1.0


def nest_memlets(sdfg: dace.SDFG) -> list[tuple[str, str]]:
    """``(parameter of the enclosing map, subset of p)`` on every edge into or out of a nest that sits in a map."""
    found = []
    for state in sdfg.states():
        for node in state.nodes():
            entry = state.entry_node(node) if isinstance(node, dace.nodes.NestedSDFG) else None
            if entry is not None:
                edges = [*state.in_edges(node), *state.out_edges(node)]
                found += [(entry.map.params[0], str(e.data.subset)) for e in edges if e.data.data == "p"]
    return found


def assert_per_iteration(memlets: list[tuple[str, str]]) -> None:
    assert memlets
    assert all(subset.startswith(f"{param}, ") for param, subset in memlets), memlets


def test_the_nest_an_interchange_builds_gets_per_iteration_memlets():
    s = Session(row_recurrence.to_sdfg(simplify=True), targets=Targets(gpu=False), sizes={"N": SIZE})
    s.canonicalize()
    (move,) = s.list_moves("interchange-loop-map")
    assert s.apply_move(move.kind, move.labels, move.epoch).status is MoveStatus.APPLIED
    assert_per_iteration(nest_memlets(s.sdfg))
    s.finish_moves()
    assert_per_iteration(nest_memlets(s.sdfg))

    rng = np.random.default_rng(0)
    p0 = rng.random(SIZE)
    p = np.zeros((SIZE, SIZE))
    s.sdfg(p0=p0, p=p, N=SIZE)
    expected = np.zeros((SIZE, SIZE))
    expected[:, 0] = p0
    for j in range(1, SIZE):
        expected[:, j] = 0.5 * expected[:, j - 1] + 1.0
    assert np.allclose(p, expected)
