# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A move that answers ``applied`` leaves a program that computes the same and still builds; one that cannot, is not
offered."""

import dace
import numpy as np

from nestforge.ir.introspect import guarded_work_depth
from nestforge.ir.names import top_level_nsdfgs
from nestforge.records import MoveStatus
from nestforge.session import Session

N = dace.symbol("N", dace.int64, positive=True)
SIZE = 12


@dace.program
def fission_then_overwrite(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
    for i in dace.map[1 : N - 1]:
        b[i] = c[i] * d[i]
        a[i] = c[i] + d[i]
    d[:] = a + b


@dace.program
def stops_early(a: dace.float64[N], b: dace.float64[N]):
    for i in range(1, N):
        a[i] = np.sin(a[i - 1]) + b[i]
        if a[i] > 1.2:
            break
        b[i] = a[i] * 2


@dace.program
def stencil_of_a_local(a: dace.float64[N], b: dace.float64[N]):
    w = np.zeros(N)
    for i in dace.map[0:N]:
        w[i] = a[i] * a[i]
    for i in dace.map[1 : N - 1]:
        b[i] = w[i - 1] + w[i + 1]


@dace.program
def while_in_a_carried_loop(a: dace.float64[N], b: dace.int64[N]):
    for i in range(1, N):
        j = 0
        while j < i and a[j] < a[i - 1]:
            j = j + 1
        a[i] = a[i - 1] + j


def session(program: dace.frontend.python.parser.DaceProgram, tmp_path) -> Session:
    sut = Session(program.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": SIZE})
    sut.canonicalize()
    return sut


def test_doall_fission_leaves_no_nest_at_the_top_and_the_overwriting_map_runs_once(tmp_path):
    sut = session(fission_then_overwrite, tmp_path)
    (move,) = sut.list_moves("doall-fission")

    assert sut.apply_move("doall-fission", move.labels).status is MoveStatus.APPLIED

    assert not top_level_nsdfgs(sut.sdfg)
    a, b, c, d = (np.arange(SIZE, dtype=np.float64) * scale + 1.0 for scale in (1.0, 2.0, 3.0, 0.5))
    expected = a.copy(), b.copy(), d.copy()
    expected[1][1:-1] = c[1:-1] * d[1:-1]
    expected[0][1:-1] = c[1:-1] + d[1:-1]
    sut.sdfg(a=a, b=b, c=c, d=d, N=SIZE)
    np.testing.assert_array_equal(a, expected[0])
    np.testing.assert_array_equal(d, expected[0] + expected[1])


def test_a_loop_with_a_break_is_not_offered_tile_for_and_refuses_it(tmp_path):
    sut = session(stops_early, tmp_path)
    (loop,) = (row for row in sut.row_index() if row.startswith("for"))

    assert sut.list_moves("tile-for") == []
    assert sut.apply_move("tile-for", [loop], tile_sizes=[4]).status is MoveStatus.ILLEGAL


def test_doall_fission_refuses_a_body_bound_to_an_expression_of_the_map_but_subgraph_fission_stays(tmp_path):
    sut = session(stencil_of_a_local, tmp_path)
    (fusion,) = sut.list_moves("doall-fusion")
    assert sut.apply_move("doall-fusion", fusion.labels).reason == "OTFMapFusion"

    assert sut.list_moves("doall-fission") == []
    assert sut.list_moves("subgraph-fission")
    sut.sdfg.validate()


def test_a_while_loop_inside_a_loop_gets_a_tree_without_costs_and_the_reason(tmp_path):
    sut = session(while_in_a_carried_loop, tmp_path)

    assert guarded_work_depth(sut.sdfg)[0] == {}
    assert sut.costs_skipped is not None and "NodeNotFoundError" in sut.costs_skipped
    assert "no work= or depth=" in sut.describe()
