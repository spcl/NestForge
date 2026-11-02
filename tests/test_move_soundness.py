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
from nestforge.stages.performance import Reference

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


@dace.program
def two_sums(a: dace.float64[N], b: dace.float64[N], d: dace.float64[N]):
    s = 0.0
    for i in dace.map[0:N]:
        s += a[i]
    d[0] = s
    s = 0.0
    for i in dace.map[0:N]:
        s += a[i]
    b[0] = s


def largest_negative() -> dace.SDFG:
    """``b[0] = max(a)`` by a map reducing into a transient, built by hand: the frontend's ``max`` keeps no WCR."""
    sdfg = dace.SDFG("largest_negative")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.float64)
    sdfg.add_scalar("s", dace.float64, transient=True)
    start = sdfg.add_state()
    below = start.add_tasklet("below", {}, {"y"}, "y = -1.0e300")
    start.add_edge(below, "y", start.add_write("s"), None, dace.Memlet("s[0]"))
    state = sdfg.add_state_after(start)
    entry, exit_node = state.add_map("reduce", {"i": "0:N"})
    tasklet = state.add_tasklet("keep", {"x"}, {"y"}, "y = x")
    state.add_memlet_path(state.add_read("a"), entry, tasklet, dst_conn="x", memlet=dace.Memlet("a[i]"))
    total = state.add_access("s")
    reduced = dace.Memlet("s[0]", wcr="lambda x, y: max(x, y)")
    state.add_memlet_path(tasklet, exit_node, total, src_conn="y", memlet=reduced)
    copy = state.add_tasklet("copy", {"x"}, {"y"}, "y = x")
    state.add_edge(total, None, copy, "x", dace.Memlet("s[0]"))
    state.add_edge(copy, "y", state.add_write("b"), None, dace.Memlet("b[0]"))
    return sdfg


@dace.program
def stencil(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[1 : N - 1]:
        b[i] = a[i - 1] + a[i + 1]


def session(program: dace.frontend.python.parser.DaceProgram, tmp_path) -> Session:
    sut = Session(program.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": SIZE})
    sut.canonicalize()
    return sut


def test_two_reduction_kernels_each_start_from_nothing_not_from_the_stack(tmp_path):
    sut = session(two_sums, tmp_path)
    a, zero = np.arange(1.0, SIZE + 1), np.zeros(SIZE)
    expected = {"b": np.array([a.sum(), *zero[1:]]), "d": np.array([a.sum(), *zero[1:]])}

    report = sut.performance({"a": a, "b": zero, "d": zero}, Reference(expected, 0.0, 0.0), codegen="first")

    assert report.structured.error == "" and report.structured.correct is True, report.structured


def test_a_max_kernel_of_negative_values_starts_below_them(tmp_path):
    sut = Session(largest_negative(), work_dir=str(tmp_path), sizes={"N": SIZE})
    sut.canonicalize()
    a, zero = -np.arange(1.0, SIZE + 1), np.zeros(SIZE)
    expected = {"b": np.array([-1.0, *zero[1:]])}

    report = sut.performance({"a": a, "b": zero}, Reference(expected, 0.0, 0.0), codegen="first")

    assert report.structured.error == "" and report.structured.correct is True, report.structured


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


def test_a_doall_tiled_twice_keeps_list_moves_costs_and_the_result(tmp_path):
    sut = session(stencil, tmp_path)
    for size in (4, 2):
        sut.apply_move("tile-doall", sut.list_moves("tile-doall")[-1].labels, tile_sizes=[size])

    assert sut.list_moves()
    sut.describe()
    assert sut.costs_skipped is None and sut.costs
    a, b = np.arange(1.0, SIZE + 1) ** 2, np.zeros(SIZE)
    sut.sdfg(a=a, b=b, N=SIZE)
    np.testing.assert_array_equal(b[1:-1], a[:-2] + a[2:])
