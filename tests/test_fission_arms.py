# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The stage 2 fission moves (:mod:`nestforge.stages.moves`): ``loop-fission`` and ``map-fission`` applied until
nothing splits, and the agent's flow of fission then fusion. Value-preservation (bit-exact vs the un-fissioned
reference) is the invariant on every case.
"""

import dace
import numpy as np
import pytest
from dace.sdfg.state import LoopRegion
from dace.transformation.helpers import nest_state_subgraph
from helpers import apply_move, fission_to_fixpoint, fusion_moves, random_vectors, run

from nestforge.ir.names import normalize_labels
from nestforge.stages.moves import legal_moves

N = dace.symbol("N")
f64 = dace.float64


def nloops(sdfg):
    return sum(1 for c in sdfg.all_control_flow_regions(recursive=True) if isinstance(c, LoopRegion))


@dace.program
def two_independent_recurrences(a: f64[N], b: f64[N], c: f64[N]):
    for i in range(1, N):
        b[i] = b[i - 1] + a[i]
        c[i] = c[i - 1] + a[i]


@dace.program
def three_independent_statements(a: f64[N], b: f64[N], c: f64[N], d: f64[N]):
    for i in range(1, N):
        b[i] = b[i - 1] + a[i]
        c[i] = c[i - 1] * a[i]
        d[i] = d[i - 1] - a[i]


@dace.program
def conditional_body(a: f64[N], b: f64[N], c: f64[N]):
    for i in range(1, N):
        if a[i] > 0.5:
            b[i] = b[i - 1] + a[i]
            c[i] = c[i - 1] + a[i]
        else:
            b[i] = b[i - 1]
            c[i] = c[i - 1]


def test_fission_splits_independent_recurrences_value_preserving():
    inputs = random_vectors(names=("a", "b", "c"))
    ref = run(two_independent_recurrences.to_sdfg(simplify=True), inputs, 48)
    sdfg = two_independent_recurrences.to_sdfg(simplify=True)
    before = nloops(sdfg)
    applied = fission_to_fixpoint(sdfg)
    got = run(sdfg, inputs, 48)
    assert applied >= 1 and nloops(sdfg) > before  # the loop split into independent statements
    for k in inputs:
        np.testing.assert_array_equal(got[k], ref[k], err_msg=k)


@pytest.mark.parametrize(
    "prog,names",
    [
        (two_independent_recurrences, ("a", "b", "c")),
        (three_independent_statements, ("a", "b", "c", "d")),
        (conditional_body, ("a", "b", "c")),
    ],
)
def test_fission_is_value_preserving(prog, names):
    inputs = random_vectors(names=names)
    ref = run(prog.to_sdfg(simplify=True), inputs, 48)
    sdfg = prog.to_sdfg(simplify=True)
    fission_to_fixpoint(sdfg)
    got = run(sdfg, inputs, 48)
    for k in inputs:
        np.testing.assert_array_equal(got[k], ref[k], err_msg=f"{prog.name}: fission changed the value")


def test_fission_then_fuse_roundtrip_value_preserving():
    # the agent's stage 2 flow: split what splits, then fuse back up -- must land on the same
    # value as the original program whatever granularity it settles on.
    inputs = random_vectors(names=("a", "b", "c", "d"))
    ref = run(three_independent_statements.to_sdfg(simplify=True), inputs, 48)
    sdfg = three_independent_statements.to_sdfg(simplify=True)
    fission_to_fixpoint(sdfg)
    while moves := fusion_moves(sdfg):
        apply_move(sdfg, moves[0])
    got = run(sdfg, inputs, 48)
    for k in inputs:
        np.testing.assert_array_equal(got[k], ref[k], err_msg=k)


def map_with_nested_body():
    """A map whose sole body child is a NestedSDFG holding two independent output groups -- MapFission's
    map-with-nested-SDFG pattern, the shape the ``map-fission`` move takes."""
    sdfg = dace.SDFG("map_with_nested_body")
    sdfg.add_array("a", [N], f64)
    sdfg.add_array("b", [N], f64)
    sdfg.add_array("c", [N], f64)
    state = sdfg.add_state()

    rnode = state.add_read("a")
    me, mx = state.add_map("outer", dict(i="0:N"))
    t1 = state.add_tasklet("one", {"x"}, {"y"}, "y = x + 1.0")
    t2 = state.add_tasklet("two", {"x"}, {"y"}, "y = x * 2.0")
    state.add_memlet_path(rnode, me, t1, memlet=dace.Memlet(data="a", subset="i"), dst_conn="x")
    state.add_memlet_path(t1, mx, state.add_write("b"), memlet=dace.Memlet(data="b", subset="i"), src_conn="y")
    state.add_memlet_path(rnode, me, t2, memlet=dace.Memlet(data="a", subset="i"), dst_conn="x")
    state.add_memlet_path(t2, mx, state.add_write("c"), memlet=dace.Memlet(data="c", subset="i"), src_conn="y")
    sdfg.validate()

    nest_state_subgraph(sdfg, state, state.scope_subgraph(me, include_entry=False, include_exit=False))
    sdfg.validate()
    return sdfg, me


def test_map_fission_is_listed_for_a_nested_sdfg_body_and_applies():
    # The move must match MapFission's map-with-nested-SDFG pattern (expr_index=1). Matched against the default
    # map-with-subgraph pattern instead, the lone NestedSDFG body reads as a single component and no move is listed.
    sdfg, me = map_with_nested_body()
    normalize_labels(sdfg)
    moves = legal_moves(sdfg, "map-fission")

    assert moves == [("map-fission", (me.map.label,))]
    assert apply_move(sdfg, moves[0]) == "MapFission"
    sdfg.validate()


def test_map_fission_preserves_values():
    inputs = random_vectors(names=("a", "b", "c"))
    sdfg, _ = map_with_nested_body()
    ref = run(map_with_nested_body()[0], inputs, 48)

    assert fission_to_fixpoint(sdfg) >= 1
    got = run(sdfg, inputs, 48)
    for k in inputs:
        np.testing.assert_array_equal(got[k], ref[k], err_msg="map fission changed the value")


def test_map_fission_no_moves_without_independent_groups():
    # The listing must stay honest in the other direction: a single-statement body has nothing to split.
    @dace.program
    def one_statement_map(a: f64[N], b: f64[N]):
        for i in dace.map[0:N]:
            b[i] = a[i] + 1.0

    assert legal_moves(one_statement_map.to_sdfg(simplify=True), "map-fission") == []


def test_fission_no_op_on_single_statement():

    @dace.program
    def one_statement(a: f64[N], b: f64[N]):
        for i in range(1, N):
            b[i] = b[i - 1] + a[i]

    inputs = random_vectors(names=("a", "b"))
    ref = run(one_statement.to_sdfg(simplify=True), inputs, 48)
    sdfg = one_statement.to_sdfg(simplify=True)
    fission_to_fixpoint(sdfg)  # nothing independent to split
    got = run(sdfg, inputs, 48)
    for k in inputs:
        np.testing.assert_array_equal(got[k], ref[k], err_msg=k)
