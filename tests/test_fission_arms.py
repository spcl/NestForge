"""The Phase-2 fission lever (:mod:`nestforge.fission_arms`): explode a program to statement granularity by
reusing the existing DaCe canon passes (SplitStatements + LoopFission + MapFission), and the agent's real
Phase-2 flow -- fission then fuse back up. Value-preservation (bit-exact vs the un-fissioned reference) is
the invariant on every case.
"""
import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion

from nestforge.fission_arms import fission_to_statements
from nestforge.fusion_arms import apply_fusion, enumerate_fusions

N = dace.symbol("N")
f64 = dace.float64


def nloops(sdfg):
    return sum(1 for c in sdfg.all_control_flow_regions(recursive=True) if isinstance(c, LoopRegion))


def run(sdfg, inputs, n):
    bufs = {k: v.copy() for k, v in inputs.items()}
    sdfg(**bufs, N=n)
    return bufs


def mk(n=48, names=("a", "b", "c"), seed=0):
    rng = np.random.default_rng(seed)
    return {k: rng.random(n) for k in names}


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
    inputs = mk(names=("a", "b", "c"))
    ref = run(two_independent_recurrences.to_sdfg(simplify=True), inputs, 48)
    sdfg = two_independent_recurrences.to_sdfg(simplify=True)
    before = nloops(sdfg)
    applied = fission_to_statements(sdfg)
    got = run(sdfg, inputs, 48)
    assert applied >= 1 and nloops(sdfg) > before  # the loop split into independent statements
    assert all(np.allclose(got[k], ref[k]) for k in inputs)


@pytest.mark.parametrize("prog,names", [
    (two_independent_recurrences, ("a", "b", "c")),
    (three_independent_statements, ("a", "b", "c", "d")),
    (conditional_body, ("a", "b", "c")),
])
def test_fission_is_value_preserving(prog, names):
    inputs = mk(names=names)
    ref = run(prog.to_sdfg(simplify=True), inputs, 48)
    sdfg = prog.to_sdfg(simplify=True)
    fission_to_statements(sdfg)
    got = run(sdfg, inputs, 48)
    assert all(np.allclose(got[k], ref[k]) for k in inputs), f"{prog.name}: fission changed the value"


def test_fission_then_fuse_roundtrip_value_preserving():
    # the agent's real Phase-2 flow: explode to statements, then fuse back up -- must land on the same
    # value as the original program whatever granularity it settles on.
    inputs = mk(names=("a", "b", "c", "d"))
    ref = run(three_independent_statements.to_sdfg(simplify=True), inputs, 48)
    sdfg = three_independent_statements.to_sdfg(simplify=True)
    fission_to_statements(sdfg)
    while True:
        moves = enumerate_fusions(sdfg)
        if not moves:
            break
        apply_fusion(sdfg, moves[0])
    got = run(sdfg, inputs, 48)
    assert all(np.allclose(got[k], ref[k]) for k in inputs)


def test_fission_no_op_on_single_statement():
    @dace.program
    def one_statement(a: f64[N], b: f64[N]):
        for i in range(1, N):
            b[i] = b[i - 1] + a[i]

    inputs = mk(names=("a", "b"))
    ref = run(one_statement.to_sdfg(simplify=True), inputs, 48)
    sdfg = one_statement.to_sdfg(simplify=True)
    fission_to_statements(sdfg)  # nothing independent to split
    got = run(sdfg, inputs, 48)
    assert all(np.allclose(got[k], ref[k]) for k in inputs)
