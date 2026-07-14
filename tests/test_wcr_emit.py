"""Emit WCR (reduction) tasklet out-edges as augmented assignments and check they accumulate.

A ``out[0] += a[i]`` reduction or a ``hist[bin] += w`` scatter is a tasklet whose output edge carries
a WCR. Sequential numpy emission turns it into ``target = target + tmp`` (Sum; ``np.maximum`` for Max,
...), correct even when many iterations hit the same element.
"""
import inspect

import numpy as np
import pytest

pytest.importorskip("dace")

import dace as dc

from nestforge.emit_numpy import sdfg_to_numpy

N = dc.symbol("N", dtype=dc.int64)
M = dc.symbol("M", dtype=dc.int64)


@dc.program
def reduce_sum(a: dc.float64[N], out: dc.float64[1]):
    out[0] = 0.0
    for i in dc.map[0:N]:
        out[0] += a[i]


@dc.program
def hist_scatter(idx: dc.int64[N], w: dc.float64[N], hist: dc.float64[M]):
    for i in dc.map[0:N]:
        hist[idx[i]] += w[i]


def run(program, fn_name, sizes, inputs):
    src = sdfg_to_numpy(program.to_sdfg(simplify=True), fn_name)
    ns = {"np": np}
    exec(src, ns)
    call = dict(inputs)
    for p in inspect.signature(ns[fn_name]).parameters:
        if p not in call:
            call[p] = sizes.get(p)
    ns[fn_name](**call)
    return call, src


def test_wcr_sum_reduction():
    rng = np.random.default_rng(0)
    a = rng.random(32)
    call, src = run(reduce_sum, "reduce_sum", dict(N=32), dict(a=a.copy(), out=np.zeros(1)))
    assert "+ __wcr_" in src  # augmented assignment, not a plain overwrite
    np.testing.assert_allclose(call["out"][0], a.sum())


def test_wcr_at_map_exit_from_nested_map_raises():
    """A reduction (WCR) reaching a map exit from a NESTED map (not an in-scope accumulator access node)
    is emitted by no numpy path -- silently dropping it would mis-emit the reduction as a no-op. Refuse
    it so the ExternalCall falls back to the DaCe variant instead of a wrong kernel."""
    from dace.sdfg import nodes  # noqa: F401  (kept explicit for the graph-build intent)
    from nestforge.emit_numpy import UnsupportedNest

    sdfg = dc.SDFG("nested_wcr")
    sdfg.add_array("A", [N, N], dc.float64)
    sdfg.add_array("out", [N], dc.float64)
    sdfg.add_transient("acc", [1], dc.float64)
    st = sdfg.add_state()
    a = st.add_access("A")
    o = st.add_access("out")
    ome, omx = st.add_map("outer", dict(i="0:N"))
    ime, imx = st.add_map("inner", dict(j="0:N"))
    acc = st.add_access("acc")
    t = st.add_tasklet("t", {"inp"}, {"res"}, "res = inp")
    st.add_memlet_path(a, ome, ime, t, dst_conn="inp", memlet=dc.Memlet("A[i, j]"))
    st.add_edge(t, "res", acc, None, dc.Memlet("acc[0]"))
    # acc reduces out through BOTH exits; the inner->outer exit edge carries the WCR from a MapExit source.
    st.add_memlet_path(acc, imx, omx, o, memlet=dc.Memlet("out[i]", wcr="lambda x, y: x + y"))
    sdfg.validate()
    with pytest.raises(UnsupportedNest, match="reduction"):
        sdfg_to_numpy(sdfg, "nested_wcr")


def test_wcr_scatter_data_dependent_index():
    """A scatter ``hist[idx[i]] += w[i]`` -- the histogram pattern -- accumulates into a data-dependent
    element; on DaCe branches where the indirect write lowers to a nested SDFG it needs the widen pass."""
    pytest.importorskip("dace.transformation.interstate.expand_nested_sdfg_inputs")
    from nestforge.emit_numpy import UnsupportedNest
    rng = np.random.default_rng(1)
    Nv, Mv = 50, 6
    idx = rng.integers(0, Mv, Nv).astype(np.int64)
    w = rng.random(Nv)
    try:
        call, src = run(hist_scatter, "hist_scatter", dict(N=Nv, M=Mv),
                        dict(idx=idx.copy(), w=w.copy(), hist=np.zeros(Mv)))
    except UnsupportedNest:
        pytest.skip("indirect-write nesting unavailable in this DaCe")
    ref = np.zeros(Mv)
    np.add.at(ref, idx, w)
    np.testing.assert_allclose(call["hist"], ref)
