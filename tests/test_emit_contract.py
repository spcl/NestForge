# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The emitter contract of docs/emitter.md: the caller allocates every buffer, the NumPy signature matches the
manifest, and the caller sizes scratch the way the emitter widened it."""

import numpy as np
import dace
from dace.sdfg.state import LoopRegion

from nestforge.build.arena import make_inputs
from nestforge.ir.emit_numpy import load_emitted, maxsize_loop_scratch, nest_to_numpy, sdfg_to_numpy
from nestforge.ir.extract import Boundary

N = dace.symbol("N")


def run(src, fn, **buffers):
    vars(load_emitted(src, fn))[fn](**buffers)


@dace.program
def dot_scale(x: dace.float64[N], y: dace.float64[N], z: dace.float64[N], out: dace.float64[N]):
    s = np.dot(x, y)
    for i in dace.map[0:N]:
        out[i] = z[i] * s


def test_scalar_transient_consistent_between_libnode_and_tasklet():
    # Dot writes a scalar transient `s`; the map tasklet reads it. Both must name it identically.
    src = sdfg_to_numpy(dot_scale.to_sdfg(simplify=True), "k")
    n = 6
    rng = np.random.default_rng(0)
    x, y, z, out = rng.random(n), rng.random(n), rng.random(n), np.zeros(n)
    run(src, "k", x=x, y=y, z=z, out=out, N=n)
    np.testing.assert_allclose(out, z * (x @ y))


@dace.program
def matvec_return(A: dace.float64[N, N], v: dace.float64[N]):
    return A @ v


def test_return_and_scratch_are_inplace_buffer_params_no_allocation():
    sdfg = matvec_return.to_sdfg(simplify=True)
    src = sdfg_to_numpy(sdfg, "k")
    assert "np.empty" not in src and "np.zeros" not in src, "C-style: caller pre-allocates, kernel must not"
    assert "return " not in src, "C-style: __return is an in-place output buffer param, not a python return"
    # __return is a parameter written in place.
    header = src.splitlines()[0]
    assert "__return" in header
    n = 5
    rng = np.random.default_rng(1)
    A, v = rng.random((n, n)), rng.random(n)
    ret = np.zeros(n)
    run(src, "k", A=A, v=v, __return=ret, N=n)
    np.testing.assert_allclose(ret, A @ v)


def nested_map_sdfg():
    sdfg = dace.SDFG("nested")
    sdfg.add_array("A", [N, N], dace.float64)
    sdfg.add_array("B", [N, N], dace.float64)
    st = sdfg.add_state()
    me_i, mx_i = st.add_map("outer", dict(i="0:N"))
    me_j, mx_j = st.add_map("inner", dict(j="0:N"))
    t = st.add_tasklet("t", {"a"}, {"b"}, "b = a * 2.0")
    rA, wB = st.add_read("A"), st.add_write("B")
    st.add_memlet_path(rA, me_i, me_j, t, dst_conn="a", memlet=dace.Memlet("A[i, j]"))
    st.add_memlet_path(t, mx_j, mx_i, wB, src_conn="b", memlet=dace.Memlet("B[i, j]"))
    return sdfg


def test_nested_map_in_map_emits_nested_for_loops():
    """A map nested inside a map (the multi-nest kernels s2275 / s152 need this) is emitted as NESTED
    ``for`` loops with the inner body at the deeper indent -- NOT dropped, and no longer refused. Guards the
    ``map_lines`` recursion that un-skipped the nested-map corpus kernels."""
    src = sdfg_to_numpy(nested_map_sdfg(), "k")
    assert "for i in range(0, N, 1):" in src and "for j in range(0, N, 1):" in src
    assert "B[i, j] = (A[i, j] * 2.0)" in src
    # numerically correct: exec the emitted kernel and compare to B = A * 2.
    n = 6
    rng = np.random.default_rng(0)
    A, B = rng.random((n, n)), np.zeros((n, n))
    load_emitted(src, "k").k(A, B, n)
    assert np.allclose(B, A * 2.0)


@dace.program
def gather(a: dace.float64[N], b: dace.int64[N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        out[i] = a[b[i]]


def test_indirect_gather_stages_map_entry_read():
    """``a[b[i]]`` is staged as ``<sym> = b[i]`` fed by the map entry; without that load the gather names an
    undefined symbol."""
    sdfg = gather.to_sdfg(simplify=True)
    src = sdfg_to_numpy(sdfg, "k")
    assert "= b[i]" in src, f"map-entry-sourced staging load not emitted:\n{src}"
    n = 8
    rng = np.random.default_rng(0)
    a, b, out = rng.random(n), rng.integers(0, n, size=n).astype(np.int64), np.zeros(n)
    run(src, "k", a=a, b=b, out=out, N=n)  # NameError here if the staged read is dropped
    np.testing.assert_allclose(out, a[b])


def loop_scratch_boundary():
    """A nest with a scratch transient shaped by the LOOP VARIABLE (``tmp[loop_i + 1]``) -- the shape the
    emitter widens to ``N + 1`` so the buffer stays a caller-allocated parameter."""
    # A dedicated symbol name: ``i`` is a common loop variable, and dace's symbol registry rejects a
    # re-declaration with a different dtype, which would couple this test to whatever ran before it.
    loop_i = dace.symbol("loop_i", dace.int64)
    sdfg = dace.SDFG("loop_scratch")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_transient("tmp", [loop_i + 1], dace.float64)
    loop = LoopRegion("loop", "loop_i < N", "loop_i", "loop_i = 0", "loop_i = loop_i + 1")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    body.add_edge(
        body.add_read("a"), None, body.add_tasklet("t", {"i0"}, {"o0"}, "o0 = i0 + 1.0"), "i0", dace.Memlet("a[0]")
    )
    return Boundary(
        inputs=["a"], outputs=["a"], symbols=["N"], nsdfg_node=None, state=None, standalone_sdfg=sdfg, parent_sdfg=None
    )


def test_make_inputs_sizes_scratch_the_way_the_emitter_widened_it():
    """make_inputs sized scratch from the RAW descriptor while the emitted kernel is written against the
    ``maxsize_loop_scratch``-widened one, so the caller handed the kernel a buffer smaller than it indexes
    -- a write past the end of the allocation across the ABI (heap corruption in the forked child)."""
    boundary = loop_scratch_boundary()
    sizes = {"N": 4}
    widened = maxsize_loop_scratch(boundary.standalone_sdfg, boundary.symbols).arrays["tmp"]
    assert str(widened.shape[0]) == "N + 1"  # the extent the emitted kernel addresses

    got = make_inputs(boundary, sizes, seed=0)["tmp"]
    assert got.shape == (sizes["N"] + 1,), "scratch allocated from the raw (smaller) shape, not the emitted one"


def test_the_emitted_signature_takes_the_scratch_buffer_the_caller_allocates():
    boundary = loop_scratch_boundary()
    assert nest_to_numpy(boundary, "k").splitlines()[0] == "def k(a, tmp, N):"
