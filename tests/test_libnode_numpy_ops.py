# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""NumPy emission of the BLAS, LinAlg and standard library nodes an SDFG can carry.

Each node is built alone, emitted with :func:`sdfg_to_numpy`, run, and compared bit for bit with the NumPy operation it
stands for: the emission renames, it does not approximate.
"""

import inspect

import numpy as np
import pytest

import dace as dc
from dace import Memlet

from nestforge.ir.emit_numpy import UnsupportedNest, load_emitted
from nestforge.ir.emit_libnode import scalar_elem

from helpers import sdfg_to_numpy

N, M, K = (dc.symbol(s, dtype=dc.int64) for s in "NMK")
F = dc.float64


def build(name, node, arrays, in_wiring, out_wiring):
    """A one-state SDFG with ``node`` wired to fresh arrays; forced connectors let one name be input and output."""
    sdfg = dc.SDFG(name)
    for data, (shape, dt) in arrays.items():
        sdfg.add_array(data, shape, dt)
    st = sdfg.add_state()
    st.add_node(node)
    for conn, data in in_wiring:
        node.add_in_connector(conn, force=True)
        st.add_edge(st.add_read(data), None, node, conn, Memlet.from_array(data, sdfg.arrays[data]))
    for conn, data in out_wiring:
        node.add_out_connector(conn, force=True)
        st.add_edge(node, conn, st.add_write(data), None, Memlet.from_array(data, sdfg.arrays[data]))
    sdfg.validate()
    return sdfg


def run(sdfg, fn, buffers, sizes):
    """Emit ``sdfg`` as ``def fn(...)``, import it, and call it with the buffers/sizes it actually takes
    (a size symbol absent from every shape is not in the signature). Returns the (mutated) buffers."""
    src = sdfg_to_numpy(sdfg, fn)
    kernel = vars(load_emitted(src, fn))[fn]
    params = set(inspect.signature(kernel).parameters)
    kernel(**{k: v for k, v in {**buffers, **sizes}.items() if k in params})
    return buffers, src


@pytest.fixture
def rng() -> np.random.Generator:
    """Each test draws the same inputs whatever ran before it."""
    return np.random.default_rng(0)


# BLAS: the nodes a MatMul expands into


def test_gemm_alpha_beta(rng):
    from dace.libraries.blas.nodes.gemm import Gemm

    g = Gemm("g")
    g.alpha, g.beta = 2.0, 3.0
    sdfg = build(
        "gemm",
        g,
        {"A": ((N, K), F), "B": ((K, M), F), "C": ((N, M), F)},
        [("_a", "A"), ("_b", "B"), ("_c", "C")],
        [("_c", "C")],
    )
    A, B, C = rng.random((3, 4)), rng.random((4, 5)), rng.random((3, 5))
    buffers, src = run(sdfg, "gemm", {"A": A.copy(), "B": B.copy(), "C": C.copy()}, {"N": 3, "M": 5, "K": 4})
    assert "@" in src
    np.testing.assert_array_equal(buffers["C"], 2 * (A @ B) + 3 * C)


def test_gemm_transA_transB(rng):
    from dace.libraries.blas.nodes.gemm import Gemm

    g = Gemm("g")
    g.transA, g.transB = True, True
    sdfg = build(
        "gemmT", g, {"A": ((K, N), F), "B": ((M, K), F), "C": ((N, M), F)}, [("_a", "A"), ("_b", "B")], [("_c", "C")]
    )
    A, B, C = rng.random((4, 3)), rng.random((5, 4)), np.zeros((3, 5))
    buffers, _ = run(sdfg, "gemmT", {"A": A.copy(), "B": B.copy(), "C": C}, {"N": 3, "M": 5, "K": 4})
    np.testing.assert_array_equal(buffers["C"], A.T @ B.T)


def test_gemv_alpha_beta_transA(rng):
    from dace.libraries.blas.nodes.gemv import Gemv

    gv = Gemv("gv")
    gv.alpha, gv.beta = 2.0, 1.0
    sdfg = build(
        "gemv",
        gv,
        {"A": ((N, M), F), "x": ((M,), F), "y": ((N,), F)},
        [("_A", "A"), ("_x", "x"), ("_y", "y")],
        [("_y", "y")],
    )
    A, x, y = rng.random((3, 4)), rng.random(4), rng.random(3)
    buffers, _ = run(sdfg, "gemv", {"A": A.copy(), "x": x.copy(), "y": y.copy()}, {"N": 3, "M": 4})
    np.testing.assert_array_equal(buffers["y"], 2 * (A @ x) + y)


def test_ger_rank1_update(rng):
    from dace.libraries.blas.nodes.ger import Ger

    gr = Ger("gr")
    gr.alpha, gr.n, gr.m = 2.0, N, M
    sdfg = build(
        "ger",
        gr,
        {"x": ((N,), F), "y": ((M,), F), "A": ((N, M), F), "res": ((N, M), F)},
        [("_x", "x"), ("_y", "y"), ("_A", "A")],
        [("_res", "res")],
    )
    x, y, A = rng.random(3), rng.random(5), rng.random((3, 5))
    buffers, src = run(
        sdfg, "ger", {"x": x.copy(), "y": y.copy(), "A": A.copy(), "res": np.zeros((3, 5))}, {"N": 3, "M": 5}
    )
    assert "np.outer" in src
    np.testing.assert_array_equal(buffers["res"], 2 * np.outer(x, y) + A)


def test_axpy(rng):
    from dace.libraries.blas.nodes.axpy import Axpy

    ax = Axpy("ax")
    ax.a, ax.n = 3.0, N
    sdfg = build(
        "axpy", ax, {"x": ((N,), F), "y": ((N,), F), "res": ((N,), F)}, [("_x", "x"), ("_y", "y")], [("_res", "res")]
    )
    x, y = rng.random(6), rng.random(6)
    buffers, _ = run(sdfg, "axpy", {"x": x.copy(), "y": y.copy(), "res": np.zeros(6)}, {"N": 6})
    np.testing.assert_array_equal(buffers["res"], 3 * x + y)


def test_batched_matmul(rng):
    from dace.libraries.blas.nodes.batched_matmul import BatchedMatMul

    sdfg = build(
        "bmm",
        BatchedMatMul("bmm"),
        {"a": ((3, N, K), F), "b": ((3, K, M), F), "c": ((3, N, M), F)},
        [("_a", "a"), ("_b", "b")],
        [("_c", "c")],
    )
    a, b = rng.random((3, 2, 4)), rng.random((3, 4, 5))
    buffers, _ = run(sdfg, "bmm", {"a": a.copy(), "b": b.copy(), "c": np.zeros((3, 2, 5))}, {"N": 2, "M": 5, "K": 4})
    np.testing.assert_array_equal(buffers["c"], a @ b)


def test_batched_matmul_transB(rng):
    from dace.libraries.blas.nodes.batched_matmul import BatchedMatMul

    bm = BatchedMatMul("bm")
    bm.transB = True
    sdfg = build(
        "bmt",
        bm,
        {"a": ((3, N, K), F), "b": ((3, M, K), F), "c": ((3, N, M), F)},
        [("_a", "a"), ("_b", "b")],
        [("_c", "c")],
    )
    a, b = rng.random((3, 2, 4)), rng.random((3, 5, 4))
    buffers, _ = run(sdfg, "bmt", {"a": a.copy(), "b": b.copy(), "c": np.zeros((3, 2, 5))}, {"N": 2, "M": 5, "K": 4})
    np.testing.assert_array_equal(buffers["c"], a @ np.swapaxes(b, -1, -2))


def test_batched_matmul_beta_refused():
    """No ``_c`` input connector exists to accumulate into, so a non-zero beta cannot be honored."""
    from dace.libraries.blas.nodes.batched_matmul import BatchedMatMul

    bm = BatchedMatMul("bm")
    bm.beta = 1.0
    sdfg = build(
        "bmb",
        bm,
        {"a": ((3, N, K), F), "b": ((3, K, M), F), "c": ((3, N, M), F)},
        [("_a", "a"), ("_b", "b")],
        [("_c", "c")],
    )
    with pytest.raises(UnsupportedNest, match="beta"):
        sdfg_to_numpy(sdfg, "bmb")


# Einsum: operand order is by (sorted) connector name


def test_einsum_three_operand(rng):
    from dace.libraries.blas.nodes.einsum import Einsum

    es = Einsum("es")
    es.einsum_str = "ik,kj,j->i"
    sdfg = build(
        "es",
        es,
        {"a": ((N, K), F), "b": ((K, M), F), "v": ((M,), F), "o": ((N,), F)},
        [("a", "a"), ("b", "b"), ("v", "v")],
        [("o", "o")],
    )
    a, b, v = rng.random((3, 4)), rng.random((4, 5)), rng.random(5)
    buffers, src = run(
        sdfg, "es", {"a": a.copy(), "b": b.copy(), "v": v.copy(), "o": np.zeros(3)}, {"N": 3, "M": 5, "K": 4}
    )
    assert "np.einsum" in src
    np.testing.assert_array_equal(buffers["o"], np.einsum("ik,kj,j->i", a, b, v))


def test_einsum_alpha_beta_properties(rng):
    """``out = alpha * einsum + beta * out_prior``; beta reads the output buffer in place (RHS-first)."""
    from dace.libraries.blas.nodes.einsum import Einsum

    es = Einsum("es")
    es.einsum_str, es.alpha, es.beta = "ik,kj->ij", 2.0, 3.0
    sdfg = build(
        "esab", es, {"a": ((N, K), F), "b": ((K, M), F), "o": ((N, M), F)}, [("a", "a"), ("b", "b")], [("o", "o")]
    )
    a, b, o = rng.random((3, 4)), rng.random((4, 5)), rng.random((3, 5))
    buffers, _ = run(sdfg, "esab", {"a": a.copy(), "b": b.copy(), "o": o.copy()}, {"N": 3, "M": 5, "K": 4})
    np.testing.assert_array_equal(buffers["o"], 2 * np.einsum("ik,kj->ij", a, b) + 3 * o)


def test_einsum_runtime_alpha_connector(rng):
    """A data-driven ``_alpha`` scalar connector multiplies the contraction (composes with the property)."""
    from dace.libraries.blas.nodes.einsum import Einsum

    es = Einsum("es")
    es.einsum_str = "ik,kj->ij"
    sdfg = build(
        "esco",
        es,
        {"a": ((N, K), F), "b": ((K, M), F), "al": ((1,), F), "o": ((N, M), F)},
        [("a", "a"), ("b", "b"), ("_alpha", "al")],
        [("o", "o")],
    )
    a, b = rng.random((3, 4)), rng.random((4, 5))
    buffers, _ = run(
        sdfg,
        "esco",
        {"a": a.copy(), "b": b.copy(), "al": np.array([4.0]), "o": np.zeros((3, 5))},
        {"N": 3, "M": 5, "K": 4},
    )
    np.testing.assert_array_equal(buffers["o"], 4.0 * np.einsum("ik,kj->ij", a, b))


# TensorDot / Inv


def test_tensordot_contract(rng):
    from dace.libraries.linalg.nodes.tensordot import TensorDot

    sdfg = build(
        "td",
        TensorDot("td", left_axes=[2], right_axes=[0]),
        {"l": ((2, 3, 4), F), "r": ((4, 5), F), "o": ((2, 3, 5), F)},
        [("_left_tensor", "l"), ("_right_tensor", "r")],
        [("_out_tensor", "o")],
    )
    L, R = rng.random((2, 3, 4)), rng.random((4, 5))
    buffers, src = run(sdfg, "td", {"l": L.copy(), "r": R.copy(), "o": np.zeros((2, 3, 5))}, {})
    assert "np.tensordot" in src
    np.testing.assert_array_equal(buffers["o"], np.tensordot(L, R, axes=([2], [0])))


def test_tensordot_permutation(rng):
    from dace.libraries.linalg.nodes.tensordot import TensorDot

    td = TensorDot("td", left_axes=[2], right_axes=[0])
    td.permutation = [2, 0, 1]
    sdfg = build(
        "tdp",
        td,
        {"l": ((2, 3, 4), F), "r": ((4, 5), F), "o": ((5, 2, 3), F)},
        [("_left_tensor", "l"), ("_right_tensor", "r")],
        [("_out_tensor", "o")],
    )
    L, R = rng.random((2, 3, 4)), rng.random((4, 5))
    buffers, _ = run(sdfg, "tdp", {"l": L.copy(), "r": R.copy(), "o": np.zeros((5, 2, 3))}, {})
    np.testing.assert_array_equal(buffers["o"], np.transpose(np.tensordot(L, R, axes=([2], [0])), [2, 0, 1]))


def test_inv(rng):
    from dace.libraries.linalg.nodes.inv import Inv

    sdfg = build("inv", Inv("inv"), {"ain": ((N, N), F), "aout": ((N, N), F)}, [("_ain", "ain")], [("_aout", "aout")])
    A = rng.random((4, 4)) + 4 * np.eye(4)
    buffers, src = run(sdfg, "inv", {"ain": A.copy(), "aout": np.zeros((4, 4))}, {"N": 4})
    assert "np.linalg.inv" in src
    np.testing.assert_allclose(buffers["aout"], np.linalg.inv(A), rtol=1e-12)


# FFT / IFFT (DaCe's forward DFT is unnormalized; its inverse omits the 1/N)

C128 = dc.complex128


def test_fft(rng):
    from dace.libraries.fft.nodes.fft import FFT

    sdfg = build("fft", FFT("fft"), {"x": ((N,), C128), "y": ((N,), C128)}, [("_inp", "x")], [("_out", "y")])
    x = rng.random(8) + 1j * rng.random(8)
    buffers, src = run(sdfg, "fft", {"x": x.copy(), "y": np.zeros(8, complex)}, {"N": 8})
    assert "np.fft.fft" in src
    np.testing.assert_allclose(buffers["y"], np.fft.fft(x), rtol=1e-12)


def test_ifft_omits_one_over_n(rng):
    """DaCe's IFFT is the raw inverse sum (no ``1/N``); numpy's ``ifft`` divides by N, so the match needs
    ``norm='forward'`` (== ``N * np.fft.ifft``)."""
    from dace.libraries.fft.nodes.fft import IFFT

    sdfg = build("ifft", IFFT("ifft"), {"x": ((N,), C128), "y": ((N,), C128)}, [("_inp", "x")], [("_out", "y")])
    x = rng.random(8) + 1j * rng.random(8)
    buffers, src = run(sdfg, "ifft", {"x": x.copy(), "y": np.zeros(8, complex)}, {"N": 8})
    assert "norm='forward'" in src
    np.testing.assert_allclose(buffers["y"], np.fft.ifft(x, norm="forward"), rtol=1e-12)


def test_fft_factor_normalization(rng):
    from dace.libraries.fft.nodes.fft import IFFT

    ifft = IFFT("ifft")
    ifft.factor = 0.125  # 1/N normalization folded into the coefficient -> matches numpy's plain ifft
    sdfg = build("ifftn", ifft, {"x": ((N,), C128), "y": ((N,), C128)}, [("_inp", "x")], [("_out", "y")])
    x = rng.random(8) + 1j * rng.random(8)
    buffers, _ = run(sdfg, "ifftn", {"x": x.copy(), "y": np.zeros(8, complex)}, {"N": 8})
    np.testing.assert_allclose(buffers["y"], np.fft.ifft(x), rtol=1e-12)


# standard: ArgReduce / Scan


@pytest.mark.parametrize("op, argfn, valfn", [("max", np.argmax, np.max), ("min", np.argmin, np.min)])
def test_argreduce(op, argfn, valfn, rng):
    from dace.libraries.standard.nodes.arg_reduce import ArgReduce

    sdfg = build(
        "ar",
        ArgReduce("ar", op=op),
        {"inp": ((N,), F), "val": ((1,), F), "idx": ((1,), dc.int64)},
        [("_in", "inp")],
        [("_out_val", "val"), ("_out_idx", "idx")],
    )
    inp = rng.random(7)
    buffers, _ = run(sdfg, "ar", {"inp": inp.copy(), "val": np.zeros(1), "idx": np.zeros(1, np.int64)}, {"N": 7})
    assert buffers["idx"][0] == argfn(inp)
    np.testing.assert_array_equal(buffers["val"][0], valfn(inp))


@pytest.mark.parametrize(
    "scanop, ref",
    [("SUM", np.cumsum), ("PRODUCT", np.cumprod), ("MAX", np.maximum.accumulate), ("MIN", np.minimum.accumulate)],
)
def test_scan_inclusive(scanop, ref, rng):
    from dace.libraries.standard.nodes.scan import Scan, ScanOp

    sdfg = build(
        "sc",
        Scan("sc", op=ScanOp[scanop]),
        {"si": ((N,), F), "so": ((N,), F)},
        [("_scan_in", "si")],
        [("_scan_out", "so")],
    )
    si = rng.random(6)
    buffers, _ = run(sdfg, "sc", {"si": si.copy(), "so": np.zeros(6)}, {"N": 6})
    np.testing.assert_array_equal(buffers["so"], ref(si))


def test_scan_exclusive_refused():
    from dace.libraries.standard.nodes.scan import Scan, ScanOp

    sc = Scan("sc", op=ScanOp.SUM)
    sc.exclusive = True
    sdfg = build("scx", sc, {"si": ((N,), F), "so": ((N,), F)}, [("_scan_in", "si")], [("_scan_out", "so")])
    with pytest.raises(UnsupportedNest, match="inclusive"):
        sdfg_to_numpy(sdfg, "scx")


def test_argreduce_over_the_absolute_value(rng):
    from dace.libraries.standard.nodes.arg_reduce import ArgReduce

    sdfg = build(
        "ara",
        ArgReduce("ara", op="max", transform="abs"),
        {"inp": ((N,), F), "val": ((1,), F), "idx": ((1,), dc.int64)},
        [("_in", "inp")],
        [("_out_val", "val"), ("_out_idx", "idx")],
    )
    inp = np.array([0.5, -3.0, 1.0, 2.0])
    buffers, _ = run(sdfg, "ara", {"inp": inp.copy(), "val": np.zeros(1), "idx": np.zeros(1, np.int64)}, {"N": 4})
    assert buffers["idx"][0] == 1 and buffers["val"][0] == 3.0


def test_argreduce_without_a_value_output_writes_the_index(rng):
    from dace.libraries.standard.nodes.arg_reduce import ArgReduce

    node = ArgReduce("ari", op="max")
    node.remove_out_connector("_out_val")
    sdfg = build("ari", node, {"inp": ((N,), F), "idx": ((1,), dc.int64)}, [("_in", "inp")], [("_out_idx", "idx")])
    inp = rng.random(7)
    buffers, src = run(sdfg, "ari", {"inp": inp.copy(), "idx": np.zeros(1, np.int64)}, {"N": 7})
    assert buffers["idx"][0] == np.argmax(inp)
    assert "np.max(" not in src


def test_scan_emits_every_chain(rng):
    from dace.libraries.standard.nodes.scan import Scan, ScanOp

    sdfg = build(
        "sc2",
        Scan("sc2", op=ScanOp.SUM, chains=2),
        {"a": ((N,), F), "b": ((N,), F), "oa": ((N,), F), "ob": ((N,), F)},
        [("_scan_in", "a"), ("_scan_in_1", "b")],
        [("_scan_out", "oa"), ("_scan_out_1", "ob")],
    )
    a, b = rng.random(5), rng.random(5)
    buffers, _ = run(sdfg, "sc2", {"a": a.copy(), "b": b.copy(), "oa": np.zeros(5), "ob": np.zeros(5)}, {"N": 5})
    np.testing.assert_array_equal(buffers["oa"], np.cumsum(a))
    np.testing.assert_array_equal(buffers["ob"], np.cumsum(b))


def build_on_row(name, node, in_conns, out_conn, out_shape):
    """``node`` reading row 2 of ``A`` (5x5) on ``in_conns[0]`` and all of ``v`` on any other input, into ``o``."""
    sdfg = dc.SDFG(name)
    sdfg.add_array("A", (5, 5), F)
    sdfg.add_array("v", (5,), F)
    sdfg.add_array("o", out_shape, F)
    st = sdfg.add_state()
    st.add_node(node)
    for i, conn in enumerate(in_conns):
        node.add_in_connector(conn, force=True)
        memlet = Memlet("A[2, 0:5]") if i == 0 else Memlet("v[0:5]")
        st.add_edge(st.add_read("A" if i == 0 else "v"), None, node, conn, memlet)
    node.add_out_connector(out_conn, force=True)
    st.add_edge(node, out_conn, st.add_write("o"), None, Memlet.from_array("o", sdfg.arrays["o"]))
    sdfg.validate()
    return sdfg


def row_nodes():
    from dace.libraries.blas.nodes.dot import Dot
    from dace.libraries.blas.nodes.einsum import Einsum
    from dace.libraries.standard.nodes.scan import Scan, ScanOp

    einsum = Einsum("es")
    einsum.einsum_str = "k,k->k"
    return {
        "scan": (Scan("sc", op=ScanOp.MAX), ["_scan_in"], "_scan_out", (5,), lambda a, v: np.maximum.accumulate(a)),
        "dot": (Dot("dot"), ["_x", "_y"], "_result", (1,), lambda a, v: np.array([a @ v])),
        "einsum": (einsum, ["a", "b"], "o", (5,), lambda a, v: a * v),
    }


@pytest.mark.parametrize("kind", ["scan", "dot", "einsum"])
def test_a_vector_node_reads_a_matrix_row_as_a_vector(kind, rng):
    """DaCe squeezes a vector node's operands, so ``A[2, 0:5]`` is a 5-vector, not a 1x5 matrix."""
    node, in_conns, out_conn, out_shape, reference = row_nodes()[kind]
    sdfg = build_on_row(f"row_{kind}", node, in_conns, out_conn, out_shape)
    A, v = np.array([[0.0] * 5, [0.0] * 5, [3.0, 1.0, 4.0, 1.0, 5.0], [0.0] * 5, [0.0] * 5]), rng.random(5)

    buffers, _ = run(sdfg, f"row_{kind}", {"A": A.copy(), "v": v.copy(), "o": np.zeros(out_shape)}, {})

    np.testing.assert_array_equal(buffers["o"], reference(A[2], v))


def test_dot_conjugate_conjugates_the_first_operand(rng):
    from dace.libraries.blas.nodes.dot import Dot

    sdfg = build(
        "dotc",
        Dot("dotc", conjugate=True),
        {"x": ((N,), C128), "y": ((N,), C128), "r": ((1,), C128)},
        [("_x", "x"), ("_y", "y")],
        [("_result", "r")],
    )
    x, y = rng.random(6) + 1j * rng.random(6), rng.random(6) + 1j * rng.random(6)
    buffers, _ = run(sdfg, "dotc", {"x": x.copy(), "y": y.copy(), "r": np.zeros(1, complex)}, {"N": 6})
    np.testing.assert_allclose(buffers["r"][0], np.vdot(x, y), rtol=1e-12, atol=0)


def test_fft_without_axes_transforms_every_axis(rng):
    from dace.libraries.fft.nodes.fft import FFT

    sdfg = build("fft2", FFT("fft2"), {"x": ((N, N), C128), "y": ((N, N), C128)}, [("_inp", "x")], [("_out", "y")])
    x = rng.random((4, 4)) + 1j * rng.random((4, 4))
    buffers, _ = run(sdfg, "fft2", {"x": x.copy(), "y": np.zeros((4, 4), complex)}, {"N": 4})
    np.testing.assert_allclose(buffers["y"], np.fft.fftn(x), rtol=1e-12, atol=1e-12)


def test_integer_sort(rng):
    from dace.libraries.sort.nodes.integer_sort import IntegerSort

    sdfg = build(
        "srt",
        IntegerSort("srt"),
        {"ki": ((N,), dc.int64), "ko": ((N,), dc.int64)},
        [("_keys_in", "ki")],
        [("_keys_out", "ko")],
    )
    ki = rng.integers(0, 1000, size=9).astype(np.int64)
    buffers, src = run(sdfg, "srt", {"ki": ki.copy(), "ko": np.zeros(9, np.int64)}, {"N": 9})
    assert "np.sort" in src
    np.testing.assert_array_equal(buffers["ko"], np.sort(ki))


# ScatterConflictCheck: TAGCOUNT duplicate count (0 iff a permutation)


def build_scatter_conflict_check(name):
    from dace.libraries.sort.nodes.scatter_conflict_check import ScatterConflictCheck

    return build(
        name,
        ScatterConflictCheck(name),
        {"idx": ((N,), dc.int64), "cnt": ((1,), dc.int64)},
        [("_idx_in", "idx")],
        [("_count_out", "cnt")],
    )


def test_scatter_conflict_check_permutation(rng):
    """A permutation has no duplicate values, so ``count == 0`` (the scatter is conflict-free)."""
    sdfg = build_scatter_conflict_check("sccp")
    idx = rng.permutation(9).astype(np.int64)
    buffers, src = run(sdfg, "sccp", {"idx": idx.copy(), "cnt": np.zeros(1, np.int64)}, {"N": 9})
    assert "np.full" in src  # TAGCOUNT ownership buffer, not a sort
    assert buffers["cnt"][0] == 0
    assert buffers["cnt"][0] == idx.shape[0] - len(np.unique(idx))


def test_scatter_conflict_check_duplicates():
    """With duplicates, ``count == N - #distinct`` -- matching the libnode's sort + adjacent-equal scan."""
    sdfg = build_scatter_conflict_check("sccd")
    idx = np.array([0, 2, 2, 5, 5, 5, 1, 9, 9], dtype=np.int64)  # 9 elems, 5 distinct -> 4 duplicates
    buffers, _ = run(sdfg, "sccd", {"idx": idx.copy(), "cnt": np.zeros(1, np.int64)}, {"N": 9})
    assert buffers["cnt"][0] == 4
    assert buffers["cnt"][0] == idx.shape[0] - len(np.unique(idx))


def desc_of_shape(shape):
    sdfg = dc.SDFG("d")
    sdfg.add_array("s", shape, dc.float64)
    return sdfg.arrays["s"]


def test_scalar_elem_indexes_every_dimension():
    # a (1, 1) buffer is also a scalar (total_size == 1); name[0] would select a sub-array
    assert scalar_elem("s", desc_of_shape([1])) == "s[0]"
    assert scalar_elem("s", desc_of_shape([1, 1])) == "s[0, 0]"
    assert scalar_elem("s", desc_of_shape([1, 1, 1])) == "s[0, 0, 0]"
