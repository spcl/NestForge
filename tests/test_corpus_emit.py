"""Emit numpy for real optarena dace kernels and check the library-node ops compute correctly.

Emission is C-style: the kernel allocates nothing, so the caller pre-allocates every buffer -- inputs,
outputs, the DaCe ``__return`` value, and scratch transients -- and reads the results back out of the
in-place buffers. ``_alloc_run`` does exactly that, driven by the emitted function's own signature.
"""
import inspect

import numpy as np
import pytest

pytest.importorskip("optarena")

from dace import symbolic

from nestforge.corpus import dace_kernel_names, iter_dace_kernels
from nestforge.emit_numpy import sdfg_to_numpy


def _kernels():
    return {k.short_name: k for k in iter_dace_kernels()}


def _alloc_run(short, fn_name, sizes, inputs, seed=0):
    """Emit ``short``, allocate every buffer parameter C-style, run it, return the buffer dict."""
    sdfg = _kernels()[short].to_sdfg(simplify=True)
    src = sdfg_to_numpy(sdfg, fn_name)
    ns = {"np": np}
    exec(src, ns)
    env = {symbolic.symbol(k): v for k, v in sizes.items()}
    call = {}
    for name in inspect.signature(ns[fn_name]).parameters:
        if name in sizes:
            call[name] = sizes[name]
            continue
        desc = sdfg.arrays[name]
        shape = tuple(int(symbolic.evaluate(d, env)) for d in desc.shape)
        dt = np.dtype(desc.dtype.type)
        call[name] = inputs[name].astype(dt) if name in inputs else np.zeros(shape, dt)
    ns[fn_name](**call)
    return call, src


def test_corpus_exposes_dace_kernels():
    assert len(dace_kernel_names("hpc")) == 50
    assert len(dace_kernel_names("ml")) == 5


def test_gemm_matmul_emits_and_computes():
    rng = np.random.default_rng(0)
    NI, NJ, NK = 8, 6, 5
    A, B, C = rng.random((NI, NK)), rng.random((NK, NJ)), rng.random((NI, NJ))
    alpha, beta = np.array([1.5]), np.array([2.0])
    ref = alpha[0] * A @ B + beta[0] * C
    call, src = _alloc_run("hpc/dense_linear_algebra/gemm/gemm", "gemm", dict(NI=NI, NJ=NJ, NK=NK),
                           dict(A=A, B=B, C=C.copy(), alpha=alpha, beta=beta))
    assert "@" in src and "np.empty" not in src  # MatMul -> numpy matmul, no internal allocation
    np.testing.assert_allclose(call["C"], ref)


def test_atax_return_value_is_inplace_buffer():
    rng = np.random.default_rng(1)
    M, N = 7, 9
    A, x = rng.random((M, N)), rng.random(N)
    call, src = _alloc_run("hpc/dense_linear_algebra/atax/atax", "atax", dict(M=M, N=N), dict(A=A, x=x))
    assert "return " not in src and "__return" in src.splitlines()[0]  # C-style: __return is a param
    np.testing.assert_allclose(call["__return"], (A @ x) @ A)


def test_nussinov_conditionalblock_emits_and_computes():
    """nussinov exercises the full control-flow path at once: ``ConditionalBlock`` (if/elif/else),
    access-node scalar copies, an inter-state indirect index, and ``dace.<cast>`` normalization."""
    N = 20
    rng = np.random.default_rng(0)
    seq = rng.integers(0, 4, size=N).astype(np.int32)
    call, src = _alloc_run("hpc/dynamic_programming/nussinov/nussinov", "nussinov", dict(N=N), dict(seq=seq))
    assert "if " in src and "else:" in src  # the guards lower through ConditionalBlock

    def match(b1, b2):
        return 1 if b1 + b2 == 3 else 0

    t = np.zeros((N, N), np.int32)
    for i in range(N - 1, -1, -1):
        for j in range(i + 1, N):
            if j - 1 >= 0:
                t[i, j] = np.maximum(t[i, j], t[i, j - 1])
            if i + 1 < N:
                t[i, j] = np.maximum(t[i, j], t[i + 1, j])
            if j - 1 >= 0 and i + 1 < N:
                if i < j - 1:
                    t[i, j] = np.maximum(t[i, j], t[i + 1, j - 1] + match(seq[i], seq[j]))
                else:
                    t[i, j] = np.maximum(t[i, j], t[i + 1, j - 1])
            for k in range(i + 1, j):
                t[i, j] = np.maximum(t[i, j], t[i, k] + t[k + 1, j])
    np.testing.assert_array_equal(call["__return"], t)  # integer DP -> bit-exact


def test_contour_integral_two_returns_solve_and_indirect_negate():
    """contour_integral returns two buffers and mixes ``np.linalg.solve``, a conditional negate, and
    a ``dace.complex128`` power cast -- a good end-to-end check of the complex + multi-output path."""
    NR, NM, slab = 5, 3, 2
    rng = np.random.default_rng(1)
    crand = lambda shape: (rng.random(shape) + 1j * rng.random(shape)).astype(np.complex128)
    Ham, int_pts, Y = crand((slab + 1, NR, NR)), crand((32, )), crand((NR, NM))
    call, src = _alloc_run("hpc/dense_linear_algebra/contour_integral/contour_integral", "contour_integral",
                           dict(NR=NR, NM=NM, slab_per_bc=slab), dict(Ham=Ham, int_pts=int_pts, Y=Y))
    assert "np.complex128(" in src and "dace." not in src  # casts normalized to numpy

    P0 = np.zeros((NR, NM), np.complex128)
    P1 = np.zeros((NR, NM), np.complex128)
    for idx in range(32):
        z = int_pts[idx]
        Tz = np.zeros((NR, NR), np.complex128)
        for n in range(slab + 1):
            Tz += np.power(z, slab / 2 - n) * Ham[n]
        X = np.linalg.solve(Tz, Y)
        if np.absolute(z) < 1.0:
            X = -X
        P0 += X
        P1 += z * X
    got = [call[k] for k in sorted(k for k in call if k.startswith("__return"))]
    np.testing.assert_allclose(got[0], P0)
    np.testing.assert_allclose(got[1], P1)


def test_jacobi_1d_loopregion_emits_and_computes():
    rng = np.random.default_rng(2)
    N, T = 32, 20
    A, B = rng.random(N), rng.random(N)
    Ar, Br = A.copy(), B.copy()
    call, src = _alloc_run("hpc/structured_grids/jacobi_1d/jacobi_1d", "jacobi_1d", dict(N=N, TSTEPS=T),
                           dict(A=A.copy(), B=B.copy()))
    assert "while" in src  # the TSTEPS time loop (a LoopRegion)
    for _ in range(1, T):
        Br[1:-1] = 0.33333 * (Ar[:-2] + Ar[1:-1] + Ar[2:])
        Ar[1:-1] = 0.33333 * (Br[:-2] + Br[1:-1] + Br[2:])
    np.testing.assert_array_equal(call["A"], Ar)
    np.testing.assert_array_equal(call["B"], Br)
