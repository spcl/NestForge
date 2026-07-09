"""Emit numpy for real optarena dace kernels and check the library-node ops compute correctly."""
import numpy as np
import pytest

pytest.importorskip("optarena")

from nestforge.corpus import dace_kernel_names, iter_dace_kernels
from nestforge.emit_numpy import sdfg_to_numpy


def _kernels():
    return {k.short_name: k for k in iter_dace_kernels()}


def _build(short, fn_name):
    src = sdfg_to_numpy(_kernels()[short].to_sdfg(simplify=True), fn_name)
    namespace = {"np": np}
    exec(src, namespace)
    return namespace[fn_name], src


def test_corpus_exposes_dace_kernels():
    assert len(dace_kernel_names("hpc")) == 50
    assert len(dace_kernel_names("ml")) == 5


def test_gemm_matmul_emits_and_computes():
    gemm, src = _build("hpc/dense_linear_algebra/gemm/gemm", "gemm")
    assert "@" in src  # MatMul -> numpy matmul
    rng = np.random.default_rng(0)
    NI, NJ, NK = 8, 6, 5
    A, B, C = rng.random((NI, NK)), rng.random((NK, NJ)), rng.random((NI, NJ))
    alpha, beta = np.array([1.5]), np.array([2.0])
    ref = alpha[0] * A @ B + beta[0] * C
    Cw = C.copy()
    gemm(A, B, Cw, NI, NJ, NK, alpha, beta)
    np.testing.assert_allclose(Cw, ref)


def test_atax_return_value_emits_and_computes():
    atax, src = _build("hpc/dense_linear_algebra/atax/atax", "atax")
    assert src.splitlines()[0] == "def atax(A, x, M, N):"  # __return dropped from signature
    assert "return" in src
    rng = np.random.default_rng(1)
    M, N = 7, 9
    A, x = rng.random((M, N)), rng.random(N)
    np.testing.assert_allclose(atax(A, x, M, N), (A @ x) @ A)


def test_jacobi_1d_loopregion_emits_and_computes():
    jacobi, src = _build("hpc/structured_grids/jacobi_1d/jacobi_1d", "jacobi_1d")
    assert "while" in src  # the TSTEPS time loop (a LoopRegion)
    rng = np.random.default_rng(2)
    N, T = 32, 20
    A, B = rng.random(N), rng.random(N)
    Ar, Br = A.copy(), B.copy()
    jacobi(A, B, N, T)
    for _ in range(1, T):
        Br[1:-1] = 0.33333 * (Ar[:-2] + Ar[1:-1] + Ar[2:])
        Ar[1:-1] = 0.33333 * (Br[:-2] + Br[1:-1] + Br[2:])
    np.testing.assert_array_equal(A, Ar)
    np.testing.assert_array_equal(B, Br)
