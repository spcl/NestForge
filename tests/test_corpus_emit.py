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
        # loop-shaped scratch is widened to its loop bound by the emitter; size any unresolved (loop)
        # symbol at the largest kernel size so the caller-allocated buffer is at least that big.
        full_env = dict(env)
        for d in desc.shape:
            for s in symbolic.pystr_to_symbolic(str(d)).free_symbols:
                full_env.setdefault(s, max(sizes.values()))
        shape = tuple(int(symbolic.evaluate(d, full_env)) for d in desc.shape)
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


def test_mandelbrot_nested_sdfg_in_map_emits_and_computes():
    """mandelbrot's per-pixel escape is a ``np.where`` -> a nested SDFG with inner control flow inside
    a 2-D map. It emits only after ``ExpandNestedSDFGInputs`` widens the nest to full arrays, so the
    masked ``if I[j,k]: Z[j,k] = Z[j,k]**2 + C[j,k]`` writes the outer buffer in place."""
    pytest.importorskip("dace.transformation.interstate.expand_nested_sdfg_inputs")
    XN, YN = 20, 16
    scal = dict(xmin=-2.0, xmax=0.5, ymin=-1.25, ymax=1.25, maxiter=25, horizon=2.0)
    sdfg = _kernels()["hpc/map_reduce/mandelbrot1/mandelbrot1"].to_sdfg(simplify=True)
    src = sdfg_to_numpy(sdfg, "mandelbrot")
    ns = {"np": np}
    exec(src, ns)
    env = {symbolic.symbol("XN"): XN, symbolic.symbol("YN"): YN}
    call = {}
    for name in inspect.signature(ns["mandelbrot"]).parameters:
        if name in ("XN", "YN"):
            call[name] = {"XN": XN, "YN": YN}[name]
        elif name in scal:
            d = sdfg.arrays.get(name)
            call[name] = np.array([scal[name]], np.dtype(d.dtype.type)) if d is not None else scal[name]
        else:
            d = sdfg.arrays[name]
            shape = tuple(int(symbolic.evaluate(x, env)) for x in d.shape)
            call[name] = np.zeros(shape, np.dtype(d.dtype.type))
    ns["mandelbrot"](**call)

    X = scal["xmin"] + np.arange(XN) * ((scal["xmax"] - scal["xmin"]) / (XN - 1))
    Y = scal["ymin"] + np.arange(YN) * ((scal["ymax"] - scal["ymin"]) / (YN - 1))
    C = X[None, :] + Y[:, None] * 1j
    Nc = np.zeros(C.shape, np.int64)
    Z = np.zeros(C.shape, np.complex128)
    for n in range(scal["maxiter"]):
        I = np.less(np.absolute(Z), scal["horizon"])
        Nc[I] = n
        for j in range(YN):
            for k in range(XN):
                if I[j, k]:
                    Z[j, k] = Z[j, k]**2 + C[j, k]
    Nc[Nc == scal["maxiter"] - 1] = 0
    np.testing.assert_array_equal(call["N_out"], Nc)
    np.testing.assert_array_equal(call["Z_out"], Z)  # bit-exact


def test_emission_does_not_mutate_caller_sdfg():
    """``sdfg_to_numpy`` must be read-only: widening nested SDFGs runs on a copy, so a caller that
    inspects or compiles the same SDFG afterwards (e.g. the DaCe-reference competitor) is unaffected."""
    pytest.importorskip("dace.transformation.interstate.expand_nested_sdfg_inputs")
    from dace.sdfg import nodes
    sdfg = _kernels()["hpc/map_reduce/mandelbrot1/mandelbrot1"].to_sdfg(simplify=True)

    def nsdfg_in_subsets(g):
        return {e.dst_conn: str(e.data.subset)
                for st in g.all_states() for n in st.nodes() if isinstance(n, nodes.NestedSDFG)
                for e in st.in_edges(n)}

    before = nsdfg_in_subsets(sdfg)
    sdfg_to_numpy(sdfg, "mandelbrot")
    assert nsdfg_in_subsets(sdfg) == before  # connectors/subsets unchanged -> no in-place widening


def test_nbody_nested_where_emits_and_computes():
    """nbody's ``np.power(inv_r3, -1.5, out=inv_r3, where=I)`` is a masked nested SDFG in a 2-D map;
    it emits correctly once ExpandNestedSDFGInputs offsets the multi-dim mask condition fully."""
    pytest.importorskip("dace.transformation.interstate.expand_nested_sdfg_inputs")
    from nestforge.emit_numpy import UnsupportedNest
    N, Nt = 6, 4
    rng = np.random.default_rng(0)
    mass, pos, vel = rng.random(N) + 0.5, rng.random((N, 3)), rng.random((N, 3))
    dt, G, soft = 0.01, 1.0, 0.1
    try:
        call, _ = _alloc_run("hpc/n_body_methods/nbody/nbody", "nbody", dict(N=N, Nt=Nt),
                             dict(mass=mass, pos=pos, vel=vel, dt=np.array([dt]), G=np.array([G]),
                                  softening=np.array([soft])))
    except UnsupportedNest:
        pytest.skip("ExpandNestedSDFGInputs multi-dim condition offset not fixed in this DaCe")

    def getAcc(pos, mass, G, softening):
        x, y, z = pos[:, 0:1], pos[:, 1:2], pos[:, 2:3]
        dx = np.add.outer(-x, x).reshape(N, N)
        dy = np.add.outer(-y, y).reshape(N, N)
        dz = np.add.outer(-z, z).reshape(N, N)
        inv_r3 = dx**2 + dy**2 + dz**2 + softening**2
        np.power(inv_r3, -1.5, out=inv_r3, where=inv_r3 > 0)
        a = np.zeros((N, 3))
        a[:, 0], a[:, 1], a[:, 2] = G * (dx * inv_r3) @ mass, G * (dy * inv_r3) @ mass, G * (dz * inv_r3) @ mass
        return a

    def getEnergy(pos, vel, mass, G):
        KE = 0.5 * np.sum(np.reshape(mass, (N, 1)) * vel**2)
        x, y, z = pos[:, 0:1], pos[:, 1:2], pos[:, 2:3]
        dx = np.add.outer(-x, x).reshape(N, N)
        dy = np.add.outer(-y, y).reshape(N, N)
        dz = np.add.outer(-z, z).reshape(N, N)
        inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
        np.divide(1.0, inv_r, out=inv_r, where=inv_r > 0)
        tmp = -np.multiply.outer(mass, mass) * inv_r
        PE = sum(tmp[j, k] for j in range(N) for k in range(j + 1, N)) * G
        return KE, PE

    vel = vel - np.mean(np.reshape(mass, (N, 1)) * vel, axis=0) / np.mean(mass)
    acc = getAcc(pos, mass, G, soft)
    KE = np.zeros(Nt + 1)
    PE = np.zeros(Nt + 1)
    KE[0], PE[0] = getEnergy(pos, vel, mass, G)
    for i in range(Nt):
        vel += acc * dt / 2.0
        pos += vel * dt
        acc = getAcc(pos, mass, G, soft)
        vel += acc * dt / 2.0
        KE[i + 1], PE[i + 1] = getEnergy(pos, vel, mass, G)
    got = [call[k] for k in sorted(k for k in call if k.startswith("__return"))]
    for ref in (KE, PE):  # KE and PE are the two returns (order-independent match)
        assert any(np.allclose(g, ref) for g in got), f"no return matches ref {ref}"


def test_azimint_naive_wcr_reduction_emits_and_computes():
    """azimint_naive is a masked mean per radial bin: ``if r1<=radius<r2: tmp += data[j]`` -- a WCR
    accumulation inside a nested SDFG. Exercises WCR augmented-assignment plus the inner/outer size-1
    descriptor reconciliation (a nested scalar accumulator read back as a size-1 array)."""
    pytest.importorskip("dace.transformation.interstate.expand_nested_sdfg_inputs")
    from nestforge.emit_numpy import UnsupportedNest
    N, npt = 150, 8
    rng = np.random.default_rng(0)
    data, radius = rng.random(N), rng.random(N)
    try:
        call, _ = _alloc_run("hpc/map_reduce/azimint_naive/azimint_naive", "azimint_naive",
                             dict(N=N, npt=npt), dict(data=data, radius=radius))
    except UnsupportedNest:
        pytest.skip("nested-SDFG emission unavailable in this DaCe")

    rmax = radius.max()
    res = np.zeros(npt)
    for i in range(npt):
        r1, r2 = rmax * i / npt, rmax * (i + 1) / npt
        mask = np.logical_and(r1 <= radius, radius < r2)
        tmp, on = 0.0, 0
        for j in range(N):
            if mask[j]:
                tmp += data[j]
                on += 1
        res[i] = tmp / on
    got = call[[k for k in call if k.startswith("__return")][0]]
    np.testing.assert_allclose(got, res, equal_nan=True)


def test_trisolv_loop_shaped_scratch_maxsized_and_computes():
    """trisolv (lower-triangular solve) stages a growing slice into a scratch buffer shaped ``[i]``.
    The emitter widens that buffer to its loop bound ``N`` and addresses it with ``0:i`` slices, so it
    can be pre-allocated C-style. Validates the max-size loop-scratch path."""
    N = 12
    rng = np.random.default_rng(0)
    L = np.tril(rng.random((N, N))) + N * np.eye(N)
    b = rng.random(N)
    call, src = _alloc_run("hpc/dense_linear_algebra/trisolv/trisolv", "trisolv", dict(N=N),
                           dict(L=L, b=b))
    assert "np.empty" not in src  # still C-style: no in-kernel allocation
    x = call.get("x", call.get("__return"))
    np.testing.assert_allclose(x, np.linalg.solve(L, b))


def test_lu_loop_shaped_scratch_maxsized_and_computes():
    """lu decomposition stages ``A[i,:j]`` / ``A[:i,j]`` dot-product operands into ``[j]``/``[i]``
    scratch buffers; the emitter max-sizes them to ``N``. Validates against an in-place Doolittle LU."""
    N = 10
    rng = np.random.default_rng(1)
    A0 = rng.random((N, N)) + N * np.eye(N)
    call, src = _alloc_run("hpc/dense_linear_algebra/lu/lu", "lu", dict(N=N), dict(A=A0.copy()))
    A = call.get("A", call.get("__return"))

    ref = A0.copy()
    for i in range(N):
        for j in range(i):
            ref[i, j] = (ref[i, j] - ref[i, :j] @ ref[:j, j]) / ref[j, j]
        for j in range(i, N):
            ref[i, j] = ref[i, j] - ref[i, :i] @ ref[:i, j]
    np.testing.assert_allclose(A, ref)


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
