"""Emit + run the tsvc2.5 kernels that exercise the harder control-flow / reduction / recurrence paths.

Each kernel is extracted (baseline opt-mode), emitted to numpy, executed, and checked against a hand
reference. These pin three emitter/extraction fixes:

  * ``ext_break_find_first`` -- a ``break`` (BreakBlock) early-exit is emitted as ``break``;
  * ``cond_reduce_sym``      -- a size-1 buffer is READ as ``x[0]`` (not the bare ``(1,)`` array, which a
    NumPy-2 scalar assignment rejects), and its canonicalized WCR *copy* accumulates;
  * ``iv_multiplicative``    -- the loop index is pre-declared on the parent so nest extraction does not
    KeyError on an unregistered symbol.

tsvc2.5 lives in DaCe's ``performance_regression_jobs/tsvc_2_5_corpus.py``; skip if it is not reachable.
"""
import inspect

import numpy as np
import pytest

pytest.importorskip("optarena")

from nestforge import tsvc
from nestforge.extract import extract_nest_to_sdfg
from nestforge.emit_numpy import sdfg_to_numpy
from nestforge.strategies import get_strategy


def load(key: str):
    try:
        found = tsvc.iter_tsvc_kernels(only=[key], corpus="tsvc2_5")
    except Exception as exc:  # corpus script not shipped with this DaCe
        pytest.skip(f"tsvc2.5 corpus unavailable: {exc}")
    if not found:
        pytest.skip(f"{key} not in the tsvc2.5 corpus")
    return found[0]


def emit_and_call(key: str, sizes: dict, inputs: dict, opt_mode: str = "baseline"):
    """Extract + emit ``key``, allocate every buffer C-style from the emitted signature, run it, return
    the call dict (buffers hold the results in place)."""
    kernel = load(key)
    sdfg = tsvc.build_sdfg(kernel, opt_mode=opt_mode)
    refs = get_strategy("skip-taskloops")(sdfg)
    assert len(refs) == 1, f"{key}/{opt_mode}: expected one compute nest, got {len(refs)}"
    boundary = extract_nest_to_sdfg(refs[0][0], refs[0][1], name=kernel.key)
    src = sdfg_to_numpy(boundary.standalone_sdfg, kernel.key)
    ns = {"np": np}
    exec(src, ns)
    fn = ns[kernel.key]
    call = {}
    for name in inspect.signature(fn).parameters:
        if name in sizes:
            call[name] = sizes[name]
            continue
        desc = boundary.standalone_sdfg.arrays[name]
        shape = tuple(int(str(d)) if str(d).isdigit() else sizes["LEN_1D"] for d in desc.shape)
        dt = np.dtype(desc.dtype.type)
        call[name] = inputs[name].astype(dt) if name in inputs else np.zeros(shape, dt)
    fn(**call)
    return call, src


def test_ext_break_find_first_emits_break_and_stops():
    """BreakBlock -> ``break``: a[i] += b[i]*c[i] until the first d[i] < 0."""
    n = 40
    rng = np.random.default_rng(3)
    a, b, c = rng.random(n), rng.random(n), rng.random(n)
    d = rng.random(n)
    d[17] = -1.0  # force the break at a known, non-trivial index
    call, src = emit_and_call("ext_break_find_first", dict(LEN_1D=n), dict(a=a.copy(), b=b.copy(), c=c.copy(), d=d))
    assert "break" in src
    ref = a.copy()
    for i in range(n):
        if d[i] < 0.0:
            break
        ref[i] = ref[i] + b[i] * c[i]
    np.testing.assert_allclose(call["a"], ref, rtol=1e-12, atol=1e-12)
    assert int(call["__sym_out_i"][0]) == 17  # stopped exactly at the negative element


@pytest.mark.parametrize("opt_mode", ["baseline", "canonicalize"])
def test_cond_reduce_sym_scalar_read_and_wcr(opt_mode):
    """Size-1 buffer read as ``x[0]`` (baseline) and WCR copy accumulates (canonicalize): out = sum of
    a[i] where a[i] > K. Both opt-modes must reduce correctly."""
    n, K = 96, 0.5
    a = np.random.default_rng(0).random(n)
    call, _ = emit_and_call("cond_reduce_sym", dict(LEN_1D=n, K=K), dict(a=a.copy()), opt_mode=opt_mode)
    acc = next(call[k] for k in call if k in ("out", "_priv_out") or k.endswith("_out"))
    np.testing.assert_allclose(float(np.ravel(acc)[0]), a[a > K].sum(), rtol=1e-12, atol=1e-12)


def test_iv_multiplicative_extracts_and_computes_recurrence():
    """Loop index pre-declared so extraction does not KeyError: s <- s * 0.99 for LEN_1D iterations."""
    n = 50
    s0 = 3.5
    call, _ = emit_and_call("iv_multiplicative", dict(LEN_1D=n), dict(s=np.array([s0])))
    np.testing.assert_allclose(float(call["s"][0]), s0 * (0.99**n), rtol=1e-12, atol=1e-12)
