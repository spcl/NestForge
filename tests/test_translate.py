# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""NumPy and C emission of an extracted nest, run and compared with NumPy."""

import subprocess

import numpy as np
import dace

from nestforge.phases.scopes import parallel_top_level_maps
from nestforge.ir.extract import extract_nest_to_sdfg
from nestforge.ir.emit_numpy import load_emitted, nest_to_numpy
from nestforge.build.arena import call_native
from nestforge.corpus.translate import prepare, emit_sources

from helpers import c_argtypes, signature_order

N = dace.symbol("N")


@dace.program
def vadd(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + B[i]


def boundary():
    sdfg = vadd.to_sdfg(simplify=True)
    psdfg, node = parallel_top_level_maps(sdfg)[0]
    return extract_nest_to_sdfg(psdfg, node, name="vadd_nest")


def test_numpy_emit_runs():
    b = boundary()
    src = nest_to_numpy(b, fn_name="vadd")
    mod = load_emitted(src, "vadd")
    A = np.random.default_rng(0).random(32)
    B = np.random.default_rng(1).random(32)
    C = np.zeros(32)
    mod.vadd(A=A, B=B, C=C, N=32)
    np.testing.assert_array_equal(C, A + B)


def test_the_translated_c_kernel_computes_what_numpy_computes(tmp_path):
    b = boundary()
    prep = prepare(b, "vadd", tmp_path / "kern")
    assert prep.numpy_path.exists() and prep.yaml_path.exists()
    srcs = emit_sources(prep, tmp_path / "gen", target="c")
    c_files = [p for p in srcs if p.suffix == ".c"]
    assert c_files, f"no C emitted; got {srcs}"
    text = c_files[0].read_text()
    assert "double *restrict A" in text and "double *restrict C" in text and "int64_t N" in text
    so = tmp_path / "libvadd.so"
    subprocess.run(["gcc", "-O2", "-fPIC", "-shared", str(c_files[0]), "-o", str(so)], check=True)
    order = signature_order(text, "vadd_fp64")
    rng = np.random.default_rng(0)
    inputs = {"A": rng.random(32), "B": rng.random(32), "C": np.zeros(32)}

    outs, _ = call_native(so, "vadd_fp64", order, c_argtypes(order, b), b, inputs, {"N": 32}, reps=1)

    assert outs is not None
    np.testing.assert_array_equal(outs["C"], inputs["A"] + inputs["B"])


@dace.program
def gather_two_map(A: dace.float64[N], idx: dace.int64[N], C: dace.float64[N]):
    T = np.empty_like(A)
    for k in dace.map[0:N]:
        T[k] = A[idx[k]]
    for k in dace.map[0:N]:
        C[k] = T[k] * 2.0


def test_a_fused_maps_scalar_transient_is_spelled_the_same_inside_and_out():
    """MapFusion's size-1 intermediate is transient outside the nested SDFG and not inside, which decides its
    spelling; run, not grepped, since what matters is that the value survives."""
    from nestforge.phases.normalize import Targets, normalize
    from nestforge.phases.schedule import full_fusion
    from nestforge.phases.scopes import lower_nests_to_external_call

    sdfg = gather_two_map.to_sdfg(simplify=True)
    full_fusion(normalize(sdfg, Targets()), Targets())
    calls = lower_nests_to_external_call(sdfg)
    assert calls, "nothing lowered; the fixture no longer produces an offloadable nest"
    _, b = calls[0]
    mod = load_emitted(nest_to_numpy(b, fn_name="fused"), "fused")

    rng = np.random.default_rng(0)
    A = rng.random(32)
    idx = rng.permutation(32).astype(np.int64)
    C = np.zeros(32)
    mod.fused(A=A, idx=idx, C=C, N=32)
    np.testing.assert_array_equal(C, A[idx] * 2.0)


def test_the_standalone_preamble_is_the_live_helpers():
    """An emitted standalone kernel must compute what the validated in-process one computes. The preamble
    is generated from int_floor/int_ceil rather than hand-copied, so an edit to either cannot leave the
    emitted text behind -- assert the property, not the generation trick."""
    from nestforge.ir import emit_numpy

    namespace = {}
    exec(emit_numpy.STANDALONE_PREAMBLE, namespace)  # noqa: S102 -- the point is that it is runnable
    for a, b in ((7, 2), (-7, 2), (7, -2), (-7, -2), (8, 4), (0, 3)):
        assert namespace["int_floor"](a, b) == emit_numpy.int_floor(a, b), (a, b)
        assert namespace["int_ceil"](a, b) == emit_numpy.int_ceil(a, b), (a, b)
