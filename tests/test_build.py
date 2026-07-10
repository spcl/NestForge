"""nest-forge owns the DaCe build (BUILD.md): generate DaCe's C++, compile+link it ourselves with a
chosen compiler+flags, and call it via ctypes with manual init/program/exit -- not ``dace.compile``.

These tests build real corpus nests through :mod:`nestforge.build` and check the owned-built kernel
matches the numpy oracle, exercising: the ``generate_program_folder`` source-tree layout + our own
compile, the ``__dace_init``/``__program``/``__dace_exit`` call sequence, and per-parameter ctype
marshaling (a size symbol is ``int`` or ``int64_t`` per its DaCe dtype; a Scalar passes by value).
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("optarena")
pytestmark = pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not on PATH")

from nestforge.corpus import iter_dace_kernels
from nestforge.strategies import get_strategy
from nestforge.extract import extract_nest_to_sdfg
from nestforge.translate import prepare
from nestforge.arena import make_inputs, run_oracle
from nestforge.build import build_sdfg, dace_runtime_include


def _kernels():
    return {k.short_name: k for k in iter_dace_kernels()}


def _first_nest(short):
    sdfg = _kernels()[short].to_sdfg(simplify=True)
    parent, node = get_strategy("skip-taskloops")(sdfg)[0]
    return extract_nest_to_sdfg(parent, node, name="nest")


def _owned_build_matches_oracle(short, size=48):
    boundary = _first_nest(short)
    shape_syms = {s for s in boundary.symbols
                  if any(s in str(d.shape) for d in boundary.standalone_sdfg.arrays.values())}
    sizes = {s: (size if s in shape_syms else 0) for s in boundary.symbols}
    inputs = make_inputs(boundary, sizes, seed=0)
    prep = prepare(boundary, "k", Path(tempfile.mkdtemp()))
    oracle = run_oracle(prep, boundary, inputs, sizes)

    built = build_sdfg(boundary.standalone_sdfg, Path(tempfile.mkdtemp(prefix="nf_build_")))
    buf = {k: v.copy() for k, v in inputs.items()}
    built.run(buf, sizes)  # init -> program -> exit
    for o in oracle:
        np.testing.assert_allclose(buf[o], oracle[o], rtol=1e-9, atol=1e-9, equal_nan=True)


def test_dace_runtime_include_exists():
    assert (dace_runtime_include() / "dace" / "dace.h").exists()


def test_owned_build_gemm_matches_oracle():
    """gemm: int64_t size symbols + a Scalar (alpha/beta) passed by value through the owned build."""
    _owned_build_matches_oracle("hpc/dense_linear_algebra/gemm/gemm")


def test_owned_build_jacobi_matches_oracle():
    """jacobi_1d: an ``int`` (not int64_t) size symbol -- guards the per-parameter ctype marshaling."""
    _owned_build_matches_oracle("hpc/structured_grids/jacobi_1d/jacobi_1d")


def test_owned_build_reusable_handle_program():
    """After one init, __program can be called repeatedly in place (the timing path) on one handle."""
    boundary = _first_nest("hpc/dense_linear_algebra/gemm/gemm")
    shape_syms = {s for s in boundary.symbols
                  if any(s in str(d.shape) for d in boundary.standalone_sdfg.arrays.values())}
    sizes = {s: (32 if s in shape_syms else 0) for s in boundary.symbols}
    inputs = make_inputs(boundary, sizes, seed=1)
    built = build_sdfg(boundary.standalone_sdfg, Path(tempfile.mkdtemp(prefix="nf_build_")))
    buf = {k: v.copy() for k, v in inputs.items()}
    built._init(sizes)
    try:
        for _ in range(5):
            built.program(buf, sizes)  # repeated in-place calls on the same state handle
    finally:
        built.close()
