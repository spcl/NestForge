# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A bool scalar input that canonicalization promotes to a symbol (vexx_k's ``gamma_only``) is bound like an integer
or a float one, and the built program receives its value (spcl/NestForge#79)."""

import ctypes

import dace
import numpy as np
from helpers import build_sdfg

from nestforge.build.toolchain import symbol_argument
from nestforge.server import symbol_inputs

#: Elements of the test vector.
N = 4


def add_flag() -> dace.SDFG:
    """``b[i] = a[i] + flag`` with ``flag`` a bool symbol, as a promoted bool input is."""
    sdfg = dace.SDFG("bool_symbol")
    sdfg.add_symbol("flag", dace.bool_)
    sdfg.add_array("a", ["N"], dace.float64)
    sdfg.add_array("b", ["N"], dace.float64)
    state = sdfg.add_state()
    state.add_mapped_tasklet(
        "shift",
        {"i": "0:N"},
        {"x": dace.Memlet("a[i]")},
        "y = x + flag",
        {"y": dace.Memlet("b[i]")},
        external_edges=True,
    )
    return sdfg


def test_a_bool_scalar_input_that_is_a_symbol_binds_it():
    inputs = {"flag": np.asarray(True), "a": np.zeros(N), "N": np.asarray(N)}

    assert symbol_inputs(add_flag(), inputs) == {"flag": True, "N": N}


def test_a_bool_symbol_is_passed_as_a_c_bool():
    true, false = symbol_argument(ctypes.c_bool, True), symbol_argument(ctypes.c_bool, False)

    assert isinstance(true, ctypes.c_bool) and true.value is True
    assert isinstance(false, ctypes.c_bool) and false.value is False


def test_the_built_program_receives_the_bool_symbol(tmp_path):
    built = build_sdfg(add_flag(), tmp_path)
    a = np.arange(N, dtype=np.float64)
    b = np.zeros(N)

    built.run({"a": a, "b": b}, {"N": N, "flag": True})

    np.testing.assert_array_equal(b, a + 1)
