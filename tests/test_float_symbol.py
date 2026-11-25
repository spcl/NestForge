# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A float scalar input that canonicalization promotes to a symbol (cp2k_density_matrix_trs4's ``threshold``) is
bound with its value: ``symbol_inputs`` keeps it, and the built program receives it as a double, not truncated to an
integer (spcl/tactics#61)."""

import ctypes
from pathlib import Path

import dace
import numpy as np
from helpers import build_sdfg

from tactics.build.toolchain import bind_argument, symbol_argument
from tactics.server import symbol_inputs

#: A threshold below one: an integer cast would make it zero.
THRESHOLD = 0.5
#: Elements of the test vector.
N = 4


def scale_by_symbol() -> dace.SDFG:
    """``b[i] = a[i] * thr`` with ``thr`` a float64 symbol, as a promoted scalar input is."""
    sdfg = dace.SDFG("float_symbol")
    sdfg.add_symbol("thr", dace.float64)
    sdfg.add_array("a", ["N"], dace.float64)
    sdfg.add_array("b", ["N"], dace.float64)
    state = sdfg.add_state()
    state.add_mapped_tasklet(
        "scale",
        {"i": "0:N"},
        {"x": dace.Memlet("a[i]")},
        "y = x * thr",
        {"y": dace.Memlet("b[i]")},
        external_edges=True,
    )
    return sdfg


def test_a_float_scalar_input_that_is_a_symbol_binds_it() -> None:
    sdfg = scale_by_symbol()
    inputs = {"thr": np.asarray(THRESHOLD), "a": np.zeros(N), "N": np.asarray(N)}

    sizes = symbol_inputs(sdfg, inputs)

    assert sizes == {"thr": THRESHOLD, "N": N}
    assert type(sizes["thr"]) is float and type(sizes["N"]) is int


def test_a_symbol_is_passed_in_its_own_c_type() -> None:
    as_double, as_int = symbol_argument(ctypes.c_double, THRESHOLD), symbol_argument(ctypes.c_int64, 3)
    bound = bind_argument("thr", ctypes.c_double, {}, {"thr": THRESHOLD})
    assert isinstance(as_double, ctypes.c_double) and as_double.value == THRESHOLD
    assert isinstance(as_int, ctypes.c_int64) and as_int.value == 3
    assert isinstance(bound, ctypes.c_double) and bound.value == THRESHOLD


def test_the_built_program_receives_the_float_symbol(tmp_path: Path) -> None:
    built = build_sdfg(scale_by_symbol(), tmp_path)
    params = {p.name: p.ctype for p in built.prog_params}
    a = np.arange(N, dtype=np.float64)
    b = np.zeros(N)

    built.run({"a": a, "b": b}, {"N": N, "thr": THRESHOLD})

    assert params["thr"] is ctypes.c_double
    np.testing.assert_array_equal(b, a * THRESHOLD)
