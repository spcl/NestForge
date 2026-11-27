# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A scalar parameter the program takes as a symbol gets its value from the bench's 0-d input in the whole-program
run, as a size would."""

import dace
import numpy as np

from tactics.stages.performance import program_symbols

N = dace.symbol("N", dtype=dace.int64)


def scaled_by_symbol() -> dace.SDFG:
    """``y = x * dvol`` with ``dvol`` a symbol of the program, as canonicalization leaves ``ls3df_scf``'s."""
    sdfg = dace.SDFG("scaled_by_symbol")
    sdfg.add_symbol("dvol", dace.float64)
    sdfg.add_array("x", [N], dace.float64)
    sdfg.add_array("y", [N], dace.float64)
    state = sdfg.add_state()
    state.add_mapped_tasklet(
        "scale",
        {"i": "0:N"},
        {"a": dace.Memlet("x[i]")},
        "b = a * dvol",
        {"b": dace.Memlet("y[i]")},
        external_edges=True,
    )
    return sdfg


# Reproducer: the whole-program run raised KeyError on a scalar parameter canonicalized to a symbol (#121 follow-up, ls3df_scf dvol, warpx_esirkepov_deposition do_ionization)
def test_a_scalar_input_the_program_takes_as_a_symbol_reaches_the_run_as_a_symbol_value() -> None:
    sdfg = scaled_by_symbol()
    inputs = {"x": np.ones(4), "dvol": np.array(0.008), "unused": np.array(3)}
    symbols = program_symbols(sdfg, inputs, {"N": 4})
    assert symbols == {"N": 4, "dvol": 0.008}
    assert isinstance(symbols["dvol"], float)


def test_a_size_given_twice_keeps_the_size() -> None:
    sdfg = scaled_by_symbol()
    symbols = program_symbols(sdfg, {"N": np.array(9), "dvol": np.array(1.0)}, {"N": 4})
    assert symbols["N"] == 4
