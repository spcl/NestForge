# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""preset_sizes: a preset's integer symbols, plus the integers a manifest fixes for every preset, which are symbols
of the program too and are bound nowhere else."""

import dace
import numpy as np
from helpers import corpus_kernel

from nestforge.corpus.bench import preset_sizes
from nestforge.server import symbol_inputs


def test_a_fixed_loop_bound_is_a_size():
    # aes_graupel's args fix kstart and ivstart (not returned by initialize); kstart bounds the level loops
    sizes = preset_sizes(corpus_kernel("aes_graupel"), "S")
    assert sizes["kstart"] == 10 and sizes["ivstart"] == 1
    assert {"nvec", "ke", "nsteps"} <= sizes.keys()


def test_a_manifest_constant_is_a_size():
    assert preset_sizes(corpus_kernel("cloudsc_sedimentation"), "S")["NSPEC"] == 3


def test_a_config_knob_is_not_a_size():
    assert "ptsphy" not in preset_sizes(corpus_kernel("cloudsc_monolith"), "M")


def test_an_integer_the_initializer_returns_binds_a_symbol_of_the_program():
    # TSVC s162's k: initialize() returns it, and the program reads it as a loop offset symbol
    sdfg = dace.SDFG("offset")
    sdfg.add_symbol("k", dace.int64)
    sdfg.add_array("a", ["N"], dace.float64)
    inputs = {"k": np.asarray(3), "a": np.zeros(4), "alpha": np.asarray(1.5), "N": np.asarray(4)}

    assert symbol_inputs(sdfg, inputs) == {"k": 3, "N": 4}
