# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""preset_sizes: a preset's integer symbols, plus the integers a manifest fixes for every preset, which are symbols
of the program too and are bound nowhere else."""

from pathlib import Path

import dace
import numpy as np
import pytest
from helpers import corpus_kernel

from tactics.corpus.bench import preset_sizes
from tactics.server import Setup, session_data, started_session, symbol_inputs


def test_a_fixed_loop_bound_is_a_size() -> None:
    # icon_aes_graupel's args fix kstart and ivstart (not returned by initialize); kstart bounds the level loops
    sizes = preset_sizes(corpus_kernel("icon_aes_graupel"), "S")
    assert sizes["kstart"] == 10 and sizes["ivstart"] == 1
    assert {"nvec", "ke", "nsteps"} <= sizes.keys()


def test_a_manifest_constant_is_a_size() -> None:
    assert preset_sizes(corpus_kernel("cloudsc_sedimentation"), "S")["NSPEC"] == 3


def test_a_config_knob_is_not_a_size() -> None:
    assert "ptsphy" not in preset_sizes(corpus_kernel("cloudsc_monolith"), "M")


def test_an_integer_the_initializer_returns_binds_a_symbol_of_the_program() -> None:
    # TSVC s162's k: initialize() returns it, and the program reads it as a loop offset symbol
    sdfg = dace.SDFG("offset")
    sdfg.add_symbol("k", dace.int64)
    sdfg.add_array("a", ["N"], dace.float64)
    inputs = {"k": np.asarray(3), "a": np.zeros(4), "alpha": np.asarray(1.5), "N": np.asarray(4)}

    assert symbol_inputs(sdfg, inputs) == {"k": 3, "N": 4}


@pytest.mark.integration
def test_the_datas_count_wins_over_the_presets(tmp_path: Path) -> None:
    """bicgstab's preset names a nominal ``nnz``; the matrix the data holds has another, and the arrays are that size.
    With the preset's count a GPU program copied part of ``A_data`` to the device and solved for a wrong ``x`` (#94)."""
    setup = Setup("bicgstab", "S", "float64", False, str(tmp_path))
    session = started_session(setup, codegen="first")
    nominal = session.sizes["nnz"]

    inputs, _ = session_data(setup, session)

    assert int(inputs["nnz"]) != nominal, "the fixture must have a preset count the data differs from"
    assert session.sizes["nnz"] == int(inputs["nnz"]) == inputs["A_data"].size
