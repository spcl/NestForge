# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""bench_data: the bench's inputs and reference outputs, named as the generated program names them."""

from helpers import corpus_kernel

from nestforge.corpus.bench import preset_sizes
from nestforge.server import Setup, bench_data


def test_a_renamed_argument_reaches_the_program_by_its_new_name():
    """graupel_sedim2b's generator renames ``frac`` to ``__frac``; under the old name the program ran on zeros."""
    assert corpus_kernel("graupel_sedim2b").renames() == {"frac": "__frac"}
    inputs, reference = bench_data(Setup("graupel_sedim2b", "S", "float64", False))
    assert "__frac" in inputs and "frac" not in inputs
    assert inputs["__frac"].any(), "the bench's fractions, not zeros"
    assert reference is not None and set(reference.outputs) == {"G", "fl", "prec", "q"}


def test_a_kernel_without_renames_keeps_its_names():
    assert corpus_kernel("tsvc_2_s231").renames() == {}


def test_params_replace_a_presets_parameters_for_inputs_and_sizes():
    """``params`` gives the same kernel at another size: the bench's inputs and the session's sizes both follow it, and
    a non-integer knob (``density``) reaches only the inputs."""
    params = {"block_size": 2, "density": 0.05}
    inputs, reference = bench_data(Setup("dbcsr", "S", "float64", False), params)
    assert inputs["a_blocks"].shape[1:] == (2, 2)
    assert reference is not None and reference.outputs["C"].shape == (8 * 2, 8 * 2)
    sizes = preset_sizes(corpus_kernel("dbcsr"), "S", params)
    assert sizes["block_size"] == 2 and sizes["n_block_rows"] == 8 and "density" not in sizes
