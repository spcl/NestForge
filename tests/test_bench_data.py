# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""bench_data: the bench's inputs and reference outputs, named as the generated program names them."""

from helpers import corpus_kernel

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
