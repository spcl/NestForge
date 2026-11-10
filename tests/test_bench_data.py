# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""bench_data: the bench's inputs and reference outputs, named as the generated program names them."""

import hpcagent_bench.numerical_oracle as oracle_mod
import numpy as np
from helpers import corpus_kernel
from hpcagent_bench.harness.grading import ReferenceUnavailable

import nestforge.server as server_mod
from nestforge.server import KERNELBENCH_DIR_VAR, Setup, bench_data, numpy_outputs, size_rung
from nestforge.stages.performance import BENCH_REFERENCE, NUMPY_REFERENCE


def test_a_renamed_argument_reaches_the_program_by_its_new_name():
    """graupel_sedim2b's generator renames ``frac`` to ``__frac``; under the old name the program ran on zeros."""
    assert corpus_kernel("graupel_sedim2b").renames() == {"frac": "__frac"}
    inputs, reference = bench_data(Setup("graupel_sedim2b", "S", "float64", False))
    assert "__frac" in inputs and "frac" not in inputs
    assert inputs["__frac"].any(), "the bench's fractions, not zeros"
    assert reference is not None and set(reference.outputs) == {"G", "fl", "prec", "q"}


def test_a_kernel_without_renames_keeps_its_names():
    assert corpus_kernel("tsvc_2_s231").renames() == {}


def test_a_missing_kernelbench_reference_names_the_setting_that_points_at_one(monkeypatch, capsys):
    def no_submodule(spec, data):
        raise ReferenceUnavailable("torch reference: not in the KernelBench submodule at /site-packages")

    def no_function(spec, data):
        raise ReferenceUnavailable("no NumPy function")

    monkeypatch.setattr(oracle_mod, "compiled_reference", no_submodule)
    monkeypatch.setattr(server_mod, "numpy_outputs", no_function)

    inputs, reference = bench_data(Setup("graupel_sedim2b", "S", "float64", False))

    assert reference is None and inputs
    assert KERNELBENCH_DIR_VAR in capsys.readouterr().err


def test_a_kernel_the_bench_cannot_grade_is_graded_against_its_numpy_function(monkeypatch, capsys):
    """#85: with the KernelBench submodule missing, the kernel's own NumPy function stands in, and says so."""

    def no_submodule(spec, data):
        raise ReferenceUnavailable("torch reference: not in the KernelBench submodule at /site-packages")

    monkeypatch.setattr(oracle_mod, "compiled_reference", no_submodule)

    inputs, reference = bench_data(Setup("conv3d_group_norm_mean", "S", "float64", False))

    assert reference is not None and set(reference.outputs) == {"out"}
    assert reference.source == NUMPY_REFERENCE != BENCH_REFERENCE
    assert np.isfinite(reference.outputs["out"]).all() and reference.outputs["out"].any()
    assert "graded against the kernel's NumPy function" in capsys.readouterr().err


def test_the_numpy_function_leaves_the_inputs_it_was_given_as_drawn():
    from hpcagent_bench.categories import Preset
    from hpcagent_bench.frameworks.benchmark import Benchmark

    kernel = corpus_kernel("conv3d_group_norm_mean")
    data = dict(Benchmark(kernel.short_name).get_data(preset=Preset.parse("S")))
    drawn = {name: np.array(value, copy=True) for name, value in data.items() if isinstance(value, np.ndarray)}

    first = numpy_outputs(kernel.spec, data)
    second = numpy_outputs(kernel.spec, data)

    assert all(np.array_equal(data[name], value) for name, value in drawn.items())
    assert np.array_equal(first["out"], second["out"])


def test_a_fuzzed_token_runs_a_session_on_its_rung():
    """A judge on ``XL+fuzz`` hands the token to the reference flavor and ``fuzzed`` to the agent's route."""
    assert [size_rung(token) for token in ("S", "M+fuzz", "XL+fuzz", "fuzzed")] == ["S", "M", "XL", "XL"]
