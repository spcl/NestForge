# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""bench_data: the bench's inputs and reference outputs, named as the generated program names them."""

from typing import Never

import hpcagent_bench.numerical_oracle as oracle_mod
import numpy as np
import pytest
from helpers import corpus_kernel
from hpcagent_bench.harness.grading import ReferenceUnavailable

import tactics.server as server_mod
from tactics.server import KERNELBENCH_DIR_VAR, Setup, bench_data, numpy_outputs, session_preset, size_rung
from tactics.stages.performance import BENCH_REFERENCE, NUMPY_REFERENCE


def test_a_renamed_argument_reaches_the_program_by_its_new_name() -> None:
    """graupel_substep_sedimentation's generator renames ``frac`` to ``__frac``; under the old name the program ran
    on zeros."""
    assert corpus_kernel("graupel_substep_sedimentation").renames() == {"frac": "__frac"}
    inputs, reference = bench_data(Setup("graupel_substep_sedimentation", "S", "float64", False))
    assert "__frac" in inputs and "frac" not in inputs
    assert inputs["__frac"].any(), "the bench's fractions, not zeros"
    assert reference is not None and set(reference.outputs) == {"G", "fl", "prec", "q"}


def test_a_kernel_without_renames_keeps_its_names() -> None:
    assert corpus_kernel("tsvc_2_s231").renames() == {}


def test_a_missing_kernelbench_reference_names_the_setting_that_points_at_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_submodule(spec: object, data: object) -> Never:
        raise ReferenceUnavailable("torch reference: not in the KernelBench submodule at /site-packages")

    def no_function(spec: object, data: object) -> Never:
        raise ReferenceUnavailable("no NumPy function")

    monkeypatch.setattr(oracle_mod, "compiled_reference", no_submodule)
    monkeypatch.setattr(server_mod, "numpy_outputs", no_function)

    inputs, reference = bench_data(Setup("graupel_substep_sedimentation", "S", "float64", False))

    assert reference is None and inputs
    assert KERNELBENCH_DIR_VAR in capsys.readouterr().err


# Reproducer: kernel without KernelBench data had no grading reference (#85)
def test_a_kernel_the_bench_cannot_grade_is_graded_against_its_numpy_function(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """#85: with the KernelBench submodule missing, the kernel's own NumPy function stands in, and says so."""

    def no_submodule(spec: object, data: object) -> Never:
        raise ReferenceUnavailable("torch reference: not in the KernelBench submodule at /site-packages")

    monkeypatch.setattr(oracle_mod, "compiled_reference", no_submodule)

    inputs, reference = bench_data(Setup("conv3d_gn_mean", "S", "float64", False))

    assert reference is not None and set(reference.outputs) == {"out"}
    assert reference.source == NUMPY_REFERENCE != BENCH_REFERENCE
    assert np.isfinite(reference.outputs["out"]).all() and reference.outputs["out"].any()
    assert "graded against the kernel's NumPy function" in capsys.readouterr().err


def test_the_numpy_function_leaves_the_inputs_it_was_given_as_drawn() -> None:
    from hpcagent_bench.categories import Preset
    from hpcagent_bench.frameworks.benchmark import Benchmark

    kernel = corpus_kernel("conv3d_gn_mean")
    data = dict(Benchmark(kernel.short_name).get_data(preset=Preset.parse("S")))
    drawn = {name: np.array(value, copy=True) for name, value in data.items() if isinstance(value, np.ndarray)}

    first = numpy_outputs(kernel.spec, data)
    second = numpy_outputs(kernel.spec, data)

    assert all(np.array_equal(data[name], value) for name, value in drawn.items())
    assert np.array_equal(first["out"], second["out"])


def test_a_fuzzed_token_runs_a_session_on_its_rung() -> None:
    """A judge on ``XL+fuzz`` hands the token to the reference flavor and ``fuzzed`` to the agent's route."""
    assert [size_rung(token) for token in ("S", "M+fuzz", "XL+fuzz", "fuzzed")] == ["S", "M", "XL", "XL"]


# Reproducer: an L+fuzz judge started every Tactics session at XL and OOMed on a Mac (T48-34, jacobi_1d, f58cc92)
def test_a_session_of_a_fuzzed_judge_keeps_the_anchor_rung() -> None:
    """``resolve_preset("L+fuzz")`` leaves ``fuzzed`` and moves L into ``fuzz.anchor``; the session's token must
    carry L, since the spawned session process does not see that override."""
    from hpcagent_bench import config, fuzz
    from hpcagent_bench.categories import Preset
    from hpcagent_bench.spec import resolve_preset

    try:
        resolved = resolve_preset("L+fuzz")
        assert resolved is Preset.FUZZED
        assert session_preset(resolved.text) == "L+fuzz"
        assert size_rung(session_preset(resolved.text)) == "L"
        assert size_rung(session_preset(resolve_preset("XL+fuzz").text)) == "XL"
        assert [session_preset(kept) for kept in ("S", "M+fuzz", "XL")] == ["S", "M+fuzz", "XL"]
    finally:
        config.clear_override("fuzz.anchor")
        config.clear_override(fuzz.PRESET_SEED_KEY)
