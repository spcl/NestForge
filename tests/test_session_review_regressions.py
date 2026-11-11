# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regressions of a review of the session, its server, the replay and the corpus loader, each on a tiny kernel."""

import sys
from http import HTTPStatus
from pathlib import Path

import dace
import numpy as np
import pytest
from helpers import corpus_kernel
from hpcagent_bench.harness.routes import ROUTES
from stub_session import Stub
from test_bench_plugin import (  # noqa: F401 -- the route tests' fixtures
    KERNEL,
    SLOT_CALLS,
    loaded,
    post,
    restart_judge,
    shared_folder,
    stub,
)

import nestforge.server as server_mod
from nestforge.corpus.bench import CorpusKernel
from nestforge.replay import Context, run_variant
from nestforge.server import Setup, bench_data, numpy_outputs
from nestforge.session import Session
from nestforge.stages.performance import Reference

N = dace.symbol("N", dtype=dace.int64)
SIZE = 64


@dace.program
def scaled_sum(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = 2.0 * a[i] + b[i]


def one_kernel_session(tmp_path: Path) -> tuple[Session, str]:
    session = Session(scaled_sum.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": SIZE})
    (kernel,) = session.default_scopes()
    return session, kernel.name


def scaled_sum_data() -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(0)
    a, b = rng.random(SIZE), rng.random(SIZE)
    return {"a": a, "b": b, "c": np.zeros(SIZE)}, 2.0 * a + b


@pytest.mark.e2e
def test_measuring_a_cpu_program_keeps_the_kernel_build_it_was_given(tmp_path):
    """A CPU program is not placed to be measured: placing would drop the written or tuned build for the default."""
    session, name = one_kernel_session(tmp_path)
    session.optimize_kernel(name)
    trial = session.trial()
    given = trial.builds[name]
    inputs, expected = scaled_sum_data()

    report = trial.measure(inputs, Reference({"c": expected}, 0.0, 0.0), 1)

    assert trial.builds[name] is given, "the measured kernel is the build the session holds"
    assert trial.unplaced is None and report.structured.correct is True


@pytest.mark.e2e
def test_each_sweep_winner_is_kept_in_a_folder_of_its_own(tmp_path):
    """Sweeping one source again keeps the count of winners; the folder named by it overwrote a linked winner."""
    session, name = one_kernel_session(tmp_path)

    winners = []
    for _ in range(3):
        assert session.sweep(name, reps=1, compilers=["gcc"]).winner is not None
        winners.append(session.builds[name])

    assert len({build.archive.parent for build in winners}) == len(winners)
    assert all(build.archive.is_file() for build in winners)


@pytest.mark.e2e
def test_performance_measures_again_for_other_arguments(tmp_path):
    session, _ = one_kernel_session(tmp_path)
    inputs, expected = scaled_sum_data()
    right, wrong = Reference({"c": expected}, 0.0, 0.0), Reference({"c": expected + 1.0}, 0.0, 0.0)

    first = session.performance(inputs, right, reps=1, codegen="first")
    second = session.performance(inputs, wrong, reps=1, codegen="first")

    assert first.structured.correct is True and second.structured.correct is False
    assert session.performance(inputs, wrong, reps=1, codegen="first") is second, "the same arguments reuse it"


def test_the_numpy_reference_is_called_by_the_specs_function_name():
    """tsvc_2_s000's module is ``tsvc_2_s000`` and its function ``s000``."""
    from hpcagent_bench.categories import Preset
    from hpcagent_bench.frameworks.benchmark import Benchmark

    kernel = corpus_kernel("tsvc_2_s000")
    assert kernel.spec.func_name != kernel.spec.module_name
    data = dict(Benchmark(kernel.short_name).get_data(preset=Preset.parse("S")))

    outputs = numpy_outputs(kernel.spec, data)

    assert set(outputs) == set(kernel.spec.output_args) and all(v.size for v in outputs.values())


def test_a_complex_scalar_of_the_bench_reaches_the_program(monkeypatch):
    from hpcagent_bench.frameworks.benchmark import Benchmark

    drawn = Benchmark.get_data
    monkeypatch.setattr(Benchmark, "get_data", lambda self, **kw: {**drawn(self, **kw), "shift": 1.0 + 2.0j})
    monkeypatch.setattr(server_mod, "numpy_outputs", lambda spec, data: {})
    monkeypatch.setattr("hpcagent_bench.numerical_oracle.compiled_reference", lambda spec, data: {})

    inputs, _ = bench_data(Setup("tsvc_2_s000", "S", "float64", False))

    assert inputs["shift"] == 1.0 + 2.0j


def test_a_stale_generated_program_is_regenerated():
    kernel = corpus_kernel("tsvc_2_s1112")
    fresh = kernel.dace_file.read_text()
    try:
        # still marked generated (without the mark it is a hand override, never regenerated), but not the emit
        kernel.dace_file.write_text(f"{fresh.splitlines()[0]}\n# stale\n")
        assert corpus_kernel("tsvc_2_s1112").dace_file.read_text() == fresh
    finally:
        kernel.dace_file.write_text(fresh)


def test_a_module_that_fails_to_import_is_not_cached(tmp_path):
    broken = tmp_path / "broken_dace.py"
    broken.write_text("raise RuntimeError('half-run')\n")
    kernel = CorpusKernel("broken", "nestforge_test_broken_dace", broken, corpus_kernel("tsvc_2_s000").spec)

    for _ in range(2):  # the second import runs the file again, not a half-run module
        with pytest.raises(RuntimeError, match="half-run"):
            kernel.module()
    assert kernel.module_path not in sys.modules


def test_a_variant_that_raises_is_recorded_and_the_replay_goes_on(tmp_path):
    session, _ = one_kernel_session(tmp_path)

    def broken(script, ctx):
        raise ValueError("no such schedule")

    record = run_variant(session, Context("cpu", 1, "S"), "broken", broken, ({}, None), 1)

    assert record.error == "ValueError: no such schedule" and record.time_ms is None


def test_a_tuned_kernel_is_journaled_and_repeated_by_a_fresh_session(stub: Stub) -> None:  # noqa: F811
    """finalize sweeps no written kernel, so a sweep lost with its session would be lost for good."""
    post("write_kernel", "episode-tuned", **SLOT_CALLS["write_kernel"])
    assert post("tune_kernel", "episode-tuned", name="kernel_0").code == HTTPStatus.OK
    restart_judge()
    before = len(stub.calls)

    post("view_tree", "episode-tuned")

    route = sys.modules[ROUTES["nestforge"].handler.__module__]
    assert [tool for tool, _ in route.journal_of("episode-tuned", KERNEL)] == ["write_kernel", "tune_kernel"]
    assert [tool for tool, _ in stub.calls[before:]] == ["write_kernel", "tune_kernel", "view_tree"]
