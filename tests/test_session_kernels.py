# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Session stages 5 and 6: optimize a kernel, replace its source, sweep its configurations, link the winner."""

from pathlib import Path

import numpy as np
import pytest

import dace

from nestforge.build import flags
from nestforge.session import Session
from nestforge.stages.canonicalize import Targets

N = dace.symbol("N", dtype=dace.int64)

pytestmark = pytest.mark.e2e


@dace.program
def scaled_sum(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = 2.0 * a[i] + b[i]


def one_kernel_session(tmp_path, n: int = 256) -> tuple:
    session = Session(scaled_sum.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": n})
    kernels = session.define_scopes()
    assert len(kernels) == 1
    return session, kernels[0]["name"]


def assert_program_computes(sdfg: dace.SDFG) -> None:
    rng = np.random.default_rng(0)
    a, b, c = rng.random(256), rng.random(256), np.zeros(256)
    sdfg(a=a, b=b, c=c, N=256)
    np.testing.assert_allclose(c, 2.0 * a + b, rtol=1e-15, atol=0)


def test_optimize_kernel_writes_one_c_entry_with_the_boundary_arguments(tmp_path):
    """The optimized kernel exposes one extern C symbol whose arguments are the boundary's names."""
    session, name = one_kernel_session(tmp_path)
    info = session.optimize_kernel(name)
    text = Path(info["unit"]).read_text()
    assert Path(info["unit"]).name == f"{info['kernel']}.cpp"
    assert text.count('extern "C"') == 1
    assert f"void {info['symbol']}(" in text
    assert info["entry"] == f"{info['symbol']}({', '.join(info['abi_order'])})"
    assert sorted(info["abi_order"]) == sorted(session.kernel_boundary(name)["boundary_order"])


def test_optimize_kernel_binds_its_default_library_and_the_program_still_computes(tmp_path):
    """Stage 5 alone links a library: the kernel calls the archive it built, with its runtimes, and 2a + b holds."""
    session, name = one_kernel_session(tmp_path)

    info = session.optimize_kernel(name)

    ext = session.kernel(name)
    assert ext.implementation == "ExternCall"
    assert ext.lib_path == info["library"] and Path(info["library"]).name == f"lib{info['kernel']}.a"
    assert ext.runtime_libraries and info["variant"].count(":") == 2
    assert_program_computes(session.sdfg)


def test_a_kernel_another_session_lowered_can_be_optimized_by_a_fresh_session(tmp_path):
    sdfg = scaled_sum.to_sdfg(simplify=True)
    Session(sdfg, work_dir=str(tmp_path / "lowering")).define_scopes()
    fresh = Session(sdfg, work_dir=str(tmp_path / "fresh"))
    (kernel,) = fresh.list_kernels()

    info = fresh.optimize_kernel(kernel["name"])

    ext = fresh.kernel(kernel["name"])
    assert ext.implementation == "ExternCall" and ext.lib_path == info["library"]
    assert sorted(info["abi_order"]) == sorted(fresh.kernel_boundary(kernel["name"])["boundary_order"])
    assert_program_computes(sdfg)


def test_a_kernel_without_its_standalone_sdfg_is_refused_before_it_is_scheduled(tmp_path):
    session, name = one_kernel_session(tmp_path)
    session.kernel(name).standalone_sdfg = None

    with pytest.raises(ValueError, match="ExternalCall 'extcall_0' has no standalone SDFG"):
        session.optimize_kernel(name)


def test_a_kernel_source_that_matches_the_oracle_is_linked_and_the_program_computes_with_it(tmp_path):
    session, name = one_kernel_session(tmp_path)
    session.optimize_kernel(name)
    original = session.kernel_source(name)
    edited = "// edited by hand\n" + original

    outcome = session.set_kernel_source(name, edited, "cpp", reps=1)

    assert outcome["status"] == "ok" and outcome["time_us"] > 0.0, outcome
    assert session.kernel_source(name) == edited
    assert "agent" in session.kernel(name).lib_path
    assert_program_computes(session.sdfg)


def test_a_kernel_source_with_a_wrong_result_is_refused_and_the_old_library_stays(tmp_path):
    session, name = one_kernel_session(tmp_path)
    before = session.optimize_kernel(name)["library"]
    wrong = session.kernel_source(name).replace("2.0", "3.0")

    outcome = session.set_kernel_source(name, wrong, "cpp", reps=1)

    assert outcome["status"] == "wrong" and "max rel err" in outcome["reason"], outcome
    assert session.kernel(name).lib_path == before


@pytest.mark.parametrize(
    ("old", "new", "language", "status"),
    [
        ("int64_t N", "int64_t M", "cpp", "refused"),  # another signature
        ("", "", "cuda", "refused"),  # a CPU kernel is C++
        ("#pragma omp parallel for", "#pragma omp parallel for\nsyntax error;", "cpp", "build-failed"),
    ],
)
def test_a_kernel_source_that_cannot_stand_in_is_not_linked(tmp_path, old, new, language, status):
    session, name = one_kernel_session(tmp_path)
    before = session.optimize_kernel(name)["library"]

    outcome = session.set_kernel_source(name, session.kernel_source(name).replace(old, new), language, reps=1)

    assert outcome["status"] == status, outcome
    assert session.kernel(name).lib_path == before


def test_sweep_links_the_fastest_correct_variant_into_the_program(tmp_path):
    """After the sweep the kernel calls the winning archive, and the program still computes 2a + b."""
    session, name = one_kernel_session(tmp_path)
    result = session.sweep(name, reps=2, compilers=["gcc"])
    assert result["winner"] is not None
    assert result["cells"] >= 1
    config = result["config"]
    assert config["compiler"] == "g++"
    assert result["winner"] == f"g++:{config['fp_mode']}:{config['cost_model']}"
    assert "-O3" in config["flags"] and config["time_us"] > 0.0
    ext = session.kernel(name)
    assert ext.implementation == "ExternCall"
    assert ext.lib_path.endswith(".a")
    assert_program_computes(session.sdfg)


def test_sweep_without_matching_compilers_reports_no_winner(tmp_path):
    """A compiler filter that matches nothing leaves the kernel on its DaCe reference expansion."""
    session, name = one_kernel_session(tmp_path, 64)
    result = session.sweep(name, reps=1, compilers=["no-such-compiler"])
    assert result["winner"] is None
    assert result["config"] == dict.fromkeys(("compiler", "fp_mode", "cost_model", "flags", "time_us"))
    assert session.kernel(name).implementation != "ExternCall"


@pytest.mark.gpu
def test_a_gpu_sweep_reports_an_nvcc_configuration_without_a_cost_model(tmp_path):
    sdfg = scaled_sum.to_sdfg(simplify=True)
    session = Session(sdfg, targets=Targets(gpu=True), work_dir=str(tmp_path), sizes={"N": 256})
    session.define_scopes()
    (kernel,) = session.place()["kernels"]

    result = session.sweep(kernel["name"], reps=2)

    assert result["winner"] is not None
    config = result["config"]
    assert config["compiler"].startswith("nvcc-") and config["cost_model"] == flags.NO_COST_MODEL
    assert config["fp_mode"] in flags.CUDA_FP and "-arch=native" in config["flags"]
    ext = session.kernel(kernel["name"])
    assert ext.implementation == "ExternCall" and ext.lib_path.endswith(".a")
