# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The LLVM and MLIR baselines: every kernel built through the agent's own pass pipeline, with NestForge's entry,
library and oracle check; the pipeline's syntax is checked when it is set, and rollback takes it back."""

from pathlib import Path

import dace
import numpy as np
import pytest
from dace.libraries.standard.nodes.external_call import external_calls

from tactics.build.pipelines import pipeline_syntax_error
from tactics.records import MoveStatus, Refusal
from tactics.session import Session
from tactics.stages.performance import Reference

N = dace.symbol("N", dtype=dace.int64)
SIZE = 64


@dace.program
def doubled(a: dace.float64[N], b: dace.float64[N]) -> None:
    for i in dace.map[0:N]:
        b[i] = 2.0 * a[i] + 1.0


def session(tmp_path: Path) -> Session:
    sut = Session(doubled.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": SIZE})
    sut.canonicalize()
    return sut


def data() -> tuple[dict[str, np.ndarray], Reference]:
    a = np.arange(1.0, SIZE + 1)
    return {"a": a, "b": np.zeros(SIZE)}, Reference({"b": 2.0 * a + 1.0}, 0.0, 0.0)


def built(sut: Session) -> list[str]:
    """The toolchain each kernel was built by, after the default scopes and builds."""
    sut.default_scopes()
    names = [ext.name for ext in external_calls(sut.sdfg)]
    for name in names:
        sut.default_build(name)
    return [sut.builds[name].variant.toolchain for name in names]


def test_every_kernel_builds_through_the_llvm_pipeline_and_the_program_stays_correct(tmp_path: Path) -> None:
    sut = session(tmp_path)

    done = sut.set_pipeline("llvm", "default<O2>")

    assert done.status is MoveStatus.APPLIED
    assert built(sut) == ["llvm-pipeline"]
    report = sut.performance(*data()).structured
    assert report.correct is True and report.time_us is not None


def test_a_pipeline_opt_cannot_parse_is_refused_with_opts_message(tmp_path: Path) -> None:
    sut = session(tmp_path)

    refused = sut.set_pipeline("llvm", "loop-unroll,no-such-pass")

    assert (refused.status, refused.refusal) == (MoveStatus.ILLEGAL, Refusal.ARGUMENT)
    assert "no-such-pass" in refused.reason
    assert sut.pipeline is None


def test_rollback_takes_the_pipeline_back_to_nestforges_build(tmp_path: Path) -> None:
    sut = session(tmp_path)
    sut.set_pipeline("llvm", "default<O1>")

    sut.apply_move("rollback", [])

    assert sut.pipeline is None
    assert built(sut) != ["llvm-pipeline"]


@pytest.mark.vendor  # Polygeist (cgeist, polygeist-opt) is in the AMD judge image, not on CI
def test_the_mlir_pipeline_builds_a_correct_kernel_through_polygeist(tmp_path: Path) -> None:
    sut = session(tmp_path)
    assert pipeline_syntax_error("mlir", "clang++", "builtin.module(canonicalize)") == ""

    done = sut.set_pipeline("mlir", "builtin.module(affine-loop-tile{tile-size=8},canonicalize)")

    assert done.status is MoveStatus.APPLIED
    assert built(sut) == ["mlir-pipeline"]
    assert sut.performance(*data()).structured.correct is True
