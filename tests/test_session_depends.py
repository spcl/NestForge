# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Session over the kernel dependency graph: one graph per epoch, the kernel listing, the tree's dependency lines,
and the per-epoch JSON snapshot."""

from pathlib import Path

import dace
import numpy as np
import pytest

from nestforge.ir.depends import UnsupportedProgram
from nestforge.session import Session

N = dace.symbol("N")


@dace.program
def chain(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    T = np.empty_like(A)
    for i in dace.map[0:N]:
        T[i] = A[i] + B[i]
    for i in dace.map[0:N]:
        C[i] = T[i] * 2.0


def scoped_session(sdfg: dace.SDFG, work_dir: Path) -> Session:
    session = Session(sdfg, work_dir=str(work_dir))
    session.default_scopes()
    return session


def test_the_kernel_graph_is_computed_once_per_epoch_and_again_after_a_mutation(tmp_path):
    session = Session(chain.to_sdfg(simplify=True), work_dir=str(tmp_path))
    before = session.kernel_graph()
    assert session.kernel_graph() is before

    session.default_scopes()

    after = session.kernel_graph()
    assert after is not before
    assert (before.kernels, after.kernels) == ((), ("extcall_0", "extcall_1"))


def test_list_kernels_names_each_argument_and_what_reaches_it(tmp_path):
    session = scoped_session(chain.to_sdfg(simplify=True), tmp_path)

    kernels = session.list_kernels()

    assert [kernel.model_dump(mode="json") for kernel in kernels] == [
        {
            "name": "extcall_0",
            "device": "cpu",
            "parallel": True,
            "inputs": ["A", "B"],
            "outputs": ["T"],
            "symbols": ["N"],
            "depends": {"A": ["program"], "B": ["program"], "N": ["program"]},
            "carried": {},
        },
        {
            "name": "extcall_1",
            "device": "cpu",
            "parallel": True,
            "inputs": ["T"],
            "outputs": ["C"],
            "symbols": ["N"],
            "depends": {"T": ["extcall_0.T"], "N": ["program"]},
            "carried": {},
        },
    ]


def test_list_kernels_reports_the_device_stage_4_placed_each_kernel_on(tmp_path):
    session = scoped_session(chain.to_sdfg(simplify=True), tmp_path)

    session.place()

    assert [kernel.device for kernel in session.list_kernels()] == ["cpu", "cpu"]


def test_describe_with_deps_prints_each_kernels_line_under_its_row(tmp_path):
    session = scoped_session(chain.to_sdfg(simplify=True), tmp_path)
    session.expand([session.row_index()["extcall_1"][1].label], session.epoch)

    lines = session.describe(deps=True).splitlines()

    row = next(index for index, line in enumerate(lines) if "extcall_1  LIBNODE" in line)
    assert lines[row + 1].endswith("extcall_1: T <- extcall_0.T, N <- program")
    assert "extcall_1: T <- extcall_0.T" not in session.describe()


def test_default_scopes_completes_when_the_analysis_refuses_the_program(tmp_path):
    sdfg = chain.to_sdfg(simplify=True)
    sdfg.add_reference("R", [N], dace.float64)
    session = Session(sdfg, work_dir=str(tmp_path))

    kernels = session.default_scopes()

    assert [kernel.name for kernel in kernels] == ["extcall_0", "extcall_1"]
    with pytest.raises(UnsupportedProgram, match="'R' .* Reference"):
        session.kernel_graph()
    with pytest.raises(UnsupportedProgram, match="'R' .* Reference"):
        session.list_kernels()
