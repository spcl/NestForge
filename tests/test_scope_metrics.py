# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scope metrics: the work and depth a scheduling agent reads per scope."""

import dace
import numpy as np
import sympy

from dace.sdfg.state import LoopRegion
from nestforge.stages.moves import ScopeMetrics, scope_metrics
from nestforge.stages.scopes import top_level_map_entries
from nestforge.session import Session

N = dace.symbol("N")
TSTEPS = dace.symbol("TSTEPS")


@dace.program
def vector_add(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + B[i]


@dace.program
def stencil_sweeps(A: dace.float64[N], B: dace.float64[N]):
    for t in range(TSTEPS):
        for i in dace.map[1 : N - 1]:
            B[i] = A[i - 1] + A[i] + A[i + 1]


@dace.program
def vertical_pair(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    T = np.empty_like(A)
    for i in dace.map[0:N]:
        T[i] = A[i] + B[i]
    for i in dace.map[0:N]:
        C[i] = T[i] * 2.0


def top_level_maps(sdfg: dace.SDFG):
    return [entry for state in sdfg.states() for entry in top_level_map_entries(state)]


def assert_symbolic(actual: sympy.Expr, expected: sympy.Expr) -> None:
    assert dace.symbolic.equal(actual, expected) is True, f"{actual} != {expected}"


def test_a_vector_add_does_one_operation_per_element_at_depth_one():
    sdfg = vector_add.to_sdfg(simplify=True)
    (entry,) = top_level_maps(sdfg)
    metrics = scope_metrics(sdfg, entry)
    assert isinstance(metrics, ScopeMetrics)
    assert_symbolic(metrics.work, N)
    assert_symbolic(metrics.depth, 1)


def test_a_loop_around_a_stencil_map_scales_work_and_depth_by_the_sweeps():
    sdfg = stencil_sweeps.to_sdfg(simplify=True)
    (loop,) = [block for block in sdfg.nodes() if isinstance(block, LoopRegion)]
    (entry,) = top_level_maps(sdfg)
    assert entry in [node for state in loop.states() for node in state.nodes()]
    in_map = scope_metrics(sdfg, entry)
    in_loop = scope_metrics(sdfg, loop)
    assert_symbolic(in_map.work, 2 * N - 4)
    assert_symbolic(in_map.depth, 2)
    assert_symbolic(in_loop.work, TSTEPS * (2 * N - 4))
    assert_symbolic(in_loop.depth, 2 * TSTEPS)


def test_scope_metrics_leaves_the_program_untouched():
    sdfg = stencil_sweeps.to_sdfg(simplify=True)
    before = sdfg.to_json()
    (loop,) = [block for block in sdfg.nodes() if isinstance(block, LoopRegion)]
    scope_metrics(sdfg, loop)
    scope_metrics(sdfg, top_level_maps(sdfg)[0])
    assert sdfg.to_json() == before


def test_describe_appends_metrics_to_kernel_lines_only_when_asked():
    session = Session(vector_add.to_sdfg(simplify=True))
    plain = session.describe()
    with_metrics = session.describe(metrics=True)
    assert "work=" not in plain
    (kernel,) = [line for line in with_metrics.splitlines() if "reads=" in line]
    assert kernel.endswith("  work=N depth=1"), kernel
    assert with_metrics.replace("  work=N depth=1", "") == plain
    assert session.epoch == 0
