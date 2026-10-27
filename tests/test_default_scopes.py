# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Session.default_scopes: which regions of the program become kernels by default, and that the program computes
the same afterwards."""

import copy
import os

import dace
import numpy as np
from dace.libraries.standard.nodes.external_call import ExternalCall
from dace.sdfg.state import ConditionalBlock, LoopRegion

from nestforge.session import Session
from nestforge.stages.canonicalize import Targets

N = dace.symbol("N", dtype=dace.int64)
SIZE = 8


@dace.program
def time_steps(A: dace.float64[N], B: dace.float64[N]):
    for t in range(3):
        for i in dace.map[1 : N - 1]:
            B[i] = (A[i - 1] + A[i] + A[i + 1]) / 3.0
        for i in dace.map[1 : N - 1]:
            A[i] = B[i] + 1.0


@dace.program
def running_sum(x: dace.float64[N], out: dace.float64[N]):
    for i in range(1, N):
        out[i] = out[i - 1] + x[i]


@dace.program
def halve_and_count(A: dace.float64[N], s: dace.float64[1]):
    for t in range(3):
        for i in dace.map[0:N]:
            A[i] = A[i] * 0.5
        s[0] = s[0] + A[0]


@dace.program
def two_loops_of_maps(A: dace.float64[N, N]):
    for t in range(2):
        for k in range(2):
            for i in dace.map[0:N]:
                A[k, i] = A[k, i] + 1.0


@dace.program
def three_loops_of_maps(A: dace.float64[N, N]):
    for t in range(2):
        for k in range(2):
            for j in range(2):
                for i in dace.map[0:N]:
                    A[k, i] = A[k, i] + j


@dace.program
def branches(A: dace.float64[N], flag: dace.int64):
    if flag > 0:
        for i in dace.map[0:N]:
            A[i] = A[i] + 1.0
    else:
        for i in dace.map[0:N]:
            A[i] = A[i] - 1.0


@dace.program
def long_body(A: dace.float64[N, 4], C: dace.float64[N, 4]):
    for i in dace.map[0:N]:
        for k in range(4):
            A[i, k] = A[i, k] + 1.0
        for k in range(4):
            C[i, k] = A[i, k] * 2.0


def session_and_reference(program, targets: Targets | None = None) -> tuple[Session, dace.SDFG]:
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = f"{sdfg.name}_{os.getpid()}"  # one build folder per xdist worker
    reference = copy.deepcopy(sdfg)
    reference.name = f"{sdfg.name}_reference"
    return Session(sdfg, targets), reference


def blocks_of(sdfg: dace.SDFG, kind: type) -> int:
    return sum(isinstance(block, kind) for block in sdfg.all_control_flow_blocks())


def same_result(session: Session, reference: dace.SDFG, **arrays: np.ndarray | int) -> None:
    expected = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in arrays.items()}
    reference(**expected, N=SIZE)
    session.sdfg(**arrays, N=SIZE)
    for name, value in arrays.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_allclose(value, expected[name], rtol=1e-14)


def test_a_loop_of_parallel_maps_keeps_the_loop_and_makes_each_map_a_parallel_kernel():
    session, reference = session_and_reference(time_steps)

    kernels = session.default_scopes()

    assert len(kernels) == 2 and all(k.parallel for k in kernels)
    assert blocks_of(session.sdfg, LoopRegion) == 1, "the time loop stays in the program"
    rng = np.random.default_rng(0)
    same_result(session, reference, A=rng.random(SIZE), B=np.zeros(SIZE))


def test_a_loop_holding_no_parallel_map_is_one_sequential_kernel():
    session, reference = session_and_reference(running_sum)

    (kernel,) = session.default_scopes()

    assert not kernel.parallel, "placement keeps a sequential kernel on the CPU"
    assert blocks_of(session.sdfg, LoopRegion) == 0, "the loop left the program"
    same_result(session, reference, x=np.arange(SIZE, dtype=np.float64), out=np.zeros(SIZE))


def test_a_loop_mixing_a_map_with_sequential_work_is_one_kernel():
    session, reference = session_and_reference(halve_and_count)

    (kernel,) = session.default_scopes()

    assert not kernel.parallel
    assert blocks_of(session.sdfg, LoopRegion) == 0
    same_result(session, reference, A=np.linspace(1.0, 2.0, SIZE), s=np.zeros(1))


def test_a_loop_of_maps_inside_a_loop_is_opened_one_level_deeper():
    session, reference = session_and_reference(two_loops_of_maps)

    (kernel,) = session.default_scopes()

    assert kernel.parallel, "the map itself is the kernel"
    assert blocks_of(session.sdfg, LoopRegion) == 2, "both loops stay in the program"
    same_result(session, reference, A=np.random.default_rng(1).random((SIZE, SIZE)))


def test_a_loop_two_levels_down_is_one_kernel_with_its_loops():
    session, reference = session_and_reference(three_loops_of_maps)

    (kernel,) = session.default_scopes()

    assert not kernel.parallel
    assert blocks_of(session.sdfg, LoopRegion) == 1, "only the outermost loop stays in the program"
    same_result(session, reference, A=np.random.default_rng(2).random((SIZE, SIZE)))


def test_each_branch_of_a_conditional_is_opened_like_a_loop_body():
    session, reference = session_and_reference(branches)

    kernels = session.default_scopes()

    assert len(kernels) == 2 and all(k.parallel for k in kernels)
    assert blocks_of(session.sdfg, ConditionalBlock) == 1, "the conditional stays in the program"
    for flag in (1, 0):
        same_result(session, reference, A=np.arange(SIZE, dtype=np.float64), flag=flag)


def test_a_map_above_the_statement_limit_is_split_into_kernels_at_a_body_block():
    session, reference = session_and_reference(long_body, Targets(max_statements=1))

    kernels = session.default_scopes()

    assert len(kernels) == 2 and all(k.parallel for k in kernels)
    calls = [n for n, _ in session.sdfg.all_nodes_recursive() if isinstance(n, ExternalCall)]
    assert len(calls) == 2
    rng = np.random.default_rng(3)
    same_result(session, reference, A=rng.random((SIZE, 4)), C=np.zeros((SIZE, 4)))


def test_a_map_within_the_statement_limit_stays_one_kernel():
    session, _ = session_and_reference(long_body)

    kernels = session.default_scopes()

    assert len(kernels) == 1
