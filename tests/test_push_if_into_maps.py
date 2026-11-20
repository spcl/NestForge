# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for ``push_if_into_maps``: ``if c: map i: B`` becomes ``map i: if c: B`` wherever the conditional sits."""

import copy
from collections.abc import Callable

import dace
import numpy as np
import pytest
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion

from nestforge.stages.sinking import push_if_into_maps

N = dace.symbol("N")
K = dace.symbol("K")
SIZE = 8


@dace.program
def map_under_symbol_guard(a: dace.float64[N], b: dace.float64[N]) -> None:
    if K > 0:
        for i in dace.map[0:N]:
            b[i] = a[i] * 2.0 + 1.0


@dace.program
def map_under_data_guard(flag: dace.int32[1], a: dace.float64[N], b: dace.float64[N]) -> None:
    if flag[0] > 0:
        for i in dace.map[0:N]:
            b[i] = a[i] * 2.0 + 1.0


@dace.program
def map_under_guard_in_loop(a: dace.float64[N], b: dace.float64[N]) -> None:
    for t in range(4):
        if K > t:
            for i in dace.map[0:N]:
                b[i] = b[i] + a[i]


@dace.program
def map_under_guard_with_else(a: dace.float64[N], b: dace.float64[N]) -> None:
    if K > 0:
        for i in dace.map[0:N]:
            b[i] = a[i] * 2.0
    else:
        for i in dace.map[0:N]:
            b[i] = a[i] * 3.0


def map_writes_what_its_guard_reads() -> dace.SDFG:
    """``if a[0] > 0: map i: a[i] = 2 * a[i]``, the condition reading the array itself (the frontend would copy
    ``a[0]`` into a scalar first, which the map does not write)."""
    sdfg = dace.SDFG("map_writes_what_its_guard_reads")
    sdfg.add_array("a", [N], dace.float64)
    cond = ConditionalBlock("guard")
    sdfg.add_node(cond, is_start_block=True)
    body = ControlFlowRegion("guard_body", sdfg=sdfg)
    cond.add_branch(CodeBlock("a[0] > 0.0"), body)
    state = body.add_state("double", is_start_block=True)
    state.add_mapped_tasklet(
        "double",
        {"i": "0:N"},
        {"inp": dace.Memlet("a[i]")},
        "out = inp * 2.0",
        {"out": dace.Memlet("a[i]")},
        external_edges=True,
    )
    return sdfg


def conditionals(sdfg: dace.SDFG) -> list[ConditionalBlock]:
    return [node for node, parent in sdfg.all_nodes_recursive() if isinstance(node, ConditionalBlock)]


def guard_inside_map(sdfg: dace.SDFG) -> bool:
    """Whether every conditional of ``sdfg`` sits in the body of a map."""
    return all(cond.sdfg.parent_nsdfg_node is not None for cond in conditionals(sdfg))


def run(sdfg: dace.SDFG, **arguments: np.ndarray | int) -> dict[str, np.ndarray]:
    arrays = {name: value.copy() for name, value in arguments.items() if isinstance(value, np.ndarray)}
    symbols = {name: value for name, value in arguments.items() if not isinstance(value, np.ndarray)}
    sdfg(**arrays, **symbols)
    return arrays


def push_the_one_conditional(sdfg: dace.SDFG) -> bool:
    (cond,) = conditionals(sdfg)
    return push_if_into_maps(cond)


@pytest.mark.parametrize("k", [0, 3])
def test_a_symbol_guard_over_a_top_level_map_moves_into_its_body(k: int) -> None:
    reference = map_under_symbol_guard.to_sdfg(simplify=True)
    sut = copy.deepcopy(reference)
    a, b = np.random.rand(SIZE), np.full(SIZE, -1.0)

    pushed = push_the_one_conditional(sut)

    assert pushed
    sut.validate()
    assert [type(block) for block in sut.nodes()] == [dace.SDFGState]
    assert len(conditionals(sut)) == 1 and guard_inside_map(sut)
    expected, actual = run(reference, a=a, b=b, N=SIZE, K=k), run(sut, a=a, b=b, N=SIZE, K=k)
    np.testing.assert_array_equal(actual["b"], expected["b"])


@pytest.mark.parametrize("flag", [0, 1])
def test_a_guard_that_reads_data_takes_the_data_into_the_map(flag: int) -> None:
    reference = map_under_data_guard.to_sdfg(simplify=True)
    sut = copy.deepcopy(reference)
    a, b = np.random.rand(SIZE), np.full(SIZE, -1.0)
    flags = np.array([flag], dtype=np.int32)

    pushed = push_the_one_conditional(sut)

    assert pushed
    sut.validate()
    assert guard_inside_map(sut)
    expected = run(reference, flag=flags, a=a, b=b, N=SIZE)
    actual = run(sut, flag=flags, a=a, b=b, N=SIZE)
    np.testing.assert_array_equal(actual["b"], expected["b"])


def test_a_guard_in_a_loop_body_that_reads_the_loop_variable_moves_into_the_map() -> None:
    reference = map_under_guard_in_loop.to_sdfg(simplify=True)
    sut = copy.deepcopy(reference)
    a, b = np.random.rand(SIZE), np.zeros(SIZE)

    pushed = push_the_one_conditional(sut)

    assert pushed
    sut.validate()
    (loop,) = [b for b in sut.all_control_flow_blocks() if isinstance(b, LoopRegion)]
    assert not any(isinstance(block, ConditionalBlock) for block in loop.nodes())
    assert guard_inside_map(sut)
    expected, actual = run(reference, a=a, b=b, N=SIZE, K=2), run(sut, a=a, b=b, N=SIZE, K=2)
    np.testing.assert_allclose(actual["b"], expected["b"], rtol=1e-15)


@pytest.mark.parametrize(
    "build", [map_writes_what_its_guard_reads, lambda: map_under_guard_with_else.to_sdfg(simplify=True)]
)
def test_a_guard_whose_value_the_map_changes_or_that_has_an_else_stays_outside(build: Callable[[], dace.SDFG]) -> None:
    sut = build()
    before = sut.to_json()

    pushed = push_the_one_conditional(sut)

    assert not pushed
    assert sut.to_json() == before


@pytest.mark.parametrize("flag", [0, 1])
def test_a_map_whose_body_also_reads_the_guard_data_computes_the_same(flag: int) -> None:
    reference = map_under_data_guard.to_sdfg(simplify=True)
    (cond,) = conditionals(reference)
    (state,) = [s for s in cond.branches[0][1].nodes() if not s.is_empty()]
    (entry,) = [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]
    tasklet = next(n for n in state.all_nodes_between(entry, state.exit_node(entry)) if isinstance(n, nodes.Tasklet))
    tasklet.add_in_connector("f")
    state.add_memlet_path(state.add_read("flag"), entry, tasklet, memlet=dace.Memlet("flag[0]"), dst_conn="f")
    sut = copy.deepcopy(reference)
    a, b = np.random.rand(SIZE), np.full(SIZE, -1.0)
    flags = np.array([flag], dtype=np.int32)

    pushed = push_the_one_conditional(sut)

    assert pushed
    sut.validate()
    expected = run(reference, flag=flags, a=a, b=b, N=SIZE)
    actual = run(sut, flag=flags, a=a, b=b, N=SIZE)
    np.testing.assert_array_equal(actual["b"], expected["b"])
