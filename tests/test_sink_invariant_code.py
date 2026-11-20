# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for ``sink_into_loop`` and ``sink_into_map``: the statements that compute a named temporary before a loop or
map move into it, and run once per iteration."""

import copy

import dace
import numpy as np
import pytest
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion

from nestforge.stages.sinking import sink_into_loop, sink_into_map

N = dace.symbol("N")
SIZE = 7


@dace.program
def coefficients_before_a_loop(a: dace.float64[2], b: dace.float64[2], X: dace.float64[N], C: dace.float64[N]) -> None:
    t = np.ndarray([2], dace.float64)
    t[0] = a[0] * b[0]
    t[1] = a[1] + b[1]
    for i in range(1, N):
        C[i] = C[i - 1] * t[0] + X[i] * t[1]


@dace.program
def coefficients_before_a_map(a: dace.float64[2], b: dace.float64[2], X: dace.float64[N], C: dace.float64[N]) -> None:
    t = np.ndarray([2], dace.float64)
    t[0] = a[0] * b[0]
    t[1] = a[1] + b[1]
    for i in dace.map[0:N]:
        C[i] = X[i] * t[0] + t[1]


@dace.program
def loop_rewrites_a_coefficient_input(a: dace.float64[N], X: dace.float64[N], C: dace.float64[N]) -> None:
    t = np.ndarray([2], dace.float64)
    t[0] = a[0] * 2.0
    t[1] = a[1] + 1.0
    for i in range(1, N):
        a[i - 1] = X[i] * t[0] + t[1]


@dace.program
def coefficients_read_again_after_the_loop(
    a: dace.float64[2], b: dace.float64[2], X: dace.float64[N], C: dace.float64[N]
) -> None:
    t = np.ndarray([2], dace.float64)
    t[0] = a[0] * b[0]
    t[1] = a[1] + b[1]
    for i in range(1, N):
        C[i] = C[i - 1] * t[0] + X[i] * t[1]
    C[0] = t[0]


def loop_of(sdfg: dace.SDFG) -> LoopRegion:
    (loop,) = [b for b in sdfg.all_control_flow_blocks() if isinstance(b, LoopRegion)]
    return loop


def map_of(sdfg: dace.SDFG) -> tuple[dace.SDFGState, nodes.MapEntry]:
    (found,) = [(state, n) for state in sdfg.states() for n in state.nodes() if isinstance(n, nodes.MapEntry)]
    return found


def states_writing(sdfg: dace.SDFG, name: str) -> list[str]:
    """The labels of the states of ``sdfg`` (nested SDFGs included) that write ``name``."""
    return sorted(
        state.label
        for graph in sdfg.all_sdfgs_recursive()
        for state in graph.states()
        for node in state.data_nodes()
        if node.data == name and state.in_degree(node) > 0
    )


def arguments() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    return {"a": rng.random(2) + 0.5, "b": rng.random(2), "X": rng.random(SIZE), "C": np.ones(SIZE)}


def run(sdfg: dace.SDFG, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    copies = {name: value.copy() for name, value in arrays.items()}
    sdfg(**copies, N=SIZE)
    return copies


def test_coefficients_computed_before_a_loop_are_recomputed_at_the_start_of_every_iteration() -> None:
    reference = coefficients_before_a_loop.to_sdfg(simplify=True)
    sut = copy.deepcopy(reference)

    reason = sink_into_loop(loop_of(sut), "t")

    assert reason is None
    sut.validate()
    loop = loop_of(sut)
    assert loop.start_block.label.startswith("sunk_t")
    assert states_writing(sut, "t") == [loop.start_block.label]
    expected, actual = run(reference, arguments()), run(sut, arguments())
    np.testing.assert_allclose(actual["C"], expected["C"], rtol=1e-15)


def test_coefficients_computed_before_a_map_are_recomputed_in_its_body() -> None:
    reference = coefficients_before_a_map.to_sdfg(simplify=True)
    sut = copy.deepcopy(reference)
    state, entry = map_of(sut)

    reason = sink_into_map(state, entry, "t")

    assert reason is None
    sut.validate()
    state, entry = map_of(sut)
    assert all(e.data.data != "t" for e in state.in_edges(entry))
    (body,) = [n for n in state.scope_subgraph(entry).nodes() if isinstance(n, nodes.NestedSDFG)]
    assert body.sdfg.start_block.label.startswith("sunk_t")
    assert "t" not in sut.arrays
    expected, actual = run(reference, arguments()), run(sut, arguments())
    np.testing.assert_allclose(actual["C"], expected["C"], rtol=1e-15)


def test_statements_whose_input_the_loop_writes_stay_before_it() -> None:
    sut = loop_rewrites_a_coefficient_input.to_sdfg(simplify=True)
    before = sut.to_json()

    reason = sink_into_loop(loop_of(sut), "t")

    assert reason is not None and "writes a" in reason, reason
    assert sut.to_json() == before


def test_a_temporary_read_again_after_the_loop_stays_before_it() -> None:
    sut = coefficients_read_again_after_the_loop.to_sdfg(simplify=True)
    before = sut.to_json()

    reason = sink_into_loop(loop_of(sut), "t")

    assert reason is not None and "outside the loop" in reason, reason
    assert sut.to_json() == before


@pytest.mark.parametrize("name", ["a", "nothing"])
def test_an_argument_or_an_unknown_name_does_not_sink(name: str) -> None:
    sut = coefficients_before_a_loop.to_sdfg(simplify=True)
    before = sut.to_json()

    reason = sink_into_loop(loop_of(sut), name)

    assert reason is not None and name in reason, reason
    assert sut.to_json() == before
