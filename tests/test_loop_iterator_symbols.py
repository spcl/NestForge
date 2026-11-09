# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A kernel left inside a sequential loop takes the loop's iterator as an argument. The iterator is a scope symbol
of the loop, never declared on the program: the kernel call reads it where it is defined, and stages 5 and 6 give
it a value."""

import copy
import os

import dace
import numpy as np
import pytest
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.analysis import loop_analysis

from nestforge.build.sdfg import compile_linked_program, generate_program
from nestforge.ir.extract import find_state_of_node
from nestforge.ir.loops import enclosing_loops, loop_symbol_values
from nestforge.session import Session
from nestforge.stages.canonicalize import Targets

T = dace.symbol("T", dtype=dace.int64)
N = dace.symbol("N", dtype=dace.int64)
SIZES = {"T": 6, "N": 5}


@dace.program
def carried_sweep(A: dace.float64[T, N], B: dace.float64[T, N]):
    for t in range(1, T):
        for i in dace.map[0:N]:
            A[t, i] = A[t - 1, i] + B[t, i]


def scoped_session(tmp_path) -> Session:
    sdfg = carried_sweep.to_sdfg(simplify=True)
    sdfg.name = f"{sdfg.name}_{os.getpid()}"  # one build folder per xdist worker
    session = Session(sdfg, Targets(gpu=False), work_dir=str(tmp_path), sizes=SIZES)
    session.canonicalize()
    session.default_scopes()
    return session


def iterator_of(session: Session) -> str:
    (kernel,) = session.list_kernels()
    (iterator,) = [s for s in kernel.symbols if s.startswith("_loop_it")]
    return iterator


def test_the_iterator_of_the_loop_around_a_kernel_stays_a_scope_symbol_of_the_loop(tmp_path):
    session = scoped_session(tmp_path)

    iterator = iterator_of(session)
    (ext,) = [session.kernel(k.name) for k in session.list_kernels()]
    state = find_state_of_node(session.sdfg, ext)

    assert iterator not in session.sdfg.symbols, "a loop iterator is defined by its loop, not by the program"
    assert iterator in state.symbols_defined_at(ext)


def test_a_kernel_inside_a_loop_gets_the_middle_iterator_value_and_the_sizes_win_over_it(tmp_path):
    session = scoped_session(tmp_path)
    iterator = iterator_of(session)
    (ext,) = [session.kernel(k.name) for k in session.list_kernels()]

    values = loop_symbol_values(find_state_of_node(session.sdfg, ext), SIZES)

    assert values == {iterator: 3}, "range(1, 6) has 5 iterations and the middle one is t=3"
    assert session.kernel_sizes(ext.name) == {**SIZES, iterator: 3}
    session.sizes[iterator] = 4
    assert session.kernel_sizes(ext.name)[iterator] == 4


@pytest.mark.e2e
def test_the_program_with_the_loop_outside_the_kernel_builds_sweeps_and_matches_numpy(tmp_path):
    session = scoped_session(tmp_path)
    session.place()
    (kernel,) = session.list_kernels()
    session.optimize_kernel(kernel.name)

    generated = generate_program(session.sdfg, tmp_path / "gen")
    swept = session.sweep(kernel.name)

    assert f"{kernel.name}(" in generated.source
    assert swept.winner is not None, "stage 6 found no configuration that matches the oracle"
    sdfg = copy.deepcopy(session.sdfg)
    sdfg.expand_library_nodes()
    compiled = compile_linked_program(sdfg, tmp_path / "parent")
    rng = np.random.default_rng(0)
    a, b = rng.random((SIZES["T"], SIZES["N"])), rng.random((SIZES["T"], SIZES["N"]))
    expected = a.copy()
    expected[1:] = a[0] + np.cumsum(b[1:], axis=0)
    compiled(A=a, B=b, T=SIZES["T"], N=SIZES["N"])
    np.testing.assert_allclose(a, expected, rtol=1e-12)


def counted_loop(label: str, iterator: str) -> LoopRegion:
    return LoopRegion(label, f"{iterator} < 4", iterator, f"{iterator} = 0", f"{iterator} = {iterator} + 1")


def test_the_loops_outside_a_nested_sdfg_enclose_a_state_inside_it_outermost_first():
    outer = dace.SDFG("outer_of_nested_loops")
    outer_loop = counted_loop("outer_loop", "i")
    outer.add_node(outer_loop, is_start_block=True)
    holder = outer_loop.add_state("holder", is_start_block=True)
    inner = dace.SDFG("inner_of_nested_loops")
    inner_loop = counted_loop("inner_loop", "j")
    inner.add_node(inner_loop, is_start_block=True)
    deepest = inner_loop.add_state("deepest", is_start_block=True)
    holder.add_nested_sdfg(inner, set(), set())

    assert enclosing_loops(deepest) == [outer_loop, inner_loop]
    assert enclosing_loops(holder) == [outer_loop]


@dace.program
def runtime_bound_sweep(A: dace.float64[T, N], B: dace.float64[T, N], K: dace.int64[1]):
    steps = K[0]
    for t in range(1, steps):
        for i in dace.map[0:N]:
            A[t, i] = A[t - 1, i] + B[t, i]


def test_a_loop_bounded_by_a_symbol_the_program_computes_leaves_its_iterator_without_a_value(tmp_path):
    sdfg = runtime_bound_sweep.to_sdfg(simplify=True)
    sdfg.name = f"{sdfg.name}_{os.getpid()}"
    session = Session(sdfg, Targets(gpu=False), work_dir=str(tmp_path), sizes=SIZES)
    session.canonicalize()
    session.default_scopes()
    iterator = iterator_of(session)
    (kernel,) = session.list_kernels()
    state = find_state_of_node(session.sdfg, session.kernel(kernel.name))

    (loop,) = enclosing_loops(state)
    assert "steps" in {str(s) for s in dace.symbolic.pystr_to_symbolic(loop_bound(loop)).free_symbols}
    assert loop_symbol_values(state, SIZES) == {}, "the bound has no value at the sizes, so the loop gets none"
    assert session.kernel_sizes(kernel.name) == SIZES
    assert iterator in kernel.symbols and iterator not in session.kernel_sizes(kernel.name), "the unchecked path"


def loop_bound(loop: LoopRegion) -> str:
    return str(loop_analysis.get_loop_end(loop))


def test_a_loop_inside_a_loop_without_a_value_gets_none_while_one_outside_keeps_its_value():
    graph = dace.SDFG("loops_without_values")
    known = LoopRegion("known", "i < 8", "i", "i = 0", "i = i + 1")
    graph.add_node(known, is_start_block=True)
    unknown = LoopRegion("unknown", "j < m", "j", "j = 0", "j = j + 1")
    known.add_node(unknown, is_start_block=True)
    depends = LoopRegion("depends", "k < j", "k", "k = 0", "k = k + 1")
    unknown.add_node(depends, is_start_block=True)
    inside = depends.add_state("inside", is_start_block=True)

    assert loop_symbol_values(inside, SIZES) == {"i": 3}, "range(0, 8) has 8 iterations and the middle one is i=3"
