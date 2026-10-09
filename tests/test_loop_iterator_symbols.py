# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A kernel left inside a sequential loop takes the loop's iterator as an argument. The iterator is defined by the
loop and declared nowhere, so stage 3 declares it on the program and stage 5 and 6 give it a value."""

import copy
import os

import dace
import numpy as np
import pytest

from nestforge.build.sdfg import compile_linked_program, generate_program
from nestforge.ir.loops import kernel_state, loop_symbol_values
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
    session.define_scopes()
    return session


def iterator_of(session: Session) -> str:
    (kernel,) = session.list_kernels()
    (iterator,) = [s for s in kernel["symbols"] if s.startswith("_loop_it")]
    return iterator


def test_the_iterator_of_the_loop_around_a_kernel_is_declared_on_the_program(tmp_path):
    session = scoped_session(tmp_path)

    iterator = iterator_of(session)

    assert iterator in session.sdfg.symbols
    assert session.sdfg.symbols[iterator] == dace.int64


def test_a_kernel_inside_a_loop_gets_the_middle_iterator_value_and_the_sizes_win_over_it(tmp_path):
    session = scoped_session(tmp_path)
    iterator = iterator_of(session)
    (ext,) = [session.kernel(k["name"]) for k in session.list_kernels()]

    values = loop_symbol_values(kernel_state(session.sdfg, ext), SIZES)

    assert values == {iterator: 3}, "range(1, 6) has 5 iterations and the middle one is t=3"
    assert session.kernel_sizes(ext.name) == {**SIZES, iterator: 3}
    session.sizes[iterator] = 4
    assert session.kernel_sizes(ext.name)[iterator] == 4


@pytest.mark.e2e
def test_the_program_with_the_loop_outside_the_kernel_builds_sweeps_and_matches_numpy(tmp_path):
    session = scoped_session(tmp_path)
    session.place()
    (kernel,) = session.list_kernels()
    session.optimize_kernel(kernel["name"])

    generated = generate_program(session.sdfg, tmp_path / "gen")
    swept = session.sweep(kernel["name"])

    assert f"{kernel['name']}(" in generated.source
    assert swept["winner"] is not None, "stage 6 found no configuration that matches the oracle"
    sdfg = copy.deepcopy(session.sdfg)
    sdfg.expand_library_nodes()
    compiled = compile_linked_program(sdfg, tmp_path / "parent")
    rng = np.random.default_rng(0)
    a, b = rng.random((SIZES["T"], SIZES["N"])), rng.random((SIZES["T"], SIZES["N"]))
    expected = a.copy()
    expected[1:] = a[0] + np.cumsum(b[1:], axis=0)
    compiled(A=a, B=b, T=SIZES["T"], N=SIZES["N"])
    np.testing.assert_allclose(a, expected, rtol=1e-12)
