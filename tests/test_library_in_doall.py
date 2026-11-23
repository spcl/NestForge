# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A library call inside a doall is offered no device library: each iteration of a parallel doall cannot launch its own
cuBLAS. The same call outside any doall is."""

import dace
import pytest
from dace.sdfg import nodes

from tactics.stages import expand
from tactics.stages.expand import implementations, in_doall, is_device_implementation

N = dace.symbol("N", dtype=dace.int64)
M = dace.symbol("M", dtype=dace.int64)


@dace.program
def rows_times_matrix(A: dace.float64[N, M], B: dace.float64[M, M], C: dace.float64[N, M]) -> None:
    for i in dace.map[0:N]:
        C[i] = A[i] @ B


@dace.program
def matrix_times_matrix(A: dace.float64[N, M], B: dace.float64[M, M], C: dace.float64[N, M]) -> None:
    C[:] = A @ B


def library_call(program: dace.frontend.python.parser.DaceProgram) -> tuple[nodes.LibraryNode, dace.SDFGState]:
    """The program's matrix product, specialized (``MatMul`` to ``Gemm`` or ``Gemv``) as canonicalization does."""
    sdfg = program.to_sdfg(simplify=True)
    (generic, state) = next((n, s) for n, s in sdfg.all_nodes_recursive() if isinstance(n, nodes.LibraryNode))
    generic.expand(state)
    (found,) = [(n, s) for n, s in sdfg.all_nodes_recursive() if isinstance(n, nodes.LibraryNode)]
    return found


@pytest.fixture
def every_library_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """This host has no GPU library; the offer is a decision about the call's place, not about the host."""
    monkeypatch.setattr(expand, "installed", lambda expansion: True)


def test_a_call_inside_a_doall_is_offered_no_device_library(every_library_installed: None) -> None:
    node, state = library_call(rows_times_matrix)

    inside = in_doall(node, state)

    assert inside, "the call sits in the doall's body"
    assert not any(is_device_implementation(i) for i in implementations(node, gpu=True, inside_doall=inside))


def test_a_call_outside_any_doall_is_offered_the_device_library(every_library_installed: None) -> None:
    node, state = library_call(matrix_times_matrix)

    inside = in_doall(node, state)

    assert not inside
    assert any(is_device_implementation(i) for i in implementations(node, gpu=True, inside_doall=inside))
