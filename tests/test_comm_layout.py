# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The layout rule of a program that communicates (``docs/design/comm-nodes.md``, section 6): the bench splits some
arguments over the ranks along their own axes (its spec's ``mpi.arrays``), so ``relayout`` refuses them and does not
list them. A permuted argument would reach every rank split along the wrong axis. The buffers of a communication row
are relayout's to pack (``pack_mpi_buffers``, ``tests/test_comm_ranks.py``)."""

import dace
import pytest

from tactics.server import distributed_arguments
from tactics.stages.layout import layout_operation, listed_operation

N = dace.symbol("N")


@dace.program
def two_arrays(a: dace.float64[N, N], b: dace.float64[N, N]) -> None:
    for i, j in dace.map[0:N, 0:N]:
        b[i, j] = a[j, i]


TRANSPOSE = {"action": "permute", "array": "a", "order": ["i1", "i0"]}


def test_a_distributed_argument_is_refused_and_not_listed() -> None:
    sdfg = two_arrays.to_sdfg(simplify=True)

    with pytest.raises(ValueError, match="the bench splits a over the ranks"):
        layout_operation(TRANSPOSE, sdfg, frozenset({"a"}))
    assert layout_operation(TRANSPOSE, sdfg).action.value == "permute", "an argument no rank splits is laid out"
    listed = listed_operation(sdfg, frozenset({"a"}))
    assert listed is not None and listed["array"] == "b"


def test_the_distributed_arguments_come_from_the_bench_spec() -> None:
    assert distributed_arguments("dist_sync_batchnorm") == frozenset({"x", "out"})
    assert distributed_arguments("hmm_forward") == frozenset()
