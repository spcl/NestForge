# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``isolate_communication`` (``docs/design/comm-nodes.md``, section 2): canonicalization gives every communication row a
state of its own, cut in the order the data flow fixes, and refuses a row that shares a state with work nothing orders
it against."""

import dace
import pytest
from dace.transformation.interstate import StateFusion
from mpi4py import MPI

from tactics.ir.comm import CommClass, comm_class, isolate_communication, isolation_refusal, skeleton
from tactics.stages.canonicalize import Targets, canonicalize

N = dace.symbol("N", dtype=dace.int64)


@dace.program
def normalized(a: dace.float64[N], r: dace.float64[1]) -> None:
    r[0] = 0.0
    for i in dace.map[0:N]:
        r[0] += a[i]
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, r, op=MPI.SUM)
    for i in dace.map[0:N]:
        a[i] = a[i] / r[0]


@dace.program
def beside(a: dace.float64[N], c: dace.float64[N], r: dace.float64[1]) -> None:
    r[0] = 0.0
    for i in dace.map[0:N]:
        r[0] += a[i]
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, r, op=MPI.SUM)
    for i in dace.map[0:N]:
        c[i] = c[i] * 2.0


def canonical(program: dace.frontend.python.parser.DaceProgram) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=True)
    canonicalize(sdfg, Targets())
    return sdfg


def row_state(sdfg: dace.SDFG) -> dace.SDFGState:
    return next(s for s in sdfg.states() if any(comm_class(n) is CommClass.BLOCKING for n in s.nodes()))


def fused_with(sdfg: dace.SDFG, neighbor: dace.SDFGState, row: dace.SDFGState) -> None:
    """Fuse the row's state with the state before or after it, as a careless pass could."""
    first, second = (neighbor, row) if sdfg.edges_between(neighbor, row) else (row, neighbor)
    StateFusion.apply_to(sdfg, verify=False, first_state=first, second_state=second)


@pytest.mark.parametrize("side", ["before", "after"])
def test_a_row_fused_with_the_work_it_orders_against_is_cut_out_again(side: str) -> None:
    sdfg = canonical(normalized)
    kept = skeleton(sdfg)
    row = row_state(sdfg)
    edges = sdfg.in_edges(row) if side == "before" else sdfg.out_edges(row)
    fused_with(sdfg, edges[0].src if side == "before" else edges[0].dst, row)
    assert isolation_refusal(sdfg) is not None, "the fusion put the row into one state with a doall"

    split = isolate_communication(sdfg)

    assert split >= 1
    assert isolation_refusal(sdfg) is None
    assert str(skeleton(sdfg)) == str(kept)
    sdfg.validate()


def test_a_row_and_work_with_no_data_flow_between_them_are_refused() -> None:
    sdfg = canonical(beside)
    row = row_state(sdfg)
    fused_with(sdfg, sdfg.out_edges(row)[0].dst, row)

    with pytest.raises(ValueError, match="no data flow between them"):
        isolate_communication(sdfg)


def test_a_canonical_program_is_already_isolated() -> None:
    sdfg = canonical(normalized)

    assert isolate_communication(sdfg) == 0
