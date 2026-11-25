# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Effects by subset (design 12.1): a crossing with a communication row conflicts only where both sides touch a
common element, so a wait crosses the interior update of the array whose ghost cells it receives, and not the
boundary update that reads them."""

import dace
from comm_rank_programs import halo_in_place, halo_stencil
from dace.libraries.comm.nodes import Waitall
from dace.sdfg import nodes

from tactics.ir.comm import CommClass, comm_class
from tactics.records import MoveStatus
from tactics.session import Session
from tactics.stages.lib_motion import accessed, plan_lib_motion
from tactics.stages.moves import SPLIT_DOALL


def canonical_halo() -> tuple[dace.SDFG, tuple[nodes.Node, dace.SDFGState]]:
    session = Session(halo_in_place.to_sdfg(simplify=True))
    session.canonicalize()
    (wait,) = [(n, s) for n, s in session.sdfg.all_nodes_recursive() if isinstance(n, Waitall)]
    return session.sdfg, wait


def neighbors(
    sdfg: dace.SDFG, wait: tuple[nodes.Node, dace.SDFGState]
) -> tuple[list[dace.SDFGState], list[dace.SDFGState]]:
    """The states before the wait that read ``ap`` and run no communication, and the states after it that read ``ap``."""
    order = [s for s in sdfg.bfs_nodes(sdfg.start_block) if isinstance(s, dace.SDFGState)]
    at = order.index(wait[1])
    computes = [s for s in order if "ap" in accessed(s)[0] and not any(comm_class(n) is not None for n in s.nodes())]
    return [s for s in computes if order.index(s) < at], [s for s in computes if order.index(s) > at]


def test_canonicalization_leaves_the_halo_buffers_as_subsets_of_ap() -> None:
    sdfg, _ = canonical_halo()

    rows = [(n, s) for n, s in sdfg.all_nodes_recursive() if comm_class(n) is CommClass.START]

    assert len(rows) == 4
    for node, state in rows:
        buffers = [e for e in state.all_edges(node) if e.data.data == "ap"]
        assert buffers, f"{node.label} sends or receives ap itself, not a view of it"


def test_the_wait_crosses_the_interior_update_but_not_the_boundary_one_and_names_both_subsets() -> None:
    sdfg, wait = canonical_halo()
    interior, _ = neighbors(sdfg, wait)
    assert interior, "the interior update runs between the starts and the wait"
    plan = plan_lib_motion(wait, (interior[0], None), "before")
    assert not isinstance(plan, str), plan

    _, boundary = neighbors(sdfg, wait)
    assert boundary, "the boundary update reads the ghost cells after the wait"
    refused = plan_lib_motion(wait, (boundary[-1], None), "after")
    assert isinstance(refused, str) and "reads ap[" in refused and "(the call: ap[" in refused, refused


def canonical_stencil() -> Session:
    session = Session(halo_stencil.to_sdfg(simplify=True))
    session.canonicalize()
    return session


def test_split_doall_is_listed_at_both_received_rows_and_two_cuts_leave_an_interior_with_plain_bounds() -> None:
    session = canonical_stencil()
    listed = session.list_moves(SPLIT_DOALL)
    assert sorted(tuple(move.at.values()) for move in listed) == [("2",), ("N",)], listed

    first = session.list_moves(SPLIT_DOALL)[0]
    assert session.apply_move(SPLIT_DOALL, first.labels, at=first.at).status is MoveStatus.APPLIED
    (second,) = [move for move in session.list_moves(SPLIT_DOALL) if move.at != first.at]
    assert session.apply_move(SPLIT_DOALL, second.labels, at=second.at).status is MoveStatus.APPLIED
    bounds = [
        str(bound)
        for node, _ in session.row_index().values()
        if isinstance(node, nodes.MapEntry)
        for bound in node.map.range[0][:2]
    ]
    assert "2" in bounds and "N - 1" in bounds, bounds
