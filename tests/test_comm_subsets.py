# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Effects by subset (design 12.1): a crossing with a communication row conflicts only where both sides touch a
common element, so a wait crosses the interior update of the array whose ghost cells it receives, and not the
boundary update that reads them."""

import dace
from comm_rank_programs import halo_in_place
from dace.libraries.comm.nodes import Waitall
from dace.sdfg import nodes

from nestforge.ir.comm import CommClass, comm_class
from nestforge.session import Session
from nestforge.stages.lib_motion import accessed, plan_lib_motion


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


def test_the_wait_crosses_the_interior_update_that_reads_no_ghost_cell() -> None:
    sdfg, wait = canonical_halo()
    interior, _ = neighbors(sdfg, wait)
    assert interior, "the interior update runs between the starts and the wait"

    plan = plan_lib_motion(wait, (interior[0], None), "before")

    assert not isinstance(plan, str), plan


def test_the_wait_does_not_cross_the_boundary_update_and_the_reason_names_both_subsets() -> None:
    sdfg, wait = canonical_halo()
    _, boundary = neighbors(sdfg, wait)
    assert boundary, "the boundary update reads the ghost cells after the wait"

    refused = plan_lib_motion(wait, (boundary[-1], None), "after")

    assert isinstance(refused, str) and "reads ap[" in refused and "(the call: ap[" in refused, refused
