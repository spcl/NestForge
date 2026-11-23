# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Canonicalization keeps the communication skeleton of every fixture (docs/design/comm-nodes.md, package 1): the
communication nodes in their order, each in as many loops as before and none inside a doall. Needs DaCe extended
6ed7dc19e6 or later, which keeps nodes with ordered side effects in their loops."""

import dace
import pytest
from comm_programs import PROGRAMS
from dace.libraries.comm.nodes.node import CommNode
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion

from tactics.session import Session

#: The communication nodes that only read the communicator; the skeleton leaves them out.
QUERIES = frozenset({"CommRank", "CommSize", "CommF2c"})


def inside_a_doall(state: dace.SDFGState, node: nodes.Node) -> bool:
    """Whether a map encloses ``node``, in its state or around the nested SDFGs that hold it."""
    while state.entry_node(node) is None:
        nest = state.sdfg.parent_nsdfg_node
        if nest is None:
            return False
        node, state = nest, state.sdfg.parent
    return True


def loop_depth(state: dace.SDFGState) -> int:
    """How many loops hold ``state``, through nested SDFGs."""
    depth, graph = 0, state.parent_graph
    while graph is not None:
        depth += isinstance(graph, LoopRegion)
        graph = graph.parent if graph is graph.sdfg else graph.parent_graph
    return depth


def skeleton(sdfg: dace.SDFG) -> list[tuple[str, int, bool]]:
    """Every communication node but the queries, in program order: its type, its loop depth, whether in a doall."""
    return [
        (type(node).__name__, loop_depth(parent), inside_a_doall(parent, node))
        for node, parent in sdfg.all_nodes_recursive()
        if isinstance(node, CommNode) and type(node).__name__ not in QUERIES
    ]


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_canonicalization_keeps_the_communication_skeleton(name: str) -> None:
    sdfg = PROGRAMS[name].to_sdfg(simplify=True)
    before = skeleton(sdfg)
    assert before, f"{name} communicates"
    sut = Session(sdfg)

    sut.canonicalize()

    assert sut.sdfg is not None
    assert skeleton(sut.sdfg) == before
