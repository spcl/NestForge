# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The gate of an authored move (``tactics/stages/gate.py``): on its cluster's examples, a move that does what the
requests asked passes, a move that computes something else is caught on the numbers, and a move that never matches
the examples is caught as not offered. Without the gate an authored move enters the catalogue on its author's word."""

from pathlib import Path

from tactics.stages.authored import load_move
from tactics.stages.gate import example_failures
from tactics.stages.requests import Example

EXPAND = """
from dace.transformation.dataflow import MapExpansion

TRANSFORMATION = MapExpansion
ROWS = ("map_entry",)
TEXT = "a doall over several dimensions becomes one doall per dimension, nested"
"""

#: A move that claims to expand a doall but also changes what it computes.
WRONG = """
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.transformation.dataflow import MapExpansion


class ExpandAndScale(MapExpansion):
    def apply(self, graph, sdfg):
        for node in graph.scope_subgraph(self.map_entry).nodes():
            inner = [n for n, _ in node.sdfg.all_nodes_recursive()] if isinstance(node, nodes.NestedSDFG) else [node]
            for tasklet in inner:
                if isinstance(tasklet, nodes.Tasklet):
                    tasklet.code = CodeBlock(tasklet.code.as_string.replace("2.0", "3.0"))
        return super().apply(graph, sdfg)


TRANSFORMATION = ExpandAndScale
ROWS = ("map_entry",)
TEXT = "a doall over several dimensions becomes one doall per dimension, nested"
"""

BEFORE = """
N = dace.symbol("N")


@dace.program
def scaled(a: dace.float64[N, N], b: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        b[i, j] = a[i, j] * 2.0
"""

AFTER = """
N = dace.symbol("N")


@dace.program
def scaled(a: dace.float64[N, N], b: dace.float64[N, N]):
    for i in dace.map[0:N]:
        for j in dace.map[0:N]:
            b[i, j] = a[i, j] * 2.0
"""

ONE_DIMENSION = """
N = dace.symbol("N")


@dace.program
def shifted(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i] + 1.0
"""


def entry(tmp_path: Path, kind: str, source: str) -> Path:
    folder = tmp_path / kind
    folder.mkdir()
    (folder / "move.py").write_text(source)
    return folder


def test_a_move_that_does_what_the_examples_ask_passes(tmp_path: Path) -> None:
    move = load_move(entry(tmp_path, "expand-doall", EXPAND))

    assert example_failures(move, [Example(before=BEFORE, after=AFTER)], tmp_path) == []


def test_a_move_that_computes_something_else_is_caught(tmp_path: Path) -> None:
    move = load_move(entry(tmp_path, "expand-and-scale", WRONG))

    failures = example_failures(move, [Example(before=BEFORE, after=AFTER)], tmp_path)

    assert failures == ["example 0: the move leaves values in b that after does not"]


def test_a_move_the_examples_never_offer_is_caught(tmp_path: Path) -> None:
    move = load_move(entry(tmp_path, "expand-doall", EXPAND))

    failures = example_failures(move, [Example(before=ONE_DIMENSION, after=ONE_DIMENSION)], tmp_path)

    assert failures == ["example 0: expand-doall is not offered on before"]
