# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``hoist-invariant-code``: the agent names a loop or a doall, LoopInvariantCodeMotion
picks the statements no iteration changes and runs them once, before it."""

import pathlib

import dace
import numpy as np
from dace import Memlet
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion

from tactics.records import MoveStatus
from tactics.session import Session

N = dace.symbol("N", dtype=dace.int64)
SIZE = 6


def product_in_a_loop(rewrites_a: bool) -> dace.SDFG:
    """``for i: t = a[0] * b[0]; C[i] = t * X[i]``, and ``a[0] = X[i]`` first when ``rewrites_a``."""
    sdfg = dace.SDFG("product_in_a_loop" + ("_rewrites_a" if rewrites_a else ""))
    for name, shape in (("a", [1]), ("b", [1]), ("X", [N]), ("C", [N])):
        sdfg.add_array(name, shape, dace.float64)
    sdfg.add_transient("t", [1], dace.float64)
    loop = LoopRegion("loop", "i < N", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    a = body.add_access("a")
    if rewrites_a:
        store = body.add_tasklet("store", {"x": None}, {"o": None}, "o = x")
        body.add_edge(body.add_read("X"), None, store, "x", Memlet("X[i]"))
        body.add_edge(store, "o", a, None, Memlet("a[0]"))
    product = body.add_tasklet("product", {"p": None, "q": None}, {"o": None}, "o = p * q")
    t = body.add_access("t")
    body.add_edge(a, None, product, "p", Memlet("a[0]"))
    body.add_edge(body.add_read("b"), None, product, "q", Memlet("b[0]"))
    body.add_edge(product, "o", t, None, Memlet("t[0]"))
    scale = body.add_tasklet("scale", {"s": None, "x": None}, {"o": None}, "o = s * x")
    body.add_edge(t, None, scale, "s", Memlet("t[0]"))
    body.add_edge(body.add_read("X"), None, scale, "x", Memlet("X[i]"))
    body.add_edge(scale, "o", body.add_write("C"), None, Memlet("C[i]"))
    sdfg.validate()
    return sdfg


def product_in_a_doall() -> dace.SDFG:
    """``doall i: t = a[0] * b[0]; C[i] = t * X[i]``."""
    sdfg = dace.SDFG("product_in_a_doall")
    for name, shape in (("a", [1]), ("b", [1]), ("X", [N]), ("C", [N])):
        sdfg.add_array(name, shape, dace.float64)
    sdfg.add_transient("t", [1], dace.float64)
    state = sdfg.add_state("body", is_start_block=True)
    entry, exit_ = state.add_map("scale", {"i": "0:N"})
    product = state.add_tasklet("product", {"p": None, "q": None}, {"o": None}, "o = p * q")
    t = state.add_access("t")
    scale = state.add_tasklet("scale", {"s": None, "x": None}, {"o": None}, "o = s * x")
    state.add_memlet_path(state.add_read("a"), entry, product, dst_conn="p", memlet=Memlet("a[0]"))
    state.add_memlet_path(state.add_read("b"), entry, product, dst_conn="q", memlet=Memlet("b[0]"))
    state.add_edge(product, "o", t, None, Memlet("t[0]"))
    state.add_edge(t, None, scale, "s", Memlet("t[0]"))
    state.add_memlet_path(state.add_read("X"), entry, scale, dst_conn="x", memlet=Memlet("X[i]"))
    state.add_memlet_path(scale, exit_, state.add_write("C"), src_conn="o", memlet=Memlet("C[i]"))
    sdfg.validate()
    return sdfg


def labels_of(sut: Session, kind: type) -> list[str]:
    return [label for label, (obj, _) in sut.row_index().items() if isinstance(obj, kind)]


def product_runs_in(sdfg: dace.SDFG) -> str:
    """Where the ``product`` tasklet is: inside the loop, inside the doall, or outside both."""
    for node, graph in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.Tasklet) and node.label.startswith("product"):
            if isinstance(graph, dace.SDFGState) and isinstance(graph.parent_graph, LoopRegion):
                return "loop"
            in_body = graph.sdfg.parent_nsdfg_node is not None  # a doall's body nested as an SDFG
            return "doall" if in_body or graph.entry_node(node) is not None else "outside"
    raise AssertionError("the product tasklet is gone")


def run(sdfg: dace.SDFG) -> np.ndarray:
    a, b, x, c = np.array([1.5]), np.array([-2.0]), np.arange(1.0, SIZE + 1), np.zeros(SIZE)
    sdfg(a=a, b=b, X=x, C=c, N=SIZE)
    return c


def test_a_product_of_loop_invariant_inputs_runs_once_before_the_loop(tmp_path: pathlib.Path) -> None:
    sut = Session(product_in_a_loop(rewrites_a=False), work_dir=str(tmp_path))
    (loop,) = labels_of(sut, LoopRegion)

    (move,) = sut.list_moves("hoist-invariant-code")
    result = sut.apply_move(move.kind, move.labels)

    assert move.labels == (loop,)
    assert (result.status, result.reason) == (MoveStatus.APPLIED, "LoopInvariantCodeMotion"), result
    assert product_runs_in(sut.sdfg) == "outside"
    assert sut.list_moves("hoist-invariant-code") == [], "nothing invariant is left in the loop"
    np.testing.assert_allclose(run(sut.sdfg), -3.0 * np.arange(1.0, SIZE + 1), rtol=1e-15)


def test_a_loop_that_rewrites_an_input_of_the_product_keeps_it_and_refuses_the_hoist(tmp_path: pathlib.Path) -> None:
    sut = Session(product_in_a_loop(rewrites_a=True), work_dir=str(tmp_path))
    (loop,) = labels_of(sut, LoopRegion)

    refused = sut.apply_move("hoist-invariant-code", [loop])

    assert sut.list_moves("hoist-invariant-code") == []
    assert refused.status is MoveStatus.ILLEGAL and "loop-invariant" in refused.reason, refused
    assert product_runs_in(sut.sdfg) == "loop"


def test_a_product_of_doall_invariant_inputs_runs_once_before_the_doall(tmp_path: pathlib.Path) -> None:
    sut = Session(product_in_a_doall(), work_dir=str(tmp_path))

    (move,) = sut.list_moves("hoist-invariant-code")
    result = sut.apply_move(move.kind, move.labels)

    assert (result.status, result.reason) == (MoveStatus.APPLIED, "LoopInvariantCodeMotion"), result
    assert product_runs_in(sut.sdfg) == "outside"
    assert sut.list_moves("hoist-invariant-code") == []
    np.testing.assert_allclose(run(sut.sdfg), -3.0 * np.arange(1.0, SIZE + 1), rtol=1e-15)


def test_a_doall_without_invariant_statements_is_not_offered(tmp_path: pathlib.Path) -> None:
    sut = Session(product_in_a_doall(), work_dir=str(tmp_path))
    (move,) = sut.list_moves("hoist-invariant-code")
    sut.apply_move(move.kind, move.labels)
    (doall,) = labels_of(sut, nodes.MapEntry)

    refused = sut.apply_move("hoist-invariant-code", [doall])

    assert refused.status is MoveStatus.ILLEGAL and "invariant" in refused.reason, refused


def test_a_hoisted_product_sinks_back_into_the_loop_and_runs_every_iteration(tmp_path: pathlib.Path) -> None:
    sut = Session(product_in_a_loop(rewrites_a=False), work_dir=str(tmp_path))
    (hoist,) = sut.list_moves("hoist-invariant-code")
    sut.apply_move(hoist.kind, hoist.labels)
    (move,) = sut.list_moves("sink-invariant-code")

    result = sut.apply_move(move.kind, move.labels, container=move.container)

    assert (result.status, result.reason) == (MoveStatus.APPLIED, "sink_into_loop"), result
    assert product_runs_in(sut.sdfg) == "loop"
    np.testing.assert_allclose(run(sut.sdfg), -3.0 * np.arange(1.0, SIZE + 1), rtol=1e-15)


def test_a_hoisted_product_sinks_back_into_the_doall(tmp_path: pathlib.Path) -> None:
    sut = Session(product_in_a_doall(), work_dir=str(tmp_path))
    (hoist,) = sut.list_moves("hoist-invariant-code")
    sut.apply_move(hoist.kind, hoist.labels)
    (move,) = sut.list_moves("sink-invariant-code")

    result = sut.apply_move(move.kind, move.labels, container=move.container)

    assert (result.status, result.reason) == (MoveStatus.APPLIED, "sink_into_map"), result
    assert product_runs_in(sut.sdfg) == "doall"
    np.testing.assert_allclose(run(sut.sdfg), -3.0 * np.arange(1.0, SIZE + 1), rtol=1e-15)


def test_sinking_without_a_container_or_with_an_argument_is_refused(tmp_path: pathlib.Path) -> None:
    sut = Session(product_in_a_loop(rewrites_a=False), work_dir=str(tmp_path))
    (loop,) = labels_of(sut, LoopRegion)

    unnamed = sut.apply_move("sink-invariant-code", [loop])
    argument = sut.apply_move("sink-invariant-code", [loop], container="X")

    assert unnamed.status is MoveStatus.ILLEGAL and "container=" in unnamed.reason, unnamed
    assert argument.status is MoveStatus.ILLEGAL and "argument" in argument.reason, argument
    assert product_runs_in(sut.sdfg) == "loop"
