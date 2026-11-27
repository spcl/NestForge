# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The ``doall-fission`` move on a map whose body is one nested SDFG with two independent statements."""

import dace
import numpy as np
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState
from dace.transformation.helpers import nest_state_subgraph

from tactics.records import MoveStatus
from tactics.session import Session

N = dace.symbol("N")
f64 = dace.float64


def two_statement_map() -> tuple[dace.SDFG, SDFGState, nodes.MapEntry]:
    """One flat map computing ``b = a + 1`` and ``c = a * 2`` -- both statements under one map."""
    sdfg = dace.SDFG("multi_statement_map")
    sdfg.add_array("a", [N], f64)
    sdfg.add_array("b", [N], f64)
    sdfg.add_array("c", [N], f64)
    state = sdfg.add_state()

    rnode = state.add_read("a")
    me, mx = state.add_map("outer", dict(i="0:N"))
    t1 = state.add_tasklet("one", {"x"}, {"y"}, "y = x + 1.0")
    t2 = state.add_tasklet("two", {"x"}, {"y"}, "y = x * 2.0")
    state.add_memlet_path(rnode, me, t1, memlet=dace.Memlet(data="a", subset="i"), dst_conn="x")
    state.add_memlet_path(t1, mx, state.add_write("b"), memlet=dace.Memlet(data="b", subset="i"), src_conn="y")
    state.add_memlet_path(rnode, me, t2, memlet=dace.Memlet(data="a", subset="i"), dst_conn="x")
    state.add_memlet_path(t2, mx, state.add_write("c"), memlet=dace.Memlet(data="c", subset="i"), src_conn="y")
    sdfg.validate()
    return sdfg, state, me


def multi_statement_map_sdfg() -> dace.SDFG:
    """``two_statement_map`` with its body wrapped into a NestedSDFG -- MapFission's map-with-nested-SDFG
    pattern, the shape the ``doall-fission`` move takes."""
    sdfg, state, me = two_statement_map()
    nest_state_subgraph(sdfg, state, state.scope_subgraph(me, include_entry=False, include_exit=False))
    sdfg.validate()
    return sdfg


def all_map_entries(sdfg: dace.SDFG) -> list[nodes.MapEntry]:
    return [
        n
        for sd in sdfg.all_sdfgs_recursive()
        for state in sd.states()
        for n in state.nodes()
        if isinstance(n, nodes.MapEntry)
    ]


def tasklet_scopes(sdfg: dace.SDFG) -> dict[str, nodes.Node | None]:
    """Tasklet label -> its innermost enclosing MapEntry, through the nested SDFG a map body is (``None`` if
    unmapped)."""
    scopes: dict[str, nodes.Node | None] = {}
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.states():
            for node in state.nodes():
                if isinstance(node, nodes.Tasklet):
                    scopes[node.label] = enclosing_map(state, node)
    return scopes


def enclosing_map(state: dace.SDFGState, node: nodes.Node) -> nodes.Node | None:
    scope = state.entry_node(node)
    nest = state.sdfg.parent_nsdfg_node
    if scope is None and nest is not None:
        return enclosing_map(state.sdfg.parent, nest)
    return scope


def test_two_statement_map_starts_as_one_map_two_statements() -> None:
    sdfg, state, me = two_statement_map()
    assert len(all_map_entries(sdfg)) == 1
    scope = state.scope_dict()
    tasklets = {n.label: n for n in state.nodes() if isinstance(n, nodes.Tasklet)}
    assert scope[tasklets["one"]] is me and scope[tasklets["two"]] is me


def test_map_fission_is_listed_for_the_multi_output_map() -> None:
    session = Session(multi_statement_map_sdfg())
    (move,) = session.list_moves("doall-fission")
    assert move.labels == (all_map_entries(session.sdfg)[0].map.label,)


def test_map_fission_splits_the_map_by_statement_and_retires_its_label_and_matches_numpy() -> None:
    session = Session(multi_statement_map_sdfg())
    assert len(all_map_entries(session.sdfg)) == 1

    (move,) = session.list_moves("doall-fission")
    result = session.apply_move(move.kind, move.labels)

    assert result.status == MoveStatus.APPLIED
    assert len(all_map_entries(session.sdfg)) == 2  # the outer map split, one map per statement
    assert session.apply_move(move.kind, move.labels).status is MoveStatus.NOT_FOUND

    scopes = tasklet_scopes(session.sdfg)
    assert scopes["one"] is not None and scopes["two"] is not None
    assert scopes["one"] is not scopes["two"]  # b = a+1 and c = a*2 now sit in separate maps

    n = 16
    a = np.random.default_rng(0).random(n)
    b = np.zeros(n)
    c = np.zeros(n)
    session.sdfg(a=a.copy(), b=b, c=c, N=n)
    np.testing.assert_array_equal(b, a + 1.0)
    np.testing.assert_array_equal(c, a * 2.0)


ZVQX = np.array([0.0, 0.13, 4.0, 1.0, 0.0])


def copy_then_use_map(constant: bool) -> dace.SDFG:
    """cloudsc's ``zvqx`` loop: ``doall i in 0:5`` whose body copies ``zvqx[i]`` into a scalar, sets ``flag[i] = 0``
    beside the copy and then writes ``out[i] = 2 * scalar``; ``zvqx`` is a constant of the body or the map's input."""
    body = dace.SDFG("body")
    body.add_array("zvqx", [5], f64, transient=constant)
    if constant:
        body.add_constant("zvqx", ZVQX, body.arrays["zvqx"])
    body.add_scalar("pick", f64, transient=True)
    body.add_array("flag", [5], f64)
    body.add_array("out", [5], f64)
    body.add_symbol("i", dace.int64)
    copy = body.add_state("copy")
    copy.add_nedge(copy.add_read("zvqx"), copy.add_write("pick"), dace.Memlet("zvqx[i]"))
    zero = copy.add_tasklet("zero", {}, {"y"}, "y = 0.0")
    copy.add_edge(zero, "y", copy.add_write("flag"), None, dace.Memlet("flag[i]"))
    use = body.add_state_after(copy, "use")
    twice = use.add_tasklet("twice", {"x"}, {"y"}, "y = 2.0 * x")
    use.add_edge(use.add_read("pick"), None, twice, "x", dace.Memlet("pick[0]"))
    use.add_edge(twice, "y", use.add_write("out"), None, dace.Memlet("out[i]"))

    sdfg = dace.SDFG(f"copy_then_use_{'constant' if constant else 'input'}")
    if not constant:
        sdfg.add_array("zvqx", [5], f64)
    sdfg.add_array("flag", [5], f64)
    sdfg.add_array("out", [5], f64)
    state = sdfg.add_state()
    me, mx = state.add_map("pick", dict(i="0:5"))
    nest = state.add_nested_sdfg(body, {} if constant else {"zvqx": None}, {"flag": None, "out": None}, {"i": "i"})
    if constant:
        state.add_nedge(me, nest, dace.Memlet())
    else:
        state.add_memlet_path(state.add_read("zvqx"), me, nest, dst_conn="zvqx", memlet=dace.Memlet("zvqx[0:5]"))
    for name in ("flag", "out"):
        state.add_memlet_path(nest, mx, state.add_write(name), src_conn=name, memlet=dace.Memlet(f"{name}[i]"))
    sdfg.validate()
    return sdfg


def test_fission_of_a_map_whose_body_copies_out_of_its_own_constant_is_refused_and_leaves_the_program() -> None:
    """MapFission moved the constant out of the map but left the copy outside every new map, reading ``zvqx[i]``
    with ``i`` unbound (cloudsc_monolith on dace-extended 1.0.5 failed to build)."""
    sut = Session(copy_then_use_map(constant=True))
    label = all_map_entries(sut.sdfg)[0].map.label
    before = sut.sdfg.hash_sdfg()

    result = sut.apply_move("doall-fission", (label,))

    assert result.status is MoveStatus.ILLEGAL and "constant of its own" in result.reason, result
    assert sut.sdfg.hash_sdfg() == before
    assert not sut.list_moves("doall-fission")


def test_fission_of_a_map_whose_body_copies_out_of_the_map_s_input_splits_it_and_matches_numpy() -> None:
    sut = Session(copy_then_use_map(constant=False))
    label = all_map_entries(sut.sdfg)[0].map.label

    result = sut.apply_move("doall-fission", (label,))

    assert result.status is MoveStatus.APPLIED, result
    assert len(all_map_entries(sut.sdfg)) > 1
    flag, out = np.ones(5), np.zeros(5)
    sut.sdfg(zvqx=ZVQX.copy(), flag=flag, out=out)
    np.testing.assert_array_equal(out, 2.0 * ZVQX)
    np.testing.assert_array_equal(flag, np.zeros(5))
