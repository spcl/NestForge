# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The Python emitter, one construct per test: the emitted text has the expected shape and the function computes
what a hand-written NumPy reference computes. Every buffer is allocated by the caller, as the C kernels do."""

import json
import math

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, ReturnBlock

from nestforge.ir.emit_python import Names, UnsupportedNest, expr, load_emitted, nest_to_python
from nestforge.ir.extract import Boundary
from helpers import run_emitted, sdfg_to_python

N = dace.symbol("N", dtype=dace.int64)
M = dace.symbol("M", dtype=dace.int64)


def emit_and_run(sdfg: dace.SDFG, inputs: dict, sizes: dict[str, int]) -> tuple[dict, str]:
    source, lowered = sdfg_to_python(sdfg, "k")
    return run_emitted(source, "k", lowered, inputs, sizes), source


def body_of(source: str) -> str:
    return source[source.index("def k(") :]


@dace.program
def scale_2d(A: dace.float64[N, M], B: dace.float64[N, M]):
    for i, j in dace.map[0:N, 0:M]:
        B[i, j] = A[i, j] * 2.0 + j


def test_a_two_dimensional_map_is_two_nested_parallel_for_loops_over_caller_buffers():
    A = np.random.default_rng(0).random((5, 3))

    call, source = emit_and_run(scale_2d.to_sdfg(simplify=True), {"A": A}, {"N": 5, "M": 3})

    assert body_of(source).count("for ") == 2 and body_of(source).count(":  # parallel\n") == 2
    assert "np.zeros" not in source and "np.empty" not in source
    np.testing.assert_array_equal(call["B"], A * 2.0 + np.arange(3))


@dace.program
def reduce_sum(a: dace.float64[N], out: dace.float64[1]):
    out[0] = 0.0
    for i in dace.map[0:N]:
        out[0] += a[i]


def test_a_sum_reduction_accumulates_every_iteration():
    a = np.random.default_rng(1).random(17)

    call, source = emit_and_run(reduce_sum.to_sdfg(simplify=True), {"a": a}, {"N": 17})

    assert "out[0] = out[0] + " in source
    np.testing.assert_allclose(call["out"][0], a.sum(), rtol=1e-14)


def running_max() -> dace.SDFG:
    """``out[0] = max(out[0], a[i])`` over a map, as a write-conflict resolution."""
    sdfg = dace.SDFG("running_max")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("out", [1], dace.float64)
    state = sdfg.add_state("only", is_start_block=True)
    entry, exit_node = state.add_map("fold", {"i": "0:N"})
    t = state.add_tasklet("pass_on", {"x"}, {"y"}, "y = x")
    state.add_memlet_path(state.add_read("a"), entry, t, dst_conn="x", memlet=dace.Memlet("a[i]"))
    folded = dace.Memlet("out[0]", wcr="lambda p, q: max(p, q)")
    state.add_memlet_path(t, exit_node, state.add_write("out"), src_conn="y", memlet=folded)
    return sdfg


def test_a_max_reduction_combines_with_np_maximum():
    a = np.random.default_rng(2).random(11) - 0.5

    call, source = emit_and_run(running_max(), {"a": a, "out": np.array([-1e9])}, {"N": 11})

    assert "np.maximum(out[0], " in source
    assert call["out"][0] == a.max()


@dace.program
def hist_scatter(idx: dace.int64[N], w: dace.float64[N], hist: dace.float64[M]):
    for i in dace.map[0:N]:
        hist[idx[i]] += w[i]


def test_a_data_dependent_scatter_accumulates_duplicate_indices():
    idx = np.array([0, 2, 2, 1, 2, 0], np.int64)
    w = np.arange(1.0, 7.0)
    expected = np.zeros(3)
    np.add.at(expected, idx, w)

    call, _ = emit_and_run(hist_scatter.to_sdfg(simplify=True), {"idx": idx, "w": w}, {"N": 6, "M": 3})

    np.testing.assert_array_equal(call["hist"], expected)


@dace.program
def three_way(a: dace.float64[N], b: dace.float64[N], sel: dace.int64):
    if sel > 0:
        b[:] = a + 1.0
    elif sel < 0:
        b[:] = a - 1.0
    else:
        b[:] = a


@pytest.mark.parametrize("sel, shift", [(3, 1.0), (-3, -1.0), (0, 0.0)])
def test_a_conditional_runs_exactly_the_branch_its_selector_picks(sel, shift):
    sdfg = three_way.to_sdfg(simplify=True)
    a = np.arange(4.0)

    call, source = emit_and_run(sdfg, {"a": a, "sel": sel}, {"N": 4})

    assert "if " in source and "else:" in source
    np.testing.assert_array_equal(call["b"], a + shift)


@dace.program
def prefix_sum(a: dace.float64[N]):
    for i in range(1, N):
        a[i] = a[i] + a[i - 1]


def test_a_sequential_loop_is_an_unmarked_while_loop_carrying_its_dependence():
    a = np.random.default_rng(3).random(9)

    call, source = emit_and_run(prefix_sum.to_sdfg(simplify=True), {"a": a}, {"N": 9})

    assert "while " in source and "# parallel" not in source
    np.testing.assert_allclose(call["a"], np.cumsum(a), rtol=1e-14)


@dace.program
def add_until_negative(a: dace.float64[N], d: dace.float64[N]):
    for i in range(N):
        if d[i] < 0.0:
            break
        a[i] = a[i] + 1.0


def test_a_break_leaves_the_loop_at_the_first_negative():
    d = np.ones(10)
    d[6] = -1.0

    call, source = emit_and_run(add_until_negative.to_sdfg(simplify=True), {"a": np.zeros(10), "d": d}, {"N": 10})

    assert "break" in source
    np.testing.assert_array_equal(call["a"], np.r_[np.ones(6), np.zeros(4)])


def early_return_sdfg() -> dace.SDFG:
    """``out[:] = a``, then ``return`` when ``sel > 0``, else ``out += 1``."""
    sdfg = dace.SDFG("earlyret")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("out", [N], dace.float64)
    sdfg.add_symbol("sel", dace.int64)
    first = sdfg.add_state("first", is_start_block=True)
    first.add_nedge(first.add_read("a"), first.add_write("out"), dace.Memlet("a[0:N]"))
    cond = ConditionalBlock("cond", sdfg=sdfg)
    sdfg.add_node(cond)
    branch = ControlFlowRegion("returning", sdfg=sdfg)
    branch.add_node(ReturnBlock("ret", sdfg=branch), is_start_block=True)
    cond.add_branch("sel > 0", branch)
    last = sdfg.add_state("last")
    entry, exit_node = last.add_map("inc", {"i": "0:N"})
    t = last.add_tasklet("inc", {"x"}, {"y"}, "y = x + 1.0")
    last.add_memlet_path(last.add_read("out"), entry, t, dst_conn="x", memlet=dace.Memlet("out[i]"))
    last.add_memlet_path(t, exit_node, last.add_write("out"), src_conn="y", memlet=dace.Memlet("out[i]"))
    sdfg.add_edge(first, cond, dace.InterstateEdge())
    sdfg.add_edge(cond, last, dace.InterstateEdge())
    return sdfg


@pytest.mark.parametrize("sel, shift", [(1, 0.0), (0, 1.0)])
def test_a_whole_program_return_skips_what_follows_it(sel, shift):
    a = np.arange(5.0)

    call, source = emit_and_run(early_return_sdfg(), {"a": a}, {"N": 5, "sel": sel})

    assert "return" in source
    np.testing.assert_array_equal(call["out"], a + shift)


def test_an_early_return_cannot_leave_an_extracted_nest():
    """In a nest, ``return`` would end only the kernel, not the program the nest came from."""
    sdfg = early_return_sdfg()
    nest = Boundary(["a"], ["out"], ["N", "sel"], nsdfg_node=None, state=None, standalone_sdfg=sdfg)

    with pytest.raises(UnsupportedNest, match="returns early"):
        nest_to_python(nest, "k")


def test_an_unstructured_conditional_edge_is_refused():
    sdfg = dace.SDFG("goto")
    sdfg.add_array("a", [1], dace.float64)
    sdfg.add_symbol("sel", dace.int64)
    first = sdfg.add_state("first", is_start_block=True)
    second = sdfg.add_state("second")
    sdfg.add_edge(first, second, dace.InterstateEdge(condition="sel > 0"))

    with pytest.raises(UnsupportedNest, match="conditional interstate edge"):
        sdfg_to_python(sdfg, "k")


@dace.program
def matmul(A: dace.float64[N, M], B: dace.float64[M, N], C: dace.float64[N, N]):
    C[:] = A @ B


def test_a_library_node_expands_to_its_pure_loops():
    sdfg = matmul.to_sdfg(simplify=True)
    assert any(isinstance(n, nodes.LibraryNode) for n, _ in sdfg.all_nodes_recursive())
    rng = np.random.default_rng(4)
    A, B = rng.random((4, 3)), rng.random((3, 4))

    call, source = emit_and_run(sdfg, {"A": A, "B": B}, {"N": 4, "M": 3})

    assert "@" not in body_of(source) and "for " in source
    np.testing.assert_allclose(call["C"], A @ B, rtol=1e-14)


@dace.program
def masked_abs(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        if a[i] > 0.0:
            b[i] = a[i]
        else:
            b[i] = -a[i]


def test_a_nested_sdfg_inside_a_map_writes_the_outer_buffer_in_place():
    sdfg = masked_abs.to_sdfg(simplify=True)
    assert any(isinstance(n, nodes.NestedSDFG) for n, _ in sdfg.all_nodes_recursive())
    a = np.random.default_rng(5).random(8) - 0.5

    call, _ = emit_and_run(sdfg, {"a": a}, {"N": 8})

    np.testing.assert_array_equal(call["b"], np.abs(a))


def test_emission_leaves_the_program_unchanged():
    sdfg = masked_abs.to_sdfg(simplify=True)
    before = json.dumps(sdfg.to_json(), sort_keys=True, default=str)

    sdfg_to_python(sdfg, "k")

    assert json.dumps(sdfg.to_json(), sort_keys=True, default=str) == before


def vector_connector_sdfg() -> dace.SDFG:
    """A tasklet whose whole-vector connector is named ``b``, like the array ``b`` a later tasklet reads."""
    sdfg = dace.SDFG("shadowing")
    for name in ("a", "b"):
        sdfg.add_array(name, [2], dace.float64)
    sdfg.add_array("out", [2], dace.float64)
    state = sdfg.add_state("only", is_start_block=True)
    first = state.add_tasklet("pair_sum", {"b"}, {"o"}, "o = b[0] + b[1]")
    state.add_edge(state.add_read("a"), None, first, "b", dace.Memlet("a[0:2]"))
    written = state.add_write("out")
    state.add_edge(first, "o", written, None, dace.Memlet("out[0]"))
    second = state.add_tasklet("copy_b", {"x"}, {"y"}, "y = x")
    state.add_edge(state.add_read("b"), None, second, "x", dace.Memlet("b[0]"))
    state.add_edge(second, "y", written, None, dace.Memlet("out[1]"))
    return sdfg


def test_a_connector_named_like_an_array_is_renamed_not_assigned_over_it():
    a, b = np.array([1.0, 2.0]), np.array([7.0, 8.0])

    call, source = emit_and_run(vector_connector_sdfg(), {"a": a, "b": b}, {})

    assert not any(line.strip().startswith("b = ") for line in source.splitlines())
    np.testing.assert_array_equal(call["out"], [3.0, 7.0])


def guarded_sdfg(language: dace.Language, code: str) -> dace.SDFG:
    """A precondition guard on the scalar ``x`` before ``out[0] = x``."""
    sdfg = dace.SDFG("guarded")
    sdfg.add_scalar("x", dace.float64)
    sdfg.add_array("out", [1], dace.float64)
    check = sdfg.add_state("check", is_start_block=True)
    guard = check.add_tasklet("guard", {"v"}, {}, code, language=language, side_effects=True)
    check.add_edge(check.add_read("x"), None, guard, "v", dace.Memlet("x[0]"))
    body = sdfg.add_state_after(check, "body")
    t = body.add_tasklet("copy", {"v"}, {"w"}, "w = v")
    body.add_edge(body.add_read("x"), None, t, "v", dace.Memlet("x[0]"))
    body.add_edge(t, "w", body.add_write("out"), None, dace.Memlet("out[0]"))
    return sdfg


def test_a_python_guard_raises_when_its_precondition_fails_and_passes_otherwise():
    sdfg = guarded_sdfg(dace.Language.Python, "if v < 0.0:\n    abort()")

    call, source = emit_and_run(sdfg, {"x": 2.5}, {})

    assert "def abort():" in source
    assert call["out"][0] == 2.5
    with pytest.raises(RuntimeError, match="DaCe precondition violated"):
        emit_and_run(sdfg, {"x": -1.0}, {})


def test_a_tasklet_in_another_language_is_refused():
    sdfg = guarded_sdfg(dace.Language.CPP, "if (v < 0.0) { std::abort(); }")

    with pytest.raises(UnsupportedNest, match="guard is CPP, not Python"):
        sdfg_to_python(sdfg, "k")


def test_two_kernels_of_equal_length_load_as_two_modules():
    """The module file is named by a hash of the source, not a counter that bytecode caching could alias."""
    first = load_emitted("import numpy as np\n\n\ndef k(a):\n    a[0] = 1.0\n", "k")
    second = load_emitted("import numpy as np\n\n\ndef k(a):\n    a[0] = 2.0\n", "k")
    a, b = np.zeros(1), np.zeros(1)

    first.k(a)
    second.k(b)

    assert (a[0], b[0]) == (1.0, 2.0)


@pytest.mark.parametrize("a, b", [(7, 2), (-7, 2), (7, -2), (-7, -2), (8, 4), (0, 3)])
def test_dace_integer_division_heads_become_python_operators_with_their_meaning(a, b):
    names = Names(frozenset(), frozenset())
    floor_text, ceil_text = expr("int_floor(a, b)", names), expr("int_ceil(a, b)", names)

    assert "int_floor" not in floor_text and "int_ceil" not in ceil_text
    assert eval(floor_text, {"a": a, "b": b}) == math.floor(a / b)  # noqa: S307 -- the emitted expression
    assert eval(ceil_text, {"a": a, "b": b}) == math.ceil(a / b)  # noqa: S307
