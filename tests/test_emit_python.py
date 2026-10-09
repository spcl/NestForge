# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The Python emitter, one construct per test: the emitted text has the expected shape and the function computes
what a hand-written NumPy reference computes. Every buffer is allocated by the caller, as the C kernels do."""

import json
import math

import dace
import numpy as np
import pytest
import sympy
from dace import subsets
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, ReturnBlock
from helpers import run_emitted, sdfg_to_python

from nestforge.ir.emit_python import (
    Names,
    UnsupportedNest,
    expr,
    index,
    indexed_shape,
    load_emitted,
    nest_to_python,
    simplify_as_sizes,
)
from nestforge.ir.extract import Boundary

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


def nested_connector_clash_sdfg() -> dace.SDFG:
    """A nested tasklet ``y = a`` whose connector ``a`` reads a scalar and whose output binds the OUTER array ``a``."""
    inner = dace.SDFG("clash_inner")
    inner.add_array("out", [3], dace.float64)
    inner.add_scalar("acc", dace.float64, transient=True)
    state = inner.add_state()
    seed = state.add_tasklet("seed", {}, {"s": None}, "s = 7.0")
    copy = state.add_tasklet("copy", {"a": None}, {"y": None}, "y = a")
    acc = state.add_access("acc")
    state.add_edge(seed, "s", acc, None, dace.Memlet("acc[0]"))
    state.add_edge(acc, None, copy, "a", dace.Memlet("acc[0]"))
    state.add_edge(copy, "y", state.add_write("out"), None, dace.Memlet("out[1]"))
    outer = dace.SDFG("clash")
    outer.add_array("a", [3], dace.float64)
    ostate = outer.add_state()
    node = ostate.add_nested_sdfg(inner, {}, {"out": None})
    ostate.add_edge(node, "out", ostate.add_write("a"), None, dace.Memlet("a[0:3]"))
    return outer


def test_a_nested_connector_named_like_the_outer_array_it_writes_stays_a_connector():
    call, _ = emit_and_run(nested_connector_clash_sdfg(), {"a": np.zeros(3)}, {})

    np.testing.assert_array_equal(call["a"], [0.0, 7.0, 0.0])


def nested_symbol_clash_sdfg() -> dace.SDFG:
    """``s = 2``, then a nested SDFG binding its own ``s = 7`` to write ``inner_out``, then ``out[0] = s``."""
    inner = dace.SDFG("symbol_inner")
    inner.add_array("inner_out", [1], dace.float64)
    first = inner.add_state("first", is_start_block=True)
    second = inner.add_state("second")
    inner.add_edge(first, second, dace.InterstateEdge(assignments={"s": "7"}))
    t = second.add_tasklet("use", {}, {"o": None}, "o = s")
    second.add_edge(t, "o", second.add_write("inner_out"), None, dace.Memlet("inner_out[0]"))
    outer = dace.SDFG("symbol_clash")
    outer.add_array("mid", [1], dace.float64)
    outer.add_array("out", [1], dace.float64)
    outer.add_symbol("s", dace.int64)
    start = outer.add_state("start", is_start_block=True)
    call = outer.add_state("call")
    outer.add_edge(start, call, dace.InterstateEdge(assignments={"s": "2"}))
    node = call.add_nested_sdfg(inner, {}, {"inner_out": None})
    call.add_edge(node, "inner_out", call.add_write("mid"), None, dace.Memlet("mid[0]"))
    after = outer.add_state_after(call, "after")
    t = after.add_tasklet("read", {}, {"o": None}, "o = s")
    after.add_edge(t, "o", after.add_write("out"), None, dace.Memlet("out[0]"))
    return outer


def test_a_symbol_a_nested_sdfg_binds_itself_leaves_the_outer_value_alone():
    call, _ = emit_and_run(nested_symbol_clash_sdfg(), {}, {})

    assert (call["mid"][0], call["out"][0]) == (7.0, 2.0)


WINDOW = 4


def strided_window_sdfg(inner: dace.SDFG, read: str, write: str) -> dace.SDFG:
    """``inner`` bound to the window ``read`` of ``a`` as ``x`` and the window ``write`` of ``b`` as ``y``."""
    outer = dace.SDFG("strided_outer")
    outer.add_array("a", [16], dace.float64)
    outer.add_array("b", [16], dace.float64)
    state = outer.add_state()
    node = state.add_nested_sdfg(inner, {"x"}, {"y"})
    state.add_edge(state.add_read("a"), None, node, "x", dace.Memlet(read))
    state.add_edge(node, "y", state.add_write("b"), None, dace.Memlet(write))
    return outer


def window_arrays(inner: dace.SDFG, x_stride: int, y_stride: int) -> None:
    inner.add_array("x", [WINDOW], dace.float64, strides=[x_stride])
    inner.add_array("y", [WINDOW], dace.float64, strides=[y_stride])


def two_state_window(x_stride: int, y_stride: int) -> dace.SDFG:
    """``y = 2 * x``, then ``y[3] = x[3] + 100`` in a second state, so the nest is not inlined as one state."""
    inner = dace.SDFG("window")
    window_arrays(inner, x_stride, y_stride)
    first = inner.add_state(is_start_block=True)
    first.add_mapped_tasklet(
        "twice",
        {"i": f"0:{WINDOW}"},
        {"v": dace.Memlet("x[i]")},
        "w = 2 * v",
        {"w": dace.Memlet("y[i]")},
        external_edges=True,
    )
    second = inner.add_state_after(first)
    last = second.add_tasklet("last", {"v"}, {"w"}, "w = v + 100")
    second.add_edge(second.add_read("x"), None, last, "v", dace.Memlet("x[3]"))
    second.add_edge(last, "w", second.add_write("y"), None, dace.Memlet("y[3]"))
    return inner


def test_a_strided_nested_window_scales_its_indices_by_the_stride():
    """The windows ``a[2::3]`` and ``b[1::2]`` read and write every third and every second element."""
    a = np.arange(16.0)

    call, source = emit_and_run(strided_window_sdfg(two_state_window(3, 2), "a[2:12:3]", "b[1:8:2]"), {"a": a}, {})

    assert "b[2 * i + 1] = 2 * a[3 * i + 2]" in source and "b[7] = a[11] + 100" in source
    expected = np.zeros(16)
    expected[1:8:2] = 2 * a[2:12:3]
    expected[7] = a[11] + 100
    np.testing.assert_array_equal(call["b"], expected)


def indexed_read_window(x_stride: int) -> dace.SDFG:
    """``y[0] = x[1]`` through an interstate symbol, whose index moves with the window."""
    inner = dace.SDFG("indexed")
    window_arrays(inner, x_stride, 1)
    inner.add_symbol("t", dace.float64)
    first = inner.add_state(is_start_block=True)
    second = inner.add_state()
    inner.add_edge(first, second, dace.InterstateEdge(assignments={"t": "x[1]"}))
    write = second.add_tasklet("write", {}, {"o"}, "o = t")
    second.add_edge(write, "o", second.add_write("y"), None, dace.Memlet("y[0]"))
    return inner


@pytest.mark.parametrize("x_stride, read", [(2, "a[1:8:2]"), (1, "a[2:6]")])
def test_a_shifted_window_read_by_index_outside_dataflow_reads_the_outer_element(x_stride, read):
    a = np.arange(16.0)

    call, source = emit_and_run(strided_window_sdfg(indexed_read_window(x_stride), read, "b[0:4]"), {"a": a}, {})

    assert "a[3]" in source and "x[" not in source
    assert call["b"][0] == a[3]


def tag_write_sdfg() -> dace.SDFG:
    """``tags[idx[i]] = i`` through a tasklet whose output connector is the whole, dynamically written array."""
    sdfg = dace.SDFG("tag_write")
    sdfg.add_array("idx", [N], dace.int64)
    sdfg.add_array("tags", [M], dace.int64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map("tag", {"i": "0:N"}, schedule=dace.ScheduleType.Sequential)
    t = state.add_tasklet("tag", {"v": None}, {"t": None}, "t[v] = i")
    state.add_memlet_path(state.add_read("idx"), entry, t, dst_conn="v", memlet=dace.Memlet("idx[i]"))
    state.add_memlet_path(
        t, exit_node, state.add_write("tags"), src_conn="t", memlet=dace.Memlet("tags[0:M]", dynamic=True)
    )
    return sdfg


def test_a_tasklet_writing_into_a_range_writes_through_to_the_array():
    call, source = emit_and_run(tag_write_sdfg(), {"idx": np.array([2, 0, 3])}, {"N": 3, "M": 5})

    assert "tags[0:M] = " not in source  # written through its view, not copied back
    np.testing.assert_array_equal(call["tags"], [1, 0, 0, 2, 0])


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


def test_a_widened_extent_names_only_sizes_not_max():
    """A loop end widened against its start folds ``Max(N - 1, 0)`` to ``N - 1``: the manifest declares no ``Max``."""
    n = sympy.Symbol("N")
    assert simplify_as_sizes(sympy.Max(n - 1, -n + n)) == n - 1


def test_a_rank_changing_copy_reshapes_to_a_literal_shape():
    """The reshape target is a tuple of extents, never ``np.shape`` of a slice the C translator cannot size."""
    column = subsets.Range([(0, M - 1, 1), (3, 3, 1)])
    assert indexed_shape(column) == "(M, )"
    assert indexed_shape(subsets.Range([(0, N - 1, 2)])) == "((int_floor(N - 1, 2) + 1), )"


def test_a_single_element_axis_is_an_index_whatever_type_its_bounds_have():
    """A Python ``int`` begin prints as ``(0)`` and a SymPy ``Zero`` end as ``0``; the axis is still one element."""
    subset = subsets.Range([(0, dace.symbol("N") - 1, 1), (0, sympy.Integer(0), 1)])

    assert index(subset) == "0:N, 0"
