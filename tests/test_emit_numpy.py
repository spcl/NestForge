# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""NumPy emission bugs that produce a wrong or unloadable oracle instead of a crash: sympy functions reaching the
source verbatim, ordering edges read as operands, a join applying every predecessor's assignments, and a dtype
with no C spelling."""

from dataclasses import dataclass

import numpy as np
import pytest
import sympy

import dace
from dace import symbolic

from dace.sdfg.state import LoopRegion

from nestforge.ir.emit_libnode import UnsupportedLibraryNode, data_edge
from nestforge.ir.emit_numpy import (
    EMITTED_BUILTINS,
    UnsupportedNest,
    int_floor,
    load_emitted,
    loop_init_value,
    normalize_casts,
    sdfg_to_numpy,
)
from nestforge.ir.libnode import ExternalCall, proto_and_call

I = sympy.Symbol("i")
N = sympy.Symbol("N")


def evaluated(expr, **env):
    """symstr the sympy expression the way a subset/bound reaches the emitter, normalize it, and evaluate
    the emitted text in the SAME namespace the oracle runs in (python builtins + EMITTED_BUILTINS)."""
    code = normalize_casts(symbolic.symstr(expr))
    assert "Max(" not in code and "Min(" not in code and "Abs(" not in code, f"un-rewritten sympy func: {code}"
    return eval(code, dict(EMITTED_BUILTINS), dict(env))


def test_clamped_index_exprs_are_emittable_and_exact():
    # a stencil/boundary-clamped subset: A[Max(0, i-1)] / bound Min(N, i+1). Before the fix symstr rendered
    # `Max(0, i - 1)` straight into the kernel and the oracle died with NameError: name 'Max'.
    assert evaluated(sympy.Max(0, I - 1), i=5) == 4
    assert evaluated(sympy.Max(0, I - 1), i=0) == 0  # the clamp actually clamps
    assert evaluated(sympy.Min(N, I + 1), i=5, N=4) == 4
    assert evaluated(sympy.Abs(I - 3), i=1) == 2


def test_variadic_max_min_are_rewritten():
    # sympy Max/Min are n-ary (a tile extent clamped by several bounds); a binary-only rewrite would
    # mis-render or drop arguments.
    assert evaluated(sympy.Max(0, N, I - 1), i=5, N=4) == 4
    assert evaluated(sympy.Min(0, N, I - 1), i=5, N=4) == 0


def test_clamped_expr_keeps_integer_type_for_indexing():
    # the rewrite must yield a Python int, not a numpy scalar: the value indexes an array and feeds range().
    value = evaluated(sympy.Max(0, I - 1), i=5)
    assert isinstance(value, int)
    assert range(0, evaluated(sympy.Min(N, I + 1), i=5, N=4))  # usable as a bound


@dataclass
class FakeEdge:
    """Minimal stand-in for a graph edge: ``data_edge`` only inspects ``e.data``."""

    data: dace.Memlet


def test_data_edge_skips_empty_ordering_edges():
    # DaCe's StateFusion adds empty (data=None) happens-before edges to sequence nodes without merging.
    # Taking edges[0] blindly picked one as the operand -> sdfg.arrays[None] KeyError mid-emission.
    empty, real = FakeEdge(dace.Memlet()), FakeEdge(dace.Memlet(data="A", subset="0:N"))
    assert empty.data.is_empty() and not real.data.is_empty()
    assert data_edge([empty, real], None, "input") is real  # the ordering edge is not the operand
    assert data_edge([real, empty], None, "input") is real


def test_data_edge_refuses_when_only_ordering_edges_exist():
    # no data-carrying edge at all: an actionable refusal, never arrays[None].
    with pytest.raises(UnsupportedLibraryNode, match="no data-carrying"):
        data_edge([FakeEdge(dace.Memlet())], None, "input")


def join_sdfg() -> dace.SDFG:
    """An unstructured join: two unconditional predecessors both assigning the same symbol into one block."""
    sdfg = dace.SDFG("join")
    sdfg.add_array("A", [2], dace.float64)
    sdfg.add_symbol("k", dace.int64)
    start = sdfg.add_state("start", is_start_block=True)
    left, right, join = sdfg.add_state("left"), sdfg.add_state("right"), sdfg.add_state("join")
    sdfg.add_edge(start, left, dace.InterstateEdge())
    sdfg.add_edge(start, right, dace.InterstateEdge())
    # both carry an assignment; straight-line emission would apply BOTH into `join`
    sdfg.add_edge(left, join, dace.InterstateEdge(assignments={"k": "k + 1"}))
    sdfg.add_edge(right, join, dace.InterstateEdge(assignments={"k": "k + 1"}))
    return sdfg


def test_unstructured_join_is_refused_not_double_applied():
    # Before the fix interstate_lines emitted `k = k + 1` once per in-edge, double-incrementing a symbol
    # that only ONE predecessor actually assigns at runtime -- a silently wrong index.
    with pytest.raises(UnsupportedNest, match="carry assignments"):
        sdfg_to_numpy(join_sdfg(), fn_name="join")


def test_single_predecessor_assignment_still_emits():
    # the refusal must not fire on the normal case: one edge carrying assignments.
    sdfg = dace.SDFG("chain")
    sdfg.add_array("A", [2], dace.float64)
    sdfg.add_symbol("k", dace.int64)
    first = sdfg.add_state("first", is_start_block=True)
    second = sdfg.add_state("second")
    sdfg.add_edge(first, second, dace.InterstateEdge(assignments={"k": "k + 1"}))
    assert "k = k + 1" in sdfg_to_numpy(sdfg, fn_name="chain")


DTYPES = {"float64": dace.float64, "complex128": dace.complex128}


def extern_call_with_dtype(dtype_name: str, shape=(8,)):
    """An ExternalCall wired into a real state: ``proto_and_call`` reads the memlets to tell a pointer
    connector from a value one, so a node without edges cannot answer the question it is asked."""
    manifest = {
        "array_args": ["A"],
        "output_args": [],
        "init": {"arrays": {"A": {"dtype": dtype_name}}, "scalars": {}},
    }
    node = ExternalCall("k", inputs={"_in_A"}, outputs=set(), config=manifest)
    node.symbol, node.abi_order = "k_fp64", ["A"]
    sdfg = dace.SDFG("host")
    sdfg.add_array("A", list(shape), DTYPES[dtype_name])
    state = sdfg.add_state()
    state.add_edge(state.add_read("A"), None, node, "_in_A", dace.Memlet.from_array("A", sdfg.arrays["A"]))
    return node, state


def test_extern_c_prototype_builds_for_a_known_dtype():
    proto, call = proto_and_call(*extern_call_with_dtype("float64"))
    assert "const double* A" in proto and "k_fp64(_in_A);" == call


def test_a_one_element_connector_is_passed_by_address():
    """DaCe defines a one-element memlet as a VALUE, but the compiled signature always takes a pointer:
    passing it directly is "cannot convert double to double*" at compile time (E3: s114/s115/s116)."""
    _, state = extern_call_with_dtype("float64", shape=(1,))
    node = next(n for n in state.nodes() if isinstance(n, ExternalCall))
    proto, call = proto_and_call(node, state)
    assert "const double* A" in proto
    assert call == "k_fp64(&_in_A);"


def test_unspellable_array_dtype_is_refused_not_keyerror():
    # a complex/fp16/unsigned array used to raise a bare KeyError from CPP_SCALAR mid-codegen; it must be an
    # actionable refusal naming the array and dtype so the caller can keep the DaceReference variant.
    with pytest.raises(ValueError, match="complex128"):
        proto_and_call(*extern_call_with_dtype("complex128"))


@pytest.mark.parametrize(
    "expr, want",
    [
        ("int_floor(a[i, j], 2)", "((a[i, j]) // (2))"),
        ("int_floor(aa[i, j] + b[k, l], 2)", "((aa[i, j] + b[k, l]) // (2))"),
        ("int_floor(x, 2)", "((x) // (2))"),
    ],
)
def test_a_subscript_comma_does_not_split_an_argument(expr, want):
    """The comma in ``a[i, j]`` is not an argument separator; splitting there emits C that does not parse."""
    from nestforge.ir.emit_numpy import apply_call

    assert apply_call(expr, "int_floor", lambda a, b: f"(({a}) // ({b}))") == want


def test_multidim_subscript_survives_the_userfunc_rewrite():
    """End to end through rewrite_userfuncs: ``int_ceil`` is not rewritten and stays a call, so a variadic
    ``Max`` carries the subscript-comma case here."""
    from nestforge.ir.emit_numpy import rewrite_userfuncs

    out = rewrite_userfuncs("d[Max(aa[i, j], 2)] = b[int_ceil(Min(c[k, l], 4), 4)]")
    assert out.count("[") == out.count("]"), out
    assert out.count("(") == out.count(")"), out
    assert "Max" not in out and "Min" not in out, out
    assert out == "d[max(aa[i, j], 2)] = b[int_ceil(min(c[k, l], 4), 4)]", out


# data-dependent scratch extents (the spmv CSR span)
M_SYM = dace.symbol("M")
NNZ_SYM = dace.symbol("NNZ")


@dace.program
def spmv_row_scratch(
    A_indptr: dace.int64[M_SYM + 1], A_vals: dace.float64[NNZ_SYM], x: dace.float64[M_SYM], y: dace.float64[M_SYM]
):
    """A CSR row loop: the row scratch is sized by a span READ OUT OF the index array."""
    for i in range(M_SYM):
        start = A_indptr[i]
        stop = A_indptr[i + 1]
        row = np.empty(stop - start, dace.float64)
        for k in range(stop - start):
            row[k] = A_vals[start + k] * x[k]
        y[i] = np.sum(row)


@pytest.mark.parametrize(
    "expr, want",
    [
        ("A_indptr[M + 1] - A_indptr[0]", False),  # the spmv span: residual symbols are kernel symbols
        ("A_indptr[i]", False),
        ("A_vals", False),  # a bare array NAME used as a value
        ("M*N", True),
        ("3", True),
        ("int_floor(N, 4)", True),  # a math head is not a data read
        ("Min(M, N)", True),  # Min/Max are not Function atoms at all
        ("i + 1", False),  # sizable for the other reason: not a kernel symbol
    ],
)
def test_sizable_rejects_a_data_read_however_it_is_spelled(expr, want):
    """``free_symbols`` is structurally blind to an indexed read: DaCe renders ``A_indptr[i]`` as
    ``Subscript(A_indptr, i)``, so the ARRAY NAME is the Function head and never appears among the free
    symbols. The predicate must walk the tree."""
    from nestforge.ir.emit_numpy import sizable

    arrays = {"A_indptr": None, "A_vals": None}
    assert sizable(symbolic.pystr_to_symbolic(expr), {"M", "N"}, arrays) is want


def test_data_dependent_scratch_extent_is_refused_not_emitted():
    """A scratch extent such as ``A_indptr[M + 1] - A_indptr[0]`` depends on data, so no caller can allocate it."""
    from nestforge.ir.emit_numpy import UnsupportedNest, sdfg_to_numpy

    sdfg = spmv_row_scratch.to_sdfg(simplify=True)
    assert "row" in sdfg.arrays and sdfg.arrays["row"].transient, "fixture no longer has the scratch buffer"
    with pytest.raises(UnsupportedNest, match="row"):
        sdfg_to_numpy(sdfg, "spmv")


def test_a_data_read_never_becomes_a_size_bound():
    """The narrow half: ``symbol_ranges`` must not ingest an interstate assignment that reads array data,
    or every shape it reaches widens to an extent the caller cannot evaluate."""
    from nestforge.ir.emit_numpy import maxsize_loop_scratch, reads_array_data

    sdfg = spmv_row_scratch.to_sdfg(simplify=True)
    widened = maxsize_loop_scratch(sdfg, ["M", "NNZ"])
    for dim in widened.arrays["row"].shape:
        assert not reads_array_data(sympy.sympify(dim), widened.arrays), f"widened to a data read: {dim}"


# copy DIRECTION (the in-place-copy inversion)
@dace.program
def shift_through_a_view(A: dace.float64[M_SYM]):
    """A slice-to-slice copy of ONE array: DaCe stages it through a view, so the copy edges are the
    shape `copy_direction` resolves."""
    A[1:M_SYM] = A[0 : M_SYM - 1]


@dace.program
def elementwise_from_a_row(A: dace.float64[M_SYM, M_SYM]):
    for i in range(M_SYM):
        A[i, 0] = A[i, M_SYM - 1]


@pytest.mark.parametrize("prog", [shift_through_a_view, elementwise_from_a_row], ids=["slice_copy", "element_copy"])
def test_copy_direction_agrees_with_dace_on_every_real_copy_edge(prog):
    """When both ends of a copy name one array, DaCe reads ``subset`` as the source; checked against DaCe itself
    so the two cannot drift apart."""
    from dace.sdfg import nodes as dnodes
    from nestforge.ir.emit_numpy import copy_direction

    sdfg = prog.to_sdfg(simplify=True)
    checked = 0
    for state in sdfg.states():
        for edge in state.edges():
            if not (isinstance(edge.src, dnodes.AccessNode) and isinstance(edge.dst, dnodes.AccessNode)):
                continue
            if edge.data.is_empty():
                continue
            src_name, src_sub, dst_sub = copy_direction(edge)
            assert src_name == edge.src.data
            assert src_sub == edge.data.get_src_subset(edge, state), f"src subset disagrees on {edge}"
            assert dst_sub == edge.data.get_dst_subset(edge, state), f"dst subset disagrees on {edge}"
            checked += 1
    assert checked, "fixture produced no access-node copy edge -- it no longer covers copy_direction"


def test_copy_direction_reads_subset_as_the_source_when_both_ends_are_one_array():
    """Backwards, ``A[i] = A[j]`` becomes ``A[j] = A[i]``, a wrong answer with no error."""
    from nestforge.ir.emit_numpy import copy_direction

    sdfg = dace.SDFG("inplace_copy")
    sdfg.add_array("A", [M_SYM], dace.float64)
    state = sdfg.add_state()
    read, write = state.add_read("A"), state.add_write("A")
    memlet = dace.Memlet(data="A", subset="1:M", other_subset="0:M - 1")
    state.add_edge(read, None, write, None, memlet)

    src_name, src_sub, dst_sub = copy_direction(state.edges()[0])
    assert src_name == "A"
    assert str(src_sub) == "1:M", "memlet.subset indexes memlet.data, which DaCe resolves as the source"
    assert str(dst_sub) == "0:M - 1"


def test_two_kernels_of_equal_length_do_not_share_bytecode():
    """CPython validates cached bytecode by mtime and size, so two equal-length kernels written to one path in the
    same second would run the first one's code."""
    first = "def k():\n    return 'AAAA'\n"
    second = "def k():\n    return 'BBBB'\n"
    assert len(first) == len(second), "the fixture only reproduces the bug at equal byte length"
    from nestforge.ir.emit_numpy import load_emitted

    assert load_emitted(first, "k").k() == "AAAA"
    assert load_emitted(second, "k").k() == "BBBB"


@pytest.mark.parametrize(
    "end, step, want_stop",
    [
        ("N - 1", "1", "N"),
        ("N - 1", "-1", "N - 2"),
        ("0", "-1", "-1"),
        ("0", "2", "1"),
    ],
)
def test_range_stop_follows_the_step_sign(end, step, want_stop):
    """A DaCe range end is INCLUSIVE, so python's exclusive stop is one past the last element IN THE
    DIRECTION OF TRAVEL. The emitter added 1 unconditionally, so a descending map lost its final
    iteration: ``range(N-1, 0, -1)`` for a DaCe range ending at 0 never yields 0."""
    from nestforge.ir.emit_numpy import range_stop

    got = range_stop(symbolic.pystr_to_symbolic(end), symbolic.pystr_to_symbolic(step), "map parameter 'i'")
    assert sympy.simplify(got - symbolic.pystr_to_symbolic(want_stop)) == 0, f"{got} != {want_stop}"


def test_a_descending_range_covers_its_last_element():
    """The behaviour the sign fix buys, checked by ENUMERATING rather than by re-deriving the formula."""
    from nestforge.ir.emit_numpy import range_stop

    stop = int(range_stop(symbolic.pystr_to_symbolic("0"), symbolic.pystr_to_symbolic("-1"), "x"))
    assert list(range(7, stop, -1)) == [7, 6, 5, 4, 3, 2, 1, 0], "element 0 must not be dropped"


@pytest.mark.parametrize(
    "rng, want",
    [
        ((7, 0, -1), list(range(7, -1, -1))),
        ((7, 2, -1), [7, 6, 5, 4, 3, 2]),
        ((0, 7, 1), list(range(8))),
        ((1, 7, 2), [1, 3, 5, 7]),
    ],
)
def test_index_str_slices_a_descending_range_to_its_last_element(rng, want):
    """A negative slice stop counts from the end, so ``A[7:-1:-1]`` selects nothing; checked against NumPy
    itself, since the formula was the bug."""
    from nestforge.ir.emit_libnode import index_str

    a = np.arange(8)
    got = eval(f"a[{index_str(dace.subsets.Range([rng]))}]", {"a": a})
    assert list(np.atleast_1d(got)) == want


def test_a_strided_slice_with_a_symbolic_stop_is_emitted():
    """An ascending stop is ``end + 1 >= 1``, so only a descending one needs proof; asking a relation on a symbol
    for its truth value raised."""
    from nestforge.ir.emit_libnode import index_str

    n = dace.symbol("N")
    assert index_str(dace.subsets.Range([(0, n - 1, 2)])) == "0:N:2"


def test_a_descending_slice_whose_stop_may_be_negative_is_refused():
    from nestforge.ir.emit_libnode import UnsupportedLibraryNode, index_str

    n, m = dace.symbol("N"), dace.symbol("M")
    with pytest.raises(UnsupportedLibraryNode, match="not provably >= 0"):
        index_str(dace.subsets.Range([(n - 1, m, -1)]))


def test_range_stop_refuses_a_step_of_unknown_sign():
    """No sound stop exists without a direction; guessing one silently drops or over-runs elements."""
    from nestforge.ir.emit_numpy import UnsupportedNest, range_stop

    with pytest.raises(UnsupportedNest, match="undecidable sign"):
        range_stop(symbolic.pystr_to_symbolic("N"), symbolic.pystr_to_symbolic("s"), "map parameter 'i'")


# nested-SDFG binding + conditional branch order
def test_symbol_mapping_binds_simultaneously_when_the_bindings_interfere():
    """``symbol_mapping`` is a substitution applied all at once. Emitted as ordered assignments, a swap
    ``{i: j, j: i}`` runs ``i = j`` then ``j = i`` and both end up holding the old ``j``."""
    from nestforge.ir.emit_numpy import symbol_mapping_lines

    namespace = {"i": 1, "j": 2}
    for line in symbol_mapping_lines({"i": "j", "j": "i"}, 7):
        exec(line, {}, namespace)
    assert (namespace["i"], namespace["j"]) == (2, 1)


def test_symbol_mapping_stays_plain_when_nothing_interferes():
    """Temps only where they are needed -- the plain form is what the reader and the C translator see."""
    from nestforge.ir.emit_numpy import symbol_mapping_lines

    assert symbol_mapping_lines({"a": "N", "b": "M + 1"}, 3) == ["a = N", "b = M + 1"]
    assert symbol_mapping_lines({"i": "i"}, 3) == [], "an identity binding emits nothing"


def test_a_non_final_unconditional_branch_is_refused():
    """DaCe's codegen refuses an unconditional branch before a keyed one; reordering it would make an
    unreachable branch live."""
    from dace.properties import CodeBlock
    from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
    from nestforge.ir.emit_numpy import UnsupportedNest, emit_conditional

    sdfg = dace.SDFG("branch_order")
    sdfg.add_array("out", [2], dace.float64)
    block = ConditionalBlock("cond", sdfg=sdfg)
    sdfg.add_node(block, is_start_block=True)
    for label, condition in (("first", "N > 0"), ("always", None), ("dead", "N < 0")):
        body = ControlFlowRegion(label, sdfg=sdfg)
        state = body.add_state(f"{label}_state", is_start_block=True)
        tasklet = state.add_tasklet(label, {}, {"o"}, "o = 1.0")
        state.add_edge(tasklet, "o", state.add_write("out"), None, dace.Memlet("out[0]"))
        block.add_branch(None if condition is None else CodeBlock(condition), body)

    with pytest.raises(UnsupportedNest, match="unconditional branch"):
        emit_conditional(block, sdfg)


def test_int_floor_is_emitted_as_the_operator_not_a_call():
    """``//`` already floors for both signs, and a translator reads ``ast.FloorDiv`` where a bare call is an
    unknown name; ``int_ceil`` has no operator and stays a call."""
    assert normalize_casts("int_floor(a, b)") == "((a) // (b))"
    assert normalize_casts("A[int_floor(i, 2)]") == "A[((i) // (2))]"
    assert "int_ceil(" in normalize_casts("int_ceil(n, 4)")


@pytest.mark.parametrize("a,b", [(7, 2), (-7, 2), (7, -2), (-7, -2), (8, 4), (-8, 4), (0, 3)])
def test_the_operator_agrees_with_the_helper_on_both_signs(a, b):
    """The rewrite is only safe because it is the SAME function; C's `/` is where they part company."""
    rendered = normalize_casts(f"int_floor({a}, {b})")
    assert eval(rendered) == int_floor(a, b)  # noqa: S307 -- a literal expression this test built


def test_classic_c_scalar_cast_names_resolve_through_dace_own_alias_table():
    """DaCe's C++ codegen accepts ``double``, ``long`` and ``short`` as casts, so the oracle must too."""
    sdfg = dace.SDFG("ccast")
    sdfg.add_array("a", [4], dace.float64)
    sdfg.add_array("out_double", [4], dace.float64)
    sdfg.add_array("out_long", [4], dace.int64)
    sdfg.add_array("out_short", [4], dace.int16)
    state = sdfg.add_state()
    me, mx = state.add_map("m", dict(i="0:4"))
    t = state.add_tasklet(
        "t",
        {"inp": None},
        {"rd": None, "rl": None, "rs": None},
        "rd = double(inp)\nrl = long(inp)\nrs = short(inp)",
    )
    state.add_memlet_path(state.add_read("a"), me, t, dst_conn="inp", memlet=dace.Memlet("a[i]"))
    state.add_memlet_path(t, mx, state.add_write("out_double"), src_conn="rd", memlet=dace.Memlet("out_double[i]"))
    state.add_memlet_path(t, mx, state.add_write("out_long"), src_conn="rl", memlet=dace.Memlet("out_long[i]"))
    state.add_memlet_path(t, mx, state.add_write("out_short"), src_conn="rs", memlet=dace.Memlet("out_short[i]"))
    sdfg.validate()
    src = sdfg_to_numpy(sdfg, "ccast")
    assert "double(" not in src and "long(" not in src and "short(" not in src  # rewritten, not left bare
    mod = load_emitted(src, "ccast")
    a = np.array([1.9, -1.9, 300.7, -2.5])
    out_double, out_long, out_short = np.zeros(4), np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int16)
    mod.ccast(a, out_double, out_long, out_short)
    np.testing.assert_array_equal(out_double, a.astype(np.float64))
    np.testing.assert_array_equal(out_long, a.astype(np.int64))
    np.testing.assert_array_equal(out_short, a.astype(np.int16))


def test_bare_math_prefix_call_is_emitted_and_runnable():
    """``math.hypot`` has no NumPy replacement and stays ``math.<fn>``, so the kernel needs ``math`` bound."""
    sdfg = dace.SDFG("mathcall")
    sdfg.add_array("a", [3], dace.float64)
    sdfg.add_array("out", [3], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map("m", dict(i="0:3"))
    t = state.add_tasklet("t", {"inp": None}, {"res": None}, "res = math.hypot(inp, 1.0)")
    state.add_memlet_path(state.add_read("a"), me, t, dst_conn="inp", memlet=dace.Memlet("a[i]"))
    state.add_memlet_path(t, mx, state.add_write("out"), src_conn="res", memlet=dace.Memlet("out[i]"))
    sdfg.validate()
    src = sdfg_to_numpy(sdfg, "mathcall")
    assert "math.hypot(" in src
    mod = load_emitted(src, "mathcall")
    a = np.array([3.0, 4.0, 0.0])
    out = np.zeros(3)
    mod.mathcall(a, out)
    np.testing.assert_allclose(out, np.hypot(a, 1.0))


def test_loop_init_statement_without_assignment_is_refused_not_indexerror():
    """``symbol_ranges`` reads a loop's init statement to find its lower bound; a statement with no
    ``=`` (initialization happens elsewhere) used to raise a bare ``IndexError`` from
    ``.split("=", 1)[1]`` instead of a named, actionable refusal."""
    loop = LoopRegion("loop", condition_expr="i < N", loop_var="i", initialize_expr="i")
    with pytest.raises(UnsupportedNest, match="init statement"):
        loop_init_value(loop)


def test_a_map_whose_body_emits_only_comments_still_loads():
    """A ``for`` whose body is only provenance comments is an IndentationError, so it needs a ``pass``."""
    sdfg = dace.SDFG("noop_map")
    sdfg.add_array("A", [4], dace.float64)
    state = sdfg.add_state()
    entry, exit_ = state.add_map("m", dict(i="0:4"))
    tasklet = state.add_tasklet("noop", {}, {}, "pass")
    state.add_nedge(entry, tasklet, dace.Memlet())
    state.add_nedge(tasklet, exit_, dace.Memlet())

    src = sdfg_to_numpy(sdfg, "noop_map")

    assert "pass" in src
    load_emitted(src, "noop_map")


@pytest.mark.parametrize(("a", "want"), [(0.9, 1.0), (0.1, 7.0)])
def test_a_branch_condition_reads_a_scalar_transient_as_its_local(a, want):
    """A scalar transient is emitted as a plain local, so ``t[0]`` in a condition would index a float."""
    from dace.properties import CodeBlock
    from dace.sdfg.state import ConditionalBlock, ControlFlowRegion

    sdfg = dace.SDFG("scalar_guard")
    sdfg.add_array("A", [1], dace.float64)
    sdfg.add_array("B", [1], dace.float64)
    sdfg.add_scalar("t", dace.float64, transient=True)
    load = sdfg.add_state("load", is_start_block=True)
    tasklet = load.add_tasklet("load", {"a"}, {"o"}, "o = a")
    load.add_edge(load.add_read("A"), None, tasklet, "a", dace.Memlet("A[0]"))
    load.add_edge(tasklet, "o", load.add_write("t"), None, dace.Memlet("t[0]"))
    guard = ConditionalBlock("guard", sdfg=sdfg)
    sdfg.add_node(guard)
    sdfg.add_edge(load, guard, dace.InterstateEdge())
    body = ControlFlowRegion("then", sdfg=sdfg)
    store = body.add_state("store", is_start_block=True)
    one = store.add_tasklet("one", {}, {"o"}, "o = 1.0")
    store.add_edge(one, "o", store.add_write("B"), None, dace.Memlet("B[0]"))
    guard.add_branch(CodeBlock("t[0] > 0.5"), body)
    kernel = vars(load_emitted(sdfg_to_numpy(sdfg, "scalar_guard"), "scalar_guard"))["scalar_guard"]
    B = np.array([7.0])

    kernel(A=np.array([a]), B=B)

    assert B[0] == want
