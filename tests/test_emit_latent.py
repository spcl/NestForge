"""Regression tests for LATENT emit bugs found in the full-repo audit -- the classes that produce a WRONG
or uncompilable numpy oracle rather than an obvious crash. Unit set, no compile.

Each test fails without its fix:
  * sympy ``Max``/``Min``/``Abs`` reached the oracle verbatim (only ``np`` is in scope -> NameError);
  * an empty (data=None) happens-before edge was taken as a library node's operand -> ``arrays[None]``;
  * an unstructured join double-applied every predecessor's inter-state assignments;
  * an array dtype with no extern-C spelling KeyError'd mid-codegen.
"""
from dataclasses import dataclass

import numpy as np
import pytest
import sympy
import dace
from dace import symbolic

from nestforge.emit_libnode import UnsupportedLibraryNode, data_edge
from nestforge.emit_numpy import UnsupportedNest, normalize_casts, sdfg_to_numpy
from nestforge.libnode import ExternalCall, proto_and_call

I = sympy.Symbol('i')
N = sympy.Symbol('N')


def evaluated(expr, **env):
    """symstr the sympy expression the way a subset/bound reaches the emitter, normalize it, and evaluate
    the emitted text in the SAME namespace the oracle runs in (builtins + np only)."""
    code = normalize_casts(symbolic.symstr(expr))
    assert "Max(" not in code and "Min(" not in code and "Abs(" not in code, f"un-rewritten sympy func: {code}"
    return eval(code, {"np": np}, dict(env))


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


def extern_call_with_dtype(dtype_name: str, shape=(8, )):
    """An ExternalCall wired into a real state: ``proto_and_call`` reads the memlets to tell a pointer
    connector from a value one, so a node without edges cannot answer the question it is asked."""
    manifest = {
        "array_args": ["A"],
        "output_args": [],
        "init": {
            "arrays": {
                "A": {
                    "dtype": dtype_name
                }
            },
            "scalars": {}
        },
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
    _, state = extern_call_with_dtype("float64", shape=(1, ))
    node = next(n for n in state.nodes() if isinstance(n, ExternalCall))
    proto, call = proto_and_call(node, state)
    assert "const double* A" in proto
    assert call == "k_fp64(&_in_A);"


def test_unspellable_array_dtype_is_refused_not_keyerror():
    # a complex/fp16/unsigned array used to raise a bare KeyError from _CPP_SCALAR mid-codegen; it must be an
    # actionable refusal naming the array and dtype so the caller can keep the DaceReference variant.
    with pytest.raises(ValueError, match="complex128"):
        proto_and_call(*extern_call_with_dtype("complex128"))
