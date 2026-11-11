# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The emitted Python oracle computes what DaCe's compiled SDFG computes, one C or DaCe semantic per tiny SDFG:
write-conflict resolutions, reductions, offsets, C arithmetic, local dtypes, library subsets and nested scratch."""

import copy

import dace
import numpy as np
from dace.libraries.standard import Reduce
from dace.sdfg.state import LoopRegion
from helpers import run_emitted

from nestforge.ir.emit_python import hoist_nested_scratch, lower, render, scratch_arrays
from nestforge.ir.extract import blocks_defined_symbol_dtypes

N = dace.symbol("N", dtype=dace.int64)
SUM = "lambda x, y: x + y"


def assert_oracle_matches_dace(sdfg: dace.SDFG, inputs: dict[str, np.ndarray], sizes: dict[str, int]) -> str:
    """Run the oracle (lowered and hoisted as for a nest) and the compiled SDFG on copies of ``inputs``; every
    argument array must agree. Returns the emitted source."""
    lowered = copy.deepcopy(sdfg)
    lower(lowered)
    hoist_nested_scratch(lowered)
    data = [name for name, desc in sdfg.arrays.items() if not desc.transient]
    source = render("k", [*data, *scratch_arrays(lowered), *sizes], lowered)
    oracle = run_emitted(source, "k", lowered, copy.deepcopy(inputs), sizes)
    compiled = copy.deepcopy(inputs)
    sdfg(**compiled, **sizes)
    for name in data:
        np.testing.assert_array_equal(oracle[name], compiled[name], err_msg=f"{name} differs\n{source}")
    return source


def test_a_dynamic_wcr_write_the_tasklet_skips_combines_nothing():
    sdfg = dace.SDFG("sem_dynamic_wcr")
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("B", [1], dace.float64)
    state = sdfg.add_state()
    entry, exit_ = state.add_map("m", {"i": "0:N"})
    tasklet = state.add_tasklet("t", {"a"}, {"o"}, "if a > 0:\n    o = a")
    state.add_memlet_path(state.add_read("A"), entry, tasklet, dst_conn="a", memlet=dace.Memlet("A[i]"))
    write = dace.Memlet("B[0]", wcr=SUM, dynamic=True)
    state.add_memlet_path(tasklet, exit_, state.add_write("B"), src_conn="o", memlet=write)

    assert_oracle_matches_dace(sdfg, {"A": np.array([1.0, -1.0, 2.0, -5.0]), "B": np.full(1, 10.0)}, {"N": 4})


def test_a_ranged_wcr_write_combines_each_element_once():
    sdfg = dace.SDFG("sem_ranged_wcr")
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("B", [2], dace.float64)
    state = sdfg.add_state()
    entry, exit_ = state.add_map("m", {"i": "0:N"})
    tasklet = state.add_tasklet("t", {"a"}, {"o"}, "o[0] = a\no[1] = 2 * a")
    state.add_memlet_path(state.add_read("A"), entry, tasklet, dst_conn="a", memlet=dace.Memlet("A[i]"))
    write = dace.Memlet("B[0:2]", wcr=SUM)
    state.add_memlet_path(tasklet, exit_, state.add_write("B"), src_conn="o", memlet=write)

    assert_oracle_matches_dace(sdfg, {"A": np.array([1.0, 2.0, 3.0]), "B": np.array([10.0, 20.0])}, {"N": 3})


def test_a_reduce_without_identity_accumulates_into_its_output():
    sdfg = dace.SDFG("sem_reduce_no_identity")
    sdfg.add_array("A", [4], dace.float64)
    sdfg.add_array("B", [1], dace.float64)
    state = sdfg.add_state()
    reduce = Reduce("r", SUM, None, None)
    state.add_node(reduce)
    state.add_edge(state.add_read("A"), None, reduce, None, dace.Memlet("A[0:4]"))
    state.add_edge(reduce, None, state.add_write("B"), None, dace.Memlet("B[0]"))

    source = assert_oracle_matches_dace(sdfg, {"A": np.ones(4), "B": np.full(1, 10.0)}, {})

    assert "B[0] = B[0] + np.add.reduce(" in source


def test_a_library_subset_spelled_with_min_is_python():
    sdfg = dace.SDFG("sem_library_min")
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("B", [1], dace.float64)
    state = sdfg.add_state()
    reduce = Reduce("r", SUM, None, 0)
    state.add_node(reduce)
    state.add_edge(state.add_read("A"), None, reduce, None, dace.Memlet("A[0:min(N, 3)]"))
    state.add_edge(reduce, None, state.add_write("B"), None, dace.Memlet("B[0]"))

    source = assert_oracle_matches_dace(sdfg, {"A": np.arange(5.0), "B": np.zeros(1)}, {"N": 5})

    assert "Min(" not in source


def test_an_offset_array_is_indexed_from_its_offset():
    sdfg = dace.SDFG("sem_offset")
    sdfg.add_array("A", [3], dace.float64, offset=[-1])
    sdfg.add_array("B", [3], dace.float64, offset=[-1])
    state = sdfg.add_state()
    state.add_mapped_tasklet(
        "m", {"i": "1:4"}, {"a": dace.Memlet("A[i]")}, "b = a * 2", {"b": dace.Memlet("B[i]")}, external_edges=True
    )

    assert_oracle_matches_dace(sdfg, {"A": np.array([1.0, 2.0, 3.0]), "B": np.zeros(3)}, {})


def integer_tasklet(name: str, code: str) -> dace.SDFG:
    """``b = code`` over ``a``, both ``int64`` elements."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [1], dace.int64)
    sdfg.add_array("B", [1], dace.int64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet("t", {"a"}, {"b"}, code)
    state.add_edge(state.add_read("A"), None, tasklet, "a", dace.Memlet("A[0]"))
    state.add_edge(tasklet, "b", state.add_write("B"), None, dace.Memlet("B[0]"))
    return sdfg


def test_tasklet_modulo_truncates_as_in_c():
    sdfg = integer_tasklet("sem_c_mod", "b = a % 3")

    assert_oracle_matches_dace(sdfg, {"A": np.array([-4]), "B": np.zeros(1, np.int64)}, {})


def test_tasklet_integer_division_truncates_as_in_c():
    sdfg = integer_tasklet("sem_c_div", "b = (a / 2) * 2")

    assert_oracle_matches_dace(sdfg, {"A": np.array([-7]), "B": np.zeros(1, np.int64)}, {})


def test_a_transient_scalar_holds_its_own_dtype():
    sdfg = dace.SDFG("sem_local_dtype")
    sdfg.add_array("A", [1], dace.int64)
    sdfg.add_array("B", [1], dace.float64)
    sdfg.add_scalar("t", dace.int64, transient=True)
    state = sdfg.add_state()
    half, double = (
        state.add_tasklet("h", {"a"}, {"o"}, "o = a * 0.5"),
        state.add_tasklet("d", {"x"}, {"o"}, "o = x * 2.0"),
    )
    scalar = state.add_access("t")
    state.add_edge(state.add_read("A"), None, half, "a", dace.Memlet("A[0]"))
    state.add_edge(half, "o", scalar, None, dace.Memlet("t[0]"))
    state.add_edge(scalar, None, double, "x", dace.Memlet("t[0]"))
    state.add_edge(double, "o", state.add_write("B"), None, dace.Memlet("B[0]"))

    assert_oracle_matches_dace(sdfg, {"A": np.array([3]), "B": np.zeros(1)}, {})


def rounding_map(name: str, code: str, dtype: type[np.generic]) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [5], dace.float64)
    sdfg.add_array("B", [5], dtype)
    sdfg.add_state().add_mapped_tasklet(
        "m", {"i": "0:5"}, {"a": dace.Memlet("A[i]")}, code, {"b": dace.Memlet("B[i]")}, external_edges=True
    )
    return sdfg


HALVES = np.array([0.5, 1.5, 2.5, -2.5, 0.49999999999999994])


def test_round_rounds_halves_away_from_zero_as_std_round():
    sdfg = rounding_map("sem_round", "b = round(a)", dace.float64)

    assert_oracle_matches_dace(sdfg, {"A": HALVES, "B": np.zeros(5)}, {})


def test_iround_rounds_halves_away_from_zero_to_an_integer():
    sdfg = rounding_map("sem_iround", "b = iround(a)", dace.int32)

    assert_oracle_matches_dace(sdfg, {"A": HALVES, "B": np.zeros(5, np.int32)}, {})


def two_state_nest(name: str, second: str, write: dace.Memlet) -> dace.SDFG:
    """``t = (2x[i], x[i] + 1)`` in a scratch ``t[2]``, then ``second`` over ``p = t[0]``, ``q = t[1]`` into
    ``write``."""
    inner = dace.SDFG(name)
    inner.add_array("x", [N], dace.float64)
    inner.add_array("y", [N], dace.float64)
    inner.add_array("t", [2], dace.float64, transient=True)
    inner.add_symbol("i", dace.int64)
    first = inner.add_state(is_start_block=True)
    tasklet = first.add_tasklet("f", {"v"}, {"o", "w"}, "o = 2 * v\nw = v + 1")
    first.add_edge(first.add_read("x"), None, tasklet, "v", dace.Memlet("x[i]"))
    scratch = first.add_write("t")
    first.add_edge(tasklet, "o", scratch, None, dace.Memlet("t[0]"))
    first.add_edge(tasklet, "w", scratch, None, dace.Memlet("t[1]"))
    then = inner.add_state_after(first)
    combine = then.add_tasklet("s", {"p", "q"}, {"o"}, second)
    reads = then.add_read("t")
    then.add_edge(reads, None, combine, "p", dace.Memlet("t[0]"))
    then.add_edge(reads, None, combine, "q", dace.Memlet("t[1]"))
    then.add_edge(combine, "o", then.add_write("y"), None, write)
    return inner


def nest_per_element(name: str, inner: dace.SDFG, out: dace.Memlet) -> tuple[dace.SDFG, dace.SDFGState]:
    """A map over ``i`` whose body is ``inner``, binding ``a`` as ``x`` and ``y`` to ``out``."""
    outer = dace.SDFG(name)
    outer.add_array("a", [N], dace.float64)
    outer.add_array("b", [N], dace.float64)
    state = outer.add_state()
    entry, exit_ = state.add_map("each", {"i": "0:N"})
    node = state.add_nested_sdfg(inner, {"x"}, {"y"}, {"N": "N", "i": "i"})
    state.add_memlet_path(state.add_read("a"), entry, node, dst_conn="x", memlet=dace.Memlet("a[0:N]"))
    state.add_memlet_path(node, exit_, state.add_write("b"), src_conn="y", memlet=out)
    return outer, state


def test_a_nested_scratch_never_shares_a_kernel_argument_of_its_name():
    inner = two_state_nest("sem_scratch_inner", "o = p + q", dace.Memlet("y[i]"))
    outer, state = nest_per_element("sem_scratch", inner, dace.Memlet("b[0:N]"))
    outer.add_array("t", [2], dace.float64)  # the caller's, same name, dtype and shape as the nest's scratch
    outer.add_array("c", [1], dace.float64)
    after = outer.add_state_after(state)
    total = after.add_tasklet("c", {"p", "q"}, {"o"}, "o = p + q")
    after.add_edge(after.add_read("t"), None, total, "p", dace.Memlet("t[0]"))
    after.add_edge(after.add_read("t"), None, total, "q", dace.Memlet("t[1]"))
    after.add_edge(total, "o", after.add_write("c"), None, dace.Memlet("c[0]"))
    a = np.array([1.0, 2.0, 3.0])

    assert_oracle_matches_dace(outer, {"a": a, "b": np.zeros(3), "t": np.array([5.0, 7.0]), "c": np.zeros(1)}, {"N": 3})


def test_a_nest_that_repeats_its_max_wcr_inside_is_inlined_with_the_bound_name():
    write = dace.Memlet("y[0]", wcr="lambda x, y: max(x, y)")
    inner = two_state_nest("sem_max_inner", "o = p + q", write)
    outer, _ = nest_per_element("sem_max", inner, dace.Memlet("b[0:N]", wcr="lambda x, y: max(x, y)"))
    a = np.array([1.0, -2.0, 3.0])

    assert_oracle_matches_dace(outer, {"a": a, "b": np.full(3, -100.0)}, {"N": 3})


def test_a_symbol_a_later_loop_derives_from_an_earlier_loop_is_typed_from_it():
    sdfg = dace.SDFG("sem_block_dtypes")
    loops = []
    for name, assignment in (("first", {"s": "1.5"}), ("second", {"u": "s * 2"})):
        loop = LoopRegion(name, f"{name}_i < 3", f"{name}_i", f"{name}_i = 0", f"{name}_i = {name}_i + 1")
        sdfg.add_node(loop, is_start_block=not loops)
        if loops:
            sdfg.add_edge(loops[-1], loop, dace.InterstateEdge())
        body = loop.add_state(f"{name}_body", is_start_block=True)
        loop.add_state_after(body, f"{name}_assign", assignments=assignment)
        loops.append(loop)

    assert blocks_defined_symbol_dtypes(sdfg, loops) == {"s": dace.float64, "u": dace.float64}


if __name__ == "__main__":
    test_a_dynamic_wcr_write_the_tasklet_skips_combines_nothing()
    test_a_ranged_wcr_write_combines_each_element_once()
    test_a_reduce_without_identity_accumulates_into_its_output()
    test_a_library_subset_spelled_with_min_is_python()
    test_an_offset_array_is_indexed_from_its_offset()
    test_tasklet_modulo_truncates_as_in_c()
    test_tasklet_integer_division_truncates_as_in_c()
    test_a_transient_scalar_holds_its_own_dtype()
    test_round_rounds_halves_away_from_zero_as_std_round()
    test_iround_rounds_halves_away_from_zero_to_an_integer()
    test_a_nested_scratch_never_shares_a_kernel_argument_of_its_name()
    test_a_nest_that_repeats_its_max_wcr_inside_is_inlined_with_the_bound_name()
    test_a_symbol_a_later_loop_derives_from_an_earlier_loop_is_typed_from_it()
