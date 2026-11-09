# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
import dace
import numpy as np
from dace.sdfg.state import LoopRegion
from dace.symbolic_facts import Predicate
from helpers import extract_nest_to_sdfg

from nestforge.ir.extract import Boundary, nest_defined_symbol_dtypes
from nestforge.stages.scopes import lower_nests_to_external_call, parallel_top_level_maps

N = dace.symbol("N")


@dace.program
def vadd(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + B[i]


def extracted_vadd() -> tuple[dace.SDFG, Boundary]:
    sdfg = vadd.to_sdfg(simplify=True)
    psdfg, node = parallel_top_level_maps(sdfg)[0]
    return sdfg, extract_nest_to_sdfg(psdfg, node, name="vadd_nest")


def run_vadd(sdfg: dace.SDFG, seed: int) -> None:
    rng = np.random.default_rng(seed)
    A, B, C = rng.random(16), rng.random(16), np.zeros(16)
    sdfg(A=A, B=B, C=C, N=16)
    np.testing.assert_allclose(C, A + B, rtol=0, atol=0)


def test_an_extracted_map_nest_reads_a_and_b_and_writes_c():
    _, b = extracted_vadd()

    assert (set(b.inputs), set(b.outputs)) == ({"A", "B"}, {"C"})
    assert "N" in b.symbols
    assert {"A", "B", "C"} <= set(b.standalone_sdfg.arrays)


def test_the_parent_still_computes_after_extraction():
    sdfg, _ = extracted_vadd()
    run_vadd(sdfg, seed=0)


def test_the_standalone_nest_computes_on_its_own():
    _, b = extracted_vadd()
    run_vadd(b.standalone_sdfg, seed=1)


def positive_size_program() -> tuple[dace.SDFG, dace.SDFG]:
    """A program whose size ``N`` is declared positive, with a map over a nested SDFG that assumes ``0 < N``."""
    positive = frozenset({Predicate.POSITIVE})
    inner = dace.SDFG("assumes_positive")
    inner.add_symbol("N", dace.int64, predicates=positive)
    inner.add_array("a", [N], dace.float64)
    body = inner.add_state()
    write = body.add_tasklet("write", {}, {"o"}, "o = 1.0")
    body.add_edge(write, "o", body.add_write("a"), None, dace.Memlet("a[0]"))

    outer = dace.SDFG("positive_size")
    outer.add_symbol("N", dace.int64, predicates=positive)
    outer.add_array("A", [N], dace.float64)
    state = outer.add_state()
    entry, leave = state.add_map("rows", {"i": "0:2"})  # a range over 0:N would itself prove N > 0
    nested = state.add_nested_sdfg(inner, set(), {"a"}, symbol_mapping={"N": "N"})
    state.add_nedge(entry, nested, dace.Memlet())
    state.add_memlet_path(nested, leave, state.add_write("A"), src_conn="a", memlet=dace.Memlet("A[0:N]"))
    return outer, inner


def test_an_extracted_nest_keeps_the_sign_its_program_declares_for_a_size():
    """Cutting the parent dropped ``N > 0``, so the nested SDFG that assumes it no longer validated in the kernel."""
    outer, _ = positive_size_program()
    psdfg, node = parallel_top_level_maps(outer)[0]

    b = extract_nest_to_sdfg(psdfg, node, name="positive_nest")

    assert b.standalone_sdfg.symbol_repo.params.predicates["N"] == frozenset({Predicate.POSITIVE})
    b.standalone_sdfg.validate()


def test_the_kernel_a_nest_becomes_keeps_the_sign_its_program_declares_for_a_size():
    """The kernel DaCe builds is ``ExternalCall.standalone_sdfg``, a copy that cut the parent as well."""
    outer, _ = positive_size_program()

    ((call, _),) = lower_nests_to_external_call(outer)

    assert call.standalone_sdfg.symbol_repo.params.predicates["N"] == frozenset({Predicate.POSITIVE})
    call.standalone_sdfg.validate()


# loop iterators are scope symbols, never SDFG symbols


def loop_with_exported_int32_iterator():
    """A loop whose int32 iterator is read after the loop, so DaCe's nesting helper exports it with its type."""
    sdfg = dace.SDFG("loopnest")
    sdfg.add_array("a", [20], dace.float64)
    sdfg.add_symbol("base", dace.int32)
    sdfg.add_symbol("bound", dace.int32)
    sdfg.add_symbol("step", dace.int32)
    loop = LoopRegion("loop", "loop_i <= bound", "loop_i", "loop_i = base", "loop_i = loop_i + step")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    tasklet = body.add_tasklet("t", {}, {"o0"}, "o0 = 1.0")
    body.add_edge(tasklet, "o0", body.add_write("a"), None, dace.Memlet("a[loop_i]"))
    after = sdfg.add_state("after")
    sdfg.add_edge(loop, after, dace.InterstateEdge(assignments={"final": "loop_i"}))
    return sdfg, loop


def test_extract_cfg_nest_adds_no_symbol_for_the_loop_iterator():
    """A loop iterator is a scope symbol, so outlining its loop declares no SDFG symbol for it."""
    sdfg, loop = loop_with_exported_int32_iterator()
    extract_nest_to_sdfg(sdfg, loop, name="loopnest_extracted")
    assert "loop_i" not in sdfg.symbols


def test_extract_cfg_nest_keeps_a_non_int64_iterator_dtype():
    """An exported iterator keeps the int32 DaCe infers from the loop bounds."""
    sdfg, loop = loop_with_exported_int32_iterator()
    boundary = extract_nest_to_sdfg(sdfg, loop, name="loopnest_extracted")
    exported = next(name for name in sdfg.arrays if "loop_i" in name)
    assert sdfg.arrays[exported].dtype == dace.int32
    assert boundary.standalone_sdfg.arrays[exported].dtype == dace.int32


def loop_with_exported_float_assignment():
    """A loop whose body assigns ``scale = 1.5`` on an interstate edge, read after the loop."""
    sdfg = dace.SDFG("floaty")
    sdfg.add_array("a", [20], dace.float64)
    loop = LoopRegion("loop", "loop_i < 10", "loop_i", "loop_i = 0", "loop_i = loop_i + 1")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    after_body = loop.add_state("after_body")
    loop.add_edge(body, after_body, dace.InterstateEdge(assignments={"scale": "1.5"}))
    tasklet = after_body.add_tasklet("t", {}, {"o0"}, "o0 = scale")
    after_body.add_edge(tasklet, "o0", after_body.add_write("a"), None, dace.Memlet("a[loop_i]"))
    after = sdfg.add_state("after")
    sdfg.add_edge(loop, after, dace.InterstateEdge(assignments={"final": "scale"}))
    return sdfg, loop


def test_extract_cfg_nest_still_types_a_genuine_interstate_assignment_target():
    """An interstate assignment target inside the nest is declared with its inferred dtype."""
    sdfg, loop = loop_with_exported_float_assignment()
    extract_nest_to_sdfg(sdfg, loop, name="floaty_extracted")
    assert sdfg.symbols["scale"] == dace.float64


def test_an_assignment_reading_an_earlier_target_gets_that_targets_dtype():
    """``twice = scale * 2`` after ``scale = 1.5`` is typed from ``scale``, so it is a double."""
    sdfg, loop = loop_with_exported_float_assignment()
    after_body = next(b for b in loop.nodes() if b.label == "after_body")
    loop.add_state_after(after_body, "doubled", assignments={"twice": "scale * 2"})

    assert nest_defined_symbol_dtypes(sdfg, loop) == {"scale": dace.float64, "twice": dace.float64}


def test_a_symbol_first_assigned_an_int_then_a_float_is_declared_double():
    sdfg = dace.SDFG("widening")
    loop = LoopRegion("loop", "loop_i < 10", "loop_i", "loop_i = 0", "loop_i = loop_i + 1")
    sdfg.add_node(loop, is_start_block=True)
    first = loop.add_state("first", is_start_block=True)
    second = loop.add_state_after(first, "second", assignments={"acc": "0"})
    loop.add_state_after(second, "third", assignments={"acc": "acc + 0.5"})

    assert nest_defined_symbol_dtypes(sdfg, loop) == {"acc": dace.float64}
