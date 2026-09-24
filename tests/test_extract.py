# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import dace
from dace.sdfg.state import LoopRegion

from nestforge.phases.scopes import parallel_top_level_maps
from nestforge.ir.extract import extract_nest_to_sdfg, nest_defined_symbol_dtypes

N = dace.symbol("N")


@dace.program
def vadd(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + B[i]


def test_outer_finds_the_map():
    sdfg = vadd.to_sdfg(simplify=True)
    refs = parallel_top_level_maps(sdfg)
    assert len(refs) == 1
    _, node = refs[0]
    assert isinstance(node, dace.sdfg.nodes.MapEntry)


def test_extract_map_nest_boundary_and_correctness():
    sdfg = vadd.to_sdfg(simplify=True)
    psdfg, node = parallel_top_level_maps(sdfg)[0]
    b = extract_nest_to_sdfg(psdfg, node, name="vadd_nest")

    assert set(b.inputs) == {"A", "B"}
    assert set(b.outputs) == {"C"}
    assert "N" in b.symbols

    standalone = b.standalone_sdfg
    for name in ("A", "B", "C"):
        assert name in standalone.arrays

    # The parent SDFG (now holding the NestedSDFG) still computes vadd.
    A = np.random.default_rng(0).random(16)
    B = np.random.default_rng(1).random(16)
    C = np.zeros(16)
    sdfg(A=A, B=B, C=C, N=16)
    np.testing.assert_allclose(C, A + B)

    # The standalone SDFG computes vadd on its own.
    A2 = np.random.default_rng(2).random(16)
    B2 = np.random.default_rng(3).random(16)
    C2 = np.zeros(16)
    standalone(A=A2, B=B2, C=C2, N=16)
    np.testing.assert_allclose(C2, A2 + B2)


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
    """``twice = scale * 2`` after ``scale = 1.5`` is a double; typed without ``scale`` it fell back to int64 and
    the parent would truncate it."""
    sdfg, loop = loop_with_exported_float_assignment()
    after_body = next(b for b in loop.nodes() if b.label == "after_body")
    loop.add_state_after(after_body, "doubled", assignments={"twice": "scale * 2"})

    assert nest_defined_symbol_dtypes(sdfg, loop) == {"scale": dace.float64, "twice": dace.float64}
