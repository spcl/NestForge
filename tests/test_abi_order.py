# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A kernel is called in the order its compiled signature declares, never the manifest's role order.

The translator sorts arrays, then scalars; the manifest lists inputs, outputs, symbols. Both are ``double*``, so a
mismatch writes through the wrong pointer without any error.
"""

import pytest

import dace

from nestforge.ir.extract import extract_nest_to_sdfg
from helpers import signature_order
from nestforge.ir.libnode import ExternLibEnv, ExternalCall, proto_and_call
from nestforge.phases.scopes import parallel_top_level_maps
from nestforge.corpus.translate import emit_sources, prepare

N = dace.symbol("N")


@dace.program
def writes_a_reads_b(a: dace.float64[N], b: dace.float64[N]):
    """The output sorts before the input, so role order ``[b, a, N]`` and sorted order ``[a, b, N]`` differ."""
    for i in dace.map[0:N]:
        a[i] = b[i] + 1.0


def prepared_nest(tmp_path):
    sdfg = writes_a_reads_b.to_sdfg(simplify=True)
    parent, node = parallel_top_level_maps(sdfg)[0]
    boundary = extract_nest_to_sdfg(parent, node, name="wab")
    prep = prepare(boundary, "wab", tmp_path, sizes={"N": 32})
    return prep, boundary


@pytest.mark.integration  # runs the numpyto emitter
def test_emitted_signature_disagrees_with_manifest_role_order(tmp_path):
    """The hazard itself: if the two orders ever coincide for this kernel, the fixture stops testing anything."""
    prep, boundary = prepared_nest(tmp_path)
    csrc = next(s for s in emit_sources(prep, tmp_path, target="c") if s.suffix == ".c" and "pluto" not in s.name)
    emitted = signature_order(csrc.read_text(), "wab_fp64")
    assert emitted == ["a", "b", "N"]  # sorted arrays, then scalars
    assert list(prep.manifest["input_args"]) == ["b", "a", "N"]  # role order: input, output, symbol
    assert emitted != list(prep.manifest["input_args"]), (
        "the two orders now coincide for this kernel -- pick one whose output still sorts before its input"
    )


def test_a_prototype_above_the_definition_does_not_widen_the_capture():
    """A non-greedy match backtracks across a preceding prototype and captures a garbage parameter list."""
    from nestforge.build.toolchain import raw_signature

    declared_then_defined = (
        "void s000_fp64(double *restrict a, const double *restrict b, int64_t LEN_1D);\n"
        "\n"
        'extern "C" void s000_fp64(double *restrict a, const double *restrict b, '
        "int64_t LEN_1D) {\n  return;\n}\n"
    )
    params = raw_signature(declared_then_defined, "s000_fp64")
    assert params == "double *restrict a, const double *restrict b, int64_t LEN_1D"
    assert ";" not in params and "void" not in params

    # a comment naming the entry is not its definition
    commented = "// s000_fp64 (s000): copy b into a\nvoid s000_fp64(double *a, int64_t n) {\n}\n"
    assert raw_signature(commented, "s000_fp64") == "double *a, int64_t n"


def extern_call(abi_order, inputs):
    manifest = {
        "array_args": list(abi_order),
        "output_args": [],
        "init": {"arrays": {a: {"dtype": "float64"} for a in abi_order}, "scalars": {}},
    }
    node = ExternalCall("k", inputs=set(inputs), outputs=set(), config=manifest)
    node.symbol, node.abi_order = "k_fp64", list(abi_order)
    sdfg = dace.SDFG("host")
    state = sdfg.add_state()
    state.add_node(node)
    for conn in inputs:
        name = conn[len("_in_") :]
        sdfg.add_array(name, [8], dace.float64)
        state.add_edge(state.add_read(name), None, node, conn, dace.Memlet.from_array(name, sdfg.arrays[name]))
    return node, state


def test_abi_arg_without_a_connector_is_refused():
    """A scratch buffer in the compiled signature never crosses the node's boundary; the call would name nothing."""
    with pytest.raises(ValueError, match="no '_in_scratch' connector"):
        proto_and_call(*extern_call(["A", "scratch"], inputs=["_in_A"]))


def test_extern_lib_env_accumulates_every_nest_library():
    """Every kernel shares the environment class, so a later kernel must not replace an earlier one's library."""
    ExternLibEnv.reset()
    assert ExternLibEnv.cmake_libraries == []
    ExternLibEnv.configure("/tmp/libone_nest.so")
    ExternLibEnv.configure("/tmp/libtwo_nest.so")
    assert ExternLibEnv.cmake_libraries == ["/tmp/libone_nest.so", "/tmp/libtwo_nest.so"]
    ExternLibEnv.configure("/tmp/libone_nest.so")  # deduplicated
    assert len(ExternLibEnv.cmake_libraries) == 2
    ExternLibEnv.reset()
    assert ExternLibEnv.cmake_libraries == [] and ExternLibEnv.cmake_link_flags == []
