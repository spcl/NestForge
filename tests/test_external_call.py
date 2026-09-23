# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Lowering a nest to an ``ExternalCall`` and running it through the ``DaceReference`` expansion. The linked-library
expansion is covered by ``test_variants_phase.py``."""

import numpy as np
import pytest
import dace

from nestforge.build.arena import make_inputs, run_oracle
from nestforge.corpus.translate import prepare
from nestforge.phases.scopes import lower_nests_to_external_call, node_boundary
from nestforge.ir.emit_numpy import nest_to_numpy
from nestforge.ir.emit_yaml import manifest_dict
from nestforge.ir.libnode import ExternalCall

N = dace.symbol("N")


@dace.program
def vadd(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + B[i]


@dace.program
def overwrite(A: dace.float64[N], B: dace.float64[N], X: dace.float64[N], C: dace.float64[N]):
    X[:] = A + 1
    X[:] = B + 2
    C[:] = X * 3


@dace.program
def scale_in_place(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:N]:
        A[i] = A[i] * 2.0 + B[i]


def reference_outputs(n):
    A = np.random.default_rng(0).random(n)
    B = np.random.default_rng(1).random(n)
    return A, B, A + B


def test_lower_inserts_external_call():
    sdfg = vadd.to_sdfg(simplify=True)
    lowered = lower_nests_to_external_call(sdfg)
    assert len(lowered) == 1
    ext, boundary = lowered[0]
    assert isinstance(ext, ExternalCall)
    assert set(ext.in_connectors) == {"_in_A", "_in_B"}
    assert set(ext.out_connectors) == {"_out_C"}
    assert "def " in ext.numpy_source


def test_an_ordering_edge_into_a_lowered_nest_gains_no_connector():
    sdfg = overwrite.to_sdfg(simplify=True)

    lowered = lower_nests_to_external_call(sdfg)

    connectors = [
        conn
        for call, boundary in lowered
        for conn in (
            *call.in_connectors,
            *call.out_connectors,
            *(edge.dst_conn for edge in boundary.state.in_edges(call)),
            *(edge.src_conn for edge in boundary.state.out_edges(call)),
        )
    ]
    assert "_in_None" not in connectors and "_out_None" not in connectors
    ordering = [
        (edge.src.data, call.label, edge.dst_conn)
        for call, boundary in lowered
        for edge in boundary.state.in_edges(call)
        if edge.data.is_empty()
    ]
    assert ordering == [("X", "extcall_1", None)]


@pytest.mark.parametrize("program", [vadd, scale_in_place], ids=["plain", "in_place"])
def test_a_kernel_node_alone_rebuilds_the_boundary_its_manifest_and_oracle_came_from(program):
    sdfg = program.to_sdfg(simplify=True)
    ((ext, boundary),) = lower_nests_to_external_call(sdfg)

    rebuilt = node_boundary(ext)

    assert (rebuilt.inputs, rebuilt.outputs, rebuilt.symbols) == (boundary.inputs, boundary.outputs, boundary.symbols)
    assert manifest_dict(rebuilt, ext.name) == ext.config
    assert nest_to_numpy(rebuilt, fn_name=ext.name) == ext.numpy_source


def test_dace_reference_runs_correctly():
    sdfg = vadd.to_sdfg(simplify=True)
    lower_nests_to_external_call(sdfg)  # default impl = DaceReference
    sdfg.expand_library_nodes()
    sdfg.validate()
    n = 1 << 12
    A, B, ref = reference_outputs(n)
    C = np.zeros(n)
    sdfg(A=A, B=B, C=C, N=n)
    np.testing.assert_allclose(C, ref)


@dace.program
def scaley(A: dace.float64[N]):
    return A * 2.0


def test_returning_kernel_survives_arena_oracle_and_manifest_matches(tmp_path):
    sdfg = scaley.to_sdfg(simplify=True)
    ext, boundary = lower_nests_to_external_call(sdfg)[0]
    assert boundary.outputs == ["__return"]
    prep = prepare(boundary, ext.name, tmp_path / "k")
    # __return is an in-place buffer parameter in the numpy signature AND the manifest -- aligned.
    assert "__return" in prep.numpy_source.splitlines()[0]
    assert "return " not in prep.numpy_source
    # emit_yaml.arg_order and emit_numpy.nest_to_numpy build the signature independently, so the manifest is
    # only usable while they agree: arrays in array_args order (inputs, extra outputs, scratch), then symbols.
    header = prep.numpy_source.splitlines()[0]
    signature = [a.strip() for a in header[header.index("(") + 1 : header.rindex(")")].split(",")]
    args = list(prep.manifest["input_args"])
    arrays = list(prep.manifest["array_args"])
    assert args == arrays + [s for s in boundary.symbols if s not in arrays]
    assert args == signature, f"manifest input_args {args} != emitted numpy signature {signature}"
    assert "__return" in prep.manifest["input_args"]
    sizes = {"N": 16}
    out = run_oracle(prep, boundary, make_inputs(boundary, sizes), sizes)
    assert "__return" in out


@dace.program
def scale_inplace(A: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        A[i] = A[i] * 2.0 + b[i]


def inplace_lowered():
    sdfg = scale_inplace.to_sdfg(simplify=True)
    ext, boundary = lower_nests_to_external_call(sdfg)[0]
    assert "A" in boundary.inputs and "A" in boundary.outputs, "A must be read+written for this to test anything"
    return sdfg, ext


def test_inplace_nest_reference_sdfg_declares_every_connector():
    """An array both read and written gets ``_in_`` and ``_out_`` connectors, and the reference expansion must
    declare both or nested-SDFG validation rejects it."""
    sdfg, ext = inplace_lowered()
    arrays = ext.standalone_sdfg.arrays
    for conn in set(ext.in_connectors) | set(ext.out_connectors):
        assert conn in arrays, f"connector {conn} has no descriptor in the reference SDFG: {sorted(arrays)}"
    sdfg.expand_library_nodes()
    sdfg.validate()  # raised InvalidSDFGNodeError('Connector "_out_A" ... not a registered data descriptor')


@pytest.mark.integration  # compiles + runs the DaceReference expansion
def test_inplace_nest_reference_expansion_is_value_preserving():
    """The reference expansion must not just validate, it must still compute: the body works on ``_out_A``
    (the one pointer connector_for also hands the extern call for an in-place arg), which the parent aliases
    to the same AccessNode as ``_in_A``, so it carries the input values on entry."""
    n = 16
    rng = np.random.default_rng(0)
    a, b = rng.random(n), rng.random(n)
    expected = a * 2.0 + b

    sdfg, _ = inplace_lowered()
    got = a.copy()
    sdfg(A=got, b=b.copy(), N=n)
    np.testing.assert_allclose(got, expected)
