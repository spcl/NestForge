# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A loop kernel that exports a symbol the program reads after it (DaCe's ``__sym_out_<s>`` output) keeps one name
for that output in its connectors, its ``abi_order`` and its body, through the label normalization every change runs."""

import dace
from dace.libraries.standard.nodes.external_call import data_name

from tactics.session import Session
from tactics.stages.canonicalize import Targets
from tactics.stages.scopes import node_boundary

N = dace.symbol("N", dtype=dace.int64)


@dace.program
def count_then_fill(x: dace.float64[N], y: dace.float64[N], z: dace.float64[N]) -> None:
    m = 0
    for i in range(1, N):
        y[i] = y[i - 1] * x[i] + x[i]
        if y[i] > 0.5:
            m = m + 1
    z[0:m] = 1.0


# Reproducer: shorten_temporaries renamed a loop kernel's __sym_out_ output in abi_order, not in its connectors (#122, qe_cegterg extcall_for42, KeyError: 'tmp165')
def test_a_kernel_exporting_a_symbol_keeps_its_output_named_alike_in_abi_order_and_connectors() -> None:
    session = Session(count_then_fill.to_sdfg(), targets=Targets(gpu=False), sizes={"N": 8})
    session.canonicalize()
    session.default_moves()
    scopes = session.default_scopes()
    (loop,) = [k for k in scopes if any(name.startswith("__sym_out_") for name in k.writes)]
    ext = session.kernel(loop.name)
    connected = {data_name(conn) for conn in [*ext.in_connectors, *ext.out_connectors]}
    exported = [name for name in loop.writes if name.startswith("__sym_out_")]
    assert set(exported) <= set(ext.abi_order)
    assert set(exported) <= connected
    boundary = node_boundary(ext)
    assert not set(boundary.symbols) & set(session.sdfg.arrays)  # no container taken for a symbol
    session.place()
    assert session.default_kernel(loop.name).boundary.outputs == boundary.outputs
