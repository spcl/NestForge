# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``lower_nests_to_external_call``: detect nests via a strategy and swap each for an ``ExternalCall``.

For each nest the strategy returns, extract it to a standalone SDFG, build the numpy + manifest
payload, and replace the outlined ``NestedSDFG`` with an :class:`ExternalCall` node carrying that
payload (defaulting to the ``DaceReference`` expansion, so the result stays runnable immediately).
"""
from __future__ import annotations

import copy
from typing import List, Tuple, Union

import dace

from nestforge.emit_numpy import nest_to_numpy
from nestforge.emit_yaml import manifest_dict
from nestforge.extract import Boundary, extract_nest_to_sdfg
from nestforge.libnode import ExternalCall, in_conn, out_conn
from nestforge.strategies import Strategy, get_strategy


def reference_sdfg(boundary: Boundary) -> "dace.SDFG":
    """A copy of the standalone SDFG whose boundary arrays are renamed to the node's connectors,
    so the ``DaceReference`` nested-SDFG expansion lines up with the ``ExternalCall`` connectors.

    An in-place array is in BOTH ``inputs`` and ``outputs`` and so carries two connectors, but it is
    one array: the body is renamed to the ``_out_`` name only -- the same single pointer
    :func:`~nestforge.libnode.connector_for` hands the extern-C call for an in-place arg, and the
    parent wires both connectors to the one AccessNode, so ``_out_`` already holds the input values.
    ``_in_`` then carries only the read dependency, but a NestedSDFG connector must still resolve to
    a descriptor, so register one for it (renaming the body to ``_in_`` instead would leave the
    ``_out_`` connector undefined and fail NestedSDFG validation).
    """
    ref = copy.deepcopy(boundary.standalone_sdfg)
    inplace = set(boundary.inputs) & set(boundary.outputs)
    for i in boundary.inputs:
        if i not in inplace:
            ref.replace(i, in_conn(i))
    for o in boundary.outputs:
        ref.replace(o, out_conn(o))
    for name in sorted(inplace):
        ref.add_datadesc(in_conn(name), copy.deepcopy(ref.arrays[out_conn(name)]))
    return ref


def replace_nsdfg_with_external(boundary: Boundary, name: str) -> ExternalCall:
    state = boundary.state
    nsdfg = boundary.nsdfg_node
    # Connectors are prefixed so they never collide with array/symbol names (a LibraryNode rule).
    ext = ExternalCall(name,
                       inputs={in_conn(i)
                               for i in boundary.inputs},
                       outputs={out_conn(o)
                                for o in boundary.outputs},
                       numpy_source=nest_to_numpy(boundary, fn_name=name),
                       config=manifest_dict(boundary, name),
                       standalone_sdfg=reference_sdfg(boundary))
    state.add_node(ext)
    # Fresh memlets per edge (never reuse subsets/memlets); remap connector names.
    for e in state.in_edges(nsdfg):
        state.add_edge(e.src, e.src_conn, ext, in_conn(e.dst_conn), copy.deepcopy(e.data))
    for e in state.out_edges(nsdfg):
        state.add_edge(ext, out_conn(e.src_conn), e.dst, e.dst_conn, copy.deepcopy(e.data))
    state.remove_node(nsdfg)
    return ext


def lower_nests_to_external_call(sdfg: dace.SDFG,
                                 strategy: Union[str, Strategy] = "skip-taskloops",
                                 name_prefix: str = "extcall") -> List[Tuple[ExternalCall, Boundary]]:
    """Lower every nest the strategy selects into an ``ExternalCall`` node.

    Defaults to ``skip-taskloops``: offload the compute-bearing nests, not the pure map/loop
    scheduling wrappers around them.

    :returns: ``[(external_call_node, boundary), ...]`` in extraction order.
    """
    strat = get_strategy(strategy) if isinstance(strategy, str) else strategy
    refs = strat(sdfg)
    out: List[Tuple[ExternalCall, Boundary]] = []
    for idx, (parent, node) in enumerate(refs):
        name = f"{name_prefix}_{idx}"
        boundary = extract_nest_to_sdfg(parent, node, name=name)
        ext = replace_nsdfg_with_external(boundary, name)
        out.append((ext, boundary))
    return out
