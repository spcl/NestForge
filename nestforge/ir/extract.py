# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Outline a map nest or a control-flow nest into a standalone SDFG with DaCe's nesting helpers. :class:`Boundary`
records its data and symbols and the nested SDFG node that phase 2 replaces with an ``ExternalCall``."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import cast

import dace
from dace.sdfg import nodes
from dace.sdfg.graph import SubgraphView
from dace.sdfg.state import ConditionalBlock, LoopRegion, SDFGState
from dace.sdfg.type_inference import infer_expr_type
from dace.transformation import helpers

CfgNest = LoopRegion | ConditionalBlock

#: ``dace.int64``, which DaCe annotates as the NumPy type rather than its typeclass.
INT64 = cast(dace.dtypes.typeclass, dace.int64)
NestNode = nodes.MapEntry | CfgNest


@dataclass(slots=True)
class Boundary:
    """The interface of an extracted nest, in argument order."""

    inputs: list[str]
    outputs: list[str]
    symbols: list[str]
    nsdfg_node: nodes.NestedSDFG | None  # placed in the parent; None for a boundary built without one
    state: SDFGState | None  # None for a boundary built without a parent
    standalone_sdfg: dace.SDFG  # detached, independently compilable copy of the nest


def detach(sdfg: dace.SDFG) -> dace.SDFG:
    """Deep-copy an outlined nested SDFG and cut its parent links so it stands alone."""
    det = copy.deepcopy(sdfg)
    det.parent = None
    det.parent_sdfg = None
    det.parent_nsdfg_node = None
    det.reset_cfg_list()
    return det


def detached_twin(
    sdfg: dace.SDFG, state: SDFGState, entry: nodes.MapEntry
) -> tuple[dace.SDFG, SDFGState, nodes.MapEntry]:
    """A detached copy of ``sdfg`` and the copies of ``state`` and ``entry`` in it."""
    twin = detach(sdfg)
    twin_state = list(twin.all_states())[list(sdfg.all_states()).index(state)]
    twin_entry = twin_state.node(state.node_id(entry))
    assert isinstance(twin_entry, nodes.MapEntry), "a deep copy keeps node ids"
    return twin, twin_state, twin_entry


def find_state_of_node(sdfg: dace.SDFG, node: nodes.Node) -> SDFGState:
    """Return the ``SDFGState`` in ``sdfg`` that contains ``node``."""
    for state in sdfg.states():
        if node in state.nodes():
            return state
    raise ValueError(f"node {node} not found in any state of SDFG {sdfg.label}")


def boundary_from_nsdfg(nsdfg_node: nodes.NestedSDFG, state: SDFGState) -> Boundary:
    inputs = sorted(nsdfg_node.in_connectors.keys())
    outputs = sorted(nsdfg_node.out_connectors.keys())
    symbols = sorted(str(s) for s in nsdfg_node.symbol_mapping.keys())
    return Boundary(
        inputs=inputs,
        outputs=outputs,
        symbols=symbols,
        nsdfg_node=nsdfg_node,
        state=state,
        standalone_sdfg=detach(nsdfg_node.sdfg),
    )


def extract_map_nest(parent_sdfg: dace.SDFG, map_entry: nodes.MapEntry, name: str | None = None) -> Boundary:
    """Outline a map scope into a nested SDFG over whole boundary arrays; a shrunk connector would change the
    kernel's C signature."""
    state = find_state_of_node(parent_sdfg, map_entry)
    subgraph = state.scope_subgraph(map_entry, include_entry=True, include_exit=True)
    nsdfg_node = helpers.nest_state_subgraph(parent_sdfg, state, subgraph, name=name or "nest", full_data=True)
    return boundary_from_nsdfg(nsdfg_node, state)


def assignment_dtype(rhs: str, table: dict[str, dace.dtypes.typeclass]) -> dace.dtypes.typeclass:
    """dtype of an interstate assignment's right-hand side over ``table``, ``int64`` when it cannot be inferred."""
    try:
        inferred = infer_expr_type(rhs, table)
    except Exception:  # inference walks arbitrary expression ASTs; an untypeable RHS keeps the default
        return INT64
    return inferred if isinstance(inferred, dace.dtypes.typeclass) else INT64


def nest_defined_symbol_dtypes(sdfg: dace.SDFG, region: CfgNest) -> dict[str, dace.dtypes.typeclass]:
    """Each symbol an interstate assignment inside the nest defines, with its dtype. Loop iterators are scope
    symbols, typed by DaCe's nesting helper, so they are left out."""
    loop_variables = {
        b.loop_variable
        for b in [region, *region.all_control_flow_blocks()]
        if isinstance(b, LoopRegion) and b.loop_variable
    }
    table = dict(sdfg.symbols) | {name: desc.dtype for name, desc in sdfg.arrays.items()}
    dtypes: dict[str, dace.dtypes.typeclass] = {}
    for e in region.all_interstate_edges():
        for target, rhs in e.data.assignments.items():
            if target in loop_variables or target in dtypes:
                continue
            # a later right-hand side may read an earlier target
            dtypes[target] = table[target] = assignment_dtype(str(rhs), table)
    return dtypes


def extract_cfg_nest(parent_sdfg: dace.SDFG, region: CfgNest, name: str | None = None) -> Boundary:
    """Outline a ``LoopRegion``, or a ``ConditionalBlock`` with all its branches, into a nested SDFG."""
    # declare with the inferred dtype: int64 would truncate a float staged across an edge
    for s, dtype in nest_defined_symbol_dtypes(parent_sdfg, region).items():
        if s not in parent_sdfg.symbols:
            parent_sdfg.add_symbol(s, dtype)
    subgraph = SubgraphView(parent_sdfg, [region])
    inner_state = helpers.nest_sdfg_subgraph(parent_sdfg, subgraph)
    nsdfg_node = next(n for n in inner_state.nodes() if isinstance(n, nodes.NestedSDFG))
    # nest_sdfg_subgraph takes no name, and unnamed nests collide in the build cache
    if name:
        nsdfg_node.sdfg.name = name  # pyright: ignore[reportAttributeAccessIssue]  # a DaCe Property, not read-only
    return boundary_from_nsdfg(nsdfg_node, inner_state)


def extract_nest_to_sdfg(parent_sdfg: dace.SDFG, node: NestNode, name: str | None = None) -> Boundary:
    """Outline a map nest or a control-flow nest; see :func:`extract_map_nest` and :func:`extract_cfg_nest`."""
    if isinstance(node, nodes.MapEntry):
        return extract_map_nest(parent_sdfg, node, name=name)
    if isinstance(node, (LoopRegion, ConditionalBlock)):
        return extract_cfg_nest(parent_sdfg, node, name=name)
    raise TypeError(
        f"cannot extract node of type {type(node).__name__}; expected MapEntry, LoopRegion, or ConditionalBlock"
    )
