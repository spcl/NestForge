# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Outline a map nest or a control-flow nest into a standalone SDFG with DaCe's nesting helpers. :class:`Boundary`
records its data and symbols and the nested SDFG node that phase 2 replaces with an ``ExternalCall``."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
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
    nsdfg_node: nodes.NestedSDFG | None  # placed in the parent; None for a whole-program boundary
    state: SDFGState | None  # None for a whole-program boundary
    standalone_sdfg: dace.SDFG  # detached, independently compilable copy of the nest
    parent_sdfg: dace.SDFG | None = field(repr=False, default=None)


def detach(sdfg: dace.SDFG) -> dace.SDFG:
    """Deep-copy an outlined nested SDFG and cut its parent links so it stands alone."""
    det = copy.deepcopy(sdfg)
    det.parent = None
    det.parent_sdfg = None
    det.parent_nsdfg_node = None
    det.reset_cfg_list()
    return det


def find_state_of_node(sdfg: dace.SDFG, node: nodes.Node) -> SDFGState:
    """Return the ``SDFGState`` in ``sdfg`` that contains ``node``."""
    for state in sdfg.states():
        if node in state.nodes():
            return state
    raise ValueError(f"node {node} not found in any state of SDFG {sdfg.label}")


def boundary_from_nsdfg(nsdfg_node: nodes.NestedSDFG, state: SDFGState, parent_sdfg: dace.SDFG) -> Boundary:
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
        parent_sdfg=parent_sdfg,
    )


def extract_map_nest(parent_sdfg: dace.SDFG, map_entry: nodes.MapEntry, name: str | None = None) -> Boundary:
    """Outline a map scope into a nested SDFG over whole boundary arrays; a shrunk connector would change the
    kernel's C signature."""
    state = find_state_of_node(parent_sdfg, map_entry)
    subgraph = state.scope_subgraph(map_entry, include_entry=True, include_exit=True)
    nsdfg_node = helpers.nest_state_subgraph(parent_sdfg, state, subgraph, name=name or "nest", full_data=True)
    return boundary_from_nsdfg(nsdfg_node, state, parent_sdfg)


def assignment_dtype(sdfg: dace.SDFG, rhs: str) -> dace.dtypes.typeclass:
    """dtype of an interstate assignment's right-hand side, ``int64`` when it cannot be inferred."""
    table = {s: t for s, t in sdfg.symbols.items()}
    table.update({name: desc.dtype for name, desc in sdfg.arrays.items()})
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
    dtypes: dict[str, dace.dtypes.typeclass] = {}
    for e in region.all_interstate_edges():
        for target, rhs in e.data.assignments.items():
            if target in loop_variables or target in dtypes:
                continue
            dtypes[target] = assignment_dtype(sdfg, str(rhs))
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
    return boundary_from_nsdfg(nsdfg_node, inner_state, parent_sdfg)


def extract_nest_to_sdfg(parent_sdfg: dace.SDFG, node: NestNode, name: str | None = None) -> Boundary:
    """Outline a map nest or a control-flow nest; see :func:`extract_map_nest` and :func:`extract_cfg_nest`."""
    if isinstance(node, nodes.MapEntry):
        return extract_map_nest(parent_sdfg, node, name=name)
    if isinstance(node, (LoopRegion, ConditionalBlock)):
        return extract_cfg_nest(parent_sdfg, node, name=name)
    raise TypeError(
        f"cannot extract node of type {type(node).__name__}; expected MapEntry, LoopRegion, or ConditionalBlock"
    )


def whole_program_boundary(sdfg: dace.SDFG) -> Boundary:
    """A :class:`Boundary` over the whole program: its non-transient arrays read and written, and its symbols."""
    detached = detach(sdfg)
    read, write = detached.read_and_write_sets()
    arrays = {n for n, desc in detached.arrays.items() if not desc.transient}
    inputs = sorted(a for a in arrays if a in read)
    outputs = sorted(a for a in arrays if a in write)
    symbols = [a for a in detached.arglist() if a not in detached.arrays]
    return Boundary(
        inputs=inputs,
        outputs=outputs,
        symbols=symbols,
        nsdfg_node=None,
        state=None,
        standalone_sdfg=detached,
        parent_sdfg=None,
    )
