# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Outline a map nest or a control-flow nest into a standalone SDFG with DaCe's nesting helpers. :class:`Boundary`
records its data and symbols and the nested SDFG node that stage 3 replaces with an ``ExternalCall``."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

import dace
from dace import dtypes
from dace.frontend.operations import detect_reduction_type
from dace.libraries.standard.nodes.external_call import one_element
from dace.sdfg import nodes
from dace.sdfg.graph import SubgraphView
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, LoopRegion, SDFGState
from dace.sdfg.type_inference import infer_expr_type
from dace.transformation import helpers

# TypeAlias, not ``type``: isinstance needs the runtime union
CfgNest: TypeAlias = LoopRegion | ConditionalBlock  # noqa: UP040

#: ``dace.int64``, which DaCe annotates as the NumPy type rather than its typeclass.
INT64 = cast(dace.dtypes.typeclass, dace.int64)
NestNode: TypeAlias = nodes.MapEntry | CfgNest  # noqa: UP040


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
    twin_state = list(twin.states())[list(sdfg.states()).index(state)]
    twin_entry = twin_state.node(state.node_id(entry))
    assert isinstance(twin_entry, nodes.MapEntry), "a deep copy keeps node ids"
    return twin, twin_state, twin_entry


def find_state_of_node(sdfg: dace.SDFG, node: nodes.Node) -> SDFGState:
    """Return the ``SDFGState`` in ``sdfg`` that contains ``node``."""
    for state in sdfg.states():
        if node in state.nodes():
            return state
    raise ValueError(f"node {node} not found in any state of SDFG {sdfg.label}")


def start_reductions(nsdfg_node: nodes.NestedSDFG, state: SDFGState) -> None:
    """A reduction leaving a nest through one element is combined into the target outside, and a kernel's call takes
    that element from a local nothing sets: the nest starts it at the reduction's identity."""
    for edge in state.out_edges(nsdfg_node):
        if edge.data.wcr is None or not one_element(edge.data) or edge.src_conn in nsdfg_node.in_connectors:
            continue
        dtype = nsdfg_node.sdfg.arrays[edge.src_conn].dtype
        identity = dtypes.reduction_identity(dtype, detect_reduction_type(edge.data.wcr))
        if identity is None:  # a custom reduction names no identity
            continue
        start = nsdfg_node.sdfg.add_state_before(nsdfg_node.sdfg.start_block, is_start_block=True)
        tasklet = start.add_tasklet("start_reduction", {}, {"out"}, f"out = {identity}")
        memlet = dace.Memlet(data=edge.src_conn, subset=edge.data.subset)
        start.add_edge(tasklet, "out", start.add_write(edge.src_conn), None, memlet)


def inherit_symbol_predicates(standalone: dace.SDFG, nsdfg_node: nodes.NestedSDFG, state: SDFGState) -> None:
    """Declare on ``standalone`` the sign predicates the enclosing programs declare for the symbols the nest receives
    under their own name. Cutting the parent drops them, and a nested SDFG inside the nest that assumes ``0 < N`` then
    has nothing to prove it from."""
    names = {name for name, value in nsdfg_node.symbol_mapping.items() if str(value) == name}
    outer = state.sdfg
    while names:
        for name in sorted(names & standalone.symbols.keys()):
            inherited = frozenset(outer.symbol_repo.params.predicates.get(name, ()))
            if inherited:
                declared = frozenset(standalone.symbol_repo.params.predicates.get(name, ()))
                standalone.set_symbol_assumptions(name, declared | inherited)
        enclosing = outer.parent_nsdfg_node
        if enclosing is None or outer.parent_sdfg is None:
            return
        names = {name for name in names if str(enclosing.symbol_mapping.get(name)) == name}
        outer = outer.parent_sdfg


def boundary_from_nsdfg(nsdfg_node: nodes.NestedSDFG, state: SDFGState) -> Boundary:
    start_reductions(nsdfg_node, state)
    inputs = sorted(nsdfg_node.in_connectors.keys())
    outputs = sorted(nsdfg_node.out_connectors.keys())
    symbols = sorted(str(s) for s in nsdfg_node.symbol_mapping.keys())
    standalone = detach(nsdfg_node.sdfg)
    inherit_symbol_predicates(standalone, nsdfg_node, state)
    return Boundary(
        inputs=inputs,
        outputs=outputs,
        symbols=symbols,
        nsdfg_node=nsdfg_node,
        state=state,
        standalone_sdfg=standalone,
    )


def extract_map_nest(parent_sdfg: dace.SDFG, map_entry: nodes.MapEntry, name: str | None = None) -> Boundary:
    """Outline a map scope into a nested SDFG over whole boundary arrays; a shrunk connector would change the
    kernel's C signature."""
    state = find_state_of_node(parent_sdfg, map_entry)
    return extract_state_nodes(parent_sdfg, state, state.scope_subgraph(map_entry).nodes(), name or "nest")


def extract_state_nodes(parent_sdfg: dace.SDFG, state: SDFGState, members: Sequence[nodes.Node], name: str) -> Boundary:
    """Outline ``members`` of ``state``, a convex set of whole map scopes and the access nodes between them, into
    one nested SDFG over whole boundary arrays."""
    subgraph = SubgraphView(state, list(members))
    nsdfg_node = helpers.nest_state_subgraph(parent_sdfg, state, subgraph, name=name)
    return boundary_from_nsdfg(nsdfg_node, state)


def assignment_dtype(rhs: str, table: dict[str, dace.dtypes.typeclass]) -> dace.dtypes.typeclass:
    """dtype of an interstate assignment's right-hand side over ``table``, ``int64`` when it cannot be inferred."""
    try:
        inferred = infer_expr_type(rhs, table)
    except Exception:  # inference walks arbitrary expression ASTs; an untypeable RHS keeps the default
        return INT64
    return inferred if isinstance(inferred, dace.dtypes.typeclass) else INT64


def symbol_table(sdfg: dace.SDFG) -> dict[str, dace.dtypes.typeclass]:
    """Every symbol and container of ``sdfg`` with its dtype, the names an assignment's right-hand side reads."""
    return dict(sdfg.symbols) | {name: desc.dtype for name, desc in sdfg.arrays.items()}


def define_symbol(
    defined: dict[str, dace.dtypes.typeclass], table: dict[str, dace.dtypes.typeclass], target: str, rhs: str
) -> None:
    """Type ``target = rhs`` into ``defined`` and ``table``, widening an earlier assignment's dtype: an int first
    value then a float one is a double, or the float would be truncated."""
    inferred = assignment_dtype(rhs, table)
    if target in defined:
        inferred = dace.dtypes.result_type_of(defined[target], inferred)
    defined[target] = table[target] = inferred


def nest_defined_symbol_dtypes(sdfg: dace.SDFG, region: CfgNest) -> dict[str, dace.dtypes.typeclass]:
    """Each symbol an interstate assignment inside the nest defines, with its dtype. Loop iterators are scope
    symbols, typed by DaCe's nesting helper, so they are left out."""
    loop_variables = {
        b.loop_variable
        for b in [region, *region.all_control_flow_blocks()]
        if isinstance(b, LoopRegion) and b.loop_variable
    }
    table = symbol_table(sdfg)
    dtypes: dict[str, dace.dtypes.typeclass] = {}
    for e in region.all_interstate_edges():
        for target, rhs in e.data.assignments.items():
            if target not in loop_variables:
                define_symbol(dtypes, table, target, str(rhs))  # a later right-hand side may read this target
    return dtypes


def extract_cfg_nest(parent_sdfg: dace.SDFG, region: CfgNest, name: str | None = None) -> Boundary:
    """Outline a ``LoopRegion``, or a ``ConditionalBlock`` with all its branches, into a nested SDFG."""
    return extract_blocks(parent_sdfg, [region], name)


def blocks_defined_symbol_dtypes(
    parent_sdfg: dace.SDFG, blocks: Sequence[ControlFlowBlock]
) -> dict[str, dace.dtypes.typeclass]:
    """Each symbol assigned inside ``blocks`` or on an edge between two of them, with its dtype."""
    defined: dict[str, dace.dtypes.typeclass] = {}
    for block in blocks:
        if isinstance(block, CfgNest):
            defined |= nest_defined_symbol_dtypes(parent_sdfg, block)
    table = symbol_table(parent_sdfg) | defined
    for edge in blocks[0].parent_graph.edges():
        if edge.src in blocks and edge.dst in blocks:
            for target, rhs in edge.data.assignments.items():
                define_symbol(defined, table, target, str(rhs))
    return defined


def extract_blocks(parent_sdfg: dace.SDFG, blocks: Sequence[ControlFlowBlock], name: str | None = None) -> Boundary:
    """Outline a single-entry, single-exit run of blocks of one region of ``parent_sdfg`` into one nested SDFG."""
    if len(blocks) == 1 and isinstance(blocks[0], SDFGState):
        # DaCe nests a lone state as itself, so its contents are nested instead
        return extract_state_nodes(parent_sdfg, blocks[0], blocks[0].nodes(), name or "nest")
    # declare with the inferred dtype: int64 would truncate a float staged across an edge
    for s, dtype in blocks_defined_symbol_dtypes(parent_sdfg, blocks).items():
        if s not in parent_sdfg.symbols:
            parent_sdfg.add_symbol(s, dtype)
    subgraph = SubgraphView(blocks[0].parent_graph, list(blocks))
    inner_state = helpers.nest_sdfg_subgraph(parent_sdfg, subgraph, start=blocks[0])
    nsdfg_node = next(n for n in inner_state.nodes() if isinstance(n, nodes.NestedSDFG))
    # nest_sdfg_subgraph takes no name, and unnamed nests collide in the build cache
    if name:
        nsdfg_node.sdfg.name = name  # pyright: ignore[reportAttributeAccessIssue]  # a DaCe Property, not read-only
    return boundary_from_nsdfg(nsdfg_node, inner_state)
