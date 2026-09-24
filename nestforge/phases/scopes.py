# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 2: turn every parallel top-level map into an ``ExternalCall`` kernel."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from collections.abc import Callable, Sequence

import dace
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowBlock, LoopRegion, SDFGState

from nestforge.ir.emit_numpy import nest_to_numpy
from nestforge.ir.emit_yaml import manifest_dict
from nestforge.ir.extract import (
    Boundary,
    NestNode,
    extract_blocks,
    extract_nest_to_sdfg,
    extract_state_nodes,
    find_state_of_node,
)
from nestforge.ir.names import in_order
from nestforge.ir.introspect import Row, nest_reads_writes
from nestforge.ir.dace_types import strings
from nestforge.ir.depends import kernel_symbols
from nestforge.ir.libnode import ExternalCall, external_calls, in_conn, out_conn


def top_level_map_entries(state: dace.SDFGState) -> list[nodes.MapEntry]:
    """The maps of ``state`` not nested in another map."""
    return [n for n in state.scope_children()[None] if isinstance(n, nodes.MapEntry)]


def is_parallel_nest(node: NestNode) -> bool:
    """Whether ``node`` is a map not scheduled sequentially."""
    return isinstance(node, nodes.MapEntry) and node.map.schedule != dace.ScheduleType.Sequential


def parallel_top_level_maps(sdfg: dace.SDFG) -> list[tuple[dace.SDFG, nodes.MapEntry]]:
    """Phase 2's scopes: every parallel top-level map, wherever it sits in the control flow."""
    return [
        (sdfg, entry)
        for state in sdfg.all_states()
        for entry in top_level_map_entries(state)
        if is_parallel_nest(entry)
    ]


def label_nest(node: nodes.MapEntry | LoopRegion) -> str:
    """A short description of a map nest or loop nest."""
    if isinstance(node, nodes.MapEntry):
        return f"map[{', '.join(strings(node.map.params))}] over {node.map.range}"
    if isinstance(node, LoopRegion):
        return f"loop {node.label}"
    raise TypeError(f"not a nest: {type(node).__name__}")


@dataclass(slots=True)
class OffloadCandidate:
    """A map phase 2 would turn into a kernel."""

    parent_sdfg: dace.SDFG
    node: nodes.MapEntry
    label: str


def offload_candidates(sdfg: dace.SDFG) -> list[OffloadCandidate]:
    """The maps phase 2 would turn into kernels, without changing ``sdfg``."""
    return [OffloadCandidate(parent, node, label_nest(node)) for parent, node in parallel_top_level_maps(sdfg)]


def reference_sdfg(boundary: Boundary) -> dace.SDFG:
    """The standalone SDFG with boundary arrays renamed to the node's connectors; an array both read and written
    gets an ``_in_`` and an ``_out_`` connector."""
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


def kernel_arguments(ext: ExternalCall) -> tuple[list[str], list[str], list[str]]:
    """``(inputs, outputs, symbols)`` of the kernel node, in boundary order."""
    inputs = [conn.removeprefix(in_conn("")) for conn in ext.in_connectors]
    outputs = [conn.removeprefix(out_conn("")) for conn in ext.out_connectors]
    return inputs, outputs, kernel_symbols(ext)


def node_boundary(ext: ExternalCall) -> Boundary:
    """The :class:`Boundary` phases 4 and 5 need, rebuilt from the node by inverting :func:`reference_sdfg`."""
    if ext.standalone_sdfg is None:
        raise ValueError(
            f"ExternalCall {ext.name!r} has no standalone SDFG (a kernel reloaded from disk does not carry one); "
            "phases 4 and 5 need the nest it replaced"
        )
    inputs, outputs, symbols = kernel_arguments(ext)
    nest = copy.deepcopy(ext.standalone_sdfg)
    for name in inputs:
        if name in outputs:
            nest.remove_data(in_conn(name))  # the in-place twin reference_sdfg added; the body uses _out_
        else:
            nest.replace(in_conn(name), name)
    for name in outputs:
        nest.replace(out_conn(name), name)
    return Boundary(inputs, outputs, symbols, nsdfg_node=None, state=None, standalone_sdfg=nest)


def kernel_connector(prefixed: Callable[[str], str], conn: str | None) -> str | None:
    """``prefixed(conn)``, or ``None`` for an ordering edge, which has no connector."""
    return None if conn is None else prefixed(conn)


def replace_nsdfg_with_external(boundary: Boundary, name: str) -> ExternalCall:
    if boundary.state is None or boundary.nsdfg_node is None:
        raise ValueError(
            "replace_nsdfg_with_external needs an extracted-nest Boundary (state + nsdfg_node); got a "
            "whole-program boundary, which has neither"
        )
    state = boundary.state
    nsdfg = boundary.nsdfg_node
    ext = ExternalCall(
        name,
        inputs=[in_conn(i) for i in boundary.inputs],
        outputs=[out_conn(o) for o in boundary.outputs],
        numpy_source=nest_to_numpy(boundary, fn_name=name),
        config=manifest_dict(boundary, name),
        standalone_sdfg=reference_sdfg(boundary),
    )
    state.add_node(ext)
    # a memlet is never shared between edges
    for e in state.in_edges(nsdfg):
        state.add_edge(e.src, e.src_conn, ext, kernel_connector(in_conn, e.dst_conn), copy.deepcopy(e.data))
    for e in state.out_edges(nsdfg):
        state.add_edge(ext, kernel_connector(out_conn, e.src_conn), e.dst, e.dst_conn, copy.deepcopy(e.data))
    state.remove_node(nsdfg)
    return ext


def is_host_length1_array(desc: dace.data.Data) -> bool:
    return (
        isinstance(desc, dace.data.Array)
        and not isinstance(desc, dace.data.View)
        and desc.total_size == 1
        and desc.storage not in GPU_RESIDENT_STORAGES
    )


def host_length1_inputs(sdfg: dace.SDFG, entry: nodes.MapEntry) -> list[str]:
    """Read-only inputs of the nest at ``entry`` that are length-1 arrays in host memory."""
    state = find_state_of_node(sdfg, entry)
    reads, writes = nest_reads_writes(state, entry)
    return [name for name in reads if name not in writes and is_host_length1_array(state.sdfg.arrays[name])]


def refuse_host_length1_inputs(refs: list[tuple[dace.SDFG, nodes.MapEntry]]) -> None:
    """Refuse a length-1 host array input: a host kernel takes a scalar by value, and only a device pointer may
    carry one."""
    offenders = [(entry.map.label, name) for parent, entry in refs for name in host_length1_inputs(parent, entry)]
    if offenders:
        raise ValueError(
            f"nest inputs {offenders} are length-1 arrays in host memory; declare each as a Scalar, which "
            "crosses the kernel boundary by value"
        )


def lower_nests_to_external_call(sdfg: dace.SDFG) -> list[tuple[ExternalCall, Boundary]]:
    """Replace every parallel top-level map with an ``ExternalCall``; returns each call with its boundary. Refuses
    before changing anything if a nest reads a length-1 host array."""
    refs = parallel_top_level_maps(sdfg)
    refuse_host_length1_inputs(refs)
    out: list[tuple[ExternalCall, Boundary]] = []
    for (parent, node), name in zip(refs, kernel_names(sdfg, len(refs))):
        boundary = extract_nest_to_sdfg(parent, node, name=name)
        ext = replace_nsdfg_with_external(boundary, name)
        out.append((ext, boundary))
    return out


def kernel_names(sdfg: dace.SDFG, count: int) -> list[str]:
    """``count`` kernel names no kernel of ``sdfg`` holds; a name keys a kernel's library and work directory."""
    taken = {ext.name for ext in external_calls(sdfg)}
    return [name for name in (f"extcall_{i}" for i in range(len(taken) + count)) if name not in taken][:count]


def map_group(state: SDFGState, entries: Sequence[nodes.MapEntry]) -> list[nodes.Node] | str:
    """The nodes one kernel over the top-level maps ``entries`` takes: their scopes and every access node one of
    them writes and another reads. A reason instead when something outside the group runs between them."""
    scope = state.scope_dict()
    nested = [entry.map.label for entry in entries if scope[entry] is not None]
    if nested:
        return f"{', '.join(nested)} sits inside another map; name top-level maps."
    members: dict[nodes.Node, None] = {}
    for entry in entries:
        members.update(dict.fromkeys(state.scope_subgraph(entry).nodes()))
    exits = [state.exit_node(entry) for entry in entries]
    for node in state.data_nodes():
        if any(e.src in exits for e in state.in_edges(node)) and any(e.dst in entries for e in state.out_edges(node)):
            members[node] = None
    # a node the group feeds that feeds the group back would have to run inside the kernel
    frontier = [e.dst for member in members for e in state.out_edges(member) if e.dst not in members]
    seen = dict.fromkeys(frontier)
    while frontier:
        node = frontier.pop()
        for edge in state.out_edges(node):
            if edge.dst in members:
                return f"{node} runs between the named maps; name it too, or name maps nothing runs between."
            if edge.dst not in seen:
                seen[edge.dst] = None
                frontier.append(edge.dst)
    return list(members)


def block_group(sdfg: dace.SDFG, blocks: Sequence[ControlFlowBlock]) -> list[ControlFlowBlock] | str:
    """``blocks`` in execution order when they are a straight, single-entry run of top-level blocks of ``sdfg``;
    otherwise the reason."""
    outer = [block.label for block in blocks if block.parent_graph is not sdfg]
    if outer:
        return f"{', '.join(outer)} is not a top-level block of the program; kernels live at the top level."
    order = [block for block in in_order(sdfg) if block in blocks]
    for first, second in zip(order, order[1:]):
        if [e.dst for e in sdfg.out_edges(first)] != [second] or len(sdfg.in_edges(second)) != 1:
            return f"{first.label} and {second.label} are not one straight run; name blocks that follow each other."
    return order


def group_inputs(sdfg: dace.SDFG, rows: Sequence[Row]) -> list[str]:
    """What the named regions read, for the host length-1 check lowering makes."""
    reads: dict[str, None] = {}
    for obj, state in rows:
        if isinstance(obj, nodes.MapEntry) and state is not None:
            reads.update(dict.fromkeys(nest_reads_writes(state, obj)[0]))
        elif isinstance(obj, ControlFlowBlock):
            reads.update(dict.fromkeys(obj.read_and_write_sets()[0]))
    return [name for name in reads if name in sdfg.arrays]


def lower_group_to_external_call(sdfg: dace.SDFG, rows: Sequence[Row]) -> tuple[ExternalCall, Boundary] | str:
    """One ``ExternalCall`` kernel over the regions ``rows`` name: several top-level maps of one state, or a
    straight run of top-level blocks. A reason, and nothing changed, when they cannot form one kernel.

    This is the scope an agent picks when no transformation fuses the regions: the kernel it writes in phase 4
    fuses them instead.
    """
    entries = [obj for obj, _ in rows if isinstance(obj, nodes.MapEntry)]
    blocks = [obj for obj, _ in rows if isinstance(obj, ControlFlowBlock)]
    if len(entries) == len(rows):
        return lower_map_group(sdfg, rows, entries)
    if len(blocks) == len(rows):
        group = block_group(sdfg, blocks)
        if isinstance(group, str):
            return group
        return lower_group(sdfg, rows, lambda name: extract_blocks(sdfg, group, name))
    return "name only maps or only control-flow blocks."


def lower_map_group(
    sdfg: dace.SDFG, rows: Sequence[Row], entries: list[nodes.MapEntry]
) -> tuple[ExternalCall, Boundary] | str:
    """:func:`lower_group_to_external_call` for maps, which must share one top-level state."""
    state = rows[0][1]
    if state is None or any(other is not state for _, other in rows):
        return "the named maps are in different states; name the blocks that hold them instead."
    if state.sdfg is not sdfg:
        return "the named maps sit inside a nested SDFG; kernels live at the top level."
    group = map_group(state, entries)
    if isinstance(group, str):
        return group
    return lower_group(sdfg, rows, lambda name: extract_state_nodes(sdfg, state, group, name))


def lower_group(
    sdfg: dace.SDFG, rows: Sequence[Row], extract: Callable[[str], Boundary]
) -> tuple[ExternalCall, Boundary] | str:
    """Extract a legal group under a fresh kernel name and put its ``ExternalCall`` in its place."""
    host_scalars = [name for name in group_inputs(sdfg, rows) if is_host_length1_array(sdfg.arrays[name])]
    if host_scalars:
        return f"inputs {host_scalars} are length-1 arrays in host memory; declare each as a Scalar."
    (name,) = kernel_names(sdfg, 1)
    boundary = extract(name)
    return replace_nsdfg_with_external(boundary, name), boundary
