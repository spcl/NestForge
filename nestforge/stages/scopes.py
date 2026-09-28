# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 3: lower each offloading scope into an ``ExternalCall`` kernel. Default: one scope per parallel top-level
map. An agent may name several maps of one state, or a straight run of top-level blocks, as one scope."""

from __future__ import annotations

import copy
from collections.abc import Callable, Collection, Iterable, Sequence
from functools import partial

import dace
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowBlock, SDFGState

from nestforge.corpus.translate import python_and_manifest
from nestforge.ir.extract import (
    Boundary,
    NestNode,
    extract_blocks,
    extract_map_nest,
    extract_state_nodes,
    find_state_of_node,
)
from nestforge.ir.introspect import Row, nest_reads_writes
from nestforge.ir.libnode import ExternalCall, external_calls, in_conn, out_conn
from nestforge.ir.names import in_order


def top_level_map_entries(state: SDFGState) -> list[nodes.MapEntry]:
    """The maps of ``state`` not nested in another map."""
    return [n for n in state.scope_children()[None] if isinstance(n, nodes.MapEntry)]


def is_parallel_nest(node: NestNode) -> bool:
    """Whether ``node`` is a map not scheduled sequentially."""
    return isinstance(node, nodes.MapEntry) and node.map.schedule != dace.ScheduleType.Sequential


def parallel_top_level_maps(sdfg: dace.SDFG) -> list[tuple[dace.SDFG, nodes.MapEntry]]:
    """The default scopes: every parallel top-level map, wherever it sits in the control flow."""
    return [
        (sdfg, entry)
        for state in sdfg.all_states()
        for entry in top_level_map_entries(state)
        if is_parallel_nest(entry)
    ]


def is_parallel_kernel(ext: ExternalCall) -> bool:
    """Whether the kernel's body is only parallel maps: no control flow and no work outside a map."""
    body = ext.standalone_sdfg
    if body is None:
        return False
    for block in body.nodes():
        if not isinstance(block, SDFGState):
            return False
        work = [node for node in block.scope_children()[None] if not isinstance(node, nodes.AccessNode)]
        if not all(isinstance(node, nodes.MapEntry) and is_parallel_nest(node) for node in work):
            return False
    return True


def reference_sdfg(boundary: Boundary) -> dace.SDFG:
    """The standalone SDFG with boundary arrays renamed to the node's connectors; an array both read and written
    gets an ``_in_`` and an ``_out_`` connector."""
    ref = copy.deepcopy(boundary.standalone_sdfg)
    inplace = [name for name in boundary.inputs if name in boundary.outputs]
    for name in boundary.inputs:
        if name not in inplace:
            ref.replace(name, in_conn(name))
    for name in boundary.outputs:
        ref.replace(name, out_conn(name))
    for name in sorted(inplace):
        ref.add_datadesc(in_conn(name), copy.deepcopy(ref.arrays[out_conn(name)]))
    return ref


def kernel_arguments(ext: ExternalCall) -> tuple[list[str], list[str], list[str]]:
    """``(inputs, outputs, symbols)`` of the kernel node, in boundary order."""
    inputs = [conn.removeprefix(in_conn("")) for conn in ext.in_connectors]
    outputs = [conn.removeprefix(out_conn("")) for conn in ext.out_connectors]
    # read once: a DaCe property the library-node decorator hides from pyright
    manifest = ext.config  # pyright: ignore[reportAttributeAccessIssue]
    symbols = sorted(arg for arg in manifest["input_args"] if arg not in manifest["array_args"]) if manifest else []
    return inputs, outputs, symbols


def node_boundary(ext: ExternalCall) -> Boundary:
    """The kernel's :class:`Boundary`, rebuilt from the node by inverting :func:`reference_sdfg`."""
    if ext.standalone_sdfg is None:
        raise ValueError(
            f"ExternalCall {ext.name!r} has no standalone SDFG (a kernel reloaded from disk does not carry one); "
            "stages 5 and 6 need the nest it replaced"
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


def connector(prefixed: Callable[[str], str], conn: str | None) -> str | None:
    """``prefixed(conn)``, or ``None`` for an ordering edge, which has no connector."""
    return None if conn is None else prefixed(conn)


def replace_nsdfg_with_external(boundary: Boundary, name: str) -> ExternalCall:
    state, nsdfg = boundary.state, boundary.nsdfg_node
    if state is None or nsdfg is None:
        raise ValueError("replace_nsdfg_with_external needs an extracted-nest Boundary (state + nsdfg_node)")
    source, manifest = python_and_manifest(boundary, name)
    ext = ExternalCall(
        name,
        inputs=[in_conn(i) for i in boundary.inputs],
        outputs=[out_conn(o) for o in boundary.outputs],
        numpy_source=source,
        config=manifest,
        standalone_sdfg=reference_sdfg(boundary),
    )
    state.add_node(ext)
    # a memlet is never shared between edges
    for e in state.in_edges(nsdfg):
        state.add_edge(e.src, e.src_conn, ext, connector(in_conn, e.dst_conn), copy.deepcopy(e.data))
    for e in state.out_edges(nsdfg):
        state.add_edge(ext, connector(out_conn, e.src_conn), e.dst, e.dst_conn, copy.deepcopy(e.data))
    state.remove_node(nsdfg)
    return ext


def host_length1_reads(sdfg: dace.SDFG, reads: Iterable[str], writes: Collection[str]) -> list[str]:
    """Read-only inputs that are length-1 arrays in host memory: a host kernel takes a scalar by value, and only a
    device pointer may carry one."""
    return [
        name
        for name in reads
        if name not in writes
        and name in sdfg.arrays
        and isinstance(desc := sdfg.arrays[name], dace.data.Array)
        and not isinstance(desc, dace.data.View)
        and desc.total_size == 1
        and desc.storage not in GPU_RESIDENT_STORAGES
    ]


def kernel_names(sdfg: dace.SDFG, count: int) -> list[str]:
    """``count`` kernel names no kernel of ``sdfg`` holds; a name keys a kernel's library and work directory."""
    taken = {ext.name for ext in external_calls(sdfg)}
    return [name for name in (f"extcall_{i}" for i in range(len(taken) + count)) if name not in taken][:count]


def lower_nests_to_external_call(sdfg: dace.SDFG) -> list[tuple[ExternalCall, Boundary]]:
    """Stage 3 default: replace every parallel top-level map with an ``ExternalCall``. Refuses before changing
    anything if a nest reads a length-1 host array."""
    refs = parallel_top_level_maps(sdfg)
    offenders = [
        (entry.map.label, name)
        for parent, entry in refs
        for name in host_length1_reads(parent, *nest_reads_writes(find_state_of_node(parent, entry), entry))
    ]
    if offenders:
        raise ValueError(
            f"nest inputs {offenders} are length-1 arrays in host memory; declare each as a Scalar, which "
            "crosses the kernel boundary by value"
        )
    out: list[tuple[ExternalCall, Boundary]] = []
    for (parent, entry), name in zip(refs, kernel_names(sdfg, len(refs))):
        boundary = extract_map_nest(parent, entry, name=name)
        out.append((replace_nsdfg_with_external(boundary, name), boundary))
    return out


#: Outlines a group into one nested SDFG under a kernel name.
Extract = Callable[[str], Boundary]


def runs_between(state: SDFGState, members: dict[nodes.Node, None]) -> nodes.Node | None:
    """A node outside ``members`` that the group feeds and that feeds the group back, so would run inside it."""
    frontier = [e.dst for member in members for e in state.out_edges(member) if e.dst not in members]
    seen = dict.fromkeys(frontier)
    while frontier:
        node = frontier.pop()
        for edge in state.out_edges(node):
            if edge.dst in members:
                return node
            if edge.dst not in seen:
                seen[edge.dst] = None
                frontier.append(edge.dst)
    return None


def map_group(sdfg: dace.SDFG, rows: Sequence[Row]) -> Extract | str:
    """Top-level maps of one state of ``sdfg``, with the access nodes one writes and another reads."""
    state = rows[0][1]
    if state is None or any(other is not state for _, other in rows):
        return "the named maps are in different states; name the blocks that hold them instead."
    if state.sdfg is not sdfg:
        return "the named maps sit inside a nested SDFG; kernels live at the top level."
    entries = [obj for obj, _ in rows]
    nested = [entry.map.label for entry in entries if state.entry_node(entry) is not None]
    if nested:
        return f"{', '.join(nested)} sits inside another map; name top-level maps."
    members: dict[nodes.Node, None] = {}
    for entry in entries:
        members.update(dict.fromkeys(state.scope_subgraph(entry).nodes()))
    exits = [state.exit_node(entry) for entry in entries]
    for node in state.data_nodes():
        if any(e.src in exits for e in state.in_edges(node)) and any(e.dst in entries for e in state.out_edges(node)):
            members[node] = None
    between = runs_between(state, members)
    if between is not None:
        return f"{between} runs between the named maps; name it too, or name maps nothing runs between."
    return partial(extract_state_nodes, sdfg, state, list(members))


def block_group(sdfg: dace.SDFG, blocks: Sequence[ControlFlowBlock]) -> Extract | str:
    """A straight, single-entry run of top-level blocks of ``sdfg``, in execution order."""
    outer = [block.label for block in blocks if block.parent_graph is not sdfg]
    if outer:
        return f"{', '.join(outer)} is not a top-level block of the program; kernels live at the top level."
    order = [block for block in in_order(sdfg) if block in blocks]
    for first, second in zip(order, order[1:]):
        if [e.dst for e in sdfg.out_edges(first)] != [second] or len(sdfg.in_edges(second)) != 1:
            return f"{first.label} and {second.label} are not one straight run; name blocks that follow each other."
    return partial(extract_blocks, sdfg, order)


def group_reads_writes(rows: Sequence[Row]) -> tuple[dict[str, None], dict[str, None]]:
    reads: dict[str, None] = {}
    writes: dict[str, None] = {}
    for obj, state in rows:
        read, written = nest_reads_writes(state, obj) if state is not None else obj.read_and_write_sets()
        reads.update(dict.fromkeys(read))
        writes.update(dict.fromkeys(written))
    return reads, writes


def plan_scope(sdfg: dace.SDFG, rows: Sequence[Row]) -> Extract | str:
    """How to outline the regions ``rows`` name into one kernel, or why they cannot form one; changes nothing."""
    if not rows:
        return "name at least one map or block."
    if all(isinstance(obj, nodes.MapEntry) for obj, _ in rows):
        extract = map_group(sdfg, rows)
    elif all(isinstance(obj, ControlFlowBlock) for obj, _ in rows):
        extract = block_group(sdfg, [obj for obj, _ in rows])
    else:
        return "name only top-level maps or only control-flow blocks."
    if isinstance(extract, str):
        return extract
    host_scalars = host_length1_reads(sdfg, *group_reads_writes(rows))
    if host_scalars:
        return f"inputs {host_scalars} are length-1 arrays in host memory; declare each as a Scalar."
    return extract


def lower_group_to_external_call(sdfg: dace.SDFG, rows: Sequence[Row]) -> tuple[ExternalCall, Boundary] | str:
    """One ``ExternalCall`` kernel over the regions ``rows`` name, or the reason nothing changed. The kernel written
    in stage 5 fuses what no transformation could."""
    extract = plan_scope(sdfg, rows)
    if isinstance(extract, str):
        return extract
    (name,) = kernel_names(sdfg, 1)
    boundary = extract(name)
    return replace_nsdfg_with_external(boundary, name), boundary
