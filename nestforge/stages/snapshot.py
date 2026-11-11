# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The ``snapshot-reads`` move: a doall that reads a slice of an array which another doall of the same state writes
(``outer = path[:, k] + path[k, :]`` beside ``path = min(path, outer)``) first copies the slice into a transient, so
the two doalls no longer touch the same array and ``doall-fusion`` can merge them.

The copy is a doall of its own, not a plain copy: a plain copy between two arrays is folded back into a read of the
source by the cleanup that follows every move."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

import dace
from dace import symbolic
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import SDFGState
from dace.subsets import Range
from dace.transformation.interstate import InlineSDFG

from nestforge.ir.extract import scope_twin

#: The prefix of the transient that holds the copied slice.
SNAPSHOT_SUFFIX = "_snapshot"
#: The code of the tasklet in the copy doall; a doall that only runs it is a snapshot already.
COPY_CODE = "snap_out = snap_in"
#: The prefix of the parameters of the copy doall.
COPY_PARAM = "__snap"

REFUSED = (
    "blocked by snapshot-reads: the doall reads no slice of an array that another doall of its state writes, or the "
    "slice is not one index per dimension of a map parameter or a value the doall does not change."
)


@dataclass(frozen=True, slots=True)
class SliceRead:
    """One use of a slice inside ``entry``: the array, the dimensions that walk a map parameter and the ones that stay
    fixed, the edge that carries the array into ``entry`` and the edge that reads the slice."""

    data: str
    #: ``(dimension, map parameter)`` of every dimension the read walks.
    walked: tuple[tuple[int, str], ...]
    #: ``(dimension, index)`` of every dimension the read holds fixed.
    fixed: tuple[tuple[int, symbolic.SymbolicType], ...]
    outer: MultiConnectorEdge
    inner: MultiConnectorEdge


def written_by_another_doall(state: SDFGState, entry: nodes.MapEntry) -> set[str]:
    """The arrays a top-level doall of ``state`` other than ``entry`` writes."""
    written: set[str] = set()
    for other in state.nodes():
        if not isinstance(other, nodes.MapExit):
            continue
        owner = state.entry_node(other)
        if owner is not None and owner is not entry and state.entry_node(owner) is None:
            written.update(e.data.data for e in state.out_edges(other) if e.data.data is not None)
    return written


def reads_another_doalls_write(state: SDFGState, entry: nodes.MapEntry) -> bool:
    """Whether ``entry`` reads an array another top-level doall of ``state`` writes. Inlining the body leaves these
    outer reads alone, so without one :func:`snapshot` has nothing to copy."""
    written = written_by_another_doall(state, entry)
    return any(e.data.data in written for e in state.in_edges(entry) if isinstance(e.src, nodes.AccessNode))


def body_tasklets(state: SDFGState, entry: nodes.MapEntry) -> list[nodes.Tasklet]:
    """The tasklets in the body of ``entry``, also inside a nested SDFG that is the body."""
    found: list[nodes.Tasklet] = []
    for node in state.all_nodes_between(entry, state.exit_node(entry)):
        if isinstance(node, nodes.Tasklet):
            found.append(node)
        elif isinstance(node, nodes.NestedSDFG):
            found.extend(n for n, _ in node.sdfg.all_nodes_recursive() if isinstance(n, nodes.Tasklet))
    return found


def is_snapshot_doall(state: SDFGState, entry: nodes.MapEntry) -> bool:
    """Whether the body of ``entry`` is the copy :func:`snapshot` makes, which never snapshots again."""
    body = body_tasklets(state, entry)
    return len(body) == 1 and body[0].code.as_string.strip() == COPY_CODE


def inline_body(state: SDFGState, entry: nodes.MapEntry) -> None:
    """Inline the nested SDFG that is the body of ``entry``, when it is one, so the reads inside are plain memlets; the
    cleanup after the move nests it again."""
    bodies = [n for n in state.all_nodes_between(entry, state.exit_node(entry)) if isinstance(n, nodes.NestedSDFG)]
    for body in bodies:
        if InlineSDFG.can_be_applied_to(state.sdfg, nested_sdfg=body):
            InlineSDFG.apply_to(state.sdfg, annotate=False, save=False, nested_sdfg=body)


def classify(
    entry: nodes.MapEntry, subset: Range
) -> tuple[tuple[tuple[int, str], ...], tuple[tuple[int, symbolic.SymbolicType], ...]] | None:
    """The dimensions of ``subset`` that walk a parameter of ``entry`` and the ones that stay fixed; ``None`` when a
    dimension is a range, mixes a parameter with other terms, or two dimensions walk one parameter."""
    params = {symbolic.pystr_to_symbolic(p): p for p in entry.map.params}
    walked: list[tuple[int, str]] = []
    fixed: list[tuple[int, symbolic.SymbolicType]] = []
    for dim, (begin, end, step) in enumerate(subset.ranges):
        if begin != end or step != 1:
            return None
        if begin in params:
            walked.append((dim, params[begin]))
        elif not (begin.free_symbols & params.keys()):
            fixed.append((dim, begin))
        else:
            return None
    if len({p for _, p in walked}) != len(walked) or not walked or not fixed:
        return None
    return tuple(walked), tuple(fixed)


def slice_reads(state: SDFGState, entry: nodes.MapEntry, written: set[str]) -> Iterator[SliceRead]:
    """Every use inside ``entry`` of a slice of an array in ``written``, one per reading edge."""
    for outer in state.in_edges(entry):
        if outer.dst_conn is None or outer.data.data not in written or not isinstance(outer.src, nodes.AccessNode):
            continue
        for inner in state.out_edges_by_connector(entry, "OUT_" + outer.dst_conn[len("IN_") :]):
            shape = classify(entry, inner.data.subset)
            if shape is not None:
                yield SliceRead(outer.data.data, shape[0], shape[1], outer, inner)


def copy_slice(state: SDFGState, entry: nodes.MapEntry, read: SliceRead) -> str:
    """Add the doall that copies the slice ``read`` uses into a new transient, and return the transient's name."""
    sdfg = state.sdfg
    ranges = dict(zip(entry.map.params, entry.map.range.ranges, strict=True))
    sizes = [symbolic.simplify(ranges[param][1] - ranges[param][0] + 1) for _, param in read.walked]
    name, _ = sdfg.add_transient(read.data + SNAPSHOT_SUFFIX, sizes, sdfg.arrays[read.data].dtype, find_new_name=True)
    params = {f"{COPY_PARAM}{i}": f"0:{size}" for i, size in enumerate(sizes)}
    source: list[str] = [""] * len(sdfg.arrays[read.data].shape)
    for i, (dim, param) in enumerate(read.walked):
        source[dim] = f"{COPY_PARAM}{i} + ({ranges[param][0]})"
    for dim, index in read.fixed:
        source[dim] = str(index)
    state.add_mapped_tasklet(
        f"snapshot_{read.data}",
        params,
        {"snap_in": Memlet(f"{read.data}[{', '.join(source)}]")},
        "snap_out = snap_in",
        {"snap_out": Memlet(f"{name}[{', '.join(params)}]")},
        input_nodes={read.data: cast(nodes.AccessNode, read.outer.src)},
        output_nodes={name: state.add_access(name)},
        external_edges=True,
    )
    return name


def snapshot(state: SDFGState, entry: nodes.MapEntry, written: set[str] | None = None) -> bool:
    """Copy every slice ``entry`` reads of an array another doall of ``state`` writes (``written``, by default read
    from ``state``) into a transient, in a doall of its own, and make ``entry`` read the transient instead. Whether
    anything was copied."""
    if is_snapshot_doall(state, entry):
        return False
    inline_body(state, entry)
    written = written_by_another_doall(state, entry) if written is None else written
    reads = list(slice_reads(state, entry, written))
    ranges = dict(zip(entry.map.params, entry.map.range.ranges, strict=True))
    for number, read in enumerate(reads):
        name = copy_slice(state, entry, read)
        snapshot_node = next(n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == name)
        connector = f"snapshot{number}_{name}"
        entry.add_in_connector("IN_" + connector)
        entry.add_out_connector("OUT_" + connector)
        state.add_edge(
            snapshot_node,
            None,
            entry,
            "IN_" + connector,
            Memlet(data=name, subset=Range.from_array(state.sdfg.arrays[name])),
        )
        index = [symbolic.pystr_to_symbolic(param) - ranges[param][0] for _, param in read.walked]
        subset = Range([(i, i, 1) for i in index])
        state.remove_edge(read.inner)
        state.add_edge(entry, "OUT_" + connector, read.inner.dst, read.inner.dst_conn, Memlet(data=name, subset=subset))
    for outer in {read.outer for read in reads}:
        conn = cast(str, outer.dst_conn)
        if not list(state.out_edges_by_connector(entry, "OUT_" + conn[len("IN_") :])):
            state.remove_edge(outer)
            entry.remove_in_connector(conn)
            entry.remove_out_connector("OUT_" + conn[len("IN_") :])
    return bool(reads)


def snapshot_applies(state: SDFGState, entry: nodes.MapEntry) -> bool:
    """Whether :func:`snapshot` has something to copy; read from the doall's own memlets, with its body inlined on a
    copy of the doall alone. A doall that reads no array another doall writes is refused without the copy."""
    written = written_by_another_doall(state, entry)
    if not any(e.data.data in written for e in state.in_edges(entry) if isinstance(e.src, nodes.AccessNode)):
        return False
    twin_state, twin_entry = scope_twin(state, entry)
    try:
        applied = snapshot(twin_state, twin_entry, written)
        # the twin lacks the doalls that write the transients it reads, which validation warns of
        with warnings.catch_warnings(action="ignore", category=UserWarning):
            twin_state.sdfg.validate()
        return applied
    except (ValueError, KeyError, dace.sdfg.InvalidSDFGError):
        return False
