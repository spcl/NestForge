"""Emit a standalone numpy/python kernel from an extracted map-nest.

M0 scope: one state, one top-level map scope, Python tasklets, no WCR. Connector identity is
ignored (per design) -- each tasklet connector is replaced by the array element ``name[idx]``
it reads/writes (from the memlet's ``data`` + ``subset``), or by a local scalar for an
intermediate transient. Unsupported constructs raise, so nothing is silently mis-emitted.
"""
from __future__ import annotations

import re
from typing import Dict, List

import dace
from dace import symbolic
from dace.sdfg import nodes
from dace.sdfg.utils import dfs_topological_sort

from nestforge.extract import Boundary


class UnsupportedNest(Exception):
    """The nest uses a construct the M0 numpy emitter does not handle."""


def _index(subset: dace.subsets.Range) -> str:
    parts: List[str] = []
    for (beg, end, step) in subset.ranges:
        if str(beg) == str(end):
            parts.append(symbolic.symstr(beg))
        elif str(step) == "1":
            parts.append(f"{symbolic.symstr(beg)}:{symbolic.symstr(end + 1)}")
        else:
            parts.append(f"{symbolic.symstr(beg)}:{symbolic.symstr(end + 1)}:{symbolic.symstr(step)}")
    return ", ".join(parts)


def _local(name: str) -> str:
    return f"_t_{name}"


def _access(sdfg: dace.SDFG, name: str, subset: dace.subsets.Range) -> str:
    """``name[idx]`` for a real array, or a local scalar for an intermediate transient."""
    desc = sdfg.arrays[name]
    if desc.transient:
        return _local(name)
    return f"{name}[{_index(subset)}]"


def _sub_connectors(code: str, conn_expr: Dict[str, str]) -> str:
    """Replace whole-word connector tokens in a tasklet's Python code with their expressions."""
    for conn, expr in sorted(conn_expr.items(), key=lambda kv: -len(kv[0])):
        code = re.sub(rf"\b{re.escape(conn)}\b", expr, code)
    return code


def nest_to_numpy(boundary: Boundary, fn_name: str = "kernel") -> str:
    """Return standalone python source ``def <fn_name>(<args>): ...`` computing the nest."""
    sdfg = boundary.standalone_sdfg
    states = sdfg.states()
    if len(states) != 1:
        raise UnsupportedNest(f"M0 emitter expects a single state, got {len(states)}")
    state = states[0]

    entries = [n for n in state.scope_children()[None] if isinstance(n, nodes.MapEntry)]
    if len(entries) != 1:
        raise UnsupportedNest(f"M0 emitter expects one top-level map, got {len(entries)}")
    map_entry = entries[0]

    # Loop headers from the map ranges.
    headers: List[str] = []
    for param, (beg, end, step) in zip(map_entry.map.params, map_entry.map.range.ranges):
        upper = symbolic.symstr(end + 1)
        step_s = symbolic.symstr(step)
        headers.append(f"for {param} in range({symbolic.symstr(beg)}, {upper}, {step_s}):")

    # Body: tasklets in the map scope, in dataflow order.
    scope = state.scope_subgraph(map_entry, include_entry=False, include_exit=False)
    body_lines: List[str] = []
    for node in dfs_topological_sort(scope):
        if not isinstance(node, nodes.Tasklet):
            continue
        if node.code.language != dace.dtypes.Language.Python:
            raise UnsupportedNest(f"tasklet {node.label} is not Python ({node.code.language})")
        conn_expr: Dict[str, str] = {}
        for e in state.in_edges(node):
            if e.dst_conn is not None:
                conn_expr[e.dst_conn] = _access(sdfg, e.data.data, e.data.subset)
        for e in state.out_edges(node):
            if e.src_conn is not None:
                if e.data.wcr is not None:
                    raise UnsupportedNest(f"tasklet {node.label} has a WCR (reduction) edge")
                conn_expr[e.src_conn] = _access(sdfg, e.data.data, e.data.subset)
        for line in node.code.as_string.splitlines():
            body_lines.append(_sub_connectors(line, conn_expr))

    # Signature: real arrays (inputs, then extra outputs), then symbols.
    args = list(boundary.inputs)
    args += [o for o in boundary.outputs if o not in boundary.inputs]
    args += [s for s in boundary.symbols if s not in args]

    indent = "    "
    lines = [f"def {fn_name}({', '.join(args)}):"]
    depth = 1
    for h in headers:
        lines.append(indent * depth + h)
        depth += 1
    for bl in body_lines:
        lines.append(indent * depth + bl)
    return "\n".join(lines) + "\n"
