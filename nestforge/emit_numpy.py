"""Emit a standalone numpy/python kernel from an extracted nest.

The emitter walks a state's dataflow in topological order and turns each node into numpy:

  * **library node** (``MatMul``, ``Dot``, ``Reduce``, ...) -> the equivalent numpy op, via
    :mod:`nestforge.emit_libnode` (the polybench/npbench kernels are almost all library nodes);
  * **map scope** -> explicit ``for`` loops with the tasklet body inlined (the fallback when a nest
    has no library node);
  * **free tasklet** -> its Python code inlined.

Connector identity is ignored (per design): each element access is the array element ``name[idx]``
its memlet names, and each array's data name doubles as the numpy variable at state level.
Unsupported constructs raise :class:`UnsupportedNest`, so nothing is silently mis-emitted.
"""
from __future__ import annotations

import re
from typing import Dict, List

import dace
from dace import symbolic
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.sdfg.utils import dfs_topological_sort

from nestforge.emit_libnode import UnsupportedLibraryNode, emit_library_node, index_str
from nestforge.extract import Boundary


class UnsupportedNest(Exception):
    """The nest uses a construct the numpy emitter does not handle."""


def _local(name: str) -> str:
    return f"_t_{name}"


def _is_scalar(desc) -> bool:
    return isinstance(desc, dace.data.Scalar) or desc.total_size == 1


def _access(sdfg: dace.SDFG, name: str, subset: dace.subsets.Range) -> str:
    """Element access ``name[idx]``, or a local scalar ``_t_name`` for a scalar transient temp.

    A *scalar* transient is a per-iteration temporary (connector-to-connector inside one map); an
    *array* transient is a state-level intermediate (e.g. a matmul operand produced by a map) and
    must be indexed like any array so it lines up with the pre-allocated buffer.
    """
    desc = sdfg.arrays[name]
    if desc.transient and _is_scalar(desc):
        return _local(name)
    return f"{name}[{index_str(subset)}]"


def _sub_connectors(code: str, conn_expr: Dict[str, str]) -> str:
    """Replace whole-word connector tokens in a tasklet's Python code with their expressions."""
    for conn, expr in sorted(conn_expr.items(), key=lambda kv: -len(kv[0])):
        code = re.sub(rf"\b{re.escape(conn)}\b", expr, code)
    return code


def _tasklet_lines(state: dace.SDFGState, sdfg: dace.SDFG, tasklet: nodes.Tasklet) -> List[str]:
    """The tasklet's Python code with connectors substituted by the array element they name."""
    if tasklet.code.language != dace.dtypes.Language.Python:
        raise UnsupportedNest(f"tasklet {tasklet.label} is not Python ({tasklet.code.language})")
    conn_expr: Dict[str, str] = {}
    for e in state.in_edges(tasklet):
        if e.dst_conn is not None:
            conn_expr[e.dst_conn] = _access(sdfg, e.data.data, e.data.subset)
    for e in state.out_edges(tasklet):
        if e.src_conn is not None:
            if e.data.wcr is not None:
                raise UnsupportedNest(f"tasklet {tasklet.label} has a WCR (reduction) edge")
            conn_expr[e.src_conn] = _access(sdfg, e.data.data, e.data.subset)
    return [_sub_connectors(line, conn_expr) for line in tasklet.code.as_string.splitlines()]


def _numpy_dtype(desc) -> str:
    return f"np.{desc.dtype.type.__name__}"


def _shape_tuple(desc) -> str:
    dims = ", ".join(symbolic.symstr(s) for s in desc.shape)
    return f"({dims},)" if len(desc.shape) == 1 else f"({dims})"


def _map_lines(state: dace.SDFGState, sdfg: dace.SDFG, entry: nodes.MapEntry, defined: set) -> List[str]:
    """Emit a map scope as ``for`` loops, pre-allocating any transient array it writes."""
    lines: List[str] = []
    # An array written by element-indexed loop stores must exist before the loop. Passed-in arrays
    # are already in ``defined``; transients and return values (excluded from ``defined``) get a
    # fresh buffer here.
    exit_node = state.exit_node(entry)
    for e in state.out_edges(exit_node):
        name = e.data.data
        desc = sdfg.arrays[name]
        if name not in defined and not _is_scalar(desc):
            lines.append(f"{name} = np.empty({_shape_tuple(desc)}, dtype={_numpy_dtype(desc)})")
            defined.add(name)

    headers: List[str] = []
    for param, (beg, end, step) in zip(entry.map.params, entry.map.range.ranges):
        headers.append(f"for {param} in range({symbolic.symstr(beg)}, {symbolic.symstr(end + 1)}, "
                       f"{symbolic.symstr(step)}):")

    body: List[str] = []
    scope = state.scope_subgraph(entry, include_entry=False, include_exit=False)
    for node in dfs_topological_sort(scope):
        if isinstance(node, nodes.Tasklet):
            body.extend(_tasklet_lines(state, sdfg, node))
        elif isinstance(node, nodes.NestedSDFG):
            raise UnsupportedNest("nested SDFG inside a map is not yet emitted")

    for depth, h in enumerate(headers):
        lines.append("    " * depth + h)
    for bl in body:
        lines.append("    " * len(headers) + bl)
    return lines


def _state_body(sdfg: dace.SDFG, state: dace.SDFGState, defined: set) -> List[str]:
    """Numpy statements for a whole state, in dataflow order (library nodes + maps + tasklets).

    ``defined`` tracks arrays already materialised (function args + transients produced by earlier
    states/nodes); it is threaded across states so a cross-state transient is only allocated once.
    """
    lines: List[str] = []
    for node in dfs_topological_sort(state):
        if state.entry_node(node) is not None:
            continue  # emitted as part of its enclosing map scope
        if isinstance(node, nodes.MapEntry):
            lines.extend(_map_lines(state, sdfg, node, defined))
        elif isinstance(node, nodes.LibraryNode):
            try:
                lines.append(emit_library_node(node, state, sdfg))
            except UnsupportedLibraryNode as exc:
                raise UnsupportedNest(str(exc)) from exc
            for e in state.out_edges(node):
                defined.add(e.data.data)
        elif isinstance(node, nodes.Tasklet):
            lines.extend(_tasklet_lines(state, sdfg, node))
        elif isinstance(node, nodes.NestedSDFG):
            raise UnsupportedNest("top-level nested SDFG is not yet emitted")
    return lines


def _ordered_blocks(region) -> List:
    """Blocks of a control-flow region (SDFG or LoopRegion) in execution order."""
    return list(dfs_topological_sort(region, [region.start_block]))


def _emit_loop(loop: LoopRegion, sdfg: dace.SDFG, defined: set) -> List[str]:
    """Emit a ``LoopRegion`` as an init + ``while`` (do-while when ``inverted``) around its body.

    The loop variable and bounds come straight from DaCe's canonical ``init``/``condition``/
    ``update`` statements (already Python), so a ``for t in range(...)`` loop round-trips as
    ``t = 1 / while (t < TSTEPS): ... / t = (t + 1)``.
    """
    init = loop.init_statement.as_string.strip()
    cond = loop.loop_condition.as_string.strip()
    update = loop.update_statement.as_string.strip()
    body = _emit_region(loop, sdfg, defined) or ["pass"]
    indent = "    "
    if loop.inverted:  # do-while: body runs before the condition is tested
        lines = [init, "while True:"]
        lines += [indent + b for b in body]
        lines += [indent + update, f"{indent}if not ({cond}):", f"{indent}{indent}break"]
    else:
        lines = [init, f"while {cond}:"]
        lines += [indent + b for b in body]
        lines += [indent + update]
    return lines


def _emit_region(region, sdfg: dace.SDFG, defined: set) -> List[str]:
    """Numpy statements for every block of a control-flow region, in execution order."""
    lines: List[str] = []
    for block in _ordered_blocks(region):
        if isinstance(block, dace.SDFGState):
            lines.extend(_state_body(sdfg, block, defined))
        elif isinstance(block, LoopRegion):
            lines.extend(_emit_loop(block, sdfg, defined))
        else:
            raise UnsupportedNest(f"control-flow block not yet emitted: {type(block).__name__}")
    return lines


def _return_names(sdfg: dace.SDFG) -> List[str]:
    """DaCe return values (``__return`` / ``__return_0`` ...), in arglist order."""
    return [name for name in sdfg.arglist() if name == "__return" or name.startswith("__return_")]


def _emit_body(sdfg: dace.SDFG, returns: List[str]) -> List[str]:
    """Numpy statements for the whole SDFG, sharing one ``defined`` set across states and loops.

    ``returns`` names are excluded from ``defined`` so they are materialised inside the body (by a
    library-node assignment or a pre-allocated map) and then handed back via a ``return``.
    """
    defined = {name for name, desc in sdfg.arrays.items() if not desc.transient} - set(returns)
    body = _emit_region(sdfg, sdfg, defined)
    if returns:
        body.append(f"return {', '.join(returns)}")
    return body


def _render(fn_name: str, args: List[str], body: List[str]) -> str:
    lines = [f"def {fn_name}({', '.join(args)}):"]
    lines += ["    " + bl for bl in body]
    return "\n".join(lines) + "\n"


def nest_to_numpy(boundary: Boundary, fn_name: str = "kernel") -> str:
    """Standalone python source ``def <fn_name>(<args>): ...`` for an extracted nest's boundary."""
    returns = _return_names(boundary.standalone_sdfg)
    args = list(boundary.inputs)
    args += [o for o in boundary.outputs if o not in boundary.inputs and o not in returns]
    args += [s for s in boundary.symbols if s not in args]
    return _render(fn_name, args, _emit_body(boundary.standalone_sdfg, returns))


def sdfg_to_numpy(sdfg: dace.SDFG, fn_name: str = "kernel") -> str:
    """Standalone python source for a whole SDFG -- the corpus entry point.

    Signature is the SDFG's argument order (non-transient arrays + free symbols) minus the DaCe
    return values, which are handed back with a ``return`` instead of being passed in.
    """
    returns = _return_names(sdfg)
    args = [a for a in sdfg.arglist() if a not in returns]
    return _render(fn_name, args, _emit_body(sdfg, returns))
