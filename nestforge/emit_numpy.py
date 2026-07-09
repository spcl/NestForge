"""Emit a standalone numpy/python kernel from an extracted nest.

The emitter walks a state's dataflow in topological order and turns each node into numpy:

  * **library node** (``MatMul``, ``Dot``, ``Reduce``, ...) -> the equivalent numpy op, via
    :mod:`nestforge.emit_libnode` (the polybench/npbench kernels are almost all library nodes);
  * **map scope** -> explicit ``for`` loops with the tasklet body inlined (the fallback when a nest
    has no library node);
  * **free tasklet** -> its Python code inlined.

Memory model is **C-style**: the kernel allocates nothing. Every array -- inputs, outputs, the DaCe
``__return`` value, and scratch transients -- is a pre-allocated buffer *parameter* the caller
passes in, and array writes go in place (``name[:] = ...``). Only scalar transients become plain
python locals (a C ``double``). There is no ``return``. Connector identity is ignored: each access
is the array element the memlet names, and each array's data name doubles as the python variable.
Unsupported constructs raise :class:`UnsupportedNest`, so nothing is silently mis-emitted.
"""
from __future__ import annotations

import re
from typing import Dict, List

import dace
from dace import symbolic
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.sdfg.utils import dfs_topological_sort

from nestforge.emit_libnode import (UnsupportedLibraryNode, emit_library_node, index_str, is_scalar, read_expr,
                                    scalar_local, write_lhs)
from nestforge.extract import Boundary


class UnsupportedNest(Exception):
    """The nest uses a construct the numpy emitter does not handle."""


def _access(sdfg: dace.SDFG, name: str, subset: dace.subsets.Range) -> str:
    """Element access: a scalar-transient local ``name``, else the indexed buffer element ``name[idx]``.

    A scalar transient is a per-iteration temporary (a C ``double``); every other container -- arrays
    and passed-in scalars alike -- is a buffer parameter and is indexed.
    """
    if scalar_local(sdfg, name):
        return name
    return f"{name}[{index_str(subset)}]"


def _sub_connectors(code: str, conn_expr: Dict[str, str]) -> str:
    """Replace whole-word connector tokens in a tasklet's Python code with their expressions.

    All connectors are substituted in a single pass so a replacement expression that happens to
    contain another connector's name is not itself re-substituted.
    """
    if not conn_expr:
        return code
    pattern = re.compile(r"\b(" + "|".join(re.escape(c) for c in sorted(conn_expr, key=len, reverse=True)) + r")\b")
    return pattern.sub(lambda m: conn_expr[m.group(0)], code)


_DACE_CAST = re.compile(r"\bdace\.([A-Za-z_][A-Za-z0-9_]*)")


def _normalize_casts(code: str) -> str:
    """Rewrite DaCe dtype casts to numpy so the emitted kernel has no ``dace`` runtime dependency.

    Codegen inserts type-promotion casts such as ``dace.int64(x)`` / ``dace.complex128(x)``; numpy
    exposes the same dtype names as scalar constructors (``np.int64(x)``), so the rewrite is value
    preserving and drops the ``import dace`` the standalone kernel would otherwise need.
    """
    return _DACE_CAST.sub(r"np.\1", code)


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
    return [_normalize_casts(_sub_connectors(line, conn_expr)) for line in tasklet.code.as_string.splitlines()]


def _copy_lines(state: dace.SDFGState, sdfg: dace.SDFG, dst: nodes.AccessNode) -> List[str]:
    """Emit ``dst[..] = src[..]`` for each memlet copy feeding ``dst`` from another access node.

    Simplified SDFGs stage tasklet operands through scratch access nodes (a scalar ``s = A[i]`` or a
    sub-array ``B[:] = A[k, :, :]``), a plain data copy with no tasklet. The memlet names one side in
    ``memlet.data``/``subset`` and the other in ``other_subset``; we resolve which is the source.
    """
    lines: List[str] = []
    for e in state.in_edges(dst):
        if not isinstance(e.src, nodes.AccessNode):
            continue
        m = e.data
        if m.wcr is not None:
            raise UnsupportedNest(f"reduction (WCR) copy into {dst.data} is not yet emitted")
        if m.data == e.dst.data:
            dst_sub, src_sub = m.subset, m.other_subset
        else:  # memlet names the source side (the common case), or an unrelated array
            src_sub, dst_sub = m.subset, m.other_subset
        lines.append(f"{write_lhs(sdfg, e.dst.data, dst_sub)} = {read_expr(sdfg, e.src.data, src_sub)}")
    return lines


def _map_lines(state: dace.SDFGState, sdfg: dace.SDFG, entry: nodes.MapEntry) -> List[str]:
    """Emit a map scope as ``for`` loops over pre-allocated buffers (no allocation of its own)."""
    headers: List[str] = []
    for param, (beg, end, step) in zip(entry.map.params, entry.map.range.ranges):
        headers.append(f"for {param} in range({symbolic.symstr(beg)}, {symbolic.symstr(end + 1)}, "
                       f"{symbolic.symstr(step)}):")

    body: List[str] = []
    scope = state.scope_subgraph(entry, include_entry=False, include_exit=False)
    for node in dfs_topological_sort(scope):
        if isinstance(node, nodes.Tasklet):
            body.extend(_tasklet_lines(state, sdfg, node))
        elif isinstance(node, nodes.AccessNode):
            body.extend(_copy_lines(state, sdfg, node))
        elif isinstance(node, (nodes.MapEntry, nodes.NestedSDFG, nodes.LibraryNode)):
            raise UnsupportedNest(f"{type(node).__name__} nested inside a map is not yet emitted")

    lines = ["    " * depth + h for depth, h in enumerate(headers)]
    lines += ["    " * len(headers) + bl for bl in (body or ["pass"])]
    return lines


def _state_body(sdfg: dace.SDFG, state: dace.SDFGState) -> List[str]:
    """Numpy statements for a whole state, in dataflow order (library nodes + maps + tasklets)."""
    lines: List[str] = []
    for node in dfs_topological_sort(state):
        if state.entry_node(node) is not None:
            continue  # emitted as part of its enclosing map scope
        if isinstance(node, nodes.MapEntry):
            lines.extend(_map_lines(state, sdfg, node))
        elif isinstance(node, nodes.LibraryNode):
            try:
                lines.append(emit_library_node(node, state, sdfg))
            except UnsupportedLibraryNode as exc:
                raise UnsupportedNest(str(exc)) from exc
        elif isinstance(node, nodes.Tasklet):
            lines.extend(_tasklet_lines(state, sdfg, node))
        elif isinstance(node, nodes.AccessNode):
            lines.extend(_copy_lines(state, sdfg, node))
        elif isinstance(node, nodes.NestedSDFG):
            raise UnsupportedNest("top-level nested SDFG is not yet emitted")
    return lines


def _ordered_blocks(region) -> List:
    """Blocks of a control-flow region (SDFG or LoopRegion) in execution order."""
    return list(dfs_topological_sort(region, [region.start_block]))


def _emit_loop(loop: LoopRegion, sdfg: dace.SDFG) -> List[str]:
    """Emit a ``LoopRegion`` as init + ``while`` (do-while when ``inverted``) around its body.

    The bounds come from DaCe's canonical ``init``/``condition``/``update`` statements, so a
    ``for t in range(...)`` loop round-trips as ``t = 1 / while (t < TSTEPS): ... / t = (t + 1)``.
    ``init``/``update`` are optional (a bare ``while``); the condition is required.
    """
    if loop.loop_condition is None:
        raise UnsupportedNest(f"loop {loop.label} has no condition")
    cond = loop.loop_condition.as_string.strip()
    init = loop.init_statement.as_string.strip() if loop.init_statement is not None else None
    update = loop.update_statement.as_string.strip() if loop.update_statement is not None else None
    body = _emit_region(loop, sdfg) or ["pass"]
    ind = "    "

    lines: List[str] = []
    if init is not None:
        lines.append(init)
    if loop.inverted:  # do-while: body executes before the condition is tested
        lines.append("while True:")
        lines += [ind + b for b in body]
        test = [f"{ind}if not ({cond}):", f"{ind}{ind}break"]
        upd = [ind + update] if update is not None else []
        # update_before_condition selects whether the increment precedes or follows the test.
        lines += (upd + test) if loop.update_before_condition else (test + upd)
    else:
        lines.append(f"while {cond}:")
        lines += [ind + b for b in body]
        if update is not None:
            lines.append(ind + update)
    return lines


def _emit_conditional(cond_block: ConditionalBlock, sdfg: dace.SDFG) -> List[str]:
    """Emit a ``ConditionalBlock`` as ``if``/``elif``/``else`` over its branches.

    Each branch is ``(condition, region)``; the first keyed branch is ``if``, later keyed branches
    ``elif``, and the lone unconditional branch (``condition is None``, always last) is ``else``.
    """
    ind = "    "
    lines: List[str] = []
    for i, (condition, region) in enumerate(cond_block.branches):
        body = _emit_region(region, sdfg) or ["pass"]
        if condition is None:
            lines.append("else:")
        else:
            keyword = "if" if i == 0 else "elif"
            lines.append(f"{keyword} {condition.as_string.strip()}:")
        lines += [ind + b for b in body]
    return lines


def _interstate_lines(region, block) -> List[str]:
    """Assignments carried on the edge(s) entering ``block`` (e.g. an indirect index ``s = A[i]``).

    DaCe hoists a data-dependent index or loop-carried scalar onto the inter-state edge that reaches
    a block; those assignments must run before the block body or the symbols they define are unbound.
    A conditional (branching) edge is old-style control flow the numpy emitter does not model, so it
    is refused rather than silently dropped.
    """
    lines: List[str] = []
    for e in region.in_edges(block):
        if not e.data.assignments:
            continue
        if not e.data.is_unconditional():
            raise UnsupportedNest(f"conditional inter-state edge into {block.label} is not yet emitted")
        for lhs, rhs in e.data.assignments.items():
            lines.append(f"{lhs} = {_normalize_casts(rhs)}")
    return lines


def _emit_region(region, sdfg: dace.SDFG) -> List[str]:
    """Numpy statements for every block of a control-flow region, in execution order."""
    lines: List[str] = []
    for block in _ordered_blocks(region):
        lines.extend(_interstate_lines(region, block))
        if isinstance(block, dace.SDFGState):
            lines.extend(_state_body(sdfg, block))
        elif isinstance(block, LoopRegion):
            lines.extend(_emit_loop(block, sdfg))
        elif isinstance(block, ConditionalBlock):
            lines.extend(_emit_conditional(block, sdfg))
        else:
            raise UnsupportedNest(f"control-flow block not yet emitted: {type(block).__name__}")
    return lines


def _scratch_arrays(sdfg: dace.SDFG) -> List[str]:
    """Transient array buffers the caller must pre-allocate (scalar transients stay locals)."""
    return sorted(name for name, desc in sdfg.arrays.items() if desc.transient and not is_scalar(desc))


def _render(fn_name: str, args: List[str], body: List[str]) -> str:
    lines = [f"def {fn_name}({', '.join(args)}):"]
    lines += ["    " + bl for bl in body]
    return "\n".join(lines) + "\n"


def nest_to_numpy(boundary: Boundary, fn_name: str = "kernel") -> str:
    """Standalone python source ``def <fn_name>(<args>): ...`` for an extracted nest's boundary.

    Signature (all pre-allocated buffers): inputs, then extra outputs, then scratch transients, then
    size symbols. Everything is written in place; there is no return.
    """
    args = list(boundary.inputs)
    args += [o for o in boundary.outputs if o not in boundary.inputs]
    args += [s for s in _scratch_arrays(boundary.standalone_sdfg) if s not in args]
    args += [s for s in boundary.symbols if s not in args]
    return _render(fn_name, args, _emit_region(boundary.standalone_sdfg, boundary.standalone_sdfg))


def sdfg_to_numpy(sdfg: dace.SDFG, fn_name: str = "kernel") -> str:
    """Standalone python source for a whole SDFG -- the corpus entry point.

    Signature is the SDFG's own arguments (non-transient arrays + ``__return`` + scalars) followed by
    scratch transient buffers and size symbols -- all caller-allocated, all written in place.
    """
    data_args = [a for a in sdfg.arglist() if a in sdfg.arrays]
    symbols = [a for a in sdfg.arglist() if a not in sdfg.arrays]
    args = data_args + _scratch_arrays(sdfg) + symbols
    return _render(fn_name, args, _emit_region(sdfg, sdfg))
