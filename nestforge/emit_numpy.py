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

import ast
import copy
import re
from typing import Dict, List

import sympy

import dace
from dace import symbolic
from dace.frontend.operations import detect_reduction_type
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.sdfg.utils import dfs_topological_sort

from nestforge.emit_libnode import (UnsupportedLibraryNode, emit_library_node, index_str, is_scalar, read_expr,
                                    scalar_local, write_lhs)
from nestforge.extract import Boundary

try:
    from dace.transformation.interstate.expand_nested_sdfg_inputs import ExpandNestedSDFGInputs
except ImportError:  # the pass ships only on the DaCe `extended` branch nest-forge targets
    ExpandNestedSDFGInputs = None


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


#: DaCe dtype cast (``dace.<name>(x)``) -> the numpy scalar constructor that spells it. Restricted to
#: the fixed-width dtypes so a non-dtype ``dace.<attr>`` (``dace.math.sqrt``, ``dace.define_local``) is
#: never blindly rewritten to a nonexistent ``np.<attr>``; ``bool`` maps to ``np.bool_`` (``np.bool``
#: was removed in NumPy 2).
_DACE_DTYPES = {
    "bool": "np.bool_", "int8": "np.int8", "int16": "np.int16", "int32": "np.int32", "int64": "np.int64",
    "uint8": "np.uint8", "uint16": "np.uint16", "uint32": "np.uint32", "uint64": "np.uint64",
    "float16": "np.float16", "float32": "np.float32", "float64": "np.float64", "complex64": "np.complex64",
    "complex128": "np.complex128",
}
_DACE_CAST = re.compile(r"\bdace\.(" + "|".join(_DACE_DTYPES) + r")\b")

#: bare math intrinsic (as DaCe exposes it in tasklet code) -> the numpy function that computes it.
_MATH_INTRINSICS = {
    "sqrt": "np.sqrt", "cbrt": "np.cbrt", "exp": "np.exp", "exp2": "np.exp2", "expm1": "np.expm1",
    "log": "np.log", "log2": "np.log2", "log10": "np.log10", "log1p": "np.log1p", "sin": "np.sin",
    "cos": "np.cos", "tan": "np.tan", "asin": "np.arcsin", "acos": "np.arccos", "atan": "np.arctan",
    "atan2": "np.arctan2", "sinh": "np.sinh", "cosh": "np.cosh", "tanh": "np.tanh", "floor": "np.floor",
    "ceil": "np.ceil", "fabs": "np.abs", "sign": "np.sign",
}
_INTRINSIC_CALL = re.compile(r"(?<![\w.])(" + "|".join(_MATH_INTRINSICS) + r")(?=\s*\()")


def _normalize_casts(code: str) -> str:
    """Rewrite DaCe dtype casts and bare math intrinsics to numpy so the kernel needs no ``dace``/
    ``math`` runtime import.

    Codegen inserts type-promotion casts such as ``dace.int64(x)`` / ``dace.complex128(x)`` and emits
    math intrinsics unqualified (``sqrt(x)``, as a tasklet runs with ``math`` in scope). Numpy exposes
    the same dtype names as scalar constructors and the same functions, so both rewrites are value
    preserving; the intrinsic rewrite skips already-qualified names (``np.sqrt``) via a lookbehind, and
    only recognised dtype names are rewritten (an unknown ``dace.<attr>`` is left for the caller to hit).
    """
    code = _DACE_CAST.sub(lambda m: _DACE_DTYPES[m.group(1)], code)
    return _INTRINSIC_CALL.sub(lambda m: _MATH_INTRINSICS[m.group(1)], code)


#: reduction type -> ``(accumulator, term) -> combined expression`` for a WCR (augmented) write.
_WCR_BINOP = {
    dace.dtypes.ReductionType.Sum: lambda acc, t: f"{acc} + {t}",
    dace.dtypes.ReductionType.Product: lambda acc, t: f"{acc} * {t}",
    dace.dtypes.ReductionType.Max: lambda acc, t: f"np.maximum({acc}, {t})",
    dace.dtypes.ReductionType.Min: lambda acc, t: f"np.minimum({acc}, {t})",
}


def _tasklet_lines(state: dace.SDFGState, sdfg: dace.SDFG, tasklet: nodes.Tasklet) -> List[str]:
    """The tasklet's Python code with connectors substituted by the array element they name.

    A plain output connector is substituted by the target element it writes. A **WCR** (reduction)
    output -- a scatter-accumulate like ``hist[bin] += w`` -- is emitted as an augmented assignment:
    the body writes a fresh temporary, then the target is combined with it (``target = target + tmp``
    for Sum, ``np.maximum`` for Max, ...). Sequential emission makes this correct even when several
    iterations hit the same target element (the whole point of the WCR).
    """
    if tasklet.code.language != dace.dtypes.Language.Python:
        raise UnsupportedNest(f"tasklet {tasklet.label} is not Python ({tasklet.code.language})")
    conn_expr: Dict[str, str] = {}
    for e in state.in_edges(tasklet):
        if e.dst_conn is not None:
            conn_expr[e.dst_conn] = _access(sdfg, e.data.data, e.data.subset)
    wcr_updates: List[str] = []
    for e in state.out_edges(tasklet):
        if e.src_conn is None:
            continue
        target = _access(sdfg, e.data.data, e.data.subset)
        if e.data.wcr is None:
            conn_expr[e.src_conn] = target
            continue
        combine = _WCR_BINOP.get(detect_reduction_type(e.data.wcr))
        if combine is None:
            raise UnsupportedNest(f"tasklet {tasklet.label} has an unsupported WCR {e.data.wcr!r}")
        temp = f"__wcr_{e.src_conn}"
        conn_expr[e.src_conn] = temp  # the body writes the temporary, then we accumulate into target
        wcr_updates.append(f"{target} = {combine(target, temp)}")
    lines = [_normalize_casts(_sub_connectors(line, conn_expr)) for line in tasklet.code.as_string.splitlines()]
    return lines + wcr_updates


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
        elif m.data == e.src.data:
            src_sub, dst_sub = m.subset, m.other_subset
        else:  # the memlet must name one of the two endpoints; a third array is unexpected
            raise UnsupportedNest(f"copy memlet {m.data!r} names neither {e.src.data!r} nor {e.dst.data!r}")
        if len(sdfg.arrays[e.src.data].shape) == len(sdfg.arrays[e.dst.data].shape):
            # Same-rank copy: a ``None`` other-side subset means "the same range as the named side";
            # mirror it so a partial copy is not silently widened to the whole array.
            src_sub = src_sub if src_sub is not None else dst_sub
            dst_sub = dst_sub if dst_sub is not None else src_sub
            lhs, rhs = write_lhs(sdfg, e.dst.data, dst_sub), read_expr(sdfg, e.src.data, src_sub)
        else:
            # Reshape copy (e.g. ``(N,) <-> (N, 1)``): a scalar-local side stays bare; otherwise the
            # reshaping side keeps its subset explicit so a point index collapses the rank to match,
            # and a ``None`` side is the whole array.
            lhs = _reshape_side(sdfg, e.dst.data, dst_sub, write=True)
            rhs = _reshape_side(sdfg, e.src.data, src_sub, write=False)
        lines.append(f"{lhs} = {rhs}")
    return lines


def _reshape_side(sdfg: dace.SDFG, name: str, subset, write: bool) -> str:
    """One side of a rank-changing copy: bare for a scalar local, explicit ``name[idx]`` for the
    reshaping (rank-collapsing) side, else the whole array via the access oracle."""
    if scalar_local(sdfg, name):
        return name
    if subset is None:
        return write_lhs(sdfg, name, None) if write else read_expr(sdfg, name, None)
    return f"{name}[{index_str(subset)}]"


def _reject_underranked_codeblock_index(inner: dace.SDFG) -> None:
    """Refuse a nested SDFG whose inter-state code indexes a multi-dim array with too few indices.

    ``ExpandNestedSDFGInputs`` widens a collapsed size-1 inner array to the full outer array and
    offsets references by the map index -- but for a reference inside an inter-state *condition* or
    *assignment* (``I_0 = I[0]``) it adds only the first map dimension, leaving ``getAcc_I_0[__i0]``
    on an ``(N, N)`` array (a numpy row, not the element). That is a DaCe-pass gap; emitting it would
    produce ``if <array>:``, so we reject with a precise reason instead of a broken kernel.
    """
    for region in inner.all_control_flow_regions():
        for e in region.edges():
            codes = list(e.data.assignments.values())
            if not e.data.is_unconditional():
                codes.append(e.data.condition.as_string)
            for code in codes:
                try:
                    tree = ast.parse(code)
                except SyntaxError as exc:
                    raise UnsupportedNest(f"inter-state code {code!r} is not parseable Python") from exc
                for sub in ast.walk(tree):
                    if not (isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name)):
                        continue
                    desc = inner.arrays.get(sub.value.id)
                    if desc is None or len(desc.shape) <= 1:
                        continue
                    ndims = len(sub.slice.elts) if isinstance(sub.slice, ast.Tuple) else 1
                    if ndims < len(desc.shape):
                        raise UnsupportedNest(
                            f"nested SDFG under-indexes {sub.value.id!r} ({ndims} of {len(desc.shape)} dims) "
                            "in inter-state code -- ExpandNestedSDFGInputs offsets multi-dim conditions incompletely")


def _emit_nested_sdfg(state: dace.SDFGState, sdfg: dace.SDFG, node: nodes.NestedSDFG) -> List[str]:
    """Inline a nested SDFG (e.g. one map iteration's sub-kernel) as flat statements, in place.

    :func:`_expand_nested_sdfg_inputs` has already widened every in/out connector to the *full* outer
    array, and DaCe offsets the inner memlets by the enclosing map index, so the inner body reads and
    writes the outer buffers directly (``Z[j, k]``) using the map symbols already in scope. Emitting
    it is then just: bind the symbol mapping, alias each connector array to the outer array it binds,
    prefix any private transient that would shadow an outer buffer, and emit the inner body -- inner
    control flow (a masked ``np.where`` writing ``Z[j, k]`` only under a condition) stays correct
    because the write lands on the outer array in place, leaving the other elements untouched.
    """
    if ExpandNestedSDFGInputs is None:
        raise UnsupportedNest("nested SDFG emission needs ExpandNestedSDFGInputs (DaCe extended branch)")
    inner = copy.deepcopy(node.sdfg)
    conns = {e.dst_conn: e.data.data for e in state.in_edges(node) if e.data.data is not None}
    conns.update({e.src_conn: e.data.data for e in state.out_edges(node) if e.data.data is not None})
    for conn, outer in conns.items():
        if conn != outer:
            inner.replace(conn, outer)
        # Make the inner descriptor agree with the outer buffer it aliases: a connector may be a
        # scalar inside but a size-1 *array* outside (a nested return), and the two must index the
        # element the same way (bare ``x`` vs ``x[0]``) or the inner write and outer read disagree.
        if outer in sdfg.arrays:
            inner.arrays[outer] = copy.deepcopy(sdfg.arrays[outer])
    _reject_underranked_codeblock_index(inner)
    # A private (non-connector) inner transient becomes a plain python local. That only works for a
    # scalar; a private *array* transient would be emitted as ``name[:] = ...`` yet appears in no
    # signature (``_scratch_arrays`` scans the outer SDFG only), so it is refused rather than left
    # undefined. Connector arrays alias an outer buffer parameter and are exempt.
    outer_names = set(sdfg.arrays)
    node_id = state.node_id(node)
    for name, desc in list(inner.arrays.items()):
        if name in conns.values():
            continue
        if not is_scalar(desc):
            raise UnsupportedNest(f"nested SDFG private transient {name!r} is a non-scalar array; not allocated")
        # A private inner name that collides with an outer buffer would shadow it -- rename it.
        if name in outer_names:
            inner.replace(name, f"_ns{node_id}_{name}")

    lines: List[str] = []
    for sym, expr in node.symbol_mapping.items():
        if str(sym) != str(expr):
            lines.append(f"{sym} = {_normalize_casts(str(expr))}")
    lines += _emit_region(inner, inner)
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
        elif isinstance(node, nodes.NestedSDFG):
            body.extend(_emit_nested_sdfg(state, sdfg, node))
        elif isinstance(node, (nodes.MapEntry, nodes.LibraryNode)):
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
            lines.extend(_emit_nested_sdfg(state, sdfg, node))
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

    Each branch is ``(condition, region)``; keyed branches emit ``if`` then ``elif`` in their given
    order, and the unconditional branch (``condition is None``) always emits ``else`` last -- so a
    block whose unconditional branch is not stored last still produces valid Python. The condition is
    run through :func:`_normalize_casts` like every other emitted expression (a guard may hold a
    ``dace.<cast>`` or a bare math intrinsic).
    """
    ind = "    "
    keyed = [(c, r) for c, r in cond_block.branches if c is not None]
    unconditional = [r for c, r in cond_block.branches if c is None]
    lines: List[str] = []
    for i, (condition, region) in enumerate(keyed):
        keyword = "if" if i == 0 else "elif"
        lines.append(f"{keyword} {_normalize_casts(condition.as_string.strip())}:")
        lines += [ind + b for b in (_emit_region(region, sdfg) or ["pass"])]
    for region in unconditional:
        lines.append("else:")
        lines += [ind + b for b in (_emit_region(region, sdfg) or ["pass"])]
    return lines


def _strip_scalar_local_subscript(code: str, sdfg: dace.SDFG) -> str:
    """Drop the ``[0]`` index off a scalar-transient array in a raw DaCe code string.

    DaCe refers to a size-1 array as ``A[0]``, but the emitter treats a scalar transient as a bare
    python local (``A``); a raw inter-state assignment string (``bin = min(ret[0], ...)``) must be
    reconciled to that convention or the bare write and the indexed read disagree (an IndexError on a
    scalar). Only genuinely scalar-transient names are stripped; real arrays keep their indices.
    """
    for name, desc in sdfg.arrays.items():
        if scalar_local(sdfg, name):
            code = re.sub(rf"\b{re.escape(name)}\s*\[[^][]*\]", name, code)
    return code


def _interstate_lines(region, sdfg: dace.SDFG, block) -> List[str]:
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
            lines.append(f"{lhs} = {_strip_scalar_local_subscript(_normalize_casts(rhs), sdfg)}")
    return lines


def _emit_region(region, sdfg: dace.SDFG) -> List[str]:
    """Numpy statements for every block of a control-flow region, in execution order."""
    lines: List[str] = []
    for block in _ordered_blocks(region):
        lines.extend(_interstate_lines(region, sdfg, block))
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


def _symbol_ranges(sdfg: dace.SDFG) -> tuple:
    """``(lo_of, hi_of)``: each non-argument symbol -> its min / max value in kernel symbols.

    Two sources feed the range of a symbol used in a buffer shape:

    * an *increasing loop variable* takes ``[init, condition-bound]`` (``i = 0`` / ``i < N`` -> ``[0,
      N]``; ``i <= N`` -> ``N + 1``);
    * an *inter-state-assigned* config symbol takes each value it is assigned (a layer size reused
      across a network -- ``N2 = S0``/``S1``/``S2``, ``S1 = H``/``H-4``/``120``), so its range spans
      all of them.

    A symbol driven by several of these takes ``Min`` of its los / ``Max`` of its his; endpoints that
    still name another such symbol are resolved recursively, so the result is in kernel symbols only.
    """
    los: Dict[str, list] = {}
    his: Dict[str, list] = {}
    for cfg in sdfg.all_control_flow_regions():
        if isinstance(cfg, LoopRegion) and cfg.loop_condition is not None:
            rel = symbolic.pystr_to_symbolic(cfg.loop_condition.as_string)
            var = cfg.loop_variable
            if isinstance(rel, (sympy.StrictLessThan, sympy.LessThan)) and str(rel.lhs) == var:
                his.setdefault(var, []).append(rel.rhs + (1 if isinstance(rel, sympy.LessThan) else 0))
                los.setdefault(var, []).append(
                    symbolic.pystr_to_symbolic(cfg.init_statement.as_string.split("=", 1)[1])
                    if cfg.init_statement is not None else sympy.Integer(0))
        for e in cfg.edges():
            for var, rhs in e.data.assignments.items():
                try:
                    value = symbolic.pystr_to_symbolic(rhs)  # the config symbol takes exactly this value
                except Exception:
                    continue  # a data-dependent / non-symbolic assignment is not a usable size bound
                los.setdefault(var, []).append(value)
                his.setdefault(var, []).append(value)

    def resolve(bounds, combine):
        def r(expr, seen):
            for sym in list(expr.free_symbols):
                name = str(sym)
                if name in bounds and name not in seen:
                    parts = [r(b, seen | {name}) for b in bounds[name]]
                    expr = expr.subs(sym, combine(*parts) if len(parts) > 1 else parts[0])
            return expr
        return {v: r(combine(*bs) if len(bs) > 1 else bs[0], {v}) for v, bs in bounds.items()}

    return resolve(los, sympy.Min), resolve(his, sympy.Max)


def _max_over_loops(dim: sympy.Expr, lo_of: Dict[str, sympy.Expr], hi_of: Dict[str, sympy.Expr],
                    known: set):
    """Largest value a shape dimension takes over the loop variables' ranges, or ``None``.

    Each loop variable is substituted by the endpoint that maximises the dimension: its resolved upper
    bound where the dimension increases in it (``i + 1``), its resolved lower bound where it decreases
    (``M - i - 1``). Monotonicity is the sign of the (constant) derivative; a variable whose slope
    still holds free symbols (a power ``R**i``, slope ``R**i*log(R)`` of unknown sign) leaves the
    dimension unresolved (``None``), as does any residual non-kernel symbol.
    """
    result = dim
    for s in list(dim.free_symbols):
        if str(s) not in hi_of:
            continue
        slope = sympy.diff(dim, s)
        if slope.free_symbols:
            return None  # non-constant slope -> monotonicity undetermined
        result = result.subs(s, hi_of[str(s)] if slope.is_nonnegative else lo_of[str(s)])
    return result if not {str(s) for s in result.free_symbols} - known else None


def _maxsize_loop_scratch(sdfg: dace.SDFG, symbols: List[str]) -> dace.SDFG:
    """Return an SDFG where a scratch transient sized by loop variables is widened to a caller-sizable
    bound, so it stays a pre-allocated parameter and is addressed with the original ``0:extent`` slices.

    Each shape dimension is replaced by its maximum over the loop ranges (:func:`_max_over_loops`); a
    dimension whose extent is not a sound function of the kernel symbols (an FFT ``R**(K-i-1)``, a
    data-dependent CSR span) is left untouched and caught later by :func:`_reject_unsizable_scratch`.
    Runs on a copy so the caller is not mutated.
    """
    known = set(symbols)
    lo_of, hi_of = _symbol_ranges(sdfg)
    resize: Dict[str, tuple] = {}
    for name, desc in sdfg.arrays.items():
        if not desc.transient or is_scalar(desc):
            continue
        if not {str(s) for s in desc.free_symbols} - known:
            continue  # already sizable from kernel symbols
        new_shape = []
        for dim in desc.shape:
            sdim = sympy.sympify(dim)  # a literal-int dimension has no free symbols to widen
            if {str(s) for s in sdim.free_symbols} - known:
                widened = _max_over_loops(sdim, lo_of, hi_of, known)
                new_shape.append(widened if widened is not None else dim)
            else:
                new_shape.append(dim)
        if new_shape != list(desc.shape):
            resize[name] = tuple(new_shape)
    if not resize:
        return sdfg
    out = copy.deepcopy(sdfg)
    for name, shape in resize.items():
        old = out.arrays[name]
        out.arrays[name] = dace.data.Array(old.dtype, shape, transient=True, storage=old.storage)
    return out


def _reject_unsizable_scratch(sdfg: dace.SDFG, scratch: List[str], symbols: List[str]) -> None:
    """Refuse a scratch buffer whose shape depends on a symbol the caller cannot know at allocation.

    C-style emission makes every transient array a caller-allocated parameter, so its shape must be
    fixed from the kernel's own symbols (the size arguments). A transient whose shape references a
    *loop* variable (e.g. a per-stage FFT buffer shaped ``R**i``) has no fixed size at call time and
    cannot be pre-allocated -- refuse rather than emit a kernel whose signature can't be satisfied.
    """
    known = set(symbols)
    for name in scratch:
        loop_syms = {str(s) for s in sdfg.arrays[name].free_symbols} - known
        if loop_syms:
            raise UnsupportedNest(
                f"scratch buffer {name!r} has shape depending on {sorted(loop_syms)} (not kernel symbols); "
                "cannot be pre-allocated C-style")


def _render(fn_name: str, args: List[str], body: List[str]) -> str:
    lines = [f"def {fn_name}({', '.join(args)}):"]
    lines += ["    " + bl for bl in body]
    return "\n".join(lines) + "\n"


def _expand_nested_sdfg_inputs(sdfg: dace.SDFG) -> dace.SDFG:
    """Return an SDFG whose nested-SDFG in/out connectors are widened to the full outer arrays.

    A nested SDFG inside a map is handed per-iteration *slices*; DaCe's ``ExpandNestedSDFGInputs``
    rewrites its descriptors and memlets to reference the whole outer array offset by the map index,
    which is exactly the form :func:`_emit_nested_sdfg` inlines (the outer buffers, indexed by the map
    symbols). It is a semantics-preserving DaCe transformation, so the emitted numerics are unchanged.

    The transformation runs on a **copy** so the caller's SDFG is never mutated (emission is
    read-only); the copy is taken only when there is a nested SDFG to widen -- otherwise the caller's
    SDFG is returned unchanged. When the pass is unavailable, this is a no-op and
    :func:`_emit_nested_sdfg` refuses any nested SDFG.
    """
    if ExpandNestedSDFGInputs is None:
        return sdfg
    if not any(isinstance(n, nodes.NestedSDFG) for state in sdfg.all_states() for n in state.nodes()):
        return sdfg
    widened = copy.deepcopy(sdfg)
    widened.apply_transformations_repeated(ExpandNestedSDFGInputs)
    return widened


def nest_to_numpy(boundary: Boundary, fn_name: str = "kernel") -> str:
    """Standalone python source ``def <fn_name>(<args>): ...`` for an extracted nest's boundary.

    Signature (all pre-allocated buffers): inputs, then extra outputs, then scratch transients, then
    size symbols. Everything is written in place; there is no return.
    """
    standalone = _expand_nested_sdfg_inputs(boundary.standalone_sdfg)
    standalone = _maxsize_loop_scratch(standalone, boundary.symbols)
    scratch = _scratch_arrays(standalone)
    _reject_unsizable_scratch(standalone, scratch, boundary.symbols)
    args = list(boundary.inputs)
    args += [o for o in boundary.outputs if o not in boundary.inputs]
    args += [s for s in scratch if s not in args]
    args += [s for s in boundary.symbols if s not in args]
    return _render(fn_name, args, _emit_region(standalone, standalone))


def sdfg_to_numpy(sdfg: dace.SDFG, fn_name: str = "kernel") -> str:
    """Standalone python source for a whole SDFG -- the corpus entry point.

    Signature is the SDFG's own arguments (non-transient arrays + ``__return`` + scalars) followed by
    scratch transient buffers and size symbols -- all caller-allocated, all written in place.
    """
    sdfg = _expand_nested_sdfg_inputs(sdfg)
    symbols = [a for a in sdfg.arglist() if a not in sdfg.arrays]
    sdfg = _maxsize_loop_scratch(sdfg, symbols)
    data_args = [a for a in sdfg.arglist() if a in sdfg.arrays]
    scratch = _scratch_arrays(sdfg)
    _reject_unsizable_scratch(sdfg, scratch, symbols)
    args = data_args + scratch + symbols
    return _render(fn_name, args, _emit_region(sdfg, sdfg))
