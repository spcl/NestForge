"""Emit a standalone numpy/python kernel from an extracted nest.

Walks a state's dataflow topologically: library nodes -> the numpy op (via
:mod:`nestforge.emit_libnode`), map scopes -> ``for`` loops with the tasklet body inlined, free
tasklets -> their Python code inlined.

Memory model is **C-style**: the kernel allocates nothing. Every array -- inputs, outputs, the DaCe
``__return`` value, scratch transients -- is a pre-allocated buffer *parameter* written in place
(``name[:] = ...``); only scalar transients become plain python locals. There is no ``return``. Each
array's data name doubles as its python variable. Unsupported constructs raise
:class:`UnsupportedNest` rather than mis-emit silently.
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
from dace.sdfg.state import BreakBlock, ConditionalBlock, ContinueBlock, LoopRegion, ReturnBlock
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


def access(sdfg: dace.SDFG, name: str, subset: dace.subsets.Range) -> str:
    """Element access: a scalar-transient local ``name``, else the indexed buffer element ``name[idx]``.

    A scalar transient is a plain local; every other container (arrays and passed-in scalars alike)
    is a buffer parameter and is indexed.
    """
    if scalar_local(sdfg, name):
        return name
    return f"{name}[{index_str(subset)}]"


def sub_connectors(code: str, conn_expr: Dict[str, str]) -> str:
    """Replace whole-word connector tokens in a tasklet's Python code with their expressions.

    Single-pass substitution, so a replacement expression containing another connector's name is not
    itself re-substituted.
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
    "bool": "np.bool_",
    "int8": "np.int8",
    "int16": "np.int16",
    "int32": "np.int32",
    "int64": "np.int64",
    "uint8": "np.uint8",
    "uint16": "np.uint16",
    "uint32": "np.uint32",
    "uint64": "np.uint64",
    "float16": "np.float16",
    "float32": "np.float32",
    "float64": "np.float64",
    "complex64": "np.complex64",
    "complex128": "np.complex128",
}
_DACE_CAST = re.compile(r"\bdace\.(" + "|".join(_DACE_DTYPES) + r")\b")

#: bare math intrinsic (as DaCe exposes it in tasklet code) -> the numpy function that computes it.
_MATH_INTRINSICS = {
    "sqrt": "np.sqrt",
    "cbrt": "np.cbrt",
    "exp": "np.exp",
    "exp2": "np.exp2",
    "expm1": "np.expm1",
    "log": "np.log",
    "log2": "np.log2",
    "log10": "np.log10",
    "log1p": "np.log1p",
    "sin": "np.sin",
    "cos": "np.cos",
    "tan": "np.tan",
    "asin": "np.arcsin",
    "acos": "np.arccos",
    "atan": "np.arctan",
    "atan2": "np.arctan2",
    "sinh": "np.sinh",
    "cosh": "np.cosh",
    "tanh": "np.tanh",
    "floor": "np.floor",
    "ceil": "np.ceil",
    "fabs": "np.abs",
    "sign": "np.sign",
}
_INTRINSIC_CALL = re.compile(r"(?<![\w.])(" + "|".join(_MATH_INTRINSICS) + r")(?=\s*\()")

#: DaCe sympy user-functions -- ``symstr`` renders a subset index / map bound in function form because
#: sympy has no operator for them -> the numpy/python expression computing the same integer value.
#: ``int_ceil`` uses ``-((-a) // b)`` (sign-robust, ``== (a+b-1)//b`` for ``b > 0``).
_USERFUNC_REWRITES = {
    "int_floor": lambda a, b: f"(({a}) // ({b}))",
    "int_ceil": lambda a, b: f"(-((-({a})) // ({b})))",
    "ipow": lambda a, b: f"(({a}) ** ({b}))",
    "Mod": lambda a, b: f"(({a}) % ({b}))",
}

#: ``(dace.)?math.<fn>`` (a qualified intrinsic the bare-name rewrite deliberately skips) -> its numpy
#: form. Extra numpy-verbatim names beyond :data:`_MATH_INTRINSICS` that a TSVC/HPC tasklet may spell
#: qualified; anything else is left as ``math.<fn>`` (the emitter refuses rather than guess a bad name).
_NP_VERBATIM_MATH = frozenset({"power", "arcsin", "arccos", "arctan", "arctan2", "maximum", "minimum", "abs"})
_MATH_PREFIX_CALL = re.compile(r"\b(?:dace\.)?math\.(\w+)(?=\s*\()")


def apply_call(code: str, name: str, fn) -> str:
    """Rewrite every ``name(arg0, arg1)`` call in ``code`` via ``fn(arg0, arg1)``, matching balanced
    parentheses so nested arguments stay intact. Leftmost-first with a rescan from the start; the outer
    :func:`rewrite_userfuncs` fixpoint loop resolves calls nested inside the replacements."""
    pat = re.compile(rf"(?<![\w.]){re.escape(name)}\s*\(")
    while True:
        m = pat.search(code)
        if not m:
            return code
        i, depth, cur, args = m.end(), 1, "", []
        while i < len(code) and depth > 0:
            ch = code[i]
            if ch == "(":
                depth += 1
                cur += ch
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
                cur += ch
            elif ch == "," and depth == 1:
                args.append(cur)
                cur = ""
            else:
                cur += ch
            i += 1
        if depth != 0:
            return code  # unbalanced (should not happen on emitted code): leave it for the caller to hit
        args.append(cur)
        code = code[:m.start()] + fn(*(a.strip() for a in args)) + code[i + 1:]


def rewrite_userfuncs(code: str) -> str:
    """Rewrite DaCe sympy user-functions (:data:`_USERFUNC_REWRITES`) to numpy/python, to a fixpoint so
    cross-function nesting (``int_ceil`` inside an ``int_floor`` argument) fully resolves."""
    for _ in range(16):  # bounded: every rewrite strictly removes one user-function call
        new = code
        for name, fn in _USERFUNC_REWRITES.items():
            new = apply_call(new, name, fn)
        if new == code:
            return code
        code = new
    return code


def rewrite_math_prefix(code: str) -> str:
    """Rewrite a qualified ``dace.math.sin`` / ``math.sin`` to ``np.sin`` (the bare-name rewrite skips
    qualified forms via its lookbehind). Uses the same intrinsic map, so ``asin`` -> ``np.arcsin``."""

    def repl(m):
        fn = m.group(1)
        if fn in _MATH_INTRINSICS:
            return _MATH_INTRINSICS[fn]
        if fn in _NP_VERBATIM_MATH:
            return f"np.{fn}"
        return f"math.{fn}"

    return _MATH_PREFIX_CALL.sub(repl, code)


#: a C++ ``decltype(<connector>)`` cast prefix that DaCe emits in generated tasklet code (e.g. the
#: ``numpy.linspace`` tasklet writes ``decltype(__out)(stop - start)`` to force the connector's type). The
#: connector name holds no parentheses, so this strips just the ``decltype(...)`` prefix and leaves the
#: parenthesized value it casts -- numpy promotes types itself, so the cast is a no-op on the reference.
_DECLTYPE_CAST = re.compile(r"\bdecltype\s*\([^()]*\)")


def normalize_casts(code: str) -> str:
    """Rewrite DaCe dtype casts, math intrinsics, and sympy user-functions to numpy so the kernel needs
    no ``dace``/``math`` runtime import.

    All rewrites are value preserving. Order matters: the C++ ``decltype`` cast is stripped first (it wraps
    a parenthesized value the later rewrites still see); the qualified ``math.`` rewrite runs before the
    bare-name one, whose lookbehind then correctly skips the produced ``np.sin``; dtype casts and
    user-functions are independent.
    """
    code = _DECLTYPE_CAST.sub("", code)
    code = _DACE_CAST.sub(lambda m: _DACE_DTYPES[m.group(1)], code)
    code = rewrite_math_prefix(code)
    code = _INTRINSIC_CALL.sub(lambda m: _MATH_INTRINSICS[m.group(1)], code)
    return rewrite_userfuncs(code)


#: reduction type -> ``(accumulator, term) -> combined expression`` for a WCR (augmented) write.
_WCR_BINOP = {
    dace.dtypes.ReductionType.Sum: lambda acc, t: f"{acc} + {t}",
    dace.dtypes.ReductionType.Product: lambda acc, t: f"{acc} * {t}",
    dace.dtypes.ReductionType.Max: lambda acc, t: f"np.maximum({acc}, {t})",
    dace.dtypes.ReductionType.Min: lambda acc, t: f"np.minimum({acc}, {t})",
}


def tasklet_lines(state: dace.SDFGState, sdfg: dace.SDFG, tasklet: nodes.Tasklet) -> List[str]:
    """The tasklet's Python code with connectors substituted by the array element they name.

    A plain output connector is substituted by the target element it writes. A **WCR** (reduction)
    output -- a scatter-accumulate like ``hist[bin] += w`` -- becomes an augmented assignment: the
    body writes a fresh temporary, then the target is combined with it (``target = target + tmp`` for
    Sum, ``np.maximum`` for Max, ...). Sequential emission keeps this correct when several iterations
    hit the same target element (the whole point of the WCR).
    """
    if tasklet.code.language != dace.dtypes.Language.Python:
        raise UnsupportedNest(f"tasklet {tasklet.label} is not Python ({tasklet.code.language})")
    conn_expr: Dict[str, str] = {}
    for e in state.in_edges(tasklet):
        if e.dst_conn is not None:
            conn_expr[e.dst_conn] = access(sdfg, e.data.data, e.data.subset)
    wcr_updates: List[str] = []
    for e in state.out_edges(tasklet):
        if e.src_conn is None:
            continue
        target = access(sdfg, e.data.data, e.data.subset)
        if e.data.wcr is None:
            conn_expr[e.src_conn] = target
            continue
        combine = _WCR_BINOP.get(detect_reduction_type(e.data.wcr))
        if combine is None:
            raise UnsupportedNest(f"tasklet {tasklet.label} has an unsupported WCR {e.data.wcr!r}")
        temp = f"__wcr_{e.src_conn}"
        conn_expr[e.src_conn] = temp  # the body writes the temporary, then we accumulate into target
        wcr_updates.append(f"{target} = {combine(target, temp)}")
    lines = [normalize_casts(sub_connectors(line, conn_expr)) for line in tasklet.code.as_string.splitlines()]
    # The accumulate lines are built here, not from tasklet code, but they embed the target's rendered
    # subset -- a strided/derived index renders as sympy ``int_floor(...)``, which is not python. They need
    # the same normalization the body lines get.
    return lines + [normalize_casts(u) for u in wcr_updates]


def copy_lines(state: dace.SDFGState, sdfg: dace.SDFG, dst: nodes.AccessNode) -> List[str]:
    """Emit ``dst[..] = src[..]`` for each memlet copy feeding ``dst`` from an access node or a map entry.

    Simplified SDFGs stage tasklet operands through scratch access nodes -- a plain tasklet-less data
    copy -- in two forms this handles:

    * **access-node -> access-node** (``s = A[i]`` / ``B[:] = A[k, :, :]``): the memlet names one side
      in ``memlet.data``/``subset`` and the other in ``other_subset``; we resolve which is the source.
    * **map-entry -> access-node**: DaCe requires an indexed array read to pass through a data node, so
      an in-map element read is staged as ``b_index = b[i]`` (a scalar access node fed by the map
      entry). The memlet names the OUTER source array in ``memlet.data``/``subset``; the scratch is the
      destination (``other_subset``). Emitting this load lets a later ``a[b_index]`` gather resolve --
      ``b_index`` is bound to ``b[i]`` before its first use.
    """
    lines: List[str] = []
    for e in state.in_edges(dst):
        m = e.data
        if isinstance(e.src, nodes.AccessNode):
            if m.data == e.dst.data:
                dst_sub, src_sub = m.subset, m.other_subset
            elif m.data == e.src.data:
                src_sub, dst_sub = m.subset, m.other_subset
            else:  # the memlet must name one of the two endpoints; a third array is unexpected
                raise UnsupportedNest(f"copy memlet {m.data!r} names neither {e.src.data!r} nor {e.dst.data!r}")
            src_name = e.src.data
        elif isinstance(e.src, nodes.MapEntry):
            # Staged array read through the map entry: the memlet names the outer source array + the
            # element/range read; the scratch access node is the destination.
            src_name, src_sub, dst_sub = m.data, m.subset, m.other_subset
        else:
            # A Tasklet / MapExit source's WCR is emitted where that edge is owned: tasklet_lines (the
            # tasklet out-edge names the outer array) and map_exit_writes (the exit's in-edges) handle the
            # accumulate, so this copy edge is a genuine no-op. A LibraryNode / NestedSDFG source ignores
            # its out-edge WCR, but that is refused at the source's own emitter (out_lhs / emit_nested_sdfg),
            # so it never reaches a silent overwrite here either.
            continue
        if len(sdfg.arrays[src_name].shape) == len(sdfg.arrays[dst.data].shape):
            # Same-rank copy: a ``None`` other-side subset means "the same range as the named side";
            # mirror it so a partial copy is not silently widened to the whole array.
            src_sub = src_sub if src_sub is not None else dst_sub
            dst_sub = dst_sub if dst_sub is not None else src_sub
            lhs, rhs = write_lhs(sdfg, dst.data, dst_sub), read_expr(sdfg, src_name, src_sub)
            dst_read = read_expr(sdfg, dst.data, dst_sub)
        else:
            # Reshape/rank-changing copy (``(N,) <-> (N, 1)``, or an array element -> scalar staging): a
            # scalar-local side stays bare; the reshaping side keeps its subset explicit so a point
            # index collapses the rank to match, and a ``None`` side is the whole array.
            lhs = reshape_side(sdfg, dst.data, dst_sub, write=True)
            rhs = reshape_side(sdfg, src_name, src_sub, write=False)
            dst_read = reshape_side(sdfg, dst.data, dst_sub, write=False)
        if m.wcr is not None:
            # A reduction *copy* (AccessNode -> AccessNode carrying a WCR, e.g. a privatized accumulator
            # copied back): accumulate rather than overwrite -- ``dst = combine(dst, src)``. Tasklet WCRs
            # go through their own path in tasklet_lines; this is the copy-edge analogue.
            combine = _WCR_BINOP.get(detect_reduction_type(m.wcr))
            if combine is None:
                raise UnsupportedNest(f"reduction (WCR) copy into {dst.data} has an unsupported WCR {m.wcr!r}")
            rhs = combine(dst_read, rhs)
        lines.append(normalize_casts(f"{lhs} = {rhs}"))  # a strided subset may render an int_floor/int_ceil index
    return lines


def reshape_side(sdfg: dace.SDFG, name: str, subset, write: bool) -> str:
    """One side of a rank-changing copy: bare for a scalar local, explicit ``name[idx]`` for the
    reshaping (rank-collapsing) side, else the whole array via the access oracle."""
    if scalar_local(sdfg, name):
        return name
    if subset is None:
        return write_lhs(sdfg, name, None) if write else read_expr(sdfg, name, None)
    return f"{name}[{index_str(subset)}]"


def reject_underranked_codeblock_index(inner: dace.SDFG) -> None:
    """Refuse a nested SDFG whose inter-state code indexes a multi-dim array with too few indices.

    ``ExpandNestedSDFGInputs`` widens a collapsed size-1 inner array to the full outer array and
    offsets references by the map index -- but for a reference inside an inter-state *condition* or
    *assignment* (``I_0 = I[0]``) it adds only the first map dimension, leaving ``getAcc_I_0[__i0]``
    on an ``(N, N)`` array (a numpy row, not the element). That DaCe-pass gap would emit
    ``if <array>:``, so we reject with a precise reason instead of a broken kernel.
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


def emit_nested_sdfg(state: dace.SDFGState, sdfg: dace.SDFG, node: nodes.NestedSDFG) -> List[str]:
    """Inline a nested SDFG (e.g. one map iteration's sub-kernel) as flat statements, in place.

    :func:`expand_nested_sdfg_inputs` has already widened every in/out connector to the *full* outer
    array, and DaCe offsets the inner memlets by the enclosing map index, so the inner body reads and
    writes the outer buffers directly (``Z[j, k]``) using the map symbols already in scope. Emitting
    it is then: bind the symbol mapping, alias each connector array to the outer array it binds,
    rename any private transient that would shadow an outer buffer, and emit the inner body. Inner
    control flow (a masked write to ``Z[j, k]``) stays correct because the write lands on the outer
    array in place, leaving the other elements untouched.
    """
    for e in state.out_edges(node):
        if e.data.wcr is not None:
            # This function replays the inner body only (emit_region below); it never applies a WCR carried
            # on the nested SDFG's OUTER output edge, so an accumulate would silently become an overwrite.
            # At a map exit map_exit_writes already refuses this; guarding here also covers a nested SDFG at
            # state-body level, which that guard never sees. Refuse -> the ExternalCall uses the DaCe variant.
            raise UnsupportedNest(
                f"nested SDFG output into {e.data.data} carries a reduction (WCR) that emit_nested_sdfg does "
                "not apply; not emittable as numpy -- fall back to the DaCe variant")
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
    reject_underranked_codeblock_index(inner)
    # A private (non-connector) inner transient becomes a plain python local. That only works for a
    # scalar; a private *array* transient would be emitted as ``name[:] = ...`` yet appears in no
    # signature (``scratch_arrays`` scans the outer SDFG only), so it is refused rather than left
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
            lines.append(f"{sym} = {normalize_casts(str(expr))}")
    lines += emit_region(inner, inner)
    return lines


def map_exit_writes(state: dace.SDFGState, sdfg: dace.SDFG, entry: nodes.MapEntry) -> List[str]:
    """Writes that leave the map through its exit from an in-scope AccessNode.

    A tasklet that writes an outer array does so directly (its out-edge memlet names the target, handled
    in :func:`tasklet_lines`). But a canonicalized reduction stages its result in a privatized
    accumulator AccessNode *inside* the map, whose out-edge to the MapExit carries the WCR into the real
    output (``_priv_out += acc`` for a Sum). :func:`copy_lines` cannot emit that write -- it hangs off
    the outer output node, which is fed by the MapExit (a passthrough it skips). Emit it here, inside the
    loop body, accumulating for a WCR edge and overwriting otherwise, so the reduction reaches the buffer.

    Mirrors the MapEntry-read convention in :func:`copy_lines`: on an edge into the exit the memlet names
    the OUTER destination array (``m.data``/``m.subset``); the in-scope access node is the source, read at
    ``m.other_subset``.
    """
    lines: List[str] = []
    for e in state.in_edges(state.exit_node(entry)):
        if not isinstance(e.src, nodes.AccessNode) or state.entry_node(e.src) is not entry:
            # Only a tasklet WCR out-edge accumulates on emit: tasklet_lines rewrites it as
            # ``target = target + tmp``. Every other source at this exit -- a nested map, a library node,
            # or a NESTED SDFG -- emits its body only (emit_nested_sdfg calls emit_region and never applies
            # the out-edge WCR), so a reduction reaching the exit from one of them would silently become an
            # overwrite (last-write-wins), mis-emitting the reduction as a no-op -- the very bug this
            # function exists to fix, one nest-level in. Refuse those so the ExternalCall falls back to the
            # DaCe variant rather than emit a wrong kernel.
            if e.data.wcr is not None and not isinstance(e.src, nodes.Tasklet):
                raise UnsupportedNest(
                    f"reduction (WCR) leaves the map exit from a {type(e.src).__name__}, not an in-scope "
                    "accumulator access node; not emittable as numpy -- fall back to the DaCe variant")
            continue
        m = e.data
        dst_name, dst_sub, src_sub = m.data, m.subset, m.other_subset
        src_name = e.src.data
        if src_name == dst_name:  # a self-edge moves nothing out of the map
            continue
        if len(sdfg.arrays[src_name].shape) == len(sdfg.arrays[dst_name].shape):
            src_sub = src_sub if src_sub is not None else dst_sub
            lhs, rhs = write_lhs(sdfg, dst_name, dst_sub), read_expr(sdfg, src_name, src_sub)
            dst_read = read_expr(sdfg, dst_name, dst_sub)
        else:
            lhs = reshape_side(sdfg, dst_name, dst_sub, write=True)
            rhs = reshape_side(sdfg, src_name, src_sub, write=False)
            dst_read = reshape_side(sdfg, dst_name, dst_sub, write=False)
        if m.wcr is not None:
            combine = _WCR_BINOP.get(detect_reduction_type(m.wcr))
            if combine is None:
                raise UnsupportedNest(f"reduction (WCR) write-out into {dst_name} has an unsupported WCR {m.wcr!r}")
            rhs = combine(dst_read, rhs)
        lines.append(normalize_casts(f"{lhs} = {rhs}"))
    return lines


def map_lines(state: dace.SDFGState, sdfg: dace.SDFG, entry: nodes.MapEntry) -> List[str]:
    """Emit a map scope as ``for`` loops over pre-allocated buffers (no allocation of its own)."""
    headers: List[str] = []
    for param, (beg, end, step) in zip(entry.map.params, entry.map.range.ranges):
        headers.append(
            normalize_casts(f"for {param} in range({symbolic.symstr(beg)}, {symbolic.symstr(end + 1)}, "
                            f"{symbolic.symstr(step)}):"))  # a bound may render an int_floor/int_ceil

    body: List[str] = []
    scope = state.scope_subgraph(entry, include_entry=False, include_exit=False)
    for node in dfs_topological_sort(scope):
        # ``scope_subgraph`` returns the whole subtree, including a nested map's own descendants; emit only
        # the DIRECT children of THIS map here and let the recursion below handle a nested map's body, so a
        # grandchild is not also emitted flat at the wrong depth (mirrors the guard in ``state_body``).
        if state.entry_node(node) is not entry:
            continue
        if isinstance(node, nodes.Tasklet):
            body.extend(tasklet_lines(state, sdfg, node))
        elif isinstance(node, nodes.AccessNode):
            body.extend(copy_lines(state, sdfg, node))
        elif isinstance(node, nodes.NestedSDFG):
            body.extend(emit_nested_sdfg(state, sdfg, node))
        elif isinstance(node, nodes.MapEntry):
            # A map nested in a map -> emit it as deeper ``for`` loops. ``map_lines`` returns a
            # self-contained header+body block; splicing it into this body and applying the uniform
            # per-level indent below preserves the relative nesting.
            body.extend(map_lines(state, sdfg, node))
        elif isinstance(node, nodes.LibraryNode):
            raise UnsupportedNest(f"{type(node).__name__} nested inside a map is not yet emitted")

    body.extend(map_exit_writes(state, sdfg, entry))  # reductions/writes leaving the map via its exit

    lines = ["    " * depth + h for depth, h in enumerate(headers)]
    lines += ["    " * len(headers) + bl for bl in (body or ["pass"])]
    return lines


def state_body(sdfg: dace.SDFG, state: dace.SDFGState) -> List[str]:
    """Numpy statements for a whole state, in dataflow order (library nodes + maps + tasklets)."""
    lines: List[str] = []
    for node in dfs_topological_sort(state):
        if state.entry_node(node) is not None:
            continue  # emitted as part of its enclosing map scope
        if isinstance(node, nodes.MapEntry):
            lines.extend(map_lines(state, sdfg, node))
        elif isinstance(node, nodes.LibraryNode):
            try:
                lines.extend(emit_library_node(node, state, sdfg))
            except UnsupportedLibraryNode as exc:
                raise UnsupportedNest(str(exc)) from exc
        elif isinstance(node, nodes.Tasklet):
            lines.extend(tasklet_lines(state, sdfg, node))
        elif isinstance(node, nodes.AccessNode):
            lines.extend(copy_lines(state, sdfg, node))
        elif isinstance(node, nodes.NestedSDFG):
            lines.extend(emit_nested_sdfg(state, sdfg, node))
    return lines


def ordered_blocks(region) -> List:
    """Blocks of a control-flow region (SDFG or LoopRegion) in execution order."""
    return list(dfs_topological_sort(region, [region.start_block]))


def emit_loop(loop: LoopRegion, sdfg: dace.SDFG) -> List[str]:
    """Emit a ``LoopRegion`` as init + ``while`` (do-while when ``inverted``) around its body.

    Bounds come from DaCe's canonical ``init``/``condition``/``update`` statements, so a
    ``for t in range(...)`` round-trips as ``t = 1 / while (t < TSTEPS): ... / t = (t + 1)``.
    ``init``/``update`` are optional (a bare ``while``); the condition is required.
    """
    if loop.loop_condition is None:
        raise UnsupportedNest(f"loop {loop.label} has no condition")
    cond = normalize_casts(loop.loop_condition.as_string.strip())
    init = normalize_casts(loop.init_statement.as_string.strip()) if loop.init_statement is not None else None
    update = normalize_casts(loop.update_statement.as_string.strip()) if loop.update_statement is not None else None
    body = emit_region(loop, sdfg) or ["pass"]
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


def emit_conditional(cond_block: ConditionalBlock, sdfg: dace.SDFG) -> List[str]:
    """Emit a ``ConditionalBlock`` as ``if``/``elif``/``else`` over its branches.

    Each branch is ``(condition, region)``; keyed branches emit ``if`` then ``elif`` in order, and the
    unconditional branch (``condition is None``) always emits ``else`` last -- so a block whose
    unconditional branch is not stored last still produces valid Python.
    """
    ind = "    "
    keyed = [(c, r) for c, r in cond_block.branches if c is not None]
    unconditional = [r for c, r in cond_block.branches if c is None]
    lines: List[str] = []
    for i, (condition, region) in enumerate(keyed):
        keyword = "if" if i == 0 else "elif"
        lines.append(f"{keyword} {normalize_casts(condition.as_string.strip())}:")
        lines += [ind + b for b in (emit_region(region, sdfg) or ["pass"])]
    for region in unconditional:
        lines.append("else:")
        lines += [ind + b for b in (emit_region(region, sdfg) or ["pass"])]
    return lines


def strip_scalar_local_subscript(code: str, sdfg: dace.SDFG) -> str:
    """Drop the ``[0]`` index off a scalar-transient array in a raw DaCe code string.

    DaCe refers to a size-1 array as ``A[0]``, but the emitter treats a scalar transient as a bare
    local (``A``); a raw inter-state assignment string (``bin = min(ret[0], ...)``) must match that
    convention or the bare write and the indexed read disagree (IndexError on a scalar). Only
    genuinely scalar-transient names are stripped; real arrays keep their indices.
    """
    for name, desc in sdfg.arrays.items():
        if scalar_local(sdfg, name):
            code = re.sub(rf"\b{re.escape(name)}\s*\[[^][]*\]", name, code)
    return code


def interstate_lines(region, sdfg: dace.SDFG, block) -> List[str]:
    """Assignments carried on the edge(s) entering ``block`` (e.g. an indirect index ``s = A[i]``).

    DaCe hoists a data-dependent index or loop-carried scalar onto the inter-state edge that reaches
    a block; those assignments must run before the block body or the symbols they define are unbound.
    A conditional (branching) edge is an unstructured goto -- old-style state-machine control flow the
    straight-line topological emission does not model (structured branches are ``ConditionalBlock`` s,
    structured loops ``LoopRegion`` s) -- so ANY conditional edge is refused, whether or not it carries
    assignments, rather than silently emitting the successor blocks as if the branch were always taken.
    """
    lines: List[str] = []
    for e in region.in_edges(block):
        if not e.data.is_unconditional():
            raise UnsupportedNest(
                f"conditional inter-state edge into {block.label} (unstructured goto/branch) is not emitted")
        for lhs, rhs in e.data.assignments.items():
            lines.append(f"{lhs} = {strip_scalar_local_subscript(normalize_casts(rhs), sdfg)}")
    return lines


def emit_region(region, sdfg: dace.SDFG) -> List[str]:
    """Numpy statements for every block of a control-flow region, in execution order."""
    lines: List[str] = []
    for block in ordered_blocks(region):
        lines.extend(interstate_lines(region, sdfg, block))
        if isinstance(block, dace.SDFGState):
            lines.extend(state_body(sdfg, block))
        elif isinstance(block, LoopRegion):
            lines.extend(emit_loop(block, sdfg))
        elif isinstance(block, ConditionalBlock):
            lines.extend(emit_conditional(block, sdfg))
        elif isinstance(block, BreakBlock):
            lines.append("break")  # exits the enclosing while (emit_loop); its region-DFS successor is the loop exit
        elif isinstance(block, ContinueBlock):
            lines.append("continue")
        elif isinstance(block, ReturnBlock):
            # Early return out of the SDFG. A whole-kernel python ``return`` matches this exactly (it exits
            # the function == exits the SDFG). Externalizing a *sub-nest* that carries one is refused up
            # front by ``reject_early_return`` (the return would only exit the nest, not the enclosing SDFG).
            lines.append("return")
        else:
            raise UnsupportedNest(f"control-flow block not yet emitted: {type(block).__name__}")
    return lines


def scratch_arrays(sdfg: dace.SDFG) -> List[str]:
    """Transient array buffers the caller must pre-allocate (scalar transients stay locals)."""
    return sorted(name for name, desc in sdfg.arrays.items() if desc.transient and not is_scalar(desc))


def symbol_ranges(sdfg: dace.SDFG) -> tuple:
    """``(lo_of, hi_of)``: each non-argument symbol -> its min / max value in kernel symbols.

    Two sources feed the range of a symbol used in a buffer shape:

    * an *increasing loop variable* takes ``[init, condition-bound]`` (``i = 0`` / ``i < N`` -> ``[0,
      N]``; ``i <= N`` -> ``N + 1``);
    * an *inter-state-assigned* config symbol takes each value it is assigned (a layer size reused
      across a network -- ``N2 = S0``/``S1``/``S2``), so its range spans all of them.

    A symbol driven by several takes ``Min`` of its los / ``Max`` of its his; endpoints naming another
    such symbol are resolved recursively, so the result is in kernel symbols only.
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
                    symbolic.pystr_to_symbolic(cfg.init_statement.as_string.split("=", 1)[1]) if cfg.
                    init_statement is not None else sympy.Integer(0))
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


def max_over_loops(dim: sympy.Expr, lo_of: Dict[str, sympy.Expr], hi_of: Dict[str, sympy.Expr], known: set):
    """Largest value a shape dimension takes over the loop variables' ranges, or ``None``.

    Each loop variable is substituted by the endpoint that maximises the dimension: its resolved upper
    bound where the dimension increases in it (``i + 1``), its resolved lower bound where it decreases
    (``M - i - 1``). Monotonicity is the sign of the (constant) derivative; a variable whose slope
    still holds free symbols (``R**i``, slope of unknown sign) leaves the dimension unresolved
    (``None``), as does any residual non-kernel symbol.
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


def maxsize_loop_scratch(sdfg: dace.SDFG, symbols: List[str]) -> dace.SDFG:
    """Return an SDFG where a scratch transient sized by loop variables is widened to a caller-sizable
    bound, so it stays a pre-allocated parameter and is addressed with the original ``0:extent`` slices.

    Each shape dimension is replaced by its maximum over the loop ranges (:func:`max_over_loops`); a
    dimension whose extent is not a sound function of the kernel symbols (an FFT ``R**(K-i-1)``, a
    data-dependent CSR span) is left untouched and caught later by :func:`reject_unsizable_scratch`.
    Runs on a copy so the caller is not mutated.
    """
    known = set(symbols)
    lo_of, hi_of = symbol_ranges(sdfg)
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
                widened = max_over_loops(sdim, lo_of, hi_of, known)
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


def reject_unsizable_scratch(sdfg: dace.SDFG, scratch: List[str], symbols: List[str]) -> None:
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


def has_enclosing_loop(block) -> bool:
    """True if ``block`` has a ``LoopRegion`` ancestor within its SDFG -- the loop a ``break`` /
    ``continue`` inside it would target (walks ``parent_graph`` up to the SDFG root)."""
    region = block.parent_graph
    while region is not None:
        if isinstance(region, LoopRegion):
            return True
        region = region.parent_graph
    return False


def reject_nonexternalizable(sdfg: dace.SDFG) -> None:
    """Refuse a nest whose control flow cannot be carried into a standalone kernel unchanged.

    Both cases arise when extraction cuts a control-flow *target* out of the nest:

    * a ``ReturnBlock`` returns out of the *enclosing* SDFG. A python ``return`` in the extracted kernel
      would exit only that kernel and let the caller resume the rest of the original program -- the
      original return instead skipped everything after the nest. (A return is still emittable for a
      *whole*-SDFG kernel via :func:`sdfg_to_numpy`, where ``return`` == exit the kernel == exit the
      SDFG.)
    * a ``BreakBlock`` / ``ContinueBlock`` targets its innermost enclosing loop. If that loop was not
      pulled into the nest, the emitted ``break`` / ``continue`` lands outside any loop -- a wrong
      target (or a python ``SyntaxError``). Externalizing must take the whole loop the break exits, not
      an inner sub-nest; a break/continue with no enclosing ``LoopRegion`` in the nest is an illegal cut.

    Refuse rather than silently mis-emit.
    """
    for block in sdfg.all_control_flow_blocks():
        if isinstance(block, ReturnBlock):
            raise UnsupportedNest(f"nest contains an early return ({block.label}); a return out of the enclosing SDFG "
                                  "cannot be externalized into a standalone kernel")
        if isinstance(block, (BreakBlock, ContinueBlock)) and not has_enclosing_loop(block):
            raise UnsupportedNest(
                f"nest contains a {type(block).__name__} ({block.label}) whose target loop is outside the "
                "extracted scope; externalize the loop it breaks out of, not an inner nest")


def render(fn_name: str, args: List[str], body: List[str]) -> str:
    lines = [f"def {fn_name}({', '.join(args)}):"]
    lines += ["    " + bl for bl in body]
    return "\n".join(lines) + "\n"


def expand_nested_sdfg_inputs(sdfg: dace.SDFG) -> dace.SDFG:
    """Return an SDFG whose nested-SDFG in/out connectors are widened to the full outer arrays.

    A nested SDFG inside a map is handed per-iteration *slices*; DaCe's ``ExpandNestedSDFGInputs``
    rewrites its descriptors and memlets to reference the whole outer array offset by the map index --
    exactly the form :func:`emit_nested_sdfg` inlines. It is semantics-preserving, so numerics are
    unchanged. Runs on a **copy** (emission is read-only), taken only when there is a nested SDFG to
    widen. When the pass is unavailable this is a no-op and :func:`emit_nested_sdfg` refuses any
    nested SDFG.
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
    reject_nonexternalizable(boundary.standalone_sdfg)  # early return / orphan break cannot be externalized
    standalone = expand_nested_sdfg_inputs(boundary.standalone_sdfg)
    standalone = maxsize_loop_scratch(standalone, boundary.symbols)
    scratch = scratch_arrays(standalone)
    reject_unsizable_scratch(standalone, scratch, boundary.symbols)
    args = list(boundary.inputs)
    args += [o for o in boundary.outputs if o not in boundary.inputs]
    args += [s for s in scratch if s not in args]
    args += [s for s in boundary.symbols if s not in args]
    return render(fn_name, args, emit_region(standalone, standalone))


def sdfg_to_numpy(sdfg: dace.SDFG, fn_name: str = "kernel") -> str:
    """Standalone python source for a whole SDFG -- the corpus entry point.

    Signature is the SDFG's own arguments (non-transient arrays + ``__return`` + scalars) followed by
    scratch transient buffers and size symbols -- all caller-allocated, all written in place.
    """
    sdfg = expand_nested_sdfg_inputs(sdfg)
    symbols = [a for a in sdfg.arglist() if a not in sdfg.arrays]
    sdfg = maxsize_loop_scratch(sdfg, symbols)
    data_args = [a for a in sdfg.arglist() if a in sdfg.arrays]
    scratch = scratch_arrays(sdfg)
    reject_unsizable_scratch(sdfg, scratch, symbols)
    args = data_args + scratch + symbols
    return render(fn_name, args, emit_region(sdfg, sdfg))
