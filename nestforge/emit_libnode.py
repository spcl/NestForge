"""Emit numpy operations for DaCe library nodes (BLAS / LinAlg / reductions / FFT).

Re-emitting a library node as the equivalent numpy op (``A @ B``, ``np.add.reduce`` ...) keeps the
reference dense so OptArena's translators recover an idiomatic kernel; only a nest with *no* library
node falls back to explicit ``for`` loops (see :mod:`nestforge.emit_numpy`). Operand resolution
(memlet -> ``name`` or ``name[slice]``) lives here so the registry stays a flat
class-name -> statement-builder table.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import dace
from dace import symbolic
from dace.frontend.operations import detect_reduction_type
from dace.sdfg import nodes


class UnsupportedLibraryNode(Exception):
    """No numpy emission is registered for this library node class."""


# ----- operand resolution -------------------------------------------------------------------------


def index_str(subset: dace.subsets.Range, keep_singleton: bool = False) -> str:
    """Format a subset as a numpy index/slice string (end-inclusive DaCe range -> ``beg:end+1``).

    A singleton range ``(k, k, 1)`` renders as the scalar ``k`` (drops the dim, for a tasklet's scalar
    operand); ``keep_singleton`` renders it ``k:k+1`` instead so an array copy/slice preserves the dim
    per numpy semantics (a ``[N,1]`` slice stays ``[N,1]``, not ``[N]``)."""
    parts = []
    for (beg, end, step) in subset.ranges:
        if str(beg) == str(end):
            parts.append(
                f"{symbolic.symstr(beg)}:{symbolic.symstr(beg + 1)}" if keep_singleton else symbolic.symstr(beg))
        elif str(step) == "1":
            parts.append(f"{symbolic.symstr(beg)}:{symbolic.symstr(end + 1)}")
        else:
            parts.append(f"{symbolic.symstr(beg)}:{symbolic.symstr(end + 1)}:{symbolic.symstr(step)}")
    return ", ".join(parts)


def covers_whole(subset: dace.subsets.Range, desc) -> bool:
    """True if ``subset`` spans the entire array descriptor (so no slice suffix is needed)."""
    if len(subset.ranges) != len(desc.shape):
        return False
    for (beg, end, step), dim in zip(subset.ranges, desc.shape):
        if str(beg) != "0" or str(step) != "1" or symbolic.symstr(end + 1) != symbolic.symstr(dim):
            return False
    return True


def is_scalar(desc) -> bool:
    """A single-element data container (a DaCe ``Scalar`` or a size-1 array)."""
    return isinstance(desc, dace.data.Scalar) or desc.total_size == 1


def scalar_local(sdfg: dace.SDFG, name: str) -> bool:
    """A *scalar transient* -- emitted as a plain python variable (a C local ``double``), not a buffer."""
    desc = sdfg.arrays[name]
    return desc.transient and is_scalar(desc)


def read_expr(sdfg: dace.SDFG, name: str, subset: Optional[dace.subsets.Range], keep_singleton: bool = False) -> str:
    """Read expression for ``name[subset]``: scalar-transient variable, whole array, or slice.

    Each array's data name doubles as its python variable (no connector renaming). ``None`` subset
    means the whole array. ``keep_singleton`` preserves a length-1 dim as ``k:k+1`` (see
    :func:`index_str`) -- set only when the counterpart of a copy keeps that dim.
    """
    if scalar_local(sdfg, name):
        return name
    desc = sdfg.arrays[name]
    if is_scalar(desc):
        # A size-1 buffer (non-transient) is a scalar value: read its sole element, mirroring write_lhs.
        # Reading the bare name yields the whole (1,) array, so ``s[0] = out`` assigns a length-1 array
        # into a scalar slot -- a NumPy 2 "setting an array element with a sequence" error (and the C
        # translator would see a double* where a double is meant).
        return f"{name}[0]"
    if subset is None or covers_whole(subset, desc):
        return name
    return f"{name}[{index_str(subset, keep_singleton=keep_singleton)}]"


def write_lhs(sdfg: dace.SDFG, name: str, subset: Optional[dace.subsets.Range], keep_singleton: bool = False) -> str:
    """Write target for ``name[subset]``. Arrays are written *in place* (``name[:]`` / ``name[slice]``)
    to fill the pre-allocated buffer rather than rebind it; scalar transients are plain assignments.
    ``None`` subset means the whole array. ``keep_singleton`` as in :func:`read_expr`."""
    if scalar_local(sdfg, name):
        return name
    desc = sdfg.arrays[name]
    if is_scalar(desc):
        # A size-1 buffer (non-transient, so not a plain local): write its sole element. ``name[:] =
        # scalar`` is valid numpy but the C translator mis-lowers it to ``name = scalar`` (double ->
        # double*); ``name[0] =`` matches how the element is read back and translates correctly.
        return f"{name}[0]"
    if subset is None or covers_whole(subset, desc):
        return f"{name}[:]"
    return f"{name}[{index_str(subset, keep_singleton=keep_singleton)}]"


def memlet_expr(memlet: dace.Memlet, sdfg: dace.SDFG) -> str:
    """Read expression for a memlet's data (see :func:`read_expr`). A library-node operand keeps its
    length-1 dims so its shape matches the numpy op (a ``[N,1]`` column stays 2-D)."""
    return read_expr(sdfg, memlet.data, memlet.subset, keep_singleton=True)


def memlet_lhs(memlet: dace.Memlet, sdfg: dace.SDFG) -> str:
    """Write target for a memlet's data (see :func:`write_lhs`). Keeps length-1 dims so the target
    shape matches the numpy op's result (``acc[0:N, 0:1] = A @ mass`` for an ``[N,1]`` product)."""
    return write_lhs(sdfg, memlet.data, memlet.subset, keep_singleton=True)


def in_expr(state: dace.SDFGState, node: nodes.Node, conn: Optional[str], sdfg: dace.SDFG) -> str:
    edges = list(state.in_edges(node))
    edge = edges[0] if conn is None else next(e for e in edges if e.dst_conn == conn)
    return memlet_expr(edge.data, sdfg)


def out_lhs(state: dace.SDFGState, node: nodes.Node, conn: Optional[str], sdfg: dace.SDFG) -> str:
    edges = list(state.out_edges(node))
    edge = edges[0] if conn is None else next(e for e in edges if e.src_conn == conn)
    if edge.data.wcr is not None:
        # No library-node emitter applies an output-edge WCR: every one writes via write_lhs (a plain
        # ``name[..] = rhs``), so a reduction accumulating the node's result into an existing buffer would
        # silently become an overwrite. Refuse it so the ExternalCall falls back to the DaCe variant.
        raise UnsupportedLibraryNode(
            f"{type(node).__name__} output into {edge.data.data} carries a reduction (WCR) that no library-node "
            "emitter applies; not emittable as numpy -- fall back to the DaCe variant")
    return memlet_lhs(edge.data, sdfg)


# ----- per-library-node numpy statements ----------------------------------------------------------

_REDUCTION_FUNC = {
    dace.dtypes.ReductionType.Sum: "np.add",
    dace.dtypes.ReductionType.Product: "np.multiply",
    dace.dtypes.ReductionType.Max: "np.maximum",
    dace.dtypes.ReductionType.Min: "np.minimum",
    dace.dtypes.ReductionType.Logical_And: "np.logical_and",
    dace.dtypes.ReductionType.Logical_Or: "np.logical_or",
}


def is_one(v) -> bool:
    """Value-aware ``v == 1`` (a ``sympy.Float(1.0)`` from a SymbolicProperty compares unequal to the
    int ``1`` under sympy's structural ``__eq__``, so a plain ``== 1`` misfires)."""
    return symbolic.equal_valued(1, v)


def is_zero(v) -> bool:
    """Value-aware ``v == 0`` (see :func:`is_one`); ``str(beta) not in ('0','0.0')`` misses e.g. ``0.00``."""
    return symbolic.equal_valued(0, v)


def scaled(expr: str, coeff) -> str:
    """``expr`` multiplied by ``coeff``, or ``expr`` unchanged when ``coeff`` is 1."""
    return expr if is_one(coeff) else f"{coeff} * ({expr})"


def transposed(expr: str, trans: bool) -> str:
    """``(expr).T`` when ``trans`` (a BLAS ``transA``/``transB`` flag), else ``expr``. Parenthesized so a
    slice/expression operand transposes as a whole (``(A[0:N, 0:K]).T``)."""
    return f"({expr}).T" if trans else expr


def emit_matmul(node, state, sdfg) -> str:
    a = in_expr(state, node, "_a", sdfg)
    b = in_expr(state, node, "_b", sdfg)
    c_read = in_expr(state, node, "_c", sdfg) if not is_zero(node.beta) else None
    expr = scaled(f"{a} @ {b}", node.alpha)
    if c_read is not None:
        expr = f"{expr} + {node.beta} * {c_read}"
    return f"{out_lhs(state, node, '_c', sdfg)} = {expr}"


def emit_gemm(node, state, sdfg) -> str:
    """``alpha * (opA(A) @ opB(B)) + beta * C`` -- the BLAS GEMM ``MatMul`` expands to, connectors
    ``_a``/``_b``/``_c`` with ``transA``/``transB`` operand transposes and scalar ``alpha``/``beta``."""
    reject_runtime_scalars(node, state)
    a = transposed(in_expr(state, node, "_a", sdfg), node.transA)
    b = transposed(in_expr(state, node, "_b", sdfg), node.transB)
    expr = scaled(f"{a} @ {b}", node.alpha)
    if not is_zero(node.beta):
        expr = f"{expr} + {node.beta} * {in_expr(state, node, '_c', sdfg)}"
    return f"{out_lhs(state, node, '_c', sdfg)} = {expr}"


def emit_gemv(node, state, sdfg) -> str:
    """``alpha * (opA(A) @ x) + beta * y`` -- BLAS GEMV (matrix-vector), connectors ``_A``/``_x``/``_y``."""
    a = transposed(in_expr(state, node, "_A", sdfg), node.transA)
    x = in_expr(state, node, "_x", sdfg)
    expr = scaled(f"{a} @ {x}", node.alpha)
    if not is_zero(node.beta):
        expr = f"{expr} + {node.beta} * {in_expr(state, node, '_y', sdfg)}"
    return f"{out_lhs(state, node, '_y', sdfg)} = {expr}"


def emit_ger(node, state, sdfg) -> str:
    """``alpha * outer(x, y) + A`` -- BLAS GER rank-1 update; connectors ``_x``/``_y``/``_A`` -> ``_res``."""
    x = in_expr(state, node, "_x", sdfg)
    y = in_expr(state, node, "_y", sdfg)
    a = in_expr(state, node, "_A", sdfg)
    return f"{out_lhs(state, node, '_res', sdfg)} = {scaled(f'np.outer({x}, {y})', node.alpha)} + {a}"


def emit_axpy(node, state, sdfg) -> str:
    """``a * x + y`` -- BLAS AXPY; connectors ``_x``/``_y`` -> ``_res``."""
    x = in_expr(state, node, "_x", sdfg)
    y = in_expr(state, node, "_y", sdfg)
    return f"{out_lhs(state, node, '_res', sdfg)} = {scaled(x, node.a)} + {y}"


def emit_batched_matmul(node, state, sdfg) -> str:
    """Batched ``A @ B`` (numpy ``@`` contracts the trailing two dims, broadcasting the batch); connectors
    ``_a``/``_b`` -> ``_c``, with ``transA``/``transB`` swapping the last two axes. The pure DaCe expansion
    ignores ``beta`` and there is no ``_c`` input connector to accumulate into, so a non-zero ``beta`` is
    refused rather than silently dropped."""
    if not is_zero(node.beta):
        raise UnsupportedLibraryNode(f"BatchedMatMul with beta={node.beta} has no _c input to accumulate")
    a = in_expr(state, node, "_a", sdfg)
    b = in_expr(state, node, "_b", sdfg)
    if node.transA:
        a = f"np.swapaxes({a}, -1, -2)"
    if node.transB:
        b = f"np.swapaxes({b}, -1, -2)"
    return f"{out_lhs(state, node, '_c', sdfg)} = {scaled(f'{a} @ {b}', node.alpha)}"


def emit_einsum(node, state, sdfg) -> str:
    """``np.einsum(einsum_str, *operands)`` -- operands ordered by connector name (the specialize
    expansion contracts ``*sorted(inputs)``, and ``LiftEinsum`` builds ``einsum_str`` from the same
    sorted order, so operand ``i`` of the string is the ``i``-th sorted connector). ``alpha``/``beta`` are
    the node properties multiplied by any ``_alpha``/``_beta`` runtime-scalar connectors (they compose)."""
    coeff = {"_alpha": str(node.alpha), "_beta": str(node.beta)}
    operands = []
    for e in state.in_edges(node):
        if e.dst_conn in coeff:
            coeff[e.dst_conn] = f"({coeff[e.dst_conn]}) * ({memlet_expr(e.data, sdfg)})"
        else:
            operands.append((e.dst_conn, memlet_expr(e.data, sdfg)))
    ordered = [expr for _, expr in sorted(operands)]
    expr = f"np.einsum('{node.einsum_str}', {', '.join(ordered)})"
    has_alpha = any(e.dst_conn == "_alpha" for e in state.in_edges(node))
    has_beta = any(e.dst_conn == "_beta" for e in state.in_edges(node))
    if has_alpha or not is_one(node.alpha):
        expr = f"({coeff['_alpha']}) * ({expr})"
    if has_beta or not is_zero(node.beta):
        out_edge = next(iter(state.out_edges(node)))
        expr = f"{expr} + ({coeff['_beta']}) * ({memlet_expr(out_edge.data, sdfg)})"
    return f"{out_lhs(state, node, None, sdfg)} = {expr}"


def emit_tensordot(node, state, sdfg) -> str:
    """``np.tensordot(L, R, axes=(left_axes, right_axes))`` with an optional output-mode ``permutation``
    (``np.transpose`` of the contraction result); connectors ``_left_tensor``/``_right_tensor`` ->
    ``_out_tensor``."""
    left = in_expr(state, node, "_left_tensor", sdfg)
    right = in_expr(state, node, "_right_tensor", sdfg)
    expr = f"np.tensordot({left}, {right}, axes=({list(node.left_axes)}, {list(node.right_axes)}))"
    if node.permutation is not None and list(node.permutation) != list(range(len(node.permutation))):
        expr = f"np.transpose({expr}, axes={list(node.permutation)})"
    return f"{out_lhs(state, node, '_out_tensor', sdfg)} = {expr}"


def emit_inv(node, state, sdfg) -> str:
    """``np.linalg.inv(A)`` -- matrix inverse; connectors ``_ain`` -> ``_aout``."""
    return f"{out_lhs(state, node, '_aout', sdfg)} = np.linalg.inv({in_expr(state, node, '_ain', sdfg)})"


def emit_fft(node, state, sdfg) -> str:
    """``factor * np.fft.fft(x)`` -- DaCe's forward DFT is unnormalized (numpy's default ``norm``), scaled
    by the ``factor`` normalization coefficient; connectors ``_inp`` -> ``_out``."""
    inp = in_expr(state, node, "_inp", sdfg)
    return f"{out_lhs(state, node, '_out', sdfg)} = {scaled(f'np.fft.fft({inp})', node.factor)}"


def emit_ifft(node, state, sdfg) -> str:
    """``factor * np.fft.ifft(x, norm='forward')`` -- DaCe's inverse DFT omits the ``1/N`` (``norm='forward'``
    puts no scale on the inverse), scaled by ``factor``; connectors ``_inp`` -> ``_out``."""
    inp = in_expr(state, node, "_inp", sdfg)
    call = "np.fft.ifft(%s, norm='forward')" % inp
    return f"{out_lhs(state, node, '_out', sdfg)} = {scaled(call, node.factor)}"


_ARGREDUCE_FUNC = {"max": ("np.argmax", "np.max"), "min": ("np.argmin", "np.min")}


def emit_argreduce(node, state, sdfg):
    """``np.argmax``/``np.argmin`` over the (contiguous) input slice -> a slice-local index plus its value;
    connector ``_in`` -> ``_out_idx`` (position) and ``_out_val`` (extreme). Two statements."""
    argfn, valfn = _ARGREDUCE_FUNC[node.op]
    inp = in_expr(state, node, "_in", sdfg)
    return [
        f"{out_lhs(state, node, '_out_idx', sdfg)} = {argfn}({inp})",
        f"{out_lhs(state, node, '_out_val', sdfg)} = {valfn}({inp})",
    ]


_SCAN_FUNC = {
    dace.dtypes.ReductionType.Sum: "np.cumsum",
    dace.dtypes.ReductionType.Product: "np.cumprod",
    dace.dtypes.ReductionType.Max: "np.maximum.accumulate",
    dace.dtypes.ReductionType.Min: "np.minimum.accumulate",
}


def emit_scan(node, state, sdfg) -> str:
    """Inclusive prefix scan -> ``np.cumsum``/``np.cumprod``/``np.maximum.accumulate``/``.minimum.``;
    connector ``_scan_in`` -> ``_scan_out``. Exclusive / seeded / strided scans have no direct numpy
    form and are refused."""
    from dace.libraries.standard.nodes.scan import ScanOp
    red = {
        ScanOp.SUM: dace.dtypes.ReductionType.Sum,
        ScanOp.PRODUCT: dace.dtypes.ReductionType.Product,
        ScanOp.MIN: dace.dtypes.ReductionType.Min,
        ScanOp.MAX: dace.dtypes.ReductionType.Max
    }.get(node.op)
    func = _SCAN_FUNC.get(red)
    if func is None:
        raise UnsupportedLibraryNode(f"Scan with unsupported op {node.op}")
    if node.exclusive or str(node.stride) != "1" or "_scan_init" in node.in_connectors:
        raise UnsupportedLibraryNode("only an inclusive unit-stride unseeded Scan maps to a numpy accumulate")
    return f"{out_lhs(state, node, '_scan_out', sdfg)} = {func}({in_expr(state, node, '_scan_in', sdfg)})"


def emit_integer_sort(node, state, sdfg) -> str:
    """Ascending 1-D sort -> ``np.sort`` (numpy sorts ascending by default); connectors ``_keys_in`` ->
    ``_keys_out``."""
    return f"{out_lhs(state, node, '_keys_out', sdfg)} = np.sort({in_expr(state, node, '_keys_in', sdfg)})"


def emit_scatter_conflict_check(node, state, sdfg) -> List[str]:
    """Count duplicate values in a 1-D integer index array (scatter no-conflict proof); connector
    ``_idx_in`` -> ``_count_out`` (a host int64 scalar, ``0`` iff the index is a permutation).

    Emits the **TAGCOUNT** form -- last-writer-wins ownership then a mismatch count -- rather than the
    libnode's sort + adjacent-equal scan; both yield ``count = N - #distinct``. The owner buffer is
    runtime-sized (``max(idx) + 1``) and initialised to ``-1`` so a zero index value cannot be mistaken
    for an already-claimed owner slot. Temp names are suffixed by the output array so two
    ScatterConflictCheck nodes in one state don't share the owner / accumulator locals. The internal max
    / count stay plain scalars; the size-1 ``_count_out`` buffer is written through :func:`out_lhs` like
    the other scalar outputs. The index is non-empty by construction (a ScatterConflictCheck exists only
    to guard a real scatter), so ``np.max`` always has an element."""
    idx = in_expr(state, node, "_idx_in", sdfg)
    count = out_lhs(state, node, "_count_out", sdfg)
    tag = next(e for e in state.out_edges(node) if e.src_conn == "_count_out").data.data
    mx, owner, i, acc = f"__scc_{tag}_max", f"__scc_{tag}_owner", f"__scc_{tag}_i", f"__scc_{tag}_count"
    return [
        f"{mx} = int(np.max({idx}))",
        f"{owner} = np.full({mx} + 1, -1, np.int64)",
        f"for {i} in range({idx}.shape[0]):",
        f"    {owner}[{idx}[{i}]] = {i}",
        f"{acc} = 0",
        f"for {i} in range({idx}.shape[0]):",
        f"    if {owner}[{idx}[{i}]] != {i}:",
        f"        {acc} += 1",
        f"{count} = {acc}",
    ]


def out_data_name(state: dace.SDFGState, node: nodes.Node, conn: str) -> str:
    """The array NAME an output connector writes (for reading the buffer's prior value, e.g. the
    untouched triangle a symmetric BLAS update preserves)."""
    return next(e for e in state.out_edges(node) if e.src_conn == conn).data.data


def has_in_conn(state: dace.SDFGState, node: nodes.Node, conn: str) -> bool:
    return any(e.dst_conn == conn for e in state.in_edges(node))


def reject_runtime_scalars(node, state: dace.SDFGState) -> None:
    """Refuse a BLAS node with a wired runtime ``_alpha`` / ``_beta`` scalar connector: the emitters
    below fold only the compile-time ``alpha``/``beta`` properties, so a runtime coefficient would be
    silently dropped. Uncommon (kernels almost always bake the scalars); refuse rather than mis-scale."""
    if has_in_conn(state, node, "_alpha") or has_in_conn(state, node, "_beta"):
        raise UnsupportedLibraryNode(f"{type(node).__name__} has a runtime _alpha/_beta scalar connector; "
                                     "only compile-time alpha/beta are emitted -- fall back to the DaCe variant")


def triangle_funcs(uplo: str):
    """``(write_fn, keep_fn, keep_offset)`` for a symmetric BLAS update that touches only the ``uplo``
    triangle: the written triangle (incl. diagonal) plus the STRICT opposite triangle of the prior value,
    so the untouched half is preserved bit-for-bit (the DaCe reference leaves it unchanged)."""
    return ("np.tril", "np.triu", 1) if uplo == "L" else ("np.triu", "np.tril", -1)


def emit_syrk(node, state, sdfg) -> str:
    """BLAS SYRK: ``C := alpha*(A@A.T) + beta*C`` (``trans='N'``, ``A`` is ``N x K``) or ``alpha*(A.T@A)``
    (``trans='T'``, ``A`` is ``K x N``), updating ONLY the ``uplo`` triangle of symmetric ``C``; the
    opposite triangle keeps its prior value. Connectors ``_a``/``_c`` -> ``_c`` (in-place); ``_c`` is read
    only when ``beta != 0``."""
    reject_runtime_scalars(node, state)
    a = in_expr(state, node, "_a", sdfg)
    prod = f"{a}.T @ {a}" if node.trans == "T" else f"{a} @ {a}.T"
    rhs = scaled(f"({prod})", node.alpha)
    c_buf = out_data_name(state, node, "_c")
    if not is_zero(node.beta):
        rhs = f"{rhs} + {node.beta} * {read_expr(sdfg, c_buf, None)}"
    write, keep, off = triangle_funcs(node.uplo)
    return f"{out_lhs(state, node, '_c', sdfg)} = {write}({rhs}) + {keep}({read_expr(sdfg, c_buf, None)}, {off})"


def emit_syr2k(node, state, sdfg) -> str:
    """BLAS SYR2K: ``C := alpha*(A@B.T + B@A.T) + beta*C`` (``trans='N'``, ``A``/``B`` are ``N x K``) or
    ``alpha*(A.T@B + B.T@A) + beta*C`` (``trans='T'``, ``A``/``B`` are ``K x N``); ``A``/``B`` read in FULL
    (rectangular), only the ``uplo`` triangle of symmetric ``C`` written. Connectors ``_a``/``_b``/``_c`` ->
    ``_c``; ``_c`` read only when ``beta != 0``."""
    reject_runtime_scalars(node, state)
    a = in_expr(state, node, "_a", sdfg)
    b = in_expr(state, node, "_b", sdfg)
    prod = f"{a}.T @ {b} + {b}.T @ {a}" if node.trans == "T" else f"{a} @ {b}.T + {b} @ {a}.T"
    rhs = scaled(f"({prod})", node.alpha)
    c_buf = out_data_name(state, node, "_c")
    if not is_zero(node.beta):
        rhs = f"{rhs} + {node.beta} * {read_expr(sdfg, c_buf, None)}"
    write, keep, off = triangle_funcs(node.uplo)
    return f"{out_lhs(state, node, '_c', sdfg)} = {write}({rhs}) + {keep}({read_expr(sdfg, c_buf, None)}, {off})"


def emit_symm(node, state, sdfg) -> str:
    """BLAS SYMM: ``C := alpha*(A@B) + beta*C`` (``side='L'``) or ``alpha*(B@A) + beta*C`` (``side='R'``),
    where ``A`` is symmetric with only its ``uplo`` triangle stored -> reconstruct the FULL symmetric ``A``
    first. Output ``C`` is FULL (no triangle masking). Connectors ``_a``/``_b``/``_c`` -> ``_c``."""
    reject_runtime_scalars(node, state)
    a = in_expr(state, node, "_a", sdfg)
    b = in_expr(state, node, "_b", sdfg)
    asym = f"(np.tril({a}) + np.tril({a}, -1).T)" if node.uplo == "L" else f"(np.triu({a}) + np.triu({a}, 1).T)"
    mat = f"{asym} @ {b}" if node.side == "L" else f"{b} @ {asym}"
    rhs = scaled(mat, node.alpha)
    if not is_zero(node.beta):
        rhs = f"{rhs} + {node.beta} * {in_expr(state, node, '_c', sdfg)}"
    return f"{out_lhs(state, node, '_c', sdfg)} = {rhs}"


def emit_potrf(node, state, sdfg) -> List[str]:
    """LAPACK POTRF (Cholesky factorization) -> ``np.linalg.cholesky`` (lower) / ``.conj().T`` (upper),
    mirroring :func:`emit_cholesky`; connectors ``_xin`` -> ``_xout`` (+ optional ``_res`` info scalar,
    always success ``0`` for a numpy reference)."""
    a = in_expr(state, node, "_xin", sdfg)
    expr = f"np.linalg.cholesky({a})"
    if not node.lower:
        expr = f"({expr}).conj().T"
    lines = [f"{out_lhs(state, node, '_xout', sdfg)} = {expr}"]
    if any(e.src_conn == "_res" for e in state.out_edges(node)):
        lines.append(f"{out_lhs(state, node, '_res', sdfg)} = np.array(0, np.int32)")
    return lines


def emit_dot(node, state, sdfg) -> str:
    x = in_expr(state, node, "_x", sdfg)
    y = in_expr(state, node, "_y", sdfg)
    return f"{out_lhs(state, node, '_result', sdfg)} = np.dot({x}, {y})"


def emit_transpose(node, state, sdfg) -> str:
    inp = in_expr(state, node, "_inp", sdfg)
    return f"{out_lhs(state, node, '_out', sdfg)} = np.transpose({inp})"


def emit_solve(node, state, sdfg) -> str:
    ain = in_expr(state, node, "_ain", sdfg)
    bin_ = in_expr(state, node, "_bin", sdfg)
    return f"{out_lhs(state, node, '_bout', sdfg)} = np.linalg.solve({ain}, {bin_})"


def emit_cholesky(node, state, sdfg) -> str:
    a = in_expr(state, node, "_a", sdfg)
    # numpy returns the lower factor L (A = L @ L.conj().T); the upper factor is L.conj().T.
    expr = f"np.linalg.cholesky({a})"
    if not node.lower:
        expr = f"({expr}).conj().T"
    return f"{out_lhs(state, node, '_b', sdfg)} = {expr}"


def emit_tensortranspose(node, state, sdfg) -> str:
    inp = in_expr(state, node, "_inp_tensor", sdfg)
    if str(node.beta) not in ("0", "0.0"):
        raise UnsupportedLibraryNode(f"TensorTranspose with beta={node.beta} (accumulate) is not emitted")
    expr = f"np.transpose({inp}, axes={list(node.axes)})"
    if str(node.alpha) != "1":
        expr = f"{node.alpha} * ({expr})"
    return f"{out_lhs(state, node, '_out_tensor', sdfg)} = {expr}"


def emit_reduce(node, state, sdfg) -> str:
    red = detect_reduction_type(node.wcr)
    func = _REDUCTION_FUNC.get(red)
    if func is None:
        raise UnsupportedLibraryNode(f"Reduce with unsupported wcr {node.wcr!r} ({red})")
    inp = in_expr(state, node, None, sdfg)
    axis = None if node.axes is None else tuple(node.axes)
    # keepdims when the output keeps the reduced axis as a size-1 dimension (a numpy ``keepdims=True``
    # reduction, e.g. softmax's ``np.max(x, axis=-1, keepdims=True)``); detected by equal rank.
    in_desc = sdfg.arrays[next(iter(state.in_edges(node))).data.data]
    out_desc = sdfg.arrays[next(iter(state.out_edges(node))).data.data]
    keepdims = axis is not None and len(out_desc.shape) == len(in_desc.shape)
    kd = ", keepdims=True" if keepdims else ""
    return f"{out_lhs(state, node, None, sdfg)} = {func}.reduce({inp}, axis={axis}{kd})"


#: class name -> ``(node, state, sdfg) -> "lhs = rhs"`` (or a list of statements). Extend here.
LIBNODE_EMITTERS: Dict[str, Callable] = {
    "MatMul": emit_matmul,
    "Gemm": emit_gemm,
    "Gemv": emit_gemv,
    "Ger": emit_ger,
    "Axpy": emit_axpy,
    "BatchedMatMul": emit_batched_matmul,
    "Dot": emit_dot,
    "Einsum": emit_einsum,
    "TensorDot": emit_tensordot,
    "Transpose": emit_transpose,
    "TensorTranspose": emit_tensortranspose,
    "Solve": emit_solve,
    "Cholesky": emit_cholesky,
    "Inv": emit_inv,
    "Symm": emit_symm,
    "Syrk": emit_syrk,
    "Syr2k": emit_syr2k,
    "Potrf": emit_potrf,
    "FFT": emit_fft,
    "IFFT": emit_ifft,
    "Reduce": emit_reduce,
    "ArgReduce": emit_argreduce,
    "Scan": emit_scan,
    "IntegerSort": emit_integer_sort,
    "ScatterConflictCheck": emit_scatter_conflict_check,
}

#: Library nodes DELIBERATELY not emitted as numpy, each with the reason. Distinct from an *unregistered*
#: node (a genuine gap): these have no faithful single-process numpy form (sparse formats / FPGA streams /
#: arbitrary stencil code / LAPACK primitives that output pivots or packed in-place factors). The whole-
#: program lane must SPLIT AROUND one of these -- isolate it in its own state and externalize the
#: pure-compute states before/after -- rather than abandon the program (see the MPI policy below).
REFUSED_LIBRARY_NODES: Dict[str, str] = {
    "CSRMM": "sparse CSR matrix-matrix product; not emitted as dense numpy",
    "CSRMV": "sparse CSR matrix-vector product; not emitted as dense numpy",
    "Gearbox": "FPGA stream rate-changer; operands are Streams, not arrays",
    "Stencil": "arbitrary stencil code (relative offsets + boundary conditions); not a numpy op",
    "Getrf": "LAPACK LU factorization outputs pivots (ipiv) + packed in-place LU; no pure-numpy form",
    "Getri": "LAPACK inverse-from-LU consumes packed LU + pivots; no pure-numpy form",
    "Getrs": "LAPACK solve-from-LU consumes packed LU + pivots; no pure-numpy form",
}

#: DaCe library subpackages whose nodes are DISTRIBUTED communication (MPI point-to-point / collectives /
#: redistribution / parallel BLAS). None has a single-process numpy equivalent, and the offload policy is
#: the same for all: never offload the comm node itself -- place it in its own state and offload only the
#: pure-compute states before and after it. Matched by module so a name collision (an MPI ``Gather`` vs a
#: layout ``Gather``) cannot mis-route.
_COMM_MODULE_PREFIXES = ("dace.libraries.mpi", "dace.libraries.pblas")


def is_comm_node(node: nodes.LibraryNode) -> bool:
    """True if ``node`` is a distributed-communication library node (dace.libraries.mpi / pblas). Matched by
    MODULE, not class name, so an MPI ``Reduce`` / ``Gather`` (which collide by name with the registered
    standard ``Reduce`` and a layout ``Gather``) is correctly identified rather than mis-routed."""
    return type(node).__module__.startswith(_COMM_MODULE_PREFIXES)


def is_emittable_library_node(node: nodes.LibraryNode) -> bool:
    """True iff :func:`emit_library_node` will actually emit ``node`` -- it is not a communication node, not
    an explicitly refused class, and has a registered emitter. The single source of truth for "supported",
    used by both the emitter and the split-around-unsupported pass so a name collision cannot make one
    treat a node as supported while the other refuses it."""
    if is_comm_node(node):
        return False
    if type(node).__name__ in REFUSED_LIBRARY_NODES:
        return False
    return type(node).__name__ in LIBNODE_EMITTERS


def emit_library_node(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> List[str]:
    """Numpy statement(s) for a library node, or raise if it is a communication node / deliberately
    unsupported / unregistered. A single-statement emitter returns a ``str`` (wrapped here); a multi-output
    node (``ArgReduce``, ``Potrf``) returns a list of statements.

    Communication and refusal are checked BEFORE the name registry: an MPI ``Reduce`` shares its class name
    with the registered standard ``Reduce``, so a name-first lookup would mis-route it to the wrong
    emitter."""
    cls = type(node).__name__
    if is_comm_node(node):
        raise UnsupportedLibraryNode(f"{cls} is a distributed communication node (dace.libraries.mpi/pblas); "
                                     "not emittable as single-process numpy -- isolate it in its own state and "
                                     "externalize the compute before/after it")
    if cls in REFUSED_LIBRARY_NODES:
        raise UnsupportedLibraryNode(f"{cls}: {REFUSED_LIBRARY_NODES[cls]}")
    emitter = LIBNODE_EMITTERS.get(cls)
    if emitter is None:
        raise UnsupportedLibraryNode(f"no numpy emission for library node {cls}")
    result = emitter(node, state, sdfg)
    return result if isinstance(result, list) else [result]
