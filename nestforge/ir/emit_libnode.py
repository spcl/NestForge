# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Emit numpy statements for DaCe library nodes (BLAS / LinAlg / reductions / FFT).

Operand resolution (read/write expressions, scalar handling) plus a flat class-name -> emitter registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from collections.abc import Callable

import math

import sympy

import dace
from dace import symbolic
from dace.frontend.operations import detect_reduction_type
from dace.libraries.standard.nodes.scan import ScanOp
from dace.libraries.standard.nodes.scan import in_connector as scan_in
from dace.libraries.standard.nodes.scan import out_connector as scan_out
from dace.sdfg import nodes
from dace.sdfg.graph import MultiConnectorEdge

from nestforge.ir.dace_types import bounds, memlet_data, memlet_subset

if TYPE_CHECKING:
    from dace.libraries.blas.nodes.axpy import Axpy
    from dace.libraries.blas.nodes.batched_matmul import BatchedMatMul
    from dace.libraries.blas.nodes.dot import Dot
    from dace.libraries.blas.nodes.einsum import Einsum
    from dace.libraries.blas.nodes.gemm import Gemm
    from dace.libraries.blas.nodes.gemv import Gemv
    from dace.libraries.blas.nodes.ger import Ger
    from dace.libraries.blas.nodes.matmul import MatMul
    from dace.libraries.blas.nodes.symm import Symm
    from dace.libraries.blas.nodes.syr2k import Syr2k
    from dace.libraries.blas.nodes.syrk import Syrk
    from dace.libraries.fft.nodes.fft import FFT, IFFT
    from dace.libraries.lapack.nodes.potrf import Potrf
    from dace.libraries.linalg.nodes.cholesky import Cholesky
    from dace.libraries.linalg.nodes.inv import Inv
    from dace.libraries.linalg.nodes.solve import Solve
    from dace.libraries.linalg.nodes.tensordot import TensorDot
    from dace.libraries.linalg.nodes.transpose import Transpose
    from dace.libraries.linalg.nodes.ttranspose import TensorTranspose
    from dace.libraries.standard.nodes.arg_reduce import ArgReduce
    from dace.libraries.standard.nodes.reduce import Reduce
    from dace.libraries.standard.nodes.scan import Scan


class UnsupportedLibraryNode(Exception):
    """No numpy emission is registered for this library node class."""


def index_str(subset: dace.subsets.Range, keep_singleton: bool = False) -> str:
    """Format a subset as a numpy index/slice string (singleton range -> scalar unless ``keep_singleton``)."""
    parts: list[str] = []
    for beg, end, step in bounds(subset):
        if str(beg) == str(end):
            parts.append(
                f"{symbolic.symstr(beg)}:{symbolic.symstr(beg + 1)}" if keep_singleton else symbolic.symstr(beg)
            )
        elif str(step) == "1":
            parts.append(f"{symbolic.symstr(beg)}:{symbolic.symstr(end + 1)}")
        else:
            stop = exclusive_stop(end, step)
            if stop is None:
                raise UnsupportedLibraryNode(
                    f"subset range ({beg}, {end}, {step}) has a step of undecidable sign; no sound numpy slice stop"
                )
            # numpy reads a negative slice stop as an offset from the end; only a descending stop can go negative
            descending = sympy.sign(sympy.sympify(step)) == -1
            if symbolic.symstr(stop) == "-1":
                parts.append(f"{symbolic.symstr(beg)}::{symbolic.symstr(step)}")
            elif descending and sympy.sympify(stop >= 0) is not sympy.true:
                raise UnsupportedLibraryNode(
                    f"descending subset range ({beg}, {end}, {step}) has stop {symbolic.symstr(stop)}, which is not "
                    "provably >= 0; numpy would read a negative value as an offset from the end of the axis"
                )
            else:
                parts.append(f"{symbolic.symstr(beg)}:{symbolic.symstr(stop)}:{symbolic.symstr(step)}")
    return ", ".join(parts)


def literal(value: object) -> str:
    """``value`` as Python source; an infinite float spells ``np.inf``."""
    if isinstance(value, float) and math.isinf(value):
        return "-np.inf" if value < 0 else "np.inf"
    return repr(value)


def exclusive_stop(end: sympy.Expr, step: sympy.Expr) -> sympy.Expr | None:
    """One past the last element for an inclusive-end range; ``None`` when the step's sign is undecidable."""
    sign = sympy.sign(sympy.sympify(step))
    if sign not in (1, -1):
        return None
    return end + sign


def covers_whole(subset: dace.subsets.Range, desc: dace.data.Data) -> bool:
    """True if ``subset`` spans the entire array descriptor (so no slice suffix is needed)."""
    if len(subset.ranges) != len(desc.shape):
        return False
    for (beg, end, step), dim in zip(bounds(subset), desc.shape):
        if str(beg) != "0" or str(step) != "1" or symbolic.symstr(end + 1) != symbolic.symstr(dim):
            return False
    return True


def is_scalar(desc: dace.data.Data) -> bool:
    """A single-element data container (a DaCe ``Scalar`` or a size-1 array)."""
    return isinstance(desc, dace.data.Scalar) or desc.total_size == 1


def scalar_local(sdfg: dace.SDFG, name: str) -> bool:
    """A *scalar transient* -- emitted as a plain python variable (a C local ``double``), not a buffer."""
    desc = sdfg.arrays[name]
    return desc.transient and is_scalar(desc)


def scalar_elem(name: str, desc: dace.data.Data) -> str:
    """Index the sole element of a size-1 buffer, one ``0`` index per dimension (not ``name[0]``)."""
    rank = len(desc.shape)
    if rank <= 1:
        return f"{name}[0]"
    return f"{name}[{', '.join(['0'] * rank)}]"


def read_expr(sdfg: dace.SDFG, name: str, subset: dace.subsets.Range | None, keep_singleton: bool = False) -> str:
    """Read expression for ``name[subset]``: scalar-transient variable, whole array, or slice."""
    desc = sdfg.arrays[name]
    if scalar_local(sdfg, name):
        return name
    if is_scalar(desc):
        # bare name is the whole (1,) array; a scalar read must index its element instead.
        return scalar_elem(name, desc)
    if subset is None or covers_whole(subset, desc):
        return name
    return f"{name}[{index_str(subset, keep_singleton=keep_singleton)}]"


def write_lhs(sdfg: dace.SDFG, name: str, subset: dace.subsets.Range | None, keep_singleton: bool = False) -> str:
    """Write target for ``name[subset]``, in place (``name[:]`` / ``name[slice]``), not rebound. A size-1 buffer
    indexes its element: ``name[:] = scalar`` is valid numpy but the C translator mis-lowers it (double -> double*)."""
    expr = read_expr(sdfg, name, subset, keep_singleton)
    return f"{name}[:]" if expr == name and not scalar_local(sdfg, name) else expr


def operand_rank(sdfg: dace.SDFG, name: str, subset: dace.subsets.Range | None) -> int:
    """Rank of the operand as rendered by :func:`read_expr`/:func:`write_lhs`, not the buffer's rank."""
    desc = sdfg.arrays[name]
    if is_scalar(desc):
        return 0
    if subset is None or covers_whole(subset, desc):
        return len(desc.shape)
    return len(subset.ranges)


def memlet_expr(memlet: dace.Memlet, sdfg: dace.SDFG, keep_singleton: bool = False) -> str:
    """Read expression for a memlet's data; ``keep_singleton`` keeps length-1 dims, so a ``[N,1]`` column stays
    2-D for a matrix operation, where DaCe's vector operations squeeze them."""
    return read_expr(sdfg, memlet_data(memlet), memlet_subset(memlet), keep_singleton)


def data_edge(edges: list[MultiConnectorEdge], node: nodes.Node, kind: str) -> MultiConnectorEdge:
    """The first edge that carries data (skips empty happens-before/ordering edges, e.g. from StateFusion)."""
    for e in edges:
        if not e.data.is_empty():
            return e
    raise UnsupportedLibraryNode(
        f"{type(node).__name__} has no data-carrying {kind} edge (only empty ordering edges); not emittable as numpy"
    )


def conn_edge(edges: list[MultiConnectorEdge], node: nodes.Node, conn: str | None, output: bool) -> MultiConnectorEdge:
    """The edge on ``conn``, or the first data edge when ``conn`` is ``None``; raises when it is unwired."""
    kind = "output" if output else "input"
    if conn is None:
        return data_edge(edges, node, kind)
    edge = next((e for e in edges if (e.src_conn if output else e.dst_conn) == conn), None)
    if edge is None:
        raise UnsupportedLibraryNode(f"{type(node).__name__} has no {conn!r} {kind} connector; not emittable as numpy")
    return edge


def in_expr(
    state: dace.SDFGState, node: nodes.Node, conn: str | None, sdfg: dace.SDFG, keep_singleton: bool = False
) -> str:
    """Read expression for one input connector."""
    return memlet_expr(conn_edge(list(state.in_edges(node)), node, conn, output=False).data, sdfg, keep_singleton)


def out_expr(
    state: dace.SDFGState, node: nodes.Node, conn: str | None, sdfg: dace.SDFG, keep_singleton: bool = False
) -> str:
    """Read expression for the buffer an output connector writes (for a ``beta`` accumulate with no input)."""
    return memlet_expr(conn_edge(list(state.out_edges(node)), node, conn, output=True).data, sdfg, keep_singleton)


def out_lhs(
    state: dace.SDFGState, node: nodes.Node, conn: str | None, sdfg: dace.SDFG, keep_singleton: bool = False
) -> str:
    """Write target for one output connector."""
    edge = conn_edge(list(state.out_edges(node)), node, conn, output=True)
    if edge.data.wcr is not None:
        # no emitter applies an output WCR; an accumulate would silently become an overwrite.
        raise UnsupportedLibraryNode(
            f"{type(node).__name__} output into {edge.data.data} carries a reduction (WCR) that no library-node "
            "emitter applies; not emittable as numpy -- fall back to the DaCe variant"
        )
    return write_lhs(sdfg, memlet_data(edge.data), memlet_subset(edge.data), keep_singleton)


REDUCTION_FUNC = {
    dace.dtypes.ReductionType.Sum: "np.add",
    dace.dtypes.ReductionType.Product: "np.multiply",
    dace.dtypes.ReductionType.Max: "np.maximum",
    dace.dtypes.ReductionType.Min: "np.minimum",
    dace.dtypes.ReductionType.Logical_And: "np.logical_and",
    dace.dtypes.ReductionType.Logical_Or: "np.logical_or",
}


def is_one(v: Any) -> bool:
    """Value-aware ``v == 1`` (a sympy ``Float(1.0)`` compares unequal to the int ``1``)."""
    return symbolic.equal_valued(1, v)


def is_zero(v: Any) -> bool:
    """Value-aware ``v == 0`` (see :func:`is_one`)."""
    return symbolic.equal_valued(0, v)


def scaled(expr: str, coeff: Any) -> str:
    """``expr`` multiplied by ``coeff``, or ``expr`` unchanged when ``coeff`` is 1."""
    return expr if is_one(coeff) else f"{coeff} * ({expr})"


def transposed(expr: str, trans: bool) -> str:
    """``(expr).T`` when ``trans``, else ``expr`` (parenthesized so a slice operand transposes as a whole)."""
    return f"({expr}).T" if trans else expr


def matmul_update(node: MatMul | Gemm, state: dace.SDFGState, sdfg: dace.SDFG, read_c: Callable[..., str]) -> str:
    """``alpha * (opA(A) @ opB(B)) + beta * C``, ``C`` read through ``read_c`` (:func:`in_expr` or :func:`out_expr`)."""
    a = transposed(in_expr(state, node, "_a", sdfg, keep_singleton=True), node.transA)
    b = transposed(in_expr(state, node, "_b", sdfg, keep_singleton=True), node.transB)
    expr = scaled(f"{a} @ {b}", node.alpha)
    if not is_zero(node.beta):
        expr = f"{expr} + {node.beta} * {read_c(state, node, '_c', sdfg, keep_singleton=True)}"
    return f"{out_lhs(state, node, '_c', sdfg, keep_singleton=True)} = {expr}"


def emit_matmul(node: MatMul, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """Beta reads the output edge: ``MatMul`` has no ``_c`` input."""
    return matmul_update(node, state, sdfg, out_expr)


def emit_gemm(node: Gemm, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """BLAS GEMM, connectors ``_a``/``_b``/``_c``."""
    reject_runtime_scalars(node, state)
    return matmul_update(node, state, sdfg, in_expr)


def emit_gemv(node: Gemv, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """``alpha * (opA(A) @ x) + beta * y`` -- BLAS GEMV (matrix-vector), connectors ``_A``/``_x``/``_y``."""
    a = transposed(in_expr(state, node, "_A", sdfg), node.transA)
    x = in_expr(state, node, "_x", sdfg)
    expr = scaled(f"{a} @ {x}", node.alpha)
    if not is_zero(node.beta):
        expr = f"{expr} + {node.beta} * {in_expr(state, node, '_y', sdfg)}"
    return f"{out_lhs(state, node, '_y', sdfg)} = {expr}"


def emit_ger(node: Ger, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """``alpha * outer(x, y) + A`` -- BLAS GER rank-1 update; connectors ``_x``/``_y``/``_A`` -> ``_res``."""
    x = in_expr(state, node, "_x", sdfg)
    y = in_expr(state, node, "_y", sdfg)
    a = in_expr(state, node, "_A", sdfg, keep_singleton=True)
    res = out_lhs(state, node, "_res", sdfg, keep_singleton=True)
    return f"{res} = {scaled(f'np.outer({x}, {y})', node.alpha)} + {a}"


def emit_axpy(node: Axpy, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """``a * x + y`` -- BLAS AXPY; connectors ``_x``/``_y`` -> ``_res``."""
    x = in_expr(state, node, "_x", sdfg)
    y = in_expr(state, node, "_y", sdfg)
    return f"{out_lhs(state, node, '_res', sdfg)} = {scaled(x, node.a)} + {y}"


def emit_batched_matmul(node: BatchedMatMul, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """Batched ``A @ B`` over the trailing two dims; ``beta != 0`` is refused (no ``_c`` input to accumulate)."""
    if not is_zero(node.beta):
        raise UnsupportedLibraryNode(f"BatchedMatMul with beta={node.beta} has no _c input to accumulate")
    a = in_expr(state, node, "_a", sdfg, keep_singleton=True)
    b = in_expr(state, node, "_b", sdfg, keep_singleton=True)
    if node.transA:
        a = f"np.swapaxes({a}, -1, -2)"
    if node.transB:
        b = f"np.swapaxes({b}, -1, -2)"
    return f"{out_lhs(state, node, '_c', sdfg, keep_singleton=True)} = {scaled(f'{a} @ {b}', node.alpha)}"


def emit_einsum(node: Einsum, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """``np.einsum`` over connectors sorted by name, with ``alpha``/``beta`` folded in."""
    coeff = {"_alpha": str(node.alpha), "_beta": str(node.beta)}
    operands: list[tuple[str, str]] = []
    has_alpha = has_beta = False
    for e in state.in_edges(node):
        if e.data.is_empty():
            continue  # ordering edge, no operand
        if e.dst_conn in coeff:
            has_alpha = has_alpha or e.dst_conn == "_alpha"
            has_beta = has_beta or e.dst_conn == "_beta"
            coeff[e.dst_conn] = f"({coeff[e.dst_conn]}) * ({memlet_expr(e.data, sdfg)})"
        else:
            operands.append((e.dst_conn, memlet_expr(e.data, sdfg)))
    ordered = [expr for _, expr in sorted(operands)]
    expr = f"np.einsum('{node.einsum_str}', {', '.join(ordered)})"
    if has_alpha or not is_one(node.alpha):
        expr = f"({coeff['_alpha']}) * ({expr})"
    if has_beta or not is_zero(node.beta):
        expr = f"{expr} + ({coeff['_beta']}) * ({out_expr(state, node, None, sdfg)})"
    return f"{out_lhs(state, node, None, sdfg)} = {expr}"


def emit_tensordot(node: TensorDot, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """``np.tensordot`` with an optional output ``permutation`` transpose."""
    left = in_expr(state, node, "_left_tensor", sdfg, keep_singleton=True)
    right = in_expr(state, node, "_right_tensor", sdfg, keep_singleton=True)
    expr = f"np.tensordot({left}, {right}, axes=({list(node.left_axes)}, {list(node.right_axes)}))"
    if node.permutation is not None and list(node.permutation) != list(range(len(node.permutation))):
        expr = f"np.transpose({expr}, axes={list(node.permutation)})"
    return f"{out_lhs(state, node, '_out_tensor', sdfg, keep_singleton=True)} = {expr}"


def emit_inv(node: Inv, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """``np.linalg.inv(A)`` -- matrix inverse; connectors ``_ain`` -> ``_aout``."""
    ain = in_expr(state, node, "_ain", sdfg, keep_singleton=True)
    return f"{out_lhs(state, node, '_aout', sdfg, keep_singleton=True)} = np.linalg.inv({ain})"


def fft_statement(node: FFT | IFFT, state: dace.SDFGState, sdfg: dace.SDFG, func: str, norm: str) -> str:
    """``factor * func(x)`` over ``node.axes``, every axis when ``None``: listed axes index the unsqueezed operand,
    while over every axis a length-1 one transforms to itself and is squeezed. A 1-D transform keeps the plain
    ``fft``/``ifft`` call the translator lowers."""
    keep = node.axes is not None
    edge = conn_edge(list(state.in_edges(node)), node, "_inp", output=False)
    inp = memlet_expr(edge.data, sdfg, keep)
    rank = operand_rank(sdfg, memlet_data(edge.data), memlet_subset(edge.data))
    suffix = "" if norm == "backward" else f", norm={norm!r}"
    if node.axes is None and rank == 1:
        call = f"np.fft.{func}({inp}{suffix})"
    else:
        axes = None if node.axes is None else list(node.axes)
        call = f"np.fft.{func}n({inp}, axes={axes}{suffix})"
    return f"{out_lhs(state, node, '_out', sdfg, keep_singleton=keep)} = {scaled(call, node.factor)}"


def emit_fft(node: FFT, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """DaCe's forward DFT is unnormalized."""
    return fft_statement(node, state, sdfg, "fft", "backward")


def emit_ifft(node: IFFT, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """DaCe's inverse DFT has no built-in ``1/N``."""
    return fft_statement(node, state, sdfg, "ifft", "forward")


ARGREDUCE_FUNC = {"max": ("np.argmax", "np.max"), "min": ("np.argmin", "np.min")}


def emit_argreduce(node: ArgReduce, state: dace.SDFGState, sdfg: dace.SDFG) -> list[str]:
    """``np.argmax``/``np.argmin`` of the transformed input, plus its extreme value when ``_out_val`` is wired."""
    argfn, valfn = ARGREDUCE_FUNC[node.op]
    inp = in_expr(state, node, "_in", sdfg)
    if node.transform == "abs":
        inp = f"np.abs({inp})"
    elif node.transform:
        raise UnsupportedLibraryNode(f"ArgReduce transform {node.transform!r} is not emitted")
    lines = [f"{out_lhs(state, node, '_out_idx', sdfg)} = {argfn}({inp})"]
    if any(e.src_conn == "_out_val" for e in state.out_edges(node)):
        lines.append(f"{out_lhs(state, node, '_out_val', sdfg)} = {valfn}({inp})")
    return lines


SCAN_FUNC = {
    ScanOp.SUM: "np.cumsum",
    ScanOp.PRODUCT: "np.cumprod",
    ScanOp.MAX: "np.maximum.accumulate",
    ScanOp.MIN: "np.minimum.accumulate",
}


def emit_scan(node: Scan, state: dace.SDFGState, sdfg: dace.SDFG) -> list[str]:
    """Each chain of an inclusive unit-stride unseeded scan -> a numpy accumulate; anything else is refused."""
    func = SCAN_FUNC.get(node.op)
    if func is None:
        raise UnsupportedLibraryNode(f"Scan with unsupported op {node.op}")
    seeded = any(conn.startswith("_scan_init") for conn in node.in_connectors)
    if node.exclusive or str(node.stride) != "1" or seeded:
        raise UnsupportedLibraryNode("only an inclusive unit-stride unseeded Scan maps to a numpy accumulate")
    return [
        f"{out_lhs(state, node, scan_out(c), sdfg)} = {func}({in_expr(state, node, scan_in(c), sdfg)})"
        for c in range(node.chains)
    ]


def emit_integer_sort(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """Ascending 1-D sort -> ``np.sort``."""
    return f"{out_lhs(state, node, '_keys_out', sdfg)} = np.sort({in_expr(state, node, '_keys_in', sdfg)})"


def emit_scatter_conflict_check(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> list[str]:
    """Duplicate count over a 1-D integer index array, via last-writer-wins ownership (TAGCOUNT form)."""
    idx = in_expr(state, node, "_idx_in", sdfg)
    count = out_lhs(state, node, "_count_out", sdfg)
    tag = memlet_data(conn_edge(list(state.out_edges(node)), node, "_count_out", output=True).data)
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


def reject_runtime_scalars(node: nodes.LibraryNode, state: dace.SDFGState) -> None:
    """Refuse a BLAS node with a runtime ``_alpha``/``_beta`` connector (only compile-time values are folded)."""
    dst_conns = {e.dst_conn for e in state.in_edges(node)}
    if "_alpha" in dst_conns or "_beta" in dst_conns:
        raise UnsupportedLibraryNode(
            f"{type(node).__name__} has a runtime _alpha/_beta scalar connector; "
            "only compile-time alpha/beta are emitted -- fall back to the DaCe variant"
        )


def triangle_update(node: Syrk | Syr2k, state: dace.SDFGState, sdfg: dace.SDFG, prod: str) -> str:
    """``C = alpha * prod + beta * C`` on the ``uplo`` triangle of the block ``_c`` writes; the other keeps ``C``."""
    c = out_expr(state, node, "_c", sdfg, keep_singleton=True)
    rhs = scaled(f"({prod})", node.alpha)
    if not is_zero(node.beta):
        rhs = f"{rhs} + {node.beta} * {c}"
    # the written triangle, and the preserved one with the offset that excludes the diagonal
    write, keep, off = ("np.tril", "np.triu", 1) if node.uplo == "L" else ("np.triu", "np.tril", -1)
    return f"{out_lhs(state, node, '_c', sdfg, keep_singleton=True)} = {write}({rhs}) + {keep}({c}, {off})"


def emit_syrk(node: Syrk, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """BLAS SYRK, updating only the ``uplo`` triangle of ``C``; the opposite triangle keeps its prior value."""
    reject_runtime_scalars(node, state)
    a = in_expr(state, node, "_a", sdfg, keep_singleton=True)
    return triangle_update(node, state, sdfg, f"{a}.T @ {a}" if node.trans == "T" else f"{a} @ {a}.T")


def emit_syr2k(node: Syr2k, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """BLAS SYR2K, updating only the ``uplo`` triangle of ``C`` (``A``/``B`` read in full)."""
    reject_runtime_scalars(node, state)
    a = in_expr(state, node, "_a", sdfg, keep_singleton=True)
    b = in_expr(state, node, "_b", sdfg, keep_singleton=True)
    prod = f"{a}.T @ {b} + {b}.T @ {a}" if node.trans == "T" else f"{a} @ {b}.T + {b} @ {a}.T"
    return triangle_update(node, state, sdfg, prod)


def emit_symm(node: Symm, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """BLAS SYMM; ``A`` is symmetric and stored as only its ``uplo`` triangle, reconstructed to full first."""
    reject_runtime_scalars(node, state)
    a = in_expr(state, node, "_a", sdfg, keep_singleton=True)
    b = in_expr(state, node, "_b", sdfg, keep_singleton=True)
    asym = f"(np.tril({a}) + np.tril({a}, -1).T)" if node.uplo == "L" else f"(np.triu({a}) + np.triu({a}, 1).T)"
    mat = f"{asym} @ {b}" if node.side == "L" else f"{b} @ {asym}"
    rhs = scaled(mat, node.alpha)
    if not is_zero(node.beta):
        rhs = f"{rhs} + {node.beta} * {in_expr(state, node, '_c', sdfg, keep_singleton=True)}"
    return f"{out_lhs(state, node, '_c', sdfg, keep_singleton=True)} = {rhs}"


def cholesky_factor(a: str, lower: bool) -> str:
    """numpy returns the lower factor L (A = L @ L.conj().T); the upper factor is L.conj().T."""
    return f"np.linalg.cholesky({a})" if lower else f"(np.linalg.cholesky({a})).conj().T"


def emit_potrf(node: Potrf, state: dace.SDFGState, sdfg: dace.SDFG) -> list[str]:
    """LAPACK POTRF; ``_res`` always reports success."""
    expr = cholesky_factor(in_expr(state, node, "_xin", sdfg, keep_singleton=True), node.lower)
    lines = [f"{out_lhs(state, node, '_xout', sdfg, keep_singleton=True)} = {expr}"]
    if any(e.src_conn == "_res" for e in state.out_edges(node)):
        lines.append(f"{out_lhs(state, node, '_res', sdfg, keep_singleton=True)} = np.array(0, np.int32)")
    return lines


def emit_dot(node: Dot, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    x = in_expr(state, node, "_x", sdfg)
    y = in_expr(state, node, "_y", sdfg)
    call = "np.vdot" if node.conjugate else "np.dot"  # vdot conjugates its first operand
    return f"{out_lhs(state, node, '_result', sdfg)} = {call}({x}, {y})"


def emit_transpose(node: Transpose, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    inp = in_expr(state, node, "_inp", sdfg, keep_singleton=True)
    return f"{out_lhs(state, node, '_out', sdfg, keep_singleton=True)} = np.transpose({inp})"


def emit_solve(node: Solve, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    ain = in_expr(state, node, "_ain", sdfg, keep_singleton=True)
    bin_ = in_expr(state, node, "_bin", sdfg, keep_singleton=True)
    return f"{out_lhs(state, node, '_bout', sdfg, keep_singleton=True)} = np.linalg.solve({ain}, {bin_})"


def emit_cholesky(node: Cholesky, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    expr = cholesky_factor(in_expr(state, node, "_a", sdfg, keep_singleton=True), node.lower)
    return f"{out_lhs(state, node, '_b', sdfg, keep_singleton=True)} = {expr}"


def emit_tensortranspose(node: TensorTranspose, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    inp = in_expr(state, node, "_inp_tensor", sdfg, keep_singleton=True)
    if not is_zero(node.beta):
        raise UnsupportedLibraryNode(f"TensorTranspose with beta={node.beta} (accumulate) is not emitted")
    transposed_input = scaled(f"np.transpose({inp}, axes={list(node.axes)})", node.alpha)
    return f"{out_lhs(state, node, '_out_tensor', sdfg, keep_singleton=True)} = {transposed_input}"


def emit_reduce(node: Reduce, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    red = detect_reduction_type(node.wcr)
    func = None if red is None else REDUCTION_FUNC.get(red)
    if func is None:
        raise UnsupportedLibraryNode(f"Reduce with unsupported wcr {node.wcr!r} ({red})")
    inp = in_expr(state, node, None, sdfg, keep_singleton=True)
    axis = None if node.axes is None else tuple(node.axes)
    # keepdims true iff the rendered output keeps the same rank as the rendered input (judged on the
    # operands, via operand_rank, since two buffers can share a descriptor rank while their renders differ).
    in_memlet = data_edge(list(state.in_edges(node)), node, "input").data
    out_memlet = data_edge(list(state.out_edges(node)), node, "output").data
    in_rank = operand_rank(sdfg, memlet_data(in_memlet), memlet_subset(in_memlet))
    out_rank = operand_rank(sdfg, memlet_data(out_memlet), memlet_subset(out_memlet))
    keepdims = axis is not None and out_rank == in_rank
    kd = ", keepdims=True" if keepdims else ""
    reduced = f"{func}.reduce({inp}, axis={axis}{kd})"
    # without an identity DaCe accumulates into the output's current value
    seed = out_expr(state, node, None, sdfg, keep_singleton=True) if node.identity is None else literal(node.identity)
    return f"{out_lhs(state, node, None, sdfg, keep_singleton=True)} = {func}({seed}, {reduced})"


#: class name -> emitter ``(node, state, sdfg) -> "lhs = rhs"`` (or a list of statements).
LIBNODE_EMITTERS: dict[str, Callable[[Any, dace.SDFGState, dace.SDFG], str | list[str]]] = {
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

#: Library nodes deliberately not emitted as numpy, each mapped to the refusal reason.
REFUSED_LIBRARY_NODES: dict[str, str] = {
    "CSRMM": "sparse CSR matrix-matrix product; not emitted as dense numpy",
    "CSRMV": "sparse CSR matrix-vector product; not emitted as dense numpy",
    "Gearbox": "FPGA stream rate-changer; operands are Streams, not arrays",
    "Stencil": "arbitrary stencil code (relative offsets + boundary conditions); not a numpy op",
    "Getrf": "LAPACK LU factorization outputs pivots (ipiv) + packed in-place LU; no pure-numpy form",
    "Getri": "LAPACK inverse-from-LU consumes packed LU + pivots; no pure-numpy form",
    "Getrs": "LAPACK solve-from-LU consumes packed LU + pivots; no pure-numpy form",
}

# distributed-communication subpackages; matched by module, not class name, so no name collides.
COMM_MODULE_PREFIXES = ("dace.libraries.mpi", "dace.libraries.pblas")


def is_comm_node(node: nodes.LibraryNode) -> bool:
    """True if ``node`` is a distributed-communication library node (dace.libraries.mpi / pblas)."""
    return type(node).__module__.startswith(COMM_MODULE_PREFIXES)


def emit_library_node(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> list[str]:
    """Numpy statement(s) for a library node; raises if it is a communication / refused / unregistered node."""
    cls = type(node).__name__
    if is_comm_node(node):  # checked before the name registry: MPI Reduce collides by name with ours
        raise UnsupportedLibraryNode(
            f"{cls} is a distributed communication node (dace.libraries.mpi/pblas); "
            "not emittable as single-process numpy -- isolate it in its own state and "
            "externalize the compute before/after it"
        )
    if cls in REFUSED_LIBRARY_NODES:
        raise UnsupportedLibraryNode(f"{cls}: {REFUSED_LIBRARY_NODES[cls]}")
    emitter = LIBNODE_EMITTERS.get(cls)
    if emitter is None:
        raise UnsupportedLibraryNode(f"no numpy emission for library node {cls}")
    result = emitter(node, state, sdfg)
    return result if isinstance(result, list) else [result]
