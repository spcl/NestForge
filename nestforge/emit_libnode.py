"""Emit numpy operations for DaCe library nodes (BLAS / LinAlg / reductions / FFT).

A DaCe SDFG expresses dense-linear-algebra and spectral compute as *library nodes* (``MatMul``,
``Dot``, ``Reduce``, ``Transpose``, ``Solve``, FFT, ...) rather than explicit loops -- the classic
polybench/npbench kernels simplify almost entirely into these. Re-emitting a library node as the
equivalent numpy op (``A @ B``, ``np.add.reduce``, ``np.fft.fft`` ...) keeps the reference dense and
lets OptArena's translators recover an idiomatic kernel; only when a nest has *no* library node do
we fall back to explicit ``for`` loops (see :mod:`nestforge.emit_numpy`).

This module owns operand resolution (memlet -> ``name`` or ``name[slice]``) so the registry stays a
flat class-name -> statement-builder table that is trivial to extend with new library nodes.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import dace
from dace import symbolic
from dace.frontend.operations import detect_reduction_type
from dace.sdfg import nodes


class UnsupportedLibraryNode(Exception):
    """No numpy emission is registered for this library node class."""


# ----- operand resolution -------------------------------------------------------------------------


def index_str(subset: dace.subsets.Range) -> str:
    """Format a subset as a numpy index/slice string (end-inclusive DaCe range -> ``beg:end+1``)."""
    parts = []
    for (beg, end, step) in subset.ranges:
        if str(beg) == str(end):
            parts.append(symbolic.symstr(beg))
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


def read_expr(sdfg: dace.SDFG, name: str, subset: Optional[dace.subsets.Range]) -> str:
    """Read expression for ``name[subset]``: a scalar-transient variable, a whole array, or a slice.

    Each array's data name doubles as the python variable, so no connector renaming is needed:
    inputs/outputs/scratch are pre-allocated buffer parameters, scalar transients are locals. A
    ``None`` subset means the whole array.
    """
    if scalar_local(sdfg, name):
        return name
    desc = sdfg.arrays[name]
    if subset is None or covers_whole(subset, desc):
        return name
    return f"{name}[{index_str(subset)}]"


def write_lhs(sdfg: dace.SDFG, name: str, subset: Optional[dace.subsets.Range]) -> str:
    """Write target for ``name[subset]``. Arrays are written *in place* (``name[:]`` / ``name[slice]``)
    so a pre-allocated buffer parameter is filled rather than rebound to a fresh array; a scalar
    transient is a plain assignment. A ``None`` subset means the whole array."""
    if scalar_local(sdfg, name):
        return name
    desc = sdfg.arrays[name]
    if subset is None or covers_whole(subset, desc):
        return f"{name}[:]"
    return f"{name}[{index_str(subset)}]"


def memlet_expr(memlet: dace.Memlet, sdfg: dace.SDFG) -> str:
    """Read expression for a memlet's data (see :func:`read_expr`)."""
    return read_expr(sdfg, memlet.data, memlet.subset)


def memlet_lhs(memlet: dace.Memlet, sdfg: dace.SDFG) -> str:
    """Write target for a memlet's data (see :func:`write_lhs`)."""
    return write_lhs(sdfg, memlet.data, memlet.subset)


def _in_expr(state: dace.SDFGState, node: nodes.Node, conn: Optional[str], sdfg: dace.SDFG) -> str:
    edges = list(state.in_edges(node))
    edge = edges[0] if conn is None else next(e for e in edges if e.dst_conn == conn)
    return memlet_expr(edge.data, sdfg)


def _out_lhs(state: dace.SDFGState, node: nodes.Node, conn: Optional[str], sdfg: dace.SDFG) -> str:
    edges = list(state.out_edges(node))
    edge = edges[0] if conn is None else next(e for e in edges if e.src_conn == conn)
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


def _emit_matmul(node, state, sdfg) -> str:
    a = _in_expr(state, node, "_a", sdfg)
    b = _in_expr(state, node, "_b", sdfg)
    c_read = _in_expr(state, node, "_c", sdfg) if str(node.beta) not in ("0", "0.0") else None
    expr = f"{a} @ {b}"
    if str(node.alpha) != "1":
        expr = f"{node.alpha} * ({expr})"
    if c_read is not None:
        expr = f"{expr} + {node.beta} * {c_read}"
    return f"{_out_lhs(state, node, '_c', sdfg)} = {expr}"


def _emit_dot(node, state, sdfg) -> str:
    x = _in_expr(state, node, "_x", sdfg)
    y = _in_expr(state, node, "_y", sdfg)
    return f"{_out_lhs(state, node, '_result', sdfg)} = np.dot({x}, {y})"


def _emit_transpose(node, state, sdfg) -> str:
    inp = _in_expr(state, node, "_inp", sdfg)
    return f"{_out_lhs(state, node, '_out', sdfg)} = np.transpose({inp})"


def _emit_solve(node, state, sdfg) -> str:
    ain = _in_expr(state, node, "_ain", sdfg)
    bin_ = _in_expr(state, node, "_bin", sdfg)
    return f"{_out_lhs(state, node, '_bout', sdfg)} = np.linalg.solve({ain}, {bin_})"


def _emit_cholesky(node, state, sdfg) -> str:
    a = _in_expr(state, node, "_a", sdfg)
    # numpy returns the lower factor L (A = L @ L.conj().T); the upper factor is L.conj().T.
    expr = f"np.linalg.cholesky({a})"
    if not node.lower:
        expr = f"({expr}).conj().T"
    return f"{_out_lhs(state, node, '_b', sdfg)} = {expr}"


def _emit_tensortranspose(node, state, sdfg) -> str:
    inp = _in_expr(state, node, "_inp_tensor", sdfg)
    if str(node.beta) not in ("0", "0.0"):
        raise UnsupportedLibraryNode(f"TensorTranspose with beta={node.beta} (accumulate) is not emitted")
    expr = f"np.transpose({inp}, axes={list(node.axes)})"
    if str(node.alpha) != "1":
        expr = f"{node.alpha} * ({expr})"
    return f"{_out_lhs(state, node, '_out_tensor', sdfg)} = {expr}"


def _emit_reduce(node, state, sdfg) -> str:
    red = detect_reduction_type(node.wcr)
    func = _REDUCTION_FUNC.get(red)
    if func is None:
        raise UnsupportedLibraryNode(f"Reduce with unsupported wcr {node.wcr!r} ({red})")
    inp = _in_expr(state, node, None, sdfg)
    axis = None if node.axes is None else tuple(node.axes)
    # keepdims when the output keeps the reduced axis as a size-1 dimension (a numpy ``keepdims=True``
    # reduction, e.g. softmax's ``np.max(x, axis=-1, keepdims=True)``); detected by equal rank.
    in_desc = sdfg.arrays[next(iter(state.in_edges(node))).data.data]
    out_desc = sdfg.arrays[next(iter(state.out_edges(node))).data.data]
    keepdims = axis is not None and len(out_desc.shape) == len(in_desc.shape)
    kd = ", keepdims=True" if keepdims else ""
    return f"{_out_lhs(state, node, None, sdfg)} = {func}.reduce({inp}, axis={axis}{kd})"


#: class name -> ``(node, state, sdfg) -> "lhs = rhs"``. Extend with new library nodes here.
LIBNODE_EMITTERS: Dict[str, Callable] = {
    "MatMul": _emit_matmul,
    "Dot": _emit_dot,
    "Transpose": _emit_transpose,
    "TensorTranspose": _emit_tensortranspose,
    "Solve": _emit_solve,
    "Cholesky": _emit_cholesky,
    "Reduce": _emit_reduce,
}


def emit_library_node(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """Return the numpy statement for a library node, or raise if none is registered."""
    emitter = LIBNODE_EMITTERS.get(type(node).__name__)
    if emitter is None:
        raise UnsupportedLibraryNode(f"no numpy emission for library node {type(node).__name__}")
    return emitter(node, state, sdfg)
