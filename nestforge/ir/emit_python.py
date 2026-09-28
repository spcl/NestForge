# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Emit an SDFG or an extracted nest as one plain Python function, the kernel's correctness oracle.

Tasklets and interstate code are already Python. :func:`lower` prepares a copy: library nodes expand to their pure
implementations, nested SDFGs bind their connectors to the outer arrays, DaCe's ``InlineTaskletConnectors`` turns
connectors into direct array accesses, and :func:`unique_connectors` names the rest uniquely, since a Python function
has one flat scope. Every array is a caller-allocated buffer; a transient scalar is a local variable."""

from __future__ import annotations

import ast
import atexit
import copy
import functools
import hashlib
import importlib.util
import re
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import sympy

import dace
from dace import data as dt
from dace import dtypes, subsets, symbolic
from dace.frontend.operations import detect_reduction_type
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import (
    BreakBlock,
    ConditionalBlock,
    ContinueBlock,
    ControlFlowBlock,
    ControlFlowRegion,
    LoopRegion,
    ReturnBlock,
    SDFGState,
)
from dace.sdfg.utils import dfs_topological_sort, inline_sdfgs
from dace.transformation.interstate.expand_nested_sdfg_inputs import ExpandNestedSDFGInputs
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.inline_tasklet_connectors import InlineTaskletConnectors

from nestforge.ir.dace_types import strings
from nestforge.ir.extract import Boundary


class UnsupportedNest(Exception):
    """The program uses a construct the Python emitter does not handle."""

    __slots__ = ()


#: Math intrinsic in tasklet code -> its NumPy spelling.
NP_FUNCTIONS = {
    **{fn: fn for fn in ("sqrt", "cbrt", "exp", "exp2", "expm1", "log", "log2", "log10", "log1p", "sin", "cos")},
    **{fn: fn for fn in ("tan", "sinh", "cosh", "tanh", "floor", "ceil", "sign", "power", "abs", "arctan2")},
    **{"asin": "arcsin", "acos": "arccos", "atan": "arctan", "atan2": "arctan2", "fabs": "abs", "pow": "power"},
    **{"re": "real", "im": "imag"},  # the frontend spells z.real and z.imag as SymPy's re and im
    **{"CMod": "fmod", "FtnMod": "fmod", "bitwise_invert": "invert"},  # C's truncating modulo is fmod
}
NP_DTYPES = {name: name for name in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")}
NP_DTYPES |= {"float16": "float16", "float32": "float32", "float64": "float64", "bool": "bool_"}
NP_DTYPES |= {"complex64": "complex64", "complex128": "complex128", "double": "float64", "float": "float32"}
NP_DTYPES |= {f"{name}_t": name for name in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")}

#: SymPy function heads DaCe prints in symbolic expressions -> a Python operator.
BINARY_OPS = {"int_floor": ast.FloorDiv, "__int_floor": ast.FloorDiv, "int_floor_ni": ast.FloorDiv, "ipow": ast.Pow}
BINARY_OPS |= {"py_floor": ast.FloorDiv}
BINARY_OPS |= {"Mod": ast.Mod, "PyMod": ast.Mod, "py_mod": ast.Mod, "left_shift": ast.LShift, "right_shift": ast.RShift}
BINARY_OPS |= {"bitwise_and": ast.BitAnd, "bitwise_or": ast.BitOr, "bitwise_xor": ast.BitXor}
BUILTINS = {"Min": "min", "Max": "max", "Abs": "abs"}

#: Reduction -> how a write-conflict resolution combines accumulator and value.
COMBINE = {
    dtypes.ReductionType.Sum: "{0} + {1}",
    dtypes.ReductionType.Product: "{0} * {1}",
    dtypes.ReductionType.Min: "np.minimum({0}, {1})",
    dtypes.ReductionType.Max: "np.maximum({0}, {1})",
}


#: Marks the loop of a map, whose iterations are independent; a sequential loop carries no mark.
PARALLEL = "# parallel"
#: How often library nodes are expanded again: an expansion may produce further library nodes.
EXPANSION_ROUNDS = 16
#: Hex digits of the source hash that name an emitted module's file.
HASH_DIGITS = 16
#: The one-trip loop a nested SDFG's early return breaks out of.
ONCE = "nested_once"


def numpy_attr(name: str) -> ast.Attribute:
    return ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr=name, ctx=ast.Load())


@dataclass(frozen=True, slots=True)
class Names:
    """How data names of one SDFG are spelled: a transient scalar is a local, a scalar argument a 1-element buffer."""

    locals: frozenset[str]
    scalars: frozenset[str]


def names_of(sdfg: dace.SDFG) -> Names:
    single = {name for name, desc in sdfg.arrays.items() if isinstance(desc, dt.Scalar) or desc.total_size == 1}
    local = frozenset(name for name in single if sdfg.arrays[name].transient)
    return Names(local, frozenset(n for n in single - local if isinstance(sdfg.arrays[n], dt.Scalar)))


class Normalize(ast.NodeTransformer):
    """Rewrite DaCe spellings to NumPy and Python, and data names to how the emitted function holds them."""

    __slots__ = ("names",)

    def __init__(self, names: Names) -> None:
        self.names = names

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        base = node.value
        if isinstance(base, ast.Name) and base.id in self.names.locals:
            return ast.copy_location(ast.Name(id=base.id, ctx=node.ctx), node)
        if isinstance(base, ast.Name):  # the base is a buffer, never a scalar to index
            node.slice = self.visit(node.slice)
            return node
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.names.scalars and isinstance(node.ctx, ast.Load):
            return ast.copy_location(ast.Subscript(value=node, slice=ast.Constant(0), ctx=node.ctx), node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        owner = ast.unparse(node.value)
        if owner in ("math", "dace.math") and node.attr in NP_FUNCTIONS:
            return ast.copy_location(numpy_attr(NP_FUNCTIONS[node.attr]), node)
        if owner == "dace" and node.attr in NP_DTYPES:
            return ast.copy_location(numpy_attr(NP_DTYPES[node.attr]), node)
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        name = node.func.id if isinstance(node.func, ast.Name) else ""
        if name in BINARY_OPS and len(node.args) == 2:
            return ast.copy_location(ast.BinOp(node.args[0], BINARY_OPS[name](), node.args[1]), node)
        if name == "int_ceil" and len(node.args) == 2:  # ceil(a / b) == -((-a) // b)
            neg = ast.BinOp(ast.UnaryOp(ast.USub(), node.args[0]), ast.FloorDiv(), node.args[1])
            return ast.copy_location(ast.UnaryOp(ast.USub(), neg), node)
        if name in BUILTINS:
            node.func = ast.Name(id=BUILTINS[name], ctx=ast.Load())
        elif name in NP_FUNCTIONS:
            node.func = numpy_attr(NP_FUNCTIONS[name])
        elif name in NP_DTYPES:
            node.func = numpy_attr(NP_DTYPES[name])
        return node


def python(code: str, names: Names) -> list[str]:
    """``code`` (statements or an expression) as emitted Python lines."""
    try:
        tree = ast.parse(code.strip())
    except SyntaxError as exc:
        raise UnsupportedNest(f"not Python: {code!r}") from exc
    return ast.unparse(ast.fix_missing_locations(Normalize(names).visit(tree))).splitlines()


def expr(value: object, names: Names) -> str:
    """A symbolic value or a code string as one Python expression."""
    text = value if isinstance(value, str) else symbolic.symstr(value, cpp_mode=False)
    return " ".join(python(text, names))


# lowering, on a copy


def placed[N: nodes.Node](sdfg: dace.SDFG, kind: type[N]) -> list[tuple[N, SDFGState]]:
    """Every ``kind`` node of ``sdfg`` and its nested SDFGs, with its state."""
    return [(n, s) for n, s in sdfg.all_nodes_recursive() if isinstance(n, kind) and isinstance(s, SDFGState)]


def memlet_parts(memlet: dace.Memlet) -> tuple[str, subsets.Subset | None, subsets.Subset | None]:
    """A data memlet's container, subset and other subset; DaCe annotates the subsets as possibly strings."""
    data, subset, other = memlet.data, memlet.subset, memlet.other_subset
    assert isinstance(data, str), f"memlet {memlet} moves no data"
    assert not isinstance(subset, str) and not isinstance(other, str), f"memlet {memlet} has an unparsed subset"
    return data, subset, other


def expand_to_pure(sdfg: dace.SDFG) -> None:
    """Expand every library node, choosing its ``pure`` implementation where it has one."""
    for attempt in range(EXPANSION_ROUNDS):
        libraries = [node for node, state in placed(sdfg, nodes.LibraryNode)]
        if not libraries:
            return
        for node in libraries:
            # dace.library.node erases the class type, hiding its implementations from pyright
            if "pure" in node.implementations:  # pyright: ignore[reportAttributeAccessIssue]
                node.implementation = "pure"
        sdfg.expand_library_nodes(recursive=False)
    raise UnsupportedNest(f"library nodes of {sdfg.name} still expand to library nodes after {attempt + 1} rounds")


def moved(subset: subsets.Subset | None) -> bool:
    """Whether a connector subset starts off the origin or strides along an axis it keeps."""
    ranges = subset.ranges if isinstance(subset, subsets.Range) else []
    return any(begin != end and (begin != 0 or step != 1) for begin, end, step in ranges)


def symbolic_reads(sdfg: dace.SDFG) -> set[str]:
    """Names ``sdfg`` itself reads outside dataflow: interstate edges, loop and branch heads, memlet subsets."""
    reads = {name for e in sdfg.all_interstate_edges() for name in e.data.read_symbols()}
    for block in sdfg.all_control_flow_blocks():
        if isinstance(block, (LoopRegion, ConditionalBlock)):
            reads |= block.used_symbols(all_symbols=True, with_contents=False)
    for state in sdfg.all_states():
        for edge in state.edges():
            for subset in (edge.data.subset, edge.data.other_subset):
                reads |= {str(s) for s in subset.free_symbols} if isinstance(subset, subsets.Subset) else set()
    return reads


def refuse_moved_symbolic_reads(sdfg: dace.SDFG, outer_moved: frozenset[str] = frozenset()) -> None:
    """``ExpandNestedSDFGInputs`` offsets and scales a widened connector's memlets, but not its indices in
    interstate edges, loop and branch heads or other memlets' subsets; refuse a moved connector read there."""
    for node, state in placed(sdfg, nodes.NestedSDFG):
        if state.sdfg is not sdfg:
            continue
        edges = [*((e, e.dst_conn) for e in state.in_edges(node)), *((e, e.src_conn) for e in state.out_edges(node))]
        inner_moved = frozenset(c for e, c in edges if c and (e.data.data in outer_moved or moved(e.data.subset)))
        misread = sorted(inner_moved & symbolic_reads(node.sdfg))
        if misread:
            raise UnsupportedNest(f"nested SDFG {node.label} reads the offset or strided connectors {misread} by index")
        refuse_moved_symbolic_reads(node.sdfg, inner_moved)


def bind_nested_data(sdfg: dace.SDFG) -> None:
    """Rename each nested SDFG's connector arrays to the outer arrays they bind, with the outer descriptors; the
    connectors cover whole arrays after ``ExpandNestedSDFGInputs``."""
    for node, state in placed(sdfg, nodes.NestedSDFG):
        outer, inner = state.sdfg, node.sdfg
        bound = {e.dst_conn: memlet_parts(e.data)[0] for e in state.in_edges(node) if e.dst_conn and e.data.data}
        bound |= {e.src_conn: memlet_parts(e.data)[0] for e in state.out_edges(node) if e.src_conn and e.data.data}
        for conn, data in bound.items():
            inner_desc, outer_desc = inner.arrays[conn], outer.arrays[data]
            same = [str(d) for d in inner_desc.shape] == [str(d) for d in outer_desc.shape]
            if not same and not (inner_desc.total_size == 1 and outer_desc.total_size == 1):
                raise UnsupportedNest(
                    f"nested connector {conn!r} is {inner_desc.shape}, but {data!r} is {outer_desc.shape}"
                )
            if data in inner.arrays and data not in bound:
                raise UnsupportedNest(f"nested SDFG {inner.name} has a private {data!r} that shadows the outer array")
        inner.replace_dict({conn: data for conn, data in bound.items() if conn != data})
        for data in bound.values():
            inner.arrays[data] = copy.deepcopy(outer.arrays[data])


def program_names(sdfg: dace.SDFG) -> set[str]:
    """Every data, symbol, map parameter and loop variable name of ``sdfg`` and its nested SDFGs."""
    taken: set[str] = set()
    for sd in sdfg.all_sdfgs_recursive():
        taken |= set(sd.arrays) | set(sd.symbols)
    for entry, state in placed(sdfg, nodes.MapEntry):
        taken |= set(strings(entry.map.params))
    for region in sdfg.all_control_flow_regions(recursive=True):
        if isinstance(region, LoopRegion) and region.loop_variable:
            taken.add(region.loop_variable)
    return taken


def own_bindings(sdfg: dace.SDFG) -> dict[str, None]:
    """The symbols ``sdfg`` itself binds, nested SDFGs excluded: interstate assignment targets and loop variables."""
    bound = dict.fromkeys(k for e in sdfg.all_interstate_edges() for k in e.data.assignments)
    for region in sdfg.all_control_flow_regions():
        if isinstance(region, LoopRegion) and region.loop_variable:
            bound[region.loop_variable] = None
    return bound


def fresh_name(base: str, taken: set[str]) -> str:
    """``base_<k>`` for the first ``k`` not in ``taken``, which it then joins."""
    name = next(f"{base}_{k}" for k in range(len(taken) + 1) if f"{base}_{k}" not in taken)
    taken.add(name)
    return name


def own_names(sdfg: dace.SDFG) -> set[str]:
    """The data, symbols, bindings and map parameters of ``sdfg`` itself, nested SDFGs excluded."""
    names = set(sdfg.arrays) | set(sdfg.symbols) | own_bindings(sdfg).keys()
    return names | {p for e, s in placed(sdfg, nodes.MapEntry) if s.sdfg is sdfg for p in strings(e.map.params)}


def unique_nested_bindings(sdfg: dace.SDFG) -> None:
    """Rename a symbol a nested SDFG binds itself when an enclosing SDFG uses the name: in one flat Python
    function the nested binding would overwrite the outer value."""
    taken = program_names(sdfg)
    for sd in list(sdfg.all_sdfgs_recursive()):  # parents first, so a renamed parent is what a child sees
        node = sd.parent_nsdfg_node
        if node is None:
            continue
        enclosing: set[str] = set()
        parent = sd.parent_sdfg
        while parent is not None:
            enclosing |= own_names(parent)
            parent = parent.parent_sdfg
        clashes = [n for n in own_bindings(sd) if n in enclosing and n not in node.symbol_mapping]
        if clashes:
            sd.replace_dict({name: fresh_name(name, taken) for name in clashes})


def unique_connectors(sdfg: dace.SDFG) -> None:
    """Rename every connector a tasklet body still reads or writes to a name unique across the program."""
    taken = program_names(sdfg)
    for node, state in placed(sdfg, nodes.Tasklet):
        if node.code.language != dtypes.Language.Python:
            continue
        tree = ast.parse(node.code.as_string)
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        renames: dict[str, str] = {}
        for conn in dict.fromkeys([*node.in_connectors, *node.out_connectors]):
            if conn in used:
                renames[conn] = fresh_name(conn.strip("_") or "c", taken)
        if not renames:
            continue
        for name_node in ast.walk(tree):
            if isinstance(name_node, ast.Name) and name_node.id in renames:
                name_node.id = renames[name_node.id]
        node.code = CodeBlock(ast.unparse(tree), node.code.language)
        for old, new in renames.items():
            rename_connector(state, node, old, new)


def rename_connector(state: SDFGState, node: nodes.Tasklet, old: str, new: str) -> None:
    if old in node.in_connectors:
        node.add_in_connector(new, node.in_connectors[old])
        node.remove_in_connector(old)
    if old in node.out_connectors:
        node.add_out_connector(new, node.out_connectors[old])
        node.remove_out_connector(old)
    for edge in state.in_edges(node):
        if edge.dst_conn == old:
            edge.dst_conn = new
    for edge in state.out_edges(node):
        if edge.src_conn == old:
            edge.src_conn = new


def lower(sdfg: dace.SDFG, expand: bool = True) -> None:
    """Prepare ``sdfg`` for emission, in place; ``expand=False`` leaves library nodes, which emission refuses."""
    if expand:
        expand_to_pure(sdfg)
    if placed(sdfg, nodes.NestedSDFG):
        inline_sdfgs(sdfg)
    # before nested data takes outer names, which a nested tasklet's connector may carry
    unique_connectors(sdfg)
    unique_nested_bindings(sdfg)
    if placed(sdfg, nodes.NestedSDFG):
        refuse_moved_symbolic_reads(sdfg)
        sdfg.apply_transformations_repeated(ExpandNestedSDFGInputs, validate=False)
        bind_nested_data(sdfg)
    InlineTaskletConnectors().apply_pass(sdfg, {})


def simplify_as_sizes(extent: sympy.Expr) -> sympy.Expr:
    """``extent`` simplified with every symbol a positive integer, as sizes are; drops a ``Max`` against zero."""
    sizes = {s: sympy.Symbol(str(s), integer=True, positive=True) for s in extent.free_symbols}
    simple = sympy.simplify(extent.subs(sizes))
    return simple.subs({size: s for s, size in sizes.items()})


def widen_scratch(sdfg: dace.SDFG, symbols: Iterable[str]) -> None:
    """Size each transient shaped by a loop variable at that variable's extreme value, so the caller can allocate it;
    refuse one whose extent the caller cannot evaluate."""
    known = set(symbols)
    ranges: dict[str, tuple[Any, Any]] = {}
    for region in sdfg.all_control_flow_regions():
        if isinstance(region, LoopRegion) and region.loop_variable:
            start, end = loop_analysis.get_init_assignment(region), loop_analysis.get_loop_end(region)
            if start is not None and end is not None:
                ranges[region.loop_variable] = (start, end)
    for name in scratch_arrays(sdfg):
        desc = sdfg.arrays[name]
        shape = []
        for dim in desc.shape:
            extent = sympy.sympify(dim)
            for level in range(len(ranges) + 1):  # a bound may itself name an outer loop's variable
                loop_vars = [s for s in extent.free_symbols if str(s) in ranges]
                if not loop_vars:
                    break
                for var in loop_vars:
                    lo, hi = ranges[str(var)]
                    extent = sympy.Max(extent.subs(var, lo), extent.subs(var, hi))  # monotone in var
            extent = simplify_as_sizes(extent)
            unknown = sorted(str(s) for s in extent.free_symbols if str(s) not in known)
            if unknown:
                raise UnsupportedNest(f"scratch buffer {name!r} has extent {dim}, which depends on {unknown}")
            shape.append(extent)
        if shape != list(desc.shape):
            sdfg.arrays[name] = dt.Array(desc.dtype, shape, transient=True, storage=desc.storage)


# emission of a lowered SDFG


def scratch_arrays(sdfg: dace.SDFG) -> list[str]:
    """Transient arrays the caller must pre-allocate; a transient scalar is a local instead."""
    return sorted(name for name, desc in sdfg.arrays.items() if desc.transient and name not in names_of(sdfg).locals)


def index(subset: subsets.Subset) -> str:
    """A subset as a NumPy index: an integer for a single element (the axis drops), else a slice."""
    ranges: list[tuple[Any, Any, Any]] = as_range(subset).ranges
    parts: list[str] = []
    for begin, end, step in ranges:
        begin_text = symbolic.symstr(begin, cpp_mode=False)
        if begin_text == symbolic.symstr(end, cpp_mode=False):
            parts.append(begin_text)
            continue
        if sympy.sympify(step).is_positive is not True:
            raise UnsupportedNest(f"subset step {step} is not provably positive; no sound NumPy slice")
        stop = symbolic.symstr(end + 1, cpp_mode=False)
        parts.append(f"{begin_text}:{stop}" if step == 1 else f"{begin_text}:{stop}:{symbolic.symstr(step)}")
    return ", ".join(parts)


def as_range(subset: subsets.Subset) -> subsets.Range:
    """``subset`` as a range; NumPy indexing needs one."""
    if isinstance(subset, subsets.Indices):
        subset = subsets.Range.from_indices(subset)
    if not isinstance(subset, subsets.Range):
        raise UnsupportedNest(f"subset {subset} is neither a range nor indices")
    return subset


def indexed_shape(subset: subsets.Subset) -> str:
    """The shape ``index(subset)`` selects, as a tuple literal: single-element axes drop, as in NumPy."""
    ranges: list[tuple[Any, Any, Any]] = as_range(subset).ranges
    kept = [(end - begin) // step + 1 for begin, end, step in ranges if begin != end]
    return "(" + "".join(f"{symbolic.symstr(extent, cpp_mode=False)}, " for extent in kept) + ")"


def access(scope: Scope, name: str, subset: subsets.Subset | None) -> str:
    """How the function reads or writes ``name[subset]``."""
    if name in scope.names.locals:
        return name
    whole = subsets.Range.from_array(scope.sdfg.arrays[name])
    return f"{name}[{index(subset if subset is not None else whole)}]"


def combine(wcr: str | ast.AST, target: str, value: str) -> str:
    """``target`` combined with ``value`` by a write-conflict resolution."""
    kind = detect_reduction_type(wcr)
    text = wcr if isinstance(wcr, str) else ast.unparse(wcr)
    return COMBINE[kind].format(target, value) if kind in COMBINE else f"({text})({target}, {value})"


@dataclass(frozen=True, slots=True)
class Scope:
    """Where emission is: the SDFG whose names are in use, and the iterators and outer names a write must not hit."""

    sdfg: dace.SDFG
    names: Names
    active: frozenset[str]


def scope_of(sdfg: dace.SDFG, active: frozenset[str] = frozenset()) -> Scope:
    return Scope(sdfg, names_of(sdfg), active)


def assign(scope: Scope, target: str, value: str) -> str:
    if target in scope.active:
        raise UnsupportedNest(f"assignment to {target!r} would overwrite an enclosing iterator or outer name")
    return f"{target} = {expr(value, scope.names)}"


def tasklet_lines(scope: Scope, state: SDFGState, node: nodes.Tasklet) -> list[str]:
    """A tasklet's body between the reads and writes of the connectors it still has."""
    if node.code.language != dtypes.Language.Python:
        raise UnsupportedNest(f"tasklet {node.label} is {node.code.language.name}, not Python")
    body = python(node.code.as_string, scope.names)
    used = {n.id for n in ast.walk(ast.parse(node.code.as_string)) if isinstance(n, ast.Name)}
    before: list[str] = []
    for edge in state.in_edges(node):
        if edge.dst_conn not in used or edge.data.is_empty():
            continue
        value = edge.src_conn if isinstance(edge.src, nodes.Tasklet) else access(scope, *memlet_parts(edge.data)[:2])
        before.append(f"{edge.dst_conn} = {value}")
    after: list[str] = []
    for edge in state.out_edges(node):
        if edge.src_conn not in used or edge.data.is_empty() or isinstance(edge.dst, nodes.Tasklet):
            continue
        data, subset, other = memlet_parts(edge.data)
        target = access(scope, data, subset)
        single = not isinstance(subset, (subsets.Range, subsets.Indices)) or subset.num_elements() == 1
        # the body writes some elements of a range, or may skip a dynamic write: start from what is there
        if edge.data.dynamic or not single:
            before.append(f"{edge.src_conn} = {target}")
        if single or edge.data.wcr is not None:  # a range is written through its view
            value = edge.src_conn if edge.data.wcr is None else combine(edge.data.wcr, target, edge.src_conn)
            after.append(f"{target} = {value}")
    return (
        [line for text in before for line in python(text, scope.names)]
        + body
        + [line for text in after for line in python(text, scope.names)]
    )


def copy_line(scope: Scope, src: nodes.AccessNode, dst: nodes.AccessNode, memlet: dace.Memlet) -> str:
    """``dst[..] = src[..]`` for one copy; a subset missing on one side mirrors the other between equal shapes."""
    data, subset, other = memlet_parts(memlet)
    if data == src.data:
        src_sub, dst_sub = subset, other
    elif data == dst.data:
        dst_sub, src_sub = subset, other
    else:
        raise UnsupportedNest(f"copy memlet {memlet.data!r} names neither {src.data!r} nor {dst.data!r}")
    src_shape, dst_shape = scope.sdfg.arrays[src.data].shape, scope.sdfg.arrays[dst.data].shape
    if [str(d) for d in src_shape] == [str(d) for d in dst_shape]:
        src_sub, dst_sub = src_sub or dst_sub, dst_sub or src_sub
    target, value = access(scope, dst.data, dst_sub), access(scope, src.data, src_sub)
    if len(src_shape) != len(dst_shape) and dst.data not in scope.names.locals:
        # a copy between ranks moves elements in row-major order
        shape = indexed_shape(dst_sub if dst_sub is not None else subsets.Range.from_array(scope.sdfg.arrays[dst.data]))
        value = f"np.reshape({value}, {shape})"
    value = value if memlet.wcr is None else combine(memlet.wcr, target, value)
    return python(f"{target} = {value}", scope.names)[0]


def copy_lines(scope: Scope, state: SDFGState, node: nodes.AccessNode) -> list[str]:
    """Copies into ``node`` from an access node, and copies out of ``node`` through a map exit."""
    lines: list[str] = []
    for edge in state.in_edges(node):
        path = [] if edge.data.is_empty() else state.memlet_path(edge)
        # a copy leaving a map is emitted where its source is written, below
        if (
            path
            and isinstance(path[0].src, nodes.AccessNode)
            and not any(isinstance(e.src, nodes.MapExit) for e in path)
        ):
            lines.append(copy_line(scope, path[0].src, node, edge.data))
    for edge in state.out_edges(node):
        if isinstance(edge.dst, nodes.MapExit) and not edge.data.is_empty():
            last = state.memlet_path(edge)[-1].dst
            if isinstance(last, nodes.AccessNode) and not (last.data == node.data and edge.data.wcr is None):
                lines.append(copy_line(scope, node, last, edge.data))
    return lines


def indent(lines: list[str]) -> list[str]:
    return ["    " + line for line in lines] if lines else ["    pass"]


def map_lines(scope: Scope, state: SDFGState, entry: nodes.MapEntry, order: list[nodes.Node]) -> list[str]:
    """A map as nested ``for`` loops; its dynamic range inputs are read first."""
    lines = [
        f"{e.dst_conn} = {access(scope, *memlet_parts(e.data)[:2])}"
        for e in state.in_edges(entry)
        if e.dst_conn and not e.dst_conn.startswith("IN_") and not e.data.is_empty()
    ]
    headers: list[str] = []
    for param, (begin, end, step) in zip(entry.map.params, entry.map.range.ranges):
        if param in scope.active:
            raise UnsupportedNest(f"map parameter {param!r} shadows an enclosing iterator")
        sign = sympy.sign(sympy.sympify(step))
        if sign not in (1, -1):
            raise UnsupportedNest(f"map parameter {param!r} has step {step} of undecidable sign")
        bounds = [expr(begin, scope.names), expr(end + sign, scope.names)]
        stepped = bounds if step == 1 else [*bounds, expr(step, scope.names)]
        headers.append(f"for {param} in range({', '.join(stepped)}):  {PARALLEL}")
    inner = Scope(scope.sdfg, scope.names, scope.active | set(strings(entry.map.params)))
    body = scope_lines(inner, state, entry, order)
    for header in reversed(headers):
        body = [header, *indent(body)]
    return [*python("\n".join(lines), scope.names), *body] if lines else body


def nested_lines(scope: Scope, state: SDFGState, node: nodes.NestedSDFG) -> list[str]:
    """A nested SDFG inlined: its symbols bound, then its body; an early return ends a one-trip loop."""
    if any(e.data.wcr is not None for e in state.out_edges(node)):
        raise UnsupportedNest(f"nested SDFG {node.label} writes through a write-conflict resolution")
    inner_sdfg = node.sdfg
    outer_names = scope.active | set(scope.sdfg.arrays) | set(scope.sdfg.symbols)
    binds = {str(k): expr(v, scope.names) for k, v in node.symbol_mapping.items() if str(k) != symbolic.symstr(v)}
    private = [n for n, desc in inner_sdfg.arrays.items() if n not in scope.sdfg.arrays and n in outer_names]
    if private or binds.keys() & outer_names:
        raise UnsupportedNest(
            f"nested SDFG {node.label} rebinds outer names {sorted(private or binds.keys() & outer_names)}"
        )
    reads = {str(s) for value in binds.values() for s in symbolic.pystr_to_symbolic(value).free_symbols}
    lines = (
        [f"{', '.join(binds)} = {', '.join(binds.values())}"]
        if binds.keys() & reads
        else [f"{k} = {v}" for k, v in binds.items()]
    )
    inner = scope_of(inner_sdfg, frozenset(outer_names | binds.keys()))
    body = region_lines(inner, inner_sdfg, None)
    returns = [b for b in inner_sdfg.all_control_flow_blocks() if isinstance(b, ReturnBlock)]
    if not returns:
        return lines + body
    if any(enclosing_loop(r) is not None for r in returns) or ONCE in outer_names:
        raise UnsupportedNest(f"nested SDFG {node.label} returns from inside a loop, or {ONCE!r} is taken")
    body = [line.replace("return", "break") if line.strip() == "return" else line for line in body]
    return [*lines, f"for {ONCE} in range(1):", *indent(body)]


def enclosing_loop(block: ControlFlowBlock) -> LoopRegion | None:
    """The loop a ``break`` or ``continue`` in ``block`` targets, within its SDFG."""
    region = block.parent_graph
    while region is not None and not isinstance(region, (LoopRegion, dace.SDFG)):
        region = region.parent_graph
    return region if isinstance(region, LoopRegion) else None


def node_lines(scope: Scope, state: SDFGState, node: nodes.Node, order: list[nodes.Node]) -> list[str]:
    if isinstance(node, nodes.Tasklet):
        return tasklet_lines(scope, state, node)
    if isinstance(node, nodes.AccessNode):
        return copy_lines(scope, state, node)
    if isinstance(node, nodes.MapEntry):
        return map_lines(scope, state, node, order)
    if isinstance(node, nodes.NestedSDFG):
        return nested_lines(scope, state, node)
    if isinstance(node, nodes.MapExit):
        return []
    raise UnsupportedNest(f"{type(node).__name__} {node} is not emitted")


def scope_lines(scope: Scope, state: SDFGState, entry: nodes.MapEntry | None, order: list[nodes.Node]) -> list[str]:
    """The nodes directly inside ``entry`` (the state's top level for ``None``), in dataflow order."""
    parents = state.scope_dict()
    return [line for node in order if parents[node] is entry for line in node_lines(scope, state, node, order)]


def state_lines(scope: Scope, state: SDFGState) -> list[str]:
    return scope_lines(scope, state, None, list(dfs_topological_sort(state)))


def interstate_lines(scope: Scope, region: ControlFlowRegion, block: ControlFlowBlock) -> list[str]:
    """The assignments on the one edge entering ``block`` that carries any."""
    carrying = []
    for edge in region.in_edges(block):
        if not edge.data.is_unconditional():
            raise UnsupportedNest(f"conditional interstate edge into {block.label}: unstructured control flow")
        if edge.data.assignments:
            carrying.append(edge)
    if len(carrying) > 1:
        raise UnsupportedNest(f"{len(carrying)} interstate edges into {block.label} carry assignments")
    return [assign(scope, k, v) for edge in carrying for k, v in edge.data.assignments.items()]


def loop_lines(scope: Scope, loop: LoopRegion) -> list[str]:
    """A loop as ``while``; a do-while tests its condition at the end of the body."""
    var = loop.loop_variable
    if var and var in scope.active:
        raise UnsupportedNest(f"loop variable {var!r} shadows an enclosing iterator")
    cond = expr(loop.loop_condition.as_string, scope.names)
    init = [] if loop.init_statement is None else python(loop.init_statement.as_string, scope.names)
    update = [] if loop.update_statement is None else python(loop.update_statement.as_string, scope.names)
    continues = any(isinstance(b, ContinueBlock) and enclosing_loop(b) is loop for b in loop.all_control_flow_blocks())
    if loop.inverted and continues:
        raise UnsupportedNest(f"do-while loop {loop.label} holds a continue, which would skip its exit test")
    inner = Scope(scope.sdfg, scope.names, scope.active | ({var} if var else set()))
    body = region_lines(inner, loop, None if loop.inverted else "\n".join(update))
    if not loop.inverted:
        return [*init, f"while {cond}:", *indent(body + update)]
    test = [f"if not ({cond}):", "    break"]
    return [*init, "while True:", *indent(body + (update + test if loop.update_before_condition else test + update))]


def conditional_lines(scope: Scope, block: ConditionalBlock, update: str | None) -> list[str]:
    lines: list[str] = []
    for position, (condition, branch) in enumerate(block.branches):
        if condition is None and position != len(block.branches) - 1:
            raise UnsupportedNest(f"conditional {block.label} has an unconditional branch before its last")
        keyword = "else" if condition is None else ("if" if position == 0 else "elif")
        test = "" if condition is None else " " + expr(condition.as_string, scope.names)
        lines += [f"{keyword}{test}:", *indent(region_lines(scope, branch, update))]
    return lines


def block_lines(scope: Scope, block: ControlFlowBlock, update: str | None) -> list[str]:
    if isinstance(block, SDFGState):
        return state_lines(scope, block)
    if isinstance(block, LoopRegion):
        return loop_lines(scope, block)
    if isinstance(block, ConditionalBlock):
        return conditional_lines(scope, block, update)
    if isinstance(block, BreakBlock):
        return ["break"]
    if isinstance(block, ContinueBlock):
        return [*([] if update is None else update.splitlines()), "continue"]
    if isinstance(block, ReturnBlock):
        return ["return"]
    if isinstance(block, ControlFlowRegion):
        return region_lines(scope, block, update)
    raise UnsupportedNest(f"control-flow block {type(block).__name__} is not emitted")


def region_lines(scope: Scope, region: ControlFlowRegion, update: str | None) -> list[str]:
    """Every block of ``region`` in execution order; ``update`` is the enclosing loop's, run before a continue."""
    lines: list[str] = []
    for block in dfs_topological_sort(region, [region.start_block]):
        lines += interstate_lines(scope, region, block) + block_lines(scope, block, update)
    return lines


def module(fn_name: str, args: list[str], body: list[str]) -> str:
    """The standalone module defining ``fn_name(args)`` with ``body``; refuses code Python rejects."""
    function = "\n".join([f"def {fn_name}({', '.join(args)}):", *indent(body)])
    imports = "import math\n\nimport numpy as np\n" if re.search(r"\bmath\.", function) else "import numpy as np\n"
    # a DaCe precondition guard calls abort()
    guard = "\n\ndef abort():\n    raise RuntimeError('DaCe precondition violated')\n" if "abort(" in function else ""
    source = f"{imports}{guard}\n\n{function}\n"
    try:
        compile(source, fn_name, "exec")
    except SyntaxError as exc:
        raise UnsupportedNest(f"emitted {fn_name} is not valid Python: {exc}") from exc
    return source


def render(fn_name: str, args: list[str], sdfg: dace.SDFG) -> str:
    """The module defining ``fn_name(args)`` over a lowered ``sdfg``."""
    return module(fn_name, args, region_lines(scope_of(sdfg), sdfg, None))


def map_python(state: SDFGState, entry: nodes.MapEntry, body_only: bool = False) -> list[str]:
    """One map of a lowered SDFG as ``for`` loops, or only what one iteration computes."""
    params = frozenset(strings(entry.map.params))
    scope = Scope(state.sdfg, names_of(state.sdfg), params if body_only else frozenset())
    order = list(dfs_topological_sort(state))
    return scope_lines(scope, state, entry, order) if body_only else map_lines(scope, state, entry, order)


# extracted nests


def oracle_sdfg(boundary: Boundary) -> dace.SDFG:
    """The lowered copy of a nest the oracle is emitted from; its scratch is sized for the caller to allocate."""
    if any(isinstance(b, ReturnBlock) for b in boundary.standalone_sdfg.all_control_flow_blocks()):
        raise UnsupportedNest("the nest returns early, out of its enclosing SDFG; it cannot stand alone")
    sdfg = copy.deepcopy(boundary.standalone_sdfg)
    lower(sdfg)
    widen_scratch(sdfg, boundary.symbols)
    return sdfg


def kernel_arrays(boundary: Boundary, sdfg: dace.SDFG) -> list[str]:
    """Array arguments in signature order: inputs, further outputs, then scratch, since a kernel allocates nothing."""
    names = list(boundary.inputs) + [o for o in boundary.outputs if o not in boundary.inputs]
    return names + [s for s in scratch_arrays(sdfg) if s not in names]


def kernel_args(boundary: Boundary, arrays: list[str]) -> list[str]:
    """Every argument in signature order: :func:`kernel_arrays`, then the symbols."""
    return [*arrays, *(s for s in boundary.symbols if s not in arrays)]


def nest_to_python(boundary: Boundary, fn_name: str = "kernel") -> str:
    """Standalone Python source ``def <fn_name>(<args>): ...`` of an extracted nest."""
    sdfg = oracle_sdfg(boundary)
    return render(fn_name, kernel_args(boundary, kernel_arrays(boundary, sdfg)), sdfg)


@functools.lru_cache(maxsize=1, typed=True)
def emitted_dir() -> Path:
    """Process-lifetime directory of emitted sources: ``linecache`` reads a traceback's source lazily."""
    path = Path(tempfile.mkdtemp(prefix="nestforge-emitted-"))
    atexit.register(shutil.rmtree, path, ignore_errors=True)
    return path


def load_emitted(source: str, name: str) -> ModuleType:
    """Import emitted ``source`` as a module; the file name hashes the source, since bytecode caching keys on mtime."""
    path = emitted_dir() / f"{name}_{hashlib.sha256(source.encode()).hexdigest()[:HASH_DIGITS]}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(f"nestforge_emitted.{name}", path)
    assert spec is not None and spec.loader is not None, f"{path} is not importable"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
