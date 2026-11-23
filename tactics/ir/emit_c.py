# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""What one iteration of a leaf doall computes, as C: the tree's second spelling of a body (the first is Python,
:func:`~tactics.ir.introspect.body_lines`). A Python tasklet is spelled by DaCe's own C++ unparser; a C++ tasklet
(intrinsics, inline assembly: what vectorization leaves) is printed as written. Each connector reads or writes the
element its memlet names, and a connector that takes more than one element (a vector lane, a tile) takes the address
of the first. Nothing is lowered or changed: the live program is read."""

import re

import dace
from dace import dtypes, subsets
from dace.codegen.common import sym2cpp
from dace.codegen.cppunparse import py2cpp
from dace.frontend.operations import detect_reduction_type
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState

from tactics.ir.names import in_order, ordered_scope_children

__all__ = ["body_c"]

#: ``ReductionType`` -> the C compound assignment it is; any other accumulation is spelled as a call.
COMPOUND = {
    dtypes.ReductionType.Sum: "+=",
    dtypes.ReductionType.Product: "*=",
    dtypes.ReductionType.Bitwise_And: "&=",
    dtypes.ReductionType.Bitwise_Or: "|=",
    dtypes.ReductionType.Bitwise_Xor: "^=",
}
#: One level of a C block.
INDENT = "    "
#: ``ReductionType`` -> the C function an accumulation calls when it is no compound assignment.
CALLED = {dtypes.ReductionType.Min: "min", dtypes.ReductionType.Max: "max"}


def element(data: str, subset: subsets.Subset) -> str:
    """The C access of ``subset`` of ``data``: its element, or the address of its first one when it spans more."""
    starts = "".join(f"[{sym2cpp(start)}]" for start, _, _ in subset.ndrange())
    return f"{data}{starts}" if subset.num_elements() == 1 else f"&{data}{starts}"


def access(memlet: dace.Memlet) -> str:
    """The C access a tasklet's memlet names."""
    assert isinstance(memlet.subset, subsets.Subset), "a data memlet has a subset"
    return element(memlet.data, memlet.subset)


def accumulate(target: str, wcr: str, value: str) -> str:
    """``target`` combined with ``value`` by ``wcr``, as one C statement."""
    kind = detect_reduction_type(wcr)
    if kind in COMPOUND:
        return f"{target} {COMPOUND[kind]} {value};"
    if kind in CALLED:
        return f"{target} = {CALLED[kind]}({target}, {value});"
    return f"{target} = ({wcr})({target}, {value});"


def tasklet_c(state: SDFGState, tasklet: nodes.Tasklet) -> list[str]:
    """``tasklet`` with each connector replaced by the access its memlet names; an accumulated output keeps its
    connector as a local and is combined into its target after the code."""
    code = tasklet.code.as_string
    # the connectors are declared (they become accesses below), so no assignment to one is spelled a declaration
    connectors = {**tasklet.in_connectors, **tasklet.out_connectors}
    text = py2cpp(code, defined_symbols=connectors) if tasklet.language is dtypes.Language.Python else code
    named: dict[str, str] = {}
    folds: list[str] = []
    for edge in state.in_edges(tasklet):
        if edge.dst_conn is not None and not edge.data.is_empty():
            named[edge.dst_conn] = access(edge.data)
    for edge in state.out_edges(tasklet):
        if edge.src_conn is None or edge.data.is_empty():
            continue
        if edge.data.wcr is None:
            named[edge.src_conn] = access(edge.data)
        else:
            folds.append(accumulate(access(edge.data), edge.data.wcr, edge.src_conn))
    if named:
        pattern = re.compile(r"\b(" + "|".join(map(re.escape, sorted(named, key=len, reverse=True))) + r")\b")
        text = pattern.sub(lambda match: named[match.group(1)], text)
    return [line for line in text.splitlines() if line.strip()] + folds


def copy_c(memlet: dace.Memlet, src: nodes.AccessNode, dst: nodes.AccessNode) -> str:
    """A copy between two access nodes: the memlet's subset is of its own data, ``other_subset`` (default the same)
    of the other end."""
    own, other = memlet.subset, memlet.other_subset if memlet.other_subset is not None else memlet.subset
    assert isinstance(own, subsets.Subset) and isinstance(other, subsets.Subset), "a copy memlet has subsets"
    read, written = (own, other) if memlet.data == src.data else (other, own)
    return f"{element(dst.data, written)} = {element(src.data, read)};"


def access_out_c(state: SDFGState, node: nodes.AccessNode) -> list[str]:
    """What leaves the access node ``node``: copies into other access nodes, and writes (an accumulation in canonical
    form: ``tasklet -> local -wcr-> map exit``) of its value into the array a memlet of the scope's exit names."""
    lines: list[str] = []
    for edge in state.out_edges(node):
        memlet = edge.data
        if memlet.is_empty():
            continue
        if isinstance(edge.dst, nodes.AccessNode):
            lines.append(copy_c(memlet, node, edge.dst))
        elif isinstance(edge.dst, nodes.MapExit) and memlet.data != node.data:
            assert isinstance(memlet.subset, subsets.Subset), "a data memlet has a subset"
            own = memlet.other_subset if memlet.other_subset is not None else subsets.Range.from_array(node.desc(state))
            value = element(node.data, own)
            target = element(memlet.data, memlet.subset)
            lines.append(f"{target} = {value};" if memlet.wcr is None else accumulate(target, memlet.wcr, value))
    return lines


def loop_header(entry: nodes.MapEntry) -> list[str]:
    """One C ``for`` per dimension of ``entry``'s map, outermost first."""
    heads = []
    for param, (start, end, step) in zip(entry.map.params, entry.map.range.ranges):
        heads.append(f"for ({param} = {sym2cpp(start)}; {param} <= {sym2cpp(end)}; {param} += {sym2cpp(step)}) {{")
    return heads


def indent(lines: list[str]) -> list[str]:
    return [INDENT + line for line in lines]


def scope_c(state: SDFGState, scope: nodes.MapEntry | None) -> list[str]:
    """The statements directly in ``scope`` of ``state`` (``None``: its top level), in execution order; a map
    becomes its loops, a nested SDFG its control flow, a library call a comment naming it."""
    lines: list[str] = []
    for node in ordered_scope_children(state)[scope]:
        if isinstance(node, nodes.Tasklet):
            lines += tasklet_c(state, node)
        elif isinstance(node, nodes.AccessNode):
            lines += access_out_c(state, node)
        elif isinstance(node, nodes.MapEntry):
            heads = loop_header(node)
            body = scope_c(state, node)
            for _ in heads:
                body = indent(body)
            lines += [INDENT * depth + head for depth, head in enumerate(heads)] + body
            lines += [INDENT * depth + "}" for depth in reversed(range(len(heads)))]
        elif isinstance(node, nodes.NestedSDFG):
            lines += region_c(node.sdfg)
        elif isinstance(node, nodes.LibraryNode):
            lines.append(f"// {node.label}: {type(node).__name__} [{node.implementation or 'auto'}]")
    return lines


def region_c(region: ControlFlowRegion) -> list[str]:
    """A control-flow region's blocks as C, in execution order, each followed by its interstate assignments (under
    the edge's condition when it has one)."""
    lines: list[str] = []
    for block in in_order(region):
        if isinstance(block, SDFGState):
            lines += scope_c(block, None)
        elif isinstance(block, LoopRegion):
            init = py2cpp(block.init_statement.as_string, expr_semicolon=False) if block.init_statement else ""
            cond = py2cpp(block.loop_condition.as_string, expr_semicolon=False)
            step = py2cpp(block.update_statement.as_string, expr_semicolon=False) if block.update_statement else ""
            lines += [f"for ({init}; {cond}; {step}) {{", *indent(region_c(block)), "}"]
        elif isinstance(block, ConditionalBlock):
            for index, (condition, branch) in enumerate(block.branches):
                if condition is None:
                    head = "} else {"
                else:
                    test = py2cpp(condition.as_string, expr_semicolon=False)
                    head = f"if ({test}) {{" if index == 0 else f"}} else if ({test}) {{"
                lines += [head, *indent(region_c(branch))]
            lines.append("}")
        elif isinstance(block, ControlFlowRegion):
            lines += region_c(block)
        for edge in region.out_edges(block):
            assigned = [
                f"{name} = {py2cpp(value, expr_semicolon=False)};" for name, value in edge.data.assignments.items()
            ]
            if assigned and not edge.data.is_unconditional():
                # an edge taken only when its condition holds assigns only then
                test = py2cpp(edge.data.condition.as_string, expr_semicolon=False)
                assigned = [f"if ({test}) {{", *indent(assigned), "}"]
            lines += assigned
    return lines


def body_c(state: SDFGState, entry: nodes.MapEntry) -> list[str]:
    """What one iteration of the leaf doall ``entry`` computes, as C, statement by statement in execution order,
    nested SDFGs (a vectorized body's tile loads, operations and stores) written out where they run."""
    return scope_c(state, entry)
