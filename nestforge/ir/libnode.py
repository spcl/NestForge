# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The ``ExternalCall`` library node: a kernel call that expands to the extracted nest (``DaceReference``) or to a
call into a linked library (``ExternCall``)."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any
from collections.abc import Collection, Sequence

import numpy as np

import dace
import dace.library
import dace.properties
from dace import dtypes, subsets
from dace.ordered import OrderedSet
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation

from nestforge.ir.dace_types import strings

CPP_SCALAR = {"float64": "double", "float32": "float", "int64": "int64_t", "int32": "int32_t"}


def in_conn(name: str) -> str:
    """Connector of an input; distinct from the data name, as a library node requires."""
    return f"_in_{name}"


def out_conn(name: str) -> str:
    """Connector of an output."""
    return f"_out_{name}"


def connector_for(arg: str, outputs: Collection[str]) -> str:
    return out_conn(arg) if arg in outputs else in_conn(arg)


def value_connectors(node: ExternalCall, state: dace.SDFGState) -> set[str]:
    """Connectors whose memlet covers one element: DaCe declares these as values, so the call takes their address."""
    single: set[str] = set()
    for edge in state.in_edges(node):
        if edge.dst_conn is not None and one_element(edge.data):
            single.add(edge.dst_conn)
    for edge in state.out_edges(node):
        # a dynamic non-WCR output stays a pointer
        dynamic_pointer = edge.data.dynamic and edge.data.wcr is None
        if edge.src_conn is not None and one_element(edge.data) and not dynamic_pointer:
            single.add(edge.src_conn)
    return single


def one_element(memlet: dace.Memlet) -> bool:
    subset = memlet.subset
    return isinstance(subset, (subsets.Range, subsets.Indices)) and bool(subset) and subset.num_elements() == 1


def scalar_inputs(node: ExternalCall, state: dace.SDFGState) -> OrderedSet:
    """Input connectors fed from a ``Scalar`` in the parent: the kernel takes those by value."""
    return OrderedSet(
        edge.dst_conn
        for edge in state.in_edges(node)
        if edge.dst_conn is not None
        and edge.data.data is not None
        and isinstance(state.sdfg.arrays[edge.data.data], dace.data.Scalar)
    )


@dataclass(slots=True)
class CallSite:
    """What the parent state says about the connectors of one ``ExternalCall``."""

    outputs: Collection[str]
    connectors: Collection[str]
    by_value: Collection[str]
    scalars: Collection[str]


def data_param(node: ExternalCall, arg: str, dtype: str, site: CallSite) -> tuple[str, str]:
    """``(parameter, call argument)`` of one data argument: a read-only Scalar input by value, the rest by pointer."""
    if dtype not in CPP_SCALAR:
        raise ValueError(
            f"ExternalCall {node.name!r}: array {arg!r} has dtype {dtype!r}, which has no "
            f"extern-C spelling (known: {sorted(CPP_SCALAR)}); keep the DaceReference "
            "implementation for this nest"
        )
    conn = connector_for(arg, site.outputs)
    if conn not in site.connectors:
        raise ValueError(
            f"ExternalCall {node.name!r}: abi_order names {arg!r}, but the node has no "
            f"{conn!r} connector (a caller-allocated scratch buffer is not passed across "
            "the ExternalCall boundary); keep the DaceReference implementation"
        )
    ctype = CPP_SCALAR[dtype]
    if conn in site.scalars:
        return f"{ctype} {arg}", conn
    const = "" if arg in site.outputs else "const "
    return f"{const}{ctype}* {arg}", f"&{conn}" if conn in site.by_value else conn


def proto_and_call(node: ExternalCall, state: dace.SDFGState) -> tuple[str, str]:
    """The ``extern "C"`` prototype and call of the linked kernel, in ``node.abi_order``. C linkage matches the name
    alone, so any other order links cleanly and swaps buffers."""
    manifest = node.config
    if manifest is None:
        raise ValueError(f"ExternalCall {node.name!r} has no manifest")
    arrays = set(manifest["array_args"])
    dtypes_map = {a: v["dtype"] for a, v in manifest["init"]["arrays"].items()}
    scalar_dtypes = {n: np.dtype(type(v)).name for n, v in manifest["init"].get("scalars", {}).items()}
    order = strings(node.abi_order or [])
    if not order:
        raise ValueError(
            f"ExternalCall {node.name!r} has no abi_order; the extern-call expansion must declare the linked "
            "symbol in the order it was compiled with"
        )
    # the connector sets are dace Properties, re-resolved on every access, so read them once
    site = CallSite(
        outputs=set(manifest["output_args"]),
        connectors={*node.in_connectors, *node.out_connectors},
        by_value=value_connectors(node, state),
        scalars=scalar_inputs(node, state),
    )
    params: list[str] = []
    call_args: list[str] = []
    for arg in order:
        if arg not in arrays:
            params.append(f"{CPP_SCALAR.get(scalar_dtypes.get(arg, 'int64'), 'int64_t')} {arg}")
            call_args.append(arg)
            continue
        param, call_arg = data_param(node, arg, dtypes_map[arg], site)
        params.append(param)
        call_args.append(call_arg)
    proto = f'extern "C" void {node.symbol}({", ".join(params)});'
    call = f"{node.symbol}({', '.join(call_args)});"
    return proto, call


def with_new_items(existing: list[str], items: Sequence[str]) -> list[str]:
    """``existing`` followed by each item of ``items`` it does not hold yet, in order."""
    return [*existing, *(item for item in dict.fromkeys(items) if item not in existing)]


@dace.library.environment
class ExternLibEnv:
    """Links the kernel libraries into the program; DaCe reads environments as classes, so ``configure`` sets
    class attributes during expansion."""

    __slots__ = ()

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []
    headers = []
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []

    @classmethod
    def reset(cls) -> None:
        """Drop every accumulated library; call before expanding a fresh SDFG."""
        cls.cmake_libraries = []
        cls.cmake_link_flags = []

    @classmethod
    def configure(cls, lib_path: str, runtime_libraries: Sequence[str] = ()) -> None:
        """Add one kernel's library and its runtimes to the link; every ``ExternalCall`` shares this class."""
        lib = os.path.abspath(lib_path)
        cls.cmake_libraries = with_new_items(cls.cmake_libraries, [lib, *runtime_libraries])
        if not lib.endswith(".a"):  # a shared library needs an rpath
            cls.cmake_link_flags = with_new_items(cls.cmake_link_flags, [f"-Wl,-rpath,{os.path.dirname(lib)}"])


@dace.library.expansion
class ExpandDaceReference(ExpandTransformation):
    """Expand to a copy of the extracted nest."""

    # no __slots__: make_properties needs a __dict__
    environments = []

    @staticmethod
    def expansion(node: ExternalCall, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> dace.SDFG:
        if node.standalone_sdfg is None:
            raise ValueError(f"ExternalCall {node.name} has no standalone SDFG to fall back to")
        return copy.deepcopy(node.standalone_sdfg)


@dace.library.expansion
class ExpandExternCall(ExpandTransformation):
    """Expand to a C++ tasklet calling the linked library's entry."""

    # no __slots__: make_properties needs a __dict__
    environments = [ExternLibEnv]

    @staticmethod
    def expansion(node: ExternalCall, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        if not node.lib_path or not node.symbol:
            raise ValueError(f"ExternalCall {node.name} needs lib_path + symbol for ExpandExternCall")
        proto, call = proto_and_call(node, parent_state)
        ExternLibEnv.configure(node.lib_path, strings(node.runtime_libraries))
        tasklet = nodes.Tasklet(
            node.name,
            node.in_connectors,
            node.out_connectors,
            call,
            language=dtypes.Language.CPP,
            code_global=proto,
            side_effects=True,
        )
        return tasklet


@dace.library.node
class ExternalCall(nodes.LibraryNode):
    """A nest lowered to a call of a separately compiled kernel."""

    # no __slots__: make_properties needs a __dict__

    implementations = {"DaceReference": ExpandDaceReference, "ExternCall": ExpandExternCall}
    default_implementation = "DaceReference"

    numpy_source = dace.properties.Property(dtype=str, default="", desc="NumPy oracle of the nest")
    config = dace.properties.DictProperty(
        key_type=str, value_type=object, default=None, desc="argument manifest (symbols, shapes, dtypes)"
    )
    symbol = dace.properties.Property(dtype=str, default="", desc="extern-C symbol to call")
    abi_order = dace.properties.ListProperty(
        element_type=str,
        default=[],
        desc="parameter order of the linked entry",
    )
    lib_path = dace.properties.Property(dtype=str, default="", desc="kernel library, .a or .so")
    runtime_libraries = dace.properties.ListProperty(
        element_type=str,
        default=[],
        desc="link items for the runtimes the linked library needs (libomp, cudart), placed after the objects",
    )
    fp_mode = dace.properties.Property(dtype=str, default="", desc="winning FP mode")

    def __init__(
        self,
        name: str,
        inputs: Sequence[str] | None = None,
        outputs: Sequence[str] | None = None,
        numpy_source: str = "",
        config: dict[str, Any] | None = None,
        standalone_sdfg: dace.SDFG | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, inputs=list(inputs or []), outputs=list(outputs or []), **kwargs)
        self.numpy_source = numpy_source
        self.config = config
        self.standalone_sdfg = standalone_sdfg

    @property
    def standalone_sdfg(self) -> dace.SDFG | None:
        """The extracted nest, in memory only; ``make_properties`` accepts no other stored attribute than an
        underscored one, which DaCe dictates here."""
        return self._standalone_sdfg

    @standalone_sdfg.setter
    def standalone_sdfg(self, value: dace.SDFG | None) -> None:
        self._standalone_sdfg = value


def external_calls(sdfg: dace.SDFG) -> list[ExternalCall]:
    """Every kernel node of ``sdfg``, nested SDFGs included."""
    return [node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, ExternalCall)]
