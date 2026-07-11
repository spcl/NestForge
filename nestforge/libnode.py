"""The ``ExternalCall`` library node and its expansions.

Payload (numpy source + manifest config + the chosen compiled lib) rides on the node as
properties. Two expansions:
  * ``ExpandDaceReference`` (default) -- rebuilds the extracted nest as a NestedSDFG (the DaCe
    competitor and the correctness fallback);
  * ``ExpandExternCall`` -- a CPP tasklet calling the extern-C entry of the chosen ``.so``,
    linked via a per-node environment.
"""
from __future__ import annotations

import copy
from typing import List

import dace
import dace.library
import dace.properties
from dace import dtypes
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation

_CPP_SCALAR = {"float64": "double", "float32": "float", "int64": "int64_t", "int32": "int32_t"}


def in_conn(name: str) -> str:
    """Connector name for an input array (kept distinct from the array name itself)."""
    return f"_in_{name}"


def out_conn(name: str) -> str:
    """Connector name for an output array."""
    return f"_out_{name}"


def connector_for(arg: str, outputs: set) -> str:
    return out_conn(arg) if arg in outputs else in_conn(arg)


def proto_and_call(node: "ExternalCall") -> (str, str):
    """Build the ``extern "C"`` prototype and the call expression from the manifest.

    Array args are passed by their connector variable (``_in_X`` / ``_out_Y``); size symbols are
    referenced by name (available in the tasklet's symbol scope).
    """
    manifest = node.config
    arrays = set(manifest["array_args"])
    outputs = set(manifest["output_args"])
    dtypes_map = {a: v["dtype"] for a, v in manifest["init"]["arrays"].items()}
    params: List[str] = []
    call_args: List[str] = []
    for arg in manifest["input_args"]:
        if arg in arrays:
            c = _CPP_SCALAR[dtypes_map[arg]]
            const = "" if arg in outputs else "const "
            params.append(f"{const}{c}* {arg}")
            call_args.append(connector_for(arg, outputs))
        else:
            params.append(f"int64_t {arg}")
            call_args.append(arg)
    proto = f'extern "C" void {node.symbol}({", ".join(params)});'
    call = f'{node.symbol}({", ".join(call_args)});'
    return proto, call


@dace.library.environment
class ExternLibEnv:
    """Links the chosen compiled ``.so`` into the SDFG program.

    The lib path is per-node, so :meth:`configure` stamps it onto the (module-level, hence
    registry-resolvable) class right before expansion. M0 assumes one extern call per build;
    per-node isolation is an M1 concern.
    """
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
    def configure(cls, lib_path: str) -> None:
        import os
        lib = os.path.abspath(lib_path)
        cls.cmake_libraries = [lib]
        cls.cmake_link_flags = [f"-Wl,-rpath,{os.path.dirname(lib)}"]


@dace.library.expansion
class ExpandDaceReference(ExpandTransformation):
    """Rebuild the extracted nest as a NestedSDFG (DaCe competitor / correctness fallback)."""
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        if node._standalone_sdfg is None:
            raise ValueError(f"ExternalCall {node.name} has no standalone SDFG to fall back to")
        return copy.deepcopy(node._standalone_sdfg)


@dace.library.expansion
class ExpandExternCall(ExpandTransformation):
    """Call the extern-C entry of the chosen compiled ``.so`` from a CPP tasklet."""
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        if not node.lib_path or not node.symbol:
            raise ValueError(f"ExternalCall {node.name} needs lib_path + symbol for ExpandExternCall")
        proto, call = proto_and_call(node)
        ExternLibEnv.configure(node.lib_path)
        ExpandExternCall.environments = [ExternLibEnv]
        tasklet = nodes.Tasklet(node.name,
                                node.in_connectors,
                                node.out_connectors,
                                call,
                                language=dtypes.Language.CPP,
                                code_global=proto,
                                side_effects=True)
        return tasklet


@dace.library.node
class ExternalCall(nodes.LibraryNode):
    """A loop-/map-nest lowered to an external, separately-compiled call."""

    implementations = {"DaceReference": ExpandDaceReference, "ExternCall": ExpandExternCall}
    default_implementation = "DaceReference"

    numpy_source = dace.properties.Property(dtype=str, default="", desc="numpy reference of the nest")
    config = dace.properties.DictProperty(key_type=str,
                                          value_type=object,
                                          default=None,
                                          desc="OptArena manifest (symbols, shapes, dtypes)")
    symbol = dace.properties.Property(dtype=str, default="", desc="extern-C symbol to call")
    lib_path = dace.properties.Property(dtype=str, default="", desc="compiled static/shared lib")
    fp_mode = dace.properties.Property(dtype=str, default="", desc="winning FP mode")

    def __init__(self, name, inputs=None, outputs=None, numpy_source="", config=None, standalone_sdfg=None, **kwargs):
        super().__init__(name, inputs=inputs or set(), outputs=outputs or set(), **kwargs)
        self.numpy_source = numpy_source
        self.config = config
        self._standalone_sdfg = standalone_sdfg  # in-memory only (not serialized in M0)
