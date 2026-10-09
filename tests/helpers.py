# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Helpers several test modules share."""

import copy
import ctypes
import inspect
import re

import dace
import numpy as np
from dace import symbolic
from dace.libraries.standard.nodes.external_call import ExpandExternCall, ExternalCall

from nestforge.build.toolchain import CType, raw_signature
from nestforge.corpus.bench import CorpusKernel, iter_dace_kernels
from nestforge.ir.emit_python import load_emitted, lower, render, scratch_arrays, widen_scratch
from nestforge.ir.extract import Boundary
from nestforge.ir.introspect import tree_rows
from nestforge.ir.names import normalize_labels
from nestforge.stages.moves import Rewrite, legal_moves, plan_move

#: NumPy dtype name -> ctypes scalar; DaCe lowers a comparison transient to C bool.
CTYPE = {
    "float64": ctypes.c_double,
    "float32": ctypes.c_float,
    "int64": ctypes.c_int64,
    "int32": ctypes.c_int32,
    "bool": ctypes.c_bool,
}


def corpus_kernel(short_name: str) -> CorpusKernel:
    """The corpus kernel named ``short_name`` (``track/.../module``)."""
    return {k.short_name: k for k in iter_dace_kernels()}[short_name]


def loop_level_kernel(key: str) -> CorpusKernel:
    """The loop_level_reasoning kernel whose module is ``key``."""
    for kernel in iter_dace_kernels("loop_level_reasoning"):
        if kernel.short_name.rsplit("/", 1)[-1] == key:
            return kernel
    raise AssertionError(f"{key} is not in the loop_level_reasoning track; the corpus this test pins has changed")


def run(sdfg, inputs: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    """Run ``sdfg`` on copies of ``inputs`` with ``N=n``; returns the buffers after the call."""
    bufs = {k: v.copy() for k, v in inputs.items()}
    sdfg(**bufs, N=n)
    return bufs


def random_vectors(n: int = 48, names: tuple[str, ...] = ("a", "b", "c"), seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {k: rng.random(n) for k in names}


def signature_order(text: str, symbol: str, lang: str = "c") -> list[str]:
    """Parameter names of a translated kernel's entry, in declaration order: the emitted C order (sorted arrays,
    then symbols) is not the manifest's ``input_args`` order, so arguments bind to this."""
    if lang == "fortran":
        m = re.search(rf"subroutine\s+{re.escape(symbol)}\s*\((.*?)\)", text, re.DOTALL | re.IGNORECASE)
        if not m:
            raise LookupError(f"subroutine {symbol} not found")
        return [a.strip() for a in m.group(1).replace("&", " ").split(",") if a.strip()]
    params = raw_signature(text, symbol)
    return [p.strip().split()[-1].lstrip("*") for p in params.split(",") if p.strip() and p.strip() != "void"]


def scalar_ctype(sdfg: dace.SDFG, name: str) -> type[ctypes._SimpleCData]:
    """ctype of a by-value argument: a float is ``c_double``, any integer ``int64_t`` whatever its SDFG width,
    since a narrower c_int leaves the upper register half undefined."""
    if name in sdfg.symbols and np.dtype(sdfg.symbols[name].type).kind == "f":
        return ctypes.c_double
    return ctypes.c_int64


def c_argtypes(order: list[str], boundary: Boundary) -> list[CType]:
    """ctypes type per C parameter: an array is a pointer to its dtype, a float symbol a double, any other int64."""
    sdfg = boundary.standalone_sdfg
    return [
        ctypes.POINTER(CTYPE[np.dtype(sdfg.arrays[a].dtype.type).name]) if a in sdfg.arrays else scalar_ctype(sdfg, a)
        for a in order
    ]


def sdfg_to_python(sdfg: dace.SDFG, fn_name: str = "kernel") -> tuple[str, dace.SDFG]:
    """Standalone Python source for a whole SDFG, whose non-array arguments are its symbols, and the lowered copy it
    was emitted from (its descriptors size the scratch buffers the caller allocates). The symbols are every free
    symbol: ``arglist`` alone drops one only a library node's memlet step reads, which its expansion then indexes by."""
    arglist = sdfg.arglist(free_symbols=sdfg.free_symbols)
    symbols = [a for a in arglist if a not in sdfg.arrays]
    lowered = copy.deepcopy(sdfg)
    lower(lowered)
    widen_scratch(lowered, symbols)
    data_args = [a for a in arglist if a in sdfg.arrays]
    args = data_args + [s for s in scratch_arrays(lowered) if s not in data_args] + symbols
    return render(fn_name, args, lowered), lowered


def run_emitted(source: str, fn_name: str, lowered: dace.SDFG, inputs: dict, sizes: dict[str, int]) -> dict:
    """Call an emitted kernel C-style: every buffer its signature names is allocated here, from ``inputs`` (cast and
    reshaped to the descriptor; a scalar is a 1-element buffer) or zeroed; returns the buffers after the call."""
    kernel = vars(load_emitted(source, fn_name))[fn_name]
    env = {symbolic.symbol(k): v for k, v in sizes.items()}
    call = {}
    for name in inspect.signature(kernel).parameters:
        if name in sizes or name not in lowered.arrays:  # a symbol, by value
            call[name] = sizes[name] if name in sizes else inputs[name]
            continue
        desc = lowered.arrays[name]
        shape = tuple(int(symbolic.evaluate(d, env)) for d in desc.shape)
        dtype = np.dtype(desc.dtype.type)
        given = inputs.get(name)
        call[name] = np.zeros(shape, dtype) if given is None else np.array(given, dtype).reshape(shape)
    kernel(**call)
    return call


def fusion_moves(sdfg: dace.SDFG) -> list[tuple[str, tuple[str, ...]]]:
    """Every legal loop or map fusion of ``sdfg`` as ``(kind, labels)``, after making its tree labels unique."""
    normalize_labels(sdfg)
    return legal_moves(sdfg, "loop-fusion") + legal_moves(sdfg, "map-fusion")


def apply_move(sdfg: dace.SDFG, move: tuple[str, tuple[str, ...]]) -> str:
    """Apply one listed ``(kind, labels)`` move; returns the transformation that ran."""
    kind, labels = move
    rows = tree_rows(sdfg)
    plan = plan_move(kind, [rows[label] for label in labels])
    assert isinstance(plan, Rewrite), plan
    plan.commit()
    return plan.name


def fission_to_fixpoint(sdfg: dace.SDFG) -> int:
    """Apply the first legal loop or map fission until none remains; returns how many applied."""
    for applied in range(200):  # bound: each fission strictly adds a nest
        normalize_labels(sdfg)
        moves = legal_moves(sdfg, "loop-fission") + legal_moves(sdfg, "map-fission")
        if not moves:
            return applied
        apply_move(sdfg, moves[0])
    raise AssertionError("fission did not converge")


def extern_declaration_and_call(ext: ExternalCall, state: dace.SDFGState) -> tuple[str, str]:
    """The forward declaration and the call DaCe's ``ExternCall`` expansion emits for ``ext``."""
    tasklet = ExpandExternCall.expansion(ext, state, state.sdfg)
    return tasklet.code_global.as_string, tasklet.code.as_string
