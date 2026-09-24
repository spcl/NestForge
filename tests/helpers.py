# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Helpers several test modules share."""

import ctypes
import re

import numpy as np

import dace

from nestforge.build.toolchain import CType, raw_signature
from nestforge.corpus.bench import CorpusKernel, iter_dace_kernels
from nestforge.ir.extract import Boundary

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
        m = re.search(rf"subroutine\s+{re.escape(symbol)}\s*\((.*?)\)", text, re.S | re.I)
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
