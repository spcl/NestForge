# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The argument manifest of an extracted nest (symbols, array shapes, dtypes), in the fields HPCAgent-Bench's
translator reads."""

from __future__ import annotations

from typing import Any
from collections.abc import Sequence

import numpy as np

import dace
from dace import symbolic

from nestforge.ir.emit_numpy import kernel_args, kernel_arrays, sized_standalone
from nestforge.ir.extract import Boundary

DEFAULT_SIZE = 1 << 16

#: Symbol dtypes that make a value, not a size.
FLOAT_DTYPES = frozenset({"float64", "float32", "float16", "float128"})


def symbol_dtype_name(sdfg: dace.SDFG, s: str) -> str:
    if s in sdfg.symbols:
        return np.dtype(sdfg.symbols[s].type).name
    return "int64"


def shape_str(shape: Sequence[Any]) -> str:
    dims = [symbolic.symstr(d) for d in shape]
    return "(" + ", ".join(dims) + ("," if len(dims) == 1 else "") + ")"


def dtype_str(desc: dace.data.Data) -> str:
    return np.dtype(desc.dtype.type).name


def manifest_dict(
    boundary: Boundary, name: str, sizes: dict[str, int] | None = None, preset: str = "S"
) -> dict[str, Any]:
    """The manifest of ``boundary``'s standalone SDFG."""
    sdfg = sized_standalone(boundary)
    arrays = kernel_arrays(boundary, sdfg)
    init_arrays: dict[str, dict[str, str]] = {}
    for a in arrays:
        desc = sdfg.arrays[a]
        init_arrays[a] = {"shape": shape_str(desc.shape), "dtype": dtype_str(desc)}
    sizes = sizes or dict.fromkeys(boundary.symbols, DEFAULT_SIZE)
    int_params: dict[str, int] = {}
    float_scalars: dict[str, float] = {}
    for s in boundary.symbols:
        # a float symbol is a staged value; the translator must declare it double, not int64
        if symbol_dtype_name(sdfg, s) in FLOAT_DTYPES:
            float_scalars[s] = 0.0
        else:
            int_params[s] = int(sizes.get(s, DEFAULT_SIZE))
    init: dict[str, Any] = {"arrays": init_arrays}
    if float_scalars:
        init["scalars"] = float_scalars
    return {
        "name": name,
        "func_name": name,
        "relative_path": "extended",
        "level": 1,
        "parameters": {preset: int_params},
        "input_args": kernel_args(boundary, arrays),
        "array_args": arrays,
        "output_args": list(boundary.outputs),
        "init": init,
    }
