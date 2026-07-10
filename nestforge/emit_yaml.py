"""Emit an OptArena ``BenchSpec`` manifest (symbols, array shapes, dtypes) for an extracted nest.

The manifest mirrors the fields OptArena's translator needs (verified against its ``gemm`` /
``vsumr`` kernels):
  * ``input_args``  -- the FULL positional signature of the numpy kernel (arrays then symbols),
  * ``array_args``  -- which of those are array pointers,
  * ``output_args`` -- which arrays are written,
  * ``init.arrays`` -- ``{name: {shape, dtype}}``,
  * ``parameters``  -- AOT representative sizes per symbol (preset -> {symbol: size}).
The numpy kernel's arg order (from :mod:`nestforge.emit_numpy`) must match ``input_args``.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import yaml

from dace import symbolic

from nestforge.extract import Boundary

DEFAULT_SIZE = 1 << 16


def _arg_order(boundary: Boundary) -> List[str]:
    """Arrays (inputs, then extra outputs), then symbols -- identical to the numpy signature."""
    args = list(boundary.inputs)
    args += [o for o in boundary.outputs if o not in boundary.inputs]
    args += [s for s in boundary.symbols if s not in args]
    return args


def _array_names(boundary: Boundary) -> List[str]:
    names = list(boundary.inputs)
    names += [o for o in boundary.outputs if o not in boundary.inputs]
    return names


def _shape_str(shape) -> str:
    dims = [symbolic.symstr(d) for d in shape]
    return "(" + ", ".join(dims) + ("," if len(dims) == 1 else "") + ")"


def _dtype_str(desc) -> str:
    return np.dtype(desc.dtype.type).name


def manifest_dict(boundary: Boundary,
                  name: str,
                  sizes: Optional[Dict[str, int]] = None,
                  preset: str = "S",
                  track: str = "foundation") -> Dict:
    """Build the OptArena manifest dict for ``boundary``'s standalone SDFG."""
    sdfg = boundary.standalone_sdfg
    arrays = _array_names(boundary)
    init_arrays = {a: {"shape": _shape_str(sdfg.arrays[a].shape), "dtype": _dtype_str(sdfg.arrays[a])} for a in arrays}
    sizes = sizes or {s: DEFAULT_SIZE for s in boundary.symbols}
    return {
        "name": name,
        "short_name": name,
        "func_name": name,
        "relative_path": "extended",
        "kind": "microkernel",
        "level": 1,
        "parameters": {
            preset: {
                s: int(sizes.get(s, DEFAULT_SIZE))
                for s in boundary.symbols
            }
        },
        "input_args": _arg_order(boundary),
        "array_args": arrays,
        "output_args": list(boundary.outputs),
        "init": {
            "arrays": init_arrays
        },
        "taxonomy": {
            "track": track
        },
    }


def manifest_yaml(boundary: Boundary, name: str, **kw) -> str:
    return yaml.safe_dump(manifest_dict(boundary, name, **kw), sort_keys=False)
