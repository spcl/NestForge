# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The manifest's ``input_args`` is the emitted NumPy kernel's positional signature.

The translator declares C parameters by walking ``input_args``, so a name missing there is never declared. Scratch
transients are caller-allocated parameters between the outputs and the symbols, and must sit there in both.
"""

import ast

import numpy as np

import dace

from nestforge.ir.emit_numpy import scratch_arrays
from nestforge.ir.emit_yaml import manifest_dict
from nestforge.ir.extract import Boundary, detach

from helpers import sdfg_to_numpy

N = dace.symbol("N")


@dace.program
def two_nest(a: dace.float64[N], out: dace.float64[N]):
    """Two maps chained through a non-scalar transient -- ``tmp`` stays internal (it is neither an input
    nor an output of the program), so it reaches the emitter as a scratch buffer."""
    tmp = np.empty(N, dace.float64)
    for i in dace.map[0:N]:
        tmp[i] = a[i] * 2.0
    for i in dace.map[0:N]:
        out[i] = tmp[i] + 1.0


def whole_program_boundary(sdfg: dace.SDFG) -> Boundary:
    """A :class:`Boundary` over the whole program: its non-transient arrays read and written, and its symbols."""
    detached = detach(sdfg)
    read, write = detached.read_and_write_sets()
    arrays = {n for n, desc in detached.arrays.items() if not desc.transient}
    inputs = sorted(a for a in arrays if a in read)
    outputs = sorted(a for a in arrays if a in write)
    symbols = [a for a in detached.arglist() if a not in detached.arrays]
    return Boundary(
        inputs=inputs,
        outputs=outputs,
        symbols=symbols,
        nsdfg_node=None,
        state=None,
        standalone_sdfg=detached,
    )


def signature_of(source: str) -> list:
    """Positional parameter names of the single ``def`` in emitted numpy source."""
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
    return [a.arg for a in fn.args.args]


def scratch_boundary():
    return whole_program_boundary(two_nest.to_sdfg(simplify=True))


def test_scratch_transient_reaches_the_emitter():
    """Guards the fixture itself: if simplify ever folds ``tmp`` away, the tests below stop testing."""
    boundary = scratch_boundary()
    assert scratch_arrays(boundary.standalone_sdfg) == ["tmp"]
    assert "tmp" not in boundary.inputs and "tmp" not in boundary.outputs


def test_input_args_equals_numpy_signature_with_scratch():
    boundary = scratch_boundary()
    manifest = manifest_dict(boundary, "two_nest")
    emitted = signature_of(sdfg_to_numpy(boundary.standalone_sdfg, fn_name="two_nest"))
    assert emitted == ["a", "out", "tmp", "N"]  # arrays, scratch, then symbols
    assert list(manifest["input_args"]) == emitted


def test_scratch_is_declared_as_an_allocatable_array():
    """A scratch buffer in ``input_args`` must also be typed as an array, or the translator reads it as a
    scalar parameter; its shape must be resolvable from the kernel's own size symbols."""
    manifest = manifest_dict(scratch_boundary(), "two_nest")
    assert "tmp" in manifest["array_args"]
    assert manifest["init"]["arrays"]["tmp"] == {"shape": "(N,)", "dtype": "float64"}
    assert "tmp" not in manifest["output_args"]  # internal, not a result the caller reads back
