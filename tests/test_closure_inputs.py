# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A subscripted NumPy call on constants (cp2k_density_matrix_trs4's ``np.arange(3, dtype=np.int64)[None, :]``) is
evaluated when the program is parsed and becomes a closure array, an argument of the SDFG that the bench's inputs do
not name: ``closure_inputs`` binds its value, so the whole program does not run on zeros."""

import dace
import numpy as np
from helpers import build_sdfg

from nestforge.server import closure_inputs
from nestforge.stages.performance import program_buffers

#: Rows and columns of the result; the closure is ``np.arange(3)``.
N = 3


@dace.program
def add_offsets(a: dace.int64[N], b: dace.int64[N, N]):
    b[:] = a[:, None] + np.arange(3, dtype=np.int64)[None, :]


def test_a_constant_the_program_computed_at_parse_time_is_bound_with_its_value():
    sdfg = add_offsets.to_sdfg()

    closure = closure_inputs(add_offsets, sdfg)

    assert len(closure) == 1
    ((name, value),) = closure.items()
    assert name in sdfg.arrays and not sdfg.arrays[name].transient
    np.testing.assert_array_equal(value, np.arange(N)[None, :])


def test_the_whole_program_runs_on_the_closure_value_not_zeros(tmp_path):
    sdfg = add_offsets.to_sdfg()
    a = np.array([10, 20, 30], dtype=np.int64)
    buffers = program_buffers(sdfg, {"a": a} | closure_inputs(add_offsets, sdfg), {})

    build_sdfg(sdfg, tmp_path).run(buffers, {})

    np.testing.assert_array_equal(buffers["b"].reshape(N, N), a[:, None] + np.arange(N)[None, :])
