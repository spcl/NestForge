# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A rewritten loop nest of the corpus kernel ``conv_3d``: same arguments, inputs and oracle.

From HPCAgent-Bench branch ``benchmarking-nestforge`` (``d22922c5c``), kernel ``conv3d_box``, which took an already
padded grid. Here the program pads ``in_grid`` at its edges itself, as ``conv_3d`` does, then runs the box stencil
output-stationary: one grid point, zeroed, accumulated over the tap loops ``di, dj, dk`` in place. The zero init
keeps the nest from being lifted to one library node.
"""

import dace as dc
from hpcagent_bench.frameworks.dace_framework import dc_float

N = dc.symbol("N", dtype=dc.int64, positive=True)
R = dc.symbol("R", dtype=dc.int64, positive=True)


@dc.program
def conv_3d(in_grid: dc_float[N, N, N], out_grid: dc_float[N, N, N], w_box: dc_float[2 * R + 1, 2 * R + 1, 2 * R + 1]):
    padded = dc.define_local([N + 2 * R, N + 2 * R, N + 2 * R], dc_float)
    for pi in range(N + 2 * R):
        for pj in range(N + 2 * R):
            for pk in range(N + 2 * R):
                padded[pi, pj, pk] = in_grid[
                    min(max(pi - R, 0), N - 1), min(max(pj - R, 0), N - 1), min(max(pk - R, 0), N - 1)
                ]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                out_grid[i, j, k] = 0.0
                for di in range(2 * R + 1):
                    for dj in range(2 * R + 1):
                        for dk in range(2 * R + 1):
                            out_grid[i, j, k] = out_grid[i, j, k] + w_box[di, dj, dk] * padded[i + di, j + dj, k + dk]
