"""conv3d_box: hand-written DaCe program (no autogen marker, so it is a hand override).

Output-stationary loop order: one grid point, zeroed, accumulated over the tap loops ``di, dj, dk`` in place. The zero
init keeps the nest from being lifted to one library node; the in-place accumulation keeps ``di`` and ``dj`` as loops.
"""

import dace as dc
from hpcagent_bench.frameworks.dace_framework import dc_float

N = dc.symbol("N", dtype=dc.int64, positive=True)
K = dc.symbol("K", dtype=dc.int64, positive=True)

__hpcagent_bench_program__ = "conv3d_box"


@dc.program
def conv3d_box(padded: dc_float[N + K - 1, N + K - 1, N + K - 1], w_box: dc_float[K, K, K], out_grid: dc_float[N, N, N]):
    for i in range(N):
        for j in range(N):
            for k in range(N):
                out_grid[i, j, k] = 0.0
                for di in range(K):
                    for dj in range(K):
                        for dk in range(K):
                            out_grid[i, j, k] = out_grid[i, j, k] + w_box[di, dj, dk] * padded[i + di, j + dj, k + dk]
