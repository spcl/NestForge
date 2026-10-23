"""conv2d_bias_relu: hand-written DaCe program (no autogen marker, so it is a hand override).

Output-stationary loop order, as a direct or thread-per-output convolution is written: one output element, the bias
init, the three summed loops (channel, kernel row, kernel column) accumulating into ``out`` in place, then ReLU.
Accumulating into the array keeps the channel and kernel-row loops as loops after stage 1 (only the innermost sum
becomes a reduction map), and the init and ReLU around them stop the frontend from lifting the nest to an einsum.
"""

import dace as dc
from hpcagent_bench.frameworks.dace_framework import dc_float

N = dc.symbol("N", dtype=dc.int64, positive=True)
CI = dc.symbol("CI", dtype=dc.int64, positive=True)
CO = dc.symbol("CO", dtype=dc.int64, positive=True)
H = dc.symbol("H", dtype=dc.int64, positive=True)
W = dc.symbol("W", dtype=dc.int64, positive=True)
KH = dc.symbol("KH", dtype=dc.int64, positive=True)
KW = dc.symbol("KW", dtype=dc.int64, positive=True)

__hpcagent_bench_program__ = "conv2d_bias_relu"


@dc.program
def conv2d_bias_relu(x: dc_float[N, CI, H + KH - 1, W + KW - 1], wt: dc_float[CO, CI, KH, KW], b: dc_float[CO],
                     out: dc_float[N, CO, H, W]):
    for n in range(N):
        for co in range(CO):
            for oh in range(H):
                for ow in range(W):
                    out[n, co, oh, ow] = b[co]
                    for ci in range(CI):
                        for kh in range(KH):
                            for kw in range(KW):
                                out[n, co, oh, ow] = out[n, co, oh, ow] + x[n, ci, oh + kh, ow + kw] * wt[co, ci, kh, kw]
                    out[n, co, oh, ow] = max(out[n, co, oh, ow], 0.0)
