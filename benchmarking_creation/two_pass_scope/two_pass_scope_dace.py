"""two_pass_scope: hand-written DaCe program (no autogen marker, so it is a hand override).

The numpy reference has the same loops. The column loop carries a scalar through the map (the next value
reads ``t``), so the loop stays a loop. The row loop does not carry, and the two are separate blocks.
``carry`` holds the scalar after each column so the consumer reads an array, not a by-value scalar.
"""

import dace
import dace as dc
from hpcagent_bench.frameworks.dace_framework import dc_float

N = dc.symbol("N", dtype=dc.int64, positive=True)
M = dc.symbol("M", dtype=dc.int64, positive=True)

__hpcagent_bench_program__ = "two_pass_scope"


@dc.program
def two_pass_scope(
    a: dc_float[N, M],
    b: dc_float[N, M],
    w: dc_float[M],
    t: dc_float[N, M],
    out: dc_float[N, M],
    carry: dc_float[M],
):
    # The carried value lives in an array indexed by column, cs[j] before column j and cs[j + 1] after it: a scalar
    # (or a one-element array, which simplification turns back into one) shared by the parent and a GPU kernel is
    # passed to the kernel by value while the parent holds it on the device, which does not compile (2026-10-22/23).
    # The update is one tasklet: written as two statements, the frontend lowers each to a length-1 map.
    cs = dc.define_local([M + 1], dc.float64)
    cs[0] = 0.0
    for j in range(M):
        for i in dc.map[0:N]:
            t[i, j] = a[i, j] * cs[j] + b[i, j]
        with dace.tasklet:
            si << cs[j]
            t0 << t[0, j]
            wj << w[j]
            so >> cs[j + 1]
            cj >> carry[j]
            so = si + t0 + wj
            cj = si + t0 + wj
    for i in range(N):
        for j in dc.map[0:M]:
            out[i, j] = t[i, j] * t[i, j] - a[i, j] * carry[M - 1]
