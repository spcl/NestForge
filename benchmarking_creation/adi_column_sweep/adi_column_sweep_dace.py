"""adi_column_sweep: hand-written DaCe program (no hpcagent_bench-autogen marker, so it is a hand override).

Same arithmetic as HPCAgent-Bench e63f9f1 adi (x-dominant configuration b1 = 2, b2 = 1, the oracle's defaults), with
every row slice written as an explicit loop over i, so each sweep is one map under its j recurrence instead of one map
per elementwise operation. The column sweep's two recurrences are written as two loops (p first, then q, which reads
p[i, j - 1] and recomputes the shared denominator), so each column sweep is one map under its own loop; written as one
loop, stage 1 splits it into two maps that stage 2 can neither fuse nor interchange. Every sweep keeps the source order
`for j { for i }`, which walks p, q and u with stride N. The column boundary values (u[i, 0], p[i, 0], q[i, 0] and
u[i, N - 1]) are set in one loop before the p sweep, which reads none of them but p[i, 0]. M6 entry, 2026-10-22."""
import numpy as np
import dace as dc

TSTEPS = dc.symbol('TSTEPS', dtype=dc.int64, positive=True)
N = dc.symbol('N', dtype=dc.int64, positive=True)

__hpcagent_bench_program__ = 'adi_column_sweep'


@dc.program
def adi_column_sweep(u: dc.float64[N, N]):
    v = np.empty((N, N), dtype=u.dtype)
    p = np.empty((N, N), dtype=u.dtype)
    q = np.empty((N, N), dtype=u.dtype)
    pt = np.empty((N, N), dtype=u.dtype)
    qt = np.empty((N, N), dtype=u.dtype)
    DX = 1.0 / N
    DY = 1.0 / N
    DT = 1.0 / TSTEPS
    mul1 = 2.0 * DT / (DX * DX)
    mul2 = 1.0 * DT / (DY * DY)
    a = -mul1 / 2.0
    b = 1.0 + mul1
    c = a + 0.0
    d = -mul2 / 2.0
    e = 1.0 + mul2
    f = d + 0.0
    for _ in range(1, TSTEPS + 1):
        # row sweep: the recurrence runs over rows j, each step a contiguous row of pt, qt, v
        for i in range(1, N - 1):
            v[0, i] = 1.0
            pt[0, i] = 0.0
            qt[0, i] = 1.0
            v[N - 1, i] = 1.0
        for j in range(1, N - 1):
            for i in range(1, N - 1):
                inv = 1.0 / (a * pt[j - 1, i] + b)
                qt[j, i] = (-d * u[j, i - 1] + (1.0 + 2.0 * d) * u[j, i] - f * u[j, i + 1] - a * qt[j - 1, i]) * inv
                pt[j, i] = -c * inv
        for j in range(N - 2, 0, -1):
            for i in range(1, N - 1):
                v[j, i] = pt[j, i] * v[j + 1, i] + qt[j, i]
        # column sweep: the recurrence runs over columns j, each step a strided column of p, q, u
        for i in range(1, N - 1):
            u[i, 0] = 1.0
            p[i, 0] = 0.0
            q[i, 0] = 1.0
            u[i, N - 1] = 1.0
        for j in range(1, N - 1):
            for i in range(1, N - 1):
                p[i, j] = -f / (d * p[i, j - 1] + e)
        for j in range(1, N - 1):
            for i in range(1, N - 1):
                q[i, j] = (-a * v[i - 1, j] + (1.0 + 2.0 * a) * v[i, j] - c * v[i + 1, j] - d * q[i, j - 1]) / (d * p[i, j - 1] + e)
        for j in range(N - 2, 0, -1):
            for i in range(1, N - 1):
                u[i, j] = p[i, j] * u[i, j + 1] + q[i, j]
