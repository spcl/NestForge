"""two_pass_scope: two grid passes the moves cannot fuse (numpy reference).

Synthetic (slot M7). Arrays are (N, M) with M contiguous. The producer walks columns. Each column is scaled
by a carried scalar, then that scalar is updated from the column just written, and the value is stored in
``carry``. The consumer walks rows and reads the finished ``t`` and the last ``carry``. The loops have
different ranges (M and N) and are separate blocks. Map fusion does not cross that boundary. The scheduling
problem is to put both passes in one kernel.
"""

import numpy as np


def two_pass_scope(a, b, w, t, out, carry, N, M):
    # array shapes (numpy->dace): a=(N,M), b=(N,M), w=(M,), t=(N,M), out=(N,M), carry=(M,)
    s = 0.0
    for j in range(M):
        for i in range(N):
            t[i, j] = a[i, j] * s + b[i, j]
        s = s + t[0, j] + w[j]
        carry[j] = s
    last = carry[M - 1]
    for i in range(N):
        for j in range(M):
            out[i, j] = t[i, j] * t[i, j] - a[i, j] * last
