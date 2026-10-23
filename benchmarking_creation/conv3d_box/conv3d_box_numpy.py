"""conv3d_box: non-separable 3-D box stencil with (2R+1)^3 taps on an edge-padded grid (numpy reference).

Derived from HPCAgent-Bench scientific_computing/structured_grids/conv_3d: the same weighted sum, with the edge
padding done by the input generator (``padded`` is the padded grid, ``K = 2R + 1``).
"""

import numpy as np


def conv3d_box(padded, w_box, out_grid, N, K):
    # array shapes (numpy->dace): padded=(N+K-1,N+K-1,N+K-1), w_box=(K,K,K), out_grid=(N,N,N)
    out_grid[:] = 0.0
    for di in range(K):
        for dj in range(K):
            for dk in range(K):
                out_grid += w_box[di, dj, dk] * padded[di:di + N, dj:dj + N, dk:dk + N]
