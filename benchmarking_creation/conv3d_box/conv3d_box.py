"""Input generation for conv3d_box: a uniform grid, edge-padded by R = (K - 1) // 2 on every side as conv_3d pads it."""

from typing import Optional

import numpy as np
from numpy.random import default_rng


def initialize(N, K, datatype=np.float64, rng: Optional[np.random.Generator] = None):
    if rng is None:
        rng = default_rng(7)
    R = (K - 1) // 2
    padded = np.pad(rng.random((N, N, N)), R, mode="edge").astype(datatype)
    w_box = (rng.random((K, K, K)) / K ** 3).astype(datatype)
    out_grid = np.zeros((N, N, N), dtype=datatype)
    return padded, w_box, out_grid
