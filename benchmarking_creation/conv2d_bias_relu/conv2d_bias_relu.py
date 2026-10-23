"""Input generation for conv2d_bias_relu (kept out of the numpy reference, as hpcagent_bench requires).

Normal inputs; weights scaled so the pre-activation sums stay O(1) and about half the outputs are clamped by ReLU.
"""

from typing import Optional

import numpy as np
from numpy.random import default_rng


def initialize(N, CI, CO, H, W, KH, KW, datatype=np.float64, rng: Optional[np.random.Generator] = None):
    if rng is None:
        rng = default_rng(7)
    x = rng.standard_normal((N, CI, H + KH - 1, W + KW - 1)).astype(datatype)
    wt = (rng.standard_normal((CO, CI, KH, KW)) / np.sqrt(CI * KH * KW)).astype(datatype)
    b = (rng.standard_normal(CO) / 4.0).astype(datatype)
    out = np.zeros((N, CO, H, W), dtype=datatype)
    return x, wt, b, out
