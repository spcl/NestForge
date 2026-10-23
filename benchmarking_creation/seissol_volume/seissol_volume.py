"""Input generation for seissol_volume (kept out of the numpy reference, as hpcagent_bench requires)."""

from typing import Optional

import numpy as np
from numpy.random import default_rng


def initialize(NB, NBF, NQ, ND, datatype=np.float64, rng: Optional[np.random.Generator] = None):
    if rng is None:
        rng = default_rng(7)
    dt = datatype(0.01)
    Q = rng.standard_normal((NB, NBF, NQ)).astype(datatype)
    upd = np.zeros((NB, NBF, NQ), dtype=datatype)
    I = rng.standard_normal((NB, NBF, NQ)).astype(datatype)
    kDivM = (rng.standard_normal((ND, NBF, NBF)) / NBF).astype(datatype)
    star = rng.standard_normal((ND, NQ, NQ)).astype(datatype)
    return dt, Q, upd, I, kDivM, star
