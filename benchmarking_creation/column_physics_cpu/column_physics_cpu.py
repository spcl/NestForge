"""Input generation for column_physics_cpu (kept out of the numpy reference, as hpcagent_bench requires).

``cap`` is fixed at 0.6, below what the tracer recurrence reaches from these inputs (up to 1.1), so its ``min``
clamp binds in part of the updates and both branches are exercised. Outputs start at zero except ``tr`` and ``out``,
whose first level is the recurrence's initial value.
"""

from typing import Optional

import numpy as np
from numpy.random import default_rng


def initialize(NLEV, KLON, NTRAC, datatype=np.float64, rng: Optional[np.random.Generator] = None):
    if rng is None:
        rng = default_rng(0)
    tr = rng.uniform(0.0, 1.0, size=(NLEV, NTRAC, KLON)).astype(datatype)
    src = rng.uniform(0.0, 0.1, size=(NLEV, KLON)).astype(datatype)
    fall = rng.uniform(0.5, 1.0, size=(NTRAC,)).astype(datatype)
    za = rng.uniform(0.0, 1.0, size=(NLEV, KLON)).astype(datatype)
    pcov = np.zeros((NLEV, KLON), dtype=datatype)
    covc = np.zeros((KLON,), dtype=datatype)
    out = rng.uniform(0.0, 1.0, size=(NLEV, KLON)).astype(datatype)
    cap = datatype(0.6)
    return tr, src, fall, za, pcov, covc, out, cap
