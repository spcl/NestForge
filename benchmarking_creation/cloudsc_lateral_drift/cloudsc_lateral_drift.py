"""Input generation for cloudsc_lateral_drift (kept out of the numpy reference, as hpcagent_bench requires).

Pressure must be strictly monotone down the column because the kernel divides by ``paph[jk+1] - paph[jk]``, and
temperature must stay well above zero; uniform random draws break both. ``cap`` is drawn small enough that the
``min`` limiter of the lateral drift binds in part of the updates (about 28% at preset S), so both of its branches
are exercised.
"""

from typing import Optional

import numpy as np
from numpy.random import default_rng


def initialize(NLEV, KLON, NCLDTOP, datatype=np.float64, rng: Optional[np.random.Generator] = None):
    if rng is None:
        rng = default_rng(0)
    dp = rng.uniform(50.0, 900.0, size=(NLEV + 1, KLON))
    paph = np.cumsum(dp, axis=0) + 1000.0  # half-level pressure (Pa), increasing downward
    pap = 0.5 * (paph[:-1] + paph[1:])  # full-level pressure
    ztp1 = rng.uniform(220.0, 300.0, size=(NLEV, KLON))  # temperature (K)
    zqr = rng.uniform(0.0, 1e-3, size=(NLEV, KLON))  # rain mixing ratio
    pfsnow = rng.uniform(0.0, 2e-4, size=(NLEV + 1, KLON))  # snow flux
    pfplsx = np.zeros((NLEV + 1, KLON))  # rain flux, zero above the cloud top
    cap = rng.uniform(1e-6, 2e-4, size=(NLEV + 1, KLON))  # lateral limiter
    east = (rng.uniform(size=NLEV + 1) < 0.5).astype(np.int32)  # drift direction per level
    z = np.zeros((NLEV + 1, KLON))
    f = lambda a: np.ascontiguousarray(a, dtype=datatype)  # noqa: E731
    return (f(pap), f(paph), f(ztp1), f(zqr), f(pfsnow), f(pfplsx), f(z), f(z), f(z), f(z), f(cap), east)
