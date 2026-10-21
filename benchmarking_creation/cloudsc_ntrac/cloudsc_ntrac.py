"""Input generation for cloudsc_ntrac (kept out of the numpy reference, as hpcagent_bench requires).

Structured inputs, so that every branch of the kernel is exercised:

- ``vfall`` spreads the tracers over CloudSC's fall speeds (``ZVQX``: ice 0.13, snow 1.0, rain 4.0 m/s); tracer 2 is
  rain (4.0) and tracer 3 is snow (1.0).
- ``dtgdp`` is ``dt * g / dp`` for a 3600 s step and a layer depth between about 450 and 1800 Pa, and ``rdtgdp`` is
  its inverse, as in CloudSC; ``rho`` grows from 0.1 at the top to 1.2 kg/m3 at the surface.
- About a third of the columns carry no rain or snow at all: ``prec`` (last step's rain plus snow flux) is 0 there
  and the precipitation gate sends the cover to 0; elsewhere ``prec`` is positive and the cover is clamped from below
  at 0.1.
- About 10% of the ``q`` values are slightly negative (as CloudSC's numerics produce), so the ``max`` clamp on the
  new tracer value binds.
- Level 0 of ``pflux`` and ``pcov`` is the inflow from above; the outputs start at 0 elsewhere.
"""

from typing import Optional

import numpy as np
from numpy.random import default_rng


def initialize(NLEV, KLON, NTRAC, datatype=np.float64, rng: Optional[np.random.Generator] = None):
    if rng is None:
        rng = default_rng(0)
    speeds = np.linspace(0.13, 4.0, NTRAC)
    speeds[2], speeds[3] = 4.0, 1.0
    vfall = speeds.astype(datatype)
    dp = rng.uniform(450.0, 1800.0, size=(NLEV, KLON))
    dtgdp = (3600.0 * 9.80665 / dp).astype(datatype)
    rdtgdp = (1.0 / dtgdp).astype(datatype)
    profile = np.linspace(0.1, 1.2, NLEV)[:, None]
    rho = (profile * rng.uniform(0.95, 1.05, size=(NLEV, KLON))).astype(datatype)
    za = rng.uniform(0.0, 1.0, size=(NLEV, KLON)).astype(datatype)
    q = (rng.uniform(-0.1, 1.0, size=(NTRAC, NLEV, KLON)) * 1e-3).astype(datatype)
    pflux = np.zeros((NTRAC, NLEV, KLON), dtype=datatype)
    pflux[:, 0, :] = rng.uniform(0.0, 1e-4, size=(NTRAC, KLON))
    dry = rng.uniform(0.0, 1.0, size=KLON) < 1.0 / 3.0
    for jm in (2, 3):
        q[jm, :, dry] = 0.0
        pflux[jm, 0, dry] = 0.0
    prec = rng.uniform(1e-6, 1e-4, size=(NLEV, KLON)).astype(datatype)
    prec[:, dry] = 0.0
    pcov = np.zeros((NLEV, KLON), dtype=datatype)
    pcov[0, :] = rng.uniform(0.0, 1.0, size=KLON)
    return q, pflux, vfall, dtgdp, rdtgdp, rho, za, prec, pcov
