# hpcagent_bench-autogen -- generated from column_physics_cpu_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
"""DaCe program auto-generated from the numpy reference by numpyto_c.dace_emit."""
import numpy as np
import dace as dc
from hpcagent_bench.frameworks.dace_framework import dc_float
import math
from math import sin, cos, log, exp, pow, sqrt

NLEV = dc.symbol('NLEV', dtype=dc.int64, positive=True)
KLON = dc.symbol('KLON', dtype=dc.int64, positive=True)
NTRAC = dc.symbol('NTRAC', dtype=dc.int64, positive=True)

__hpcagent_bench_program__ = 'column_physics_cpu'


@dc.program
def column_physics_cpu(tr: dc_float[NLEV, NTRAC, KLON], src: dc_float[NLEV, KLON], fall: dc_float[NTRAC], za: dc_float[NLEV, KLON], pcov: dc_float[NLEV, KLON], covc: dc_float[KLON], out: dc_float[NLEV, KLON], cap: dc_float):
    for jl in range(KLON):
        covc[jl] = 0.0
        for jk in range(1, NLEV):
            for jm in range(NTRAC):
                tr[jk, jm, jl] = min(tr[jk - 1, jm, jl] * fall[jm] + src[jk, jl], cap)
            cov = max(covc[jl], za[jk, jl])
            covc[jl] = cov
            pcov[jk, jl] = cov
    for jk in range(1, NLEV):
        out[jk, :] = np.minimum(out[jk - 1, :] * 0.5 + pcov[jk, :] * tr[jk, 0, :], cap)
