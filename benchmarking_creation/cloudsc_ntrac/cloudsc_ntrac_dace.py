# hpcagent_bench-autogen -- generated from cloudsc_ntrac_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
"""DaCe program auto-generated from the numpy reference by numpyto_c.dace_emit."""
import numpy as np
import dace as dc
from hpcagent_bench.frameworks.dace_framework import dc_float
import math
from math import sin, cos, log, exp, pow, sqrt

NLEV = dc.symbol('NLEV', dtype=dc.int64, positive=True)
KLON = dc.symbol('KLON', dtype=dc.int64, positive=True)
NTRAC = dc.symbol('NTRAC', dtype=dc.int64, positive=True)

__hpcagent_bench_program__ = 'cloudsc_ntrac'


@dc.program
def cloudsc_ntrac(q: dc_float[NTRAC, NLEV, KLON], pflux: dc_float[NTRAC, NLEV, KLON], vfall: dc_float[NTRAC], dtgdp: dc_float[NLEV, KLON], rdtgdp: dc_float[NLEV, KLON], rho: dc_float[NLEV, KLON], za: dc_float[NLEV, KLON], prec: dc_float[NLEV, KLON], pcov: dc_float[NLEV, KLON]):
    for jk in range(1, NLEV):
        for jm in range(NTRAC):
            for jl in range(KLON):
                src = pflux[jm, jk - 1, jl] * dtgdp[jk, jl]
                sink = dtgdp[jk, jl] * vfall[jm] * rho[jk, jl]
                qn = max((q[jm, jk, jl] + src) / (1.0 + sink), 0.0)
                pflux[jm, jk, jl] = sink * qn * rdtgdp[jk, jl]
        for jl in range(KLON):
            cov = 1.0 - (1.0 - pcov[jk - 1, jl]) * (1.0 - max(za[jk, jl], za[jk - 1, jl])) / (1.0 - min(za[jk - 1, jl], 1.0 - 1e-06))
            pcov[jk, jl] = max(cov, 0.1) * min(prec[jk, jl] * 100000000000000.0, 1.0)
