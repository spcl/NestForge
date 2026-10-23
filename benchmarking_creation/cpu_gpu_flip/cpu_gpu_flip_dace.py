# hpcagent_bench-autogen -- generated from cpu_gpu_flip_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
"""DaCe program auto-generated from the numpy reference by numpyto_c.dace_emit."""
import numpy as np
import dace as dc
from hpcagent_bench.frameworks.dace_framework import dc_float
import math
from math import sin, cos, log, exp, pow, sqrt

nlev = dc.symbol('nlev', dtype=dc.int64, positive=True)
ncol = dc.symbol('ncol', dtype=dc.int64, positive=True)

__hpcagent_bench_program__ = 'cpu_gpu_flip'


@dc.program
def cpu_gpu_flip(t: dc_float[nlev, ncol], q: dc_float[nlev, ncol], p: dc_float[nlev, ncol], x: dc_float[nlev, ncol]):
    zqs = np.empty((nlev, ncol), dtype=t.dtype)
    zsm = np.empty((nlev, ncol), dtype=t.dtype)
    cp = np.empty((nlev, ncol), dtype=t.dtype)
    dp = np.empty((nlev, ncol), dtype=t.dtype)
    zqs[:, :] = 380.1608703442847 * np.exp(17.502 * (t - 273.16) / (t - 32.19)) / p
    zsm[:, 1:ncol - 1] = 0.25 * zqs[:, 0:ncol - 2] + 0.5 * zqs[:, 1:ncol - 1] + 0.25 * zqs[:, 2:ncol]
    zsm[:, 0] = zqs[:, 0]
    zsm[:, ncol - 1] = zqs[:, ncol - 1]
    cp[0, :] = -10.0 * zsm[1, :] / (1.0 + 10.0 * zsm[1, :])
    dp[0, :] = (q[0, :] - zsm[0, :]) / (1.0 + 10.0 * zsm[1, :])
    for k in range(1, nlev - 1):
        m = 1.0 + 10.0 * zsm[k, :] + 10.0 * zsm[k + 1, :] + 10.0 * zsm[k, :] * cp[k - 1, :]
        cp[k, :] = -10.0 * zsm[k + 1, :] / m
        dp[k, :] = (q[k, :] - zsm[k, :] + 10.0 * zsm[k, :] * dp[k - 1, :]) / m
    m__v2 = 1.0 + 10.0 * zsm[nlev - 1, :] + 10.0 * zsm[nlev - 1, :] * cp[nlev - 2, :]
    dp[nlev - 1, :] = (q[nlev - 1, :] - zsm[nlev - 1, :] + 10.0 * zsm[nlev - 1, :] * dp[nlev - 2, :]) / m__v2
    x[nlev - 1, :] = dp[nlev - 1, :]
    for k in range(nlev - 2, -1, -1):
        x[k, :] = dp[k, :] - cp[k, :] * x[k + 1, :]
