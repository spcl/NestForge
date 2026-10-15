# hpcagent_bench-autogen -- generated from sweep_scale_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
"""DaCe program auto-generated from the numpy reference by numpyto_c.dace_emit."""
import numpy as np
import dace as dc
from hpcagent_bench.frameworks.dace_framework import dc_float
import math
from math import sin, cos, log, exp, pow, sqrt

NJ = dc.symbol('NJ', dtype=dc.int64, positive=True)
NI = dc.symbol('NI', dtype=dc.int64, positive=True)

__hpcagent_bench_program__ = 'sweep_scale'


@dc.program
def sweep_scale(aa: dc_float[NJ, NI], bb: dc_float[NJ, NI], cc: dc_float[NJ, NI], w: dc_float[NI]):
    for i in range(NI):
        for j in range(1, NJ):
            aa[j, i] = aa[j - 1, i] + bb[j, i]
    for i in range(NI):
        for j in range(NJ):
            cc[j, i] = aa[j, i] * w[i]
