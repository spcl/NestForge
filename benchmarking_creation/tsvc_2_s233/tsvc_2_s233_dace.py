# hpcagent_bench-autogen -- generated from tsvc_2_s233_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
"""DaCe program auto-generated from the numpy reference by numpyto_c.dace_emit."""
import numpy as np
import dace as dc
from hpcagent_bench.frameworks.dace_framework import dc_float
import math
from math import sin, cos, log, exp, pow, sqrt

LEN_2D = dc.symbol('LEN_2D', dtype=dc.int64, positive=True)

__hpcagent_bench_program__ = 'tsvc_2_s233'


@dc.program
def tsvc_2_s233(aa: dc_float[LEN_2D, LEN_2D], bb: dc_float[LEN_2D, LEN_2D], cc: dc_float[LEN_2D, LEN_2D]):
    for i in range(8, LEN_2D):
        for j in range(8, LEN_2D):
            aa[j, i] = aa[j - 1, i] + cc[j, i]
        for j in range(8, LEN_2D):
            bb[j, i] = bb[j, i - 1] + cc[j, i]
