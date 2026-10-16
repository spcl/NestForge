# hpcagent_bench-autogen -- generated from tsvc_2_s231_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
"""DaCe program auto-generated from the numpy reference by numpyto_c.dace_emit."""
import numpy as np
import dace as dc
from hpcagent_bench.frameworks.dace_framework import dc_float
import math
from math import sin, cos, log, exp, pow, sqrt

LEN_2D = dc.symbol('LEN_2D', dtype=dc.int64, positive=True)

__hpcagent_bench_program__ = 'tsvc_2_s231'


@dc.program
def tsvc_2_s231(aa: dc_float[LEN_2D, LEN_2D], bb: dc_float[LEN_2D, LEN_2D]):
    for i in range(LEN_2D):
        for j in range(1, LEN_2D):
            aa[j, i] = aa[j - 1, i] + bb[j, i]
