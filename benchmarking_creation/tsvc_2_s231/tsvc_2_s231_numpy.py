# Adapted from TSVC_2 -- Test Suite for Vectorizing Compilers (github.com/UoB-HPC/TSVC_2),
# NCSA/MIT license (UIUC). Reimplemented in NumPy as the HPCAgent-Bench correctness reference.

"""TSVC tsvc_2 kernel ``s231`` (numpy reference).

Copied unchanged from HPCAgent-Bench `26a4f0cfc` (`benchmarks/loop_level_reasoning/tsvc_2_s231`); only the entry is renamed
from ``s231`` to ``tsvc_2_s231``, as repo-local kernels require."""


def tsvc_2_s231(aa, bb, LEN_2D):
    # array shapes (numpy->dace): aa=(LEN_2D,LEN_2D), bb=(LEN_2D,LEN_2D)
    for i in range(LEN_2D):
        for j in range(1, LEN_2D):
            aa[j, i] = aa[j - 1, i] + bb[j, i]
