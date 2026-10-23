# hpcagent_bench-autogen -- generated from dbcsr_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
"""DaCe program auto-generated from the numpy reference by numpyto_c.dace_emit."""
import numpy as np
import dace as dc
from hpcagent_bench.frameworks.dace_framework import dc_float
import math
from math import sin, cos, log, exp, pow, sqrt

block_size = dc.symbol('block_size', dtype=dc.int64, positive=True)
n_block_rows = dc.symbol('n_block_rows', dtype=dc.int64, positive=True)
n_block_cols = dc.symbol('n_block_cols', dtype=dc.int64, positive=True)
n_block_inner = dc.symbol('n_block_inner', dtype=dc.int64, positive=True)

__hpcagent_bench_renames__ = {'product': '__product'}

__hpcagent_bench_program__ = 'dbcsr'


@dc.program
def dbcsr(a_index: dc.int32[n_block_rows * n_block_inner, 3], b_index: dc.int32[n_block_inner * n_block_cols, 3], a_blocks: dc_float[n_block_rows * n_block_inner, block_size, block_size], b_blocks: dc_float[n_block_inner * n_block_cols, block_size, block_size], m_sizes: dc.int32[n_block_rows], n_sizes: dc.int32[n_block_cols], k_sizes: dc.int32[n_block_inner], C: dc_float[n_block_rows * block_size, n_block_cols * block_size]):
    C[:, :] = 0.0
    row_offsets = np.zeros(n_block_rows + 1, dtype=np.int64)
    col_offsets = np.zeros(n_block_cols + 1, dtype=np.int64)
    for i in range(n_block_rows):
        row_offsets[i + 1] = row_offsets[i] + m_sizes[i]
    for j in range(n_block_cols):
        col_offsets[j + 1] = col_offsets[j] + n_sizes[j]
    for ia in range(n_block_rows * n_block_inner):
        a_bid = int(a_index[ia, 2])
        if a_bid < 0:
            continue
        a_row = int(a_index[ia, 0])
        a_k = int(a_index[ia, 1])
        for ib in range(n_block_inner * n_block_cols):
            b_bid = int(b_index[ib, 2])
            if b_bid < 0:
                continue
            b_k = int(b_index[ib, 0])
            if a_k != b_k:
                continue
            b_col = int(b_index[ib, 1])
            __product = a_blocks[a_bid] @ b_blocks[b_bid]
            for bi in range(block_size):
                for bj in range(block_size):
                    C[row_offsets[a_row] + bi, col_offsets[b_col] + bj] = C[row_offsets[a_row] + bi, col_offsets[b_col] + bj] + __product[bi, bj]
    return C
