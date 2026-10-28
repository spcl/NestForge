# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A rewritten loop nest of the corpus kernel ``seissol_tensor_contraction``: same arguments, inputs and oracle.

From HPCAgent-Bench branch ``benchmarking-nestforge`` (``d22922c5c``), kernel ``seissol_volume``, which also took a
time step ``dt`` and returned the update. Here ``Q += upd`` as the corpus kernel computes it. Output-stationary: one
entry ``upd[b, k, p]``, zeroed, accumulated over the summed loops ``d, l, q`` in place, then added to ``Q``. The zero
init and the separate update keep it a loop nest rather than one einsum library node. The basis, quantity and
dimension counts are symbols, as in ``seissol_volume`` (with them as constants, replay at preset S stopped in
NestForge's kernel emitter on a write-conflict nested SDFG); the solution script binds them to the corpus values.
"""

import dace as dc
from hpcagent_bench.frameworks.dace_framework import dc_float

batch = dc.symbol("batch", dtype=dc.int64, positive=True)
NBF = dc.symbol("NBF", dtype=dc.int64, positive=True)
NQ = dc.symbol("NQ", dtype=dc.int64, positive=True)
ND = dc.symbol("ND", dtype=dc.int64, positive=True)


@dc.program
def kernel(
    Q: dc_float[batch, NBF, NQ], I: dc_float[batch, NBF, NQ], kDivM: dc_float[ND, NBF, NBF], star: dc_float[ND, NQ, NQ]
):
    upd = dc.define_local([batch, NBF, NQ], dc_float)
    for b in range(batch):
        for k in range(NBF):
            for p in range(NQ):
                upd[b, k, p] = 0.0
                for d in range(ND):
                    for l in range(NBF):
                        for q in range(NQ):
                            upd[b, k, p] = upd[b, k, p] + kDivM[d, k, l] * I[b, l, q] * star[d, q, p]
                Q[b, k, p] = Q[b, k, p] + upd[b, k, p]
