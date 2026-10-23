"""seissol_volume: SeisSol ADER-DG volume-integral update of one time step (numpy reference).

Derived from HPCAgent-Bench scientific_computing/dense_linear_algebra/seissol_tensor_contraction: the same contraction
``upd[b, k, p] = sum_{d, l, q} kDivM[d, k, l] * I[b, l, q] * star[d, q, p]`` (einsum "dkl,blq,dqp->bkp"), applied as the
time step applies it, ``Q += dt * upd``. b: elements, k and l: basis functions, p and q: quantities, d: dimensions.
"""

import numpy as np


def seissol_volume(dt, Q, upd, I, kDivM, star, NB, NBF, NQ, ND):
    # array shapes (numpy->dace): Q=(NB,NBF,NQ), upd=(NB,NBF,NQ), I=(NB,NBF,NQ), kDivM=(ND,NBF,NBF), star=(ND,NQ,NQ)
    upd[:] = np.einsum("dkl,blq,dqp->bkp", kDivM, I, star, optimize=True)
    Q[:] = Q + dt * upd
