"""column_physics_cpu: a column model written in the GPU order, to be scheduled back to the CPU order (numpy reference).

Synthetic (slot H3 of the thesis benchmark set). Each column ``jl`` runs a cloud-cover recurrence down the levels (a
carried maximum in ``covc``) and a tracer recurrence for ``NTRAC`` tracers (the level above times a fall factor, plus a
source, clamped at ``cap``). A diagnostic pass then combines cover and the first tracer, level by level, carried over
levels (``out[jk]`` reads ``out[jk - 1]``) so that stage 1 keeps it as a level loop; ``np.minimum`` stops it from
becoming a scan. ``tr`` is (NLEV, NTRAC, KLON), the others (NLEV, KLON); ``KLON`` is contiguous. The column loop is
outermost, which is the GPU order; on a CPU the level loops belong outside and ``KLON`` innermost.
"""

import numpy as np


def column_physics_cpu(tr, src, fall, za, pcov, covc, out, cap, NLEV, KLON, NTRAC):
    # array shapes (numpy->dace): tr=(NLEV,NTRAC,KLON), src=(NLEV,KLON), fall=(NTRAC,), za=(NLEV,KLON),
    # pcov=(NLEV,KLON), covc=(KLON,), out=(NLEV,KLON), cap=()
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
