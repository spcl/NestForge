"""cloudsc_ntrac: sedimentation of NTRAC tracers and a cover recurrence down one level loop (numpy reference).

Source: CloudSC (ecmwf-ifs/dwarf-p-cloudsc, ``src/cloudsc_fortran/cloudsc.F90``): 4.2 per-species fall (1715-1755)
and precipitation cover (1757-1789), the 5.2 implicit diagonal (2594-2610), the clamp (2681-2698) and the 5.3 flux
(2700-2719).

Edits from CloudSC: ``NTRAC`` independent tracers (as for the IFS-AER aerosols) instead of the 5 coupled species,
each with its own fall speed and no off-diagonal solver terms; the flux is stored as "out of level jk"
(``pflux[jm, jk]``), not as CloudSC's ``ZPFPLSX(JK+1)``; the cloud top is level 1; the cover is gated by ``prec``,
the precipitation flux of the previous time step (an input). ``prec`` is 0 or at least 1e-6, so
``min(prec * 1e14, 1)`` is exactly 0 or 1. The new tracer value ``qn`` is not stored; the flux carries the
recurrence (``qn = pflux / (vfall * rho)``).

Shapes: ``q`` and ``pflux`` are (NTRAC, NLEV, KLON); ``dtgdp``, ``rdtgdp``, ``rho``, ``za``, ``prec``, ``pcov`` are
(NLEV, KLON); ``vfall`` is (NTRAC,). ``KLON`` (columns) is contiguous; ``jm`` and ``jl`` are parallel, and both
recurrences are carried down the levels ``jk``.
"""

import numpy as np


def cloudsc_ntrac(q, pflux, vfall, dtgdp, rdtgdp, rho, za, prec, pcov, NLEV, KLON, NTRAC):
    # array shapes (numpy->dace): q=(NTRAC,NLEV,KLON), pflux=(NTRAC,NLEV,KLON), vfall=(NTRAC,),
    # dtgdp=(NLEV,KLON), rdtgdp=(NLEV,KLON), rho=(NLEV,KLON), za=(NLEV,KLON), prec=(NLEV,KLON),
    # pcov=(NLEV,KLON)
    for jk in range(1, NLEV):
        for jm in range(NTRAC):
            for jl in range(KLON):
                src = pflux[jm, jk - 1, jl] * dtgdp[jk, jl]
                sink = dtgdp[jk, jl] * vfall[jm] * rho[jk, jl]
                qn = max((q[jm, jk, jl] + src) / (1.0 + sink), 0.0)
                pflux[jm, jk, jl] = sink * qn * rdtgdp[jk, jl]
        for jl in range(KLON):
            cov = 1.0 - (1.0 - pcov[jk - 1, jl]) * (1.0 - max(za[jk, jl], za[jk - 1, jl])) / (
                1.0 - min(za[jk - 1, jl], 1.0 - 1e-6)
            )
            pcov[jk, jl] = max(cov, 0.1) * min(prec[jk, jl] * 1e14, 1.0)
