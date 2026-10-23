"""graupel_sedim2b: hand-written DaCe program (no autogen marker, so it is a hand override).

The same loops as the numpy reference, with the column axis as a map inside the level loop (the Fortran order).
``fl[jl]`` carries the flux down one species' levels; ``G[jk, jl]`` holds the previous species' outgoing flux at level
``jk`` and is overwritten in place, so every flux is read and written at the same element (an offset access on a
flux array blocks both pulls: 2026-10-22, job 5119404).
"""

import dace as dc
from hpcagent_bench.frameworks.dace_framework import dc_float

NSPEC = dc.symbol("NSPEC", dtype=dc.int64, positive=True)
NLEV = dc.symbol("NLEV", dtype=dc.int64, positive=True)
KLON = dc.symbol("KLON", dtype=dc.int64, positive=True)
NSUB = dc.symbol("NSUB", dtype=dc.int64, positive=True)

__hpcagent_bench_program__ = "graupel_sedim2b"


@dc.program
def graupel_sedim2b(
    q: dc_float[NSPEC, NLEV, KLON],
    c: dc_float[NSPEC, NLEV, KLON],
    frac: dc_float[KLON],
    nsub: dc.int32[KLON],
    fl: dc_float[KLON],
    G: dc_float[NLEV, KLON],
    prec: dc_float[KLON],
    ftop: dc_float[NSPEC, KLON],
    w: dc_float[NSPEC],
):
    for it in range(NSUB):
        for js in range(NSPEC):
            for jk in range(NLEV):
                for jl in dc.map[0:KLON]:
                    if it < nsub[jl]:
                        fin = fl[jl]
                        if jk == 0:
                            fin = ftop[js, jl] * frac[jl]
                        fnew = c[js, jk, jl] * frac[jl] * q[js, jk, jl]
                        q[js, jk, jl] = q[js, jk, jl] + fin - fnew + w[js] * G[jk, jl]
                        G[jk, jl] = fnew
                        fl[jl] = fnew
        for jl in dc.map[0:KLON]:
            if it < nsub[jl]:
                prec[jl] = prec[jl] + fl[jl]
