"""Hand-written DaCe program for cloudsc_lateral_drift.

This is the program that was timed, transcribed from the 2026-10-09 probe ``b1`` (same loops as
``cloudsc_lateral_drift_numpy.py``). It was not produced by ``scripts/emit_local_dace.py``. The
first line is not the autogen marker, so the loader treats it as an override.
"""

import dace as dc

NLEV = dc.symbol("NLEV", dtype=dc.int64, positive=True)
KLON = dc.symbol("KLON", dtype=dc.int64, positive=True)
NCLDTOP = dc.symbol("NCLDTOP", dtype=dc.int64, positive=True)

PTSPHY = 3600.0  # physics timestep (s)
RG = 9.80665  # gravity (m s-2)
RD = 287.0597  # dry-air gas constant (J kg-1 K-1)
VQR = 4.0  # rain fall speed (m s-1)
CFL = 0.5  # lateral Courant number

f64 = dc.float64
i32 = dc.int32


@dc.program
def cloudsc_lateral_drift(
    pap: f64[NLEV, KLON],
    paph: f64[NLEV + 1, KLON],
    ztp1: f64[NLEV, KLON],
    zqr: f64[NLEV, KLON],
    pfsnow: f64[NLEV + 1, KLON],
    pfplsx: f64[NLEV + 1, KLON],
    flx: f64[NLEV + 1, KLON],
    outE: f64[NLEV + 1, KLON],
    outW: f64[NLEV + 1, KLON],
    out: f64[NLEV + 1, KLON],
    cap: f64[NLEV + 1, KLON],
    east: i32[NLEV + 1],
):
    for jk in range(NCLDTOP, NLEV):
        for jl in range(KLON):
            zdtgdp = PTSPHY * RG / (paph[jk + 1, jl] - paph[jk, jl])
            zsink = zdtgdp * VQR * pap[jk, jl] / (RD * ztp1[jk, jl])
            zqn = (zqr[jk, jl] + pfplsx[jk, jl] * zdtgdp) / (1.0 + zsink)
            pfplsx[jk + 1, jl] = zsink * zqn / zdtgdp
    for k in range(NLEV + 1):
        for jl in range(KLON):
            flx[k, jl] = pfplsx[k, jl] + pfsnow[k, jl]
    for k in range(NLEV + 1):
        outE[k, 0] = flx[k, 0]
        outW[k, KLON - 1] = flx[k, KLON - 1]
    for j in range(1, KLON):
        for k in range(NLEV + 1):
            outE[k, j] = (flx[k, j] + min(CFL * outE[k, j - 1], cap[k, j])) / (1.0 + CFL)
    for j in range(KLON - 2, -1, -1):
        for k in range(NLEV + 1):
            outW[k, j] = (flx[k, j] + min(CFL * outW[k, j + 1], cap[k, j])) / (1.0 + CFL)
    for k in range(NLEV + 1):
        for jl in range(KLON):
            out[k, jl] = east[k] * outE[k, jl] + (1 - east[k]) * outW[k, jl]
