"""cloudsc_lateral_drift: CloudSC rain sedimentation followed by an implicit lateral drift of the precipitation flux
(numpy reference).

Source: ecmwf-ifs/dwarf-p-cloudsc ``src/cloudsc_fortran/cloudsc.F90`` (v1.5.3, ``f7ba9f8``), section 4.2 rain
sedimentation (lines 1721-1756), the 5.2 implicit diagonal (2603), the 5.3 flux handed to the next level (2705-2709)
and the section 8 flux sum (2795-2800). Edit: after the vertical step, the total flux drifts sideways with an
implicit-upwind scheme, eastward on some levels and westward on others (``east`` is a per-level int32 flag), and
each step is limited by ``cap`` (a ``min``).

Arrays are row-major ``(level, column)`` with the column index contiguous. Carried values: ``pfplsx`` is carried
down the levels (``jk`` to ``jk + 1``, columns independent); ``outE`` is carried eastward and ``outW`` westward
along the columns (levels independent). ``flx``, ``outE`` and ``outW`` are work arrays; ``out`` is the result.
"""

PTSPHY = 3600.0  # physics timestep (s)
RG = 9.80665  # gravity (m s-2)
RD = 287.0597  # dry-air gas constant (J kg-1 K-1)
VQR = 4.0  # rain fall speed (m s-1)
CFL = 0.5  # lateral Courant number


def cloudsc_lateral_drift(pap, paph, ztp1, zqr, pfsnow, pfplsx, flx, outE, outW, out, cap, east, NLEV, KLON, NCLDTOP):
    # array shapes (numpy->dace): pap=(NLEV,KLON), paph=(NLEV+1,KLON), ztp1=(NLEV,KLON), zqr=(NLEV,KLON),
    # pfsnow=(NLEV+1,KLON), pfplsx=(NLEV+1,KLON), flx=(NLEV+1,KLON), outE=(NLEV+1,KLON), outW=(NLEV+1,KLON),
    # out=(NLEV+1,KLON), cap=(NLEV+1,KLON), east=(NLEV+1,) int32
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
