"""cloudsc_nproma: CloudSC precipitation fall and cover in the driver's blocked NPROMA layout (numpy reference).

Source (dwarf-p-cloudsc f7ba9f8): the block loop of ``cloudsc_driver_mod.F90`` 126-135, whose last block is ragged
(``ICEND = MIN(NPROMA, NGPTOT - JKGLO + 1)``), with the loop order of ``cloudsc_driver_gpu_scc_mod.F90`` 138 and
``cloudsc_gpu_scc_mod.F90`` 492-495 (block, then column, then level). The body is ``cloudsc.F90`` 4.2 implicit
per-species fall (1715-1755), the 4.3 precipitation cover (1757-1789) and the 5.3 cover reset (2700-2719); each
column first zeroes its boundary values at the cloud top.

Layout: every field is ``(nblk, [nclv,] nlev[+1], nproma)`` in C order, the driver's Fortran ``(NPROMA, NLEV, NBLK)``
reversed, so ``nproma`` (the column inside a block) is the contiguous axis. ``nblk = ceil(ngptot / nproma)``; the
last block has only ``ngptot - (nblk - 1) * nproma`` valid columns and its padded tail is never read or written.

Carried down the levels of each column: the species fluxes ``pfplsx[.., jk + 1, ..]`` and the precipitation cover
``pcov[.., jk, ..]``. Columns are independent. The 5.3 reset ``if qpre < eps: cover = 0`` is the ramp
``min(qpre * RQPRE, 1)``.
"""

NCLV = 5
IRAIN = 2
ISNOW = 3
PTSPHY = 3600.0
RG = 9.80665
RD = 287.0597
RCOVPMIN = 0.1
RQPRE = 1.0e7
ZAMAX = 1.0 - 1.0e-6


def cloudsc_nproma(paph, pap, pt, za, q, vfall, pfplsx, pcov, nlev, nproma, ngptot, nblk, ncldtop):
    # array shapes (numpy->dace): paph=(nblk, nlev+1, nproma), pap=(nblk, nlev, nproma), pt=(nblk, nlev, nproma),
    # za=(nblk, nlev, nproma), q=(nblk, 5, nlev, nproma), vfall=(5,), pfplsx=(nblk, 5, nlev+1, nproma),
    # pcov=(nblk, nlev, nproma)
    for ibl in range(nblk):
        for jl in range(min(nproma, ngptot - ibl * nproma)):
            pcov[ibl, ncldtop - 1, jl] = 0.0
            for jm in range(NCLV):
                pfplsx[ibl, jm, ncldtop, jl] = 0.0
            for jk in range(ncldtop, nlev):
                dtgdp = PTSPHY * RG / (paph[ibl, jk + 1, jl] - paph[ibl, jk, jl])
                rho = pap[ibl, jk, jl] / (RD * pt[ibl, jk, jl])
                for jm in range(NCLV):
                    src = pfplsx[ibl, jm, jk, jl] * dtgdp
                    sink = dtgdp * vfall[jm] * rho
                    qnew = max((q[ibl, jm, jk, jl] + src) / (1.0 + sink), 0.0)
                    pfplsx[ibl, jm, jk + 1, jl] = sink * qnew / dtgdp
                qpre = pfplsx[ibl, IRAIN, jk + 1, jl] + pfplsx[ibl, ISNOW, jk + 1, jl]
                cov = 1.0 - (1.0 - pcov[ibl, jk - 1, jl]) * (1.0 - max(za[ibl, jk, jl], za[ibl, jk - 1, jl])) / (
                    1.0 - min(za[ibl, jk - 1, jl], ZAMAX))
                pcov[ibl, jk, jl] = min(qpre * RQPRE, 1.0) * max(cov, RCOVPMIN)
