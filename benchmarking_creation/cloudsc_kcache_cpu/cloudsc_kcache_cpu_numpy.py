"""cloudsc_kcache_cpu: rain and snow precipitation column physics in the loop structure of CloudSC's k-caching GPU
port (numpy reference).

Loop structure from dwarf-p-cloudsc ``src/cloudsc_gpu/cloudsc_gpu_scc_k_caching_mod.F90`` (f7ba9f8): one loop over
columns ``jl`` whose body is a per-column prologue (lines 635-658: the cross-level state is set to its top-of-column
value) followed by one fused sequential level loop ``jk`` (lines 665-2636). Physics from ``cloudsc.F90``:

  - 4.2 sedimentation (1715-1793): implicit fall of rain and snow, source = flux from the level above;
  - cover of the precipitation (1757-1789): random-overlap total cover ``zcov``, clear-sky part ``zcovpclr``,
    running maximum ``zcovmax``;
  - 4.5 rain evaporation, ``IEVAPRAIN == 1`` (2128-2196), with the guards written as min/max limiters, plus a
    simple melting of snow above freezing;
  - 5.3 fluxes to the next level (2700-2725) and section 8 cumulative fluxes and enthalpy fluxes (2795-2879).

Arrays are (level, column), row-major, so ``klon`` is the contiguous axis; half-level arrays have ``nlev + 1`` rows.
Columns are independent; levels are sequential. Six values are carried from one level to the next within a column:
the total precipitation cover ``zcov``, its running maximum ``zcovmax``, the rain and snow fluxes ``zpfr``/``zpfs``
and the cumulative rain and snow flux sums ``zsqrf``/``zsqsf``; they live in per-column arrays of length ``klon``,
set in the prologue and read once and written once per level. ``zpaphd`` is the per-column inverse surface pressure.
"""

import numpy as np
from math import exp, sqrt

PTSPHY = 3600.0
QTMST = 1.0 / 3600.0
RG = 9.80665
RD = 287.0597
RETV = 0.6077667
RTT = 273.16
R2ES = 380.1608703442847
R3LES = 17.502
R4LES = 32.19
R5LES = 4217.456
RLVTT = 2.5008e6
RLSTT = 2.8345e6
RALVDCP = 2489.08
RALFDCP = 330.38
RPRECRHMAX = 0.7
RVRFACTOR = 0.00509
RPECONS = 5.547e-5
RCOVPMIN = 0.1
ZEPSEC = 1e-14
VQR = 4.0
VQS = 1.0
KMELT = 0.05


def cloudsc_kcache_cpu(pt, tt, pq, qr, qs, pa, pap, paph, tnd_t, tnd_q, tnd_qr, tnd_qs, pcovptot, pfplsl, pfplsn,
                       pfhpsl, pfhpsn, pfsqrf, pfsqsf, nlev, klon, ncldtop):
    # array shapes (numpy->dace): pt, tt, pq, qr, qs, pa, pap, tnd_t, tnd_q, tnd_qr, tnd_qs, pcovptot = (nlev, klon),
    # paph, pfplsl, pfplsn, pfhpsl, pfhpsn, pfsqrf, pfsqsf = (nlev + 1, klon)
    zcov = np.zeros((klon,), dtype=pt.dtype)
    zcovmax = np.zeros((klon,), dtype=pt.dtype)
    zpfr = np.zeros((klon,), dtype=pt.dtype)
    zpfs = np.zeros((klon,), dtype=pt.dtype)
    zsqrf = np.zeros((klon,), dtype=pt.dtype)
    zsqsf = np.zeros((klon,), dtype=pt.dtype)
    zpaphd = np.zeros((klon,), dtype=pt.dtype)
    for jl in range(klon):
        # per-column prologue: top-of-column state
        zcov[jl] = 0.0
        zcovmax[jl] = 0.0
        zpfr[jl] = 0.0
        zpfs[jl] = 0.0
        zsqrf[jl] = 0.0
        zsqsf[jl] = 0.0
        zpaphd[jl] = 1.0 / paph[nlev, jl]
        for jk in range(ncldtop, nlev):
            zcov0 = zcov[jl]
            zcovmax0 = zcovmax[jl]
            zpfr0 = zpfr[jl]
            zpfs0 = zpfs[jl]
            zsqrf0 = zsqrf[jl]
            zsqsf0 = zsqsf[jl]
            ztp1 = pt[jk, jl] + PTSPHY * tt[jk, jl]
            zdp = paph[jk + 1, jl] - paph[jk, jl]
            zdtgdp = PTSPHY * RG / zdp
            zrdtgdp = zdp / (RG * PTSPHY)
            zrho = pap[jk, jl] / (RD * ztp1)
            # saturation over liquid (FOEELIQ / FOEDEM inlined)
            zfoeew = R2ES * exp(R3LES * (ztp1 - RTT) / (ztp1 - R4LES))
            zqs0 = min(zfoeew / pap[jk, jl], 0.5)
            zcor = 1.0 / (1.0 - RETV * zqs0)
            zqsliq = zqs0 * zcor
            zcorqsliq = 1.0 + zqsliq * zcor * R5LES / ((ztp1 - R4LES) * (ztp1 - R4LES))
            # precipitation cover, random overlap
            zcov1 = 1.0 - (1.0 - zcov0) * (1.0 - max(pa[jk, jl], pa[jk - 1, jl])) / (
                1.0 - min(pa[jk - 1, jl], 1.0 - 1e-6))
            zcov2 = min(zcov1, 1.0)
            zcovpclr = max(0.0, zcov2 - pa[jk, jl])
            zcovmax1 = max(zcov2, zcovmax0)
            # sedimentation: sinks and first guess with the flux from above
            zsinkr = zdtgdp * VQR * zrho
            zsinks = zdtgdp * VQS * zrho
            zqrf = qr[jk, jl] + zpfr0 * zdtgdp
            zqsf = qs[jk, jl] + zpfs0 * zdtgdp
            # rain evaporation in the clear-sky part of the precipitation cover
            zzrh0 = RPRECRHMAX + (1.0 - RPRECRHMAX) * zcovmax1 / max(ZEPSEC, 1.0 - pa[jk, jl])
            zzrh = min(max(zzrh0, RPRECRHMAX), 1.0)
            zqe0 = (pq[jk, jl] - pa[jk, jl] * zqsliq) / max(ZEPSEC, 1.0 - pa[jk, jl])
            zqe = max(0.0, min(zqe0, zqsliq))
            zpreclr = zqrf * zcovpclr / max(zcov2 * zdtgdp, ZEPSEC)
            zbeta1 = sqrt(pap[jk, jl] * zpaphd[jl]) / RVRFACTOR * zpreclr / max(zcovpclr, ZEPSEC)
            zbeta = RG * RPECONS * 0.5 * zbeta1 ** 0.5777
            zdenom = 1.0 + zbeta * PTSPHY * zcorqsliq
            zdpr = zcovpclr * zbeta * max(0.0, zzrh * zqsliq - zqe) / zdenom * zdp / RG
            zevap = min(zdpr * zdtgdp, zqrf)
            zcov3 = max(RCOVPMIN, zcov2 - max(0.0, (zcov2 - pa[jk, jl]) * zevap / max(zqrf, ZEPSEC)))
            # melting of snow above freezing
            zmelt = zqsf * min(1.0, max(0.0, (ztp1 - RTT) * KMELT))
            # implicit update and fluxes to the next level
            zqrn = (zqrf - zevap + zmelt) / (1.0 + zsinkr)
            zqsn = (zqsf - zmelt) / (1.0 + zsinks)
            zpfr1 = zsinkr * zqrn * zrdtgdp
            zpfs1 = zsinks * zqsn * zrdtgdp
            zgdph = -zdp / (RG * PTSPHY)
            zsqrf1 = zsqrf0 + (zqrn - qr[jk, jl]) * zgdph
            zsqsf1 = zsqsf0 + (zqsn - qs[jk, jl]) * zgdph
            tnd_qr[jk, jl] = (zqrn - qr[jk, jl]) * QTMST
            tnd_qs[jk, jl] = (zqsn - qs[jk, jl]) * QTMST
            tnd_q[jk, jl] = zevap * QTMST
            tnd_t[jk, jl] = -(RALVDCP * zevap + RALFDCP * zmelt) * QTMST
            pcovptot[jk, jl] = zcov3
            pfplsl[jk + 1, jl] = zpfr1
            pfplsn[jk + 1, jl] = zpfs1
            pfhpsl[jk + 1, jl] = -RLVTT * zpfr1
            pfhpsn[jk + 1, jl] = -RLSTT * zpfs1
            pfsqrf[jk + 1, jl] = zsqrf1
            pfsqsf[jk + 1, jl] = zsqsf1
            zcov[jl] = zcov3
            zcovmax[jl] = zcovmax1
            zpfr[jl] = zpfr1
            zpfs[jl] = zpfs1
            zsqrf[jl] = zsqrf1
            zsqsf[jl] = zsqsf1
