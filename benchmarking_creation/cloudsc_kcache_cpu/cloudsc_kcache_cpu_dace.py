"""Hand-written DaCe program for cloudsc_kcache_cpu.

This is the program that was probed and timed as ``kc_arr2`` on 2026-10-09 and 2026-10-21 (same loops as
``cloudsc_kcache_cpu_numpy.py``): the six per-column carries are ``(klon,)`` transient arrays, each read once at the top
of a level iteration into a local scalar and written once at its end, which is what lets ``interchange-map-loop`` apply
after the fission. Only the size symbols were renamed (``NLEV``, ``KLON``, ``NCLDTOP`` to the manifest's lowercase
names) and the constants inlined. It was not produced by ``scripts/emit_local_dace.py``. The first line is not the
autogen marker, so the loader treats it as an override.
"""

from math import exp, sqrt

import dace as dc
import numpy as np

nlev = dc.symbol("nlev", dtype=dc.int64, positive=True)
klon = dc.symbol("klon", dtype=dc.int64, positive=True)
ncldtop = dc.symbol("ncldtop", dtype=dc.int64, positive=True)

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

f64 = dc.float64


@dc.program
def cloudsc_kcache_cpu(
    pt: f64[nlev, klon],
    tt: f64[nlev, klon],
    pq: f64[nlev, klon],
    qr: f64[nlev, klon],
    qs: f64[nlev, klon],
    pa: f64[nlev, klon],
    pap: f64[nlev, klon],
    paph: f64[nlev + 1, klon],
    tnd_t: f64[nlev, klon],
    tnd_q: f64[nlev, klon],
    tnd_qr: f64[nlev, klon],
    tnd_qs: f64[nlev, klon],
    pcovptot: f64[nlev, klon],
    pfplsl: f64[nlev + 1, klon],
    pfplsn: f64[nlev + 1, klon],
    pfhpsl: f64[nlev + 1, klon],
    pfhpsn: f64[nlev + 1, klon],
    pfsqrf: f64[nlev + 1, klon],
    pfsqsf: f64[nlev + 1, klon],
):
    zcov = np.zeros((klon,), dtype=np.float64)
    zcovmax = np.zeros((klon,), dtype=np.float64)
    zpfr = np.zeros((klon,), dtype=np.float64)
    zpfs = np.zeros((klon,), dtype=np.float64)
    zsqrf = np.zeros((klon,), dtype=np.float64)
    zsqsf = np.zeros((klon,), dtype=np.float64)
    zpaphd = np.zeros((klon,), dtype=np.float64)
    for jl in range(klon):
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
            zfoeew = R2ES * exp(R3LES * (ztp1 - RTT) / (ztp1 - R4LES))
            zqs0 = min(zfoeew / pap[jk, jl], 0.5)
            zcor = 1.0 / (1.0 - RETV * zqs0)
            zqsliq = zqs0 * zcor
            zcorqsliq = 1.0 + zqsliq * zcor * R5LES / ((ztp1 - R4LES) * (ztp1 - R4LES))
            zcov1 = 1.0 - (1.0 - zcov0) * (1.0 - max(pa[jk, jl], pa[jk - 1, jl])) / (
                1.0 - min(pa[jk - 1, jl], 1.0 - 1e-6))
            zcov2 = min(zcov1, 1.0)
            zcovpclr = max(0.0, zcov2 - pa[jk, jl])
            zcovmax1 = max(zcov2, zcovmax0)
            zsinkr = zdtgdp * VQR * zrho
            zsinks = zdtgdp * VQS * zrho
            zqrf = qr[jk, jl] + zpfr0 * zdtgdp
            zqsf = qs[jk, jl] + zpfs0 * zdtgdp
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
            zmelt = zqsf * min(1.0, max(0.0, (ztp1 - RTT) * KMELT))
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
