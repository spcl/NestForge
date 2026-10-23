"""Input generation for cloudsc_kcache_cpu (kept out of the numpy reference, as hpcagent_bench requires; the loader
finds ``initialize`` here).

Monotone pressure with a ~101 kPa surface, temperature rising downward through freezing, humidity below saturation,
rain and snow mixing ratios and cloud fraction with about 30% exact zeros, outputs zero-initialized.
"""

import numpy as np

PTSPHY = 3600.0
RTT = 273.16
R2ES = 380.1608703442847
R3LES = 17.502
R4LES = 32.19


def initialize(nlev, klon, ncldtop, datatype=np.float64, seed=0):
    rng = np.random.default_rng(seed)
    dp = rng.uniform(0.5, 1.5, size=(nlev + 1, klon))
    paph = 100.0 + 101000.0 * np.cumsum(dp, axis=0) / dp.sum(axis=0)
    pap = 0.5 * (paph[:-1] + paph[1:])
    lev = (np.arange(nlev) / max(nlev - 1, 1))[:, None]
    pt = 210.0 + 85.0 * lev + rng.uniform(-3.0, 3.0, (nlev, klon))
    tt = rng.uniform(-2e-4, 2e-4, (nlev, klon))
    ztp1 = pt + PTSPHY * tt
    qsat = np.minimum(R2ES * np.exp(R3LES * (ztp1 - RTT) / (ztp1 - R4LES)) / pap, 0.5)
    pq = rng.uniform(0.3, 1.0, (nlev, klon)) * qsat
    qr = np.maximum(rng.uniform(-5e-5, 1e-4, (nlev, klon)), 0.0)
    qs = np.maximum(rng.uniform(-5e-5, 1e-4, (nlev, klon)), 0.0)
    pa = np.maximum(rng.uniform(-0.3, 0.7, (nlev, klon)), 0.0)
    full = np.zeros((nlev, klon), dtype=datatype)
    half = np.zeros((nlev + 1, klon), dtype=datatype)
    arrays = [pt, tt, pq, qr, qs, pa, pap, paph]
    outs = [full.copy() for _ in range(5)] + [half.copy() for _ in range(6)]
    return tuple(a.astype(datatype) for a in arrays) + tuple(outs)
