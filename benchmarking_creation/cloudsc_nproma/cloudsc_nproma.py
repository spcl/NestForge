"""Input generation for cloudsc_nproma (kept out of the numpy reference; the loader finds ``initialize`` here).

Columns are drawn flat ``(ngptot, ...)`` and then blocked to ``(nblk, ..., nproma)``. Half-level pressure ``paph``
increases downwards (cumulative layer thicknesses of 50-900 Pa on top of 1000 Pa), ``pap`` is the mid-level pressure,
``pt`` is 220-300 K, cloud fraction ``za`` is in [0, 1) with about 40% clear cells, and the mixing ratios ``q`` are
in [-2e-6, 1e-5] kg/kg (the negative part exercises ``max(., 0)``) with about 20% dry levels and 15% dry columns
(which exercise the cover ramp). The padded tail of the last block repeats valid draws; the kernel never touches it.
``vfall`` holds CloudSC-like fall speeds per species (vapour, liquid, ice, rain, snow).
"""

import numpy as np

NCLV = 5


def initialize(nlev, nproma, ngptot, nblk, ncldtop, datatype=np.float64, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    n = nblk * nproma
    dp = rng.uniform(50.0, 900.0, size=(n, nlev + 1))
    paph = np.cumsum(dp, axis=1) + 1000.0
    pap = 0.5 * (paph[:, :-1] + paph[:, 1:])
    pt = rng.uniform(220.0, 300.0, size=(n, nlev))
    za = rng.uniform(0.0, 1.0, size=(n, nlev)) * (rng.uniform(size=(n, nlev)) < 0.6)
    q = rng.uniform(-2e-6, 1e-5, size=(n, NCLV, nlev))
    q *= (rng.uniform(size=(n, 1, nlev)) < 0.8) * (rng.uniform(size=(n, 1, 1)) < 0.85)

    def blk(a):
        a = a.reshape(nblk, nproma, *a.shape[1:])
        return np.ascontiguousarray(np.moveaxis(a, 1, -1), dtype=datatype)

    vfall = np.array([0.0, 0.15, 4.0, 1.0, 0.0], dtype=datatype)
    pfplsx = np.zeros((nblk, NCLV, nlev + 1, nproma), dtype=datatype)
    pcov = np.zeros((nblk, nlev, nproma), dtype=datatype)
    return blk(paph), blk(pap), blk(pt), blk(za), blk(q), vfall, pfplsx, pcov
