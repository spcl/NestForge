# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""OMEN scattering self-energies, written as an explicit loop nest (``programs/scattering_self_energies_m3_dace.py``):
on the CPU, pull the ``a`` map out through the reduction loops so ``k, E, a`` form one parallel map."""

from nestforge.replay import Context, Script

KERNEL = "scattering_self_energies"
PROGRAM = "programs/scattering_self_energies_m3_dace.py"
#: Preset M of ``scattering_self_energies_m3``: fewer ``(k, E)`` points than threads, so map ``a`` fills them.
PARAMS = {"Nkz": 2, "NE": 8}
CONFIGS = (Context("cpu", 72, "M"),)
#: Loops map ``a`` is pulled out past after the first interchange: ``b``, ``i``, ``w``, ``q``.
PULLS = 4


def solve(s: Script, ctx: Context) -> None:
    s.apply(s.pick("interchange-map-loop", writes=["Sigma"]))  # map a inside loop b
    for _ in range(PULLS):  # then map a out past loops b, i, w and q, one at a time
        s.apply(s.pick("interchange-loop-map", writes=["Sigma"]))
    s.finish()
