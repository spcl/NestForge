# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""OMEN scattering self-energies, written as an explicit loop nest (``programs/scattering_self_energies_m2_dace.py``):
on the CPU, pull the ``a`` map out through the reduction loops so ``k, E, a`` form one parallel map."""

from nestforge.replay import Context, Script

KERNEL = "scattering_self_energies"
PROGRAM = "programs/scattering_self_energies_m2_dace.py"
#: Preset M of ``scattering_self_energies_m2``: fewer ``(k, E)`` points than threads, so map ``a`` fills them.
PARAMS = {"Nkz": 2, "NE": 4}
CONFIGS = (Context("cpu", 72, "M"),)


def solve(s: Script, ctx: Context) -> None:
    s.apply(s.pick("interchange-loop-map", writes=["Sigma"]))  # map a out past loop w
    s.apply(s.pick("interchange-loop-map", writes=["Sigma"]))  # and past loop q, offered only after the first
    s.finish()
