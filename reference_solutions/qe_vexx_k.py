# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Quantum ESPRESSO exact exchange (``vexx_k``): the default stage 2, no move on top.

No route beats the default yet; the script keeps the kernel's whole pipeline (its bool ``gamma_only`` symbol and the
nesting that once mapped an unused symbol into a nested graph) replayed against the bench oracle in CI."""

from nestforge.replay import Context, Script

KERNEL = "qe_vexx_k"
CONFIGS = (Context("cpu", 72, "M"),)


def solve(s: Script, ctx: Context) -> None:
    s.defaults()
