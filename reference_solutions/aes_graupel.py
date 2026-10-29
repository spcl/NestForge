# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""ICON AES graupel: split the map computing the sedimentation sink and its ratio (``sink``, ``sx2x``), then the
default stage 2 on top."""

from nestforge.replay import Context, Script

KERNEL = "aes_graupel"
CONFIGS = (Context("cpu", 72, "M"),)
CI_SKIP = (
    "on dace extended 3ef7de40e the session does not start: canonicalization leaves a nested connector "
    "(__inl1_flux_7) unlike its outer container and memlet propagation refuses it; first bad dace commit 9c9a6a7a4 "
    "(StateFusionExtended aggressive sink merge), starts on b737e5050"
)


def solve(s: Script, ctx: Context) -> None:
    s.apply(s.pick("doall-fission", writes=["__inl1_sink", "__inl1_sx2x"]))
    s.defaults()
