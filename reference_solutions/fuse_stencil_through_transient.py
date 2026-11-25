# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stencil through a transient: the default's fusions, then the one ``doall-fusion`` it leaves (the stencil into the
map that writes its transient), then the stages after fusion."""

from tactics.replay import Context, Script

KERNEL = "fuse_stencil_through_transient"
CONFIGS = (Context("cpu", 72, "M"),)


def solve(s: Script, ctx: Context) -> None:
    s.session.run_defaults("moves")
    s.apply(s.pick("doall-fusion"))
    s.finish()
