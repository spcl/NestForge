# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Decoys of ``aes_graupel`` (branch ``decoys``): schedules an agent could reasonably take that are not the reference solution."""

from nestforge.replay import Context, Decoy, Script


def stage1(s: Script, ctx: Context) -> None:
    """No move and no default fusions: the species map stays 6 wide, yet faster than the default (1.38x at M, 72
    threads, job 5256868); the reference beats it by 1.33x."""
    s.finish()


DECOYS = {
    "stage1": Decoy(
        stage1,
        "stopped-early",
        "taking no move already beats the default, so stopping there looks like a win",
    )
}
