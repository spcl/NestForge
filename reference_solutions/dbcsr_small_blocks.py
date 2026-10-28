# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""DBCSR block-sparse multiply with small, sparse blocks (4 x 4 at density 0.05, the ``dbcsr_small_blocks`` presets
of HPCAgent-Bench branch ``benchmarking-nestforge``): hoist the padded-slot guard out of the slot loop."""

from nestforge.replay import Context, Script

KERNEL = "dbcsr"
PARAMS = {"block_size": 4, "density": 0.05}
CONFIGS = (Context("cpu", 1, "M"), Context("cpu", 72, "M"))
CI_SKIP = (
    "the baseline (default_moves) of dbcsr fails to build on DaCe 7683e143 with KeyError 'a_index_index_0', with or "
    "without PARAMS; the reference builds and matches the oracle"
)


def solve(s: Script, ctx: Context) -> None:
    s.apply(s.pick("interchange-loop-if"))
    s.finish()
