# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cholesky factor plus the strict upper triangle, on the CPU: the default keeps ``np.linalg.cholesky`` as a library
call, and on the CPU that call is the slow part of the program. ``expand-library`` turns it into NestForge's own
loops, which the default then parallelizes: 22.3-23.3 s to 121 ms at preset M on 72 threads. The default never
expands a library call; pinning the call to OpenBLAS with ``set-library-implementation`` does not help (the same time
as the default). On the GPU the answer flips: the default's vendor solver takes 8.5 ms and the expansion 9.5 s, so the
default stays the reference there."""

from nestforge.replay import Context, Script

KERNEL = "cholesky2"
CONFIGS = (Context("cpu", 72, "M"),)
CI_SKIP = (
    "the replay's per-kernel check emits Python for the expanded Cholesky kernel that calls an undefined conj "
    "(NameError: name 'conj' is not defined); the whole program builds and matches the bench oracle"
)


def solve(s: Script, ctx: Context) -> None:
    if ctx.device == "cpu":
        s.apply(s.pick("expand-library"))
    s.defaults()


def openblas(s: Script, ctx: Context) -> None:
    """The library call pinned to OpenBLAS: looks like the fix, times like the default."""
    if ctx.device == "cpu":
        s.apply(next(m for m in s.session.list_moves("set-library-implementation") if m.implementation == "OpenBLAS"))
    s.defaults()


DECOYS = {"openblas": openblas}
