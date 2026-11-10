# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Floyd-Warshall: copy row ``k`` and column ``k`` of ``path`` into vectors (``snapshot-reads``), then the default.

Each step computes ``outer = path[:, k] + path[k, :]`` in one doall and ``path = min(path, outer)`` in another. The
first reads ``path`` and the second writes it, so the default cannot fuse them and writes the N x N ``outer`` every
step. Once the two slices are copied, the first doall no longer touches ``path`` and the default's own fusion removes
``outer``. The gain grows with N (the vectors cost more than the saved traffic at small N): see the README row."""

from nestforge.replay import Context, Script

KERNEL = "floyd_warshall"
CONFIGS = (Context("cpu", 72, "M"),)


def solve(s: Script, ctx: Context) -> None:
    s.apply(s.pick("snapshot-reads"))
    s.defaults()


def stage1(s: Script, ctx: Context) -> None:
    """No move and no default fusions: the same time as the default (0.98-1.01x at M)."""
    s.finish()


DECOYS = {"stage1": stage1}
