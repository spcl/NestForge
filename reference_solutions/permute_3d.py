# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""3D tensor permutation ``B = A.transpose(...)``: expand the library transpose into NestForge's own loops, tile the
map 16 x 16 x 16, then the default stage 2 on top.

The default keeps the transpose as one library call (HPTT). Expanded alone it is no faster (0.98x); tiled, each
16 x 16 x 16 block of ``A`` is read and written while it is in cache, so the strided side of the permutation stops
missing on every element (49.6 ms to 14.6 ms, 3.40x at preset M, 72 threads, 3 alternated rounds 3.19-3.54x, on
690de2f, job 5303150)."""

from nestforge.replay import Context, Script

KERNEL = "permute_3d"
CONFIGS = (Context("cpu", 72, "M"),)
#: Tile edge of the reference: a 16^3 block of float64 is 32 KiB, one block of ``A`` and one of ``B`` per core.
TILE = 16


def expand_and_tile(s: Script, edge: int) -> None:
    """The transpose as loops, its map tiled ``edge`` per dimension, then the default."""
    s.apply(s.pick("expand-library", reads=["A"], writes=["B"]))
    tile = s.pick("tile-doall", reads=["A"], writes=["B"])
    s.apply(tile.model_copy(update={"tile_sizes": (edge,) * len(tile.tile_sizes)}))
    s.defaults()


def solve(s: Script, ctx: Context) -> None:
    expand_and_tile(s, TILE)
