# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""3D tensor permutation ``B = A.transpose(...)``: expand the library transpose into NestForge's own loops, tile the
map 16 x 16 x 16, then the default stage 2 on top.

The default keeps the transpose as one library call (HPTT). Expanded alone it is no faster (0.98x); tiled, each
16 x 16 x 16 block of ``A`` is read and written while it is in cache, so the strided side of the permutation stops
missing on every element (49.6 ms to 14.6 ms, 3.40x at preset M, 72 threads, 3 alternated rounds 3.19-3.54x, on
690de2f, job 5303150)."""

from nestforge.replay import Context, Decoy, Script

KERNEL = "permute_3d"
CONFIGS = (Context("cpu", 72, "M"),)
#: Tile edge of the reference: a 16^3 block of float64 is 32 KiB, one block of ``A`` and one of ``B`` per core.
TILE = 16
#: Tile edge of the decoy: a 256^3 block no longer fits any cache.
DECOY_TILE = 256


def expand_and_tile(s: Script, edge: int) -> None:
    """The transpose as loops, its map tiled ``edge`` per dimension, then the default."""
    s.apply(s.pick("expand-library", reads=["A"], writes=["B"]))
    tile = s.pick("tile-doall", reads=["A"], writes=["B"])
    s.apply(tile.model_copy(update={"tile_sizes": (edge,) * len(tile.tile_sizes)}))
    s.defaults()


def solve(s: Script, ctx: Context) -> None:
    expand_and_tile(s, TILE)


def expand_only(s: Script, ctx: Context) -> None:
    """The expansion without the tile: the loops are as fast as HPTT (0.98-0.99x at M), so the tile is the lever."""
    s.apply(s.pick("expand-library", reads=["A"], writes=["B"]))
    s.defaults()


def large_tile(s: Script, ctx: Context) -> None:
    """The same two moves with 256-wide tiles: slower than the default (0.35-0.39x at M; 64-wide gave 1.73x on c2daa40)."""
    expand_and_tile(s, DECOY_TILE)


DECOYS = {
    "expand-only": Decoy(
        expand_only,
        "missing-enabler",
        "the expansion alone is as fast as the library call, so it looks safe, and the tile is what pays",
    ),
    "large-tile": Decoy(
        large_tile,
        "went-past-best",
        "a bigger tile looks like more reuse, but 256 per dimension no longer fits in cache",
    ),
}
