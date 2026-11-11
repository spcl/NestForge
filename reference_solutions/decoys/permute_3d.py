# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Decoys of ``permute_3d`` (branch ``decoys``): schedules an agent could reasonably take that are not the reference solution."""

from nestforge.replay import Context, Decoy, Script

#: Tile edge of the decoy: a 256^3 block no longer fits any cache.
DECOY_TILE = 256


def expand_and_tile(s: Script, edge: int) -> None:
    """The transpose as loops, its map tiled ``edge`` per dimension, then the default."""
    s.apply(s.pick("expand-library", reads=["A"], writes=["B"]))
    tile = s.pick("tile-doall", reads=["A"], writes=["B"])
    s.apply(tile.model_copy(update={"tile_sizes": (edge,) * len(tile.tile_sizes)}))
    s.defaults()


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
