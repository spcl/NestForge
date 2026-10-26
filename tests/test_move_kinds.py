# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The move catalog the agent reads: every kind with the tree rows it takes and the transformation it runs, and the
skill page naming every kind."""

from nestforge.paths import REPO_ROOT
from nestforge.stages.moves import MOVE_SHAPES, move_kinds

PAGE = REPO_ROOT / "plugin" / "nestforge" / "skills" / "nestforge" / "SKILL.md"


def test_every_move_kind_is_listed_with_the_tree_rows_it_takes_in_order() -> None:
    sut = {kind.kind: kind for kind in move_kinds()}

    assert list(sut) == list(MOVE_SHAPES)
    assert sut["subgraph-fission"].rows == ("kernel", "state")
    assert sut["interchange-loop-map"].rows == ("for", "kernel")
    assert sut["map-fission"].runs == "MapFission" and "map-fusion" in sut["map-fission"].does


def test_the_skill_page_names_every_move_kind() -> None:
    page = PAGE.read_text()

    unnamed = [kind for kind in MOVE_SHAPES if f"`{kind}`" not in page and not kind.startswith("interchange-")]

    assert not unnamed
    assert "`interchange-*`" in page
