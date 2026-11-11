# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The move catalog the agent reads: every kind with the tree rows it takes and the transformation it runs, and the
skill page naming every kind."""

import re

from nestforge.paths import REPO_ROOT
from nestforge.stages.moves import MOVE_SHAPES, move_kinds

PAGE = REPO_ROOT / "plugin" / "nestforge" / "skills" / "nestforge" / "SKILL.md"
#: A backticked kebab-case token with one of these parts reads as a move name (``kernel-optimizer`` does not).
MOVE_WORDS = frozenset(
    {"for", "doall", "doalls", "if", "fusion", "fission", "interchange", "tile", "untile", "peel", "expand", "copy"}
)


def test_every_move_kind_is_listed_with_the_tree_rows_it_takes_in_order() -> None:
    sut = {kind.kind: kind for kind in move_kinds()}

    assert list(sut) == list(MOVE_SHAPES)
    assert sut["subgraph-fission"].rows == ("doall", "state")
    assert sut["interchange-for-doall"].rows == ("for", "doall")
    assert sut["doall-fission"].runs == "MapFission" and "doall-fusion" in sut["doall-fission"].does


def test_the_skill_page_names_every_move_kind() -> None:
    page = PAGE.read_text()

    unnamed = [kind for kind in MOVE_SHAPES if f"`{kind}`" not in page]

    assert not unnamed


def test_the_plugin_pages_name_only_move_kinds_that_exist() -> None:
    named = {
        (page.relative_to(REPO_ROOT).as_posix(), token)
        for page in sorted((REPO_ROOT / "plugin").rglob("*.md"))
        for token in re.findall(r"`([a-z]+(?:-[a-z0-9]+)+)`", page.read_text())
        if MOVE_WORDS & set(token.split("-"))
    }

    assert named, "no move named on any plugin page"
    assert sorted(entry for entry in named if entry[1] not in MOVE_SHAPES) == []
