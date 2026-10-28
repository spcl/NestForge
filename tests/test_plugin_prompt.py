# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The nestforge packet's prompt overlay reshapes both of the bench's prompts: the agent writes no code, delivers
finalize's library, and reads what NestForge does right after the intro."""

import pytest
from hpcagent_agent.plugins import Plugin
from hpcagent_bench import config, prompt_build
from hpcagent_bench.categories import Language, SourceMode
from hpcagent_bench.harness.prompts import build_prompt
from hpcagent_bench.harness.task import Task

from nestforge.paths import REPO_ROOT

PLUGIN = REPO_ROOT / "plugin" / "nestforge"
WHAT_IT_DOES = (PLUGIN / "prompts" / "nestforge" / "what_it_does.md").read_text(encoding="utf-8")
TASK = Task("jacobi_1d", SourceMode.ANY, Language.CPP)


@pytest.fixture(scope="module")
def registered(nestforge_plugin: tuple[Plugin, ...]) -> None:
    pass


def test_the_overlay_names_its_agent_blocks_and_every_text_it_reads_exists(registered: None) -> None:
    """The agent half is rendered by the bench's user-plugins CI (the bench checkout holds agent/prompt.md); here
    every file it reads loads and it inserts what NestForge does after the intro."""
    overlay = prompt_build.load_overlay(str(PLUGIN / "prompts" / "nestforge.yaml"))

    assert {"build", "syntax_check", "index_buffers"} <= {k for k, e in overlay.agent.set.items() if e.enabled is False}
    assert [(i.after, i.block.text) for i in overlay.agent.insert] == [("intro", WHAT_IT_DOES)]
    assert "expand-library" in (overlay.agent.set["keep_improving"].text or "")


def test_the_task_prompt_delivers_finalizes_library_and_drops_the_coding_guides(registered: None) -> None:
    plain = build_prompt(TASK)

    with config.overridden("record.packet", "nestforge"):
        sut = build_prompt(TASK)

    assert WHAT_IT_DOES.splitlines()[0] not in plain
    assert sut.index(WHAT_IT_DOES.splitlines()[0]) < sut.index("## Benchmark")
    assert "## Delivery: the library NestForge assembles" in sut
    assert "## How to optimize" in plain
    assert "## How to optimize" not in sut
    assert "### Build flags per compiler family" not in sut


def test_the_full_packet_says_the_agent_may_write_kernels_where_the_structured_one_says_it_writes_no_code(
    registered: None,
) -> None:
    with config.overridden("record.packet", "nestforge-full"):
        sut = build_prompt(TASK)

    assert "You may rewrite any kernel's source" in sut
    assert "you write no code" not in sut and "You do not write code" not in sut
    assert "## Delivery: the library NestForge assembles" in sut
