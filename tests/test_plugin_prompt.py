# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The nestforge packet's prompt table reshapes both of the bench's prompts: the agent writes no code, delivers
finalize's library, reads what NestForge does right after the intro, and the workflow says which phases are its own."""

import pytest
from hpcagent_agent.plugins import Plugin
from hpcagent_bench import config, prompt_build
from hpcagent_bench.categories import Language, SourceMode
from hpcagent_bench.harness.prompts import build_prompt
from hpcagent_bench.harness.task import Task

from nestforge.paths import REPO_ROOT

PLUGIN = REPO_ROOT / "plugin" / "nestforge"
ABOUT = (PLUGIN / "prompt" / "about.md").read_text(encoding="utf-8")
TASK = Task("jacobi_1d", SourceMode.ANY, Language.CPP)


@pytest.fixture(scope="module")
def registered(nestforge_plugin: tuple[Plugin, ...]) -> None:
    pass


def test_the_table_switches_the_coding_blocks_off_and_inserts_what_nestforge_does_after_the_intro(
    registered: None,
) -> None:
    """The agent half is rendered by the bench's user-plugins CI (the bench checkout holds agent/prompt.md); here
    the compiled overlay reads every text and inserts what NestForge does after the intro."""
    sut = prompt_build.load_overlay("plugin:nestforge/nestforge").agent

    edits = dict(sut.set)
    assert {"build", "syntax_check", "index_buffers"} <= {k for k, e in edits.items() if e.enabled is False}
    assert [(i.after, i.block.text) for i in sut.insert] == [("intro", ABOUT)]
    assert "expand-library" in dict(edits["workflow"].vars)["WORKFLOW"]


def test_the_task_prompt_delivers_finalizes_library_and_drops_the_coding_guides(registered: None) -> None:
    plain = build_prompt(TASK)

    with config.overridden("record.packet", "nestforge"):
        sut = build_prompt(TASK)

    assert ABOUT.splitlines()[0] not in plain
    assert sut.index(ABOUT.splitlines()[0]) < sut.index("## Workflow") < sut.index("## Benchmark")
    assert "### 1. Moves (you)" in sut and "### 4. Kernel code (default)" in sut
    assert "## Delivery: the library NestForge assembles" in sut
    assert "## How to optimize" in plain
    assert "## How to optimize" not in sut
    assert "### Build flags per compiler family" not in sut


def test_the_full_packet_gives_the_agent_every_phase_but_placement(registered: None) -> None:
    with config.overridden("record.packet", "nestforge-full"):
        sut = build_prompt(TASK)

    assert "### 2. Scopes (you)" in sut and "### 3. Placement (default)" in sut
    assert "### 4. Kernel code (you)" in sut and "### 5. Compiler configuration (you)" in sut
    assert "## Delivery: the library NestForge assembles" in sut
