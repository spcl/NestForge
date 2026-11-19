# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The nestforge packet's prompt overlays reshape both of the bench's prompts: the agent writes no code, delivers
finalize's library, reads what NestForge does right after the intro, and the agent's prompt carries the guide of its
first phase, the one ``NESTFORGE_AGENT_PHASES`` names first."""

import pytest
from hpcagent_agent.plugins import Plugin
from hpcagent_bench import config, packets, prompt_build
from hpcagent_bench.categories import Language, SourceMode
from hpcagent_bench.harness.prompts import build_prompt
from hpcagent_bench.harness.task import Task

from nestforge.paths import REPO_ROOT
from nestforge.stages.phases import checked_phases

PLUGIN = REPO_ROOT / "plugin" / "nestforge"
ABOUT = (PLUGIN / "prompt" / "about.md").read_text(encoding="utf-8")
TASK = Task("jacobi_1d", SourceMode.ANY, Language.CPP)


@pytest.fixture(scope="module")
def registered(nestforge_plugin: tuple[Plugin, ...]) -> None:
    pass


def test_the_overlay_switches_the_coding_blocks_off_and_inserts_what_nestforge_does_after_the_intro(
    registered: None,
) -> None:
    """The agent half is rendered by the bench's user-plugins CI (the bench checkout holds agent/prompt.md); here
    the overlay reads every text and inserts what NestForge does after the intro."""
    sut = prompt_build.load_overlay("nestforge:prompt/overlay.yaml").agent

    edits = dict(sut.set)
    assert {"build", "syntax_check", "index_buffers"} <= {k for k, e in edits.items() if e.enabled is False}
    assert [(i.after, i.block.text) for i in sut.insert] == [("intro", ABOUT)]


def test_the_task_prompt_delivers_finalizes_library_and_drops_the_coding_guides(registered: None) -> None:
    plain = build_prompt(TASK)

    with config.overridden("record.packet", "nestforge"):
        sut = build_prompt(TASK)

    assert ABOUT.splitlines()[0] not in plain
    assert sut.index(ABOUT.splitlines()[0]) < sut.index("## Benchmark")
    assert "## Delivery: the library NestForge assembles" in sut
    assert "## How to optimize" in plain
    assert "## How to optimize" not in sut
    assert "### Build flags per compiler family" not in sut


@pytest.mark.parametrize("packet", ["nestforge", "nestforge-full", "nestforge-full-no-moves", "nestforge-subagents"])
def test_a_packets_prompt_carries_the_guide_of_the_first_phase_its_agent_runs(registered: None, packet: str) -> None:
    resolved = packets.resolve(packet, "cpp")
    names = [n for n in dict(resolved.env).get("NESTFORGE_AGENT_PHASES", "").split(",") if n]
    first = checked_phases(names)[0]

    guides = [
        i.block.text
        for overlay in prompt_build.overlays_for((), packet)
        for i in overlay.agent.insert
        if i.block.id == "nestforge_phase"
    ]

    assert guides == [(PLUGIN / "prompt" / "phases" / f"{first}.md").read_text(encoding="utf-8")]
