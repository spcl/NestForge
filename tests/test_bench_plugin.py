# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The NestForge bench plugin: loaded through HPCAgent-Bench's own plugin loader, every tool registered, the
structured and kernel phases served with performance feedback, and ``finalize``'s library graded by the judge."""

import asyncio
import contextlib
import copy
import gc
import inspect
import json
import multiprocessing
import os
import pathlib
import re
import subprocess
import sys
import threading
import tomllib
from collections.abc import AsyncIterator, Callable, Iterator
from http import HTTPMethod, HTTPStatus
from types import SimpleNamespace
from typing import Any

import anthropic
import hpcagent_agent.tools.next_phase  # noqa: F401 -- registers a tool of the packet
import pytest
from anthropic.types import Message
from dace.codegen.common import get_gpu_backend
from hpcagent_agent import plugins as agent_plugins
from hpcagent_agent.driver import agent_driver
from hpcagent_agent.harness import run_anthropic, runner_common, subagents
from hpcagent_agent.plugins import Plugin
from hpcagent_agent.tools import http_json, registry, submit
from hpcagent_bench import config, packets, plugins
from hpcagent_bench.api import RunConfig
from hpcagent_bench.categories import Baseline, InputMode, Language, Oracle, Preset, SourceMode
from hpcagent_bench.columns import FRAMEWORKS
from hpcagent_bench.config import SubmitFeedback
from hpcagent_bench.harness import sandbox, tools
from hpcagent_bench.harness.envelope import Submission
from hpcagent_bench.harness.judge_scheduler import DeviceSlot
from hpcagent_bench.harness.judge_wire import GradeAnswer, JudgeAnswer
from hpcagent_bench.harness.routes import ROUTES, RouteRequest, next_phase
from hpcagent_bench.harness.service import JUDGE_FAULT_CAUSE, JudgeHandler, ServiceConfig, make_server
from hpcagent_bench.harness.task import Task
from hpcagent_bench.skill_frontmatter import split_frontmatter
from hpcagent_bench.vocabulary import PACKETS
from stub_session import Stub, install

import nestforge_bench
from nestforge.paths import REPO_ROOT
from nestforge.records import KernelStatus, MoveStatus
from nestforge.server import CALLS, STOP_GRACE_S, TOOL_TIMEOUT_S, Reply, Setup
from nestforge.session import KernelOutcome, MoveResult
from nestforge_bench import ReferenceOptimizer

PLUGIN = REPO_ROOT / "plugin" / "nestforge"
PAGE = PLUGIN / "skills" / "nestforge" / "SKILL.md"
KERNEL = "loop_level_reasoning/fuse_diamond/fuse_diamond"
#: The bench tools the packet replaces: each takes no code and sends the library finalize built.
OVERRIDDEN = {"profile", "score", "submit"}
#: The device slot a fake judge hands out.
SLOT = 3
#: Seconds a spawned judge has to import the bench and bind its port.
JUDGE_START_TIMEOUT_S = 300.0
#: Seconds one tuner episode may take; the bench's own user-plugins test allows the same.
TUNER_EPISODE_TIMEOUT_S = 900
#: One valid call of tools that read, change or measure.
SLOT_CALLS: dict[str, dict[str, object]] = {
    "get_cpf": {},
    "view_tree": {},
    "list_moves": {},
    "applied_moves": {},
    "apply_move": {"kind": "fuse", "labels": ["map_0", "map_1"]},
    "run_defaults": {"stage": "moves"},
    "set_scope": {"labels": ["map_0"]},
    "get_performance": {},
    "finalize": {},
    "write_kernel": {"name": "kernel_0", "source": "x", "language": "cpp"},
    "tune_kernel": {"name": "kernel_0"},
}


class FakeJudge(JudgeHandler):
    """The part of ``service.JudgeHandler`` a route uses; keeps the last answer."""

    def __init__(self) -> None:
        self.cfg = RunConfig(preset=Preset.S)
        self.code = 0
        self.answer: dict[str, object] = {}
        self.gone = threading.Event()
        self.slots = 0
        self.terminal = False

    def reply(self, code: int, payload: JudgeAnswer) -> None:
        self.code, self.answer = code, payload.to_json()

    def watch_client(self, finished: threading.Event) -> None:
        finished.wait()

    @contextlib.contextmanager
    def device_slot(self, terminal: bool = False) -> Iterator[DeviceSlot]:
        self.slots += 1
        self.terminal = terminal
        yield DeviceSlot("cpu", SLOT)


@pytest.fixture(autouse=True)
def shared_folder(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Journals live on disk: each test has its own shared folder, so no run repeats another's moves."""
    monkeypatch.setenv("HPCAGENT_BENCH_SHARED_DIR", str(tmp_path))


@pytest.fixture(scope="module")
def loaded(nestforge_plugin: tuple[Plugin, ...]) -> Iterator[None]:
    assert [plugin.name for plugin in nestforge_plugin if plugin.root == PLUGIN] == ["nestforge"]
    yield
    # every session process this module started, so no other test inherits a live child
    episodes = sys.modules[ROUTES["nestforge"].handler.__module__].EPISODES
    for _, (episode, _) in list(episodes.states.items()):
        for session in episode.sessions.values():
            session.close()


def post(
    tool: str, episode: str = "episode-cpu", kernel: str = KERNEL, judge: FakeJudge | None = None, **body: object
) -> FakeJudge:
    judge = judge if judge is not None else FakeJudge()
    request = RouteRequest(HTTPMethod.POST, ["nestforge", tool], {}, {"kernel": kernel, **body}, episode)
    ROUTES["nestforge"].handler(judge, request)
    return judge


@pytest.fixture
def stub(loaded: None, monkeypatch: pytest.MonkeyPatch) -> Iterator[Stub]:
    """Every session the route starts answers from a :class:`Stub`; its episodes are forgotten afterwards."""
    episodes = sys.modules[ROUTES["nestforge"].handler.__module__].EPISODES
    before = set(episodes.states)
    yield install(monkeypatch, Stub())
    for episode in set(episodes.states) - before:
        episodes.drop(episode)


def test_every_phase_registers_its_tools(loaded: None) -> None:
    names = {"get_cpf", "apply_move", "set_scope", "list_kernels", "write_kernel", "tune_kernel", "get_performance"}
    assert names <= registry.REGISTRY.keys()
    assert all(not registry.REGISTRY[name].core for name in names)


def test_a_scope_set_through_the_route_can_be_unset_again(loaded: None) -> None:
    rows = [
        line.lstrip("|`- ").split()[0]
        for line in post("get_cpf", episode="episode-scope").answer["result"].splitlines()[1:]
    ]
    states = [row for row in rows if row.startswith("state")]
    tree = post("view_tree", episode="episode-scope", expand=states).answer["result"]
    maps = [line.lstrip("|`- ").split()[0] for line in tree.splitlines() if line.lstrip("|`- ").startswith("doall")]

    scoped = post("set_scope", episode="episode-scope", labels=maps[:2]).answer["result"]
    unset = post("unset_scope", episode="episode-scope", name=scoped["reason"]).answer["result"]

    assert (scoped["status"], unset["status"]) == ("applied", "applied"), (scoped, unset)


def test_each_manifest_phase_serves_the_tools_of_its_wire_phase(loaded: None) -> None:
    from hpcagent_bench_plugins.nestforge.wire import TOOLS, Phase

    manifest = tomllib.loads((PLUGIN / "plugin.toml").read_text())

    for name, phase in manifest["phases"].items():
        assert sorted(phase.get("tools", [])) == sorted(t for t, d in TOOLS.items() if d.phase == name), name
    assert sorted(manifest["packets"]["nestforge"]["tools"]) == sorted(
        (*(t for t, d in TOOLS.items() if d.phase == Phase.ANY), "profile", "score", "submit")
    )


def test_a_blind_agent_is_not_served_a_measurement_and_the_tool_list_says_one_submission(loaded: None) -> None:
    body = {"LANGUAGE": "cpp", registry.PACKET_TOOLS_ENV: ",".join(PACKETS["nestforge"].tools)}

    blind = registry.served(registry.Setup.read({**body, "AGENT_SUBMISSION_MODE": "blind"}))
    single = registry.served(registry.Setup.read({**body, "AGENT_SUBMISSION_MODE": "single"}))

    assert "get_performance" not in blind and "score" not in blind
    assert "get_performance" in single
    assert single["submit"].prompt == submit.PROMPT


def test_the_packet_exposes_the_structured_phase_and_its_page_and_every_tool_is_served(loaded: None) -> None:
    from hpcagent_bench_plugins.nestforge.wire import TOOLS, Phase

    packet = PACKETS["nestforge"].tools

    assert packet[-1] == plugins.NEXT_PHASE
    assert {TOOLS[tool].phase for tool in packet[:-1] if tool in TOOLS} == {Phase.MOVES, Phase.ANY}
    assert OVERRIDDEN <= registry.OVERRIDES.keys() and OVERRIDDEN <= set(packet)
    assert "set_scope" not in packet and "list_move_kinds" in packet
    assert split_frontmatter(PAGE.read_text())[0].name == "nestforge"
    assert PACKETS["nestforge"].skills == ("nestforge",)
    assert TOOLS.keys() == CALLS.keys()
    for tool in TOOLS:
        params = inspect.signature(CALLS[tool]).parameters
        missing = TOOLS[tool].args.model_fields.keys() - params.keys()
        assert not missing, f"{tool}: the session call takes no {sorted(missing)}"


def test_profile_takes_no_code_and_sends_the_library_finalize_built_with_its_scratch(
    loaded: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run's agents sent ``source`` the library judge refuses, or the library without the ``workspace_bytes`` that
    library's scratch needs, a NULL workspace and a SIGSEGV. Neither can be said now, and finalize gets no instrument:
    its arguments are forbidden."""
    calls: list[tuple[str, dict[str, object]]] = []

    def judge(route: str, body: dict[str, object]) -> dict[str, object]:
        calls.append((route, body))
        return {"result": {"library": "/shared/lib.so", "workspace_bytes": "16*K + 128"}, "profiled": True}

    monkeypatch.setattr(http_json, "post_judge", judge)
    setup = registry.Setup.read({"LANGUAGE": "cpp", registry.PACKET_TOOLS_ENV: ",".join(PACKETS["nestforge"].tools)})
    sut = registry.served(setup)["profile"]

    answer = sut.run({"tool": "none", "threads": 24, "size": "S"})
    refused = sut.run({"tool": "none", "source": "int main;"})

    arguments = sut.input_schema["properties"].keys()
    assert {"tool", "threads", "counter_group", "size"} <= arguments
    assert not {"source", "source_file", "library", "workspace_bytes", "build"} & arguments
    assert answer["profiled"] is True and refused["ok"] is False, (answer, refused)
    assert [route for route, _ in calls] == ["/nestforge/finalize", "/profile"], "the refused call reaches no judge"
    assert "tool" not in calls[0][1]
    assert calls[1][1]["library"] == "/shared/lib.so" and calls[1][1]["workspace_bytes"] == "16*K + 128"
    assert (calls[1][1]["tool"], calls[1][1]["threads"], calls[1][1]["size"]) == ("none", 24, "S")


def test_the_full_packet_serves_every_tool_and_lists_reads_as_read_only(loaded: None) -> None:
    from hpcagent_bench_plugins.nestforge.wire import TOOLS

    packet = PACKETS["nestforge-full"].tools

    assert sorted(packet) == sorted((*TOOLS, *OVERRIDDEN, plugins.NEXT_PHASE))
    specs = {name: registry.REGISTRY[name].spec for name in TOOLS}
    hints = {name: spec.hints for name, spec in specs.items() if isinstance(spec, registry.Spec)}
    assert hints.keys() == TOOLS.keys()
    assert {name for name, hint in hints.items() if hint.read_only} == {
        "get_cpf",
        "view_tree",
        "list_move_kinds",
        "list_moves",
        "list_kernels",
        "read_kernel",
        "get_performance",
        "applied_moves",
    }
    assert hints["write_kernel"].destructive and not hints["apply_move"].destructive


def test_a_session_that_dies_is_the_judges_fault_and_the_next_call_starts_a_fresh_one(stub: Stub) -> None:
    stub.dies_on = "apply_move"

    died = post("apply_move", "episode-crash", kind="fuse", labels=["map_0", "map_1"])
    again = post("get_cpf", "episode-crash")

    assert (died.code, died.answer["cause"]) == (HTTPStatus.INTERNAL_SERVER_ERROR, JUDGE_FAULT_CAUSE), died.answer
    assert "fresh session" in died.answer["error"]
    assert again.code == HTTPStatus.OK and len(stub.started) == 2


def test_a_fresh_session_repeats_every_change_the_last_one_answered_so_no_move_is_lost(stub: Stub) -> None:
    first = {"kind": "fuse", "labels": ["map_0", "map_1"]}
    post("apply_move", "episode-replay", **first)
    post("view_tree", "episode-replay")
    post("set_scope", "episode-replay", labels=["map_0"])
    stub.dies_on = "get_performance"

    died = post("get_performance", "episode-replay")
    stub.dies_on = ""
    before = len(stub.calls)
    answered = post("list_moves", "episode-replay")

    assert died.code == HTTPStatus.INTERNAL_SERVER_ERROR and answered.code == HTTPStatus.OK
    repeated = [tool for tool, _ in stub.calls[before:]]
    assert repeated == ["apply_move", "set_scope", "list_moves"], "the changes in order, then the call; no read"
    assert len(stub.started) == 2


MOVE = {"kind": "fuse", "labels": ["map_0", "map_1"]}


def test_a_session_that_died_between_calls_is_replaced_and_repeats_the_agents_moves(stub: Stub) -> None:
    post("apply_move", "episode-idle", **MOVE)
    route = sys.modules[ROUTES["nestforge"].handler.__module__]
    route.episode_of("episode-idle").sessions[KERNEL].alive = False
    before = len(stub.calls)

    answered = post("get_cpf", "episode-idle")

    assert answered.code == HTTPStatus.OK and len(stub.started) == 2
    assert [tool for tool, _ in stub.calls[before:]] == ["apply_move", "get_cpf"]


def restart_judge() -> None:
    """What a judge restart or an idle eviction leaves: nothing in memory."""
    episodes = sys.modules[ROUTES["nestforge"].handler.__module__].EPISODES
    for episode in list(episodes.states):
        episodes.drop(episode)


def test_a_judge_restart_repeats_the_agents_changes_so_nothing_grades_the_canonical_program(stub: Stub) -> None:
    post("apply_move", "episode-restart", **MOVE)
    post("set_scope", "episode-restart", labels=["map_0"])
    restart_judge()
    before = len(stub.calls)

    answered = post("finalize", "episode-restart")

    assert answered.code == HTTPStatus.OK
    assert [tool for tool, _ in stub.calls[before:]] == ["apply_move", "set_scope", "finalize"]
    assert stub.calls[before][1]["labels"] == MOVE["labels"] and len(stub.started) == 2


@pytest.mark.parametrize("line", ["not json", '["apply_move"]', "7"])
def test_a_journal_that_cannot_be_read_fails_loudly_and_starts_no_session(stub: Stub, line: str) -> None:
    post("apply_move", "episode-torn", **MOVE)
    restart_judge()
    route = sys.modules[ROUTES["nestforge"].handler.__module__]
    with route.journal_file("episode-torn", KERNEL).open("a") as journal:
        journal.write(line + "\n")
    started = len(stub.started)

    judge = post("finalize", "episode-torn")

    assert (judge.code, judge.answer["cause"]) == (HTTPStatus.INTERNAL_SERVER_ERROR, JUDGE_FAULT_CAUSE), judge.answer
    assert "journal" in judge.answer["error"] and len(stub.started) == started


def test_a_change_a_rebuild_refuses_fails_loudly_instead_of_grading_a_program_without_it(stub: Stub) -> None:
    post("apply_move", "episode-diverged", **MOVE)
    restart_judge()
    refused = MoveResult(status=MoveStatus.NOT_FOUND, kind="fuse", labels=("map_0",), reason="no such row")
    stub.replies["apply_move"] = Reply(result=refused)

    judge = post("finalize", "episode-diverged")

    assert (judge.code, judge.answer["cause"]) == (HTTPStatus.INTERNAL_SERVER_ERROR, JUDGE_FAULT_CAUSE), judge.answer
    assert "no such row" in judge.answer["error"] and "finalize" not in [tool for tool, _ in stub.calls]


def test_a_refused_change_is_not_journaled(stub: Stub) -> None:
    refused = MoveResult(status=MoveStatus.ILLEGAL, kind="fuse", labels=("map_0",), reason="cannot fuse")
    stub.replies["apply_move"] = Reply(result=refused)

    answered = post("apply_move", "episode-refused", **MOVE)

    route = sys.modules[ROUTES["nestforge"].handler.__module__]
    assert answered.code == HTTPStatus.OK and route.journal_of("episode-refused", KERNEL) == []


def test_a_kernel_written_without_a_check_is_journaled_as_the_change_it_is(stub: Stub) -> None:
    written = KernelOutcome(kernel="kernel_0", status=KernelStatus.UNCHECKED, reason="the whole program checks it")
    stub.replies["write_kernel"] = Reply(result=written)

    post("write_kernel", "episode-unchecked", **SLOT_CALLS["write_kernel"])

    route = sys.modules[ROUTES["nestforge"].handler.__module__]
    assert [tool for tool, _ in route.journal_of("episode-unchecked", KERNEL)] == ["write_kernel"]


def test_a_change_that_cannot_be_journaled_is_dropped_with_its_session_and_the_agent_is_told(
    stub: Stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    route = sys.modules[ROUTES["nestforge"].handler.__module__]

    def full(episode_id: str, kernel: str, call: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(route, "record", full)

    judge = post("apply_move", "episode-full", **MOVE)

    assert (judge.code, judge.answer["cause"]) == (HTTPStatus.INTERNAL_SERVER_ERROR, JUDGE_FAULT_CAUSE), judge.answer
    assert "disk full" in judge.answer["error"] and not route.episode_of("episode-full").sessions


def test_a_relaunched_attempt_works_in_its_own_folder_and_inherits_no_moves(
    stub: Stub, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    (tmp_path / "agent-0").mkdir()
    (tmp_path / "agent-1").mkdir()
    folders = tmp_path / sandbox.AGENT_FOLDERS_LOG
    line = '{{"episode_id": "episode-again", "folder": "{}"}}\n'
    folders.write_text(line.format("agent-0"))
    monkeypatch.setenv("RUN_DIR", str(tmp_path))
    post("apply_move", "episode-again", **MOVE)

    folders.write_text(line.format("agent-0") + line.format("agent-1"))
    post("get_cpf", "episode-again")

    assert [setup.shared_root for setup in stub.started] == [str(tmp_path / f"agent-{n}") for n in range(2)]
    assert [tool for tool, _ in stub.calls] == ["apply_move", "get_cpf"], "the second attempt starts anew"


def test_an_unused_episode_ends_its_session_within_the_hour_and_never_under_a_running_call(loaded: None) -> None:
    idle = sys.modules[ROUTES["nestforge"].handler.__module__].EPISODES.idle_seconds

    assert TOOL_TIMEOUT_S + STOP_GRACE_S < idle <= 3600.0


def test_a_read_and_a_move_take_no_device_slot_and_what_runs_code_takes_one(stub: Stub) -> None:
    slots = {tool: post(tool, "episode-slots", **body).slots for tool, body in SLOT_CALLS.items()}

    assert slots == {
        tool: int(tool in {"get_performance", "finalize", "write_kernel", "tune_kernel"}) for tool in slots
    }


def test_only_the_finalize_of_a_submission_queues_as_the_submission_does(
    stub: Stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def judge(route: str, body: dict[str, object]) -> dict[str, object]:
        calls.append(route)
        return {"result": {"library": "/shared/lib.so", "workspace_bytes": "0"}}

    monkeypatch.setattr(http_json, "post_judge", judge)
    setup = registry.Setup.read({"LANGUAGE": "cpp", registry.PACKET_TOOLS_ENV: ",".join(PACKETS["nestforge"].tools)})
    registry.served(setup)["submit"].run({})
    registry.served(setup)["score"].run({})
    terminal = FakeJudge()
    request = {"terminal": ["1"]}, {"kernel": KERNEL}, "episode-terminal"
    ROUTES["nestforge"].handler(terminal, RouteRequest(HTTPMethod.POST, ["nestforge", "finalize"], *request))

    assert [calls[0], calls[2]] == ["/nestforge/finalize?terminal=1", "/nestforge/finalize"]
    assert terminal.slots == 1 and terminal.terminal is True
    assert post("finalize", "episode-terminal").terminal is False


def test_a_measuring_call_reaches_its_session_with_its_slot_and_a_read_with_none(stub: Stub) -> None:
    post("get_performance", "episode-pinned")
    post("view_tree", "episode-pinned")

    assert [held[0] for held in stub.held] == [SLOT, None]


def written_then_lost(episode: str) -> None:
    """``episode``'s journal holds a written kernel, and its session is gone."""
    post("write_kernel", episode, **SLOT_CALLS["write_kernel"])
    post("tune_kernel", episode, name="kernel_0")
    restart_judge()


def test_a_read_that_rebuilds_a_written_kernel_takes_a_slot_measures_nothing_and_sees_its_client(stub: Stub) -> None:
    written_then_lost("episode-rebuild")
    before = len(stub.calls)

    judge = post("view_tree", "episode-rebuild")

    assert judge.slots == 1
    assert [tool for tool, _ in stub.calls[before:]] == ["write_kernel", "view_tree"], "a sweep is finalize's to repeat"
    assert stub.held[before:] == [(SLOT, False, True), (SLOT, True, True)]


def test_a_read_after_a_rebuild_takes_no_slot(stub: Stub) -> None:
    written_then_lost("episode-rebuilt-once")
    post("view_tree", "episode-rebuilt-once")

    assert post("view_tree", "episode-rebuilt-once").slots == 0


def test_a_rebuild_out_of_budget_fails_loudly_and_the_next_call_tries_again(stub: Stub) -> None:
    written_then_lost("episode-out")
    stub.replies["write_kernel"] = Reply(error="ran out of its 1500 s budget", stopped=True)

    judge = post("finalize", "episode-out")

    assert (judge.code, judge.answer["cause"]) == (HTTPStatus.INTERNAL_SERVER_ERROR, JUDGE_FAULT_CAUSE), judge.answer
    stub.replies["write_kernel"] = Reply(result={"status": "ok", "reason": ""})
    assert post("finalize", "episode-out").code == HTTPStatus.OK


def test_an_episodes_second_measuring_call_waits_without_holding_a_slot(stub: Stub) -> None:
    episode = sys.modules[ROUTES["nestforge"].handler.__module__].episode_of("episode-queue")
    judge = FakeJudge()
    judge.gone.set()

    with episode.measuring:
        left = post("get_performance", "episode-queue", judge=judge)

    assert left.slots == 0 and left.code == 0 and stub.calls == []


def test_a_call_whose_client_left_while_it_waited_for_the_session_never_runs(stub: Stub) -> None:
    episode = sys.modules[ROUTES["nestforge"].handler.__module__].episode_of("episode-gone")
    judge = FakeJudge()
    judge.gone.set()

    with episode.lock:
        left = post("apply_move", "episode-gone", judge=judge, kind="fuse", labels=["map_0", "map_1"])

    assert left.code == 0 and stub.calls == [], "no answer, and the session never saw it"


def test_a_retry_reads_the_answer_of_a_change_whose_client_left_instead_of_applying_it_twice(stub: Stub) -> None:
    move = {"kind": "fuse", "labels": ["map_0", "map_1"], "tile_sizes": [], "implementation": None}
    judge = FakeJudge()
    judge.gone.set()
    route = sys.modules[ROUTES["nestforge"].handler.__module__]
    request = RouteRequest(HTTPMethod.POST, [], {}, {}, "episode-retry")
    reply = route.run_call(route.episode_of("episode-retry"), judge, request, KERNEL, ("apply_move", move), judge.gone)

    retried = post("apply_move", "episode-retry", **move)

    assert reply.result == {"status": "applied", "reason": ""}
    assert retried.code == HTTPStatus.OK and retried.answer["result"] == reply.result
    assert [tool for tool, _ in stub.calls].count("apply_move") == 1, "the retry did not move the program again"


def test_a_measuring_call_that_ran_out_of_budget_says_so_and_keeps_the_session(stub: Stub) -> None:
    stub.replies["get_performance"] = Reply(error="ran out of its 1500 s budget", stopped=True)

    stopped = post("get_performance", "episode-budget")

    assert stopped.code == HTTPStatus.SERVICE_UNAVAILABLE and "budget" in stopped.answer["error"]
    assert post("view_tree", "episode-budget").code == HTTPStatus.OK and len(stub.started) == 1


def test_a_session_that_cannot_start_is_the_judges_fault(loaded: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def refused(setup: object) -> None:
        raise LookupError("canonicalize: RecursionError")

    monkeypatch.setattr(sys.modules[ROUTES["nestforge"].handler.__module__], "SessionProcess", refused)

    judge = post("get_cpf", "episode-unstartable")

    assert (judge.code, judge.answer["cause"]) == (HTTPStatus.INTERNAL_SERVER_ERROR, JUDGE_FAULT_CAUSE), judge.answer


def test_the_subagent_packet_adds_the_kernel_optimizer_to_the_full_packet(loaded: None) -> None:
    packet, full = PACKETS["nestforge-subagents"], PACKETS["nestforge-full"]
    agent = subagents.parse((PLUGIN / "agents" / "kernel-optimizer.md").read_text())

    assert packet.agents == (agent.name,) and sorted(packet.tools) == sorted(full.tools)
    assert agent.tools is None, "a subagent lists no tools: the harnesses name them differently and the gate holds"
    assert packet.phases == full.phases


def test_the_two_stage_name_is_the_full_packet_and_only_that_packet_answers_to_it(loaded: None) -> None:
    resolved = packets.resolve("nestforge-two-stage", "cpp")

    assert resolved == packets.resolve("nestforge-full", "cpp")
    assert next_phase.phases_of("nestforge-two-stage") == PACKETS["nestforge-full"].phases
    assert PACKETS.canonical("nestforge-two-stage") == "nestforge-full"
    assert "nestforge-two-stage" not in PACKETS.entries, "an alias is no packet of its own"


def test_a_bench_setup_of_the_subagent_packet_stages_and_offers_the_kernel_optimizer(
    loaded: None, tmp_path: pathlib.Path
) -> None:
    environ = {
        agent_plugins.USER_PLUGIN_DIRS_ENV: str(PLUGIN),
        agent_plugins.PACKET_AGENTS_ENV: ",".join(packets.resolve("nestforge-subagents", "cpp").agents),
    }

    staged = agent_plugins.stage_agents(tmp_path, environ)
    offered = subagents.load(tmp_path, environ)

    shipped = subagents.parse((PLUGIN / "agents" / "kernel-optimizer.md").read_text())
    assert staged == ("kernel-optimizer",) and offered == (shipped,)
    assert subagents.task_schema(offered)["properties"]["subagent_type"]["enum"] == [shipped.name]
    assert shipped.name in subagents.task_description(offered)


def walk(episode: str, monkeypatch: pytest.MonkeyPatch) -> Callable[[], dict[str, object]]:
    """``next_phase`` of ``episode`` on the full packet."""
    monkeypatch.setattr(next_phase, "run_phases", lambda: PACKETS["nestforge-full"].phases)

    def step() -> dict[str, object]:
        judge = FakeJudge()
        next_phase.serve(judge, RouteRequest(HTTPMethod.POST, ["next_phase"], {}, {"kernel": KERNEL}, episode))
        assert judge.code == HTTPStatus.OK, judge.answer
        return judge.answer

    return step


@pytest.mark.integration
def test_the_full_packet_goes_from_structured_to_each_kernel_and_a_new_round_reopens_the_moves(
    loaded: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode = "episode-walk"
    step = walk(episode, monkeypatch)
    moves = post("list_moves", episode=episode).answer["result"]
    assert moves

    scopes, kernels = step(), step()
    names = [kernel["name"] for kernel in post("list_kernels", episode=episode).answer["result"]]
    closed = post("set_scope", episode=episode, labels=["doall0"])
    written = [
        post(
            "write_kernel",
            episode=episode,
            name=name,
            source=post("read_kernel", episode=episode, name=name).answer["result"],
            language="cpp",
        ).answer["result"]["status"]
        for name in names
    ]
    codegen = step()
    tuned = post("tune_kernel", episode=episode, name=names[0], compilers=["gcc"])
    again = step()

    assert (scopes["phase"], kernels["phase"], codegen["phase"]) == ("scopes", "kernels", "codegen")
    assert names and closed.code == HTTPStatus.BAD_REQUEST and "scopes are fixed" in closed.answer["error"]
    assert written == ["ok"] * len(names) and tuned.code == HTTPStatus.OK
    assert (again["phase"], again["round"]) == ("moves", 2)
    assert post("list_moves", episode=episode).answer["result"] == moves


def test_next_phase_runs_the_default_scopes_and_placement_and_returns_to_the_moves(
    stub: Stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(next_phase, "run_phases", lambda: PACKETS["nestforge"].phases)
    judge = FakeJudge()
    request = RouteRequest(HTTPMethod.POST, ["next_phase"], {}, {"kernel": KERNEL}, "episode-round")

    next_phase.serve(judge, request)

    assert judge.code == HTTPStatus.OK, judge.answer
    assert (judge.answer["phase"], judge.answer["round"]) == ("moves", 2)
    assert [(run["phase"], run["status"]) for run in judge.answer["ran"]] == [
        ("scopes", 200),
        ("placement", 200),
        ("round", 200),
    ]
    assert [call for call in stub.calls if call[0] == "run_defaults"] == [
        ("run_defaults", {"stage": "scopes"}),
        ("run_defaults", {"stage": "placement"}),
        ("run_defaults", {"stage": "round"}),
    ]


@pytest.mark.integration
def test_next_phase_keeps_the_doall_moves_and_finalize_still_gets_the_kernels(
    loaded: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(next_phase, "run_phases", lambda: PACKETS["nestforge"].phases)
    before = post("list_moves", "episode-stays").answer["result"]
    judge = FakeJudge()
    request = RouteRequest(HTTPMethod.POST, ["next_phase"], {}, {"kernel": KERNEL}, "episode-stays")

    next_phase.serve(judge, request)

    assert judge.code == HTTPStatus.OK, judge.answer
    assert post("list_moves", "episode-stays").answer["result"] == before
    assert post("finalize", "episode-stays").answer["result"]["kernels"]


def test_a_tool_that_refuses_the_agents_call_is_still_a_bad_request(stub: Stub) -> None:
    stub.replies["apply_move"] = Reply(error="ValueError: doall-fusion takes 2 label(s); got 1")

    judge = post("apply_move", "episode-stale", kind="fuse", labels=["map_0"])

    assert judge.code == HTTPStatus.BAD_REQUEST and "takes 2 label" in judge.answer["error"], judge.answer


def test_concurrent_calls_of_one_episode_start_one_session_and_reach_it_one_at_a_time(stub: Stub) -> None:
    calls = [threading.Thread(target=post, args=("get_cpf", "episode-race")) for _ in range(4)]
    for call in calls:
        call.start()
    for call in calls:
        call.join()

    assert len(stub.started) == 1, "a second session of the same kernel would leak until garbage collection"
    assert len(stub.calls) == 4 and stub.most_in_flight == 1, "two calls on one pipe read each other's answers"


def test_an_unknown_argument_is_refused(loaded: None) -> None:
    assert post("list_moves", depth=3).code == HTTPStatus.BAD_REQUEST


@pytest.mark.integration
def test_the_structured_phase_returns_the_tree_and_get_performance_measures_a_move(loaded: None) -> None:
    tree = post("get_cpf")
    moves = post("list_moves")
    first = moves.answer["result"][0]

    moved = post("apply_move", **first)
    performance = post("get_performance").answer["performance"]

    assert tree.code == HTTPStatus.OK and "doall" in str(tree.answer["result"])
    assert moved.code == HTTPStatus.OK and moved.answer["result"]["status"] == "applied"
    assert moved.answer["performance"] is None, "a move answers at once; a time is asked for"
    structured = performance["structured"]
    assert "error" not in structured and structured["time_us"] > 0
    assert structured["correct"] is True, structured
    assert set(performance["offloading"]["devices"].values()) == {"cpu"}
    assert performance["kernels"] and all(kernel["correct"] for kernel in performance["kernels"])
    assert post("get_performance").answer["performance"] == performance, "the unchanged program is measured once"


def kernel_round(episode: str, kernel: str = KERNEL) -> tuple[str, str]:
    """Default scopes on ``episode``, then the first kernel's name and source."""
    assert post("run_defaults", episode=episode, kernel=kernel, stage="scopes").code == HTTPStatus.OK
    name = post("list_kernels", episode=episode, kernel=kernel).answer["result"][0]["name"]
    return name, post("read_kernel", episode=episode, kernel=kernel, name=name).answer["result"]


@pytest.mark.integration
def test_a_real_session_restarted_from_its_journal_is_the_program_the_agent_left(loaded: None) -> None:
    move = post("list_moves", "episode-rebuilt").answer["result"][0]
    post("apply_move", "episode-rebuilt", **move)
    route = sys.modules[ROUTES["nestforge"].handler.__module__]
    assert route.journal_of("episode-rebuilt", KERNEL)[0][1]["labels"] == list(move["labels"])
    tree = post("get_cpf", "episode-rebuilt").answer["result"]
    applied = post("applied_moves", "episode-rebuilt").answer["result"]

    restart_judge()

    assert post("get_cpf", "episode-rebuilt").answer["result"] == tree
    assert post("applied_moves", "episode-rebuilt").answer["result"] == applied and applied


@pytest.mark.integration
def test_each_finalize_keeps_its_library_and_records_what_was_graded(loaded: None) -> None:
    first = post("finalize", "episode-twice").answer["result"]
    moved = post("run_defaults", "episode-twice", stage="moves")
    second = post("finalize", "episode-twice").answer["result"]

    assert moved.code == HTTPStatus.OK and first["library"] != second["library"]
    assert pathlib.Path(first["library"]).is_file() and pathlib.Path(second["library"]).is_file()
    folder = pathlib.Path(first["library"]).parent.parent
    lines = [json.loads(line) for line in (folder / "applied_moves.jsonl").read_text().splitlines()]
    assert [line["library"] for line in lines] == [first["library"], second["library"]]
    assert [move["kind"] for move in lines[1]["moves"]] == ["defaults-moves"]
    assert lines[1]["kernels"] == second["kernels"] and lines[1]["device_code"] is False


@pytest.mark.integration
def test_a_kernel_written_back_unchanged_is_linked_and_a_wrong_entry_is_refused(loaded: None) -> None:
    name, source = kernel_round("episode-kernel")

    written = post("write_kernel", episode="episode-kernel", name=name, source=source, language="cpp")
    renamed = source.replace(f"void {name}(", f"void {name}_other(", 1)
    refused = post("write_kernel", episode="episode-kernel", name=name, source=renamed, language="cpp")

    assert written.answer["result"]["status"] == "ok", written.answer
    assert written.answer["performance"]["structured"]["correct"] is True
    assert refused.answer["result"]["status"] == "refused", refused.answer


@pytest.fixture
def decoy(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> pathlib.Path:
    """A secret in the run folder, which the judge reads and agent code must not."""
    secret = tmp_path / "run" / "episode-secret"
    secret.parent.mkdir()
    secret.write_text("the-decoy-secret\n")
    monkeypatch.setenv("RUN_DIR", str(secret.parent))
    return secret


@pytest.mark.integration
def test_a_written_kernel_cannot_include_what_the_judge_hides(loaded: None, decoy: pathlib.Path) -> None:
    name, source = kernel_round("episode-include")

    written = post(
        "write_kernel", "episode-include", name=name, source=f'{source}\n#include "{decoy}"\n', language="cpp"
    )

    assert written.answer["result"]["status"] == "build-failed", written.answer
    assert "the-decoy-secret" not in written.answer["result"]["reason"]
    assert "No such file" in written.answer["result"]["reason"]


@pytest.mark.integration
def test_a_written_kernel_cannot_open_what_the_judge_hides_when_it_runs(loaded: None, decoy: pathlib.Path) -> None:
    name, source = kernel_round("episode-open")
    entry = re.search(rf"void {name}\([^)]*\)\s*\{{", source)
    assert entry is not None, source
    # the kernel skips its work if it reads the file, so its oracle calls a kernel outside the seal wrong
    guard = f'FILE* secret = fopen("{decoy}", "r"); if (secret != NULL) return;'
    spying = f"#include <cstdio>\n{source[: entry.end()]}{guard}{source[entry.end() :]}"

    written = post("write_kernel", "episode-open", name=name, source=spying, language="cpp")

    assert written.answer["result"]["status"] == "ok", written.answer
    assert written.answer["performance"]["structured"]["correct"] is True


@pytest.mark.integration
def test_a_tuned_kernel_reports_the_winning_configuration(loaded: None) -> None:
    name, _ = kernel_round("episode-tune")

    tuned = post("tune_kernel", episode="episode-tune", name=name, compilers=["gcc"]).answer

    assert tuned["result"]["cells"] > 1 and tuned["result"]["winner"] is not None, tuned
    assert tuned["performance"]["structured"]["correct"] is True


def serve_judge(ports: "multiprocessing.Queue[int]") -> None:
    """A judge taking library submissions at S, answering the whole grade, until terminated."""
    with (
        config.overridden("fuzz.hidden_correctness_presets", ["S"] * 5),
        config.overridden("service.submit_feedback", SubmitFeedback.FULL),
    ):
        server = make_server(
            "127.0.0.1", 0, ServiceConfig(baseline=Baseline.C, oracle=Oracle.AUTO, input_mode=InputMode.ANY, repeat=3)
        )
        ports.put(server.server_address[1])
        server.serve_forever()


def spawned_judge(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> Iterator[str]:
    """A judge in a spawned process, grading at S as the bench's own suite does. Not a thread of this one: a GPU
    grade initializes the device runtime in the judge's process, and a later test forking this one dies with it."""
    # The judge's shared folder, and the one the session process a test starts writes its library into.
    monkeypatch.setenv("HPCAGENT_BENCH_SHARED_DIR", str(tmp_path))
    monkeypatch.setenv("HPCAGENT_BENCH_FUZZ_SIZE_CAP", "4096")
    monkeypatch.setenv("HPCAGENT_BENCH_MEASUREMENT_TIMING_BACKEND", "min_of_k")
    monkeypatch.setenv("HPCAGENT_BENCH_SERVICE_PRESET", "S")
    monkeypatch.setenv("HPCAGENT_BENCH_FUZZ_ANCHOR", "S")
    spawn = multiprocessing.get_context("spawn")
    ports: multiprocessing.Queue[int] = spawn.Queue()
    judge = spawn.Process(target=serve_judge, args=(ports,))  # not daemonic: the judge forks its grades
    judge.start()
    try:
        yield f"http://127.0.0.1:{ports.get(timeout=JUDGE_START_TIMEOUT_S)}"
    finally:
        judge.terminate()
        judge.join()


@pytest.fixture
def judge_url(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> Iterator[str]:
    yield from spawned_judge(monkeypatch, tmp_path)


@pytest.fixture
def packet_judge(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> Iterator[str]:
    """A spawned judge of a run on the packet the test's parameter names, its routes and phase gate live, as a study's
    judge has them."""
    monkeypatch.setenv("HPCAGENT_BENCH_USER_PLUGIN_DIRS", str(PLUGIN))
    monkeypatch.setenv("HPCAGENT_BENCH_RECORD_PACKET", request.param)
    yield from spawned_judge(monkeypatch, tmp_path)


def graded(judge_url: str, kernel: str, final: dict[str, object], language: Language = Language.CPP) -> GradeAnswer:
    """``final``'s library as a ``language`` submission, graded by the judge at ``judge_url``."""
    submission = Submission(
        language=language, library=str(final["library"]), workspace_bytes=str(final["workspace_bytes"])
    )
    grade = tools.JudgeClient(judge_url).submit(submission, kernel)
    assert isinstance(grade, GradeAnswer), grade
    return grade


def test_the_reference_column_starts_on_the_preset_the_token_resolves_to(monkeypatch: pytest.MonkeyPatch) -> None:
    class Started(Exception):
        pass

    presets: list[str] = []

    def recording_start(setup: Setup) -> None:
        presets.append(setup.preset)
        raise Started

    monkeypatch.setattr(nestforge_bench, "start", recording_start)
    sut = ReferenceOptimizer("dace_cpu_canon_nestforge_reference")

    with config.scoped_environment({"HPCAGENT_BENCH_SERVICE_PRESET": "XL+fuzz"}), pytest.raises(Started):
        sut.tune(Task(KERNEL, SourceMode.ANY, Language.CPP), "canon_cpu")

    assert presets == ["fuzzed"]


def test_the_reference_column_leaves_an_episodes_library_in_the_episodes_folder(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FinalizedSession:
        def finalize(self, out: pathlib.Path) -> SimpleNamespace:
            library = out / "libkernel.so"
            library.touch()
            return SimpleNamespace(library=library, workspace_bytes="0")

    monkeypatch.setattr(nestforge_bench, "start", lambda setup: (FinalizedSession(), None, None))
    monkeypatch.setattr(nestforge_bench, "baseline", lambda script, ctx: None)
    monkeypatch.setattr(nestforge_bench, "reference_solution", lambda kernel: None)
    sut = ReferenceOptimizer("dace_cpu_canon_nestforge_reference", workdir=tmp_path)

    library = pathlib.Path(sut.tune(Task(KERNEL, SourceMode.ANY, Language.CPP), "canon_cpu").library)
    gc.collect()

    assert library.parent.parent == tmp_path
    assert library.is_file(), "a deferred /submit still needs it after the submission is gone"


def tuner_episode(column: str, kernel: str, url: str, folder: pathlib.Path) -> subprocess.CompletedProcess[str]:
    """One episode of ``column`` as the bench runs a tuner: its own process, identity and folder, on the judge ``url``."""
    episode, work = folder / "two-stage-episode", folder / "work"
    episode.mkdir()
    work.mkdir()
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "hpcagent_bench.harness.run_tuner",
            *("--workdir", str(work), "--cwd", str(episode), "--kernel", kernel, "--language", "cpp"),
        ],
        env={
            **os.environ,
            "TUNER": column,
            "JUDGE_URL": url,
            "JUDGE_RANK": "0",
            "JUDGE_INPUT_MODE": InputMode.ANY.text,
            "HPCAGENT_BENCH_EPISODE_ID": episode.name,
        },
        capture_output=True,
        text=True,
        timeout=TUNER_EPISODE_TIMEOUT_S,
        check=False,
    )


@pytest.mark.judge
@pytest.mark.parametrize("packet_judge", ["nestforge-two-stage"], indirect=True)
def test_the_two_stage_stub_walks_every_phase_through_the_gate_to_a_correct_submission(
    packet_judge: str, tmp_path: pathlib.Path
) -> None:
    done = tuner_episode("dace_cpu_canon_nestforge_two_stage", "jacobi_1d", packet_judge, tmp_path)

    assert done.returncode == 0, done.stdout + done.stderr
    assert "submitted, correct=True" in done.stdout, done.stdout + done.stderr


class ScriptedModel:
    """The model behind a harness: ``messages.stream`` answers each request with the next reply, keeping every request."""

    def __init__(self, replies: list[list[dict[str, object]]]) -> None:
        self.replies = iter(replies)
        self.sent: list[dict[str, Any]] = []
        self.messages = self

    @contextlib.asynccontextmanager
    async def stream(self, **params: Any) -> AsyncIterator[SimpleNamespace]:  # noqa: ANN401 -- the SDK's own keywords
        self.sent.append({**params, "messages": copy.deepcopy(params["messages"])})
        content = next(self.replies)
        reply = Message(
            id="scripted",
            type="message",
            role="assistant",
            model="scripted",
            content=content,
            stop_reason="tool_use" if content[0]["type"] == "tool_use" else "end_turn",
            usage={"input_tokens": 1, "output_tokens": 1},
        )

        async def final() -> Message:
            return reply

        yield SimpleNamespace(get_final_message=final)


def calls(name: str, **arguments: object) -> list[dict[str, object]]:
    return [{"type": "tool_use", "id": f"use-{name}", "name": name, "input": arguments}]


def says(text: str) -> list[dict[str, object]]:
    return [{"type": "text", "text": text}]


@pytest.mark.judge
@pytest.mark.parametrize("packet_judge", ["nestforge-subagents"], indirect=True)
def test_a_subagent_calls_the_judge_as_the_episode_that_started_it_and_passes_the_gate_of_its_phase(
    loaded: None, packet_judge: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, work = tmp_path / "home", tmp_path / "work"
    work.mkdir()
    packet = packets.resolve("nestforge-subagents", "cpp")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("JUDGE_URL", packet_judge)
    monkeypatch.setenv("JUDGE_RANK", "0")
    monkeypatch.setenv(agent_plugins.PACKET_AGENTS_ENV, ",".join(packet.agents))
    monkeypatch.setenv(registry.PACKET_TOOLS_ENV, ",".join(packet.tools))
    for key, value in packet.env:
        monkeypatch.setenv(key, value)
    agent_plugins.stage_agents(home, os.environ)
    model = ScriptedModel(
        [
            calls("next_phase"),
            calls("next_phase"),
            calls(subagents.TASK_TOOL, subagent_type="kernel-optimizer", prompt="Optimize the first kernel."),
            calls("list_kernels"),
            says("nothing to gain"),
            says("every kernel is done"),
        ]
    )
    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda **options: model)
    (work / "prompt.txt").write_text("Optimize the program.")
    args = SimpleNamespace(
        mcp_config=agent_driver.write_mcp_config(work, work, {"kernel": KERNEL}, 0, 0),
        prompt=work / "prompt.txt",
        workdir=work,
        command_dir=work,
        base_url="http://scripted.invalid",
        request_timeout=None,
        model="scripted",
        max_output_tokens=100,
        reasoning_effort="",
        compaction_trigger=None,
        resume_prompt="",
    )
    usage = runner_common.UsageLog(work / "usage.jsonl")

    reason, detail = asyncio.run(run_anthropic.run_episode_async(args, usage))

    first, _, _, started, listed, *_ = model.sent
    assert reason is runner_common.EndReason.FINISHED, detail
    assert started["system"] == subagents.parse((PLUGIN / "agents" / "kernel-optimizer.md").read_text()).prompt
    offered = {tool["name"] for tool in first["tools"]}
    assert runner_common.READ_TOOL in offered, "the agent reads its skill pages"
    assert runner_common.SHELL_TOOL not in offered, "the packet forbids a shell"
    answer = listed["messages"][-1]["content"][0]
    assert "is_error" not in answer, answer
    assert re.search(r"extcall_\w+", str(answer["content"])), answer


# s231 has a reference solution; the diamond has none and runs the default moves
@pytest.mark.judge
@pytest.mark.parametrize("kernel", ["tsvc_2_s231", KERNEL])
def test_the_reference_column_submits_a_library_the_judge_grades_correct(
    loaded: None, judge_url: str, kernel: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    column = "dace_cpu_canon_nestforge_reference"
    monkeypatch.setenv("HPCAGENT_BENCH_USER_PLUGIN_DIRS", str(PLUGIN))
    if column not in FRAMEWORKS.entries:
        plugins.register_flavors.cache_clear()
        plugins.register_flavors()
    sut = ReferenceOptimizer(column)

    submission = sut.tune(Task(kernel, SourceMode.ANY, Language.CPP), "canon_cpu")

    assert FRAMEWORKS.entries[column].pipelines == ("canon_cpu",), "the manifest registers the column"
    grade = tools.JudgeClient(judge_url).submit(submission, kernel)
    assert isinstance(grade, GradeAnswer), grade
    assert grade.build_ok is True and grade.correct is True, grade.detail


# a straight diamond of maps; maps inside a time loop
@pytest.mark.judge
@pytest.mark.parametrize("kernel", [KERNEL, "jacobi_1d"])
def test_a_finalized_program_is_a_library_the_judge_grades_correct(
    loaded: None, judge_url: str, kernel: str, tmp_path: pathlib.Path
) -> None:
    episode = f"episode-final-{kernel}"
    name, _ = kernel_round(episode, kernel)
    post("tune_kernel", episode=episode, kernel=kernel, name=name, compilers=["gcc"])

    final = post("finalize", episode=episode, kernel=kernel).answer["result"]

    assert pathlib.Path(final["library"]).is_relative_to(tmp_path), final
    assert f"{name}(" in final["source"] and final["kernels"], final
    grade = graded(judge_url, kernel, final)
    assert grade.build_ok is True and grade.correct is True, grade.detail


@pytest.mark.judge
def test_a_program_finalized_after_the_structured_phase_gets_the_default_kernels(loaded: None, judge_url: str) -> None:
    episode = "episode-structured"
    post("run_defaults", episode=episode, stage="moves")

    final = post("finalize", episode=episode).answer["result"]

    assert final["kernels"], final
    grade = graded(judge_url, KERNEL, final)
    assert grade.build_ok is True and grade.correct is True, grade.detail


@pytest.mark.judge
def test_a_profile_of_a_program_with_scratch_runs_only_with_the_library_s_workspace_bytes(
    loaded: None, judge_url: str
) -> None:
    kernel = "atax"
    final = post("finalize", episode="episode-profile-atax", kernel=kernel).answer["result"]
    assert final["workspace_bytes"] != "0", final
    library = str(final["library"])
    client = tools.JudgeClient(judge_url)

    sut = client.profile(
        Submission(language=Language.CPP, library=library, workspace_bytes=str(final["workspace_bytes"])),
        kernel,
        tool="none",
        size=Preset.S,
    )
    without = client.profile(Submission(language=Language.CPP, library=library), kernel, tool="none", size=Preset.S)

    assert sut.root.exit_code == 0 and sut.root.elapsed_ns, sut
    assert without.root.exit_code != 0 and "SIGSEGV" in without.root.stderr, without


@pytest.mark.judge
def test_each_library_product_is_a_kernel_and_the_program_asks_for_its_scratch(loaded: None, judge_url: str) -> None:
    # atax's two products are library nodes, each its own default kernel; tmp = A @ x lives in the judge's scratch

    final = post("finalize", episode="episode-final-atax", kernel="atax").answer["result"]

    assert len(final["kernels"]) == 2 and final["workspace_bytes"] != "0", final
    grade = graded(judge_url, "atax", final)
    assert grade.build_ok is True and grade.correct is True, grade.detail


@pytest.mark.gpu
def test_a_gpu_target_reports_its_offloading(loaded: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGUAGE", "cuda")

    placed = post("run_defaults", episode="episode-gpu", stage="placement")
    judge = post("get_performance", episode="episode-gpu")

    assert placed.code == HTTPStatus.OK and judge.code == HTTPStatus.OK
    offloading = judge.answer["performance"]["offloading"]
    assert "gpu" in offloading["devices"].values()
    assert offloading["transfer_bytes"] > 0
    assert "error" not in judge.answer["performance"]["structured"]


@pytest.mark.gpu
def test_a_finalized_gpu_program_is_a_two_unit_library_the_judge_grades_correct(
    loaded: None, judge_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = get_gpu_backend()
    monkeypatch.setenv("LANGUAGE", backend)

    judge = post("finalize", episode="episode-final-gpu")  # never placed: the default placement

    assert judge.code == HTTPStatus.OK, judge.answer
    final = judge.answer["result"]

    grade = graded(judge_url, KERNEL, final, Language[backend.upper()])
    assert grade.build_ok is True and grade.correct is True, grade.detail
