# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The Tactics bench plugin: loaded through HPCAgent-Bench's own plugin loader, every tool registered, the
structured and kernel phases served with performance feedback, and ``finalize``'s library graded by the judge."""

import asyncio
import contextlib
import copy
import gc
import hashlib
import inspect
import json
import multiprocessing
import os
import pathlib
import re
import sys
import threading
import tomllib
from collections.abc import AsyncIterator, Callable, Iterator
from http import HTTPMethod, HTTPStatus
from types import SimpleNamespace
from typing import Any

import anthropic
import dace
import numpy as np
import pytest
from anthropic.types import Message
from dace.codegen.common import get_gpu_backend
from hpcagent_agent import judge_mcp
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
from hpcagent_bench.harness.plugin_records import FOLDER as RECORDS_FOLDER
from hpcagent_bench.harness.routes import ROUTES, RouteRequest
from hpcagent_bench.harness.service import JUDGE_FAULT_CAUSE, JudgeHandler, ServiceConfig, make_server
from hpcagent_bench.harness.task import Task
from hpcagent_bench.skill_frontmatter import split_frontmatter
from hpcagent_bench.vocabulary import PACKETS
from stub_session import Stub, install

import tactics_bench
from tactics import plugin_record
from tactics.paths import REPO_ROOT
from tactics.records import KernelStatus, MoveStatus, Refusal, Refused
from tactics.server import CALLS, CHANGING, STOP_GRACE_S, TOOL_PHASES, TOOL_TIMEOUT_S, Reply, Setup
from tactics.session import KernelOutcome, Move, MoveResult, Restored, Session, TunedValue, TuneResult
from tactics.stages.families import FAMILIES
from tactics_bench import DefaultOptimizer, NoWaveOptimizer, ReferenceOptimizer

PLUGIN = REPO_ROOT / "plugin" / "tactics"
PAGE = PLUGIN / "skills" / "tactics" / "SKILL.md"
# the bench files a plugin's record under the plugin's name, which stays tactics
RECORD = f"{plugin_record.PLUGIN}.json"
KERNEL = "loop_level_reasoning/fuse_diamond/fuse_diamond"
#: The bench tools the packet replaces: each takes no code and sends the library finalize built.
OVERRIDDEN = {"profile", "score", "submit"}
#: Tools only one packet of its own serves, never tactics-full: transformation requests (E6) and the LLVM, MLIR and
#: pass-move baselines, whose agents choose passes instead of moves.
PACKET_OWN_TOOLS = {
    "request_transformation": "tactics-requests",
    "set_llvm_passes": "tactics-llvm",
    "set_mlir_passes": "tactics-mlir",
    "view_ir": "tactics-passes",
    "list_pass_moves": "tactics-passes",
    "apply_pass_move": "tactics-passes",
    "advance_pass_phase": "tactics-passes",
}
#: The device slot a fake judge hands out.
SLOT = DeviceSlot("cpu", 3)
#: Seconds a spawned judge has to import the bench and bind its port.
JUDGE_START_TIMEOUT_S = 300.0
#: One valid call of tools that read, change or measure, its JSON body.
SLOT_CALLS: dict[str, dict[str, Any]] = {
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
        self.answer: dict[str, Any] = {}  # the JSON answer
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
        yield SLOT


@pytest.fixture(autouse=True)
def shared_folder(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Journals live on disk: each test has its own shared folder, so no run repeats another's moves."""
    monkeypatch.setenv("HPCAGENT_BENCH_SHARED_DIR", str(tmp_path))


@pytest.fixture(scope="module")
def loaded(tactics_plugin: tuple[Plugin, ...]) -> Iterator[None]:
    assert [plugin.name for plugin in tactics_plugin if plugin.root == PLUGIN] == ["tactics"]
    yield
    # every session process this module started, so no other test inherits a live child
    episodes = sys.modules[ROUTES["tactics"].handler.__module__].EPISODES
    for _, (episode, _) in list(episodes.states.items()):
        for session in episode.sessions.values():
            session.close()


def post(
    tool: str, episode: str = "episode-cpu", kernel: str = KERNEL, judge: FakeJudge | None = None, **body: object
) -> FakeJudge:
    judge = judge if judge is not None else FakeJudge()
    request = RouteRequest(HTTPMethod.POST, ["tactics", tool], {}, {"kernel": kernel, **body}, episode)
    ROUTES["tactics"].handler(judge, request)
    return judge


@pytest.fixture
def stub(loaded: None, monkeypatch: pytest.MonkeyPatch) -> Iterator[Stub]:
    """Every session the route starts answers from a :class:`Stub`; its episodes are forgotten afterwards."""
    episodes = sys.modules[ROUTES["tactics"].handler.__module__].EPISODES
    before = set(episodes.states)
    yield install(monkeypatch, Stub())
    for episode in set(episodes.states) - before:
        episodes.drop(episode)


@pytest.fixture
def over_mcp(loaded: None, tactics_plugin: tuple[Plugin, ...], monkeypatch: pytest.MonkeyPatch) -> None:
    """The judge's tools registered as the agent's MCP server registers them (``[judge_mcp]``): the listing goes
    to the route in this process."""

    def post_judge(route: str, body: dict[str, object]) -> dict[str, object]:
        judge = FakeJudge()
        parts = route.lstrip("/").partition("?")[0].split("/")
        ROUTES["tactics"].handler(judge, RouteRequest(HTTPMethod.POST, parts, {}, body, "episode-agent"))
        return judge.answer

    monkeypatch.setattr(http_json, "post_judge", post_judge)
    (block,) = [plugin.manifest.judge_mcp for plugin in tactics_plugin if plugin.root == PLUGIN]
    assert block is not None, "plugin.toml has a [judge_mcp] block"
    judge_mcp.register(block, os.environ)


def test_every_tool_the_judge_serves_over_mcp_is_registered_as_a_plugin_tool(over_mcp: None) -> None:
    from hpcagent_bench_plugins.tactics.wire import TOOLS

    assert TOOLS.keys() <= registry.REGISTRY.keys()
    assert all(not registry.REGISTRY[name].core for name in TOOLS)


def test_a_scope_set_through_the_route_can_be_unset_again(loaded: None) -> None:
    to_phase("episode-scope", "scopes")
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


def test_tactics_owns_its_phases_and_its_gate_matches_the_tools_phases(loaded: None) -> None:
    from hpcagent_bench_plugins.tactics.answers import PHASE_TOOLS
    from hpcagent_bench_plugins.tactics.wire import TOOLS, Phase

    manifest = tomllib.loads((PLUGIN / "plugin.toml").read_text())
    gated = {tool: {p.value for p in phases} for tool, phases in TOOL_PHASES.items()}
    labelled = {tool: {p.value for p in d.phases} for tool, d in TOOLS.items() if Phase.ANY not in d.phases}

    assert "phases" not in manifest, "the bench's phases are not Tactics': Tactics gates its own tools"
    assert gated == labelled, "the server's gate and the phases each tool says it works in"
    assert sorted(manifest["packets"]["tactics"]["tools"]) == sorted(
        (
            *(
                t
                for t, d in TOOLS.items()
                if (Phase.ANY in d.phases or Phase.STRUCTURED in d.phases)
                and t not in PACKET_OWN_TOOLS
                and t not in PHASE_TOOLS
            ),
            "profile",
            "score",
            "submit",
        )
    )


def test_a_blind_agent_is_not_served_a_measurement_and_the_tool_list_says_one_submission(over_mcp: None) -> None:
    body = {"LANGUAGE": "cpp", registry.PACKET_TOOLS_ENV: ",".join(PACKETS["tactics"].tools)}

    blind = registry.served(registry.Setup.read({**body, "AGENT_SUBMISSION_MODE": "blind"}))
    single = registry.served(registry.Setup.read({**body, "AGENT_SUBMISSION_MODE": "single"}))

    assert "get_performance" not in blind and "score" not in blind
    assert "get_performance" in single
    assert single["submit"].prompt == submit.PROMPT


def test_the_packet_exposes_the_structured_phase_and_its_page_and_every_tool_is_served(loaded: None) -> None:
    from hpcagent_bench_plugins.tactics.wire import TOOLS, Phase

    packet = PACKETS["tactics"].tools

    assert plugins.NEXT_PHASE not in packet, "Tactics' own phase tools, not the bench's"
    assert all(Phase.ANY in TOOLS[t].phases or Phase.STRUCTURED in TOOLS[t].phases for t in packet if t in TOOLS)
    assert OVERRIDDEN <= registry.OVERRIDES.keys() and OVERRIDDEN <= set(packet)
    assert "set_scope" not in packet and "list_move_kinds" in packet
    assert split_frontmatter(PAGE.read_text())[0].name == "tactics"
    assert PACKETS["tactics"].skills == ("tactics",)
    assert TOOLS.keys() == CALLS.keys()
    for tool in TOOLS:
        params = inspect.signature(CALLS[tool]).parameters
        passed = {name for name, field in TOOLS[tool].args.model_fields.items() if not field.exclude}
        missing = passed - params.keys()
        assert not missing, f"{tool}: the session call takes no {sorted(missing)}"


def test_profile_takes_no_code_and_sends_the_library_finalize_built_with_its_scratch(
    over_mcp: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run's agents sent ``source`` the library judge refuses, or the library without the ``workspace_bytes`` that
    library's scratch needs, a NULL workspace and a SIGSEGV. Neither can be said now, and finalize gets no instrument:
    its arguments are forbidden."""
    calls: list[tuple[str, dict[str, object]]] = []

    def judge(route: str, body: dict[str, object]) -> dict[str, object]:
        calls.append((route, body))
        return {"result": {"library": "/shared/lib.so", "workspace_bytes": "16*K + 128"}, "profiled": True}

    monkeypatch.setattr(http_json, "post_judge", judge)
    setup = registry.Setup.read({"LANGUAGE": "cpp", registry.PACKET_TOOLS_ENV: ",".join(PACKETS["tactics"].tools)})
    sut = registry.served(setup)["profile"]

    answer = sut.run({"tool": "none", "threads": 24, "size": "S"})
    refused = sut.run({"tool": "none", "source": "int main;"})

    arguments = sut.input_schema["properties"].keys()
    assert {"tool", "threads", "counter_group", "size"} <= arguments
    assert not {"source", "source_file", "library", "workspace_bytes", "build"} & arguments
    assert answer["profiled"] is True and refused["ok"] is False, (answer, refused)
    assert [route for route, _ in calls] == ["/tactics/finalize", "/profile"], "the refused call reaches no judge"
    assert "tool" not in calls[0][1]
    assert calls[1][1]["library"] == "/shared/lib.so" and calls[1][1]["workspace_bytes"] == "16*K + 128"
    assert (calls[1][1]["tool"], calls[1][1]["threads"], calls[1][1]["size"]) == ("none", 24, "S")


def test_profile_offers_ranks_on_a_distributed_task_only_and_sends_them_to_the_judge(
    over_mcp: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bench's own ``profile`` drops ``ranks`` (bench B2), so an agent could never see ``mpi_time_us``."""
    calls: list[tuple[str, dict[str, object]]] = []

    def judge(route: str, body: dict[str, object]) -> dict[str, object]:
        calls.append((route, body))
        return {"result": {"library": "/shared/lib.so", "workspace_bytes": "0"}, "runs": []}

    monkeypatch.setattr(http_json, "post_judge", judge)
    packet = {"LANGUAGE": "cpp", registry.PACKET_TOOLS_ENV: ",".join(PACKETS["tactics"].tools)}
    distributed = {**packet, http_json.REQUEST_ACTIONS_ENV: json.dumps({"operations": ["distribute"]})}
    single = registry.served(registry.Setup.read(packet))["profile"]
    sut = registry.served(registry.Setup.read(distributed))["profile"]

    sut.run({"ranks": [1, 2]})
    sut.run({})
    refused = sut.run({"ranks": []})

    assert "ranks" not in single.input_schema["properties"]
    assert "ranks" in sut.input_schema["properties"] and refused["ok"] is False, refused
    profiled = [body for route, body in calls if route == "/profile"]
    assert len(profiled) == 2 and profiled[0]["ranks"] == [1, 2] and "ranks" not in profiled[1]


def test_the_full_packet_serves_every_tool_and_lists_reads_as_read_only(over_mcp: None) -> None:
    from hpcagent_bench_plugins.tactics.wire import TOOLS

    packet = PACKETS["tactics-full"].tools
    own = {tool: PACKETS[name].tools for tool, name in PACKET_OWN_TOOLS.items()}

    assert sorted(packet) == sorted((*(t for t in TOOLS if t not in PACKET_OWN_TOOLS), *OVERRIDDEN))
    assert all(tool in tools and tool not in packet for tool, tools in own.items())
    specs = {name: registry.REGISTRY[name].spec for name in TOOLS}
    hints = {name: spec.hints for name, spec in specs.items() if isinstance(spec, registry.Spec)}
    assert hints.keys() == TOOLS.keys()
    assert {name for name, hint in hints.items() if hint.read_only} == {
        "get_cpf",
        "view_tree",
        "list_move_kinds",
        "list_moves",
        "list_scope_moves",
        "list_kernels",
        "read_kernel",
        "applied_moves",
        "get_work_depth",
        "request_transformation",
        "view_ir",
        "list_pass_moves",
    }
    assert hints["write_kernel"].destructive and not hints["apply_move"].destructive


def rpc(method: str, episode: str = "episode-mcp", message_id: int | None = 1, **params: object) -> FakeJudge:
    """One JSON-RPC message to ``POST /tactics/mcp``, as an MCP client sends it."""
    judge = FakeJudge()
    body: dict[str, object] = {"jsonrpc": "2.0", "method": method, **({"params": params} if params else {})}
    if message_id is not None:
        body["id"] = message_id
    ROUTES["tactics"].handler(judge, RouteRequest(HTTPMethod.POST, ["tactics", "mcp"], {}, body, episode))
    return judge


def test_the_mcp_endpoint_lists_every_served_tool_with_the_schema_of_its_model(loaded: None) -> None:
    listed = rpc("tools/list").answer

    tools = {tool["name"]: tool for tool in listed["result"]["tools"]}
    assert listed["jsonrpc"] == "2.0" and listed["id"] == 1
    assert "kind" in tools["apply_move"]["inputSchema"]["required"]
    assert tools["list_moves"]["annotations"]["readOnlyHint"] is True


def test_a_tool_called_over_mcp_shows_what_the_agent_side_tool_shows(stub: Stub) -> None:
    from hpcagent_bench_plugins.tactics.answers import agent_answer
    from hpcagent_bench_plugins.tactics.wire import MOVE_DEFAULTS

    over_rest = post("list_moves", "episode-mcp-rest").answer
    over_mcp = rpc("tools/call", "episode-mcp-call", name="list_moves", arguments={}, _meta={"kernel": KERNEL})

    result = over_mcp.answer["result"]
    assert result["isError"] is False
    assert result["structuredContent"] == agent_answer("list_moves", over_rest, MOVE_DEFAULTS)


def test_a_call_over_mcp_the_model_refuses_is_an_error_result_not_a_protocol_error(stub: Stub) -> None:
    refused = rpc("tools/call", name="apply_move", arguments={"kind": "fuse"}, _meta={"kernel": KERNEL}).answer

    assert refused["result"]["isError"] is True and "labels" in refused["result"]["content"][0]["text"]


def test_an_mcp_notification_is_accepted_without_an_answer(loaded: None) -> None:
    done = rpc("notifications/initialized", message_id=None)

    assert (done.code, done.answer) == (HTTPStatus.ACCEPTED, {})


def test_a_move_that_kills_its_session_is_undone_and_the_next_call_runs_on_a_session_without_it(stub: Stub) -> None:
    post("apply_move", "episode-crash", kind="fuse", labels=["map_0", "map_1"])
    stub.dies_on = "apply_move"

    died = post("apply_move", "episode-crash", kind="fuse", labels=["map_2", "map_3"])
    stub.dies_on = ""
    before = len(stub.calls)
    again = post("get_cpf", "episode-crash")

    assert died.code == HTTPStatus.BAD_REQUEST and "cause" not in died.answer, died.answer
    assert died.answer["error"] == (
        "apply_move: the session process died (exit code -9) during apply_move (fuse on map_2, map_3); that change was"
        " undone and the session restarts from the last good program; try another move"
    )
    assert again.code == HTTPStatus.OK and len(stub.started) == 2
    rebuilt = [(tool, arguments.get("labels")) for tool, arguments in stub.calls[before:]]
    assert rebuilt == [("apply_move", ["map_0", "map_1"]), ("get_cpf", None)], "rebuilt without the crashed move"


def test_a_fresh_session_repeats_every_change_the_last_one_answered_so_no_move_is_lost(stub: Stub) -> None:
    first: dict[str, Any] = {"kind": "fuse", "labels": ["map_0", "map_1"]}
    post("apply_move", "episode-replay", **first)
    post("view_tree", "episode-replay")
    post("set_scope", "episode-replay", labels=["map_0"])
    stub.dies_on = "get_performance"

    died = post("get_performance", "episode-replay")
    stub.dies_on = ""
    before = len(stub.calls)
    answered = post("list_moves", "episode-replay")

    assert died.code == HTTPStatus.BAD_REQUEST and answered.code == HTTPStatus.OK
    repeated = [tool for tool, _ in stub.calls[before:]]
    assert repeated == ["apply_move", "set_scope", "list_moves"], "the changes in order, then the call; no read"
    assert len(stub.started) == 2


def test_a_focus_is_journaled_so_a_fresh_session_works_on_the_same_kernel(stub: Stub) -> None:
    post("focus_kernel", "episode-focus", name="kernel_0")
    stub.dies_on = "get_performance"
    post("get_performance", "episode-focus")
    stub.dies_on = ""
    before = len(stub.calls)

    written = post("write_kernel", "episode-focus", **SLOT_CALLS["write_kernel"])

    route = sys.modules[ROUTES["tactics"].handler.__module__]
    assert written.code == HTTPStatus.OK
    assert route.journal_of("episode-focus", KERNEL)[0] == ("focus_kernel", {"name": "kernel_0"})
    assert [tool for tool, _ in stub.calls[before:]] == ["focus_kernel", "write_kernel"]


MOVE: dict[str, Any] = {"kind": "fuse", "labels": ["map_0", "map_1"]}

N = dace.symbol("N", dtype=dace.int64)


@dace.program
def factor_and_scale(a: dace.float64[N, N], b: dace.float64[N, N], c: dace.float64[N, N]) -> None:
    b[:] = np.linalg.cholesky(a)
    for i, j in dace.map[0:N, 0:N]:
        c[i, j] = 2.0 * a[i, j]
    for i, j in dace.map[0:N, 0:N]:
        b[i, j] = b[i, j] + c[i, j]


def listed_moves(tmp_path: pathlib.Path) -> list[Move]:
    """Every move a real session lists for a program with a library call and two map nests."""
    session = Session(factor_and_scale.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": 16})
    session.canonicalize()
    return session.list_moves()


def test_every_listed_move_is_taken_back_as_list_moves_answered_it(stub: Stub, tmp_path: pathlib.Path) -> None:
    moves = listed_moves(tmp_path)
    stub.replies["list_moves"] = Reply(result=moves)
    listed = post("list_moves", "episode-listed").answer["result"]

    answers = [post("apply_move", "episode-listed", **move) for move in listed]

    assert {"set-library-implementation", "tile-doall"} <= {move["kind"] for move in listed}, listed
    assert [answer.code for answer in answers] == [HTTPStatus.OK] * len(listed), [a.answer for a in answers]
    sent = [arguments for tool, arguments in stub.calls if tool == "apply_move"]
    taken = [Move(**arguments) for arguments in sent]
    assert taken == [move.model_copy(update={"current": None}) for move in moves], "current is a note, not an argument"


def test_a_session_that_died_between_calls_is_replaced_and_repeats_the_agents_moves(stub: Stub) -> None:
    post("apply_move", "episode-idle", **MOVE)
    route = sys.modules[ROUTES["tactics"].handler.__module__]
    route.episode_of("episode-idle").sessions[KERNEL].alive = False
    before = len(stub.calls)

    answered = post("get_cpf", "episode-idle")

    assert answered.code == HTTPStatus.OK and len(stub.started) == 2
    assert [tool for tool, _ in stub.calls[before:]] == ["apply_move", "get_cpf"]


def restart_judge() -> None:
    """What a judge restart or an idle eviction leaves: nothing in memory."""
    episodes = sys.modules[ROUTES["tactics"].handler.__module__].EPISODES
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
    route = sys.modules[ROUTES["tactics"].handler.__module__]
    with route.journal_file("episode-torn", KERNEL).open("a") as journal:
        journal.write(line + "\n")
    started = len(stub.started)

    judge = post("finalize", "episode-torn")

    assert (judge.code, judge.answer["cause"]) == (HTTPStatus.INTERNAL_SERVER_ERROR, JUDGE_FAULT_CAUSE), judge.answer
    assert "journal" in judge.answer["error"] and len(stub.started) == started


def test_a_change_a_rebuild_refuses_fails_loudly_instead_of_grading_a_program_without_it(stub: Stub) -> None:
    post("apply_move", "episode-diverged", **MOVE)
    restart_judge()
    reason = Refused("no such row", Refusal.UNKNOWN_LABEL)
    refused = MoveResult(status=MoveStatus.NOT_FOUND, kind="fuse", labels=("map_0",), reason=reason)
    stub.replies["apply_move"] = Reply(result=refused)

    judge = post("finalize", "episode-diverged")

    assert (judge.code, judge.answer["cause"]) == (HTTPStatus.INTERNAL_SERVER_ERROR, JUDGE_FAULT_CAUSE), judge.answer
    assert "no such row" in judge.answer["error"] and "finalize" not in [tool for tool, _ in stub.calls]


def test_a_refused_change_is_not_journaled(stub: Stub) -> None:
    refused = MoveResult(
        status=MoveStatus.ILLEGAL, kind="fuse", labels=("map_0",), reason=Refused("cannot fuse", Refusal.DEPENDENCE)
    )
    stub.replies["apply_move"] = Reply(result=refused)

    answered = post("apply_move", "episode-refused", **MOVE)

    route = sys.modules[ROUTES["tactics"].handler.__module__]
    assert answered.code == HTTPStatus.OK and route.journal_of("episode-refused", KERNEL) == []


def test_finalize_keeps_a_record_of_every_call_with_its_outcome_refusal_class_and_tokens(stub: Stub) -> None:
    route = sys.modules[ROUTES["tactics"].handler.__module__]
    episode = "episode-record"
    post("view_tree", episode, tokens=100)
    post("apply_move", episode, tokens=200, **MOVE)
    refused = MoveResult(
        status=MoveStatus.ILLEGAL, kind="fuse", labels=("map_2",), reason=Refused("cannot fuse", Refusal.DEPENDENCE)
    )
    stub.replies["apply_move"] = Reply(result=refused)
    post("apply_move", episode, tokens=300, kind="fuse", labels=["map_2", "map_3"])
    post("apply_move", episode, tokens=350, kind="fuse")  # no labels: refused before any session
    post("write_kernel", episode, tokens=400, **SLOT_CALLS["write_kernel"])

    post("finalize", episode, tokens=500)

    record = json.loads((route.agent_root(episode) / RECORDS_FOLDER / RECORD).read_text())
    data = record["data"]
    assert record["schema_version"] == route.SCHEMA and data["kernel"] == KERNEL
    seen = [(c["index"], c["tool"], c["status"], c["refusal"], c["tokens"]) for c in data["calls"]]
    assert seen == [
        (0, "view_tree", "ok", "", 100),
        (1, "apply_move", "applied", "", 200),
        (2, "apply_move", "illegal", "dependence", 300),
        (3, "apply_move", "failed", "", 350),
        (4, "write_kernel", "ok", "", 400),
        (5, "finalize", "ok", "", 500),
    ]
    assert data["first_write_kernel"] == 4
    assert [tool for tool, _ in data["journal"]] == ["apply_move", "write_kernel"], "the changes, refusals left out"
    assert data["journal"][1][1]["source"] == f"sha256:{hashlib.sha256(b'x').hexdigest()}", "the source is hashed"


def test_an_episode_switches_families_off_at_its_first_call_for_its_whole_length(stub: Stub) -> None:
    route = sys.modules[ROUTES["tactics"].handler.__module__]
    episode = "episode-ablation"

    first = post("apply_move", episode, disabled_families=["all"], **MOVE)
    changed = post("view_tree", episode, disabled_families=[])
    post("finalize", episode, disabled_families=["all"])

    assert first.code == HTTPStatus.OK and stub.started[-1].disabled_families == tuple(sorted(FAMILIES))
    assert changed.code == HTTPStatus.BAD_REQUEST and "at its first call" in changed.answer["error"]
    data = json.loads((route.agent_root(episode) / RECORDS_FOLDER / RECORD).read_text())["data"]
    assert data["disabled_families"] == sorted(FAMILIES)
    assert [(c["tool"], c["family"], c["status"]) for c in data["calls"]] == [
        ("apply_move", None, "applied"),  # MOVE's kind "fuse" is the stub's, in no family
        ("view_tree", None, "failed"),
        ("finalize", None, "ok"),
    ]


def test_an_unknown_family_is_refused_at_the_first_call(stub: Stub) -> None:
    judge = post("view_tree", "episode-unknown-family", disabled_families=["fusions"])

    assert judge.code == HTTPStatus.BAD_REQUEST and "no move family" in judge.answer["error"]
    assert not stub.started, "no session starts for a run that names no family"


def test_every_call_over_mcp_carries_the_packets_switched_off_families(
    over_mcp: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[dict[str, Any]] = []

    def post_judge_waiting(route: str, body: dict[str, Any], timeout: float) -> dict[str, object]:
        sent.append(body)
        return {"result": None}

    # a tools/call waits its route's timeout, so it posts through post_judge_waiting
    monkeypatch.setattr(http_json, "post_judge_waiting", post_judge_waiting)
    monkeypatch.setattr(http_json, "assigned_kernel", lambda payload: KERNEL)
    monkeypatch.setenv("TACTICS_DISABLED_FAMILIES", "fusion, tiling")

    setup = registry.Setup.read({"LANGUAGE": "cpp", registry.PACKET_TOOLS_ENV: "view_tree"})
    registry.served(setup)["view_tree"].run({})

    assert sent[-1]["params"]["_meta"]["disabled_families"] == ["fusion", "tiling"]


def test_a_kernel_written_without_a_check_is_journaled_as_the_change_it_is(stub: Stub) -> None:
    written = KernelOutcome(kernel="kernel_0", status=KernelStatus.UNCHECKED, reason="the whole program checks it")
    stub.replies["write_kernel"] = Reply(result=written)

    post("write_kernel", "episode-unchecked", **SLOT_CALLS["write_kernel"])

    route = sys.modules[ROUTES["tactics"].handler.__module__]
    assert [tool for tool, _ in route.journal_of("episode-unchecked", KERNEL)] == ["write_kernel"]


def test_a_change_that_cannot_be_journaled_is_dropped_with_its_session_and_the_agent_is_told(
    stub: Stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    route = sys.modules[ROUTES["tactics"].handler.__module__]

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
    idle = sys.modules[ROUTES["tactics"].handler.__module__].EPISODES.idle_seconds

    assert TOOL_TIMEOUT_S + STOP_GRACE_S < idle <= 3600.0


def test_a_read_and_a_move_take_no_device_slot_and_what_runs_code_takes_one(stub: Stub) -> None:
    slots = {tool: post(tool, "episode-slots", **body).slots for tool, body in SLOT_CALLS.items()}

    assert slots == {
        tool: int(tool in {"get_performance", "finalize", "write_kernel", "tune_kernel"}) for tool in slots
    }


def test_only_the_finalize_of_a_submission_queues_as_the_submission_does(
    stub: Stub, over_mcp: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def judge(route: str, body: dict[str, object]) -> dict[str, object]:
        calls.append(route)
        return {"result": {"library": "/shared/lib.so", "workspace_bytes": "0"}}

    monkeypatch.setattr(http_json, "post_judge", judge)
    setup = registry.Setup.read({"LANGUAGE": "cpp", registry.PACKET_TOOLS_ENV: ",".join(PACKETS["tactics"].tools)})
    registry.served(setup)["submit"].run({})
    registry.served(setup)["score"].run({})
    terminal = FakeJudge()
    request = {"terminal": ["1"]}, {"kernel": KERNEL}, "episode-terminal"
    ROUTES["tactics"].handler(terminal, RouteRequest(HTTPMethod.POST, ["tactics", "finalize"], *request))

    assert [calls[0], calls[2]] == ["/tactics/finalize?terminal=1", "/tactics/finalize"]
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


def test_a_written_kernel_and_a_sweep_are_journaled_as_their_unchecked_untimed_repeats(stub: Stub) -> None:
    written_then_lost("episode-repeats")

    route = sys.modules[ROUTES["tactics"].handler.__module__]
    assert route.journal_of("episode-repeats", KERNEL) == [
        ("write_kernel", {**SLOT_CALLS["write_kernel"], "checked": False}),
        ("tune_kernel", {"name": "kernel_0", "compilers": None, "variant": "gcc-O3"}),
    ]


def test_a_tune_move_is_journaled_as_the_move_at_the_size_it_kept_or_not_at_all(loaded: None) -> None:
    route = sys.modules[ROUTES["tactics"].handler.__module__]
    call = ("tune_move", {"kind": "tile-doall", "labels": ["doall8"], "values": [[32, 32], [64, 64]]})
    now = TunedValue(status="now", time_us=30.0, correct=True)
    kept = TuneResult(
        kind="tile-doall",
        labels=("doall8",),
        values=[
            now,
            TunedValue(sizes=(32, 32), status="measured"),
            TunedValue(sizes=(64, 64), status="kept"),
        ],
    )
    none = TuneResult(
        kind="tile-doall", labels=("doall8",), values=[now, TunedValue(sizes=(32, 32), status="measured")]
    )

    assert route.repeats_of(call, Reply(result=kept)) == [
        ("apply_move", {"kind": "tile-doall", "labels": ["doall8"], "tile_sizes": [64, 64]})
    ]
    assert route.repeats_of(call, Reply(result=none)) == []


def test_a_get_performance_that_restored_the_fastest_version_is_journaled_as_its_rollbacks(loaded: None) -> None:
    route = sys.modules[ROUTES["tactics"].handler.__module__]
    call: tuple[str, dict[str, Any]] = ("get_performance", {})

    restored = route.repeats_of(call, Reply(result=Restored(fastest_us=10.0, now_us=30.0, rolled_back=2)))

    assert restored == [("apply_move", {"kind": "rollback", "labels": []})] * 2
    assert route.repeats_of(call, Reply(result=None)) == []


def test_a_read_that_rebuilds_a_written_kernel_takes_a_slot_measures_nothing_and_sees_its_client(stub: Stub) -> None:
    written_then_lost("episode-rebuild")
    before = len(stub.calls)

    judge = post("view_tree", "episode-rebuild")

    assert judge.slots == 1
    assert [tool for tool, _ in stub.calls[before:]] == ["write_kernel", "tune_kernel", "view_tree"], (
        "a tuning is replayed: finalize never sweeps a kernel that already has a build"
    )
    assert stub.held[before:] == [(SLOT, False, True), (SLOT, False, True), (SLOT, True, True)]


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
    episode = sys.modules[ROUTES["tactics"].handler.__module__].episode_of("episode-queue")
    judge = FakeJudge()
    judge.gone.set()

    with episode.measuring:
        left = post("get_performance", "episode-queue", judge=judge)

    assert left.slots == 0 and left.code == 0 and stub.calls == []


def test_a_call_whose_client_left_while_it_waited_for_the_session_never_runs(stub: Stub) -> None:
    episode = sys.modules[ROUTES["tactics"].handler.__module__].episode_of("episode-gone")
    judge = FakeJudge()
    judge.gone.set()

    with episode.lock:
        left = post("apply_move", "episode-gone", judge=judge, kind="fuse", labels=["map_0", "map_1"])

    assert left.code == 0 and stub.calls == [], "no answer, and the session never saw it"


def test_a_retry_reads_the_answer_of_a_change_whose_client_left_instead_of_applying_it_twice(stub: Stub) -> None:
    from hpcagent_bench_plugins.tactics.wire import ApplyMove

    move = ApplyMove(kind="fuse", labels=["map_0", "map_1"]).model_dump()
    judge = FakeJudge()
    judge.gone.set()
    route = sys.modules[ROUTES["tactics"].handler.__module__]
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

    monkeypatch.setattr(sys.modules[ROUTES["tactics"].handler.__module__], "SessionProcess", refused)

    judge = post("get_cpf", "episode-unstartable")

    assert (judge.code, judge.answer["cause"]) == (HTTPStatus.INTERNAL_SERVER_ERROR, JUDGE_FAULT_CAUSE), judge.answer


def test_the_subagent_packet_adds_the_kernel_optimizer_to_the_full_packet(loaded: None) -> None:
    packet, full = PACKETS["tactics-subagents"], PACKETS["tactics-full"]
    agent = subagents.parse((PLUGIN / "agents" / "kernel-optimizer.md").read_text())

    assert packet.agents == (agent.name,) and sorted(packet.tools) == sorted(full.tools)
    assert agent.tools is None, "a subagent lists no tools: the harnesses name them differently and the gate holds"
    assert packet.phases == full.phases


def test_the_two_stage_name_is_the_full_packet_and_only_that_packet_answers_to_it(loaded: None) -> None:
    resolved = packets.resolve("tactics-two-stage", "cpp")

    assert resolved == packets.resolve("tactics-full", "cpp")
    assert PACKETS.canonical("tactics-two-stage") == "tactics-full"
    assert "tactics-two-stage" not in PACKETS.entries, "an alias is no packet of its own"


def test_a_bench_setup_of_the_subagent_packet_stages_and_offers_the_kernel_optimizer(
    loaded: None, tmp_path: pathlib.Path
) -> None:
    environ = {
        agent_plugins.USER_PLUGIN_DIRS_ENV: str(PLUGIN),
        agent_plugins.PACKET_AGENTS_ENV: ",".join(packets.resolve("tactics-subagents", "cpp").agents),
    }

    staged = agent_plugins.stage_agents(tmp_path, environ)
    offered = subagents.load(tmp_path, environ)

    shipped = subagents.parse((PLUGIN / "agents" / "kernel-optimizer.md").read_text())
    assert staged == ("kernel-optimizer",) and offered == (shipped,)
    assert subagents.task_schema(offered)["properties"]["subagent_type"]["enum"] == [shipped.name]
    assert shipped.name in subagents.task_description(offered)


def walk(episode: str) -> Callable[..., dict[str, Any]]:
    """A phase tool of ``episode``, through Tactics' route; its result."""

    def step(tool: str, **body: Any) -> dict[str, Any]:  # noqa: ANN401 -- a tool's JSON body
        judge = post(tool, episode=episode, **body)
        assert judge.code == HTTPStatus.OK, judge.answer
        result: dict[str, Any] = judge.answer["result"]
        return result

    return step


@pytest.mark.integration
def test_the_full_packet_goes_from_structured_to_each_kernel_and_returns_to_the_moves_as_left(loaded: None) -> None:
    episode = "episode-walk"
    step = walk(episode)
    moves = post("list_moves", episode=episode).answer["result"]
    assert moves

    scopes, scoped, kernels = step("advance_phase"), step("advance_phase"), step("advance_phase")
    names = [kernel["name"] for kernel in post("list_kernels", episode=episode).answer["result"]]
    closed = post("set_scope", episode=episode, labels=["doall0"])
    written = []
    for name in names:
        post("focus_kernel", episode=episode, name=name)
        source = post("read_kernel", episode=episode, name=name).answer["result"]
        written.append(post("write_kernel", episode=episode, name=name, source=source, language="cpp"))
    post("focus_kernel", episode=episode, name=names[0])
    tuned = post("tune_kernel", episode=episode, name=names[0], compilers=["gcc"])
    last = step("advance_phase")
    back = step("return_to_phase", phase="structured")

    assert (scopes["phase"], scoped["phase"], kernels["phase"]) == ("scopes", "scoped", "kernels")
    assert names and closed.code == HTTPStatus.BAD_REQUEST and "phase scopes" in closed.answer["error"]
    assert [answer.answer["result"]["status"] for answer in written] == ["ok"] * len(names)
    assert tuned.code == HTTPStatus.OK
    assert (last["status"], back["status"], back["phase"]) == ("refused", "moved", "structured")
    assert post("list_moves", episode=episode).answer["result"] == moves


def test_the_phase_tools_reach_the_session_and_are_journaled_as_changes(stub: Stub) -> None:
    for tool in ("advance_phase", "undo_phase", "return_to_phase"):
        body: dict[str, Any] = {"phase": "structured"} if tool == "return_to_phase" else {}
        post(tool, "episode-phases", **body)

    assert [call[0] for call in stub.calls if call[0].endswith("phase")] == [
        "advance_phase",
        "undo_phase",
        "return_to_phase",
    ]
    assert {"advance_phase", "undo_phase", "return_to_phase"} <= CHANGING


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


def to_phase(episode: str, phase: str, kernel: str = KERNEL) -> None:
    """``advance_phase`` on ``episode`` until it is in ``phase``; the phases passed run their defaults."""
    for _ in range(4):
        step = post("advance_phase", episode=episode, kernel=kernel).answer["result"]
        if step["phase"] == phase:
            return
    raise AssertionError(f"{episode} never reached the phase {phase}")


def kernel_round(episode: str, kernel: str = KERNEL) -> tuple[str, str]:
    """``episode`` in the kernels phase (every doall its default scope), then the first kernel's name and source."""
    to_phase(episode, "kernels", kernel)
    name = post("list_kernels", episode=episode, kernel=kernel).answer["result"][0]["name"]
    assert post("focus_kernel", episode=episode, kernel=kernel, name=name).code == HTTPStatus.OK
    return name, post("read_kernel", episode=episode, kernel=kernel, name=name).answer["result"]


@pytest.mark.integration
def test_a_real_session_restarted_from_its_journal_is_the_program_the_agent_left(loaded: None) -> None:
    move = post("list_moves", "episode-rebuilt").answer["result"][0]
    post("apply_move", "episode-rebuilt", **move)
    route = sys.modules[ROUTES["tactics"].handler.__module__]
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


@pytest.fixture
def decoy(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> pathlib.Path:
    """A secret in the run folder, which the judge reads and agent code must not."""
    secret = tmp_path / "run" / "episode-secret"
    secret.parent.mkdir()
    secret.write_text("the-decoy-secret\n")
    monkeypatch.setenv("RUN_DIR", str(secret.parent))
    return secret


# One kernel round serves every write, as starting one costs about a minute: the writes the judge refuses or that
# break the build change nothing, then a kernel that tries to read the judge's secret, then the source written back.
@pytest.mark.integration
def test_a_written_kernel_is_checked_sealed_from_what_the_judge_hides_and_tuned(
    loaded: None, decoy: pathlib.Path
) -> None:
    name, source = kernel_round("episode-kernel")
    entry = re.search(rf"void {name}\([^)]*\)\s*\{{", source)
    assert entry is not None, source
    # the kernel skips its work if it reads the file, so its oracle calls a kernel outside the seal wrong
    guard = f'FILE* secret = fopen("{decoy}", "r"); if (secret != NULL) return;'
    spying = f"#include <cstdio>\n{source[: entry.end()]}{guard}{source[entry.end() :]}"
    renamed = source.replace(f"void {name}(", f"void {name}_other(", 1)

    including = post(
        "write_kernel", "episode-kernel", name=name, source=f'{source}\n#include "{decoy}"\n', language="cpp"
    ).answer
    refused = post("write_kernel", episode="episode-kernel", name=name, source=renamed, language="cpp").answer
    opening = post("write_kernel", "episode-kernel", name=name, source=spying, language="cpp").answer
    written = post("write_kernel", episode="episode-kernel", name=name, source=source, language="cpp").answer
    tuned = post("tune_kernel", episode="episode-kernel", name=name, compilers=["gcc"]).answer

    assert including["result"]["status"] == "build-failed", including
    assert "the-decoy-secret" not in including["result"]["reason"]
    assert "No such file" in including["result"]["reason"]
    assert refused["result"]["status"] == "refused", refused
    assert opening["result"]["status"] == "ok" and opening["performance"]["structured"]["correct"] is True, opening
    assert written["result"]["status"] == "ok" and written["performance"]["structured"]["correct"] is True, written
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

    monkeypatch.setattr(tactics_bench, "start", recording_start)
    sut = ReferenceOptimizer("dace_cpu_canon_tactics_reference")

    with config.scoped_environment({"HPCAGENT_BENCH_SERVICE_PRESET": "XL+fuzz"}), pytest.raises(Started):
        sut.tune(Task(KERNEL, SourceMode.ANY, Language.CPP), "canon_cpu")

    assert presets == ["fuzzed"]


class FinalizedSession:
    """What a tuner column reads of a session: the moves applied, and the library finalize builds."""

    def __init__(self) -> None:
        self.applied: list[MoveResult] = []

    def finalize(self, out: pathlib.Path) -> SimpleNamespace:
        library = out / "libkernel.so"
        library.touch()
        return SimpleNamespace(library=library, workspace_bytes="0", kernels={"kernel_0": "gcc-O3"})


def applied(kind: str, *labels: str) -> MoveResult:
    return MoveResult(status=MoveStatus.APPLIED, kind=kind, labels=labels, reason="")


def column_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, sut: ReferenceOptimizer, solution: object
) -> dict[str, Any]:
    """The plugin record ``sut`` leaves for one episode; a reference solution fuses, the defaults run their stage."""
    session = FinalizedSession()
    monkeypatch.setattr(tactics_bench, "start", lambda setup: (session, None, None))
    monkeypatch.setattr(tactics_bench, "Script", lambda live: live)
    monkeypatch.setattr(tactics_bench, "baseline", lambda live, ctx: live.applied.append(applied("defaults-moves")))
    monkeypatch.setattr(tactics_bench, "reference_solution", lambda kernel: solution)
    sut.tune(Task(KERNEL, SourceMode.ANY, Language.CPP), "canon_cpu")
    record: dict[str, Any] = json.loads((tmp_path / RECORDS_FOLDER / RECORD).read_text())["data"]
    return record


FUSING = SimpleNamespace(solve=lambda live, ctx: live.applied.append(applied("doall-fusion", "doall2", "doall4")))


def test_the_reference_column_records_the_solutions_moves_and_the_default_column_the_default_stage_alone(
    loaded: None, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    reference = column_record(
        monkeypatch,
        tmp_path / "ref",
        ReferenceOptimizer("dace_cpu_canon_tactics_reference", workdir=tmp_path / "ref"),
        FUSING,
    )
    default = column_record(
        monkeypatch,
        tmp_path / "def",
        DefaultOptimizer("dace_cpu_canon_tactics_default", workdir=tmp_path / "def"),
        FUSING,
    )

    assert (reference["reference"], reference["journal"]) == (
        True,
        [["apply_move", {"kind": "doall-fusion", "labels": ["doall2", "doall4"]}]],
    )
    assert (default["reference"], default["journal"]) == (None, [["run_defaults", {"stage": "moves"}]])
    assert reference["final"] == default["final"] == {"kernels": 1} and reference["calls"] == []


def test_the_column_without_wavefronts_starts_its_session_with_every_move_family_off(
    loaded: None, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    started: list[Setup] = []
    real_start = FinalizedSession()
    sut = NoWaveOptimizer("dace_cpu_canon_tactics_cpf_nowave", workdir=tmp_path)

    record = column_record(monkeypatch, tmp_path, sut, FUSING)

    def start(setup: Setup) -> tuple[FinalizedSession, None, None]:
        started.append(setup)
        return real_start, None, None

    monkeypatch.setattr(tactics_bench, "start", start)
    sut.tune(Task(KERNEL, SourceMode.ANY, Language.CPP), "canon_cpu")

    assert started[-1].disabled_families == ("all",)
    assert (record["reference"], record["disabled_families"]) == (None, sorted(FAMILIES))


def test_the_reference_column_says_when_a_kernel_had_no_solution_and_it_ran_the_defaults(
    loaded: None, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    sut = ReferenceOptimizer("dace_cpu_canon_tactics_reference", workdir=tmp_path)

    record = column_record(monkeypatch, tmp_path, sut, None)

    assert (record["reference"], record["journal"]) == (False, [["run_defaults", {"stage": "moves"}]])


def test_the_plugin_declares_the_reference_and_the_default_column(
    loaded: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HPCAGENT_BENCH_USER_PLUGIN_DIRS", str(PLUGIN))
    manifest = tomllib.loads((PLUGIN / "plugin.toml").read_text())

    assert manifest["flavors"]["tactics_default"]["optimizer"] == "tactics_bench:DefaultOptimizer"
    assert manifest["flavors"]["tactics_cpf_nowave"]["optimizer"] == "tactics_bench:NoWaveOptimizer"
    assert "dace_cpu_canon_tactics_default" in manifest["ci"]["optimizers"]
    plugins.register_flavors.cache_clear()
    plugins.register_flavors()
    assert FRAMEWORKS.entries["dace_cpu_canon_tactics_default"].pipelines == ("canon_cpu",)
    assert FRAMEWORKS.entries["dace_gpu_canon_tactics_default"].pipelines == ("canon_gpu",)


def test_the_reference_column_leaves_an_episodes_library_in_the_episodes_folder(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tactics_bench, "start", lambda setup: (FinalizedSession(), None, None))
    monkeypatch.setattr(tactics_bench, "Script", lambda live: live)
    monkeypatch.setattr(tactics_bench, "baseline", lambda script, ctx: None)
    monkeypatch.setattr(tactics_bench, "reference_solution", lambda kernel: None)
    sut = ReferenceOptimizer("dace_cpu_canon_tactics_reference", workdir=tmp_path)

    library = pathlib.Path(sut.tune(Task(KERNEL, SourceMode.ANY, Language.CPP), "canon_cpu").library)
    gc.collect()

    assert library.parent.parent == tmp_path
    assert library.is_file(), "a deferred /submit still needs it after the submission is gone"


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
        reply = Message.model_validate(
            {
                "id": "scripted",
                "type": "message",
                "role": "assistant",
                "model": "scripted",
                "content": content,
                "stop_reason": "tool_use" if content[0]["type"] == "tool_use" else "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )

        async def final() -> Message:
            return reply

        yield SimpleNamespace(get_final_message=final)


def calls(name: str, **arguments: object) -> list[dict[str, object]]:
    return [{"type": "tool_use", "id": f"use-{name}", "name": name, "input": arguments}]


def says(text: str) -> list[dict[str, object]]:
    return [{"type": "text", "text": text}]


def scripted_episode(
    packet_name: str,
    judge: str,
    replies: list[list[dict[str, object]]],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ScriptedModel, runner_common.EndReason, str]:
    """One episode of the bench's Anthropic harness on ``packet_name``: its MCP server lists the judge's tools over
    ``[judge_mcp]`` and ``replies`` are the model's turns. Answers the model, with every request it got, and the end."""
    home, work = tmp_path / "home", tmp_path / "work"
    work.mkdir()
    packet = packets.resolve(packet_name, "cpp")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("JUDGE_URL", judge)
    monkeypatch.setenv("JUDGE_RANK", "0")
    monkeypatch.setenv(agent_plugins.PACKET_AGENTS_ENV, ",".join(packet.agents))
    monkeypatch.setenv(registry.PACKET_TOOLS_ENV, ",".join(packet.tools))
    for key, value in packet.env:
        monkeypatch.setenv(key, value)
    agent_plugins.stage_agents(home, os.environ)
    model = ScriptedModel(replies)
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
    reason, detail = asyncio.run(run_anthropic.run_episode_async(args, runner_common.UsageLog(work / "usage.jsonl")))
    return model, reason, detail


def tool_answer(request: dict[str, Any]) -> dict[str, Any]:
    """The tool result a model request carries last."""
    answer: dict[str, Any] = request["messages"][-1]["content"][0]
    assert answer["type"] == "tool_result" and "is_error" not in answer, answer
    return answer


@pytest.mark.judge
@pytest.mark.parametrize("packet_judge", ["tactics"], indirect=True)
def test_a_mock_agent_reads_moves_measures_and_submits_through_the_mcp_server_to_a_correct_grade(
    loaded: None, packet_judge: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hpcagent_bench_plugins.tactics.wire import TOOLS

    replies = [
        calls("get_cpf"),
        calls("list_moves"),
        calls("run_defaults", stage="moves"),
        calls("get_performance"),
        calls("submit"),
        says("submitted"),
    ]

    model, reason, detail = scripted_episode("tactics", packet_judge, replies, tmp_path, monkeypatch)

    assert reason is runner_common.EndReason.FINISHED, detail
    offered = {tool["name"]: tool for tool in model.sent[0]["tools"]}
    assert set(PACKETS["tactics"].tools) <= offered.keys(), "every packet tool, the judge's listed over MCP"
    assert offered["apply_move"]["description"].startswith(f"[{TOOLS['apply_move'].label}]"), "as the judge lists it"
    read, listed, moved, measured, submitted = (tool_answer(request) for request in model.sent[1:6])
    assert "doall" in str(read["content"]) and str(listed["content"]).strip(), (read, listed)
    assert "error" not in str(moved["content"]).lower(), moved
    assert re.search(r"correct\W+true", str(measured["content"]), re.IGNORECASE), measured
    assert re.search(r"correct\W+true", str(submitted["content"]), re.IGNORECASE), submitted


@pytest.mark.judge
@pytest.mark.parametrize("packet_judge", ["tactics-subagents"], indirect=True)
def test_a_subagent_calls_the_judge_as_the_episode_that_started_it_and_passes_the_gate_of_its_phase(
    loaded: None, packet_judge: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replies = [
        calls("advance_phase"),
        calls("advance_phase"),
        calls("advance_phase"),
        calls(subagents.TASK_TOOL, subagent_type="kernel-optimizer", prompt="Optimize the first kernel."),
        calls("list_kernels"),
        says("nothing to gain"),
        says("every kernel is done"),
    ]

    model, reason, detail = scripted_episode("tactics-subagents", packet_judge, replies, tmp_path, monkeypatch)

    # three advance_phase calls, the Task call, then the subagent's first request and the one after list_kernels
    first, started, listed = model.sent[0], model.sent[4], model.sent[5]
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
    column = "dace_cpu_canon_tactics_reference"
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


# a straight diamond of maps (jacobi_1d's maps inside a time loop: the bench-plugin job's stub episodes)
@pytest.mark.judge
@pytest.mark.parametrize("kernel", [KERNEL])
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
    # The entry checks workspace_size against its layout and aborts before it touches the scratch
    assert without.root.exit_code != 0 and "signal SIGABRT" in without.root.stderr, without


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
    to_phase("episode-gpu", "scopes")

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
