# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``scripts/answer_tokens.py``: the answers of a fixed call sequence, rendered as the agent reads them and counted."""

import importlib.util
import json
import sys
from collections.abc import Iterator
from types import ModuleType
from typing import Any

import pytest
from hpcagent_agent.plugins import Plugin

from tactics.paths import REPO_ROOT
from tactics.server import Reply


def script() -> ModuleType:
    """The script as a module (``scripts/`` is no package)."""
    spec = importlib.util.spec_from_file_location("answer_tokens", REPO_ROOT / "scripts" / "answer_tokens.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["answer_tokens"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def answers(tactics_plugin: tuple[Plugin, ...]) -> Iterator[ModuleType]:
    from hpcagent_bench_plugins.tactics import answers  # importable once the bench loaded the plugin

    yield answers


def assistant(*blocks: dict[str, Any]) -> str:
    return json.dumps({"type": "assistant", "message": {"content": list(blocks)}})


def test_a_transcript_gives_its_tactics_calls_in_order_once_each() -> None:
    sut = script()
    lines = [
        assistant({"type": "tool_use", "id": "a", "name": "mcp__hpcagent__view_tree", "input": {"expand": ["s1"]}}),
        assistant({"type": "tool_use", "id": "b", "name": "mcp__hpcagent__score", "input": {}}),
        assistant({"type": "tool_use", "id": "a", "name": "mcp__hpcagent__view_tree", "input": {"expand": ["s1"]}}),
        assistant({"type": "thinking", "thinking": "Let me make the call."}),
        assistant(
            {"type": "tool_use", "id": "c", "name": "apply_move", "input": {"kind": "tile-for", "labels": ["f"]}}
        ),
    ]

    calls = list(sut.transcript_calls(lines))

    assert calls == [("view_tree", {"expand": ["s1"]}), ("apply_move", {"kind": "tile-for", "labels": ["f"]})]


def test_every_answer_is_re_sent_once_per_later_call() -> None:
    sut = script()

    assert sut.resend_cost([10, 20, 30]) == 10 * 2 + 20 * 1
    assert sut.resend_cost([]) == 0


def test_the_v1_answer_is_the_bench_mcp_text_of_the_route_answer() -> None:
    sut = script()

    ok = sut.rendered_v1("list_moves", Reply(result=[{"kind": "tile-for", "labels": ["for3"]}]))
    failed = sut.rendered_v1("apply_move", Reply(error="no row for9"))

    assert json.loads(ok) == {
        "tool": "list_moves",
        "result": [{"kind": "tile-for", "labels": ["for3"]}],
        "performance": None,
    }
    assert ok.startswith('{\n  "performance": null,')  # indented and key-sorted, as the MCP server sends it
    assert json.loads(failed) == {"ok": False, "error": "apply_move: no row for9"}


def test_the_current_answer_is_the_plugins_rendering_and_shorter_than_v1(
    tactics_plugin: tuple[Plugin, ...], answers: ModuleType
) -> None:
    """``current`` is what the agent reads since #102: the plugin's own rendering of the route answer, so the
    measurement follows the plugin, not a copy of it."""
    sut = script()
    moves = [{"kind": "tile-for", "labels": ["for3"]}, {"kind": "interchange-for", "labels": ["for3", "for4"]}]
    reply = Reply(result=moves)
    route = sut.ToolAnswer(tool="list_moves", result=moves, performance=None).to_json()
    from hpcagent_bench_plugins.tactics.wire import MOVE_DEFAULTS  # importable once the bench loaded the plugin

    current = sut.rendered_current("list_moves", reply)

    assert json.loads(current) == json.loads(json.dumps(answers.RENDERED["list_moves"](route, MOVE_DEFAULTS)))
    assert len(current) < len(sut.rendered_v1("list_moves", reply))
    assert sut.rendered_current("apply_move", Reply(error="no row for9")) == sut.rendered_v1(
        "apply_move", Reply(error="no row for9")
    )


def test_counting_skips_timed_tools_unless_measured_and_counts_each_format() -> None:
    sut = script()
    calls = [("view_tree", {}), ("get_performance", {}), ("apply_move", {"kind": "rollback", "labels": []})]
    ran: list[str] = []

    def run(tool: str, arguments: dict[str, Any]) -> Reply:
        ran.append(tool)
        return Reply(result=tool)

    counted = sut.count(calls, run, {"chars": lambda tool, reply: str(reply.result)}, len, measure=False)

    assert ran == ["view_tree", "apply_move"]
    assert [(row.tool, row.tokens) for row in counted] == [("view_tree", {"chars": 9}), ("apply_move", {"chars": 10})]
    assert counted[1].call_tokens == len('{"name": "apply_move", "input": {"kind": "rollback", "labels": []}}')


def test_the_scripts_answer_record_has_the_routes_fields(tactics_plugin: tuple[Plugin, ...]) -> None:
    sut = script()
    (route,) = [m for name, m in sys.modules.items() if name.endswith("routes.tactics") and "ToolAnswer" in vars(m)]

    assert sut.ToolAnswer.model_fields.keys() == vars(route)["ToolAnswer"].model_fields.keys()


def user(*blocks: dict[str, Any]) -> str:
    return json.dumps({"type": "user", "message": {"content": list(blocks)}})


def test_a_recorded_transcript_counts_every_tool_call_with_the_result_the_agent_read() -> None:
    sut = script()
    lines = [
        assistant({"type": "tool_use", "id": "a", "name": "Bash", "input": {"command": "sed -n 1,40p kernel.cpp"}}),
        user({"type": "tool_result", "tool_use_id": "a", "content": "for (int i = 0; i < n; ++i)"}),
        assistant({"type": "tool_use", "id": "b", "name": "Edit", "input": {"file_path": "k.cpp", "old_string": "x"}}),
        user(
            {
                "type": "tool_result",
                "tool_use_id": "b",
                "content": [{"type": "text", "text": "no match"}],
                "is_error": True,
            }
        ),
        assistant({"type": "tool_use", "id": "c", "name": "mcp__hpcagent__view_tree", "input": {}}),
    ]

    counted = sut.count_recorded(sut.recorded_calls(lines), len)

    assert [(row.tool, row.status, row.tokens["recorded"]) for row in counted] == [
        ("Bash:read", "ok", len("for (int i = 0; i < n; ++i)")),
        ("Edit:edit", "error", len("no match")),
        ("view_tree", "ok", 0),
    ]
    assert counted[0].call_tokens == len('{"name": "Bash", "input": {"command": "sed -n 1,40p kernel.cpp"}}')


def test_shell_commands_count_as_read_edit_build_or_run() -> None:
    sut = script()
    kinds = {
        "cat kernel.cpp | head -50": "read",
        "grep -n omp kernel.cpp": "read",
        "sed -i 's/i++/++i/' kernel.cpp": "edit",
        "cat > kernel.cpp <<'EOF'\nint main() {}\nEOF": "edit",
        "cd build && make -j8 2>&1 | tail -20": "build",
        "g++ -O3 -fopenmp kernel.cpp -o k": "build",
        "./k > out.txt": "run",
        "echo done": "other",
    }

    assert {command: sut.shell_kind(command) for command in kinds} == kinds


def test_an_unknown_tokenizer_name_is_refused_by_name() -> None:
    sut = script()

    with pytest.raises(ValueError, match="unknown tokenizer 'o200k'"):
        sut.token_counter("o200k")


def test_o200k_base_counts_special_token_text_as_plain_text() -> None:
    sut = script()
    tokens_of = sut.token_counter("o200k_base")

    assert tokens_of("hello world") == 2
    assert tokens_of("<|endoftext|>") > 1


def test_a_wrapped_command_counts_as_the_command_it_wraps() -> None:
    sut = script()

    assert sut.shell_kind("nice -n 5 timeout 600 make -j4") == "build"
    assert sut.shell_kind("OMP_NUM_THREADS=8 ./kernel") == "run"


def test_a_call_with_invalid_arguments_gets_the_routes_error_and_never_reaches_the_session(
    tactics_plugin: tuple[Plugin, ...],
) -> None:
    """An agent's ``apply_move`` with no arguments (bfs, 11-03) was refused by the route's schema; replayed straight
    into the session it crashed the replay with ``KeyError: 'kind'``."""
    module = script()
    refused = module.invalid_arguments("apply_move", {})
    assert refused is not None and refused.error is not None
    assert refused.error.startswith("apply_move: ") and "kind" in refused.error and "labels" in refused.error
    assert module.invalid_arguments("apply_move", {"kind": "doall-fusion", "labels": ["doall6", "doall7"]}) is None
    assert module.invalid_arguments("Bash", {}) is None
