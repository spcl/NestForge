# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``scripts/answer_tokens.py``: the answers of a fixed call sequence, rendered as the agent reads them and counted."""

import importlib.util
import json
import sys
from types import ModuleType

from nestforge.paths import REPO_ROOT
from nestforge.server import Reply


def script() -> ModuleType:
    """The script as a module (``scripts/`` is no package)."""
    spec = importlib.util.spec_from_file_location("answer_tokens", REPO_ROOT / "scripts" / "answer_tokens.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["answer_tokens"] = module
    spec.loader.exec_module(module)
    return module


def assistant(*blocks: dict) -> str:
    return json.dumps({"type": "assistant", "message": {"content": list(blocks)}})


def test_a_transcript_gives_its_nestforge_calls_in_order_once_each():
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


def test_every_answer_is_re_sent_once_per_later_call():
    sut = script()

    assert sut.resend_cost([10, 20, 30]) == 10 * 2 + 20 * 1
    assert sut.resend_cost([]) == 0


def test_the_v1_answer_is_the_bench_mcp_text_of_the_route_answer():
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


def test_counting_skips_timed_tools_unless_measured_and_counts_each_format():
    sut = script()
    calls = [("view_tree", {}), ("get_performance", {}), ("apply_move", {"kind": "rollback", "labels": []})]
    ran: list[str] = []

    def run(tool: str, arguments: dict) -> Reply:
        ran.append(tool)
        return Reply(result=tool)

    counted = sut.count(calls, run, {"chars": lambda tool, reply: str(reply.result)}, len, measure=False)

    assert ran == ["view_tree", "apply_move"]
    assert [(row.tool, row.tokens) for row in counted] == [("view_tree", {"chars": 9}), ("apply_move", {"chars": 10})]
    assert counted[1].call_tokens == len('{"name": "apply_move", "input": {"kind": "rollback", "labels": []}}')


def test_the_scripts_answer_record_has_the_routes_fields(nestforge_plugin):
    sut = script()
    (route,) = [m for name, m in sys.modules.items() if name.endswith("routes.nestforge") and "ToolAnswer" in vars(m)]

    assert sut.ToolAnswer.model_fields.keys() == vars(route)["ToolAnswer"].model_fields.keys()
