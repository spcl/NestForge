# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``POST /tactics/mcp``: the tactics tools over MCP, inside the judge.

The body is one JSON-RPC message of MCP's streamable HTTP transport (JSON answers, no stream), beside the judge's own
``episode_id`` and ``rank``. ``tools/list`` gives every served tool with the schema of its pydantic model
(:func:`~tactics.wire.mcp_tools`); ``tools/call`` runs exactly what ``POST /tactics/<tool>`` runs, with the identity
fields (``kernel``, ``tokens``, ``disabled_families``, ``agent_phases``) in ``params._meta``, and answers the text the
agent-side tool shows (:func:`~hpcagent_bench_plugins.tactics.answers.agent_answer`). Plain dicts, so the judge needs
no MCP library.
"""

import json
from collections.abc import Callable, Mapping
from http import HTTPStatus
from importlib.metadata import version
from typing import Any

from hpcagent_bench.plugin_api import Judge, JudgeAnswer, RouteRequest
from hpcagent_bench_plugins.tactics.answers import agent_answer
from hpcagent_bench_plugins.tactics.wire import IDENTITY, MOVE_DEFAULTS, mcp_tools

from tactics.server import CALLS

#: The MCP revision the endpoint speaks when the client names none.
PROTOCOL = "2025-06-18"
#: JSON-RPC's code for a method the endpoint does not serve.
NO_METHOD = -32601

#: A route's answer: the HTTP status and the payload; ``None`` when the client left before it.
Outcome = tuple[int, JudgeAnswer] | None
#: The route's tool call: ``(judge, tool, body, episode_id, terminal)`` -> its answer.
ToolCall = Callable[[Judge, str, Mapping[str, object], str, bool], Outcome]


class RpcAnswer(JudgeAnswer):
    """One JSON-RPC answer; a notification's is empty."""

    jsonrpc: str = "2.0"
    id: int | str | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


def call_result(name: str, done: tuple[int, JudgeAnswer]) -> dict[str, Any]:
    """The route's answer to ``name`` as an MCP tool result: the agent's text, or the error."""
    code, payload = done
    body = payload.to_json()
    if code != HTTPStatus.OK:
        return {"content": [{"type": "text", "text": str(body.get("error", body))}], "isError": True}
    shown = agent_answer(name, body, MOVE_DEFAULTS)
    return {"content": [{"type": "text", "text": json.dumps(shown)}], "structuredContent": shown, "isError": False}


def mcp_answer(judge: Judge, request: RouteRequest, call: ToolCall) -> Outcome:
    """The answer to the JSON-RPC message in ``request.body``; tool calls go through ``call``, the route's own."""
    body = request.body
    method, message_id = body.get("method"), body.get("id")
    params = body.get("params") if isinstance(body.get("params"), dict) else {}
    assert isinstance(params, dict), "checked above"
    if method == "initialize":
        result: dict[str, Any] = {
            "protocolVersion": params.get("protocolVersion") or PROTOCOL,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "tactics", "version": version("tactics")},
        }
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": mcp_tools(CALLS)}
    elif method == "tools/call":
        name = str(params.get("name") or "")
        arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        meta = params.get("_meta") if isinstance(params.get("_meta"), dict) else {}
        assert isinstance(arguments, dict) and isinstance(meta, dict), "checked above"
        identity = {key: value for key, value in meta.items() if key in IDENTITY}
        done = call(judge, name, {**arguments, **identity}, request.episode_id, bool(meta.get("terminal")))
        if done is None:
            return None
        result = call_result(name, done)
    elif message_id is None:
        # a notification (``notifications/initialized``): accepted, nothing to answer
        return HTTPStatus.ACCEPTED, RpcAnswer()
    else:
        error = {"code": NO_METHOD, "message": f"the tactics endpoint serves no {method!r}"}
        return HTTPStatus.OK, RpcAnswer(jsonrpc="2.0", id=message_id, error=error)
    return HTTPStatus.OK, RpcAnswer(jsonrpc="2.0", id=message_id, result=result)
