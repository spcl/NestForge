# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""tactics.mcp_server: the tools of tactics.wire as MCP tools, their schemas from the pydantic models, calls checked
before they reach the session."""

from pathlib import Path

import anyio
import pytest
from mcp import types
from mcp.client.session import ClientSession
from mcp.shared.memory import create_client_server_memory_streams

from tactics.mcp_server import checked_call, listed_tools, server
from tactics.paths import REPO_ROOT
from tactics.server import CALLS, SessionProcess, Setup
from tactics.wire import TOOLS


def tool(name: str) -> types.Tool:
    (found,) = [t for t in listed_tools() if t.name == name]
    return found


def test_every_tool_the_session_serves_is_listed_and_no_other() -> None:
    assert {t.name for t in listed_tools()} == set(TOOLS) & set(CALLS)


def test_reads_are_read_only_and_a_replacing_tool_is_destructive() -> None:
    listing, rewrite = tool("list_moves").annotations, tool("write_kernel").annotations

    assert listing is not None and (listing.read_only_hint, listing.destructive_hint) == (True, False)
    assert rewrite is not None and (rewrite.read_only_hint, rewrite.destructive_hint) == (False, True)


def test_arguments_the_model_rejects_never_reach_the_session() -> None:
    refused = checked_call("apply_move", {"kind": "tile-doall"})

    assert isinstance(refused, str) and refused.startswith("apply_move:") and "labels" in refused


def test_an_unknown_tool_is_refused_with_the_served_ones() -> None:
    refused = checked_call("no_such_tool", {})

    assert isinstance(refused, str) and "list_moves" in refused


async def over_mcp(setup: Setup) -> tuple[list[str], types.CallToolResult, types.CallToolResult]:
    session = SessionProcess(setup)
    try:
        mcp = server(session)
        async with create_client_server_memory_streams() as (client_streams, server_streams):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(mcp.run, *server_streams, mcp.create_initialization_options())
                async with ClientSession(*client_streams) as client:
                    await client.initialize()
                    names = [t.name for t in (await client.list_tools()).tools]
                    moves = await client.call_tool("list_moves", {"kind": "interchange-for-doall"})
                    wrong = await client.call_tool("apply_move", {"kind": "interchange-for-doall"})
                tasks.cancel_scope.cancel()
    finally:
        session.close()
    assert isinstance(moves, types.CallToolResult) and isinstance(wrong, types.CallToolResult)
    return names, moves, wrong


@pytest.mark.integration
def test_a_client_lists_the_tools_and_lists_moves_of_a_real_kernel(tmp_path: Path) -> None:
    names, moves, wrong = anyio.run(over_mcp, Setup("tsvc_2_s231", "S", "float64", False, str(tmp_path)))

    assert "apply_move" in names
    assert not moves.is_error and moves.structured_content is not None
    assert [m["kind"] for m in moves.structured_content["result"]] == ["interchange-for-doall"]
    assert wrong.is_error, "a call without labels is refused before it reaches the session"


def test_the_plugin_reads_the_same_tool_table() -> None:
    # the agent image has no tactics, so the plugin keeps a copy until the bench serves the tools over MCP (bench 1.1)
    plugin = REPO_ROOT / "plugin" / "tactics" / "wire.py"

    assert plugin.read_text() == (REPO_ROOT / "tactics" / "wire.py").read_text()
