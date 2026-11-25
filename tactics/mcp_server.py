# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The tactics tools as an MCP server over one kernel's session.

Every tool of :data:`tactics.wire.TOOLS` the session serves is listed with the JSON schema its pydantic model gives
and the MCP annotations of its effect; a call is validated by that model and runs in a :class:`SessionProcess`, the
isolated session the HPCAgent-Bench judge runs too. ``tactics-mcp KERNEL`` serves a bench kernel over stdio.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from importlib.metadata import version

import anyio
import pydantic
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from tactics.server import CALLS, Reply, SessionProcess, Setup
from tactics.wire import TOOLS, mcp_tools

INSTRUCTIONS = (
    "Optimize one DaCe program only through legal, checked moves: list the moves, apply them, measure with "
    "get_performance, and finalize. Each tool names the phases it serves in brackets."
)


def listed_tools() -> list[types.Tool]:
    """Every tool the session serves, as :func:`~tactics.wire.mcp_tools` lists them (the judge's endpoint too)."""
    return [types.Tool.model_validate(tool) for tool in mcp_tools(CALLS)]


def failed(error: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(text=error)], is_error=True)


def tool_result(reply: Reply) -> types.CallToolResult:
    """``reply`` as the client reads it: the result (and performance) as JSON, or the error."""
    if reply.error:
        return failed(reply.error)
    payload = reply.model_dump(mode="json", include={"result", "performance"}, exclude_none=True)
    return types.CallToolResult(content=[types.TextContent(text=json.dumps(payload))], structured_content=payload)


def checked_call(name: str, arguments: dict[str, object] | None) -> tuple[str, dict[str, object]] | str:
    """``name`` and its arguments as the session takes them, or why the call is refused."""
    tool = TOOLS.get(name)
    if tool is None or name not in CALLS:
        return f"no tool {name!r}; tools: {sorted(n for n in TOOLS if n in CALLS)}"
    try:
        return name, tool.args.model_validate(arguments or {}).model_dump()
    except pydantic.ValidationError as err:
        return f"{name}: {err}"


def server(session: SessionProcess) -> Server:
    """The MCP server over ``session``; calls run one at a time, as the session takes them."""
    lock = anyio.Lock()

    async def on_list_tools(context: object, params: object) -> types.ListToolsResult:
        return types.ListToolsResult(tools=listed_tools())

    async def on_call_tool(context: object, params: types.CallToolRequestParams) -> types.CallToolResult:
        call = checked_call(params.name, params.arguments)
        if isinstance(call, str):
            return failed(call)
        async with lock:
            reply = await anyio.to_thread.run_sync(session.call, *call)
        return tool_result(reply)

    return Server(
        "tactics",
        version=version("tactics"),
        instructions=INSTRUCTIONS,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


async def serve_stdio(setup: Setup) -> None:
    session = SessionProcess(setup)
    try:
        async with stdio_server() as (read, write):
            mcp = server(session)
            await mcp.run(read, write, mcp.create_initialization_options())
    finally:
        session.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tactics-mcp", description=__doc__.splitlines()[0])
    parser.add_argument("kernel", help="a HPCAgent-Bench kernel, e.g. jacobi_2d")
    parser.add_argument("--preset", default="S", help="the size preset (S, M, L, ...)")
    parser.add_argument("--datatype", default="float64", help="the graded datatype")
    parser.add_argument("--gpu", action="store_true", help="target the GPU as well as the CPU")
    parser.add_argument("--root", default="", help="the folder finalize builds in; empty: a shared one")
    options = parser.parse_args(argv)
    anyio.run(serve_stdio, Setup(options.kernel, options.preset, options.datatype, options.gpu, options.root))


if __name__ == "__main__":
    main()
