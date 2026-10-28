# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The NestForge tools on the agent side: each a thin ``POST /nestforge/<tool>`` (DaCe lives in the judge)."""

from collections.abc import Callable
from typing import Any

from hpcagent_agent.tools import http_json
from hpcagent_agent.tools.registry import Hints, Spec, tool
from hpcagent_bench_plugins.nestforge.wire import ROUTE, TOOLS, Effect

#: The MCP annotations of each effect. Every call reaches only the judge's own NestForge session.
HINTS = {
    Effect.READ: Hints(read_only=True, idempotent=True, open_world=False),
    Effect.ADD: Hints(read_only=False, destructive=False, open_world=False),
    Effect.REPLACE: Hints(read_only=False, destructive=True, open_world=False),
}


def poster(name: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def run(payload: dict[str, Any]) -> dict[str, Any]:
        return http_json.post_judge(f"/{ROUTE}/{name}", {**payload, "kernel": http_json.assigned_kernel(payload)})

    return run


for name, entry in TOOLS.items():
    spec = Spec(
        f"[{entry.phase}] {entry.description}", entry.args, f"- `{name}` -- {entry.description}", HINTS[entry.effect]
    )
    tool(spec=spec, core=False, name=name)(poster(name))
