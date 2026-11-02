# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The NestForge tools on the agent side: each a thin ``POST /nestforge/<tool>`` (DaCe lives in the judge)."""

from collections.abc import Callable
from typing import Any

import pydantic
from hpcagent_agent.tools import http_json, profile_tool, score, submit
from hpcagent_agent.tools.registry import Hints, Setup, Spec, tool
from hpcagent_bench_plugins.nestforge.wire import ROUTE, TOOLS, Effect, NoArgs

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


def delivering(deliver: Callable[[dict[str, Any]], dict[str, Any]]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """``deliver`` of the library ``finalize`` builds from the session: the program is what the moves made of it."""

    def run(payload: dict[str, Any]) -> dict[str, Any]:
        # finalize takes no arguments: what the call adds (an instrument, say) is for ``deliver`` alone
        built = poster("finalize")({"kernel": http_json.assigned_kernel(payload)})
        if "error" in built:
            return {"ok": False, "error": built["error"]}
        final = built["result"]
        return deliver({**payload, "library": final["library"], "workspace_bytes": final["workspace_bytes"]})

    return run


SUBMIT = "Finalize the program as your moves left it and submit that library for the recorded grade. Takes no code."
SCORE = "Finalize the program as your moves left it and preview that library's grade. Takes no code; never recorded."
tool(spec=Spec(SUBMIT, NoArgs, f"- `submit` -- {SUBMIT}", HINTS[Effect.ADD]), name="submit", overrides=True)(
    delivering(submit.run)
)
tool(spec=Spec(SCORE, NoArgs, f"- `score` -- {SCORE}", HINTS[Effect.READ]), name="score", preview=True, overrides=True)(
    delivering(score.run)
)

PROFILE = (
    "Finalize the program as your moves left it and show where that library's time goes. Takes no code; never scored."
)


def profile_spec(setup: Setup) -> Spec:
    """The bench's ``profile`` on ``setup`` with its instruments only: the library comes from the session."""
    offered = profile_tool.profile_input(setup).model_fields
    code = {*http_json.SubmissionInput.model_fields, "language", "requests"}
    fields: dict[str, Any] = {name: (field.annotation, field) for name, field in offered.items() if name not in code}
    args = pydantic.create_model("ProfileArgs", __config__=pydantic.ConfigDict(extra="forbid"), **fields)
    return Spec(PROFILE, args, f"- `profile` -- {PROFILE}", HINTS[Effect.READ])


tool(spec=profile_spec, name="profile", preview=True, overrides=True)(delivering(profile_tool.run))
