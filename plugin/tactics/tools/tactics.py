# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The agent side of Tactics: submit, score and profile. The other tools come from the judge's ``POST /tactics/mcp``
([judge_mcp] in plugin.toml); these three finalize the session's program first, so they stay here."""

import os
from collections.abc import Callable
from typing import Any

import pydantic
from hpcagent_agent.tools import http_json, profile_tool, score, submit
from hpcagent_agent.tools.registry import Hints, Setup, Spec, tool
from hpcagent_bench_plugins.tactics.answers import agent_answer, profile_answer, score_answer
from hpcagent_bench_plugins.tactics.wire import AGENT_PHASES_ENV, MOVE_DEFAULTS, ROUTE, Effect, NoArgs

#: The MCP annotations of each effect. Every call reaches only the judge's own Tactics session.
HINTS = {
    Effect.READ: Hints(read_only=True, idempotent=True, open_world=False),
    Effect.ADD: Hints(read_only=False, destructive=False, open_world=False),
    Effect.REPLACE: Hints(read_only=False, destructive=True, open_world=False),
}


#: The packet setting naming the move families a run switches off, comma-separated (``all``: every family). A
#: packet's ``env`` reaches only this side, so every call carries it to the judge, which fixes it at the first.
DISABLED_FAMILIES_ENV = "TACTICS_DISABLED_FAMILIES"


def disabled_families() -> list[str]:
    return [name.strip() for name in os.environ.get(DISABLED_FAMILIES_ENV, "").split(",") if name.strip()]


def agent_phases() -> list[str]:
    """The packet's phases for the agent (``TACTICS_AGENT_PHASES``), which every call carries like the families."""
    return [name.strip() for name in os.environ.get(AGENT_PHASES_ENV, "").split(",") if name.strip()]


def poster(name: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """The tool ``name``: its call posted to the judge, and the answer as the agent reads it (:mod:`answers`)."""

    def run(payload: dict[str, Any]) -> dict[str, Any]:
        kernel = http_json.assigned_kernel(payload)
        answer = http_json.post_judge(
            f"/{ROUTE}/{name}",
            {**payload, "kernel": kernel, "disabled_families": disabled_families(), "agent_phases": agent_phases()},
        )
        return agent_answer(name, answer, MOVE_DEFAULTS)

    return run


def rendered(
    run: Callable[[dict[str, Any]], dict[str, Any]], render: Callable[[dict[str, Any]], dict[str, Any]]
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """``run``, its answer as the agent reads it (:mod:`answers`)."""

    def answer(payload: dict[str, Any]) -> dict[str, Any]:
        return render(run(payload))

    return answer


def delivering(
    deliver: Callable[[dict[str, Any]], dict[str, Any]], finalize: str = "finalize"
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """``deliver`` of the library ``finalize`` builds from the session: the program is what the moves made of it."""

    def run(payload: dict[str, Any]) -> dict[str, Any]:
        # finalize takes no arguments: what the call adds (an instrument, say) is for ``deliver`` alone
        built = poster(finalize)({"kernel": http_json.assigned_kernel(payload)})
        if "error" in built:
            return {"ok": False, "error": built["error"]}
        final = built["result"]
        # the argument layouts the program works on; the harness lays the arguments out by them
        laid = {"requests": final["requests"]} if final.get("requests") else {}
        return deliver({**payload, "library": final["library"], "workspace_bytes": final["workspace_bytes"], **laid})

    return run


SUBMIT = "Finalize the program as your moves left it and submit that library for the recorded grade. Takes no code."
SCORE = "Finalize the program as your moves left it and preview that library's grade. Takes no code; never recorded."
tool(spec=Spec(SUBMIT, NoArgs, submit.PROMPT, HINTS[Effect.ADD]), name="submit", overrides=True)(
    delivering(submit.run, "finalize?terminal=1")  # the build queues as the /submit it delivers to does
)
tool(spec=Spec(SCORE, NoArgs, f"- `score` -- {SCORE}", HINTS[Effect.READ]), name="score", preview=True, overrides=True)(
    rendered(delivering(score.run), score_answer)
)

PROFILE = (
    "Finalize the program as your moves left it and show where that library's time goes. Takes no code; never scored."
)

#: The bench's agent-side ``profile`` does not offer the judge's ``ranks`` (bench B2), so the plugin adds it.
RANKS = pydantic.Field(
    None,
    min_length=1,
    description="Distributed task only: run the program at each of these rank counts as a grade runs it, never "
    "scored. Each run reports time_us (the timed bracket on the slowest rank), mpi_time_us (the time inside MPI calls "
    "on that rank: what an overlap of communication and compute reduces) and whether the outputs are correct. "
    "No instrument runs with it.",
)


def profile_spec(setup: Setup) -> Spec:
    """The bench's ``profile`` on ``setup`` with its instruments only: the library comes from the session."""
    offered = profile_tool.profile_input(setup).model_fields
    code = {*http_json.SubmissionInput.model_fields, "language", "requests"}
    fields: dict[str, Any] = {name: (field.annotation, field) for name, field in offered.items() if name not in code}
    if setup.request_actions.distributed and "ranks" not in fields:
        fields["ranks"] = (list[pydantic.PositiveInt] | None, RANKS)
    args = pydantic.create_model("ProfileArgs", __config__=pydantic.ConfigDict(extra="forbid"), **fields)
    return Spec(PROFILE, args, f"- `profile` -- {PROFILE}", HINTS[Effect.READ])


def profile_run(payload: dict[str, Any]) -> dict[str, Any]:
    """The bench's ``/profile`` body, with ``ranks`` when the agent sent it."""
    body = profile_tool.profile_body(payload)
    if payload.get("ranks"):
        body["ranks"] = payload["ranks"]
    return http_json.post_judge("/profile", body)


tool(spec=profile_spec, name="profile", preview=True, overrides=True)(rendered(delivering(profile_run), profile_answer))
