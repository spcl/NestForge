# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The NestForge tools on the agent side: each a thin ``POST /nestforge/<tool>`` (DaCe lives in the judge)."""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pydantic
from hpcagent_agent.tools import http_json, profile_tool, score, submit
from hpcagent_agent.tools.registry import Hints, Setup, Spec, tool
from hpcagent_bench_plugins.nestforge.answers import RENDERED, phase_answer, profile_answer, score_answer
from hpcagent_bench_plugins.nestforge.wire import AGENT_PHASES_ENV, MOVE_DEFAULTS, ROUTE, TOOLS, Effect, NoArgs

#: The MCP annotations of each effect. Every call reaches only the judge's own NestForge session.
HINTS = {
    Effect.READ: Hints(read_only=True, idempotent=True, open_world=False),
    Effect.ADD: Hints(read_only=False, destructive=False, open_world=False),
    Effect.REPLACE: Hints(read_only=False, destructive=True, open_world=False),
}


#: The packet setting naming the move families a run switches off, comma-separated (``all``: every family). A
#: packet's ``env`` reaches only this side, so every call carries it to the judge, which fixes it at the first.
DISABLED_FAMILIES_ENV = "NESTFORGE_DISABLED_FAMILIES"


def disabled_families() -> list[str]:
    return [name.strip() for name in os.environ.get(DISABLED_FAMILIES_ENV, "").split(",") if name.strip()]


def agent_phases() -> list[str]:
    """The packet's phases for the agent (``NESTFORGE_AGENT_PHASES``), which every call carries like the families."""
    return [name.strip() for name in os.environ.get(AGENT_PHASES_ENV, "").split(",") if name.strip()]


def poster(name: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """The tool ``name``: its call posted to the judge, and the answer as the agent reads it (:mod:`answers`)."""
    render = RENDERED.get(name)

    def run(payload: dict[str, Any]) -> dict[str, Any]:
        kernel = http_json.assigned_kernel(payload)
        answer = http_json.post_judge(
            f"/{ROUTE}/{name}",
            {**payload, "kernel": kernel, "disabled_families": disabled_families(), "agent_phases": agent_phases()},
        )
        return answer if render is None else render(answer, MOVE_DEFAULTS)

    return run


#: The guides of the phases, one file each; a phase tool's answer carries the guide of the phase it reached.
PHASE_GUIDES = Path(__file__).resolve().parents[1] / "prompt" / "phases"
#: The tools that move between phases.
PHASE_TOOLS = frozenset({"advance_phase", "undo_phase", "return_to_phase"})


def guided(run: Callable[[dict[str, Any]], dict[str, Any]]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """``run``, its answer with the guide of the phase it moved to."""

    def answer(payload: dict[str, Any]) -> dict[str, Any]:
        reply = run(payload)
        result = reply.get("result")
        if isinstance(result, dict) and result.get("status") == "moved":
            guide = PHASE_GUIDES / f"{result.get('phase')}.md"
            if guide.is_file():
                return {**reply, "guide": guide.read_text()}
        return reply

    return answer


def rendered(
    run: Callable[[dict[str, Any]], dict[str, Any]], render: Callable[[dict[str, Any]], dict[str, Any]]
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """``run``, its answer as the agent reads it (:mod:`answers`)."""

    def answer(payload: dict[str, Any]) -> dict[str, Any]:
        return render(run(payload))

    return answer


for name, entry in TOOLS.items():
    spec = Spec(
        f"[{entry.label}] {entry.description}", entry.args, f"- `{name}` -- {entry.description}", HINTS[entry.effect]
    )
    # a phase tool's guide is added from its record, so the answer renders after it
    run = rendered(guided(poster(name)), phase_answer) if name in PHASE_TOOLS else poster(name)
    # in blind mode no tool answers how the program does
    tool(spec=spec, core=False, name=name, preview=name == "get_performance")(run)


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


def profile_spec(setup: Setup) -> Spec:
    """The bench's ``profile`` on ``setup`` with its instruments only: the library comes from the session."""
    offered = profile_tool.profile_input(setup).model_fields
    code = {*http_json.SubmissionInput.model_fields, "language", "requests"}
    fields: dict[str, Any] = {name: (field.annotation, field) for name, field in offered.items() if name not in code}
    args = pydantic.create_model("ProfileArgs", __config__=pydantic.ConfigDict(extra="forbid"), **fields)
    return Spec(PROFILE, args, f"- `profile` -- {PROFILE}", HINTS[Effect.READ])


tool(spec=profile_spec, name="profile", preview=True, overrides=True)(
    rendered(delivering(profile_tool.run), profile_answer)
)
