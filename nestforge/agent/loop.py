# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A minimal agent loop over stages 2 to 5 of a :class:`~nestforge.session.Session`. Each turn the prompt is
rebuilt from the task, the current tree or kernels, the latest feedback report and a one-line log of every action
tried; no chat history grows. Whatever the agent leaves undone, the stage's deterministic default completes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from nestforge.agent.llm import Reply, Tool, ToolCall
from nestforge.session import MoveResult, Session
from nestforge.stages.moves import MOVE_SHAPES

#: One model call: system text, user prompt, tools -> reply.
Chat = Callable[[str, str, list[Tool]], Reply]

SYSTEM = (
    "You optimize a program for speed with NestForge. Act only through the tools; labels and epochs come from the "
    "program view. Reply without a tool call when this stage needs nothing more."
)
ANALYST = (
    "Rewrite this compiler and profiler feedback into at most five short lines of concrete guidance for the next "
    "optimization step, most important first. Keep kernel names and numbers."
)

#: Characters of a tool result one log line keeps.
LOG_LINE = 160

#: Model calls per stage.
TURNS = 4

LABELS = {"type": "array", "items": {"type": "string"}}
EPOCH = {"type": "integer"}


def tool(name: str, description: str, **properties: dict[str, Any]) -> Tool:
    schema = {"type": "object", "properties": properties, "required": [p for p in properties if p != "kind"]}
    return Tool(name, description, schema)


METRICS = tool("metrics", "Symbolic work and depth of a row.", label={"type": "string"})
MOVES = (
    tool("list_moves", "Legal moves now, optionally of one kind.", kind={"type": "string", "enum": list(MOVE_SHAPES)}),
    tool("apply_move", "Apply a fusion, fission or interchange.", kind={"type": "string"}, labels=LABELS, epoch=EPOCH),
    METRICS,
)
SCOPES = (
    tool("define_scope", "Make one kernel of maps of one state or of a run of blocks.", labels=LABELS, epoch=EPOCH),
)
PLACE = (
    tool(
        "place",
        "Run kernels on devices.",
        devices={"type": "object", "additionalProperties": {"enum": ["cpu", "gpu"]}},
        epoch=EPOCH,
    ),
)
KERNELS = (
    tool(
        "set_kernel_source",
        "Replace a kernel's source; same extern C entry. It is built and checked against the reference.",
        kernel={"type": "string"},
        source={"type": "string"},
        language={"type": "string", "enum": ["cpp", "cuda"]},
    ),
    METRICS,
)


def outcome(result: object) -> str:
    if isinstance(result, MoveResult):
        return f"{result.status}: {result.reason}"
    if isinstance(result, dict):
        return f"{result['status']}: {result.get('reason') or result.get('time_us') or result.get('kernels')}"
    if isinstance(result, list):
        return "\n".join(f"{move['kind']} {move['labels']}" for move in result) or "none"
    return str(result)


def act(session: Session, call: ToolCall) -> str:
    """Run one requested tool; a bad request comes back as text for the next turn instead of ending the run."""
    a = call.arguments
    handlers: dict[str, Callable[[], object]] = {
        "list_moves": lambda: session.list_moves(a.get("kind")),
        "apply_move": lambda: session.apply_move(a["kind"], a["labels"], a["epoch"]),
        "metrics": lambda: session.metrics(a["label"]),
        "define_scope": lambda: session.define_scope(a["labels"], a["epoch"]),
        "place": lambda: session.place(a["devices"], a["epoch"]),
        "set_kernel_source": lambda: session.set_kernel_source(a["kernel"], a["source"], a["language"]),
    }
    if call.name not in handlers:
        return f"error: no tool {call.name}"
    try:
        return outcome(handlers[call.name]())
    except Exception as err:  # the model's arguments are untrusted; its next turn sees the error
        return f"error: {type(err).__name__}: {err}"


def kernels_view(session: Session) -> str:
    lines = [
        f"{k['name']} on {k['device']} parallel={k['parallel']} depends={k['depends']}" for k in session.list_kernels()
    ]
    return "\n".join(lines)


def sources_view(session: Session) -> str:
    return "\n\n".join(
        f"// {k['name']} ({k['device']})\n{session.kernel_source(k['name'])}" for k in session.list_kernels()
    )


def default_moves(session: Session, applied: bool) -> None:
    # hand-chosen moves keep their granularity; untouched, the program gets canonicalization's own fusion
    if applied:
        session.finish_moves()
    else:
        session.default_moves()


def default_placement(session: Session, applied: bool) -> None:
    if not applied:
        session.place()


@dataclass(frozen=True, slots=True)
class Stage:
    """What the agent sees and may call in one stage, and the default that completes it (told whether any call
    applied)."""

    name: str
    goal: str
    tools: tuple[Tool, ...]
    view: Callable[[Session], str]
    finish: Callable[[Session, bool], object]


STAGES = (
    Stage("moves", "Shape loop nests with fusion, fission and interchange.", MOVES, Session.describe, default_moves),
    Stage(
        "scopes",
        "Group maps or blocks into kernels; each parallel map left becomes its own kernel.",
        SCOPES,
        Session.describe,
        lambda session, applied: session.define_scopes(),
    ),
    Stage("placement", "Choose each kernel's device; transfers cost time.", PLACE, kernels_view, default_placement),
    Stage(
        "kernels",
        "Rewrite slow kernels; keep results within the reference's tolerance.",
        KERNELS,
        sources_view,
        lambda session, applied: None,
    ),
)


def prompt(task: str, stage: Stage, session: Session, report: str, log: list[str], last: list[str]) -> str:
    return "\n\n".join(
        [
            f"Task: {task}",
            f"Stage {stage.name}: {stage.goal}\nEpoch: {session.epoch}",
            f"Program:\n{stage.view(session)}",
            f"Latest feedback:\n{report or 'none yet'}",
            "Actions so far:\n" + ("\n".join(log) or "none"),
            "Results of your last calls:\n" + ("\n".join(last) or "none"),
        ]
    )


def summary(call: ToolCall) -> str:
    args = {k: f"<{len(str(v).splitlines())} lines>" if k == "source" else v for k, v in call.arguments.items()}
    return f"{call.name}({', '.join(f'{k}={v}' for k, v in args.items())})"


def analyzed(chat: Chat, report: str, enabled: bool) -> str:
    return (chat(ANALYST, report, []).text or report) if enabled else report


def run(session: Session, chat: Chat, task: str, turns: int = TURNS, analyst: bool = False) -> list[str]:
    """Drive stages 2 to 5 of a canonicalized session, at most ``turns`` model calls per stage; returns the log.

    :param chat: One model call, e.g. ``functools.partial(llm.chat, client)``.
    :param analyst: Let one more model call rewrite each feedback report into guidance before the agent sees it.
    """
    log: list[str] = []
    report = ""
    for stage in STAGES:
        if stage.name == "kernels":
            for kernel in session.list_kernels():
                session.optimize_kernel(kernel["name"])
            report = analyzed(chat, session.feedback(), analyst)
        last: list[str] = []
        applied = False
        for _ in range(turns):
            reply = chat(SYSTEM, prompt(task, stage, session, report, log, last), list(stage.tools))
            if not reply.calls:
                break
            last = []
            for call in reply.calls:
                result = act(session, call)
                applied = applied or result.startswith(("applied", "ok"))
                log.append(f"[{stage.name}] {summary(call)} -> {result.splitlines()[0][:LOG_LINE] if result else ''}")
                last.append(f"{summary(call)}:\n{result}")
            if stage.name == "kernels" and any(c.name == "set_kernel_source" for c in reply.calls):
                report = analyzed(chat, session.feedback(), analyst)
        stage.finish(session, applied)
    return log
