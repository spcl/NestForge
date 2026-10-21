# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``POST /nestforge/<tool>``: the judge half of every NestForge tool. Each episode's kernel is served by its own
session process (:mod:`nestforge.server`), so the judge's threads never fork and a crash ends that process only.
A tool the session does not serve yet answers 501."""

from http import HTTPMethod, HTTPStatus
from typing import Any

import pydantic
from hpcagent_bench.cpf.cache import Target
from hpcagent_bench.harness.judge_wire import JudgeAnswer, JudgeError
from hpcagent_bench.harness.routes import EpisodeStore, Judge, RouteRequest, route
from hpcagent_bench.harness.routes.canonical_parallel_form import canonical_parallel_form_target
from hpcagent_bench_plugins.nestforge.wire import IDENTITY, ROUTE, TOOLS

from nestforge.server import CALLS, Reply, SessionProcess, Setup
from nestforge.stages.performance import Performance

#: Episode -> kernel name -> its session process.
EPISODES: EpisodeStore[dict[str, SessionProcess]] = EpisodeStore(lambda episode: {})


class ToolAnswer(JudgeAnswer):
    tool: str
    result: Any = None
    performance: Performance | None = None


@route(ROUTE, methods=(HTTPMethod.POST,))
def serve(judge: Judge, request: RouteRequest) -> None:
    name = request.parts[1] if len(request.parts) > 1 else ""
    entry = TOOLS.get(name)
    if entry is None:
        error = f"no NestForge tool {name!r}; tools: {sorted(TOOLS)}"
        return judge.reply(HTTPStatus.NOT_FOUND, JudgeError(error=error))
    if name not in CALLS:
        error = f"{name} ({entry.phase} phase) is not implemented yet; this run serves {sorted(CALLS)}"
        return judge.reply(HTTPStatus.NOT_IMPLEMENTED, JudgeError(error=error, cause="not-implemented"))
    try:
        args = entry.args.model_validate({k: v for k, v in request.body.items() if k not in IDENTITY})
    except pydantic.ValidationError as err:
        return judge.reply(HTTPStatus.BAD_REQUEST, JudgeError(error=f"{name}: {err}"))
    kernel = str(request.body.get("kernel") or "")
    with judge.device_slot() as slot:
        if slot is None:
            return None
        kernels = EPISODES.get(request.episode_id)
        if kernel not in kernels:
            gpu = canonical_parallel_form_target() is Target.GPU
            setup = Setup(kernel, judge.cfg.preset.text, judge.cfg.datatype, gpu)
            try:
                kernels[kernel] = SessionProcess(setup)
            except LookupError as err:
                return judge.reply(HTTPStatus.BAD_REQUEST, JudgeError(error=f"{name}: {err}"))
        reply = kernels[kernel].call(name, args.model_dump())
        if not kernels[kernel].alive:
            # The next call starts a fresh session from the canonical program; this one's moves are lost.
            del kernels[kernel]
            reply = Reply(error=f"{reply.error}; the next call starts a fresh session")
    if reply.error:
        return judge.reply(HTTPStatus.BAD_REQUEST, JudgeError(error=f"{name}: {reply.error}"))
    return judge.reply(HTTPStatus.OK, ToolAnswer(tool=name, result=reply.result, performance=reply.performance))
