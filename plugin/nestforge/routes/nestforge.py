# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``POST /nestforge/<tool>``: the judge half of every NestForge tool, one :class:`Session` per episode and kernel.
Only the structured phase and ``get_performance`` are served; every other tool answers 501 until it is enabled."""

from collections.abc import Callable
from http import HTTPMethod, HTTPStatus
from typing import Any

import numpy as np
import pydantic
from hpcagent_bench.categories import Preset
from hpcagent_bench.cpf_cache import Target
from hpcagent_bench.frameworks.benchmark import Benchmark
from hpcagent_bench.harness.judge_wire import JudgeAnswer, JudgeError
from hpcagent_bench.harness.routes import EpisodeStore, Judge, RouteRequest, route
from hpcagent_bench.harness.routes.canonical_parallel_form import canonical_parallel_form_target
from hpcagent_bench_plugins.nestforge.wire import IDENTITY, ROUTE, TOOLS, RunDefaults, ViewTree

from nestforge.corpus.bench import corpus_kernel, preset_sizes
from nestforge.session import Session
from nestforge.stages.canonicalize import Targets
from nestforge.stages.performance import Performance


class Kernel:
    """One kernel's session and the bench inputs its whole program is timed on."""

    __slots__ = ("inputs", "session")

    def __init__(self, name: str, preset: Preset) -> None:
        kernel = corpus_kernel(name)
        if kernel is None:
            raise LookupError(f"{name!r} has no DaCe program NestForge can start from")
        sizes = preset_sizes(kernel, preset.text)
        targets = Targets(gpu=canonical_parallel_form_target() is Target.GPU)
        self.session = Session(kernel.to_sdfg(), targets=targets, sizes=sizes)
        self.session.canonicalize()
        data = Benchmark(kernel.short_name).get_data(preset=preset)
        self.inputs = {key: value for key, value in data.items() if isinstance(value, np.ndarray)}


#: Episode -> kernel name -> its state.
EPISODES: EpisodeStore[dict[str, Kernel]] = EpisodeStore(lambda episode: {})


class ToolAnswer(JudgeAnswer):
    tool: str
    result: Any = None
    performance: Performance | None = None


def view_tree(session: Session, args: ViewTree) -> str:
    notes = [session.collapse(args.collapse)] if args.collapse else []
    if args.expand:
        notes.append(session.expand(args.expand, args.epoch if args.epoch is not None else -1))
    return "\n".join([*notes, session.describe()])


def run_defaults(session: Session, args: RunDefaults) -> object:
    match args.stage:
        case "moves":
            return session.default_moves()
        case "finish":
            return session.finish_moves()
        case "scopes":
            return session.define_scopes()
        case "placement":
            return session.place()


#: The served tools: name -> (handler, whether its answer carries performance). A tool left out answers 501.
HANDLERS: dict[str, tuple[Callable[[Session, Any], object], bool]] = {
    "get_cpf": (lambda session, args: session.describe(), False),
    "view_tree": (view_tree, False),
    "list_moves": (lambda session, args: session.list_moves(args.kind), False),
    "apply_move": (lambda session, args: session.apply_move(args.kind, args.labels, args.epoch), True),
    "run_defaults": (run_defaults, True),
    "set_scope": (lambda session, args: session.define_scope(args.labels, args.epoch), False),
    "get_performance": (lambda session, args: None, True),
}


@route(ROUTE, methods=(HTTPMethod.POST,))
def serve(judge: Judge, request: RouteRequest) -> None:
    name = request.parts[1] if len(request.parts) > 1 else ""
    entry = TOOLS.get(name)
    if entry is None:
        return judge.reply(
            HTTPStatus.NOT_FOUND, JudgeError(error=f"no NestForge tool {name!r}; tools: {sorted(TOOLS)}")
        )
    if name not in HANDLERS:
        error = f"{name} ({entry.phase} phase) is not implemented yet; this run serves {sorted(HANDLERS)}"
        return judge.reply(HTTPStatus.NOT_IMPLEMENTED, JudgeError(error=error, cause="not-implemented"))
    try:
        args = entry.args.model_validate({k: v for k, v in request.body.items() if k not in IDENTITY})
    except pydantic.ValidationError as err:
        return judge.reply(HTTPStatus.BAD_REQUEST, JudgeError(error=f"{name}: {err}"))
    kernel = str(request.body.get("kernel") or "")
    with judge.device_slot() as slot:
        if slot is None:
            return None
        try:
            kernels = EPISODES.get(request.episode_id)
            if kernel not in kernels:
                kernels[kernel] = Kernel(kernel, judge.cfg.preset)
            state = kernels[kernel]
            handler, timed = HANDLERS[name]
            result = handler(state.session, args)
            performance = state.session.performance(state.inputs) if timed else None
        except (LookupError, ValueError) as err:
            return judge.reply(HTTPStatus.BAD_REQUEST, JudgeError(error=f"{name}: {err}"))
    return judge.reply(HTTPStatus.OK, ToolAnswer(tool=name, result=result, performance=performance))
