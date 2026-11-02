# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``POST /nestforge/<tool>``: the judge half of every NestForge tool. Each episode's kernel is served by its own
session process (:mod:`nestforge.server`), so the judge's threads never fork and a crash ends that process only.
A tool in the wire table that the session does not serve answers 501. A session that cannot start, crashes or stops
answering is the judge's fault (500, :data:`JUDGE_FAULT_CAUSE`); 400 is only for the agent's own call; 503 is a call
that ran out of its budget and goes on when repeated. Moves are journaled here, so none of these loses one."""

import contextlib
import threading
from collections.abc import Iterator
from http import HTTPMethod, HTTPStatus
from typing import Any

import pydantic
from hpcagent_bench.cpf.cache import Target
from hpcagent_bench.harness import sandbox
from hpcagent_bench.harness.judge_wire import JudgeAnswer, JudgeError
from hpcagent_bench.harness.routes import EpisodeStore, Judge, RouteRequest, route
from hpcagent_bench.harness.routes.canonical_parallel_form import canonical_parallel_form_target
from hpcagent_bench.harness.service import JUDGE_FAULT_CAUSE
from hpcagent_bench_plugins.nestforge.wire import IDENTITY, ROUTE, TOOLS

from nestforge.server import CALLS, CHANGING, MEASURING, POLL_S, Reply, SessionProcess, Setup
from nestforge.stages.performance import Performance

#: What a tool that needs no device holds instead of a slot.
NO_SLOT = contextlib.nullcontext(object())


class Episode:
    """One episode's session processes by kernel name. A session answers one call at a time over its pipe, so
    ``lock`` serializes the episode's calls (an agent may send several at once). ``journals`` is what made each
    program: a session that dies is rebuilt from it, so its moves survive. ``orphans`` keeps, per kernel, a finished
    change whose client had left, so that the client's retry reads the answer instead of repeating the change."""

    __slots__ = ("journals", "lock", "orphans", "sessions")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.sessions: dict[str, SessionProcess] = {}
        self.journals: dict[str, list[Call]] = {}
        self.orphans: dict[str, tuple[Call, Reply]] = {}


#: A tool and its validated arguments.
Call = tuple[str, dict[str, Any]]

EPISODES: EpisodeStore[Episode] = EpisodeStore(lambda episode: Episode())


class ToolAnswer(JudgeAnswer):
    tool: str
    result: Any = None
    performance: Performance | None = None


@contextlib.contextmanager
def client_left(judge: Judge) -> Iterator[threading.Event]:
    """An event set when this request's client leaves. The bench watches only its own graded routes, so this one
    watches for itself; a default phase's call comes from ``next_phase``, not from a client of its own."""
    from hpcagent_bench.harness.service import JudgeHandler  # service imports this module while it loads

    if not isinstance(judge, JudgeHandler):
        yield threading.Event()
        return
    finished = threading.Event()
    watcher = threading.Thread(target=judge.watch_client, args=(finished,), daemon=True)
    watcher.start()
    try:
        yield judge.gone
    finally:
        finished.set()
        watcher.join()


def locked(lock: threading.Lock, gone: threading.Event) -> bool:
    """Take ``lock``, unless the client leaves first."""
    while not lock.acquire(timeout=POLL_S):
        if gone.is_set():
            return False
    return True


def session_of(episode: Episode, judge: Judge, kernel: str, episode_id: str) -> SessionProcess:
    """The kernel's session; a new one repeats the episode's changes, so it is the program the agent left.

    :raises LookupError: The session could not start, or could not repeat a change."""
    session = episode.sessions.get(kernel)
    if session is not None:
        return session
    gpu = canonical_parallel_form_target() is Target.GPU
    setup = Setup(kernel, judge.cfg.preset.text, judge.cfg.datatype, gpu, str(sandbox.agent_root(episode_id)))
    session = SessionProcess(setup)
    for tool, arguments in episode.journals.get(kernel, ()):
        repeated = session.call(tool, arguments)
        if repeated.error:
            session.close()
            raise LookupError(f"repeating {tool} {arguments} failed: {repeated.error}")
    episode.sessions[kernel] = session
    return session


def run_call(
    episode: Episode, judge: Judge, request: RouteRequest, kernel: str, call: Call, gone: threading.Event
) -> Reply:
    """``call`` on the kernel's session; a change whose client left keeps its answer for the retry.

    :raises LookupError: The session could not start."""
    name, arguments = call
    if name in CHANGING:
        orphan = episode.orphans.pop(kernel, None)
        if orphan is not None and orphan[0] == call:
            return orphan[1]
    session = session_of(episode, judge, kernel, request.episode_id)
    reply = session.call(name, arguments, gone)
    if not session.alive:
        # The call's change is not journaled, so the next session is the program before it.
        del episode.sessions[kernel]
        return reply.model_copy(update={"ended": True})
    if name in CHANGING and not reply.error:
        episode.journals.setdefault(kernel, []).append(call)
        if gone.is_set():
            episode.orphans[kernel] = (call, reply)
    return reply


def answer(judge: Judge, name: str, reply: Reply) -> None:
    if reply.ended:
        error = f"{name}: {reply.error}; your moves are kept: the next call repeats them in a fresh session"
        return judge.reply(HTTPStatus.INTERNAL_SERVER_ERROR, JudgeError(error=error, cause=JUDGE_FAULT_CAUSE))
    if reply.stopped:
        return judge.reply(HTTPStatus.SERVICE_UNAVAILABLE, JudgeError(error=reply.error))
    if reply.error:
        return judge.reply(HTTPStatus.BAD_REQUEST, JudgeError(error=f"{name}: {reply.error}"))
    return judge.reply(HTTPStatus.OK, ToolAnswer(tool=name, result=reply.result, performance=reply.performance))


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
    episode = EPISODES.get(request.episode_id)
    # The slot first: a call waiting for a slot blocks nothing of its own episode. Only what builds or runs code takes
    # one; a read or a move answers while other agents measure.
    with client_left(judge) as gone, judge.device_slot() if name in MEASURING else NO_SLOT as slot:
        if slot is None or not locked(episode.lock, gone):
            return None
        try:
            reply = run_call(episode, judge, request, kernel, (name, args.model_dump()), gone)
        except LookupError as err:
            error = f"{name}: the session could not start: {err}"
            return judge.reply(HTTPStatus.INTERNAL_SERVER_ERROR, JudgeError(error=error, cause=JUDGE_FAULT_CAUSE))
        finally:
            episode.lock.release()
        return answer(judge, name, reply)
