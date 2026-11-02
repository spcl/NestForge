# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``POST /nestforge/<tool>``: the judge half of every NestForge tool. Each episode's kernel is served by its own
session process (:mod:`nestforge.server`), so the judge's threads never fork and a crash ends that process only.
A tool in the wire table that the session does not serve answers 501. A session that cannot start, crashes or stops
answering is the judge's fault (500, :data:`JUDGE_FAULT_CAUSE`); 400 is only for the agent's own call; 503 is a call
that ran out of its budget and goes on when repeated. Moves are journaled on disk before they are answered, so none
of these loses one, nor does a judge restart; a journal that cannot be read or repeated is a judge fault, never a
program without the agent's moves."""

import contextlib
import json
import threading
from collections.abc import Iterator
from http import HTTPMethod, HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pydantic
from hpcagent_bench.cpf.cache import Target
from hpcagent_bench.harness import sandbox
from hpcagent_bench.harness.judge_scheduler import DeviceSlot
from hpcagent_bench.harness.judge_wire import JudgeAnswer, JudgeError
from hpcagent_bench.harness.routes import EpisodeStore, Judge, RouteRequest, route
from hpcagent_bench.harness.routes.canonical_parallel_form import canonical_parallel_form_target
from hpcagent_bench.harness.service import JUDGE_FAULT_CAUSE
from hpcagent_bench_plugins.nestforge.wire import IDENTITY, ROUTE, TOOLS

from nestforge.server import CALLS, CHANGING, MEASURING, POLL_S, REPEATED, Reply, SessionProcess, Setup, applied
from nestforge.stages.performance import Performance

#: What a tool that needs no device holds instead of a slot.
NO_SLOT = contextlib.nullcontext(object())

#: Seconds an unused episode keeps its processes and data; its journal stays. Above any one call's time (client
#: timeout, or budget and stop grace), or a running call's episode could get a second session beside it.
SESSION_IDLE_S = 3600.0


class Episode:
    """One episode's session processes by kernel name. A session answers one call at a time over its pipe, so
    ``lock`` serializes the episode's calls (an agent may send several at once), and ``measuring`` lets one of its
    measuring calls queue for a device slot. ``orphans`` keeps, per kernel, a finished change whose client had left, so
    that the client's retry reads the answer instead of repeating the change."""

    __slots__ = ("lock", "measuring", "orphans", "sessions")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.measuring = threading.Lock()
        self.sessions: dict[str, SessionProcess] = {}
        self.orphans: dict[str, tuple[Call, Reply]] = {}


#: A tool and its validated arguments.
Call = tuple[str, dict[str, Any]]

EPISODES: EpisodeStore[Episode] = EpisodeStore(lambda episode: Episode(), idle_seconds=SESSION_IDLE_S)


def episode_of(episode_id: str) -> Episode:
    """The state of ``episode_id``'s attempt: a relaunched attempt has a new folder and starts anew. An empty id stays
    empty, for the store to refuse."""
    return EPISODES.get(episode_id and f"{episode_id}@{sandbox.agent_root(episode_id)}")


def journal_file(episode_id: str, kernel: str) -> Path:
    """Where the journal of ``kernel`` lives: beside its library. The folder may be shared, so the name says whose."""
    return sandbox.agent_root(episode_id) / "nestforge" / f"journal-{quote(f'{episode_id}@{kernel}', safe='')}.jsonl"


def journal_of(episode_id: str, kernel: str) -> list[Call]:
    """The changes that made ``kernel``'s program, in order; none before the first.

    :raises LookupError: There is a journal and it cannot be read."""
    file = journal_file(episode_id, kernel)
    try:
        return [(tool, arguments) for tool, arguments in map(json.loads, file.read_text("utf-8").splitlines())]
    except FileNotFoundError:
        return []
    except (OSError, TypeError, ValueError) as err:
        raise LookupError(f"the journal {file} cannot be read: {err}") from err


def record(episode_id: str, kernel: str, call: Call) -> None:
    """Add ``call`` to the journal, before its answer goes out."""
    file = journal_file(episode_id, kernel)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("a", encoding="utf-8") as journal:
        journal.write(json.dumps(call) + "\n")


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


def holds_device(episode: Episode, kernel: str, name: str, episode_id: str) -> bool:
    """Whether the call ``name`` builds or runs code: a measuring tool, or a call a rebuild has to build for."""
    if name in MEASURING:
        return True
    session = episode.sessions.get(kernel)
    if session is not None and session.alive:
        return False
    with contextlib.suppress(LookupError):  # session_of answers it
        return any(tool in MEASURING for tool, _ in journal_of(episode_id, kernel))
    return False


@contextlib.contextmanager
def device_slot(
    judge: Judge, episode: Episode, gone: threading.Event, terminal: bool
) -> Iterator[DeviceSlot | object | None]:
    """A device slot, ``None`` if the client left; the episode's other measuring calls wait here, holding none."""
    if not locked(episode.measuring, gone):
        yield None
        return
    try:
        with judge.device_slot(terminal) as slot:
            yield slot
    finally:
        episode.measuring.release()


def session_of(
    episode: Episode, judge: Judge, kernel: str, episode_id: str, gone: threading.Event, slot: int | None
) -> SessionProcess:
    """The kernel's session; a new one repeats the episode's changes, so it is the program the agent left. A repeat
    measures nothing, and stops as the call that caused it does.

    :raises LookupError: The session could not start, or could not repeat a change."""
    session = episode.sessions.get(kernel)
    if session is not None and session.alive:
        return session
    episode.sessions.pop(kernel, None)  # its finalizer ends what a dead session left running
    journal = journal_of(episode_id, kernel)
    gpu = canonical_parallel_form_target() is Target.GPU
    setup = Setup(kernel, judge.cfg.preset.text, judge.cfg.datatype, gpu, str(sandbox.agent_root(episode_id)))
    session = SessionProcess(setup)
    for tool, arguments in journal:
        repeated = session.call(tool, arguments, gone, slot=slot, timed=False)
        if not applied(repeated):
            session.close()
            raise LookupError(f"repeating {tool} {arguments} failed: {repeated.error or repeated.result}")
    episode.sessions[kernel] = session
    return session


def run_call(
    episode: Episode,
    judge: Judge,
    request: RouteRequest,
    kernel: str,
    call: Call,
    gone: threading.Event,
    slot: int | None = None,
) -> Reply:
    """``call`` on the kernel's session; a change whose client left keeps its answer for the retry.

    :raises LookupError: The session could not start or rebuild its program."""
    name, arguments = call
    if name in CHANGING:
        orphan = episode.orphans.pop(kernel, None)
        if orphan is not None and orphan[0] == call:
            return orphan[1]
    session = session_of(episode, judge, kernel, request.episode_id, gone, slot)
    reply = session.call(name, arguments, gone, slot=slot)
    if not session.alive:
        # The call's change is not journaled, so the next session is the program before it.
        del episode.sessions[kernel]
        return reply.model_copy(update={"ended": True})
    if name in CHANGING and not reply.error:
        if name in REPEATED and applied(reply):
            try:
                record(request.episode_id, kernel, call)
            except OSError as err:
                # Unrecorded, the change would vanish with this session: it goes now, and the agent is told.
                session.close()
                del episode.sessions[kernel]
                return Reply(error=f"the change could not be journaled: {err}", ended=True)
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
    episode = episode_of(request.episode_id)
    # The slot first: a call waiting for a slot blocks nothing of its own episode. Only what builds or runs code takes
    # one; a read or a move answers while other agents measure.
    measuring = holds_device(episode, kernel, name, request.episode_id)
    terminal = bool(request.query.get("terminal"))
    with client_left(judge) as gone, device_slot(judge, episode, gone, terminal) if measuring else NO_SLOT as slot:
        if slot is None or not locked(episode.lock, gone):
            return None
        index = slot.index if isinstance(slot, DeviceSlot) else None
        try:
            reply = run_call(episode, judge, request, kernel, (name, args.model_dump()), gone, index)
        except LookupError as err:
            error = f"{name}: the session could not start: {err}"
            return judge.reply(HTTPStatus.INTERNAL_SERVER_ERROR, JudgeError(error=error, cause=JUDGE_FAULT_CAUSE))
        finally:
            episode.lock.release()
        return answer(judge, name, reply)
