# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``POST /nestforge/<tool>``: the judge half of every NestForge tool. Each episode's kernel is served by its own
session process (:mod:`nestforge.server`), so the judge's threads never fork and a crash ends that process only.
A tool in the wire table that the session does not serve answers 501. A session that cannot start is the judge's fault
(500, :data:`JUDGE_FAULT_CAUSE`); 400 is the agent's own call, one that crashed its session included; 503 is a call
that ran out of its budget and goes on when repeated. Moves are journaled on disk before they are answered, so none
of these loses one, nor does a judge restart; a journal that cannot be read or repeated is a judge fault, never a
program without the agent's moves."""

import contextlib
import json
import sys
import threading
from collections.abc import Iterator
from http import HTTPMethod, HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pydantic
from hpcagent_bench.plugin_api import (
    JUDGE_FAULT_CAUSE,
    DeviceSlot,
    EpisodeStore,
    Judge,
    JudgeAnswer,
    JudgeError,
    RouteRequest,
    Target,
    agent_root,
    canonical_parallel_form_target,
    keep_record,
    route,
)
from hpcagent_bench_plugins.nestforge.wire import IDENTITY, ROUTE, TOOLS

from nestforge.plugin_record import PLUGIN, SCHEMA, episode_record, journal_entry, keep_capped
from nestforge.server import CALLS, CHANGING, MEASURING, POLL_S, Reply, SessionProcess, Setup, applied, move_text
from nestforge.session import Restored, Sweep, TuneResult
from nestforge.stages.families import checked_families, family_of
from nestforge.stages.moves import ROLLBACK, size_argument
from nestforge.stages.performance import Performance
from nestforge.stages.phases import checked_phases
from nestforge.stages.requests import RequestResult

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

    __slots__ = ("families", "lock", "measuring", "orphans", "phases", "sessions")

    def __init__(self) -> None:
        #: The move families the episode switches off, fixed by its first call; ``None`` before it.
        self.families: tuple[str, ...] | None = None
        #: The phases the episode's agent runs, fixed by its first call; ``None`` before it.
        self.phases: tuple[str, ...] | None = None
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
    return EPISODES.get(episode_id and f"{episode_id}@{agent_root(episode_id)}")


def journal_file(episode_id: str, kernel: str, log: str = "journal") -> Path:
    """Where the journal of ``kernel`` lives (or its ``log``, ``calls``): beside its library. The folder may be shared,
    so the name says whose."""
    return agent_root(episode_id) / "nestforge" / f"{log}-{quote(f'{episode_id}@{kernel}', safe='')}.jsonl"


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


def call_entry(name: str, arguments: dict[str, Any], reply: Reply, tokens: object) -> dict[str, Any]:
    """One call as the plugin record counts it, without its arguments: its outcome, the refusal's class, whether
    the agent forced a move past the compiler's check, and the agent's cumulative tokens when it made the call (the
    bench sends them with every call)."""
    result = reply.result
    dumped = result.model_dump(mode="json") if isinstance(result, pydantic.BaseModel) else result
    outcome = dumped if isinstance(dumped, dict) and "status" in dumped else {}
    if reply.error:
        status = "failed"
    elif outcome:
        status = str(outcome["status"])
    elif name == "get_performance":
        status = "restored" if isinstance(result, Restored) and result.rolled_back else "ok"
    else:
        status = "applied" if name in CHANGING else "ok"
    return {
        "tool": name,
        "kind": arguments.get("kind"),
        "family": family_of(str(arguments["kind"])) if arguments.get("kind") else None,
        "status": status,
        "refusal": outcome.get("refusal") or "",
        "forced": bool(outcome.get("forced", False)),
        "phase": TOOLS[name].label,
        "tokens": tokens if isinstance(tokens, int) else None,
    }


def log_call(episode_id: str, kernel: str, entry: dict[str, Any]) -> None:
    """Append ``entry`` to the episode's call log with its index. A log that cannot be written costs the record a
    line, never the agent its call."""
    file = journal_file(episode_id, kernel, "calls")
    try:
        file.parent.mkdir(parents=True, exist_ok=True)
        index = len(file.read_text("utf-8").splitlines()) if file.exists() else 0
        with file.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"index": index, **entry}) + "\n")
    except OSError as err:
        print(f"nestforge: call {entry['tool']} not logged: {err}", file=sys.stderr)


def keep_episode_record(episode_id: str, kernel: str, final: object, families: tuple[str, ...]) -> None:
    """The plugin record of the episode so far (``docs/plugin-record.md``), written at every finalize, since a graded
    submission always follows one; a later write replaces it."""
    file = journal_file(episode_id, kernel, "calls")
    try:
        calls = [json.loads(line) for line in file.read_text("utf-8").splitlines()] if file.exists() else []
        dumped = final.model_dump(mode="json") if isinstance(final, pydantic.BaseModel) else final
        kernels = dumped.get("kernels") if isinstance(dumped, dict) else None
        journal = [journal_entry(*call) for call in journal_of(episode_id, kernel)]
        data = episode_record(
            kernel, calls, journal, len(kernels) if isinstance(kernels, dict) else 0, families=families
        )
        keep_capped(lambda record: keep_record(agent_root(episode_id), PLUGIN, SCHEMA, record), data)
    except (LookupError, OSError, ValueError) as err:
        print(f"nestforge: the plugin record of {episode_id} was not written: {err}", file=sys.stderr)


def fix_families(episode: Episode, sent: object) -> str | None:
    """Fix the episode's switched-off move families at its first call; why a call is refused, or ``None``. A call
    that names other families than the first did is refused: a run switches families off for its whole length."""
    names = sent if isinstance(sent, list) else []
    try:
        families = tuple(sorted(checked_families(str(name) for name in names)))
    except ValueError as err:
        return str(err)
    if episode.families is None:
        episode.families = families
    if episode.families != families:
        return f"this episode switched off {list(episode.families)} at its first call; a call cannot change that"
    return None


def fix_phases(episode: Episode, sent: object) -> str | None:
    """Fix the phases the episode's agent runs at its first call; why a call is refused, or ``None``. A call naming
    other phases than the first did is refused: a packet fixes them for the whole run."""
    names = sent if isinstance(sent, list) else []
    try:
        phases = tuple(p.value for p in checked_phases(str(name) for name in names))
    except ValueError as err:
        return str(err)
    if episode.phases is None:
        episode.phases = phases
    if episode.phases != phases:
        return f"this episode's agent runs the phases {list(episode.phases)}, fixed at its first call"
    return None


def record(episode_id: str, kernel: str, call: Call) -> None:
    """Add ``call`` to the journal, before its answer goes out."""
    file = journal_file(episode_id, kernel)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("a", encoding="utf-8") as journal:
        journal.write(json.dumps(call) + "\n")


def record_request(episode_id: str, kernel: str, arguments: dict[str, Any]) -> None:
    """Keep a recorded transformation request whole, examples included, beside the journal: the authoring step between
    rounds reads it there. A request changes nothing in the program, so the journal never repeats it."""
    file = journal_file(episode_id, kernel, "requests")
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("a", encoding="utf-8") as requests:
        requests.write(json.dumps(arguments) + "\n")


class ToolAnswer(JudgeAnswer):
    tool: str
    result: Any = None
    performance: Performance | None = None


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


def repeats_of(call: Call, reply: Reply) -> list[Call]:
    """How a new session repeats an applied change: a written kernel passed its check and a sweep timed its winner, so
    the repeat builds the source, or that winner, without checking or timing again; a sweep without one changed
    nothing. A tune_move repeats as the move at the size it kept, or not at all; a get_performance that restored the
    fastest version as its rollbacks, or not at all."""
    name, arguments = call
    if name == "write_kernel":
        return [(name, {**arguments, "checked": False})]
    if name == "tune_kernel":
        assert isinstance(reply.result, Sweep), "a tune_kernel answers its sweep"
        return [] if reply.result.winner is None else [(name, {**arguments, "variant": reply.result.winner})]
    if name == "tune_move":
        assert isinstance(reply.result, TuneResult), "a tune_move answers what it measured"
        kept = [value.sizes for value in reply.result.values if value.status == "kept"]
        if not kept:
            return []
        move = {
            "kind": arguments["kind"],
            "labels": arguments["labels"],
            size_argument(arguments["kind"]): list(kept[0]),
        }
        return [("apply_move", move if arguments.get("scope") is None else {**move, "scope": arguments["scope"]})]
    if name == "get_performance":
        restored = reply.result.rolled_back if isinstance(reply.result, Restored) else 0
        return [("apply_move", {"kind": ROLLBACK, "labels": []})] * restored
    return [call]


def session_of(
    episode: Episode, judge: Judge, kernel: str, episode_id: str, gone: threading.Event, slot: DeviceSlot | None
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
    root, families, phases = str(agent_root(episode_id)), episode.families or (), episode.phases or ()
    setup = Setup(
        kernel, judge.cfg.preset.text, judge.cfg.datatype, gpu, root, disabled_families=families, agent_phases=phases
    )
    session = SessionProcess(setup)
    for tool, arguments in journal:
        repeated = session.call(tool, arguments, gone, slot=slot, timed=False)
        if not applied(repeated):
            session.close()
            raise LookupError(f"repeating {tool} {arguments} failed: {repeated.error or repeated.result}")
    episode.sessions[kernel] = session
    return session


def crashed(call: Call, reply: Reply) -> Reply:
    """The answer to ``call``, whose session process died answering ``reply``: the agent's error, not the judge's."""
    name, arguments = call
    during = f"{name} ({move_text(arguments)})" if name == "apply_move" else name
    then = (
        "that change was undone and the session restarts from the last good program; try another move"
        if name in CHANGING
        else "the program is unchanged and the session restarts from it"
    )
    return Reply(error=f"{reply.error} during {during}; {then}")


def run_call(
    episode: Episode,
    judge: Judge,
    request: RouteRequest,
    kernel: str,
    call: Call,
    gone: threading.Event,
    slot: DeviceSlot | None = None,
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
        return crashed(call, reply)
    if (
        name == "request_transformation"
        and isinstance(reply.result, RequestResult)
        and reply.result.status == "recorded"
    ):
        record_request(request.episode_id, kernel, arguments)
    if name in CHANGING and not reply.error:
        for repeat in repeats_of(call, reply) if applied(reply) else []:
            try:
                record(request.episode_id, kernel, repeat)
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
        error = f"{name} ({entry.label} phase) is not implemented yet; this run serves {sorted(CALLS)}"
        return judge.reply(HTTPStatus.NOT_IMPLEMENTED, JudgeError(error=error, cause="not-implemented"))
    kernel = str(request.body.get("kernel") or "")
    tokens = request.body.get("tokens")
    try:
        args = entry.args.model_validate({k: v for k, v in request.body.items() if k not in IDENTITY})
    except pydantic.ValidationError as err:
        log_call(request.episode_id, kernel, call_entry(name, dict(request.body), Reply(error=str(err)), tokens))
        return judge.reply(HTTPStatus.BAD_REQUEST, JudgeError(error=f"{name}: {err}"))
    episode = episode_of(request.episode_id)
    refused = fix_families(episode, request.body.get("disabled_families")) or fix_phases(
        episode, request.body.get("agent_phases")
    )
    if refused is not None:
        log_call(request.episode_id, kernel, call_entry(name, args.model_dump(), Reply(error=refused), tokens))
        return judge.reply(HTTPStatus.BAD_REQUEST, JudgeError(error=f"{name}: {refused}"))
    # The slot first: a call waiting for a slot blocks nothing of its own episode. Only what builds or runs code takes
    # one; a read or a move answers while other agents measure.
    measuring = holds_device(episode, kernel, name, request.episode_id)
    terminal = bool(request.query.get("terminal"))
    with judge.client_left() as gone, device_slot(judge, episode, gone, terminal) if measuring else NO_SLOT as slot:
        if slot is None or not locked(episode.lock, gone):
            return None
        held = slot if isinstance(slot, DeviceSlot) else None
        try:
            reply = run_call(episode, judge, request, kernel, (name, args.model_dump()), gone, held)
            # under the episode's lock, so the log's indices follow the order the calls ran in
            log_call(request.episode_id, kernel, call_entry(name, args.model_dump(), reply, tokens))
            if name == "finalize" and not reply.error:
                keep_episode_record(request.episode_id, kernel, reply.result, episode.families or ())
        except LookupError as err:
            error = f"{name}: the session could not start: {err}"
            return judge.reply(HTTPStatus.INTERNAL_SERVER_ERROR, JudgeError(error=error, cause=JUDGE_FAULT_CAUSE))
        finally:
            episode.lock.release()
        return answer(judge, name, reply)
