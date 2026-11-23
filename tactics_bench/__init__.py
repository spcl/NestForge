# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The HPCAgent-Bench columns of NestForge. ``nestforge_reference`` is the column an agent's run is compared against. No
LLM: the kernel's reference solution (``reference_solutions/<kernel>.py``) picks the moves, NestForge's default moves
when the kernel has none, and scopes, placement and code generation run on the same defaults the agent's session uses.
``nestforge_stub`` and ``nestforge_two_stage`` stand in for an LLM agent in CI, through the tools an agent calls.

A package of its own beside ``nestforge``: it imports HPCAgent-Bench, which no ``nestforge`` module may."""

import hashlib
import pathlib
import random
import re
import shutil
import sys
import tempfile
import weakref
from collections.abc import Sequence
from typing import Any, ClassVar

from hpcagent_agent.tools import http_json
from hpcagent_bench import config
from hpcagent_bench.flags import ncores
from hpcagent_bench.harness.envelope import Submission
from hpcagent_bench.harness.optimizers import FlavorOptimizer
from hpcagent_bench.harness.sandbox import shared_dir
from hpcagent_bench.harness.task import Task
from hpcagent_bench.spec import resolve_preset

import reference_solutions
from tactics.plugin_record import SCHEMA, episode_record, keep_capped
from tactics.replay import Context, Script, Solution, baseline, load_solution
from tactics.records import KernelStatus, MoveStatus, spelled
from tactics.server import Setup, start
from tactics.stages.families import ALL, checked_families
from tactics.session import DEFAULTS_PREFIX, MoveResult, kernel_language

__all__ = [
    "DefaultOptimizer",
    "NoWaveOptimizer",
    "ReferenceOptimizer",
    "StubAgent",
    "TwoStageStub",
    "reference_solution",
    "solution_paths",
]

#: Where the reference solutions live, one ``<kernel>.py`` each: the installed package, wherever the install put it.
SOLUTIONS = pathlib.Path(reference_solutions.__file__).parent


def solution_paths() -> list[pathlib.Path]:
    """Every reference solution, in name order; the package's own ``__init__.py`` is none."""
    return sorted(path for path in SOLUTIONS.glob("*.py") if path.name != "__init__.py")


def reference_solution(kernel: str) -> Solution | None:
    """The reference solution of ``kernel``, or ``None`` when it has none."""
    path = SOLUTIONS / f"{pathlib.PurePath(kernel).name}.py"
    return load_solution(path) if path.is_file() else None


def applied_journal(applied: Sequence[MoveResult]) -> list[list[object]]:
    """The changes a column made, as the plugin record's journal holds an agent's tool calls."""
    return [
        ["run_defaults", {"stage": move.kind.removeprefix(DEFAULTS_PREFIX)}]
        if move.kind.startswith(DEFAULTS_PREFIX)
        else ["apply_move", {"kind": move.kind, "labels": list(move.labels)}]
        for move in applied
    ]


class ReferenceOptimizer(FlavorOptimizer):
    """Replays the kernel's reference moves (default moves without one), then finalizes on NestForge's defaults. Its
    plugin record says which it was."""

    __slots__ = ()
    #: Whether the column reads the kernel's reference solution; the default column does not.
    references: ClassVar[bool] = True
    #: The move families the column switches off (``tactics.stages.families``); none by default.
    families: ClassVar[tuple[str, ...]] = ()

    def tune(self, task: Task, pipeline: str) -> Submission:
        gpu = task.image != "cpu"
        service = config.current().service
        # the service's preset is a token ("XL+fuzz"); the judge grades on the preset it resolves to
        preset = resolve_preset(service.preset).text
        session, _, _ = start(Setup(task.kernel, preset, service.datatype, gpu, disabled_families=self.families))
        script = Script(session)
        solution = reference_solution(task.kernel) if self.references else None
        if solution is None and self.references:
            print(f"nestforge_reference: {task.kernel} has no reference solution; default moves", file=sys.stderr)
        solve = solution.solve if solution is not None else baseline
        solve(script, Context("gpu" if gpu else "cpu", ncores(), preset))
        # the judge reads an episode's library from the episode's folder, the one writable folder under /shared
        root = self._workdir or pathlib.Path(shared_dir()) / "nestforge"
        root.mkdir(parents=True, exist_ok=True)
        out = pathlib.Path(tempfile.mkdtemp(prefix=f"reference-{pathlib.PurePath(task.kernel).name}-", dir=root))
        final = session.finalize(out)
        self.keep_episode_record(
            task.kernel, applied_journal(session.applied), len(final.kernels), solution is not None
        )
        submission = Submission(
            language=task.language, library=str(final.library), workspace_bytes=final.workspace_bytes
        )
        # an episode's folder outlives this process: a deferred /submit grades the library after the run
        if self._workdir is None:
            weakref.finalize(submission, shutil.rmtree, str(out), ignore_errors=True)
        return submission

    def keep_episode_record(self, kernel: str, journal: list[list[object]], kernels: int, replayed: bool) -> None:
        """The column's plugin record, beside the library in the episode's folder; none without one."""
        if self._workdir is None:
            return
        reference = replayed if self.references else None
        data = episode_record(kernel, [], journal, kernels, reference, families=sorted(checked_families(self.families)))
        try:
            keep_capped(lambda record: self.keep_record(SCHEMA, record), data)
        except ValueError as err:
            print(f"{self.fname}: the plugin record was not written: {err}", file=sys.stderr)


class DefaultOptimizer(ReferenceOptimizer):
    """NestForge's default moves on every kernel, with no model and no reference solution: the floor every
    comparison starts from."""

    __slots__ = ()
    references: ClassVar[bool] = False


class NoWaveOptimizer(DefaultOptimizer):
    """The canonical form without wavefronts, finished by NestForge's default scopes, placement and code generation:
    every move family is off, so the default moves stage skews and fuses nothing. Against ``nestforge_default`` it
    gives the default moves' gain, the wavefronts included."""

    __slots__ = ()
    families: ClassVar[tuple[str, ...]] = (ALL,)


#: Moves the stub agent applies at most.
STUB_MOVES = 3


def tool(name: str, kernel: str, /, **args: object) -> dict[str, Any]:
    """One NestForge tool call as the agent's tool makes it (``POST /nestforge/<name>`` under the episode's identity);
    raises with the judge's error."""
    answer: dict[str, Any] = http_json.post_judge(f"/nestforge/{name}", {**args, "kernel": kernel})
    if "error" in answer:
        raise RuntimeError(f"{name}: {answer['error']}")
    return answer


def advance_phase(kernel: str, phase: str) -> None:
    """End the kernel's phase as the agent's ``advance_phase`` tool does; raises unless NestForge moved it to ``phase``."""
    step = tool("advance_phase", kernel)["result"]
    if not isinstance(step, dict) or step.get("status") != "moved" or step.get("phase") != phase:
        raise RuntimeError(f"advance_phase, expected {phase}: {step}")


def seeded(kernel: str) -> random.Random:
    """A generator seeded by the kernel's name, so a stub's run is repeatable."""
    return random.Random(int.from_bytes(hashlib.sha256(kernel.encode()).digest()[:8], "big"))


def apply_moves(kernel: str, pick: random.Random) -> None:
    """Up to :data:`STUB_MOVES` legal moves, chosen by ``pick``."""
    tool("get_cpf", kernel)
    for _ in range(STUB_MOVES):
        moves = tool("list_moves", kernel)["result"]
        if not isinstance(moves, list) or not moves:
            break
        move = pick.choice(moves)
        tool("apply_move", kernel, kind=move["kind"], labels=move["labels"])


def finalized(task: Task) -> Submission:
    """The library ``finalize`` builds from the kernel's session, as the submission."""
    final = tool("finalize", task.kernel)["result"]
    assert isinstance(final, dict), final
    return Submission(
        language=task.language, library=str(final["library"]), workspace_bytes=str(final["workspace_bytes"])
    )


def doalls(kernel: str) -> list[str]:
    """The labels of the top-level doalls, read from the tree with its states open."""
    states = re.findall(r"\bstate\d+\b", str(tool("view_tree", kernel)["result"]))
    return re.findall(r"\bdoall\d+\b", str(tool("view_tree", kernel, expand=states)["result"]))


def set_one_scope(kernel: str, pick: random.Random) -> None:
    """Make one doall, chosen by ``pick``, a kernel, when the program has a doall."""
    labels = doalls(kernel)
    if labels:
        done = tool("set_scope", kernel, labels=[pick.choice(labels)])["result"]
        if done["status"] != spelled(MoveStatus.APPLIED):
            raise RuntimeError(f"set_scope: {done}")


def write_back(kernel: str, info: dict[str, str]) -> bool:
    """``write_kernel`` of the source ``read_kernel`` showed, unchanged; whether its oracle checked it, which a sweep
    needs."""
    name = info["name"]
    tool("focus_kernel", kernel, name=name)
    source = tool("read_kernel", kernel, name=name)["result"]
    done = tool("write_kernel", kernel, name=name, source=source, language=kernel_language(info["device"]))["result"]
    if done["status"] not in (spelled(KernelStatus.OK), spelled(KernelStatus.UNCHECKED)):
        raise RuntimeError(f"write_kernel {name}: {done}")
    return bool(done["status"] == spelled(KernelStatus.OK))


class StubAgent(FlavorOptimizer):
    """``nestforge_stub``: the benchmark's CI stand-in for an LLM agent on the NestForge packet. Through the packet's
    own tools it reads the CPF, applies up to :data:`STUB_MOVES` legal moves picked by a generator seeded by the
    kernel's name (so a run is repeatable), finalizes and returns the library to submit."""

    __slots__ = ()

    def tune(self, task: Task, pipeline: str) -> Submission:
        apply_moves(task.kernel, seeded(task.kernel))
        return finalized(task)


class TwoStageStub(FlavorOptimizer):
    """``nestforge_two_stage``: the stand-in for an LLM agent on the two-stage packet, structured then unstructured.
    It walks every phase of ``nestforge-full`` through the agent's tools and NestForge's phase gate: the moves
    :class:`StubAgent` applies, one scope, the scoped phase passed, each kernel read, written back unchanged and
    tuned, then finalize. Repeatable for the same reason."""

    __slots__ = ()

    def tune(self, task: Task, pipeline: str) -> Submission:
        kernel, pick = task.kernel, seeded(task.kernel)
        apply_moves(kernel, pick)
        advance_phase(kernel, "scopes")
        set_one_scope(kernel, pick)
        advance_phase(kernel, "scoped")
        advance_phase(kernel, "kernels")
        kernels = tool("list_kernels", kernel)["result"]
        assert isinstance(kernels, list), kernels
        checked = [info["name"] for info in kernels if write_back(kernel, info)]
        for name in checked:
            tool("focus_kernel", kernel, name=name)
            tool("tune_kernel", kernel, name=name)
        return finalized(task)
