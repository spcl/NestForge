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

from hpcagent_agent.tools import http_json
from hpcagent_bench import config
from hpcagent_bench.flags import ncores
from hpcagent_bench.harness.envelope import Submission
from hpcagent_bench.harness.optimizers import FlavorOptimizer
from hpcagent_bench.harness.sandbox import shared_dir
from hpcagent_bench.harness.task import Task
from hpcagent_bench.spec import resolve_preset

from nestforge.paths import REPO_ROOT
from nestforge.replay import Context, Script, Solution, baseline, load_solution
from nestforge.records import KernelStatus, MoveStatus, spelled
from nestforge.server import Setup, start
from nestforge.session import kernel_language

__all__ = ["ReferenceOptimizer", "StubAgent", "TwoStageStub", "reference_solution"]

#: Where the reference solutions live, one ``<kernel>.py`` each.
SOLUTIONS = REPO_ROOT / "reference_solutions"


def reference_solution(kernel: str) -> Solution | None:
    """The reference solution of ``kernel``, or ``None`` when it has none."""
    path = SOLUTIONS / f"{pathlib.PurePath(kernel).name}.py"
    return load_solution(path) if path.is_file() else None


class ReferenceOptimizer(FlavorOptimizer):
    """Replays the kernel's reference moves (default moves without one), then finalizes on NestForge's defaults."""

    __slots__ = ()

    def tune(self, task: Task, pipeline: str) -> Submission:
        gpu = task.image != "cpu"
        service = config.current().service
        # the service's preset is a token ("XL+fuzz"); the judge grades on the preset it resolves to
        preset = resolve_preset(service.preset).text
        session, _, _ = start(Setup(task.kernel, preset, service.datatype, gpu))
        script = Script(session)
        solution = reference_solution(task.kernel)
        if solution is None:
            print(f"nestforge_reference: {task.kernel} has no reference solution; default moves", file=sys.stderr)
        solve = solution.solve if solution is not None else baseline
        solve(script, Context("gpu" if gpu else "cpu", ncores(), preset))
        # the judge reads an episode's library from the episode's folder, the one writable folder under /shared
        root = self._workdir or pathlib.Path(shared_dir()) / "nestforge"
        root.mkdir(parents=True, exist_ok=True)
        out = pathlib.Path(tempfile.mkdtemp(prefix=f"reference-{pathlib.PurePath(task.kernel).name}-", dir=root))
        final = session.finalize(out)
        submission = Submission(
            language=task.language, library=str(final.library), workspace_bytes=final.workspace_bytes
        )
        # an episode's folder outlives this process: a deferred /submit grades the library after the run
        if self._workdir is None:
            weakref.finalize(submission, shutil.rmtree, str(out), ignore_errors=True)
        return submission


#: Moves the stub agent applies at most.
STUB_MOVES = 3


def tool(name: str, kernel: str, /, **args: object) -> dict[str, object]:
    """One NestForge tool call as the agent's tool makes it (``POST /nestforge/<name>`` under the episode's identity);
    raises with the judge's error."""
    answer = http_json.post_judge(f"/nestforge/{name}", {**args, "kernel": kernel})
    if "error" in answer:
        raise RuntimeError(f"{name}: {answer['error']}")
    return answer


def next_phase(kernel: str, phase: str) -> None:
    """End the kernel's phase as the agent's ``next_phase`` tool does; raises unless the judge moved it to ``phase``."""
    answer = http_json.post_judge("/next_phase", {"kernel": kernel})
    if answer.get("phase") != phase:
        raise RuntimeError(f"next_phase, expected {phase}: {answer}")


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
    source = tool("read_kernel", kernel, name=name)["result"]
    done = tool("write_kernel", kernel, name=name, source=source, language=kernel_language(info["device"]))["result"]
    if done["status"] not in (spelled(KernelStatus.OK), spelled(KernelStatus.UNCHECKED)):
        raise RuntimeError(f"write_kernel {name}: {done}")
    return done["status"] == spelled(KernelStatus.OK)


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
    It walks every phase of ``nestforge-full`` through the agent's tools and the judge's phase gate: the moves
    :class:`StubAgent` applies, one scope, each kernel read and written back unchanged, each kernel tuned, then
    finalize. Repeatable for the same reason. The judge must run with that packet as its ``record.packet``."""

    __slots__ = ()

    def tune(self, task: Task, pipeline: str) -> Submission:
        kernel, pick = task.kernel, seeded(task.kernel)
        apply_moves(kernel, pick)
        next_phase(kernel, "scopes")
        set_one_scope(kernel, pick)
        next_phase(kernel, "kernels")
        kernels = tool("list_kernels", kernel)["result"]
        assert isinstance(kernels, list), kernels
        checked = [info["name"] for info in kernels if write_back(kernel, info)]
        next_phase(kernel, "codegen")
        for name in checked:
            tool("tune_kernel", kernel, name=name)
        return finalized(task)
