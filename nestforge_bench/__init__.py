# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``nestforge_reference``: the HPCAgent-Bench column an agent's NestForge run is compared against. No LLM: the
kernel's reference solution (``reference_solutions/<kernel>.py``) picks the moves, NestForge's default moves when the
kernel has none, and scopes, placement and code generation run on the same defaults the agent's session uses.

A package of its own beside ``nestforge``: it imports HPCAgent-Bench, which no ``nestforge`` module may."""

import hashlib
import pathlib
import random
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

from nestforge.paths import REPO_ROOT
from nestforge.replay import Context, Script, Solution, baseline, load_solution
from nestforge.server import Setup, start

__all__ = ["ReferenceOptimizer", "StubAgent", "reference_solution"]

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
        session, _, _ = start(Setup(task.kernel, service.preset, service.datatype, gpu))
        script = Script(session)
        solution = reference_solution(task.kernel)
        if solution is None:
            print(f"nestforge_reference: {task.kernel} has no reference solution; default moves", file=sys.stderr)
        solve = solution.solve if solution is not None else baseline
        solve(script, Context("gpu" if gpu else "cpu", ncores(), service.preset))
        # the judge takes a library only from its shared folder
        root = pathlib.Path(shared_dir()) / "nestforge"
        root.mkdir(parents=True, exist_ok=True)
        out = pathlib.Path(tempfile.mkdtemp(prefix=f"reference-{pathlib.PurePath(task.kernel).name}-", dir=root))
        final = session.finalize(out)
        submission = Submission(
            language=task.language, library=str(final.library), workspace_bytes=final.workspace_bytes
        )
        weakref.finalize(submission, shutil.rmtree, str(out), ignore_errors=True)
        return submission


#: Moves the stub agent applies at most.
STUB_MOVES = 3


def tool(name: str, kernel: str, **args: object) -> dict[str, object]:
    """One NestForge tool call as the agent's tool makes it (``POST /nestforge/<name>`` under the episode's identity);
    raises with the judge's error."""
    answer = http_json.post_judge(f"/nestforge/{name}", {**args, "kernel": kernel})
    if "error" in answer:
        raise RuntimeError(f"{name}: {answer['error']}")
    return answer


class StubAgent(FlavorOptimizer):
    """``nestforge_stub``: the benchmark's CI stand-in for an LLM agent on the NestForge packet. Through the packet's
    own tools it reads the CPF, applies up to :data:`STUB_MOVES` legal moves picked by a generator seeded by the
    kernel's name (so a run is repeatable), finalizes and returns the library to submit."""

    __slots__ = ()

    def tune(self, task: Task, pipeline: str) -> Submission:
        seed = int.from_bytes(hashlib.sha256(task.kernel.encode()).digest()[:8], "big")
        pick = random.Random(seed)
        tool("get_cpf", task.kernel)
        for _ in range(STUB_MOVES):
            moves = tool("list_moves", task.kernel)["result"]
            if not isinstance(moves, list) or not moves:
                break
            move = pick.choice(moves)
            tool("apply_move", task.kernel, kind=move["kind"], labels=move["labels"], epoch=move["epoch"])
        final = tool("finalize", task.kernel)["result"]
        assert isinstance(final, dict), final
        return Submission(
            language=task.language, library=str(final["library"]), workspace_bytes=str(final["workspace_bytes"])
        )
