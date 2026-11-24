# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The gate an authored move passes before it enters the catalogue (``docs/design/transformation-requests.md``):

1. on every example of its cluster, the session offers it on ``before``, applies it, and the program then computes
   what ``after`` computes on random inputs;
2. after each application the program is a fixed point of the session's cleanup, so later moves still find it in
   canonical form;
3. at every site a corpus kernel offers, the whole program still matches the bench's reference.

The suites and the owner's review are the script's (``scripts/gate_authored_move.py``). An authored move is tested,
not proven: the gate only finds the inputs it runs."""

import copy
import inspect
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dace
import numpy as np
from dace.frontend.python.parser import DaceProgram

from tactics.ir.introspect import describe_graph
from tactics.records import MoveStatus
from tactics.session import Session
from tactics.stages.authored import AuthoredMove
from tactics.stages.canonicalize import move_cleanup
from tactics.stages.performance import Reference, time_program
from tactics.stages.requests import ATOL, EXAMPLE_SIZE, RTOL, Example, ExampleError, drawn, program_of, runnable


@dataclass(slots=True)
class GateReport:
    """What the gate found for one authored move; it passes when every list is empty and some site was tried."""

    kind: str
    example_failures: list[str] = field(default_factory=list)
    sites: int = 0
    site_failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.example_failures and not self.site_failures and self.sites > 0


def fixed_point_failure(session: Session, where: str) -> str:
    """Empty when the session's cleanup leaves its program as it is; else what it changed, by the tree before."""
    tree = describe_graph(session.sdfg)
    cleaned = copy.deepcopy(session.sdfg)
    move_cleanup(cleaned)
    return "" if describe_graph(cleaned) == tree else f"{where}: the cleanup changes the program the move leaves"


def applied_everywhere(session: Session, kind: str, where: str) -> tuple[int, list[str], list[Session]]:
    """Every site of ``kind`` the session offers, each applied on its own trial: the count, the failures, and the
    trials it applied on."""
    failures, trials = [], []
    moves = session.list_moves(kind)
    for index, move in enumerate(moves):
        trial = session.trial()
        done = trial.apply_move(kind, list(move.labels))
        site = f"{where} site {index} {list(move.labels)}"
        if done.status is not MoveStatus.APPLIED:
            failures.append(f"{site}: offered but {done.status.name.lower()}: {done.reason}")
            continue
        failed = fixed_point_failure(trial, site)
        if failed:
            failures.append(failed)
            continue
        trials.append(trial)
    return len(moves), failures, trials


def computes_like(session: Session, after: tuple[DaceProgram, dict[str, Any]], where: str) -> str:
    """Empty when the session's program leaves the values ``after`` leaves in every argument, on one draw of random
    inputs at :data:`~tactics.stages.requests.EXAMPLE_SIZE`; else why not."""
    parameters = inspect.signature(after[0].f).parameters
    symbols = {str(v): EXAMPLE_SIZE for v in after[1].values() if isinstance(v, dace.symbolic.symbol)}
    rng = np.random.default_rng(0)
    args = {name: drawn(p.annotation, symbols, rng) for name, p in parameters.items()}
    want, got = copy.deepcopy(args), copy.deepcopy(args)
    runnable(*after)(**want)
    sdfg = copy.deepcopy(session.sdfg)
    sdfg.name = "gate_" + re.sub(r"\W+", "_", where)
    sdfg(**got, **{name: value for name, value in symbols.items() if name in sdfg.symbols})
    for name in args:
        if not np.allclose(want[name], got[name], rtol=RTOL, atol=ATOL, equal_nan=True):
            return f"{where}: the move leaves values in {name} that after does not"
    return ""


def example_failures(move: AuthoredMove, examples: list[Example], work_dir: Path) -> list[str]:
    """Check 1 and 2 on every example: offered on ``before``, applied at each site, and computing like ``after``."""
    failures = []
    for index, example in enumerate(examples):
        where = f"example {index}"
        try:
            before = program_of(example.before, f"{where} before")
            after = program_of(example.after, f"{where} after")
        except ExampleError as err:
            failures.append(str(err))
            continue
        session = Session(before[0].to_sdfg(simplify=True), work_dir=str(work_dir), catalogue={move.kind: move})
        session.canonicalize()
        sites, failed, trials = applied_everywhere(session, move.kind, where)
        if sites == 0:
            failures.append(f"{where}: {move.kind} is not offered on before")
        failures += failed
        failures += [text for trial in trials if (text := computes_like(trial, after, where))]
    return failures


def site_failures(
    session: Session, kind: str, inputs: dict[str, np.ndarray], reference: Reference
) -> tuple[int, list[str]]:
    """Check 2 and 3 on one corpus kernel: every site offered, applied on its own trial, and the whole program run
    against the bench's reference. Returns the count of sites and the failures."""
    where = session.bench.kernel if session.bench is not None else "program"
    sites, failures, trials = applied_everywhere(session, kind, where)
    for index, trial in enumerate(trials):
        out = Path(tempfile.mkdtemp(dir=trial.work_dir)) / "program"
        report = time_program(trial.sdfg, out, inputs, reference, trial.sizes, 1, False)
        if report.error or not report.correct:
            failures.append(f"{where} site {index}: {report.error or f'mismatched {report.mismatched}'}")
    return sites, failures
