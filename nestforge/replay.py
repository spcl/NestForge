# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Replay a benchmark's reference solution: a Python script of stage 2 calls, run next to the default.

A solution script (``reference_solutions/<kernel>.py``) names its corpus kernel and the configurations it is
measured on, and defines ``solve(script, ctx)``: the moves of the reference schedule, picked by kind and by what the
rows read and write, never by label. Replay starts the kernel's canonicalized session, runs the baseline
(``default_moves()``), the reference and every decoy on their own copies, takes each through the default stages 3 to
5, times the whole program and checks it against the bench oracle, writes one JSON record per configuration under
``experiments/replay/<kernel>/`` and prints a Markdown table of them::

    python -m nestforge.replay reference_solutions/tsvc_2_s231.py
    python -m nestforge.replay <solution.py> --config cpu:72:M --md <note>.md

Each configuration runs in its own process with ``OMP_NUM_THREADS`` set before any OpenMP runtime starts.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import importlib.util
import json
import os
import resource
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dace.sdfg.state import ControlFlowBlock

from nestforge.ir.introspect import Row, nest_reads_writes
from nestforge.paths import REPO_ROOT
from nestforge.records import MoveStatus, Record
from nestforge.server import Setup, start
from nestforge.session import Move, Session
from nestforge.stages.performance import Reference

#: The command line's one-line help.
DESCRIPTION = "Replay a benchmark's reference solution: a Python script of stage 2 calls, run next to the default."
#: Microseconds per millisecond.
US_PER_MS = 1e3
#: Whole-program calls timed per variant (each time is their mean).
DEFAULT_REPS = 5
#: The datatype the bench oracle grades.
DATATYPE = "float64"
#: Where the records go, one folder per kernel; kept out of the repository like quickstart's runs.
REPLAY_DIR = REPO_ROOT / "experiments" / "replay"
#: Thread placement of every timed run, as the benchmark records were measured.
OMP_PLACEMENT = {"OMP_PROC_BIND": "close", "OMP_PLACES": "cores"}
#: Overrides the NestForge commit a record names, for a tree copied without its ``.git``.
COMMIT_ENV = "NESTFORGE_COMMIT"
#: Spaces per level of a written record.
JSON_INDENT = 2
#: Fields of a ``--config`` spelling: device, threads, preset.
CONFIG_FIELDS = 3
#: Characters of a variant's first error line the table shows.
ERROR_CELL = 80
#: Hex digits of a commit the table shows.
SHORT_SHA = 9
#: Hex digits of a full git commit SHA.
FULL_SHA = 40
#: The characters of a git SHA.
HEX_DIGITS = frozenset("0123456789abcdef")


class ReplayError(RuntimeError):
    """A solution script asked for a move that is not offered, or a move that did not apply."""


@dataclass(frozen=True, slots=True)
class Context:
    """One configuration a solution is measured on; ``solve`` may branch on it."""

    device: str
    threads: int = 1
    preset: str = "M"

    @property
    def key(self) -> str:
        return f"{self.device}-{self.threads}-{self.preset}"

    @property
    def spelling(self) -> str:
        return f"{self.device}:{self.threads}:{self.preset}"

    @classmethod
    def parse(cls, text: str) -> Context:
        """``cpu:72:M`` -> ``Context("cpu", 72, "M")``."""
        parts = text.split(":")
        if len(parts) != CONFIG_FIELDS or parts[0] not in ("cpu", "gpu"):
            raise ValueError(f"a config is device:threads:preset, e.g. cpu:72:M; got {text!r}")
        return cls(parts[0], int(parts[1]), parts[2])


class AppliedMove(Record):
    """A move a script applied, with the labels it had on this run."""

    kind: str
    labels: tuple[str, ...]
    reason: str
    tile_sizes: tuple[int, ...] = ()
    implementation: str | None = None


def row_containers(row: Row) -> tuple[set[str], set[str]]:
    """The containers a tree row reads and writes."""
    obj, state = row
    if state is not None:
        reads, writes = nest_reads_writes(state, obj)
        return set(reads), set(writes)
    assert isinstance(obj, ControlFlowBlock), "a row without a state is a control-flow block"
    reads, writes = obj.read_and_write_sets()
    return set(map(str, reads)), set(map(str, writes))


def move_matches(
    rows: Sequence[Row], reads: Iterable[str], writes: Iterable[str], where: Callable[[Sequence[Row]], bool] | None
) -> bool:
    """Whether the rows of a move read every one of ``reads``, write every one of ``writes`` and satisfy ``where``."""
    seen_reads: set[str] = set()
    seen_writes: set[str] = set()
    for row in rows:
        row_reads, row_writes = row_containers(row)
        seen_reads |= row_reads
        seen_writes |= row_writes
    return set(reads) <= seen_reads and set(writes) <= seen_writes and (where is None or where(rows))


@dataclass(slots=True)
class Script:
    """What a solution script drives: a session's stage 2, by moves picked without labels."""

    session: Session
    applied: list[AppliedMove] = field(default_factory=list)
    finished: bool = False

    def pick(
        self,
        kind: str,
        *,
        reads: Iterable[str] = (),
        writes: Iterable[str] = (),
        where: Callable[[Sequence[Row]], bool] | None = None,
    ) -> Move:
        """The one legal move of ``kind`` whose rows read ``reads`` and write ``writes`` (all of them) and satisfy
        ``where``.

        :raises ReplayError: No move or several moves match; the message lists what is offered and the tree.
        """
        rows = self.session.row_index()
        offered = self.session.list_moves(kind)
        hits = [m for m in offered if move_matches([rows[label] for label in m.labels], reads, writes, where)]
        if len(hits) == 1:
            return hits[0]
        found = "no move" if not hits else f"{len(hits)} moves"
        listing = ", ".join(str(list(m.labels)) for m in offered) or "none"
        raise ReplayError(
            f"{found} of {kind} match (reads {sorted(reads)}, writes {sorted(writes)}); offered: {listing}\n"
            + self.session.describe()
        )

    def apply(self, move: Move) -> AppliedMove:
        """Apply ``move``; raises :class:`ReplayError` unless it applied."""
        if self.finished:
            raise ReplayError(f"{move.kind} after finish(): stage 2 is closed")
        result = self.session.apply_move(move.kind, move.labels, move.tile_sizes, move.implementation)
        if result.status is not MoveStatus.APPLIED:
            raise ReplayError(f"{move.kind} {list(move.labels)}: {result.status.name.lower()}: {result.reason}")
        done = AppliedMove(
            kind=move.kind,
            labels=result.labels,
            reason=result.reason,
            tile_sizes=move.tile_sizes,
            implementation=move.implementation,
        )
        self.applied.append(done)
        return done

    def finish(self) -> None:
        """Close stage 2 after hand-picked moves (:meth:`Session.finish_moves`)."""
        self.session.finish_moves()
        self.finished = True

    def defaults(self) -> None:
        """Close stage 2 with its default (:meth:`Session.default_moves`), as the baseline does."""
        self.session.default_moves()
        self.finished = True


#: A reference or decoy: stage 2 calls on a script, for one configuration.
Steps = Callable[[Script, Context], None]


@dataclass(frozen=True, slots=True)
class Solution:
    """A loaded solution script."""

    path: Path
    kernel: str
    solve: Steps
    decoys: dict[str, Steps]
    configs: tuple[Context, ...]
    ci_skip: str = ""  # why CI does not replay it yet; empty when it does


def load_solution(path: Path) -> Solution:
    """Import a solution script: ``KERNEL``, ``CONFIGS``, ``solve``, optional ``DECOYS`` and ``CI_SKIP``. Loaded by
    file path, as :meth:`CorpusKernel.module` loads a ``_dace.py``: a benchmark folder is not a package."""
    spec = importlib.util.spec_from_file_location(f"nestforge_solution_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"{path} is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = vars(module)
    missing = [name for name in ("KERNEL", "CONFIGS", "solve") if name not in names]
    if missing:
        raise ImportError(f"{path} defines no {', '.join(missing)}")
    return Solution(
        path=path.resolve(),
        kernel=names["KERNEL"],
        solve=names["solve"],
        decoys=dict(names.get("DECOYS", {})),
        configs=tuple(names["CONFIGS"]),
        ci_skip=str(names.get("CI_SKIP", "")),
    )


def baseline(script: Script, ctx: Context) -> None:
    """Stage 2's default: what a benchmark claims to beat."""
    script.defaults()


class VariantRecord(Record):
    """One schedule of a configuration: its moves, tree, whole-program time and oracle check."""

    name: str
    moves: tuple[AppliedMove, ...] = ()
    tree: str = ""
    time_ms: float | None = None
    correct: bool | None = None
    mismatched: tuple[str, ...] = ()
    error: str = ""
    speedup: float | None = None  # baseline time over this time


class ReplayReport(Record):
    """Every variant of one configuration, and the versions it ran on."""

    kernel: str
    solution: str
    device: str
    threads: int
    preset: str
    reps: int
    date: str
    pins: dict[str, str]
    variants: tuple[VariantRecord, ...]


def run_variant(
    session: Session,
    ctx: Context,
    name: str,
    steps: Steps,
    data: tuple[dict[str, np.ndarray], Reference | None],
    reps: int,
) -> VariantRecord:
    """``steps`` on a copy of the canonicalized session, then stages 3 to 5 by default, timed and checked."""
    trial = session.trial()
    script = Script(trial)
    try:
        steps(script, ctx)
        if not script.finished:
            script.finish()
    except ReplayError as err:
        return VariantRecord(name=name, moves=tuple(script.applied), error=str(err))
    tree = trial.describe()
    perf = trial.performance(data[0], data[1], reps).structured
    return VariantRecord(
        name=name,
        moves=tuple(script.applied),
        tree=tree,
        time_ms=perf.time_us / US_PER_MS if perf.time_us is not None else None,
        correct=perf.correct,
        mismatched=tuple(perf.mismatched),
        error=perf.error,
    )


def with_speedups(variants: list[VariantRecord]) -> tuple[VariantRecord, ...]:
    """Each variant with the baseline's time over its own."""
    base = variants[0].time_ms
    return tuple(
        v.model_copy(update={"speedup": base / v.time_ms}) if base is not None and v.time_ms else v for v in variants
    )


def distribution_commit(name: str) -> str:
    """The git commit pip installed ``name`` from, or its version when it was not installed from git."""
    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"
    raw = dist.read_text("direct_url.json")
    commit = json.loads(raw).get("vcs_info", {}).get("commit_id") if raw else None
    return commit or dist.version


def nestforge_commit() -> str:
    if os.environ.get(COMMIT_ENV):
        return os.environ[COMMIT_ENV]
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("+dirty" if dirty.strip() else "")


def pins() -> dict[str, str]:
    return {
        "nestforge": nestforge_commit(),
        "dace": distribution_commit("dace"),
        "hpcagent_bench": distribution_commit("hpcagent_bench"),
    }


def replay(solution: Solution, ctx: Context, reps: int = DEFAULT_REPS) -> ReplayReport:
    """The baseline, the reference and every decoy of ``solution`` on ``ctx``, in that order."""
    # the first configuration per kernel: a replay compares moves, not compiler sweeps
    session, inputs, reference = start(Setup(solution.kernel, ctx.preset, DATATYPE, ctx.device == "gpu"), "first")
    steps: dict[str, Steps] = {"baseline": baseline, "reference": solution.solve, **solution.decoys}
    variants = [run_variant(session, ctx, name, fn, (inputs, reference), reps) for name, fn in steps.items()]
    return ReplayReport(
        kernel=solution.kernel,
        solution=solution.path.relative_to(REPO_ROOT).as_posix()
        if solution.path.is_relative_to(REPO_ROOT)
        else str(solution.path),
        device=ctx.device,
        threads=ctx.threads,
        preset=ctx.preset,
        reps=reps,
        date=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        pins=pins(),
        variants=with_speedups(variants),
    )


def report_path(solution: Solution, ctx: Context) -> Path:
    return REPLAY_DIR / solution.kernel / f"{ctx.key}.json"


def write_report(report: ReplayReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(mode="json"), indent=JSON_INDENT) + "\n")


def read_reports(folder: Path) -> list[ReplayReport]:
    return [ReplayReport.model_validate_json(p.read_text()) for p in sorted(folder.glob("*.json"))]


def short(commit: str) -> str:
    """A full commit SHA cut to :data:`SHORT_SHA` digits; any other spelling (a version, a label) kept whole."""
    return commit[:SHORT_SHA] if len(commit) == FULL_SHA and all(c in HEX_DIGITS for c in commit) else commit


def cell(value: float | None, fmt: str) -> str:
    return "n/a" if value is None else format(value, fmt)


def oracle_cell(variant: VariantRecord) -> str:
    if variant.error:
        return "error: " + variant.error.splitlines()[0][:ERROR_CELL]
    return {True: "ok", False: "wrong " + ", ".join(variant.mismatched), None: "unchecked"}[variant.correct]


def markdown(reports: Sequence[ReplayReport]) -> str:
    """One table over every configuration: the time of each variant, its speedup and its oracle check."""
    lines = [
        "| Config | Variant | Moves | Time (ms) | Speedup | Oracle |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for rep in reports:
        config = f"{rep.device}, {rep.threads} thread(s), preset {rep.preset}"
        for v in rep.variants:
            moves = ", ".join(m.kind for m in v.moves) or ("default_moves" if v.name == "baseline" else "none")
            lines.append(
                f"| {config} | {v.name} | {moves} | {cell(v.time_ms, '.2f')} | {cell(v.speedup, '.2f')}x |"
                f" {oracle_cell(v)} |"
            )
    lines.append("")
    for rep in reports:
        pinned = ", ".join(f"{name} `{short(sha)}`" for name, sha in rep.pins.items())
        lines.append(f"- {rep.device}-{rep.threads}-{rep.preset}: {rep.date}, mean of {rep.reps} calls; {pinned}.")
    return "\n".join(lines) + "\n"


def markers(kernel: str) -> tuple[str, str]:
    """The comments around ``kernel``'s generated table in a document that holds several."""
    return f"<!-- replay:{kernel}:begin -->", f"<!-- replay:{kernel}:end -->"


def splice(document: str, kernel: str, section: str) -> str:
    """``document`` with the text between ``kernel``'s replay markers replaced by ``section`` (appended when
    unmarked)."""
    begin, end = markers(kernel)
    block = f"{begin}\n{section}{end}"
    if begin in document and end in document:
        head, _, rest = document.partition(begin)
        _, _, tail = rest.partition(end)
        return head + block + tail
    return document.rstrip("\n") + "\n\n" + block + "\n"


def openmp_env(ctx: Context) -> dict[str, str]:
    """The OpenMP variables ``ctx`` runs under: the thread count and placement, and the launch environment the bench
    oracle refuses to run without (:func:`hpcagent_bench.flags.openmp_launch_env`)."""
    # deferred like every hpcagent_bench import, so importing nestforge never loads it
    from hpcagent_bench.flags import openmp_launch_env

    return {**openmp_launch_env(), "OMP_NUM_THREADS": str(ctx.threads), **OMP_PLACEMENT}


def full_stack() -> bool:
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    return soft == hard


def raise_stack() -> None:
    """The stack limit up to its hard limit, as ``ulimit -s "$(ulimit -H -s)"`` does; run in the child before exec."""
    hard = resource.getrlimit(resource.RLIMIT_STACK)[1]
    resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))


def in_place(ctx: Context) -> bool:
    """Whether this process was launched with ``ctx``'s OpenMP environment: the runtimes read it once, at load."""
    return full_stack() and all(os.environ.get(name) == value for name, value in openmp_env(ctx).items())


def run_config(solution: Solution, ctx: Context, reps: int) -> int:
    """Replay one configuration here when the environment fits, else in a child process launched with it."""
    if not in_place(ctx):
        argv = [sys.executable, "-m", "nestforge.replay", str(solution.path), "--config", ctx.spelling]
        env = {**os.environ, **openmp_env(ctx)}
        return subprocess.run([*argv, "--reps", str(reps)], env=env, preexec_fn=raise_stack, check=False).returncode
    report = replay(solution, ctx, reps)
    write_report(report, report_path(solution, ctx))
    print(markdown([report]))
    checked = all(v.correct is True and not v.error for v in report.variants[:2])  # baseline and reference
    return 0 if checked else 1


def parse(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=DESCRIPTION)
    ap.add_argument("solution", type=Path, help="a reference_solutions/<kernel>.py")
    ap.add_argument("--config", action="append", type=Context.parse, help="device:threads:preset (default: CONFIGS)")
    ap.add_argument("--reps", type=int, default=DEFAULT_REPS, help="timed calls per variant")
    ap.add_argument(
        "--md", type=Path, help="also write the table into this document, between the kernel's replay markers"
    )
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse(argv)
    solution = load_solution(args.solution)
    configs: list[Context] = args.config or list(solution.configs)
    status = max(run_config(solution, ctx, args.reps) for ctx in configs)
    if args.md is not None:
        table = markdown(read_reports(REPLAY_DIR / solution.kernel))
        args.md.write_text(splice(args.md.read_text() if args.md.exists() else "", solution.kernel, table))
    return status


if __name__ == "__main__":
    sys.exit(main())
