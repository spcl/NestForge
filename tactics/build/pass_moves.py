# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The MLIR and LLVM agent's moves (``docs/design/mlir-llvm-agent.md``): compiler passes applied one at a time to a
kernel's unit, each a named move with arguments, MLIR passes first, then LLVM passes on the IR the fixed lowering
writes. The moves are kept as one JSON pipeline (:func:`spec_text`), so the session's rollback and rebuilds treat them
as they treat the baselines' pipeline strings.

MLIR: ``cgeist`` lowers the unit to affine MLIR, ``polygeist-opt`` runs every MLIR move followed by
``canonicalize,cse``, and the fixed lowering of :mod:`tactics.build.pipelines` takes it to LLVM IR. LLVM: one ``opt``
call per move, since the knobs a move takes (``-force-vector-width``, ``-unroll-count``) are process-wide. The back
end then compiles what the moves left."""

import json
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from tactics.build.pipelines import (
    ERROR_TAIL,
    MLIR_LOWERING,
    PipelineError,
    archived,
    cgeist_flags,
    compiled_ir,
    missing,
    run,
    sibling,
)

#: The tools the moves run, beside the compiler.
TOOLS = ("cgeist", "polygeist-opt", "mlir-translate", "opt")


@dataclass(frozen=True, slots=True)
class PassKind:
    """One kind of move: the pass it runs, and for each argument the pass option (``{name=value}``) or the ``opt``
    knob (``-name=value``) it fills."""

    passes: str
    options: tuple[tuple[str, str], ...] = ()
    knobs: tuple[tuple[str, str], ...] = ()
    help: str = ""


MLIR_KINDS: dict[str, PassKind] = {
    "fuse": PassKind("affine-loop-fusion", (("mode", "mode"),), help="mode: greedy, producer or sibling"),
    "tile": PassKind("affine-loop-tile", (("tile_sizes", "tile-sizes"),), help="tile_sizes: one size per loop"),
    "unroll": PassKind("affine-loop-unroll", (("factor", "unroll-factor"),)),
    "unroll-jam": PassKind("affine-loop-unroll-jam", (("factor", "unroll-jam-factor"),)),
    "scalar-replace": PassKind("affine-scalrep"),
    "hoist-invariant": PassKind("affine-loop-invariant-code-motion"),
    "copy-to-buffer": PassKind("affine-data-copy-generate"),
    "vectorize": PassKind("affine-super-vectorize", (("width", "virtual-vector-size"),)),
    "coalesce": PassKind("affine-loop-coalescing"),
    "peel": PassKind("scf-for-loop-peeling"),
    "parallelize": PassKind("affine-parallelize"),
}
LLVM_KINDS: dict[str, PassKind] = {
    "cleanup": PassKind(
        "cgscc(inline),function(sroa,early-cse,instcombine,simplifycfg)", help="inlines calls, then scalar cleanup"
    ),
    "hoist-invariant": PassKind("loop-mssa(licm)"),
    "interchange": PassKind("loop(loop-interchange)"),
    "fuse": PassKind("loop-fusion"),
    "distribute": PassKind("loop-distribute", knobs=(("", "enable-loop-distribute"),)),
    "flatten": PassKind("loop(loop-flatten)"),
    "unroll-jam": PassKind("loop(loop-unroll-and-jam)", knobs=(("factor", "unroll-and-jam-count"),)),
    "vectorize": PassKind(
        "loop-vectorize",
        knobs=(("width", "force-vector-width"), ("interleave", "force-vector-interleave")),
        help="width: lanes, interleave: vector bodies per iteration",
    ),
    "slp": PassKind("slp-vectorizer"),
    "unroll": PassKind("loop-unroll", knobs=(("factor", "unroll-count"),)),
    "version-aliasing": PassKind("loop-versioning"),
    "prefetch": PassKind("loop-data-prefetch", knobs=(("distance", "prefetch-distance"),)),
    "polly": PassKind("polly", knobs=(("tile_size", "polly-default-tile-size"),)),
}
KINDS: dict[str, dict[str, PassKind]] = {"mlir": MLIR_KINDS, "llvm": LLVM_KINDS}


@dataclass(frozen=True, slots=True)
class PassMove:
    """One applied move: its phase, kind and arguments (each a string, a list joined by ``,``)."""

    ir: str
    kind: str
    arguments: tuple[tuple[str, str], ...] = ()

    def text(self) -> str:
        """The pass string the move runs; for an LLVM move, without its knobs."""
        kind, given = KINDS[self.ir][self.kind], dict(self.arguments)
        options = " ".join(f"{option}={given[name]}" for name, option in kind.options if name in given)
        return f"{kind.passes}{{{options}}}" if options else kind.passes

    def knobs(self) -> list[str]:
        """The ``opt`` flags of an LLVM move."""
        given = dict(self.arguments)
        flags = []
        for name, knob in KINDS[self.ir][self.kind].knobs:
            if not name:
                flags.append(f"-{knob}")
            elif name in given:
                flags.append(f"-{knob}={given[name]}")
        return flags


@dataclass(frozen=True, slots=True)
class PassProgram:
    """The moves applied so far and the phase the agent is in."""

    phase: str = "mlir"
    moves: tuple[PassMove, ...] = ()

    def of(self, ir: str) -> tuple[PassMove, ...]:
        return tuple(move for move in self.moves if move.ir == ir)


def spec_text(program: PassProgram) -> str:
    """``program`` as the pipeline string the session keeps."""
    moves = [[move.ir, move.kind, [list(pair) for pair in move.arguments]] for move in program.moves]
    return json.dumps({"phase": program.phase, "moves": moves}, separators=(",", ":"))


def parsed(spec: str) -> PassProgram:
    """The program :func:`spec_text` wrote."""
    raw = json.loads(spec)
    moves = tuple(
        PassMove(ir, kind, tuple((str(k), str(v)) for k, v in arguments)) for ir, kind, arguments in raw["moves"]
    )
    return PassProgram(str(raw["phase"]), moves)


def argument_text(value: object) -> str:
    """An argument as its pass option or knob spells it: a list joined by ``,``."""
    if isinstance(value, (list, tuple)):
        return ",".join(str(int(v)) for v in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def made_move(ir: str, kind: str, arguments: dict[str, object]) -> PassMove | str:
    """The move ``kind`` of phase ``ir`` with ``arguments``, or why it is refused."""
    kinds = KINDS[ir]
    if kind not in kinds:
        return f"{kind} is no {ir} move; the {ir} moves are {sorted(kinds)}."
    takes = {name for name, _ in kinds[kind].options} | {name for name, _ in kinds[kind].knobs if name}
    unknown = sorted(set(arguments) - takes)
    if unknown:
        return f"{kind} takes {sorted(takes) or 'no arguments'}; not {unknown}."
    return PassMove(ir, kind, tuple(sorted((name, argument_text(value)) for name, value in arguments.items())))


def mlir_pipeline(moves: Sequence[PassMove]) -> str:
    """The ``polygeist-opt --pass-pipeline`` of the MLIR ``moves``, each followed by ``canonicalize,cse``; with no
    move, ``canonicalize,cse`` alone, so a move that changes nothing leaves the IR as it was."""
    steps = ",".join(f"{move.text()},canonicalize,cse" for move in moves)
    return f"builtin.module(func.func({steps or 'canonicalize,cse'}))"


def tools(compiler: str) -> dict[str, str]:
    """The tools the moves run.

    :raises PipelineError: The machine lacks one."""
    absent = missing(TOOLS, compiler)
    if absent:
        raise PipelineError(f"this machine has no {absent}; the MLIR and LLVM moves need Polygeist and LLVM")
    return {tool: sibling(compiler, tool) or tool for tool in TOOLS}


def front_end_flags(compiler: str) -> tuple[str, ...]:
    """cgeist's flags for the moves: -O2, so the unit's index helpers are inlined and its accesses are affine; at -O0
    they stay calls, which no tiling can see through and no LLVM vectorizer takes (beverin job 762802)."""
    return tuple("-O2" if flag == "-O0" else flag for flag in cgeist_flags(compiler))


def mlir_after(unit: Path, compiler: str, moves: Sequence[PassMove]) -> Path:
    """The affine MLIR of ``unit`` after the MLIR ``moves``, beside it."""
    found = tools(compiler)
    affine, moved = unit.with_suffix(".mlir"), unit.with_suffix(".moved.mlir")
    if not affine.exists():
        run("cgeist", [found["cgeist"], str(unit), *front_end_flags(compiler), "-o", str(affine)])
    run(
        "MLIR moves", [found["polygeist-opt"], str(affine), f"--pass-pipeline={mlir_pipeline(moves)}", "-o", str(moved)]
    )
    return moved


def llvm_after(unit: Path, compiler: str, program: PassProgram) -> Path:
    """The LLVM IR of ``unit`` after every move of ``program``, beside it."""
    found = tools(compiler)
    moved = mlir_after(unit, compiler, program.of("mlir"))
    lowered, ir = unit.with_suffix(".llvm.mlir"), unit.with_suffix(".ll")
    run("lowering", [found["polygeist-opt"], str(moved), *MLIR_LOWERING, "-o", str(lowered)])
    run("mlir-translate", [found["mlir-translate"], "--mlir-to-llvmir", str(lowered), "-o", str(ir)])
    for number, move in enumerate(program.of("llvm")):
        out = unit.with_suffix(f".llvm{number}.ll")
        run(
            f"LLVM move {move.kind}",
            [found["opt"], f"-passes={move.text()}", *move.knobs(), str(ir), "-S", "-o", str(out)],
        )
        ir = out
    return ir


def build_pass_moves(units: Sequence[Path], compiler: str, spec: str, archive: Path, link: Sequence[str]) -> None:
    """``archive`` from ``units`` through the moves of ``spec`` (:func:`spec_text`).

    :raises PipelineError: A tool is missing or a step failed, with its output."""
    program = parsed(spec)
    archived(compiler, [compiled_ir(compiler, llvm_after(unit, compiler, program)) for unit in units], archive, link)


@lru_cache(maxsize=None, typed=True)
def registered(compiler: str, ir: str, passes: str) -> bool:
    """Whether the tool of ``ir`` on this machine parses ``passes`` (``polly`` only with Polly loaded)."""
    tool = sibling(compiler, "polygeist-opt" if ir == "mlir" else "opt")
    if tool is None:
        return False
    flag = f"--pass-pipeline=builtin.module(func.func({passes}))" if ir == "mlir" else f"-passes={passes}"
    command = [tool, flag, "-o=/dev/null" if ir == "mlir" else "-disable-output"]
    stdin = "module {}\n" if ir == "mlir" else ""
    return subprocess.run(command, input=stdin, capture_output=True, text=True, check=False).returncode == 0


# The outline of the loops, as view_ir shows it

#: An MLIR loop: ``affine.for %i = 0 to 10 step 2``, ``scf.for %i = %c0 to %n step %c1``, ``scf.parallel``.
MLIR_LOOP = re.compile(r"^(\s*)(?:%\S+\s*=\s*)?(affine\.for|scf\.for|affine\.parallel|scf\.parallel)\s+(.*?)\s*\{\s*$")
MLIR_PREFIX = {"affine.for": "for", "scf.for": "sfor", "affine.parallel": "par", "scf.parallel": "par"}


def mlir_outline(text: str) -> list[tuple[str, int, str]]:
    """One ``(label, depth, header)`` per loop of the MLIR ``text``, in preorder."""
    rows: list[tuple[str, int, str]] = []
    counts: dict[str, int] = {}
    indents: list[int] = []
    for line in text.splitlines():
        found = MLIR_LOOP.match(line)
        if found is None:
            continue
        indent = len(found.group(1))
        while indents and indents[-1] >= indent:
            indents.pop()
        prefix = MLIR_PREFIX[found.group(2)]
        counts[prefix] = counts.get(prefix, -1) + 1
        rows.append((f"{prefix}{counts[prefix]}", len(indents), found.group(3)))
        indents.append(indent)
    return rows


#: ``Loop at depth 2 containing: %for.body<header><exiting>,%for.inc<latch>``, from ``print<loops>``.
LLVM_LOOP = re.compile(r"Loop at depth (\d+) containing: %([\w.]+)")
#: A block's label line in LLVM IR, and the widest vector type in a line.
BLOCK = re.compile(r"^([\w.]+):")
VECTOR = re.compile(r"<(\d+) x ")


def llvm_outline(ir: Path, opt: str) -> list[tuple[str, int, str]]:
    """One ``(label, depth, header)`` per loop of the LLVM ``ir``, in preorder: its header block and, when its header
    computes on vectors, their widest lane count."""
    done = subprocess.run([opt, "-passes=print<loops>", "-disable-output", str(ir)], capture_output=True, text=True)
    if done.returncode != 0:
        raise PipelineError(f"print<loops> failed\n{done.stderr[-ERROR_TAIL:]}")
    widths: dict[str, int] = {}
    block = ""
    for line in ir.read_text().splitlines():
        start = BLOCK.match(line)
        if start is not None:
            block = start.group(1)
        for lanes in VECTOR.findall(line):
            widths[block] = max(widths.get(block, 1), int(lanes))
    rows = []
    for number, (depth, header) in enumerate(LLVM_LOOP.findall(done.stderr + done.stdout)):
        vec = f" vec={widths[header]}" if widths.get(header, 1) > 1 else ""
        rows.append((f"loop{number}", int(depth) - 1, f"%{header}{vec}"))
    return rows


def outline_text(rows: Sequence[tuple[str, int, str]]) -> str:
    return "\n".join(f"{'  ' * depth}{label}: {header}" for label, depth, header in rows) or "(no loops)"
