# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A kernel built through the agent's own LLVM or MLIR pass pipeline: the baselines that choose compiler passes
instead of Tactics' moves (the paper's LLVM and MLIR agents). They start from the same kernel unit, keep the
same C entry, and end in the same static library with its shared twin, so the oracle check, ``get_performance``
and ``finalize`` treat the build like any other.

LLVM: the unit is lowered to IR at ``-O0`` without ``optnone`` (so passes may change it), ``opt -passes=<the
agent's pipeline>`` transforms it, and clang emits the object with the IR passes off (``-disable-llvm-optzns``), so
what runs is what the pipeline left; the back end still schedules and allocates registers at ``-O3``.

MLIR: ``cgeist`` lowers the unit to affine MLIR, ``polygeist-opt --pass-pipeline=<the agent's pipeline>``
transforms it, the fixed lowering (:data:`MLIR_LOWERING`) takes it to the LLVM dialect with C-style memrefs (bare
pointers, the entry's own ABI), and ``mlir-translate`` writes the IR the LLVM steps above compile."""

import os
import shlex
import shutil
import subprocess
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

#: The front end's flags for IR that passes may still change: no optimization, and no ``optnone`` on functions.
EMIT_FLAGS: tuple[str, ...] = ("-O0", "-Xclang", "-disable-O0-optnone", "-fopenmp", "-fPIC", "-march=native")
#: The back end's flags: code generation at -O3, with no IR pass of clang's own.
BACKEND_FLAGS: tuple[str, ...] = ("-O3", "-Xclang", "-disable-llvm-optzns", "-fPIC", "-march=native", "-c")
#: cgeist's flags: affine MLIR from the unit (the bench's Polygeist column uses the same).
CGEIST_FLAGS: tuple[str, ...] = ("-S", "-O0", "--raise-scf-to-affine")
#: polygeist-opt's lowering after the agent's pipeline: to the LLVM dialect with bare-pointer memrefs.
MLIR_LOWERING: tuple[str, ...] = (
    "--lower-affine",
    "--convert-scf-to-openmp",
    "--convert-polygeist-to-llvm=use-c-style-memref=1",
)
#: Characters of a tool's error output the agent reads.
ERROR_TAIL = 2000


@lru_cache(maxsize=None, typed=True)
def cgeist_flags(compiler: str) -> tuple[str, ...]:
    """cgeist's flags, with the compiler's own builtin headers (``stddef.h``): Polygeist installs no clang resource
    directory of its own, so without them every unit that includes a C++ standard header fails."""
    done = subprocess.run([compiler, "-print-resource-dir"], capture_output=True, text=True, check=False)
    resource = Path(done.stdout.strip()) / "include"
    return (*CGEIST_FLAGS, "-I", str(resource)) if done.returncode == 0 and resource.is_dir() else CGEIST_FLAGS


class PipelineError(RuntimeError):
    """A step of the pipeline failed; the message names the step and ends with the tool's output."""


def run(step: str, command: Sequence[str]) -> None:
    done = subprocess.run(list(command), capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise PipelineError(f"{step} failed: {shlex.join(command)}\n{(done.stdout + done.stderr)[-ERROR_TAIL:]}")


def sibling(compiler: str, tool: str) -> str | None:
    """``tool`` from the LLVM installation of ``compiler`` (its IR must match the compiler's version), else from PATH."""
    found = shutil.which(compiler)
    if found is not None:
        beside = Path(os.path.realpath(found)).parent / tool
        if beside.exists():
            return str(beside)
    return shutil.which(tool)


def missing(tools: Sequence[str], compiler: str) -> list[str]:
    """The ``tools`` this machine lacks for ``compiler``."""
    return [tool for tool in tools if sibling(compiler, tool) is None]


def archived(compiler: str, objects: Sequence[Path], archive: Path, link: Sequence[str]) -> None:
    """``archive`` from ``objects``, and its shared twin, which the oracle check loads."""
    archive.unlink(missing_ok=True)
    run("ar", ["ar", "rcs", str(archive), *map(str, objects)])
    run("link", [compiler, "-shared", "-fopenmp", *map(str, objects), *link, "-o", str(archive.with_suffix(".so"))])


def compiled_ir(compiler: str, ir: Path) -> Path:
    obj = ir.with_suffix(".o")
    run("back end", [compiler, *BACKEND_FLAGS, str(ir), "-o", str(obj)])
    return obj


def build_llvm_pipeline(units: Sequence[Path], compiler: str, passes: str, archive: Path, link: Sequence[str]) -> None:
    """``archive`` from ``units`` through ``opt -passes=<passes>``.

    :raises PipelineError: A step failed, with its output."""
    opt = sibling(compiler, "opt")
    if opt is None:
        raise PipelineError(f"no opt beside {compiler} on this machine")
    objects = []
    for unit in units:
        ir, optimized = unit.with_suffix(".ll"), unit.with_suffix(".opt.ll")
        run("front end", [compiler, *EMIT_FLAGS, "-std=c++20", "-emit-llvm", "-S", str(unit), "-o", str(ir)])
        run("opt", [opt, f"-passes={passes}", str(ir), "-S", "-o", str(optimized)])
        objects.append(compiled_ir(compiler, optimized))
    archived(compiler, objects, archive, link)


def build_mlir_pipeline(
    units: Sequence[Path], compiler: str, pipeline: str, archive: Path, link: Sequence[str]
) -> None:
    """``archive`` from ``units`` through cgeist, ``polygeist-opt --pass-pipeline=<pipeline>`` and the fixed lowering.

    :raises PipelineError: A tool is missing or a step failed, with its output."""
    absent = missing(("cgeist", "polygeist-opt", "mlir-translate"), compiler)
    if absent:
        raise PipelineError(f"this machine has no {absent}; the MLIR pipeline needs Polygeist")
    tools = {tool: sibling(compiler, tool) or tool for tool in ("cgeist", "polygeist-opt", "mlir-translate")}
    objects = []
    for unit in units:
        affine, moved = unit.with_suffix(".mlir"), unit.with_suffix(".opt.mlir")
        lowered, ir = unit.with_suffix(".llvm.mlir"), unit.with_suffix(".ll")
        run("cgeist", [tools["cgeist"], str(unit), *cgeist_flags(compiler), "-o", str(affine)])
        run("pipeline", [tools["polygeist-opt"], str(affine), f"--pass-pipeline={pipeline}", "-o", str(moved)])
        run("lowering", [tools["polygeist-opt"], str(moved), *MLIR_LOWERING, "-o", str(lowered)])
        run("mlir-translate", [tools["mlir-translate"], "--mlir-to-llvmir", str(lowered), "-o", str(ir)])
        objects.append(compiled_ir(compiler, ir))
    archived(compiler, objects, archive, link)


#: What a pipeline's syntax is checked on: an empty module of each IR.
EMPTY_MODULE: dict[str, str] = {"llvm": "", "mlir": "module {}\n"}


def pipeline_syntax_error(ir: str, compiler: str, pipeline: str) -> str:
    """Empty when the tool of ``ir`` parses ``pipeline`` (run on an empty module); else the tool's message."""
    tool = sibling(compiler, "opt" if ir == "llvm" else "polygeist-opt")
    if tool is None:
        return f"this machine has no {'opt' if ir == 'llvm' else 'polygeist-opt'} for the {ir} pipeline"
    flag = f"-passes={pipeline}" if ir == "llvm" else f"--pass-pipeline={pipeline}"
    done = subprocess.run(
        [tool, flag, "-disable-output" if ir == "llvm" else "-o=/dev/null"],
        input=EMPTY_MODULE[ir],
        capture_output=True,
        text=True,
        check=False,
    )
    return "" if done.returncode == 0 else (done.stdout + done.stderr)[-ERROR_TAIL:].strip()
