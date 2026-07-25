# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``optimize_program`` -- the executing entry point (``docs/PLAN_optimize_contract.md`` item 1).

Takes what a user actually has (an ``.sdfg``/``.sdfgz``, a ``.py`` holding a ``@dace.program``, or an SDFG
already in hand), lowers it, and measures every nest across the arena's compiler x FP matrix, returning
the winner per nest. Until now this chain existed only hand-assembled in ``examples/demo_fma.py``.

Lives outside :mod:`nestforge.entry` on purpose: that module PLANS and must stay dace-free so planning
stays import-light, which a test pins. This is the half that runs.

Every variant validates against the numpy oracle before its timing counts, and a nest that fails to
lower, emit or build is reported with its reason -- never silently dropped.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import dace

from nestforge.arena import ArenaResult, Cell, run_arena
from nestforge.extract import Boundary, trip_count_symbols
from nestforge.pass_lower import lower_nests_to_external_call
from nestforge.report import render_markdown
from nestforge.translate import emit_sources, prepare
from nestforge.tsvc import shape_symbols

#: Suffixes :func:`load_sdfg` reads directly as a serialized SDFG.
SDFG_SUFFIXES = (".sdfg", ".sdfgz")


@dataclass(frozen=True, slots=True)
class NestReport:
    """One lowered nest and how its variants measured. ``arena`` is ``None`` when the nest never got that
    far; ``error`` then says why."""
    name: str
    arena: Optional[ArenaResult] = None
    error: Optional[str] = None

    def winner(self, fp_mode: str = "strict-ieee") -> Optional[Cell]:
        return self.arena.winners.get(fp_mode) if self.arena is not None else None


@dataclass(frozen=True, slots=True)
class ProgramReport:
    """Every nest of one program, measured."""
    source: str
    nests: List[NestReport] = field(default_factory=list)

    def markdown(self) -> str:
        out = [f"# nest-forge report -- `{self.source}`", ""]
        for nest in self.nests:
            if nest.arena is None:
                out += [f"## nest `{nest.name}`", "", f"FAILED: {nest.error}", ""]
            else:
                out.append(render_markdown(nest.arena))
        return "\n".join(out)

    def winners(self, fp_mode: str = "strict-ieee") -> Dict[str, Cell]:
        """nest name -> its fastest correct cell at ``fp_mode``, skipping nests that have none."""
        found = ((n.name, n.winner(fp_mode)) for n in self.nests)
        return {name: cell for name, cell in found if cell is not None}


def dace_programs(module: object) -> Dict[str, dace.frontend.python.parser.DaceProgram]:
    """Every ``@dace.program`` defined at a module's top level, by name."""
    return {n: v for n, v in vars(module).items() if isinstance(v, dace.frontend.python.parser.DaceProgram)}


def load_python_program(path: Path, func_name: Optional[str] = None) -> dace.SDFG:
    """Import a ``.py`` by PATH and lower its ``@dace.program`` to an SDFG.

    Closes ``docs/PLAN_optimize_contract.md`` item 3. Imported by path rather than by dotted name so a
    user's file needs no package, no ``__init__.py`` and no ``sys.path`` surgery. ``func_name`` picks the
    entry when a module defines several -- a module with more than one and no choice given is an error,
    not a guess, because measuring the wrong function still produces a plausible-looking report.
    """
    spec = importlib.util.spec_from_file_location(f"nestforge_user_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"{path} is not importable as a python module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    programs = dace_programs(module)
    if not programs:
        raise ValueError(f"{path} defines no @dace.program at module level (found: {sorted(vars(module))[:12]})")
    if func_name is not None:
        if func_name not in programs:
            raise ValueError(f"{path} has no @dace.program {func_name!r} (has: {sorted(programs)})")
        return programs[func_name].to_sdfg(simplify=True)
    if len(programs) > 1:
        raise ValueError(f"{path} defines {sorted(programs)}; pass func_name to say which one to optimize")
    return next(iter(programs.values())).to_sdfg(simplify=True)


def load_sdfg(source: Union[str, Path, dace.SDFG], func_name: Optional[str] = None) -> dace.SDFG:
    """Whatever the user has -> an SDFG: an SDFG passed through, a serialized ``.sdfg``/``.sdfgz`` read,
    a ``.py`` imported and lowered."""
    if isinstance(source, dace.SDFG):
        return source
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"no such kernel source: {path}")
    if path.suffix in SDFG_SUFFIXES:
        return dace.SDFG.from_file(str(path))
    if path.suffix == ".py":
        return load_python_program(path, func_name)
    raise ValueError(f"cannot read {path.suffix or path.name!r} as a kernel; expected .py or one of "
                     f"{SDFG_SUFFIXES} (a provided C/C++/Fortran source has no numpy oracle to validate "
                     f"against -- see docs/PLAN_optimize_contract.md item 2)")


def resolve_nest_symbols(boundary: Boundary, sizes: Dict[str, int]) -> Dict[str, int]:
    """``sizes`` plus a value for every boundary symbol it does not name.

    Outlining a nest carries symbols the PROGRAM never exposed -- a leaked outer index, an interstate
    temporary. Two of those are provably safe to send in as ``0``: one the nest never takes as an
    argument (it cannot reach the computation), and one that cannot change the TRIP COUNT (it selects
    WHICH element is touched, never HOW MUCH work runs). Anything else raises, because sizing a real
    bound to 0 would validate vacuously against a degenerate loop. Same rule as
    :func:`nestforge.tsvc.sample_sizes`, which proved it on the TSVC corpus.
    """
    nest = boundary.standalone_sdfg
    arglist = set(nest.arglist())
    work_syms = trip_count_symbols(nest) | shape_symbols(nest)
    resolved = dict(sizes)
    for sym in boundary.symbols:
        if sym in resolved:
            continue
        if sym not in arglist or sym not in work_syms:
            resolved[sym] = 0
        else:
            raise ValueError(f"boundary symbol {sym!r} is read by the nest and sizes its work, so it needs a "
                             f"real value; pass it in sizes (given: {sorted(sizes)})")
    return resolved


def optimize_program(source: Union[str, Path, dace.SDFG],
                     sizes: Dict[str, int],
                     out_dir: Optional[Union[str, Path]] = None,
                     func_name: Optional[str] = None,
                     strategy: str = "skip-taskloops",
                     reps: int = 50,
                     seed: int = 0) -> ProgramReport:
    """Lower ``source``, then measure every nest across the arena's compiler x FP matrix.

    :param sizes: value per free symbol. Required, never inferred -- a guessed size silently changes what
        was measured, and a too-small one measures cache-resident behaviour that will not hold.
    :param out_dir: build tree; a temp dir when unset (kept, since the winner's ``.so`` lives there).
    :param func_name: which ``@dace.program`` to take from a ``.py`` defining several.
    :param strategy: offload granularity (:mod:`nestforge.strategies`); ``skip-taskloops`` takes the
        compute-bearing nests rather than the scheduling wrappers around them.
    """
    sdfg = load_sdfg(source, func_name)
    root = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp(prefix="nestforge_"))
    root.mkdir(parents=True, exist_ok=True)

    report = ProgramReport(source=str(source if not isinstance(source, dace.SDFG) else sdfg.name))
    for ext, boundary in lower_nests_to_external_call(sdfg, strategy):
        work = root / ext.name
        try:
            nest_sizes = resolve_nest_symbols(boundary, sizes)
            prep = prepare(boundary, ext.name, work / "kern")
            c_source = next(p for p in emit_sources(prep, work / "gen", target="c") if p.suffix == ".c")
            arena = run_arena(prep, boundary, c_source, work / "build", sizes=nest_sizes, reps=reps, seed=seed)
            report.nests.append(NestReport(ext.name, arena=arena))
        except Exception as e:  # a nest the translator or arena cannot handle is reported, not fatal
            report.nests.append(NestReport(ext.name, error=f"{type(e).__name__}: {e}"))
    return report
