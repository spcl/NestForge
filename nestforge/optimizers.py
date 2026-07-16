"""Optimizers: every arena variant, and the agent, under ONE contract.

The unit under evaluation is an OPTIMIZER -- a named, deterministic procedure that, given a nest, proposes
how to build it (which representation, compiler, flags, DaCe knobs). A *deterministic* optimizer IS one
arena variant: a fixed ``(opt-mode, codegen, compiler)`` DaCe cell or a ``(language, compiler, fp, cost)``
external cell. The agentic optimizer (Phase 2-4 of :mod:`docs.agentic_optimizer`) is one more optimizer
under the same contract; its stub -- the NO-OP AGENT -- proposes the Phase-1 baseline unchanged, so the
whole loop (propose -> build -> validate -> time) runs in CI with no model and no inference.

This mirrors ``optarena.harness.optimizers`` (its ``NoOpOptimizer`` / ``StubAgent``): the same "each
variant is an optimizer, the agent is just another one" philosophy, expressed over nest-forge's own
``nest`` / :class:`~nestforge.build.BuildOptions` / measure model rather than optarena's Task/Binding.

A :class:`Proposal` is only a RECIPE -- what the existing arena measure path already consumes. Nothing here
compiles or validates; that stays in :mod:`nestforge.perf.tsvc_full` / :mod:`nestforge.arena`. Predicting
which optimizer wins WITHOUT building them all is :mod:`nestforge.predictive`.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from nestforge import tsvc
from nestforge.build import DEFAULT_COMPILER, BuildOptions
from nestforge.perf import flags

#: The Phase-1 baseline opt-mode -- DaCe simplify + LoopToMap + MapFusion, the arena's speedup denominator.
#: The no-op agent proposes exactly this (see :class:`NoOpAgent`).
BASELINE_OPT_MODE = "simplify-parallel"


@dataclass(frozen=True)
class Proposal:
    """A named recipe for ONE variant of a nest -- exactly what the arena measure path consumes.

    Lane-tagged so the harness routes it; every field is an existing knob, so a proposal is a *description*
    of a cell, never new build machinery. ``fp_mode`` and ``cost_model`` are carried explicitly (not buried
    in ``flags``) because the predictor reasons over them -- e.g. "no FP error" is ``fp_mode ==
    'strict-ieee'``. The DaCe lane is strict-ieee by construction (its codegen pins ``-ffp-contract=off``
    and emits tree reductions), so its ``fp_mode`` is reported as such.
    """
    name: str
    lane: str  # "dace" | "external"
    fp_mode: str = "strict-ieee"  # the FP-precision rung this variant runs at (flags.FP_LEVELS)
    cost_model: str = "default"  # the vectorizer cost model (flags.COST_MODELS)
    # --- DaCe lane ---
    opt_mode: Optional[str] = None  # tsvc.OPT_MODES: the pre-split SDFG optimization
    build: Optional[BuildOptions] = None  # compiler, flags, codegen_impl, vectorize, ...
    # --- external lane (numpyto C / Fortran) ---
    language: Optional[str] = None  # "c" | "fortran"
    compiler: Optional[str] = None
    flags: Optional[Tuple[str, ...]] = None

    def __post_init__(self):
        if self.lane == "dace":
            if self.opt_mode is None or self.build is None:
                raise ValueError(f"dace proposal {self.name!r} needs opt_mode and build")
        elif self.lane == "external":
            if not (self.language and self.compiler and self.flags is not None):
                raise ValueError(f"external proposal {self.name!r} needs language, compiler, flags")
        else:
            raise ValueError(f"unknown lane {self.lane!r} in proposal {self.name!r}")


class Optimizer(abc.ABC):
    """A named, deterministic procedure that proposes how to build a nest.

    ``propose`` returns the same :class:`Proposal` for the same nest every time; ``None`` means the
    optimizer DECLINES this nest (e.g. an external optimizer whose flag combination is unsupported on this
    compiler). The nest argument is optional because a deterministic variant is nest-independent -- it is a
    fixed compiler/flag/opt-mode choice -- while the agent and the predictor read the nest.
    """
    name: str

    @abc.abstractmethod
    def propose(self, nest: Optional[object] = None) -> Optional[Proposal]:
        ...


class DaceOptimizer(Optimizer):
    """One DaCe-lane variant: a fixed ``(opt_mode, BuildOptions)``. Never declines (the DaCe lane builds any
    nest)."""

    def __init__(self, opt_mode: str, build: BuildOptions, name: Optional[str] = None):
        if opt_mode not in tsvc.OPT_MODES:
            raise ValueError(f"opt_mode {opt_mode!r} not in {tsvc.OPT_MODES}")
        self.opt_mode = opt_mode
        self.build = build
        vec = ",vec" if build.vectorize is not None else ""
        self.name = name or f"dace:opt={opt_mode},cc={build.compiler},codegen={build.codegen_impl}{vec}"

    def propose(self, nest: Optional[object] = None) -> Optional[Proposal]:
        return Proposal(self.name, "dace", opt_mode=self.opt_mode, build=self.build)


class ExternalOptimizer(Optimizer):
    """One external-lane variant: a numpyto ``language`` compiled by ``compiler`` at a fixed FP + cost cell.

    Flags are composed once, at construction, through the arena's own :func:`flags.lane_flags`, so this
    optimizer sweeps the identical flag set the full-matrix job does. When that combination is unsupported
    (``lane_flags`` returns ``(None, reason)`` -- e.g. clang auto-par), the optimizer DECLINES: ``propose``
    returns ``None`` and ``skip_reason`` records why, exactly as a variant is dropped with a reason today.
    """

    def __init__(self,
                 language: str,
                 family: str,
                 compiler: str,
                 fp_mode: str = "strict-ieee",
                 cost_model: str = "cheap",
                 parallel: str = "sequential",
                 nthreads: int = 1,
                 name: Optional[str] = None):
        self.language = language
        self.family = family
        self.compiler = compiler
        self.fp_mode = fp_mode
        self.cost_model = cost_model
        self.name = name or f"{language}:cc={compiler},fp={fp_mode},cost={cost_model}"
        composed, reason = flags.lane_flags(family,
                                            fp_mode,
                                            cost_model,
                                            parallel,
                                            language,
                                            nthreads,
                                            compiler=compiler)
        self.flags: Optional[Tuple[str, ...]] = tuple(composed) if composed is not None else None
        self.skip_reason: Optional[str] = reason if composed is None else None

    def propose(self, nest: Optional[object] = None) -> Optional[Proposal]:
        if self.flags is None:
            return None
        return Proposal(self.name,
                        "external",
                        fp_mode=self.fp_mode,
                        cost_model=self.cost_model,
                        language=self.language,
                        compiler=self.compiler,
                        flags=self.flags)


class NoOpAgent(Optimizer):
    """The agent's stub: propose the Phase-1 baseline UNCHANGED -- no fuse/fission, default codegen.

    The identity "agent". It lets the whole optimizer loop (propose -> build -> validate -> time) run in CI
    with no model and no inference, and it is the baseline every other optimizer is measured against.
    Mirrors ``optarena.harness.optimizers.NoOpOptimizer``. Never runs inference -- there is none to run.
    """
    name = "noop"

    def propose(self, nest: Optional[object] = None) -> Optional[Proposal]:
        return Proposal(self.name, "dace", opt_mode=BASELINE_OPT_MODE, build=BuildOptions())


def deterministic_optimizers(
    compilers: Sequence[str] = (DEFAULT_COMPILER, ),
    opt_modes: Sequence[str] = tsvc.OPT_MODES,
    external: Sequence[Tuple[str, str, str]] = (("c", "gnu", "gcc"), ),
    fp_modes: Sequence[str] = ("strict-ieee", ),
    cost_models: Sequence[str] = ("cheap", "default")
) -> List[Optimizer]:
    """Every arena variant as one optimizer -- "each variant is an optimizer".

    The DaCe lane fans ``opt_mode x compiler`` (codegen rides the build's default); the external lane fans
    ``(language, family, compiler) x fp_mode x cost_model``. Small, explicit defaults keep CI cheap;
    widen the axes for a real sweep. An external cell whose flags are unsupported is still returned as an
    optimizer -- it simply declines via ``propose`` -> ``None``, which the caller records as a skip.
    """
    out: List[Optimizer] = []
    for opt_mode in opt_modes:
        for cc in compilers:
            out.append(DaceOptimizer(opt_mode, BuildOptions(compiler=cc)))
    for language, family, cc in external:
        for fp in fp_modes:
            for cost in cost_models:
                out.append(ExternalOptimizer(language, family, cc, fp_mode=fp, cost_model=cost))
    return out
