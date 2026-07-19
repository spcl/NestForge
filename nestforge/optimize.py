"""Phase 3 of the 4-phase optimizer: optimize each externalized nest individually.

Phase 1 fixed the fusion granularity; Phase 2 externalized the chosen nests (each is now an
``ExternalCall``). Phase 3 tunes ONE nest at a time: pick a knob bundle -- representation (DaCe lane
vs numpyto C/Fortran), compiler, flags, DaCe codegen + vectorization -- and get the build recipe for
that nest.

The knob bundle IS an :class:`~nestforge.optimizers.Optimizer` (the module contract: "each variant is
an optimizer"). A deterministic optimizer is one fixed cell of the arena grid; the agent is one more
optimizer under the same contract. So Phase 3 needs no new type -- only the phase verb and the choice
inspector:

  * :func:`optimization_choices` -- the knob-bundle grid the agent picks from. Non-mutating, the
    Phase-3 analog of ``enumerate_fusions`` (Phase 1) / ``offload_candidates`` (Phase 2). Each choice
    is a named optimizer that describes its own knobs.
  * :func:`optimize` -- the verb: apply a chosen bundle to one nest, get its :class:`Proposal`
    (a build recipe -- nothing compiles here; the arena measure path consumes the recipe). ``None``
    means the bundle DECLINES this nest (an unsupported flag combo, e.g. clang auto-par).

Feeding a nest's measured outcome back to re-fuse (Phase 1) or re-granularize (Phase 2) is Phase 4
(:func:`nestforge.optimizers.run_agent_loop`), not here.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from nestforge.build import DEFAULT_COMPILER, BuildOptions
from nestforge.optimizers import (BASELINE_OPT_MODE, DaceOptimizer, ExternalOptimizer, Optimizer, Proposal,
                                  deterministic_optimizers)
from nestforge.tsvc import OPT_MODES

#: The default per-nest opt-mode -- the Phase-1 baseline (DaCe simplify + LoopToMap + MapFusion).
DEFAULT_OPT_MODE = BASELINE_OPT_MODE


def optimize(nest: Optional[object], knobs: Optimizer) -> Optional[Proposal]:
    """Apply a chosen knob bundle to ONE nest, returning its build recipe.

    ``knobs`` is an :class:`~nestforge.optimizers.Optimizer` (the knob bundle -- e.g. a
    :class:`DaceOptimizer` or :class:`ExternalOptimizer` from :func:`optimization_choices`). Returns a
    :class:`Proposal` recipe, or ``None`` when the bundle declines this nest (unsupported flag combo).
    Nothing compiles here -- the arena measure path consumes the recipe.
    """
    if not isinstance(knobs, Optimizer):
        raise TypeError(f"knobs must be an Optimizer knob bundle, got {type(knobs).__name__}")
    return knobs.propose(nest)


def optimization_choices(
    compilers: Sequence[str] = (DEFAULT_COMPILER, ),
    opt_modes: Sequence[str] = OPT_MODES,
    external: Sequence[Tuple[str, str, str]] = (("c", "gnu", "gcc"), ),
    fp_modes: Sequence[str] = ("strict-ieee", ),
    cost_models: Sequence[str] = ("cheap", "default")
) -> List[Optimizer]:
    """The knob-bundle grid the agent picks from for one nest, WITHOUT mutating anything.

    Non-mutating (each bundle is just a recipe generator), so the agent can read the whole choice set
    -- each optimizer's ``.name`` describes its knobs -- before committing to one via :func:`optimize`.
    The Phase-3 analog of ``enumerate_fusions`` / ``offload_candidates``. Delegates to
    :func:`nestforge.optimizers.deterministic_optimizers`; widen the axes for a deeper sweep, or
    construct a custom :class:`DaceOptimizer` / :class:`ExternalOptimizer` directly for a finer knob
    (codegen implementation, vectorization config).
    """
    return deterministic_optimizers(compilers=compilers,
                                    opt_modes=opt_modes,
                                    external=external,
                                    fp_modes=fp_modes,
                                    cost_models=cost_models)


__all__ = [
    "DEFAULT_OPT_MODE",
    "optimize",
    "optimization_choices",
    # knob bundles + recipe (from nestforge.optimizers)
    "Optimizer",
    "Proposal",
    "DaceOptimizer",
    "ExternalOptimizer",
    "OPT_MODES",
    "BuildOptions",
]
