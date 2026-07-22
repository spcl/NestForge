---
name: phase3-optimize
description: Phase 3 of the nest-forge 4-phase optimizer — optimize each externalized nest individually. Pick a knob bundle (representation, compiler, flags, DaCe codegen + vectorization) for one nest and get its build recipe. Use after Phase 2 has externalized the nests, before Phase 4 feeds measurements back.
---

# Phase 3 — per-nest optimization

Phase 1 fixed *fusion* granularity; Phase 2 externalized the chosen nests (each is now an
`ExternalCall`). Phase 3 tunes **one nest at a time**: choose a knob bundle, get the build recipe for
that nest, hand it to the measure path.

## Preconditions

- **Phase 2 has run.** Each nest you tune is already an `ExternalCall`; `optimize` takes that nest,
  not a raw map or loop.
- **Nothing compiles here.** This phase is pure: it returns a `Proposal` (a build recipe) that the
  arena measure path consumes. No toolchain is invoked, so it runs on a machine with no compiler.
- **Reach it as `from nestforge.optimize import optimize`.** The commit function is deliberately NOT
  re-exported at package top level, because the name would bind over the `nestforge.optimize` module.

A knob bundle IS an `Optimizer` — the module contract is "each variant is an optimizer". A
deterministic bundle is one fixed arena cell (a `(opt-mode, codegen, compiler)` DaCe cell, or a
`(language, compiler, fp, cost)` external cell). The agent is one more optimizer under the same
contract.

## Inspect, then commit

```python
from nestforge.optimize import optimization_choices, optimize

for k in optimization_choices():              # non-mutating: the knob-bundle grid for this nest
    print(k.name)                             # each bundle describes its own knobs
recipe = optimize(nest, optimization_choices()[0])   # apply one bundle -> a Proposal (build recipe)
```

`optimization_choices` is the Phase-3 analog of `enumerate_fusions` (Phase 1) / `offload_candidates`
(Phase 2): read-only, so the agent sees every bundle before committing. `optimize(nest, knobs)`
returns a `Proposal` — a recipe the arena measure path consumes. **Nothing compiles here.**
`optimize` returns `None` when the bundle **declines** the nest (an unsupported flag combo, e.g. clang
auto-par) — exactly how a variant is dropped with a reason today.

## The two lanes of knobs

`optimization_choices` fans the arena grid; widen its axes for a deeper sweep:

```python
optimization_choices(compilers=("g++", "clang++"), fp_modes=("strict-ieee", "fast-math"))
```

For a **finer knob than the grid exposes** (DaCe codegen implementation, vectorization config),
construct the bundle directly:

```python
from nestforge.optimize import DaceOptimizer, ExternalOptimizer, BuildOptions

# DaCe lane: opt-mode x compiler x codegen (legacy | experimental) x vectorization
DaceOptimizer("auto-opt", BuildOptions(compiler="g++", codegen_impl="experimental"))
# external lane: numpyto C/Fortran x compiler x fp-precision x cost-model
ExternalOptimizer("c", "gnu", "gcc", fp_mode="strict-ieee", cost_model="default")
```

- **DaCe lane** — `opt_mode ∈ OPT_MODES` (`simplify-parallel` | `canonicalize` | `auto-opt`), compiler,
  `codegen_impl` (`legacy` | `experimental`), and an optional vectorizer config on `BuildOptions`.
- **external lane** — the numpyto-emitted C/Fortran source, a compiler family, an FP-precision rung,
  and a vectorizer cost model. Flags compose once through the arena's own `flags.lane_flags`, so a
  bundle sweeps the identical flag set the full-matrix job does.

`DEFAULT_OPT_MODE` is the Phase-1 baseline (`simplify-parallel`) — the arena's speedup denominator.

## Guardrails

- **Handle `None`.** `optimize` returns `None` when a bundle DECLINES the nest (an unsupported combo,
  e.g. clang auto-par). That is a variant dropped with a reason, not an error — record it, never
  substitute a different bundle silently.
- **Do not compare across FP modes.** Each mode has its own comparison tolerance; a `fast-math` cell
  that beats a `strict-ieee` one has not been shown to be faster at the same accuracy.
- **Never re-derive the parameter order** when binding the built kernel — take `abi_order` from the
  measured cell. See the Phase 2 guardrail.
- A `Proposal` is only a recipe; the measure path builds it, validates it **bit-exact vs the numpy
  oracle**, then times it. A candidate that loses the correctness gate never competes on speed — so
  optimization can only change *how fast* a nest runs, never its result.

## Next

Phase 3 tunes each nest → **Phase 4** feeds the measured `Outcome` back to Phase 1 (re-fuse) or
Phase 2 (re-granularize) — `nestforge.feedback` (`run_feedback_loop`; per-nest inner loop
`run_agent_loop`).
