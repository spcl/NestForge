---
name: phase4-feedback
description: Phase 4 of the nest-forge 4-phase optimizer — change granularity from measured feedback and loop. Read each round's Outcome (built? bit-exact? how fast?), request a different fuse/fission, re-measure, and stop when a round yields no improvement. Closes the loop back to Phase 1. Use after Phase 3 has measured the current granularity.
---

# Phase 4 — measurement feedback loop

Phases 1–3 fixed a granularity, externalized it, and tuned each nest. Phase 4 reads the **measured**
result and requests a *different* fuse/fission — then re-runs the earlier phases on the changed
granularity. This is the arrow back to Phase 1. The decision is driven by measurement, not estimate, so
the loop needs to see results: an `Outcome` per round.

```python
from nestforge.optimizers import Outcome   # proposal, ok (bit-exact), median_us, error
```

## The two rules

```python
from nestforge.feedback import best_outcome, improved

best_outcome(outcomes)        # fastest bit-exact (ok=True) Outcome so far -- the selection rule
improved(prior, candidate)    # did this round beat every prior bit-exact outcome? -- the stop rule
```

`best_outcome` never picks a candidate that lost the correctness gate (`ok=False`) — a wrong
granularity never wins on speed. `improved` is the termination rule: **a round that does not improve
ends the loop.**

## The loop

```python
from nestforge.feedback import run_feedback_loop

res = run_feedback_loop(sdfg, measure)   # measure: SDFG -> Outcome (build + validate bit-exact + time)
res.best        # the winning Outcome
res.sdfg        # the granularity it was measured at (snapshotted at each new best)
res.rounds      # rounds run before it stopped
```

Each round applies one granularity move, re-measures, and stops when the move stops helping or the
granularity hits its fixed point. The default lever is **fuse** — re-enumerate the Phase-1 fusion moves
and apply one, coarsening from a fine granularity:

```python
from nestforge.feedback import default_fuse_step
default_fuse_step(sdfg)       # apply ONE re-enumerated fusion move; False at the fixed point
```

Pass `apply_move=` a fission step (`nestforge.fusion.map_fission_moves`) to search the other direction.
`measure` is the caller's build+validate+time step, so CI drives the whole loop with a **fake measure
and no compiler** — the loop plumbing (measure → decide → stop) is what breaks, and none of it needs a
model.

`max_rounds` is the hard bound: a loop that never stalls must still terminate, or a CI job hangs instead
of failing.

## Per-nest inner loop

Phase 4 changes granularity (an outer loop over SDFG states). The per-nest **inner** loop — propose a
build recipe, measure, observe, propose again until it stops — is `run_agent_loop`:

```python
from nestforge.feedback import run_agent_loop, AgenticOptimizer
```

An `AgenticOptimizer` reads the `Outcome` of the round it just proposed (`observe`) and proposes the
next. `run_agent_loop` enforces the round bound, so a buggy agent that never says `stop` fails a test
instead of hanging.

## Correctness

Every granularity move is value-preserving (legality-gated + fuzzed bit-exact) and `measure`
re-validates each round, so Phase 4 changes only *how fast* the program runs, never its result.

## Done

Phase 4 closes the cycle: Phase 1 (fuse/fission) → Phase 2 (offload granularity) → Phase 3 (optimize
each nest) → Phase 4 (re-fuse/re-granularize from measurements) → back to Phase 1.
