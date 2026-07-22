# Predictive mode

## High-level design

Pick the winning optimizer WITHOUT building them all. The arena sweeps every variant; predictive
mode ranks them and builds only the top pick. The arena still validates that pick, so a wrong
prediction costs time, never correctness.

```
nest --> Predictor.rank(nest, optimizers) --> [(optimizer, score, reason)]  (no compile)
     --> choose top --> arena builds + validates + times it
```

One contract: `Predictor.rank(nest, optimizers) -> List[Prediction]`. Every predictor -- hardcoded
strategy now, cost model later -- implements it, so swapping the policy changes no caller.

## Now: hardcoded strategy (`nestforge/predictive.py::HardcodedStrategy`)

No learned model, no compiling. One fixed policy, the safe correct-and-fast floor:

- **no FP error** -- `fp_mode == "strict-ieee"` (`+2`, the load-bearing clause)
- **-O3** -- every variant carries it (`base_flags`), so it is a constant, not a clause
- **cheap vectorizer cost model** globally -- `cost_model == "cheap"` (`+1`)
- an optimizer that declines the nest scores `-inf`

Deterministic; ties break on optimizer name. This is a deliberate baseline, not the goal.

## Planned: the real cost model

`HardcodedStrategy` is a placeholder for a predictor using any available static signal. Each
becomes a scoring term under the same `Predictor` contract; none requires running the kernel.

### 1. Numerical stability
Analyze the nest for FP risk (reduction depth, catastrophic cancellation, magnitude spread from
the OptArena input spec); let the predicted FP rung follow the risk -- stable kernel tolerates
`fast-math`, cancellation-prone one pinned `strict-ieee`. Feeds the FP clause instead of hardcoding it.

### 2. Opt-report diffing
Compile-`-fopt-info` / `-Rpass=loop-vectorize` (gcc/clang) and `-qopt-report` (icx) emit what the
compiler actually did -- which loops vectorized, at what width, why one didn't. Diff the reports
ACROSS variants of the same nest: the variant whose hot loop vectorized at the widest width, no
"not vectorized: ..." on the hot path, is the predicted winner. Cheap probe (compile only, no run)
-- a middle tier between the free static score and the full timed sweep.

### 3. Instruction scanning (per hardware)
Disassemble each variant's hot object (`objdump -d`) and count what matters on THIS hardware:
- **gather/scatter intrinsics** (`vgather*`/`vscatter*`) -- present iff the compiler vectorized an
  indirect access; cost varies wildly by microarch, scored against a per-hardware table.
- **cache-line / alignment ops** -- aligned vs unaligned move forms (`vmovaps` vs `vmovups`),
  software prefetch (`prefetcht*`), streaming stores (`vmovntps`) -- signal whether the layout is
  cache-friendly.
- **ISA level actually emitted** -- AVX-512 vs AVX2 vs scalar, matched against the host (see
  `nestforge/device_profile.py::host_isas`, over dace's `detect_host_isa`); an ISA the host lacks is a mis-target, not a win.

Per-hardware because the same instruction stream ranks differently on different cores. Reuses the
device profile the vectorizer already characterizes.

### 4. Learned term (last)
Once the static terms exist, a small model over (nest features, variant features) -> measured
time, trained on the arena's own sweep results, can replace the hand-weighted sum. LAST tier --
the static terms above are interpretable and need no training data; the learned model only refines
their weighting.

## How to use

```python
from nestforge.optimizers import deterministic_optimizers, NoOpAgent
from nestforge.predictive import HardcodedStrategy

optimizers = [NoOpAgent(), *deterministic_optimizers()]
predictor = HardcodedStrategy()
winner = predictor.choose(nest=None, optimizers=optimizers)   # build only this one
for p in predictor.rank(None, optimizers):                    # or inspect the whole ranking
    print(p.optimizer, p.score, p.reason)
```

## Status

**DONE** -- `Optimizer`/`Proposal` contract + deterministic variants + `NoOpAgent` (`nestforge/optimizers.py`); `Predictor`/`HardcodedStrategy` (`nestforge/predictive.py`); unit tests + the no-op-agent end-to-end smoke.

**OPEN** -- the four cost-model tiers above, in order (stability -> opt-report -> instruction scan -> learned); a `run_proposal` that hands a chosen `Proposal` to the arena measure path (today the proposal is a recipe the full-matrix job already knows how to build).
