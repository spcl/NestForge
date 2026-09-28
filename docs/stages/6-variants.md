# Stage 6: Variants

prev: [5 Kernels](5-kernels.md) · next: [7 Feedback](7-feedback.md)

Stage 6 compiles each kernel's current source into variants, checks each against the oracle and keeps the fastest
correct one. It is deterministic, never agent-driven.

- **CPU.** Compiler on PATH (GNU, LLVM, oneAPI) × FP mode (strict-ieee, contract-fma, fast-math) × vectorizer cost
  model.
- **GPU.** Every nvcc on PATH × FP mode (strict-ieee, contract-fma), all with `-arch=native`.

Variants that compile to the same object are timed once. Each CPU variant runs in a forked child and each GPU
variant in a spawned interpreter, so a crash or hang is a recorded result, never a dead harness.
[FP modes](../fp-and-vectorization.md) lists the flags.

| | |
|---|---|
| default | `sweep(name)` |
| code | `nestforge/stages/variants.py`, `nestforge/build/arena.py`, `nestforge/build/flags.py` |
