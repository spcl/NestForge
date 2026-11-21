# FP modes and vectorization

[Overview](../README.md) · related: [2 Moves](stages.md#stage-2-moves), [5 Kernels](stages.md#stage-5-kernels),
[6 Variants](stages.md#stage-6-variants)

By default the compiler vectorizes: stage 5 emits the CPF loops, and stage 6 sweeps compiler, FP mode and the
compiler's vectorizer cost model over each kernel's CPF unit (`nestforge/build/flags.py` defines both axes). The
`vectorize-doall` move ([stage 2](stages.md#stage-2-moves)) vectorizes one doall on request instead: DaCe
rewrites its body into SIMD tiles for one ISA, and the kernel holding it is checked by the whole program, since its
C++ body has no NumPy oracle.

## FP modes

`FP_LEVELS` lists three modes, strictest first. Every mode is graded against the NumPy oracle by the same rule,
HPCAgent-Bench's: its comparator at the band of each output's precision (fp64 rtol 1e-9, atol 1e-11), with the
reassociation floor `eps_acc * sqrt(l)` at the output's accumulation length `l` ([Grading](#grading)). A mode is
kept when the bench would accept its result, so a faster mode wins only if the final grade still passes.

| Mode | GNU | LLVM | oneAPI |
|---|---|---|---|
| `strict-ieee` | `-ffp-contract=off -fexcess-precision=standard` | `-ffp-contract=off` | `-fp-model=strict` |
| `contract-fma` | `-ffp-contract=fast -fexcess-precision=standard` | `-ffp-contract=fast` | `-fp-model=precise` |
| `fast-math` | `-ffast-math -mrecip` | `-ffast-math -mrecip` | `-fp-model=fast=2 -ftz` |

oneAPI compilers default to `-fp-model=fast`, so every mode sets an explicit model. GPU variants use only
`strict-ieee` and `contract-fma`.

## Grading

`nestforge/build/grading.py` grades a kernel's outputs with the bench's `compare_arrays`, band
(`precision.tolerance_band`) and grading knobs (`grading.strict_small`, `grading.relative_atol`), as `grade_outputs`
grades a program. The bench derives `l` per program output from the BenchSpec; a kernel's outputs are often
transients no spec names, so `nestforge/ir/accumulation.py` derives it from the kernel's SDFG at the checked sizes:
the writes that read their own target (directly, through scratch, or as a WCR) over the chains they extend, one per
element and one per initialization. A matmul gets `K`, a sum over `N` gets `N`, an elementwise kernel 1; an output
computed from scratch takes the scratch's `l`. A writer that cannot be counted (a library node, a while loop) gives
the size of the largest input, the bench's own fallback. When `eps_acc * sqrt(l)` reaches the band, the bench
refuses to grade and the kernel is left to the whole program's check.

## Cost models

`COST_MODELS` holds three settings. `default` keeps the compiler's own model. `cheap` vectorizes less;
only gcc has the knob (`-fvect-cost-model=cheap`), so the other compilers dedup it onto `default`.
`no-vec` turns the vectorizer off as a scalar baseline.
