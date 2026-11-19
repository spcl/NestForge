# FP modes and vectorization

[Overview](../README.md) · related: [2 Moves](stages.md#stage-2-moves), [5 Kernels](stages.md#stage-5-kernels),
[6 Variants](stages.md#stage-6-variants)

By default the compiler vectorizes: stage 5 emits the CPF loops, and stage 6 sweeps compiler, FP mode and the
compiler's vectorizer cost model over each kernel's CPF unit (`nestforge/build/flags.py` defines both axes). The
`vectorize-doall` move ([stage 2](stages.md#stage-2-moves)) vectorizes one doall on request instead: DaCe
rewrites its body into SIMD tiles for one ISA, and the kernel holding it is checked by the whole program, since its
C++ body has no NumPy oracle.

## FP modes

`FP_LEVELS` lists three modes, strictest first. Each mode has a relative tolerance against the NumPy
float64 oracle (`FP_RTOL`). `strict-ieee` evaluates in the oracle's order, so only the dtype floor
below applies to it. A reduction is the one exception: CPF emits it as an OpenMP `simd reduction`, which may
reassociate the sum, so a reduction kernel matches its sequential oracle within the dtype floor, not bit for bit.

| Mode | tolerance | GNU | LLVM | oneAPI |
|---|---|---|---|---|
| `strict-ieee` | 0 | `-ffp-contract=off -fexcess-precision=standard` | `-ffp-contract=off` | `-fp-model=strict` |
| `contract-fma` | 1e-13 | `-ffp-contract=fast -fexcess-precision=standard` | `-ffp-contract=fast` | `-fp-model=precise` |
| `fast-math` | 1e-5 | `-ffast-math -mrecip` | `-ffast-math -mrecip` | `-fp-model=fast=2 -ftz` |

oneAPI compilers default to `-fp-model=fast`, so every mode sets an explicit model. `DTYPE_RTOL` adds a per-dtype floor of about one
ULP, combined as `max(mode, dtype)`. GPU variants use only `strict-ieee` and `contract-fma`.

## Cost models

`COST_MODELS` holds three settings. `default` keeps the compiler's own model. `cheap` vectorizes less;
only gcc has the knob (`-fvect-cost-model=cheap`), so the other compilers dedup it onto `default`.
`no-vec` turns the vectorizer off as a scalar baseline.
