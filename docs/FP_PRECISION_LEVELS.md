# FP-precision levels

The FP-mode axis of the nest-forge arena, generalized to a **4-rung graded ladder** (level 0
strictest → level 3 fastest) swept as a matrix crossed with the vectorizer cost-model axis. The
machine-usable source of truth is [`nestforge/perf/flags.py`](../nestforge/perf/flags.py)
(`FP_LEVELS`, `FP_ATOL`, `flag_matrix`); this document is the rationale.

Flag lists below are the **FP-mode component only** — no `-O3`/`-march`/`-tp`/`-fPIC`/`-shared`
(added by `base_flags`) and no vectorizer flags (added by `cost_flags`). Every list here was
**verified to compile** on the local gcc 15.2, clang 21.1, nvc 26.3 and icx/ifx 2026.1.

## Why even level 0 is not bit-exact

The ladder validates against a **numpy float64** oracle, which is itself *not* bit-reproducible:
`np.sum`/`np.mean` use pairwise (128-block) summation, `np.dot`/`np.matmul` dispatch to BLAS (its
own FMA + blocked reassociation), and transcendentals (`exp/log/sin/…`) are not correctly-rounded
and differ ≥1 ULP between glibc libm, nvhpc libm and numpy's SIMD loops. So tolerances are
**relative** and cover O(N)·eps accumulation in reductions; element-wise `{+,−,*,/,sqrt}`
straight-line kernels match level 0 to ~1 ULP.

## The ladder

| L | Name | Guarantee vs the numpy fp64 oracle | atol | gnu | llvm | nvidia | intel |
|---|---|---|---|---|---|---|---|
| 0 | `strict-ieee` | Every op correctly rounded RNE at declared type; **no FMA**, no reassociation, correctly-rounded div/sqrt, denormals preserved. | `1e-14` | `-ffp-contract=off -fexcess-precision=standard` | `-ffp-contract=off` | `-Kieee -Mnofma` | `-fp-model=strict` |
| 1 | `contract-fma` | As L0 plus `a*b+c` may fuse to one FMA (single rounding), across statements. No reassociation, no approximation, no value assumptions. Accumulation error grows ~O(N)·eps. | `1e-13` | `-ffp-contract=fast -fexcess-precision=standard` | `-ffp-contract=fast` | `-Kieee -Mfma` | `-fp-model=precise` |
| 2 | `assume-finite` | Numerically identical to L1 for finite inputs with no meaningful −0.0; frees codegen of exceptional bookkeeping (no errno, non-trapping FP, NaN/Inf guards removed, ±0 ignored). Diverges from L1 only if a NaN/Inf/−0.0 actually occurs. `-fassociative-math` deliberately **not** set. | `1e-13` | `-ffp-contract=fast -fexcess-precision=standard -fno-math-errno -fno-trapping-math -ffinite-math-only -fno-signed-zeros` | `-ffp-contract=fast -fno-math-errno -fno-trapping-math -ffinite-math-only -fno-signed-zeros` | `-Kieee -Mfma` | `-fp-model=precise -ffinite-math-only -fno-math-errno` |
| 3 | `fast-math` | Full unsafe math: algebraic reassociation (reduction-order-dependent), div→reciprocal-multiply, approximate div/sqrt/rsqrt (~12-bit + Newton), FTZ/DAZ denormal flush. Matches the oracle in relative error only. | `1e-5` | `-ffast-math -mrecip` | `-ffast-math -mrecip` | `-fast -Mfma -Mfprelaxed=div,sqrt,rsqrt,recip` | `-fp-model=fast=2 -ftz` |

Monotonicity holds: aggressiveness and tolerance are non-decreasing (`1e-14 ≤ 1e-13 = 1e-13 ≤ 1e-5`),
and each rung's flag set strictly escalates the prior.

### Per-family notes (verified on the local toolchains)

- **nvidia** has only whole-model FP knobs, so `assume-finite` collapses to `contract-fma` numerically
  (there is no per-assumption flag; `nvc` **rejects `-fno-math-errno`**). The matrix dedups the
  duplicate cell, so nvidia sweeps 3 distinct FP levels, not 4.
- **intel** (icx/icpx/ifx) **defaults to `-fp-model=fast`** — a no-flag Intel build is already
  non-reproducible. A bare `-ffp-contract=off` would leave reassociation + reciprocal + FTZ on, so
  every rung sets an explicit `-fp-model` to reset the baseline. This is why Intel is a separate FP
  family even though `icx` is clang-based. `-fp-model=strict` already implies contraction off;
  `-fp-model=precise` keeps FMA on (so it is *not* bit-identical to a no-FMA reference except on
  non-FMA hardware); `-fp-model=fast=2` adds finite-math + FTZ (≡ `-Ofast`).
- **nvidia arch flag** is `-tp=native`, not `-march=native` (handled by `base_flags`).

### Cross-language (Fortran) deltas — applied by `fortran_fp_flags`

- **gfortran, L0–L2:** add `-fno-frontend-optimize` (its front-end optimizer reassociates source at
  `-O`, which `-ffp-contract=off` does not disable). `-fno-math-errno` is dropped (Fortran intrinsics
  never set `errno`).
- **gfortran, L3:** add `-fno-protect-parens` so gfortran reassociates across parentheses like
  clang/nvfortran do (`-ffast-math` does **not** set it).
- **flang** mirrors clang; **nvfortran** mirrors nvc; **ifx** drops `-fno-math-errno` (no errno in
  Fortran) and otherwise mirrors icx's `-fp-model` spellings.

## When to use each level

- **L0 `strict-ieee`** — the reproducible floor and the reference every higher rung is measured
  against. Use for correctness bring-up and differential debugging, and any kernel where
  cross-compiler bit-identity of `{+,−,*,/,sqrt}` matters. This is the rung the cross-language job
  compiles at so C and Fortran are bit-exact against each other.
- **L1 `contract-fma`** — the single most impactful portable relaxation, expressible identically on
  all four vendors. FMA halves rounding error on multiply-add chains (often moving results *toward* a
  BLAS-backed oracle) while ~doubling FMA-unit throughput. Default rung for numerically sensitive
  production kernels.
- **L2 `assume-finite`** — same numbers as L1 for finite data, but real speedups from branch
  elimination and vectorization of guarded loops. Use when inputs are provably finite with no
  meaningful signed zero. Numerically == L1 on nvidia (no per-assumption flags); the perf win lands
  on gcc/clang/intel.
- **L3 `fast-math`** — the throughput tier. Reassociation, reciprocal-multiply, approximate
  div/sqrt and FTZ/DAZ are bundled because nvidia cannot portably separate them. Bits vary with
  vectorization; validate per-kernel against a conditioning estimate, not the `1e-5` constant —
  cancellation-heavy kernels legitimately exceed it and should be pinned lower.
