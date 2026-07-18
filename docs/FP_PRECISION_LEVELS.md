# FP-precision levels

FP-mode axis of the nest-forge arena: a **4-rung graded ladder** (level 0 strictest → level 3
fastest), swept as a matrix against the vectorizer cost-model axis. Source of truth:
[`nestforge/perf/flags.py`](../nestforge/perf/flags.py) (`FP_LEVELS`, `FP_ATOL`, `flag_matrix`);
this doc is the rationale.

Flags below are the **FP-mode component only** — no `-O3`/`-march`/`-tp`/`-fPIC`/`-shared` (added
by `base_flags`) and no vectorizer flags (`cost_flags`). All **verified to compile** on local gcc
15.2, clang 21.1, nvc 26.3, icx/ifx 2026.1.

## Why even level 0 is not bit-exact

The ladder validates against a **numpy float64** oracle, itself *not* bit-reproducible: `np.sum`/
`np.mean` use pairwise (128-block) summation, `np.dot`/`np.matmul` dispatch to BLAS (own FMA +
blocked reassociation), and transcendentals (`exp/log/sin/…`) aren't correctly-rounded and differ
≥1 ULP between glibc libm, nvhpc libm, numpy's SIMD loops. Tolerances are therefore **relative**,
covering O(N)·eps accumulation in reductions; element-wise `{+,−,*,/,sqrt}` straight-line kernels
match level 0 to ~1 ULP.

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

- **nvidia** has only whole-model FP knobs: `assume-finite` collapses to `contract-fma` numerically
  (no per-assumption flag; `nvc` **rejects `-fno-math-errno`**). Matrix dedups the duplicate cell —
  nvidia sweeps 3 distinct FP levels, not 4.
- **intel** (icx/icpx/ifx) **defaults to `-fp-model=fast`** — a no-flag build is already
  non-reproducible. Bare `-ffp-contract=off` would leave reassociation + reciprocal + FTZ on, so
  every rung sets an explicit `-fp-model`. This is why Intel is a separate FP family despite `icx`
  being clang-based. `-fp-model=strict` already implies contraction off; `-fp-model=precise` keeps
  FMA on (not bit-identical to a no-FMA reference except on non-FMA hardware); `-fp-model=fast=2`
  adds finite-math + FTZ (≡ `-Ofast`).
- **nvidia arch flag** is `-tp=native`, not `-march=native` (handled by `base_flags`).

### Cross-language (Fortran) deltas — applied by `fortran_fp_flags`

- **gfortran, L0–L2:** add `-fno-frontend-optimize` (front-end optimizer reassociates source at
  `-O`; `-ffp-contract=off` doesn't disable it). Drops `-fno-math-errno` (Fortran intrinsics never
  set `errno`).
- **gfortran, L3:** add `-fno-protect-parens` so it reassociates across parentheses like
  clang/nvfortran (`-ffast-math` does **not** set it).
- **flang** mirrors clang; **nvfortran** mirrors nvc; **ifx** drops `-fno-math-errno` (no Fortran
  errno), otherwise mirrors icx's `-fp-model` spellings.

## When to use each level

- **L0 `strict-ieee`** — the reproducible floor, the reference every higher rung is measured
  against. Use for correctness bring-up, differential debugging, and any kernel needing
  cross-compiler bit-identity of `{+,−,*,/,sqrt}`. The cross-language job compiles here so C and
  Fortran stay bit-exact.
- **L1 `contract-fma`** — the single most impactful portable relaxation, identical on all four
  vendors. FMA halves rounding error on multiply-add chains (often moving results *toward* a
  BLAS-backed oracle) while ~doubling FMA-unit throughput. Default rung for numerically sensitive
  production kernels.
- **L2 `assume-finite`** — same numbers as L1 for finite data, but real speedups from branch
  elimination and vectorization of guarded loops. Use when inputs are provably finite with no
  meaningful signed zero. Numerically == L1 on nvidia (no per-assumption flags); perf win lands on
  gcc/clang/intel.
- **L3 `fast-math`** — the throughput tier. Reassociation, reciprocal-multiply, approximate
  div/sqrt and FTZ/DAZ are bundled because nvidia can't portably separate them. Bits vary with
  vectorization; validate per-kernel against a conditioning estimate, not the `1e-5` constant —
  cancellation-heavy kernels legitimately exceed it and need pinning lower.
