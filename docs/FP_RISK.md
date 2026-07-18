# `fp_risk` — predicting when fast-math (and parallelism) is numerically dangerous

`fp_risk(kernel) -> {subflag: risk_score, explanation}` is nest-forge's static classifier: given an
extracted nest as a typed SDFG (elementwise tasklets, reduction/WCR nodes, library nodes like
Cholesky/solve/eigh/matmul, loops with known trip counts), it predicts — *before running anything* —
whether fast-math (wholesale, a specific sub-flag, or a parallel reduction) will corrupt the answer. It
**prunes** the arena's compile matrix (skips building variants classified unsafe) and **explains** each
verdict; the empirical max-diff-vs-oracle gate stays the hard safety net.

Central thesis: **fast-math is not one transformation but ~7 semantically distinct sub-flags**, each
with its own failure mechanism, structural signature, and numerical-analysis theory. `fp_risk` reasons
per-sub-flag, not "fast-math" monolithically — green-lighting the safe ones (often `reassoc` on
same-sign reductions, `contract` on dot-products) while disabling only the dangerous ones, instead of
the all-or-nothing `-ffast-math`/`-fno-fast-math` choice.

Two axes share this analysis as the same risk class: **fast-math flags** and **parallel reductions** (a
thread-parallel or SIMD-lane reduction reorders the sum exactly as `-fassociative-math` does — see
`PARALLEL.md` §4). A separate, orthogonal axis — cross-language **operator semantics** (`mod`, integer
division, `**`, NaN in min/max) — is covered in §6; it's not an FP-rounding effect, don't conflate the
two.

## 0. Empirical anchor — gramschmidt

`tests/test_gramschmidt_fma.py` / `examples/demo_gramschmidt_fma.py` compile gramschmidt's compute nest
(two `np.dot` reductions) across FP modes and measure relative error vs the ieee-strict **sequential**
baseline:

| input | ieee-strict-seq | `-ffp-contract=fast` (FMA) | `-O3` auto-vec | `-ffast-math` (reassoc) |
|---|---|---|---|---|
| well-conditioned | 0 | 0 | 0 | 5.2e-16 |
| ill-conditioned (cond ~1e14) | 0 | 0 | 0 | **1.5e-3** |

The result validates and *sharpens* the intuition that "fast-math is dangerous for reductions/solvers":

- **FMA contraction alone is bit-exact** here (can even help accuracy). The danger is `reassoc` —
  it reorders the dot-product accumulation — not `contract`.
- Danger is **gated by the condition number**. Well-conditioned: reassociation shuffles only
  `O(nε)` benign noise (`κ≈1`), rel-err stays at machine epsilon. Ill-conditioned: a near-zero pivot
  `R[k,k]` divides `A[:,k]`, amplifying the reassociated-dot difference by `~1/R[k,k]` — rel-err ≈
  cond·ε ≈ 1e-3. Same kernel, same flag — danger comes purely from input conditioning.

This is the whole theory in one experiment: risk ∝ (reduction reassociability) × (condition number).
Both are readable off the SDFG — the reduction node type, its trip count, and the sign/κ of its inputs.

## 1. The sub-flag taxonomy

`-ffast-math` is a bundle. GCC: `-ffast-math` ⊃ `-fno-math-errno, -funsafe-math-optimizations,
-ffinite-math-only, -fno-rounding-math, -fno-signaling-nans, -fcx-limited-range, -fexcess-precision=fast`,
and `-funsafe-math-optimizations` ⊃ `-fno-signed-zeros, -fno-trapping-math, -fassociative-math,
-freciprocal-math`. Clang implies `-fno-honor-infinities, -fno-honor-nans, -fapprox-func,
-fno-math-errno, -ffinite-math-only, -fassociative-math, -freciprocal-math, -fno-signed-zeros,
-fno-trapping-math, -fno-rounding-math, -ffp-contract=fast`, defines `__FAST_MATH__`, and — critically —
**links `crtfastmath.o` unless `-shared` or `-mno-daz-ftz`** (the FTZ/DAZ hazard, §1.3).

The cleanest per-effect vocabulary to key on is LLVM's fast-math-flags:

| LLVM flag | GCC equivalent | meaning | fp_risk axis |
|---|---|---|---|
| `nnan` | part of `-ffinite-math-only` | assume no NaN; result undefined if present | finite-math |
| `ninf` | part of `-ffinite-math-only` | assume no ±Inf | finite-math |
| `nsz` | `-fno-signed-zeros` | sign of zero insignificant | signed-zero |
| `arcp` | `-freciprocal-math` | may use reciprocal for division | reciprocal |
| `contract` | `-ffp-contract=fast` | may fuse `a*b+c` into FMA (one rounding) | contract |
| `afn` | `-fapprox-func`/`-funsafe-math` | approximate `sin/log/sqrt/…` | approx-func |
| `reassoc` | `-fassociative-math` | "may **dramatically change results**" | reassoc |
| `fast` | `-ffast-math` | all the above | — |

A kernel can be perfectly safe under `reassoc` yet catastrophically broken under `nnan`, or vice-versa,
so `fp_risk` reports against all seven independently.

### 1.1 Reassociation (`reassoc`) — the summation footgun

Real addition is associative; FP addition is not. `-fassociative-math` lets the compiler reorder
additions and, most importantly, **vectorize reductions** by splitting one sequential accumulator into
several partial sums. Forward error of a naive length-`n` sum is `O(n·ε)`; pairwise/tree is
`O(log n · ε)`; compensated (Kahan) is `O(ε)`. Reassociation silently moves you *up* that ladder.

The catastrophe is that it **breaks compensated summation** (Simon Byrne, *Beware of fast-math*). Kahan
carries a compensation term `c = ((s + y) - s) - y`; reassociation proves `c` is identically zero and
deletes it, so the loop "optimizes" to naive summation. The same collapse destroys every error-free
transformation (TwoSum, Fast2Sum, Dekker TwoProduct, Neumaier) — each computes a round-off residual that
is algebraically zero. These idioms are exactly what §4.2 detects. Byrne: "compensated arithmetic is
often used to implement core math functions… allowing the compiler to reassociate inside these can give
**catastrophically wrong answers**."

### 1.2 `-ffinite-math-only` (`nnan`+`ninf`) — deletes your NaN/Inf guards

Tells the compiler no op produces or consumes NaN/±Inf, so `isnan(x)`, `isinf(x)`, and the
self-comparison `x != x` fold to constant `false` and their guarded bodies are deleted. Empirically the
**#1 real-world fast-math bug** (Cantera #1155; the long-running LLVM-dev "should isnan be optimized out
in fast-math?" thread). Breaks input validation and the common HPC/ML idiom of using NaN as a sentinel
for missing data. Failure is **silent** — no diff on well-formed inputs; the guard only mattered on the
bad inputs it was meant to catch. If any `isnan`/`isinf`/`x!=x`/NaN-sentinel guard is reachable in the
kernel, `nnan`/`ninf` risk = CRITICAL.

### 1.3 Flush-to-zero / denormals-are-zero (FTZ/DAZ) — breaks Sterbenz, acts non-locally

Fast-math sets the FTZ (flush denormal results to 0) and DAZ (treat denormal inputs as 0) bits in the
x86 `MXCSR`. **Local danger**: breaks Sterbenz' lemma (`y/2 ≤ x ≤ 2y ⟹ fl(x−y)=x−y` exactly — false
without gradual underflow), so compensated arithmetic, some Newton iterations, and convergence tests can
silently fail. **Non-local danger (the spooky one)**: FTZ/DAZ is thread-global process state, and
`-ffast-math`/`-funsafe-math` link `crtfastmath.o`, a static constructor that sets those MXCSR bits at
program start — so **simply loading a shared library built with fast-math can change the results of
completely unrelated IEEE-strict code in the same process** (LLVM #81204; Clang now disables FTZ/DAZ for
shared libraries by default). For nest-forge this is why fast-math node libraries must be built with
care, and why the differential oracle often *cannot* observe this effect in isolation. Ties directly to
`PARALLEL.md` — a fast-math `.a` linked into the multi-library driver can perturb every other library.

### 1.4 `-fno-signed-zeros` (`nsz`) — breaks branch-on-sign, `atan2`, complex branch cuts

IEEE-754 distinguishes `+0.0`/`−0.0`; `nsz` says the sign of zero is insignificant. The sign bit of
zero is load-bearing for discontinuous functions: `x/0.0` vs `x/-0.0` are opposite-signed infinities;
`atan2(0,-0)≈π` but `atan2(0,0)=0` (a whole quadrant); `sqrt(complex(-1,0))` and `sqrt(complex(-1,-0))`
pick different branch sheets; any `if (signbit(x))` at a `±0` value becomes nondeterministic. Elevate
`nsz` risk when the kernel has `atan2`, complex `sqrt`/`log`/`pow`, `copysign`/`signbit`, reciprocals of
possibly-zero denominators, or a sign branch on a value that can reach `±0`.

### 1.5 FMA contraction (`contract`) — one rounding vs two (can help *or* hurt)

`contract` fuses `a*b+c` into a single FMA (one rounding instead of two). **Ambiguous sign**: it
*improves* dot-products / Horner / Newton residuals (smaller error), but *breaks* code that relied on
the two-rounding result — especially **`a*b − c*d` orientation / determinant / sign predicates**
(Bartels–Fisikopoulos–Weiser 2022; Shewchuk robust predicates): fusing into `fma(a,b,−c*d)` loses the
anticommutativity of the difference, so swapping two inputs no longer flips the predicate sign,
breaking the geometric invariant. So `contract` risk is LOW for accumulation (may even help) but HIGH
for `a*b − c*d` sign tests. gramschmidt (§0) confirms the benign case: contract alone was bit-exact.

### 1.6 The remainder: `arcp`, `afn`

`-freciprocal-math` (`arcp`): `x/y → x*(1/y)`, loses ~1 ulp and can change sign/overflow near zero —
MEDIUM where a division result feeds a comparison. `-fapprox-func` (`afn`)/`-funsafe-math`: allows
`sqrt(x)*sqrt(x)→x`, `exp(x)*exp(y)→exp(x+y)`, `sin/cos→tan`, and lower-accuracy libm — MEDIUM.

## 2. Condition-number theory — the "why"

`fp_risk` can be static because the danger magnitude is governed by **condition numbers** and **trip
counts**, both readable off the SDFG.

### 2.1 Summation error bounds (Higham; Blanchard–Higham–Mary 2020)

Standard model `fl(a op b) = (a op b)(1+δ), |δ| ≤ u`. With `γ_n = nu/(1−nu)`:

- Recursive (sequential) sum: `|ŝ − s| ≤ γ_{n−1} Σ|xᵢ|`  — grows **linearly** in `n`.
- Blocked (block `b`): `(b + n/b − 2)u + O(u²)`.
- Pairwise/fan-in: `≈ (log₂ n) u Σ|xᵢ|` — the `O(log n · ε)` result.
- Compensated (Kahan): `[2ε + O(nε²)] Σ|xᵢ|` — leading term **independent of `n`**.

**Condition number of summation**: `κ = Σ|xᵢ| / |Σxᵢ|`. Relative forward error ≈ (method backward
error) × `κ`. Three consequences that drive the reassociation rule:

1. All `xᵢ` same sign ⟹ `κ = 1` — perfectly conditioned; reassociation only shuffles benign `O(nu)`
   noise. **`reassoc` risk LOW.**
2. Mixed signs with cancellation ⟹ `κ ≫ 1` (e.g. `κ > 10⁸` for FP64). Reassociation demoting a
   Kahan/pairwise reduction to naive loses `~log₁₀ n` digits. **`reassoc` risk HIGH, scaling with `κ`
   and `n`.**
3. Very large `n` at low precision: `nu·κ` can exceed 1 — "not even a correct sign is guaranteed."

Both `κ` (from a representative input sample, or static sign analysis of the reduction inputs) and `n`
(the loop trip count) are available to `fp_risk`. gramschmidt §0 is case (2): cond ~1e14 → rel-err ~1e-3.

### 2.2 Catastrophic cancellation = condition number of subtraction

Condition number of `x − y` is `(|x|+|y|)/|x−y|` — blows up as `x → y`. Benign cancellation subtracts
*exact* quantities (harmless); catastrophic cancellation subtracts *already-rounded* quantities,
exposing their error (Goldberg). Cancellation is not caused by fast-math, but reassociation and
contraction **change where the cancellation lands and how much prior error it exposes** — so
cancellation-heavy kernels are fast-math-fragile.

### 2.3 Why solvers are the most dangerous library nodes

For `Ax=b`, a backward-stable solver's forward error is `‖Δx‖/‖x‖ ≲ κ(A)·u`. Three amplification
channels make Cholesky/LU/`solve`/`eigh`/triangular-solve high-risk: (1) `κ(A)` multiplies any tiny
reassociation/contraction perturbation, and `κ(A)` is unknown at analysis time — so assign solver nodes
a **high prior**; (2) **pivoting decisions can flip** — LU partial pivoting picks `argmax|aᵢⱼ|` from
Schur-complement reductions; reassociating those sums can change the pivot near ties, sending the
factorization down a different, unstable path (a discrete failure the differential may miss on a given
input); (3) iterative refinement / triangular solves accumulate rounding, and refinement's convergence
assumes a bounded residual that reassociation/FTZ can perturb. Rule: **every solver/factorization
library node = HIGH for `reassoc`/`contract`, MED for FTZ**, unless the caller supplies well-conditioning
evidence (SPD, diagonally dominant, small `κ(A)`).

## 3. Related tools — static (sound) vs dynamic

| Tool | Static/Dynamic | Sound bound? | Needs | Method | Gives |
|---|---|---|---|---|---|
| **FPTaylor** | Static | Yes (abs+rel) | source→FPCore; ranges | symbolic Taylor + global opt; HOL-Light certs | tight per-expression roundoff bound |
| **Gappa** | Static | Yes | DSL / annotated expr | interval + rewriting; Coq/Rocq certs | certified bounds on rounded expr |
| **PRECiSA** | Static | Yes | PVS subset / FPCore | abstract interp (lfp) + branch&bound | certified over-approx; sound unstable-branch |
| **Rosa/Daisy** | Static | Yes | Scala/Daisy DSL; ranges | affine/interval + SMT; `--rewrite` | sound abs/rel + mixed-precision + rewriting |
| **Fluctuat** | Static | Yes (over-approx) | annotated C/Ada; ranges | zonotope/affine abstract interp | error bounds + provenance (commercial, CEA) |
| **Satire** | Static | Yes | expression DAG | symbolic Taylor + abstraction | rigorous bounds at **>200K operators** |
| **Herbie** | Dynamic (sampling) | No | FPCore expr | random-input sampling vs MPFR; rewrite | more-accurate rewritten expression |
| **Herbgrind** | Dynamic (Valgrind) | No | unmodified binary | shadow high-precision + taint | root-cause FP error sites in binaries |
| **Verrou** | Dynamic (MCA) | No (statistical) | **unmodified binary** | random-rounding CESTAC; delta-debug | # stable significant digits + localization |
| **Verificarlo** | Dynamic (MCA) | No | source + LLVM recompile | MCA on post-opt IR | # significant digits; CI tracking |
| **CADNA** | Dynamic (DSA) | No | source + typed recompile | CESTAC N=3 | significant digits + unstable-branch/cancel flags |
| **EXPLANIFLOAT** | Dynamic | No | executable | double-double condition numbers + log oracle | flags instability; 80%/96% prec/recall |
| **FLiT / pLiner** | Dynamic (differential) | No | source ×N configs | vary compiler/flags/arch, diff | detects+localizes flag-induced divergence |

Key facts for nest-forge:

- **Sound static bounds are conservative** — they bound worst-case error but do not say whether
  *reassociation specifically* changes the answer. The reassociation-safety argument (FPTaylor/Satire):
  the exact real value is reassociation-invariant, so if a tool certifies bound `B₁` for ordering `O₁`
  and `B₂` for `O₂`, then `|fl_{O1} − fl_{O2}| ≤ B₁ + B₂`; a tiny bound ⟹ all orderings agree to ~2×
  that bound ⟹ reassociation-safe. Satire is the only static tool near whole-kernel scale (>200K ops);
  FPTaylor is the tight per-expression oracle. Both require straight-line code (loops unrolled, no
  data-dependent control flow).
- **Monte-Carlo Arithmetic needs no oracle**: Verrou/Verificarlo/CADNA run the *same* program several
  times, perturbing each op with random rounding, and treat inter-run variance as the error estimate —
  `s = −log₁₀(σ̂/|μ̂|)` significant digits. **Verrou is the ideal optional backend for nest-forge**: a
  Valgrind tool, so it needs no recompile — it runs the unmodified ieee-strict binary. It models the
  `reassoc`/`contract`/`arcp` rounding class but **not** `nnan`/`ninf`/`nsz`/FTZ semantics (those are
  branch/semantic changes, not rounding), so those stay static-only detections.
- **FPCore/FPBench** is the interchange format tying the ecosystem together; `fp_risk` should emit a
  suspect sub-expression as FPCore to route to FPTaylor/Satire (sound bound) or Herbie (rewrite).

## 4. `fp_risk` implementation

### 4.1 Rule table: pattern → sub-flag → severity → theory

| # | SDFG pattern | sub-flag | severity | mechanism | citation |
|---|---|---|---|---|---|
| R1 | compensated-sum / EFT idiom (Kahan/Neumaier/TwoSum/Fast2Sum; §4.2) | reassoc | **CRITICAL** | compensation term → 0 | Byrne |
| R2 | `Reduce(+)` float, large `n`, mixed-sign / large κ | reassoc | **HIGH** (∝ log n, κ) | naive `nu·κ` vs compensated `u·κ` | Higham; BHM |
| R3 | `Reduce(+)` float, provably same-sign (κ≈1) | reassoc | **LOW** | only shuffles `O(nu)` noise | Higham |
| R4 | `Reduce(+)` fed by a subtraction (cancellation into the sum) | reassoc,contract | **HIGH** | reassoc moves cancellation; κ explodes | §2.2 |
| R5 | sign/branch on `a*b − c*d` (orientation/determinant) | contract | **HIGH** | FMA loses anticommutativity → sign flip | Bartels; Shewchuk |
| R6 | `isnan`/`isinf`/`x!=x`/NaN-sentinel guard reachable | nnan,ninf | **CRITICAL** (silent) | guard folded false, body deleted | Byrne; Cantera #1155 |
| R7 | `x−x`, `x+0.0`, Inf arithmetic assumed absent | nnan,ninf | **MED** | folds invalid for NaN/Inf | Krister |
| R8 | branch on signbit / `x<0` where `x` can be `±0`; `copysign` | nsz | **HIGH** | sign-of-zero lost → wrong branch | Kahan branch-cuts |
| R9 | `atan2`, complex `sqrt`/`log`/`pow`, branch cuts | nsz | **HIGH** | wrong quadrant / sheet | Kahan |
| R10 | `1/x` or `a/b`, possibly-zero denom feeding sign/compare | nsz,arcp | **MED** | ±Inf sign flip; reciprocal −1 ulp | Krister |
| R11 | solver library node (Cholesky/LU/solve/eigh/tri-solve) | reassoc,contract | **HIGH** | κ(A) amplifies; pivot flip; refinement | §2.3; Higham |
| R12 | iterative/convergence loop; Sterbenz-exact subtraction; subnormal-range | ftz/daz | **HIGH** | Sterbenz breaks; no gradual underflow | Byrne; Sterbenz |
| R13 | kernel ships in a shared library / DSO (any FP) | ftz/daz | **HIGH (non-local)** | `crtfastmath.o` sets process-global MXCSR | Byrne; LLVM #81204 |
| R14 | transcendental identity reachable (`exp*exp`, `sqrt*sqrt`, `sin/cos`) | afn | **MED** | approx funcs; range/accuracy loss | LLVM afn |
| R15 | dot-product / matmul / Horner accumulation, no sign predicate | contract | **LOW** | FMA usually *improves* accuracy | §1.5 |
| **R16** | **parallel reduction** (`reduction(+:s)` / WCR partial-sum tree) | reassoc (parallel) | **= R2/R3** | thread-count-dependent sum reorder | `PARALLEL.md` §4 |

R16 is the parallelism bridge: a parallel reduction scores exactly like R2/R3 (same-sign → LOW,
cancelling → HIGH) — it's a reassociation whose order depends on thread count.

### 4.2 Detecting the compensated-summation / EFT idiom structurally

Every compensation/EFT has a sub-expression **identically zero in exact real arithmetic** but nonzero in
FP (it captures the round-off tail); reassociation collapses it. Detection: (1) find loop-carried scalar
accumulators (candidate `sum` `s`, candidate `compensation` `c`) via the SDFG's loop-carried dependence
edges; (2) symbolically expand each accumulator tasklet over the loop-body dataflow (treat `+ − × /` as
exact — a read-only `sympy` pass is fine for analysis) and test whether the term the code keeps
simplifies to 0; (3) cheap syntactic fallback matches the EFT triangles — Kahan `c=(t−s)−y` with
`t=s+y`, Fast2Sum `err=b−(s−a)`, FMA product-error `e=fma(a,b,−(a*b))`; (4) recognize a balanced
reduction tree (already `O(log n)`, reassoc-tolerant) vs a linear compensated chain (not). Tag each
reduction with `idiom ∈ {naive, pairwise, kahan, neumaier, twosum, …}` so R1/R2/R3 fire correctly.

### 4.3 Static risk score per sub-flag

`risk_f = clamp[0,1]( max over matched rules p of severity(p)·confidence(p)·magnitude(p) )`, with
`explanation = matched rules + node ids + citations`. Magnitude from graph-readable quantities:

- **reassoc**: `0.5·(log₁₀ n / 16) + 0.5·cond_penalty`; `cond_penalty` from `κ = Σ|xᵢ|/|Σxᵢ|` mapped
  `clamp(log₁₀ κ / 8)` if samplable, else static sign analysis — same-sign `0.1`, mixed `0.6`,
  subtraction-fed `0.9`. A detected EFT idiom overrides to `1.0`.
- **nnan/ninf**: binary — any reachable `isnan`/`isinf`/sentinel guard ⟹ `1.0`, else `~0.1`.
- **nsz**: sign-sensitive ops (`atan2`, complex `sqrt`/`log`, `signbit`, `1/x` near 0) → `0.7`; a
  *branch* on the sign of a possibly-`±0` value → `0.9`.
- **contract**: `a*b − c*d` sign predicate → `0.8`; plain accumulation → `0.2`.
- **ftz/daz**: iterative loop / Sterbenz subtraction / subnormal-range → `0.7`; always OR-in a standing
  `0.4` shared-library advisory (R13, non-local).
- **arcp, afn**: `0.3–0.5` where a division/transcendental feeds a comparison / tight-accuracy path.

### 4.4 Combining the static score with the differential oracle

The arena already compiles ieee-strict AND the variant and compares max-diff-vs-numpy. Per output:
`d_strict` (intrinsic algorithmic error, *independent* of the flag), `d_fast`, and
`Δ = d_fast − d_strict` (the flag-induced degradation). **Always subtract `d_strict`** — comparing
`d_fast` to numpy alone conflates conditioning with the fast-math effect.

| static risk_f | Δ on test inputs | verdict | action |
|---|---|---|---|
| HIGH | large | **CONFIRMED** | report subflag + Δ + rule + citation; disable `f` |
| HIGH | small | **LATENT / input-dependent** | test inputs miss the ill-conditioned regime (κ small for this sample) — keep the warning; suggest an adversarial input (max-κ sign pattern, near-cancellation). *This is exactly gramschmidt §0: well-conditioned Δ≈0 hides the ill-conditioned Δ≈1e-3.* |
| LOW | large | **UNMODELED** | rule gap or an effect the diff can't attribute (FTZ-global, unexpected FMA) — check `d_strict`, run Verrou |
| LOW | small | **SAFE** | green-light `f` |

Use `Δ` as a **calibration label**: over the kernel corpus, fit the severity/magnitude weights so
`risk_f` predicts `sign(Δ > tol)` — turning `fp_risk` into a supervised predictor whose ground truth is
the arena's own strict-vs-fast differential.

### 4.5 Optional Verrou backend

Verrou confirms the rounding class with no recompile — it runs the unmodified ieee-strict binary:
`valgrind --tool=verrou --rounding-mode=random --vr-seed=$seed ./kernel` over N seeds →
`s = −log₁₀(stdev/|mean|)` significant digits. Stable ⟹ safe for the reassoc/contract class; unstable ⟹
the kernel is intrinsically ill-conditioned (remediation is Kahan/pairwise, not just disabling a flag),
and `verrou_dd_line`/`verrou_dd_sym` localize the digit loss to the offending SDFG node. Verrou does
**not** model `nnan`/`ninf`/`nsz`/FTZ — keep R6/R8/R9/R12/R13 static-only.

### 4.6 Output schema

```python
fp_risk(kernel) -> {
  "reassoc":  {risk: 0.92, severity: "CRITICAL", status: "confirmed",
               evidence: [{rule:"R1", node:"reduce_7", detail:"Kahan compensation term simplifies to 0"}],
               citations: ["Byrne fastmath", "Higham 2002 §4"]},
  "nnan_ninf":{risk: 1.00, severity: "CRITICAL", status: "predicted",
               evidence:[{rule:"R6", node:"guard_3", detail:"isnan(x) guard will be deleted"}]},
  "nsz": {...}, "contract": {...}, "arcp": {...}, "afn": {...},
  "ftz_daz": {risk: 0.70, severity:"HIGH", note_nonlocal: true},
  "_overall": {verdict: "UNSAFE under -ffast-math",
               recommended_flags: "-fassociative-math OK (same-sign reduction) BUT "
                                  "-fno-finite-math-only -fsigned-zeros -ffp-contract=off required",
               d_strict: 3e-14, d_fast: 2e-3, delta: 2e-3}
}
```

The `recommended_flags` line is the payoff: per-sub-flag reasoning, not all-or-nothing `-ffast-math`.

## 5. Parallelism is a reassociation event

A thread-parallel reduction (`#pragma omp parallel for reduction(+:s)`) or a SIMD-lane reduction splits
one sequential sum into partial sums whose combination order **depends on the thread/lane count** — a
reassociation, scored by R16 exactly like `-fassociative-math`. Consequences (detailed in
`PARALLEL.md`): the result is `OMP_NUM_THREADS`-dependent; stability is measured vs the ieee-strict
**sequential** baseline; a parallel reduction is offered as a variant only if its max-diff vs that
baseline stays under tolerance across the thread counts swept. `fp_risk` supplies the static prior:
same-sign reduction → parallelize freely; cancelling reduction → keep sequential or compensated. This is
why "seq vs par loop-nest opt" lands on the FP axis — sequential opts (unroll, tile, strict-order SIMD)
are order-preserving and always safe; parallel opts are order-changing and carry the reassociation risk.

## 6. Operator semantics are a *separate* correctness axis

`fp_risk` and the arena's max-diff gate reason about **rounding**. A distinct, orthogonal source of
divergence is **operator semantics that differ across languages**. nest-forge follows **numpy semantics
by default** (the oracle is numpy), but it **also ingests Fortran-origin SDFGs** (`fortran-to-sdfg`),
which carry Fortran operator semantics. The repository must **not assume that an operator means the same
thing in Python, C, and Fortran.** A mismatch here is a *wrong value*, not a rounding error — it shows
up as a huge max-diff, and must not be misread as an FP-tolerance problem.

The divergences that matter for the numeric corpus:

| op | numpy / Python | C | Fortran | note |
|---|---|---|---|---|
| **modulo** | `%`/`np.mod`: sign of **divisor** (floored) | `%` (int), `fmod`: sign of **dividend** (truncated) | `MOD`: sign of dividend (trunc); `MODULO`: sign of divisor (floored) | numpy `mod` ↔ Fortran `MODULO`; C needs `((a%b)+b)%b` |
| **integer division** | `//`: floor (toward −∞) | `/`: truncate toward 0 | `/`: truncate toward 0 | differ for negative operands |
| **power `**`** | `x**y`: neg base + frac exp → NaN; int-exp fast path | no `**`; `pow()` (real) or manual `ipow` | `**`: integer-exp = exact repeated multiply; real-exp = `exp(y·log x)` (domain error for neg base) | int-vs-real exponent is a real semantic fork; `0**0` conventions |
| **min/max + NaN** | `np.maximum`/`np.minimum`: **propagate** NaN | `fmax`/`fmin`: **suppress** NaN (return the non-NaN operand) | `MAX`/`MIN`: NaN behaviour processor-dependent | (emit_libnode already ships `__npb_fmax`/`__npb_fmin` that propagate, to match numpy) |
| **int cast / round** | `int()` trunc toward 0; `np.round` half-to-even | `(int)` trunc toward 0 | `INT` trunc; `NINT` half-away-from-zero | rounding-tie convention differs |
| **array base** | 0-based | 0-based | 1-based | SDFG normalizes indices; the translator must too |

Design consequence: **the operator's intended semantics travel with the SDFG node** (recorded by the
frontend that produced it — numpy-origin or Fortran-origin), and **each backend lowering realizes that
semantics in the target language, not the target language's default.** An SDFG `mod` meaning
numpy-floored must emit Fortran `MODULO` (never `MOD`) and a corrected C expression (never bare `%`); an
SDFG `**` with an integer exponent must lower to an exact `ipow`, not `pow(x, (double)e)`. The
translator's existing C helpers (`__npb_fmax`/`__npb_fmin` for NaN-propagating min/max, `__npb_sign`,
`__npb_conj`) are exactly this pattern — per-operator shims making C reproduce numpy semantics instead of
assuming the languages agree. Same discipline for `mod`, integer `//`, and integer `**` across C **and**
Fortran; the arena must classify an operator-semantics mismatch (huge, discrete max-diff, insensitive to
FP mode) distinctly from an FP-rounding divergence (small, FP-mode-sensitive).

## Sources

Sub-flag mechanisms & real bugs — Simon Byrne, *Beware of fast-math* (https://simonbyrne.github.io/notes/fastmath/);
Krister Walfridsson, *Optimizations enabled by -ffast-math* (https://kristerw.github.io/2021/10/19/fast-math/),
*-ffp-contract=fast* (https://kristerw.github.io/2021/11/09/fp-contract/); LLVM LangRef fast-math-flags
(https://llvm.org/docs/LangRef.html#fast-math-flags); Clang FP control
(https://clang.llvm.org/docs/UsersManual.html#controlling-floating-point-behavior);
GCC `crtfastmath.c`; LLVM FTZ/DAZ issue #81204 (https://github.com/llvm/llvm-project/issues/81204);
Cantera #1155 (https://github.com/Cantera/cantera/issues/1155);
signed-zero/branch-cuts — Lishman (https://binhbar.com/posts/2024/04/signed-zeroes-and-complex-literals/),
Kahan *Branch Cuts for Complex Elementary Functions*;
FMA & robust predicates — Bartels/Fisikopoulos/Weiser arXiv:2208.00497, Shewchuk (https://www.cs.cmu.edu/~quake/robust.html).

Condition-number & summation — Blanchard, Higham, Mary, *A Class of Fast and Accurate Summation
Algorithms*, SIAM SISC 2020 (https://eprints.maths.manchester.ac.uk/2704/1/paper.pdf); Higham,
*Accuracy and Stability of Numerical Algorithms* 2nd ed. Ch. 4; Kahan summation & catastrophic
cancellation (Wikipedia); Goldberg, *What Every Computer Scientist Should Know About Floating-Point
Arithmetic* (https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html); Sterbenz lemma (Wikipedia).

Dynamic tools — Verrou (https://github.com/edf-hpc/verrou, HAL:hal-01383417); Verificarlo arXiv:1509.01347
(https://github.com/verificarlo/verificarlo); CADNA (https://cadna.lip6.fr); Parker, *Monte Carlo
Arithmetic* UCLA CSD-970002; interflop (https://github.com/interflop); Sohier et al. arXiv:1807.09655;
Herbie (https://herbie.uwplse.org/, PLDI 2015), Herbgrind (arXiv:1705.10416, PLDI 2018),
Odyssey (https://github.com/herbie-fp/odyssey, arXiv:2305.10599, UIST 2023); FPBench (https://fpbench.org,
NSV 2016); FPChecker (https://github.com/LLNL/FPChecker); Precimonious (Rubio-González et al., SC 2013).

Static/sound analyzers — FPTaylor (https://github.com/soarlab/FPTaylor, FM 2015 / TOPLAS 2018);
Gappa (https://gappa.gitlabpages.inria.fr/, TOMS 2010, arXiv:cs/0701186); PRECiSA (https://github.com/nasa/PRECiSA,
SAFECOMP 2017 / FM 2024); Rosa/Daisy (https://malyzajko.github.io/, POPL 2014 / TACAS 2018);
Fluctuat (Goubault & Putot, VMCAI 2011); Satire (arXiv:2004.11960, https://github.com/arnabd88/Satire, SC 2020).

Fast-math safety / oracle-free — EXPLANIFLOAT arXiv:2503.11884 (ARITH 2025); FLiT
(https://github.com/PRUNERS/FLiT, IISWC 2017) + pLiner + Varity; verified FP compilation arXiv:2509.09019
(2025); on-the-fly instability detection (OOPSLA 2013); Andy Kaylor, *Towards Useful Fast-Math* (LLVM 2024).
