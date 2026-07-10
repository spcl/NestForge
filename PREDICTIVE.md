# nest-forge — profile-based & offline-predictive modes

Two modes, complementary. Profile = ground truth by measurement. Predictive = rank / prune
without running the full matrix, from static analysis + compile-only reports.

## 1. Profile-based mode (measure)

The current arena, made first-class: for each nest, sweep compiler × flag-combo × FP-mode; compile,
run on AOT data, record **both** time AND max-diff-vs-numpy-oracle per cell; winner = fastest
*correct-for-that-mode* variant. This is the label source the predictive model is validated against.
Already scaffolded in `arena.py` (discover_compilers, FP modes ieee-strict / fast-but-ieee /
fast-math). Persist every cell to SQLite → the training/validation corpus.

## 2. Offline predictive mode

### (A) Predictive compiling — which compiler/flags win, without running

Compile-only (cheap, no execution, no data), parse the optimization reports, rank.

- Emit reports: GCC `-fopt-info-vec -fopt-info-missed`; Clang `-fsave-optimization-record`
  (`-Rpass=loop-vectorize -Rpass-missed=…` → YAML opt-remarks); icx `-qopt-report=3`; nvc `-Minfo`.
- Per-loop features from the report: vectorized? vector width; interleave/unroll factor; **missed-vec
  reason** (dependence, cost-model, alignment, unsupported reduction); FMA formed; register spills.
- nest-forge advantage over generic ML: it **extracted the nest** and knows the iteration space
  (trip counts from the size symbols). So a cheap, explainable analytic score beats a black box:
  `predicted_throughput ≈ Σ_loops (vector_width · trip_count / latency)`, penalize missed-vec,
  spills, remainder loops. Rank compilers by score; only *profile* the top-k → prunes the matrix.
- Related work: NeuroVectorizer (deep-RL predicts vectorization pragmas); RF/FNN/SVM ensembles on
  vectorization features predict speedup within ~15%. Start rule-based on the reports; graduate to a
  model trained on the profile-mode SQLite labels once there is data.

### (B) Predictive numerical safety — when is fast-math dangerous, offline

The user's intuitions match the literature exactly: **reductions** (reassociation-sensitive, error
grows ~ε·n naively), **compensated summation** (Kahan — reassociation *deletes* the correction,
compiling it to naive summation), and **solvers** (conditioning + catastrophic cancellation) are the
dangerous patterns. Two more the literature flags as the top real-world bugs: **`-ffinite-math-only`**
silently removes `isnan`/`isinf`/`x==x` checks (the single most common fast-math bug report), and
**flush-to-zero** (via `-funsafe-math`) breaks Sterbenz' lemma AND is non-local (leaks into code not
compiled with fast-math, even other shared libs).

Per-subflag risk (from the fast-math breakdown):

| flag | breaks | danger |
|---|---|---|
| `-ffinite-math-only` | isnan/isinf/x==x checks removed | HIGHEST (most frequent bug) |
| `-fassociative-math` (reassoc) | Kahan/compensated sums, reduction order, Sterbenz | HIGHEST for reductions/solvers |
| FTZ / subnormals (`-funsafe-math`) | Sterbenz, denormal-dependent code; **non-local** | HIGHEST (insidious) |
| `-freciprocal-math` (x/y→x·(1/y)) | modest accuracy | low |
| `-fno-signed-zeros` | sign-of-zero branches, atan2, complex | medium |
| `-ffp-contract=fast` (FMA) | changes rounding of a·b+c | the pivot (gramschmidt: 17.4 vs 0) |
| `-fno-trapping-math`, `-fno-math-errno` | exceptions/errno | low |

**Safe way to know danger — three signals, cheapest first:**

1. **Static pattern classifier over the SDFG (offline, no run, no oracle).** nest-forge already has
   the node types, so it can *see* the dangerous patterns:
   - a **WCR / Reduce** node → reassociation-sensitive; risk scales with trip count → refuse
     `-fassociative-math` when the reduction feeds a tight tolerance.
   - a **Solve / Cholesky / eigh** libnode → solver pattern → default ieee-strict; fast-math only on
     explicit opt-in + a passing differential.
   - a tasklet AST containing `isnan`/`isinf`/`x==x` → refuse `-ffinite-math-only`.
   - a **compensated-summation idiom** (`c = (t - sum) - y` shape) → refuse `-fassociative-math`.
   - `→ fp_risk(nest) -> {subflag: risk}` — reuses the emitter's own node walk.
2. **Empirical differential (profile-mode, needs the oracle nest-forge already has).** Compile
   ieee-strict AND fast-math, compare max-diff-vs-oracle. `md_fast ≫ md_strict` ⇒ fast-math is
   *empirically* dangerous here. This is already the winner-selection gate (a fast-math variant is
   only chosen if it passes its tolerance) — predictive mode just uses it to *label* nests.
3. **Oracle-free instability score (offline, when no reference exists).** Monte-Carlo Arithmetic —
   Verrou (Valgrind, no recompile), Verificarlo, CADNA (CESTAC/DSA): perturb every FP op with
   stochastic rounding, run a handful of times; large output variance ⇒ ill-conditioned ⇒ fast-math
   dangerous. Condition-number estimation (EXPLANIFLOAT) and Herbie's sampled local-error localize
   *which* expression is unstable; Herbie can even rewrite it to a stable form.

**Design:** predictive `fp_risk` **prunes** the matrix (don't even compile a fast-math variant for a
nest the classifier calls unsafe → cheaper) and **explains** the choice; the profile-mode max-diff
gate stays the hard safety net; optionally wire Verrou as a backend for an oracle-free instability
score. Winner selection stays per-FP-mode so the report shows the accuracy/speed trade-off rather
than silently trading correctness for speed.

## Related work (sources)

- Simon Byrne, *Beware of fast-math* — per-subflag breakdown + dangers.
- Verificarlo / Verrou / CADNA — Monte-Carlo arithmetic instability detection without a reference.
- Herbie / Herbgrind / FPBench — sampled rounding-error localization, stable-form rewriting, benchmark format.
- EXPLANIFLOAT — condition numbers + oracle for rounding/over-underflow detection.
- NeuroVectorizer; RF/FNN/SVM vectorization-speedup predictors; awesome-machine-learning-in-compilers.
