# Experiment: Characterizing and Closing the Frontend-Semantics Optimization Gap

## 0. Thesis & falsifiable predictions

**Object.** DaCe emits already-good C++ (`__restrict__` on every array, `-fno-math-errno`, `-O3 -march=native`). A semantically identical loop nest emitted as Fortran and compiled by the *same-family* compiler runs faster on some nests. The residual survives restrict + errno + good flags. This residual is the scientific object.

**Thesis (H1), reframed.** The claim is not that C-the-language cannot express the fast form. The claim is that the *canonical, semantics-preserving* C a naive emitter produces systematically drops information the Fortran frontend supplies for free by language contract (alignment, divisibility, stride, IV/SCEV facts, reduction shape), and that recovering it in C requires targeted annotations a naive emitter omits. Holding the middle-end fixed (optimizer library + backend + version + target) and varying only the frontend that produces the optimizer's input IR, a compute-bound gap persists between the naive C form and a layout-matched Fortran form of the same nest at pinned FP-contract parity; the gap is attributable to specific IR features; and a targeted DaCe C++-emitter change closes a majority of it.

Per-nest, per-backend residual gap (higher = C slower):
```
G(nest, backend) = t_C(fastest correct variant) / t_Fortran(fastest correct variant)
```
measured after the control ladder (§3) neutralizes restrict, math-errno, IV signedness, storage/loop order, MXCSR state, and FP-contract.

**Falsifiable predictions.**
- **P1 (frontend, not middle-end).** `G > 1` persists with shared-backend pairs at FP parity, storage-order parity, and unit-stride-axis parity. *Falsified if:* the gap vanishes once backend, FP-contract, and layout are equalized → it was compiler-brand skew, silent fast-math, or a layout artifact.
- **P2 (compute-bound, vanishes memory-bound).** `G_compute > G_memory` with `G_memory ∈ [0.98, 1.02]`. The compute/memory contrast is the prediction; monotonic slope in AI is a secondary, weaker form (see §6.2, T4). *Falsified if:* `G_memory` is not near 1, or the gap is largest on memory-bound nests → not a vectorization/frontend-IR mechanism.
- **P3 (attributable + closable).** The sign/magnitude of `G` is predicted by a small set of IR features (alignment/`dereferenceable`, divisibility/trip-count/`nsw` SCEV facts, stride-axis form, and — only at the `fast` rung — reduction associativity), and a codegen change emitting the Fortran-like IR form drives `G → 1` on exactly the nests those features flagged. *Falsified if:* `G` correlates with no single IR feature (real but unattributable → downgrade to a measurement paper), or the fix trades nests rather than strictly improving (a cost-model result).

**Prior on mechanism (S3).** Because `__restrict__` is *already emitted*, aliasing is predicted to contribute ≈0 to the residual; if the ablation shows alias-repair contributing a lot, that is a concrete emission bug (restrict not reaching the optimizer as `noalias`), which the `noalias_coverage` gate (§4.5) catches. Divisibility/remainder contributions are predicted to vanish when both sides have compile-time-constant bounds. The residual is expected to live in alignment/`dereferenceable`, stride-axis, and SCEV facts.

**Pre-registration (T1).** A pilot set of 8 kernels, disjoint from the confirmation corpus, sets: the AI split (compute vs memory); the practical-significance threshold (`G > max(1.05, upper-CI of the A-vs-A layout floor)`); the VF-forcing decisions; the Part-B recovery target (≥70% of the attributed category's gap on the held-out fix-evaluation split); and the FP rung the headline is quoted at (`strict-ieee` / contract-off). One **primary endpoint** is named and git-tagged before the confirmation run: *Stratum-1 geomean `G` at strict-ieee, LLVM pair, L2-long size, one-sided, cluster-bootstrapped by idiom family.* Everything else is secondary/exploratory.

---

## 1. Setup

### 1.1 Shared-backend compiler pairs

Two frontends feeding one optimizer *library*:

| Pair | C frontend | Fortran frontend | Shared middle-end library |
|------|-----------|------------------|---------------------------|
| **LLVM** | `clang` | `flang-new` | LLVM `opt`/`llc` — identical `libLLVM.so` |
| **GCC** | `gcc` | `gfortran` | GCC `tree-ssa` + RTL |

Honest framing (S2, S3-Q3): the two frontends configure their `-O3` pipelines in frontend C++ (`PassBuilder` + `PipelineTuningOptions`); the pipelines are **not** byte-identical and no `-mllvm` knob makes them so. The isolation claim is therefore "same optimizer *library/binary*, frontend-configured pipeline," and the §4.1 crux tests IR *content* under a fixed reference pipeline. For GCC no freeze-and-refeed is possible; the GCC pair is **correlational only** and is reported as such — never pooled with LLVM.

**Premise-check first (S1).** flang-new is a young frontend; the folk claim is historically about gfortran/ifort/ifx. Before trusting the flang pair, replicate the effect on gcc/gfortran (and ifx/ifort as an external cross-check, excluded from the same-backend headline per §7 threats). If the effect is gfortran-specific and absent in flang, the LLVM crux is studying a different phenomenon; report per-pair and re-scope. Never pool pairs into one geomean.

### 1.2 Version and target pinning (preflight, mandatory)

Build one LLVM commit; both frontends come from it. Same for GCC.
```bash
LLVM=/opt/llvm-21-pinned
clang --version; flang-new --version          # assert SAME commit hash
readelf -d $(which clang)     | grep libLLVM
readelf -d $(which flang-new) | grep libLLVM  # identical soname+path
gcc --version; gfortran --version             # identical "gcc version" line
gcc -v 2>&1 | grep 'Configured with'          # same configure string
```
Record all four version strings into each result row. Any pair with a mismatched backend revision is dropped, not adjusted (C3).

### 1.3 Flags and target-string resolution

Every compile fixes `-O3 -march=native -mtune=native`. Force `-march=native` parity or replace with an explicit shared feature list:
```bash
clang     -march=native -E - -### 2>&1 | tr ' ' '\n' | grep -E '^-target-feature|^\+' | sort > feat.clang
flang-new -march=native -### -c dummy.f90 2>&1 | tr ' ' '\n' | grep -E '^-target-feature|^\+' | sort > feat.flang
diff feat.clang feat.flang    # MUST be empty; else pin -mcpu=<x> explicitly
```
**Resolve `native` to a concrete CPU string once (B1)** and use it verbatim everywhere, because `llc` and `llvm-mca` do **not** accept `native` and silently fall back to generic (no AVX-512):
```bash
CPU=$(clang -march=native -### -c dummy.c 2>&1 | tr ' ' '\n' | grep -A1 -- '-target-cpu' | tail -1 | tr -d '"')
echo "$CPU"   # e.g. znver4 ; use -mcpu=$CPU / -mcpu=$CPU for llc and llvm-mca
```
Dump the real pipelines for the record (they will differ; do not claim equality — S2):
```bash
clang     -O3 -mllvm -debug-pass-manager ... 2>&1 | grep 'Running pass' > pipe.clang
flang-new -O3 -mllvm -debug-pass-manager ... 2>&1 | grep 'Running pass' > pipe.flang
diff pipe.clang pipe.flang     # EXPECTED non-empty; archived, not forced equal
```

### 1.4 Machine & measurement hygiene

- Frequency: `cpupower frequency-set -g performance`; turbo off (`echo 1 > .../intel_pstate/no_turbo`, or `scaling_min_freq == scaling_max_freq` on AMD/acpi).
- Pinning: `taskset -c 3 <runner>` on an isolated core (`isolcpus=`/`nohz_full=`, SMT sibling idle). Same physical core for every variant of a kernel.
- NUMA: `numactl --membind=0 --cpunodebind=0`; first-touch local. ASLR off.
- **MXCSR/FTZ-DAZ parity (B4):** assert `stmxcsr` is identical at kernel entry in both the C and Fortran harnesses (the Fortran runtime's `_gfortran_set_options` can set FTZ/DAZ differently). Set it explicitly in both trampolines. Denormal-producing kernels otherwise diverge in both bits and speed and silently break the bit-parity gate.
- Warmup + steady state: discard first W=3 timed reps; use the `calloverhead.py` trampoline.
- Repeats: R=31 reps per cell. **Central statistic = minimum with a lower-tail CI** (interference-free, closest to machine capability), plus a shift-function/quantile comparison and a paired permutation test on per-rep differences (T2). Median + IQR reported only as robustness; if median and min disagree in ranking, the machine is noisy — re-run.
- Cache state per AI regime: compute-size cells run warm; memory-size cells `clflush` (or size > LLC) between reps.
- Determinism: `rm -rf .dacecache .pytest_cache perf_results` before authoritative runs; record the exact flag list per cell.
- Run compiled kernels under fork isolation (`nestforge/isolation.py run_isolated`).

---

## 2. Corpus & sizes

### 2.1 Corpus, stratified by measured roofline arithmetic intensity

Kernels from TSVC / TSVC-2.5, PolyBench/C 4.2.1, NPBench. Compute nest-local `AI = FLOPs/bytes` analytically from the SDFG (FLOPs from tasklet ops, bytes from distinct memlet footprints), cross-checked against the machine ridge (`likwid-bench`/STREAM) and against LIKWID `MEM_DP`/`FLOPS_DP`. **Stratify by *measured* AI.**

**Stratum 1 — COMPUTE-BOUND, vectorizable (gap EXPECTED; ≥20 nests).**
- TSVC: `s176` (conv), `s251`/`s1251`, `s211`/`s212`, `s1244`/`s2244`, `s243`, `s271`/`s272`, `s311`/`s312`/`s3111`/`s314`, `s352`, `s1113`, `vpvts`, `vtvtv`, `vdotr`.
- PolyBench: `gemm`, `syrk`, `syr2k`, `doitgen`, `2mm`/`3mm` per-nest, `cholesky` inner, `gemver` compute part.
- NPBench: `gemm`, `mlp`, `softmax`, `azimint_naive` inner reduction, `arc_distance`.

**Stratum 2 — BALANCED (gap MODEST; ≥12 nests).**
- TSVC: `s121`, `s1221`, `s231`, `s232`, `s2711`.
- PolyBench: `gesummv`, `bicg`, `mvt`, `atax`, `fdtd-2d`, `heat-3d`.
- NPBench: `jacobi_2d`, `gemver`, `crc16`.

**Stratum 3 — MEMORY-BOUND CONTROLS (gap must VANISH; ≥12 nests).**
- TSVC: `s000`/`s1`, `s112`, `s1112`, `va`, `vag`/`vas`, `vsumr` large, `s311` at PROF size.
- PolyBench: `jacobi-1d`; `atax`, `bicg` at large N (same source as Stratum 2 — within-kernel paired size test).
- NPBench: `stream`/`vadd`, `cavity_flow` streaming nest, `spmv` (CSR — indirect access).

**Idiom-family tagging (T3).** Near-duplicate idioms (s211/s212/s1244/s2244; the s3xx reductions) are tagged with an `idiom_family` column. Independent idioms number ~10–15, not 44. All corpus-level inference clusters by family; the effective family count is reported.

Target ~44 unique kernels → ~60–70 (nest × size) cells. Reuse across strata by size change only (§2.2) makes "AI vanishes the gap" a within-kernel paired test.

### 2.2 Three sizes per kernel

- **`--size l1` (COMPUTE, NEW).** Working set in L1D (~32 KiB), high reuse. Measures the vectorization signal but *maximizes* remainder-loop and per-call artifacts — used only in conjunction with the call-overhead subtraction (§4) and never as the headline size.
- **`--size l2-long` (COMPUTE HEADLINE, NEW — added per critique).** Mid working set (L2-resident, ~1 MiB, GEMM N=64–128 so `3·N²·8 B ≲ L2`) with *long trip counts / large `ntimes`* so steady-state throughput dominates and remainder overhead is amortized. This is the size the primary endpoint is quoted at.
- **`--size prof` (MEMORY, EXISTING).** Working set > L3, DRAM-bound. Negative control.

All sizes keep the timed kernel ≫ 100× timer resolution. Report *measured* AI next to analytic AI; an L1/L2 cell that spilled is dropped from Stratum 1.

---

## 3. Control ladder

Emit C++ and Fortran from **one** SDFG (`extract.py` → `emit_cpp` and the Fortran emitter) so the source pair is provably the same IR-level computation; `translate.py` emits both from one `Prepared` object with a shared `<key>_fp64` C-ABI symbol.

### 3.1 Storage-order and stride parity (the previously-missing control — L2)

DaCe C++ is row-major with C loop order; Fortran is column-major. Emitting "the same SDFG" to Fortran forces a layout choice, and either choice can fake or hide the effect. **Pin identical storage order in both:** emit Fortran arrays as `bind(c)` explicit-shape with C index order and row-major storage; then verify the innermost memlet stride is 1 on the **same axis** in both `.ll` files. Any nest where the emitters disagree on the unit-stride axis is dropped. Unit-stride-axis parity joins FMA-count parity in the §3.3 admission gate. Without this, "Fortran faster" is confounded with "column-major aligned better on this access pattern."

### 3.2 The rung ladder (each rung a toggled control; build all)

| Rung | C emission | Fortran emission | Isolates |
|------|-----------|------------------|----------|
| R0 baseline | plain C, default FP | plain F90 | naive gap |
| R1 restrict | `__restrict__` all arrays | (non-aliasing by std) | aliasing confound |
| R2 errno | `-fno-math-errno` | `-fno-math-errno` | libm-errno |
| R3 induction | `size_t` **and** `int64_t` IV | `integer(8)` | signed/unsigned IV wrap-UB |
| R4 **FP-contract parity** | `-ffp-contract=off` (headline) **and** `=fast` (reduction rung) | matched contract, `-fno-fast-math`, matched reassoc | silent fast-math |
| R5 alignment | `__builtin_assume_aligned(p,64)` | `!dir$ assume_aligned` / `-falign` | alignment/peeling |
| R6 flag-skew audit | dump `-O3` pipeline | same | pipeline divergence recorded |

**Headline = the gap at R4 contract-off (`strict-ieee`) with the shared-backend pair (L3-fix).** The reduction-associativity axis is reported *only* at R4-`fast` and *separately* (§4, Fig. 2), because at contract-off neither frontend may reorder a reduction, so that axis contributes zero by construction. If `G > 1` at R0 but `G ≈ 1` at contract-off, the gap was FP-model (report as such). If `G > 1` survives R4-off through R6, the remaining variable is the IR the frontend handed the optimizer.

### 3.3 FP-contract parity, layout parity, and correctness precheck (gate before any timing)

Timing rung for the causal headline = **`strict-ieee` / `-ffp-contract=off`**, validated to `atol 1e-14` vs the fp64 numpy oracle, where bit-parity between the two variants is *achievable*. `nestforge/perf/flags.py` already asserts `-fno-frontend-optimize` on every Fortran cell (gfortran reassociates in the frontend at `-O` even under `-ffp-contract=off`); assert its presence per cell.

- **FMA-count parity:** `grep -cE 'vfmadd|vfmsub' s.clang s.flang` equal at the same rung; unequal ⇒ discard the cell.
- **Fast-math-flag IR audit:** grep the pre-vectorizer `.ll` for `fmul contract`, `fadd reassoc`, `fast`, `ninf`, `nnan`. At contract-off both `.ll` files carry **zero** fast-math flags on FP ops. GCC side: `-fdump-tree-optimized`, diff `.FMA`/`__builtin_fma`.
- **Variant-to-variant bit parity:** dump both variants' output arrays; `numpy.array_equal(view_c, view_f)` must be `True` before admission. At contract-off this is achievable; at the `fast`/reduction rung it is **not** (clang and flang contract/reorder different mul-add pairs), so at that rung the gate is replaced by FMA-count parity + a pre-registered ULP bound (L3-fix). Never gate the `fast` rung on `array_equal`.
- **Stride-axis parity (§3.1):** unit-stride axis identical in both `.ll`.
- **MXCSR parity (§1.4):** identical FTZ/DAZ at kernel entry.
- **Oracle gate:** `max_diff_vs_oracle` ≤ per-rung tol (0 ulp at contract-off; documented ULP bound at `fast`). Fastest-correct-variant selection is *within* each rung so a fast-math variant never silently wins a bit-exact comparison.

---

## 4. Instrumentation & attribution

Instrumentation runs on a **shadow build** of the identical timed source, same `-O`/`-march`/FP flags. Timing runs stay flag-clean. Artifacts land in `nest-forge/records/<kernel>/<variant>/`.

### 4.1 The crux experiment: same optimizer library, different frontend IR — with a valid "pristine" definition

Emit each frontend's pre-optimization IR, feed both through one identical reference `opt` pipeline. **The critical correction (L1, B2):** `-disable-llvm-passes` disables only the LLVM pipeline; flang-new lowers Fortran → HLFIR → FIR → LLVM through an **MLIR pass pipeline** (array-copy elision, `SimplifyIntrinsics`, FIR loop versioning, alias-attribute attachment) that runs *before* LLVM IR exists. flang's `.fe.ll` is therefore not comparable to clang's near-AST output unless the MLIR opts are also disabled. Two enforced safeguards:

1. Disable flang's MLIR optimization (verify the correct spelling — likely `-Xflang -disable-llvm-passes`, and additionally suppress the FIR/HLFIR optimization pipeline via the `-mmlir` pass-disabling path); do not rely on `-mllvm -disable-llvm-passes`, which may silently no-op on the frontend flag.
2. **Hard-fail assertion:** the "pristine" `.ll` for *both* frontends MUST contain zero vector ops and zero `!llvm.loop` vectorize metadata. Grep and abort the cell if not. This is what makes the disable actually verified rather than assumed.

```bash
CPU=znver4   # resolved once, §1.3
clang     -O3 -march=native -ffp-contract=off -fno-math-errno \
          -Xclang -disable-llvm-passes -emit-llvm -S s176.dace.cpp -o s176.clang.fe.ll
flang-new -O3 -march=native -ffp-contract=off \
          -Xflang -disable-llvm-passes -emit-llvm -S s176.f90 -o s176.flang.fe.ll

# HARD GATE: pristine == no vectorization already present
for fe in clang flang; do
  if grep -qE '<[0-9]+ x (float|double)>|!llvm.loop' s176.$fe.fe.ll; then
     echo "NOT PRISTINE: $fe" >&2; exit 1
  fi
done

PIPE='default<O3>'   # reference pipeline; see S2 framing below
for fe in clang flang; do
  opt -passes="$PIPE" -mtriple=$(llvm-config --host-target) \
      -print-before=loop-vectorize -print-after=loop-vectorize \
      s176.$fe.fe.ll -S -o s176.$fe.opt.ll 2> s176.$fe.vec.beforeafter.ll
  llc -O3 -mcpu=$CPU s176.$fe.opt.ll -o s176.$fe.s
done
```
`opt`/`llc` are the same binaries for both inputs. **Framing (S2):** `default<O3>` is a *reference* pipeline neither shipped driver runs verbatim; the crux tests IR *content* under a fixed reference optimizer, and its validity as a causal statement about the timed result is established by the §4.6 crux↔timing correlation criterion, not asserted.

### 4.2 Optimization remarks (did the pass fire, and why not?)

Both frontends route through the same LLVM remark infrastructure → diffable YAML.
```bash
clang     -O3 -march=native -ffp-contract=off -fsave-optimization-record \
          -foptimization-record-file=$OUT/opt.clang.yaml \
          -Rpass=loop-vectorize -Rpass-missed='loop-vectorize|slp-vectorize' \
          -Rpass-analysis='loop-vectorize' -gline-tables-only -c kernel.c 2> $OUT/remarks.clang.txt
flang-new -O3 -march=native -ffp-contract=off -fsave-optimization-record \
          -foptimization-record-file=$OUT/opt.flang.yaml \
          -Rpass=loop-vectorize -Rpass-missed='loop-vectorize' \
          -Rpass-analysis='loop-vectorize' -gline-tables-only -c kernel.f90 2> $OUT/remarks.flang.txt
```
GCC: `-fopt-info-vec-all` / `-fopt-info-vec-missed` + `-fdump-tree-vect-details`. Parse per hot loop keyed by `DebugLoc` line so the C loop aligns to its Fortran twin: `Vectorized`+`VF`/`IC`; `CantComputeNumberOfIterations`; `CantProveAliasing`/`LoopVersioningRequired`; `SLPVectorized`; interleave/remainder; EH bailouts. A remark Missed-for-C and Passed-for-Fortran, with a named reason, is a taxonomy entry.

### 4.3 Pristine-IR structural diff

From §4.1 pre-vectorizer IR, `llvm-diff` + targeted greps mapping to taxonomy axes:
```bash
grep -c '!alias.scope\|!noalias'  s176.*.fe.ll   # aliasing metadata (predicted ~0 delta given restrict)
grep -c 'align 64\|dereferenceable' s176.*.fe.ll # alignment/deref (predicted residual carrier)
grep -c 'llvm.assume'             s176.*.fe.ll   # trip-count/alignment assumes
grep -c ' nsw\| nuw'              s176.*.fe.ll   # IV wrap flags (SCEV impact)
grep -c '!llvm.loop'              s176.*.fe.ll   # loop metadata/hints
grep -c 'invoke\|landingpad'      s176.*.fe.ll   # EH edges in pure arithmetic
grep -c 'sext\|zext'              s176.*.fe.ll   # 32-bit index sign-extend chains
grep -c 'llvm.vector.reduce.fadd' s176.*.fe.ll   # tree-reduce vs serial (fast rung only)
```
Record the unit-stride axis per innermost memlet here as well (§3.1 gate input).

### 4.4 Static hot-loop throughput

Extract the vectorized inner block from each `.s`, bracket with `# LLVM-MCA-BEGIN/END`, score with the **concrete** CPU string (B1):
```bash
llvm-mca -mcpu=$CPU -iterations=1000 -timeline -bottleneck-analysis s176.clang.hot.s > mca.clang.txt
uiCA.py  --arch <uarch> --iterations 1000 s176.clang.hot.s > uica.clang.txt
```
Store cycles/iter, bottleneck class, per-port µop distribution, packed-vs-scalar, spill count. uiCA is the higher-accuracy arbiter when it and llvm-mca disagree; its model-error floor gates attribution.

### 4.5 Per-nest feature vector

```
kernel, idiom_family, variant, frontend, backend,
vf_intended, vf_achieved, interleave,
noalias_coverage∈[0,1], tbaa_present, align64_coverage, dereferenceable_present,
loop_versioned, scalar_prologue, scalar_epilogue,
iv_nsw, iv_sext_chain, tripcount_opaque, tripcount_const, unit_stride_axis,
landingpad_present, reduction_reassoc,
spill_count, load_count, store_count, fma_count,
mca_cycles_per_iter, mca_bottleneck, uica_cycles_per_iter,
measured_cycles_per_iter, arithmetic_intensity, mxcsr_at_entry
```

### 4.6 Attribution: Shapley decomposition + one unified ablation ladder

`G_cyc = measured_cycles_C − measured_cycles_Fortran` (per-iteration, same backend).

**Call-overhead subtraction (confound).** At L1/L2-long sizes per-call ABI overhead is non-negligible. Emit an empty-body nest per language via `calloverhead.py`; report `G` net of measured per-call overhead so the ladder attributes steady-state IR shape, not trampoline cost.

**One unified ablation ladder, replacing the two divergent ladders in the original §4.6 and C9 (S4-fix).** Each rung is one IR-repair applied to the C variant and re-timed on the arena's normal path with full statistics. Each rung carries an expressibility tag — this tag **is** the science-vs-engineering verdict:

```
rung                                   tag
+noalias-repair (restrict→noalias)     {expressible-in-conforming-C}   [predicted ~0]
+align64/dereferenceable annotation    {annotation-only}
+assume-divisible (n%VF==0)            {annotation-only, or expressible if bound const}
+size_t / nsw index form               {expressible-in-conforming-C}
+no-exceptions                         {expressible-in-conforming-C}
+tree-reduce intrinsic (reassoc)       {unsound-assertion-only}  [fast rung ONLY]
→ Fortran_parity
```
`{expressible-in-conforming-C}` and `{annotation-only}` gaps are **wins for both Part A and Part B** (a smarter emitter recovers them). Only `{unsound-assertion-only}` gaps are the rare fundamental subset. This retires the C9 internal tension where Part B success retracted Part A.

**Order-invariant magnitude (T2/Q2).** With k≈5–6 rungs, decompose `G_cyc` by **Shapley value over the 2^k timed configurations** (order-invariant, interaction-aware, tractable at k≤6) rather than an order-dependent additive ladder. The static-mca "what C could have been" counterfactual (§4.4) is retained strictly as *hypothesis generation*, never as authoritative attribution. Each confirmed taxonomy row = ⟨kernel, diverging pass (§4.2), throughput cost (§4.4), IR feature (§4.3), expressibility tag⟩.

---

## 5. Confound controls

| # | Attack | Control | Falsification test |
|---|--------|---------|--------------------|
| **C1** | Fortran default FP reassociates/contracts | contract-off both sides; matrix at `off` and `fast`; FMA-count parity + fast-math IR grep (§3.3); `array_equal(view_c,view_f)` at contract-off | Gap only where outputs differ in ULPs → **FALSE** (silent fast-math). Survives bit-parity → cleared. |
| **C2** | Different default flags | `-march=native` feature-diff empty; real pipelines dumped (not forced equal); primary crux = frozen frontend IR through identical reference `opt` (§4.1) | Gap vanishes under identical reference `opt` → **FALSE** (flag skew). Survives → in the IR bytes. |
| **C3** | Different backend versions | Same-revision pairs only; assert git hash / configure string; record all four | Mismatched pair dropped, not adjusted. |
| **C4** | Fortran intrinsics vs libm | Arithmetic-only vs intrinsic-bearing partition; core thesis argued on arithmetic-only; `-fveclib` matched; `nm variant.o | grep -E 'exp|matmul|pgmath|sleef'` matches | Gap only in intrinsic stratum → **RESTRICTED** (library). Healthy in arithmetic-only → cleared. |
| **C5** | Layout/alignment/descriptor ABI | `contiguous` explicit-shape `bind(c)`; matched alignment; no `.desc` in flang `.ll` | C `align 1` vs Fortran `align 64` → fixable emission (→C9 annotation rung). Persists with C IR already `noalias align 64 dereferenceable` → genuine residual. |
| **C5b** | **Storage/loop order (L2, NEW)** | Identical row-major storage + C index order both sides; **unit-stride-axis parity in `.ll`** in the §3.3 gate; drop nests where emitters disagree | Gap tracks a stride-axis difference → **FALSE** (layout artifact). Survives stride-axis parity → genuine. |
| **C6** | Memory washout hides/fakes effect | Roofline stratify (§2); L1/L2-long/>L3; confirm with `perf stat` counters; admit to compute stratum only if DRAM traffic ≪ ridge AND 512-bit vectorization observed | Memory-bound gap that vanishes compute-bound → **FALSE/inverted**. `G_compute > G_memory≈1` → confirmation. |
| **C7** | Cost-model threshold flip = bimodal noise | Capture decision+reason via opt-remarks; classify decision-flip vs same-VF-different-codegen; sweep `-mllvm -force-vector-width=8`, trip-count ±20% | Winner flips under perturbation with no consistent reason → **FALSE** (threshold noise). Consistent reason + mca/uiCA agreement → robust. Survives forced-equal-VF → post-vectorization codegen result. |
| **C8** | Noise / DVFS / layout luck | Turbo off, `performance`, `taskset`, NUMA-local, ASLR off; ≥31 reps, minimum + lower-tail CI, shift function; `-falign-loops` sweep. **Layout floor (T3):** same source, N fresh compilations with permuted link order / `-frandom-seed` and k heap/mmap base offsets — the real nuisance, not statement reorder | Gap within the layout floor on compute stratum → **NULL**. |
| **C9** | Gap is a DaCe-emission detail, not a C-language limit | The unified §4.6 ladder with expressibility tags; diff C `.ll` vs Fortran `.ll` at each rung | Closed by `{expressible-in-conforming-C}`/`{annotation-only}` rungs → **win for both parts** (not a retraction). Closable only by `{unsound-assertion-only}` → the fundamental subset. |
| **C10** | **Fortran runtime/temp pollution (B3, NEW)** | `-Warray-temporaries -fcheck=all` (detection); grep `.ll`/`.s` for `_gfortran_internal_pack`/`malloc` inside the loop; assert none | Copy-in/out present → cell discarded until emission fixed. |

---

## 6. Analysis plan

### 6.1 Per-cell and per-nest metrics

- Per cell: **minimum + lower-tail bootstrap CI** (primary), shift function vs the paired variant, plus median+IQR (robustness). If min and median rank differently, re-run.
- Per-nest gap: paired ratio `G = t_C/t_Fortran`, BCa bootstrap 95% CI (10 000 resamples) on the ratio. Real only if the CI excludes 1.0 **and** `G > max(1.05, upper-CI of the C8 layout floor)`. Multiple-comparison control across cells via Benjamini–Hochberg FDR at q=0.05, applied within the secondary/exploratory family only (the primary endpoint is single, T1).

### 6.2 Corpus-level statistics

- Geometric mean of `G` per stratum, **cluster-bootstrapped by idiom family** (T3), with the effective family count reported.
- Effect test: Wilcoxon signed-rank on paired per-family medians (right-skewed runtimes), pseudo-median speedup + bootstrap CI. Report the fraction of families whose gap exceeds the C8 layout floor.
- **P2, dual model (T4):** vectorization is often a step at the ridge, not a slope. Fit both `log G ~ log(AI_measured) + (1|kernel)` **and** a segmented/changepoint model; report which fits better by cross-validated error. The pre-registered P2 statement is the contrast `G_compute > G_memory` with `G_memory ∈ [0.98,1.02]`, not slope monotonicity. Within-kernel paired one-sided Wilcoxon on `G(l2-long) − G(prof) > 0` (kernel identity fixed) is the cleanest evidence.
- **P3 regression:** `log G` on §4.5 IR features; a feature is "attributed" only if the §4.6 Shapley/ablation moves `G → 1` on flagged nests and **not** on control nests lacking the feature.
- **Crux↔timing bridge (S2, required):** across nests, correlate the §4.1 reference-pipeline mca-cycle gap with the §3 real-pipeline measured gap; report Pearson/Spearman r with cluster-bootstrap CI. Low correlation invalidates the crux as a model of the timed result.

### 6.3 Money figures/tables

- **Fig. 1 — Gap vs arithmetic intensity**, two panels (clang/flang, gcc/gfortran, **never pooled**). Memory-bound left ≈1, compute-bound right >1. Delimits where the effect is not.
- **Fig. 2 — Gap attribution (Shapley) stacked bar**, per axis (alignment/`dereferenceable`; divisibility/trip-count; stride/SCEV; residual). Reduction-associativity is a **separate bar shown only for the `fast` rung**, never folded into the strict-headline geomean (L3-fix). A tall residual is a reportable result.
- **Fig. 3 — Gap closed after the emitter fix**, per-nest before/after, on the **held-out fix-evaluation split** (§7). Fixed-category bars drop to ≈1; unaddressed categories don't move.
- **Table 1 — Arena winner distribution** (`perf/plot_winners.py`) + single-core cross-compiler gap (`perf/plot_vectorization.py`); post-fix, the C lane wins nests it previously lost — and any nest the fix regresses is shown in a win/loss ledger (a fix that trades nests is a cost-model result, not a strict improvement).

---

## 7. Success vs null criteria

**Held-out split (pre-registered, T1/S2).** The corpus is partitioned up front into pilot (8, threshold-setting), attribution (feature/ablation development), and **fix-evaluation** (held-out; Part-B recovery quoted only here). Partition is git-tagged before the confirmation run.

**H1 / P1–P3 CONFIRMED iff all hold:**
1. Stratum-1 geomean `G > max(1.05, layout-floor CI)` (CI excludes 1) at contract-off, L2-long, for the LLVM pair, **and** replicated in direction on the GCC pair — reported per-pair, never pooled.
2. `G_compute > G_memory` with Stratum-3 geomean `G ∈ [0.98,1.02]`, within-kernel size Wilcoxon significant, and the better of {linear, segmented} P2 model consistent with the compute>memory contrast.
3. §4.1 shows identical reference `opt` on the two frontends' *verified-pristine* IR yields divergent remarks / mca throughput, and the pristine-IR diff names a specific feature.
4. **Crux↔timing correlation (S2):** r between §4.1 idealized gap and §3 measured gap is positive with CI excluding 0.
5. Shapley attribution: ≥1 category explains ≥50% of the geomean gap, residual bar smaller than the largest named bar, median unattributed residual < 15% of `G`; the attributed category is *not* aliasing (predicted ≈0 given restrict) unless `noalias_coverage < 1` reveals a concrete emission bug.
6. Part-B emitter change recovers ≥70% of the attributed category's gap **on the held-out fix-evaluation split**, with zero contract-off correctness-gate failures.
7. FMA-count, `-march=native` feature, stride-axis, and MXCSR parity audits all pass; the C8 layout floor < effect.

**NULL / alternative outcomes — reported, not buried:**
- **Gap gone at contract-off, present only at `fast`** → fast-math artifact (C1). Report: "at matched FP contract and matched backend, the folk Fortran-is-faster claim is silent fast-math on this corpus." Publishable negative (CC/TACO).
- **Gap tracks the stride/unit-stride axis (C5b)** → layout artifact, not frontend contract. Do not reframe as a win.
- **Effect gfortran-specific, absent in flang (S1)** → the LLVM crux studies a different phenomenon; report per-pair and re-scope.
- **Crux and timing uncorrelated (S2)** → the reference-pipeline experiment is a toy decoupled from the shipped result; drop the causal claim, keep the measurement.
- **Gap flat across AI / largest memory-bound** → not the vectorization mechanism (C6); localize to call overhead or layout.
- **Gap real but unattributable (tall residual)** → ship the roofline-stratified arena + attribution method as the contribution; drop "we fixed the emitter."
- **Every attributed gap closed by `{expressible}`/`{annotation}` rungs** → this is the *expected win*, not a retraction (S4 reframing); the durable thesis stands.
- **Fix trades nests** → per-nest ledger (Table 1); a cost-model result, not a strict improvement.

De-risking spine (S4/Q6): the paper's backbone is the **roofline-stratified same-backend arena + Shapley attribution method**; the sign of `G` is a secondary empirical result, so the paper survives every NULL branch. Pre-commit all thresholds via the git-tagged pilot before the confirmation run.

---

## 8. Phase plan on the existing nest-forge harness

**Reuse.**
- `nestforge/extract.py`; `translate.py` / `emit_cpp` (both frontends, shared `<key>_fp64` symbol); `nestforge/arena.py` (oracle, `max_diff_vs_oracle`, fastest-correct selection); `nestforge/isolation.py run_isolated`; `nestforge/perf/crosslang_xl.py`; `nestforge/perf/flags.py` (`FP_LEVELS` ladder + `-fno-frontend-optimize` — the C1 control); `nestforge/perf/tsvc_full.py`; `nestforge/perf/tsvc_arena.py`; `nestforge/perf/calloverhead.py`; `perf/plot_vectorization.py`, `perf/plot_winners.py`; `perf/daint_all.sh` + SLURM jobs (also the GH200 aarch64 second-µarch check); `docs/OPT_RECORDS.md`, `docs/FP_RISK.md`.

**Add.**
- `scripts/census_ai.py` — FLOP/byte + roofline census (tasklet FLOPs, distinct memlet bytes), LIKWID cross-check. Required before stratification.
- The `--size l2-long` compute-headline preset **and** `--size l1`, alongside PROF.
- **Storage/loop-order pinning + unit-stride-axis extraction** in the Fortran emitter and the §3.3 gate.
- The §4.1 IR harness with the **verified-pristine hard gate** (zero vector ops / `!llvm.loop` in the "pristine" `.ll`; flang MLIR opts disabled via `-Xflang`/`-mmlir`) and resolved concrete `-mcpu` string.
- The variant-to-variant bit-parity gate (contract-off) and FMA-count+ULP gate (`fast` rung) in the arena.
- The unified §4.6 ladder driver with expressibility tags + Shapley harness over 2^k configs; call-overhead subtraction.
- The C8 layout noise floor (permuted link order / `-frandom-seed` / heap-base offsets).
- MXCSR assertion + `_gfortran_*`/temporary detection (C10) in both harnesses.

**Runnable skeleton.**
```bash
# 0. clean state
rm -rf .dacecache .pytest_cache perf_results
CPU=$(clang -march=native -### -c dummy.c 2>&1 | tr ' ' '\n' | grep -A1 -- '-target-cpu' | tail -1 | tr -d '"')

# 1. corpus manifest + AI census + git-tagged pilot/attribution/heldout split
python -m nestforge.corpus build --sources tsvc2,tsvc2_5,polybench,npbench \
  --stratify arithmetic_intensity --bins compute,balanced,memory \
  --idiom-family-tag --split pilot=8,attribution,heldout --out corpus/gap_study.json
python scripts/census_ai.py corpus/gap_study.json
git tag prereg-$(date +%s)   # freeze thresholds + split

# 2. three sizes: L1, L2-long (headline), PROF
python -m nestforge.corpus size corpus/gap_study.json \
  --size l1 --fit-cache L1 --reuse high --ntimes auto \
  --size l2-long --fit-cache L2 --long-trip --ntimes auto \
  --size prof --working-set '>L3'

# 3. same-backend cross-language gap, FP+layout+MXCSR pinned, per size; NEVER pool pairs
python -m nestforge.perf.crosslang_xl --corpora tsvc2 tsvc2_5 \
  --languages c fortran --pairs "clang:flang-new" "gcc:gfortran" \
  --pin-storage-order rowmajor --assert-stride-axis --assert-mxcsr \
  --preset L2LONG --opt-mode canonicalize --fp-mode strict-ieee --reps 31 --out perf_results/xl_l2long
python -m nestforge.perf.crosslang_xl --corpora tsvc2 tsvc2_5 \
  --languages c fortran --pairs "clang:flang-new" "gcc:gfortran" \
  --pin-storage-order rowmajor --assert-stride-axis --assert-mxcsr \
  --preset PROF --opt-mode canonicalize --fp-mode strict-ieee --reps 31 --out perf_results/xl_prof

# 4. attribution matrix + cost-model axis (C7); reduction axis only at fast rung, tagged separately
python -m nestforge.perf.tsvc_full --corpora tsvc2 tsvc2_5 \
  --languages c c++ fortran --opt-modes canonicalize --parallelism sequential \
  --cost-models default no-vec --fp-modes strict-ieee fast \
  --profile-preset L2LONG --matrix-preset full --reps 31 --out perf_results/full

# 5. IR crux (NEW): verified-pristine -> identical reference opt -> remark/mca diff
for nest in $(python -m nestforge.corpus --list --stratum arith_only --split attribution); do
  python -m nestforge.extract $nest --emit cpp fortran --pin-storage-order rowmajor
  clang     -O3 -Xclang -disable-llvm-passes -emit-llvm -S -ffp-contract=off $nest.cpp -o cpp.fe.ll
  flang-new -O3 -Xflang -disable-llvm-passes -emit-llvm -S -ffp-contract=off $nest.f90 -o ftn.fe.ll
  for ll in cpp ftn; do
    grep -qE '<[0-9]+ x (float|double)>|!llvm.loop' $ll.fe.ll && { echo "NOT PRISTINE $ll"; exit 1; }
    opt -passes='default<O3>' $ll.fe.ll -S -o $ll.opt.ll
    llc -O3 -mcpu=$CPU $ll.opt.ll -o $ll.s
  done
  clang -O3 -ffp-contract=off -fsave-optimization-record -c $nest.cpp   # -> nest.opt.yaml
  python -m nestforge.arena --variants cpp.s ftn.s \
     --require-bitexact-between-variants --oracle numpy --sizes L1 L2LONG DRAM \
     --reps 31 --stat min-ci --shift-function --pin-freq --taskset 3 --assert-mxcsr
done

# 6. readers + stats
python perf/plot_vectorization.py --results-dir perf_results/full --lang c --gap-threshold 1.5
python perf/plot_winners.py       --results-dir perf_results/full
python -m nestforge.stats gap perf_results/full/gap.parquet \
  --pair c=clang f=flang-new --pair c=gcc f=gfortran --no-pool-pairs \
  --group stratum,idiom_family --ratio G=t_c/t_f \
  --stat min --shift-function --bootstrap bca --resamples 10000 \
  --cluster-bootstrap idiom_family --fdr 0.05 \
  --model 'log G ~ log(AI) + (1|kernel)' --also segmented \
  --within-kernel-size-test l2-long,prof \
  --crux-timing-correlation perf_results/crux \
  --shapley perf_results/ablation --report tables/gap_summary.md
```

**Reproducibility manifest (per run):** LLVM/GCC commit hashes; `-march=native` feature diff (empty); resolved `-mcpu` string; FP rung + FMA-count parity per cell; storage order + unit-stride axis per cell; MXCSR at entry; measured AI + roofline point; governor/turbo/pinning/NUMA/ASLR state; R, W, min/CI/shift/median/IQR; C8 layout floor; verified-pristine assertion pass/fail; crux↔timing r; opt-remark YAMLs, pristine `.ll`s, mca/uiCA reports, git pre-reg tag archived alongside the arena JSON.

**Threats to validity:** silent fast-math (mitigated by `FP_LEVELS` + `-fno-frontend-optimize`; nvfortran/ifx excluded from the same-backend headline — their whole-model FP knobs can't be split per-assumption); real pipelines are frontend-configured and differ (do not claim identity; the crux is a reference-pipeline model, validated by the crux↔timing correlation); flang MLIR pre-optimization (mitigated by the verified-pristine gate); storage/loop order (mitigated by C5b stride-axis parity); single-machine/arch and µarch cost-model tuning — AVX-512 downclock can flip `G`'s sign between Skylake-SP and Zen4, so results are reported per-µarch and a **sign-flip across µarch is pre-registered as NULL for the contract claim**; direction is confirmed on GH200 aarch64 before any generality claim; corpus bias (TSVC over-samples compute-bound → effect size is an upper bound; spot-check PolyBench/NPBench).

**Venue:** primary CGO (characterize → attribute → fix emitter → measure recovery); fallbacks CC (if tool/method dominates or the result is the matched-FP negative), PACT (if the vectorization-obstacle taxonomy dominates — lineage to Maleki et al. 2011), TACO (if the arena grows large).

---

> ## Kill criteria — the earliest cheap test that says STOP
>
> Run these **before** the full confirmation sweep, in order, on the pilot set. Any one firing means stop or re-scope, not push harder.
>
> 1. **FP-parity kill (cheapest, first).** Take the 5 kernels where Fortran is folk-known to win (e.g. `s176`, `gemm`, `s311`, `s352`, `dot`). Pin FP-contract to parity (`-ffp-contract=off`, `-fno-frontend-optimize`, FMA-count equal, `array_equal(view_c, view_f)` True). If the geomean gap collapses into `[0.98, 1.02]` once contract is off, **STOP the causal-mechanism paper** — the effect was silent fast-math; pivot to the matched-FP negative-result paper (CC/TACO). Cost: ~1 hour, no crux, no attribution.
>
> 2. **Layout kill.** On the same 5, pin identical row-major storage + unit-stride-axis parity. If the gap tracks the stride axis and vanishes at parity, **STOP** — it is a column-major layout artifact (C5b), not a frontend contract.
>
> 3. **Pristine-IR kill.** If the flang "pristine" `.ll` still contains vector ops / `!llvm.loop` after the `-Xflang`/`-mmlir` disable (the hard gate in §4.1 keeps failing), **STOP the crux** — flang's MLIR pipeline cannot be frozen comparably here; downgrade to the correlational GCC-pair study or a pure measurement paper.
>
> 4. **Premise kill.** If the folk effect does not reproduce on gcc/gfortran (the mature pair) at parity, do not trust the flang pair to carry it. **STOP** and re-scope to whichever pair actually shows it, reported per-pair.
>
> 5. **Crux↔timing kill.** If the §4.1 reference-pipeline mca gap does not correlate with the §3 measured gap (r CI includes 0) on the pilot, **STOP claiming causality** — keep only the roofline-stratified arena as the contribution.
>
> If (1)–(4) all pass on the pilot, the effect is real, FP-clean, layout-clean, pair-robust, and crux-coupled — proceed to the full sweep. If (1) or (2) fires, the cheap negative is itself the paper.
