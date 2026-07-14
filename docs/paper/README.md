# Paper research corpus — semantic-lifting cross-compiler arena

Research and experiment plan for the paper. **Nothing here has been run yet** — these are the
literature grounding, the design, and the experiment protocol. Produced by a multi-agent literature
sweep; every citation should be re-verified against the primary source before it enters the paper.

## Governing claim

From a lifted, semantics-explicit IR (DaCe SDFG) whose dependence and aliasing facts are known by
construction, emit each loop-nest offload unit under a **per-transform, per-array maximally-permissive
language contract** — `restrict`, alignment, induction-variable width/signedness, and a **per-transform
floating-point association license instead of a global `-ffast-math`** — with soundness discharged by SDFG
dataflow, gated on numerical correctness against a reproducible oracle, and select the fastest correct
variant **across compilers and languages**. The falsifiable claim is a **measured performance residual
beyond the SDFG's existing global `restrict` + `-fno-math-errno` emission** (Calotoiu et al., ICS 2022).
Without that residual the contribution is "ICS 2022 with more flags."

The single strongest differentiator is the four-way conjunction that no prior system holds:
**cross-compiler × cross-language × per-transform-FP-under-a-numerical-oracle × causal attribution.**
The most dangerous citation is Calotoiu et al. (ICS 2022) — our own group's SDFG-lifting predecessor.

## Deliverables

- [DEEP_RESEARCH_fp_aware_cross_compiler_vectorization.md](DEEP_RESEARCH_fp_aware_cross_compiler_vectorization.md)
  — the initial landscape + a brutally critical assessment of the FP-gate framing, with the offload and
  variant-space addenda.
- [RELATED_WORK_offload_fp_arena.md](RELATED_WORK_offload_fp_arena.md) — optimizer definition, the
  capability comparison table, the FP-flag lattice, and the danger/safety detector design.
- [RELATED_WORK_cross_language_semantic_lifting.md](RELATED_WORK_cross_language_semantic_lifting.md) —
  the related-work section on cross-language / cross-compiler optimization and semantic lifting; isolates
  the closest prior art (ICS 2022, Polygeist, MCompiler) and the opening.
- [EXPERIMENT_frontend_semantics_gap.md](EXPERIMENT_frontend_semantics_gap.md) — the executable protocol
  to characterize and close the C++-vs-Fortran optimization gap on shared-backend compilers.
- [RANKED_BIBLIOGRAPHY.md](RANKED_BIBLIOGRAPHY.md) — every reference ordered by importance to the paper,
  each with a positioning note and a threat level.

## Experiment plan (planned — NOT run)

Sequenced cheapest-first; each stage gates the next. Full detail in `EXPERIMENT_frontend_semantics_gap.md`.

1. **Kill tests (≈1 hr, run first).** On 5 folk-Fortran-wins kernels (`s176`, `gemm`, `s311`, `s352`,
   `dot`):
   - *FP-parity kill* — pin `-ffp-contract=off`, equal FMA count, `array_equal(view_c, view_f)`. If the
     gap collapses into `[0.98, 1.02]`, the effect was silent fast-math → stop the causal paper, pivot to
     the matched-FP negative result.
   - *Layout kill* — pin row-major + unit-stride-axis parity. If the gap tracks the stride axis, it is a
     layout artifact → stop.
   - *Premise check* — replicate on gcc/gfortran before trusting the young `flang-new`; never pool
     compiler pairs into one geomean.
2. **Pre-registration.** 8-kernel pilot (disjoint from the confirmation corpus) fixes the arithmetic-
   intensity split, the significance threshold, the vector-length decisions, the Part-B recovery target,
   and one git-tagged primary endpoint — all before the confirmation run.
3. **Confirmation study.** ~44 kernels (TSVC / PolyBench / NPBench) stratified by measured arithmetic
   intensity; compute-bound (gap expected) vs memory-bound controls (gap must vanish). Measure the
   per-nest residual `G = t_C / t_Fortran` at FP-contract parity; the same-optimizer-library crux
   (verified-pristine IR through one reference `opt` pipeline) plus Shapley attribution to IR features.
4. **Part B (the tool).** A DaCe C++-emitter change that emits the Fortran-like IR form, evaluated on the
   held-out fix-evaluation split; success = it recovers a majority of the attributed gap at zero
   contract-off correctness-gate failures.

De-risking spine: the paper's backbone is the **roofline-stratified same-backend arena + attribution
method**; the sign of `G` is a secondary result, so every null branch is still publishable.
