# Backlog

One ordered list, worked top to bottom. Every item is verified against the source at 2026-07-22;
anything already done was deleted rather than ticked. `->` marks a dependency.

## A. Emitter correctness (DESIGN.md §3 has the full write-up)

Independent of each other unless noted. A2 must land before A6, which rewrites the same function.

- [x] **A1** DONE. `reads_array_data` + `sizable` in `emit_numpy.py`, gating `symbol_ranges`
      acceptance, the `max_over_loops` residual, and `reject_unsizable_scratch` per DIMENSION.
      Pre-fix, spmv widened its row scratch to `-Subscript(A_indptr, 0) + Subscript(A_indptr, M + 1)`
      and the `free_symbols` residual was empty, so it was ACCEPTED; now refused with a reason.
      Regression tests in `tests/test_emit_latent.py`, fixture built through the frontend.
- [x] **A2** DONE. Direction extracted into `emit_numpy.copy_direction`, which tests the SOURCE first.
      Both endpoints of an in-place copy carry the same name, so the order decides; DaCe resolves the
      same tie the same way (`Memlet.try_initialize` prefers `is_data_src=True`). Tested as a property
      against DaCe's own `get_src_subset`/`get_dst_subset` over frontend-built SDFGs, plus one direct
      pin of the in-place case.
- [x] **A3** DONE. New `emit_libnode.operand_rank` gives the rank of an operand AS RENDERED (0 for a
      scalar-local or size-1 buffer, the descriptor rank for a whole-array reference, one axis per
      subset range otherwise -- libnode operands render `keep_singleton=True`, so a length-1 axis
      survives). `emit_reduce` decides keepdims on that instead of the buffer descriptors.
- [x] **A4** DONE. `emit_numpy.reconcile_connector_descriptor` syncs a connector descriptor ONLY for
      the one legitimate disagreement -- `Scalar` inside, size-1 ARRAY outside (a nested return), where
      bare `x` and `x[0]` must agree -- and refuses any other extent mismatch instead of overwriting
      the inner shape, which silently re-ranked the body (`Z[j]` for `Z[j, k]`). Instrumented the old
      blanket replace across every emit suite first: it NEVER changed a shape, so refusing cannot
      regress a working kernel.
- [x] **A5** DONE. `emit_numpy.range_stop` turns a DaCe INCLUSIVE end into python's exclusive stop in
      the direction of travel (`end + 1` ascending, `end - 1` descending) and refuses a step whose sign
      is undecidable. The blanket `+ 1` emitted `range(N-1, 0, -1)` for a range ending at 0 and dropped
      the last element.

      On normalizing instead: dace's canonicalize pipeline already ships `NormalizeNegativeStride`
      (LoopRegions with a literal negative stride, run twice), so a canonicalized nest has none. Not
      relied on here for two reasons -- nest-forge's DEFAULT opt mode is `simplify-parallel`, not
      `canonicalize`, so emitter correctness would depend on which mode ran; and that pass rewrites
      LoopRegions, while `map_lines` reads a Map's own range. Normalizing MAP ranges would be sound and
      free (a Map's iteration order is unspecified), but no path is known to produce one -- so the
      emitter defends at the point of use and nothing is normalized on spec.
- [x] **A6** DONE, but NOT as written. Folding `access` / `copy_side` / `reshape_side` into one
      flag-parameterised function was rejected: their differences (what a `None` subset means, what a
      size-1 BUFFER renders as) are semantic, not accidental, and two policy flags on one function
      reads worse than three named policies. The real duplication was elsewhere and is now gone -- the
      twelve-line "render both sides of a copy" decision was copy-pasted in `copy_lines` and
      `map_exit_writes`, and both now call `copy_sides`. A2's cause was the direction test, not the
      access rule, so the premise of this item was weaker than DESIGN.md claimed.
- [x] **A7** DONE. `emit_numpy.symbol_mapping_lines` stages through temps when a target symbol appears
      on any right-hand side, and emits plain assignments otherwise, so the readable form survives
      wherever the bindings do not interfere.
- [x] **A8** DONE, and stricter than planned. `emit_conditional` emits branches in stored order and
      REFUSES an unconditional branch that is not final -- because DaCe refuses it too: its codegen
      raises `Missing branch condition for non-final conditional branch`, verified by building and
      running the SDFG. Hoisting unconditional branches to a trailing `else` compared the emitted
      kernel against semantics no DaCe build has, made a branch stored after one live, and turned two
      unconditional branches into two `else:` clauses. `tests/test_conditional_emit.py` asserted the
      old hoisting; it now asserts the refusal.

## B. The agent's view — SDFG structure as a string tree (CORE)

An SDFG is a graph; an agent reasons badly about graphs and well about text. The whole agent-facing
design therefore rests on two projections: **structure becomes an indented string tree**
(`introspect.describe_graph`, `Session.describe`, `Session.region_tree`) and **each nest body becomes
numpy** (`emit_numpy.nest_to_numpy`). The agent reads those, controls fusion/fission granularity, and
nothing else; each nest is then optimized by TRANSLATION to Fortran/C/C++ and measured.

That makes the string tree a core API surface, not a debugging aid — it is the agent's only view of
the program. Today it renders, but an agent cannot act on what it reads:

- [x] **B1** **The text tree carries no handles.** `describe_graph` prints `loop 'for_10'`, while the
      actionable ids come from a separate `Session.list_nests()` / `region_tree()` call with a
      different shape. The agent must join two views by eyeballing labels. Put the epoch-scoped id on
      the line it belongs to, so reading and acting use one vocabulary.
- [x] **B2** -> B1. **The tree shows containers, not bodies.** It lists regions, states and nests but
      never the numpy the agent is reasoning about. Render each nest's body (or a bounded excerpt of
      it) inline, or give the line an id that fetches it, so one view answers both "where" and "what".
- [x] **B3** **Labels are frontend-generated and unstable** (`for_10`, `slice_A_vals_15` — the source
      LINE NUMBER is in the name). Re-parsing the same program after an edit renames every node, so an
      agent's notes across rounds silently refer to different nests. Decide a stable naming scheme.
- [x] **B4** **Pin the format.** The tree is a prompt: a silent format change breaks every agent
      reading it, and nothing tests the rendering today. Add a golden-tree test the way
      `tests/test_phase_api_contract.py` pins the skill surface.

## BK. The kernel surface — body as a function, and a language the agent chose

Design: `docs/kernel_surface/README.md`. A kernel is (iteration domain, body function); today the
body is reachable only by string-slicing `for` headers off a re-emit, and anything compiled drags
`dace::math::*` in through `build.include_flags`.

- [ ] **BK1** `NormalizeWCR` + `NormalizeWCRSource` into `normalize_for_tree`, plus
      `reduce=(op over axes -> target)` on the kernel line so the tree stops hiding a reduction.
      `detect_reduction_type` names the op; `ReductionType.Custom` refused by name. Gate on the
      3.5ms warm normalize. NOT `NestInnermostMapBodyIntoNSDFG` -- emitting does not need it.
- [ ] **BKp** Re-normalize after a move is 37ms on cavity_flow (was 120ms before stable naming), of
      which 32ms is ONE `sdfg.replace_dict` global walk for a single stale name. Dirty/clean region
      tracking saves the ~3ms of scans, not the 32ms -- build it when the scans grow. A state-scoped
      replace measures 9.1ms but hand-rolling a rename risks a silent wrong answer; the right home is
      dace's `replace_dict`.
- [x] **BK2** Re-cut `introspect.kernel_body` against the scope tree; delete the `lines[headers:]` /
      `line[4 * headers:]` string surgery.
- [x] **BK3** -> BK1, BK2. Reduction representations the agent picks between: `folded` (explicit
      loop, DEFAULT -- measured to lower with no temp buffer) and `declared` (`np.sum`, reads better,
      today lowers to buffer-then-reduce). Whatever is emitted must be valid runnable numpy.
- [ ] **BK4** -> BK2. `form="slice"` for straight-line bodies (where `declared` reductions live).
- [x] **BK5** -> BK2. `lang="c"|"cpp"|"fortran"` through numpyto, point form only. NO nest-forge
      intrinsic layer -- emit `np.<op>` and let numpyto spell it. `Session.kernel_source(lang=...)`
      extracts the nest on a DETACHED copy (projection, no mutation), reuses `prepare` + `emit_sources`.
      Bare C++ is the `.cpp` of the plain `c` target (one emit -> C-family), not `cpp_omp`.

## C. Scratchpad allocation pass

Today the C-style contract makes EVERY non-scalar transient a caller-allocated parameter
(`emit_yaml.array_names`, `emit_numpy.scratch_arrays`), and `reject_unsizable_scratch` refuses the
nest outright when a shape is not a function of the kernel's own symbols. A nest that merely needs
scratch should not push that allocation onto the caller.

- [ ] **C1** -> A1. Classify each scratch shape with the same `sizable` predicate, into three cases:
      **constant** (a literal extent), **entry-symbolic** (free symbols only, all bound at program
      entry, so the extent is known once the SDFG is initialised), and **loop-dependent** (an extent
      that changes inside the nest — still refused, that is the sound case).
- [ ] **C2** -> C1. A pass that allocates the first two cases in SDFG init rather than emitting them
      as parameters: allocate in `__dace_init`, free in `__dace_exit`, keep the buffer transient so it
      never reaches the signature. Entry-symbolic is legal there precisely because every symbol it
      names is already bound when init runs.
- [ ] **C3** -> C2. Drop the allocated names from `array_names` / `scratch_arrays` / the manifest, so
      the emitted signature, the manifest and the ctypes bind stay the same positional list. This is
      the failure mode DESIGN.md's closed finding #1 was: a signature/manifest split slides size
      symbols into pointer slots and corrupts memory silently. Test the three cases end to end.

## D. Test infrastructure

- [x] **D1** DONE, and it was NOT a test bug. `emit_numpy.load_emitted` named the emitted module file
      by a COUNTER, so two different kernels could share a path; CPython invalidates `__pycache__` on
      (mtime, size) only, so a same-second rewrite of equal byte length serves the FIRST kernel's
      bytecode. The caller then validates and times a kernel it never emitted -- a wrong-answer bug in
      the arena, surfacing as the flaky BLAS shape mismatch. Forking made it certain rather than
      unlikely: every `run_isolated` child inherited the same next counter value. The file name is now
      a hash of the source, so distinct sources are distinct files and identical sources legitimately
      share one cache entry. Regression test in `tests/test_emit_latent.py`.

## E. Entry contract (`nestforge/entry.py`, docs/PLAN_optimize_contract.md)

- [ ] **E1** An arena entry for a PROVIDED source. `run_arena` assumes a `Prepared` built from numpy
      emission; a C/C++/Fortran input has none. Smallest adapter, not a parallel path.
- [ ] **E2** -> E1. `optimize_program(source, *, agent=None, ...) -> Report` — the executing entry.
      Today `plan_search` returns a `SearchPlan` nothing consumes.
- [ ] **E3** `lower_to_sdfg` for `InputKind.NUMPY` (raises `NotImplementedError`).
- [ ] **E4** Map `FLAG_AXES['vectorize']`'s three values to per-compiler flags; reuse
      `perf/flags.py::cost_flags`.
- [ ] **E5** ccache on `arena.compile_object` (2.57x measured on repeat compiles);
      `BuildOptions.use_ccache` already exists on the `build.py` path.

## F. Offload granularity (docs/MEGAKERNEL_TODO.md)

- [ ] **F1** The libnode invariant is unenforced: `whole_program.default_offloadable` returns `True`
      unconditionally and `offload.state_has_compute` counts a `LibraryNode` AS compute, so the
      `state` unit externalizes a BLAS state that should stay a libnode.
- [ ] **F2** Multi-node granularity (a line graph of consecutive blocks) through
      `nest_sdfg_subgraph`; `extract_loop_nest` passes a single block today.
- [ ] **F3** `ConditionalBlock` as a single-node granularity (currently a `TypeError`).
- [ ] **F4** -> F2, F3. Numeric (oracle) checks per granularity. `tests/test_offload_units.py`
      asserts SDFG validity only.

## G. Documented-but-unbuilt — decide build vs. relabel

Each of these reads as BUILT in its doc and has zero implementation. Every one needs a call: build
it, or rewrite the doc to say "planned". Do not leave them reading as fact.

- [ ] **G1** `Boundary.nest_parallelism` / `Boundary.parent_is_parallel` (PARALLEL.md §6, and
      BUILD.md:150 depends on them). Nearest built thing is `strategies.is_parallel_nest` surfaced as
      `OffloadCandidate.parallel`; there is no ancestor walk at all.
- [ ] **G2** -> G1. The three-row compile-intent dispatch table (PARALLEL.md §2). Nothing consumes a
      parallelism descriptor to gate `-fopenmp`.
- [ ] **G3** Driver-owned OpenMP init (PARALLEL.md §3.4 already reads as fact; thread count is only
      READ from the environment today). **Decided:** warm the pool by running a full-thread region in
      the SDFG-INIT phase — an `omp parallel` block in `__dace_init` that every thread enters, so the
      runtime has created and bound the whole team before the first timed nest. Pool creation
      otherwise lands inside the first measured region and is charged to it. `omp_set_num_threads` /
      `omp_set_proc_bind` are set there too, ahead of the warm-up, so the team is the right size and
      pinned before it is built.
- [ ] **G4** Reject a link set that MIXES runtimes (PARALLEL.md:156). `OpenMPRuntime.check`
      validates one compiler against one runtime; nothing validates a set.
- [ ] **G5** `fp_risk` static classifier. Cited as a live gate in PARALLEL.md and PREDICTIVE.md;
      only `docs/FP_RISK.md` exists.
- [ ] **G6** Predictive mode: opt-report emission + parsing (`-fopt-info` / `-Rpass` /
      `-qopt-report` / `-Minfo`) and the SQLite result corpus. PREDICTIVE.md §2A and
      docs/OPT_RECORDS.md both read as built; results are per-kernel JSON and no report is parsed.
- [ ] **G7** BUILD.md leftovers: the `time_kernel(entry, reps)` harness, the dead-strip flags
      (`-ffunction-sections` / `--gc-sections` / `-fno-semantic-interposition`), and §7's
      translate-comments feature (optarena emits OpenMP from dependence analysis instead, so §7 is
      describing a design that was not taken).

## H. Cleanup (KISS/YAGNI)

- [ ] **H1** `perf/crosslang_xl.py::family_of` duplicates `build.compiler_family` and disagrees with
      it on the Intel label (`intel` vs `intel-classic`). `tsvc_arena.Toolchain.fp_family` already
      does the job correctly — delete `family_of`.
- [ ] **H2** Three FP-flag tables: `arena.FP_MODES` (gcc/clang, 3 levels), `perf/flags._FP` (4
      families x 4 levels), `perf/flags._REDUCED_FP`. `arena._BASE` is byte-identical to
      `flags.base_flags("gnu")`. Collapse the arena's onto `perf/flags`.
- [ ] **H3** `device_profile.py` has the repo's only two underscore-prefixed FUNCTIONS
      (`_probe_source`, `_run_probe`) — rename. (The ~58 underscore CONSTANTS are a separate call;
      13 of them are read directly by tests, so the underscore is already a lie for those.)
- [ ] **H4** `granularity.to_canonical_atoms` is a one-statement passthrough to
      `fission_to_statements` with no added behaviour. Inline its 4 callers.
- [ ] **H5** Unreferenced by anything: `prototypes/gpu_stream_interop/` (11 tracked files, zero
      references outside itself) and `scripts/census.py`. Also
      `docs/paper/EXPERIMENT_frontend_semantics_gap.md:345,364` cite `scripts/census_ai.py`, which
      does not exist. Decide keep-and-reference or delete.
- [ ] **H6** `nestforge/report.py` (34 lines) has one consumer, `examples/demo_fma.py`. No test, no
      driver, no CI. Keep only if the example is kept.
- [ ] **H7** `Session.emit_variant` has zero callers and zero tests, and its docstring advertises
      `target="numpy"|"cpp"` -- neither is a numpyto `--target` (it would fail argparse `choices`).
      BK5's `kernel_source(lang=...)` is the tested per-language surface now; either delete
      `emit_variant` or route it through `LANG_LOWERING` and fix the contract.

## I. Unverified numbers in docs

- [ ] **I1** `PARALLEL.md:182` claims a ~1e-3 divergence; the test that would pin it
      (`tests/test_gramschmidt_fma.py`) asserts only `> 1e-6`. `PREDICTIVE.md:50`'s "gramschmidt:
      17.4 vs 0" has no assertion behind it either. Pin them or drop the numbers.

## K. Fusion foundation audit (user priority — the passes the agent rides on)

The agent's success rides entirely on the quality of the fusion/fission foundation. Deep
audit + improve, AFTER the active tasks above, BEFORE J. Do not reinvent — the passes exist in
dace/extended; audit and improve them.

- [ ] **K1** **state-fusion-with-happens-before** — the substrate everything else sits on. Audit the
      WAR-edge-targets-consumer redesign for missed cases.
- [ ] **K2** **loop-fusion / loop-fission**, **map-fusion / map-fission**. Ruling: **statement-fission
      is the canonical floor**, not loop-fission. `fission_to_statements` already reaches statement
      granularity (SplitStatements → LoopFission → MapFission); LoopFission is only a STEP in that
      descent, never where we stop. Loop-fission alone is the worse primitive.
- [ ] **K3** **Fuse MORE than statement granularity.** The lattice runs atoms→maximal; the fusion arms
      must GROUP many statement-atoms into one kernel, not leave them one-atom-per-nest. Decompose to
      statements, then re-fuse UP to the chosen granularity.
- [ ] **K4** **fuse-loops / fuse-maps with close iteration domains** (dace/extended). Audit + expose the
      granularity choice cleanly. CPU vs GPU have different perf characteristics — the agent LEARNS the
      tradeoff; the passes must not hardcode it. This is the paper's C1 (backend-dependent optimum).

## J. Last — mega-kernel

An OFFLOADING STRATEGY, so it comes after everything above is done. Nothing is built.

- [ ] **J1** -> F2. Take a nested SDFG and rewrite it as ONE persistent kernel (GPU device kernel, or
      CPU persistent multicore). Assess FIRST whether the readable/experimental CPU codegen already
      emits multi-dimensional OpenMP parallel scopes; if not, that is the one codegen change needed.
      Thread ids inside the scope, every launch becomes a grid-strided loop, global sync in its own
      state. jacobi2d (a time loop wrapping two maps) is the worked example.
