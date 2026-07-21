# PLAN — the `optimize` entry contract + e2e tests in CI

Status: PLAN ONLY, nothing implemented. Written 2026-07-22.

## The contract the user asked for

NestForge should expose one entry point that takes *whatever the user has* and returns the
best-performing build, by compiling and measuring variants rather than trusting language- or
compiler-defined semantics.

Three input cases, each selecting a different SEARCH SPACE:

| # | input | what we can change | search space |
|---|---|---|---|
| A | a source file to compile: **C / C++ / Fortran** | only how it is compiled | vectorizer flags x cost-model flags x fp flags x compilers |
| B | **NumPy or Fortran** to be parsed | the generated code too | parse -> SDFG -> C++; OLD codegen vs MANY new-codegen combinations, x flags x compilers |
| C | default, **no agent driving** | everything | SDFG -> C++, test ALL variants exhaustively |

Purpose (user's words): *"decrease reliance on language and compiler dependent semantics as much as
we can"* — i.e. do not assume a flag or a codegen path is better, measure it.

## What already exists (verified, do not rebuild)

- **4-phase agent API** (README): P1 `fusion` -> P2 `offload` -> P3 `optimize` -> P4 `feedback`.
  `optimize(nest, knobs)` in `nestforge/optimize.py` already dispatches to an `Optimizer` knob
  bundle. The new entry sits ABOVE this, it does not replace it.
- `nestforge/translate.py`: `prepare`, `prepare_whole_program`, `prepare_regions`,
  `emit_sources(prep, out_dir, target="c", precision="float64")`.
- `nestforge/session.py:268` already documents targets `"numpy" | "c" | "cpp" | "fortran"`.
- `nestforge/optimizers.py:52` already carries `language: Optional[str]  # "c" | "fortran"`.
- `nestforge/arena.py: run_arena(prep, ...)` — the compiler x flag x FP-mode matrix, `Cell.compile_us`.
- `nestforge/build.py` — compiler family detection (gnu / intel / nvidia), ccache, linkers.
- `nestforge/{fusion,fission,region}_arms.py`, `vectorize_variants.py`, `sweep.py`, `strategies.py`,
  `whole_program.py` — existing variant axes.

## Design

### 1. One entry point + input-kind detection

`nestforge/entry.py` (new): `optimize_program(source, *, agent=None, ...) -> Report`.

Detect the input kind explicitly, never by guessing from content alone:
- suffix + an explicit `kind=` override (`c` / `cpp` / `fortran-source` / `fortran-parse` / `numpy` / `sdfg`)
- the ambiguity that matters: **Fortran is in BOTH case A and case B.** A `.f90` can be handed
  straight to a Fortran compiler (case A) or parsed into an SDFG (case B). The caller must be able to
  say which; default to B (parse) because that is the strictly larger search space, and fall back to
  A with a recorded reason if the frontend refuses the file.

### 2. Case A — provided source, flag-space only

No SDFG, no extraction. Feed the file to the arena's compile matrix directly.
- Needs a `Prepared`-like path that skips numpy emission and translation. Check whether
  `arena.run_arena` can accept a pre-existing source; if not, add the smallest adapter.
- Axes: vectorizer flags, cost-model flags, fp flags, compilers. These already exist in `build.py` /
  the arena matrix — this case is mostly plumbing, not new search.

### 3. Case B — parsed input, codegen-space search

- NumPy -> SDFG via the DaCe Python frontend; Fortran -> SDFG via **dace-fortran** (a SIBLING repo,
  not currently a nest-forge dependency — see risks).
- Then SDFG -> C++ and search **old codegen vs many new-codegen combinations**. The DaCe knob is
  `compiler.cpu.implementation` (`experimental_readable` vs legacy) plus the codegen_params knobs
  (`explicit_copy`, `scalar_emission_type`, ...). Confirm the current knob names against dace
  `extended` at implementation time — my notes on these are days old and have already been wrong once.
- Cross with the existing flag/compiler matrix.

### 4. Case C — default, no agent

Exhaustive sweep over the same axes. `sweep.py` / `strategies.py` likely already provide this; the
entry point should select it when `agent is None` rather than introducing a parallel mechanism.

### 5. Correctness gate (non-negotiable, already the repo's rule)

Every variant is validated bit-exact against the numpy oracle before its timing counts. A variant that
does not validate is reported as failed, never silently dropped. This is what makes "measure, don't
trust semantics" sound rather than reckless.

## e2e integration tests in CI

Current state: `ci.yml` runs on push-to-main / PR / dispatch, ubuntu-latest, 60 min timeout, and runs
ONLY the unit set `-m "not integration"` with **zero skips enforced** (`NESTFORGE_CI_NO_SKIP`, repo-root
conftest fails the session on any skip). The `integration` marker means "needs a vendor compiler
(nvc/icx/icpx) or a heavy launcher" — those genuinely cannot run on the runner.

So e2e tests must NOT reuse the `integration` marker, or they stay excluded forever.

Plan:
- New marker **`e2e`** in `pyproject.toml`: "runs the full optimize contract end to end using only
  g++/clang++".
- Add a CI job (or a second pytest phase in the existing job) running `-m e2e`.
- Keep it credential-free — the workflow comments are explicit that needing a secret would stop CI
  running on fork PRs. Deps already install over public HTTPS.
- **Budget the matrix.** A full variant sweep will not fit 60 min. In CI: one tiny kernel, preset `S`
  shapes, 2 compilers (g++, clang++), a small flag set, 1 fp mode. The test asserts the CONTRACT
  (each input kind dispatches to the right search space, every variant validates bit-exact, a winner
  is selected and reported), not that a particular variant wins — a "which flag is fastest" assertion
  would flake on a shared runner.
- Zero-skip policy means an e2e test must not skip on the runner; if a case cannot run there it must
  be `integration`, not `e2e`.

## Risks / open questions to settle at implementation time

1. **dace-fortran is not a nest-forge dependency.** Case B's Fortran path needs it. Options: make it
   optional and mark those tests `integration`, or add the dep. Do not silently import it.
2. **Case A vs B for Fortran** must be caller-selectable (above).
3. **Codegen knob names** must be re-read from dace `extended`, not taken from my notes.
4. **CI time budget** — measure the e2e job before enabling it on every PR; if it is slow, keep it on
   push-to-main + dispatch only.
5. `arena.run_arena` may assume a `Prepared` built from numpy emission; case A needs a way in.

## Sequencing

1. Re-verify the four facts above (knob names, `emit_sources` targets, arena entry, dace-fortran).
2. `entry.py` + input-kind detection + dispatch, with unit tests (no compiling).
3. Case A path, then case C, then case B (B is the largest and depends on the dace-fortran decision).
4. e2e marker + tests + CI lane, budgeted.
5. Then the three measured perf wins from the assessment (`extract_all_nests` deepcopy 2x,
   `symbol_ranges` reorder, `compile_object` ccache 2.57x) — independent of this contract.
