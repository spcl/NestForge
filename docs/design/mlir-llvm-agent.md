# MLIR and LLVM agent

[Designs](README.md) · [Stages](../stages.md) · [Bench requests](bench-requests.md)

Status: design. Written 2026-11-24 against NestForge `b98494f`. It builds on the survey of 2026-11-19
(`nestforge-workspace/notes/mlir-llvm-agent/survey.md`, pass names checked in the LLVM 22.1.2 headers and tools) and
the skill draft beside it (`SKILL.md`). Nothing here is built.

## Goal

An agent rewrites a kernel by applying compiler passes one at a time, the way the NestForge agent applies moves:
each pass is a named move with arguments, the agent sees the loops after each move, it can roll a move back, and the
reference checks the result. The agent works in two phases on the same kernel: MLIR passes on affine and scf loops,
then LLVM passes on the IR the fixed lowering writes. It is a baseline for the study: what an agent reaches with a
compiler's own passes, without NestForge's moves.

## What exists

The two pass baselines (the paper's LLVM and MLIR agents) build every kernel through one pipeline string:

- `Session.set_pipeline` (`tactics/session.py`) checks the string's syntax on an empty module and stores
  `self.pipeline = (ir, pipeline)`; `rollback` restores the previous value through `program_change`.
- `Session.pipeline_build` copies the kernel's CPF unit into a folder of its own and calls
  `build_llvm_pipeline` or `build_mlir_pipeline` (`tactics/build/pipelines.py`).
- `build_mlir_pipeline` runs `cgeist -S -O0 --raise-scf-to-affine`, then `polygeist-opt --pass-pipeline=<pipeline>`,
  then the fixed `MLIR_LOWERING` (`lower-affine`, `convert-scf-to-openmp`, `convert-polygeist-to-llvm` with C-style
  memrefs), then `mlir-translate`, then clang's back end with the IR passes off.
- `build_llvm_pipeline` lowers the unit with clang at `-O0` without `optnone`, runs `opt -passes=<pipeline>`, and
  compiles with the same back end.
- `get_performance` checks each build against the kernel's oracle; a correct program more than 3% slower than the
  fastest one measured is rolled back to it (`plugin/tactics/prompt/passes/*.md`).
- Only CPU kernels take a pipeline (`src.device == "cpu"` in `Session.default_kernel_library`).

So ingestion, the lowering, the back end, the oracle check and the rollback exist. What the baselines lack is the
step between: the agent writes one opaque string, sees no loops, and learns only a time.

## Design

### The program

The program of the agent is the kernel's CPF unit plus two lists of moves:

```python
@dataclass(slots=True, frozen=True)
class PassMove:
    kind: str  # "tile", "fuse", "vectorize", ...
    arguments: tuple[tuple[str, str], ...]
    text: str  # the pass string the harness built from kind and arguments


# Session state, replacing self.pipeline for this agent
self.mlir_moves: tuple[PassMove, ...]
self.llvm_moves: tuple[PassMove, ...]
```

The MLIR pipeline is `builtin.module(func.func(<move 1>,canonicalize,cse,<move 2>,canonicalize,cse,...))`, the LLVM
pipeline is the LLVM moves joined by `,`. `build_mlir_pipeline` then hands its IR to `opt` with the LLVM pipeline
before the back end: one new function, `build_mlir_llvm_pipeline`, that is the two existing functions in sequence.
Rollback stays `program_change`: a move is a change of the two tuples.

### Calls

| Call | Does |
|---|---|
| `view_ir` | runs the pipeline up to now and prints one row per loop: `for<n>` (affine), `sfor<n>` (scf), `par<n>` (parallel), with bounds; after the switch to LLVM, one row per loop of LoopInfo with its trip count and vector width. `mlir=[labels]` or `llvm=[labels]` prints those loops' IR |
| `list_moves` | the moves of the current phase, their arguments and the pass each runs |
| `apply_move kind arguments` | appends one move; answers `applied`, `no change` (the IR came out identical, nothing is recorded), or `failed: <tool output>` |
| `rollback` | the existing call |
| `advance_phase` | from MLIR to LLVM; MLIR moves are closed after it |
| `get_performance`, `finalize`, `score`, `submit` | the existing calls; the oracle check stays at `get_performance` |

The labels are the loop's position in a preorder walk of the outline, recomputed after each move. A move that
retires a loop (tiling makes two) gives new labels; the outline after the move shows them.

### Moves

The moves are the survey's lists, restricted to the passes `polygeist-opt` and `opt` register. Each kind maps to one
pass string; the arguments fill the pass options.

- MLIR (`polygeist-opt`): `fuse` (`affine-loop-fusion{mode=}`), `tile` (`affine-loop-tile{tile-sizes=}`), `unroll`
  (`affine-loop-unroll`), `unroll-jam`, `scalar-replace` (`affine-scalrep`), `hoist-invariant`
  (`affine-loop-invariant-code-motion`), `copy-to-buffer` (`affine-data-copy-generate`), `vectorize`
  (`affine-super-vectorize`), `coalesce`, `peel` (`scf-for-loop-peeling`), `parallelize` (`affine-parallelize`).
- LLVM (`opt`): `cleanup`, `hoist-invariant` (`licm`), `interchange`, `fuse` (`loop-fusion`), `distribute`, `flatten`,
  `unroll-jam`, `vectorize` (`loop-vectorize` with forced width and interleave), `slp`, `unroll`,
  `version-aliasing`, `prefetch`, `polly`.
- `cl::opt` knobs (`-force-vector-width`, `-unroll-count`, `-polly-default-tile-size`) are process-wide, so two LLVM
  moves of one kind with different knobs cannot share one `opt` call. The harness runs one `opt` per LLVM move, each
  with its own knobs.

The CPU threads come from `convert-scf-to-openmp` in the fixed lowering, as today: a `par` loop after `parallelize`
becomes an OpenMP loop. There is no placement phase in the first version.

### Answers to the draft's open questions

| # | Question | Answer |
|---|---|---|
| 1 | Targeting one loop | Whole-function passes. A CPF kernel unit holds one scope's nest, so a function-wide pass already acts on one nest. Per-loop transform scripts wait until a kernel shows two nests that need different arguments |
| 2 | Cleanup after MLIR moves | Append `canonicalize,cse` to every MLIR move, as the draft does: the outline stays readable and no move depends on debris |
| 3 | Aliasing lost in cgeist | Re-attach `noalias` in the lowering: DaCe's units mark every array parameter `__restrict__`, and the entry's parameters are known, so the harness sets `llvm.noalias` on the `llvm.func` arguments before `mlir-translate` |
| 4 | Fortran | Not a question here: every kernel reaches the agent as NestForge's C++ unit, whatever language the program came from |
| 7 | A raw pipeline call | No: the two baselines already offer raw strings (`set_mlir_passes`, `set_llvm_passes`). This agent offers named moves only, so its column measures the move interface |
| 8 | The reference | NestForge's default build of the kernel, as for the two baselines; `default<O3>` is not a move |
| 9 | linalg and the transform dialect | Later: cgeist emits no linalg from loop code |
| 10 | One agent or two | One agent, two phases, on NestForge's kernel units |

Questions 5 and 6 (GPU) are the owner's; see the end.

## Tests

- `tests/test_mlir_llvm_moves.py`, each on a small kernel (`jacobi_1d`, a 2D stencil, a matrix product):
  1. the outline of an ingested kernel lists its loops with their bounds;
  2. `tile` on the 2D stencil gives four `for` rows, and the program still matches its oracle;
  3. a move that changes nothing answers `no change` and records nothing;
  4. `rollback` after two moves gives back the IR and labels after the first;
  5. an LLVM `vectorize` with width 4 shows `vec=4x1` in the outline;
  6. a pass that fails (an option out of range) answers `failed` with the tool's message and changes nothing;
  7. two LLVM moves with different knobs keep their own knobs (two `opt` calls).
- The tests need `cgeist`, `polygeist-opt` and `mlir-translate`; they skip with that reason where the machine lacks
  them, as `build_mlir_pipeline` refuses today. They run in the bench image, which builds Polygeist on LLVM 22.1.8.

## Done

The plugin offers the agent as a packet beside the two baselines, the skill (`SKILL.md`, updated to these calls) is
in `plugin/tactics/skills/`, the tests above are green in CI, and one episode on `jacobi_2d` reaches `submit`.

## Decisions for the owner

1. **GPU kernels (draft questions 5 and 6).** The first version is CPU only, like the two baselines. A GPU version
   needs `gpu-map-parallel-loops`, `convert-parallel-loops-to-gpu` and the NVVM lowering, explicit copies or managed
   memory for the arrays, and `libmlir_cuda_runtime.so` in the bench image. Recommendation: CPU only for the study,
   and GPU when the study's GPU configurations need a compiler baseline.
2. **Whether this agent is a study column.** It costs one more configuration per kernel. Recommendation: run it on
   `scicomp40` only, beside the LLVM baseline (B13), and keep the raw-string MLIR baseline out of the study if the
   budget needs a cut.
