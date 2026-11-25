# MLIR and LLVM agent

[Designs](README.md) · [Stages](../stages.md) · [Bench requests](bench-requests.md)

Status: CPU part built (2026-11-24): `tactics/build/pass_moves.py`, the session calls below, and the
`tactics-passes` packet. The GPU moves wait on bench request B16. Written against Tactics `b98494f`; it builds on
the survey of 2026-11-19 (`nestforge-workspace/notes/mlir-llvm-agent/survey.md`, pass names checked in the LLVM 22.1.2
headers and tools) and the skill draft beside it (`SKILL.md`).

## Goal

An agent rewrites a kernel by applying compiler passes one at a time, the way the Tactics agent applies moves:
each pass is a named move with arguments, the agent sees the loops after each move, it can roll a move back, and the
reference checks the result. The agent works in two phases on the same kernel: MLIR passes on affine and scf loops,
then LLVM passes on the IR the fixed lowering writes, for CPU and GPU kernels. It is a baseline for the study: what an
agent reaches with a compiler's own passes, without Tactics' moves. The owner decided on 2026-11-24 that the
first version covers the GPU, and that the agent is a study column on `scicomp40` only.

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

### As built

The moves live in `self.pipeline = ("moves", <JSON of the phase and the moves>)` instead of two new tuples, so
`set_pipeline`'s rollback, the checkpoints and `pipeline_build` take them unchanged (`PIPELINES["moves"]` is
`build_pass_moves`). The tools carry their own names beside Tactics': `view_ir loops=[...]`, `list_pass_moves`,
`apply_pass_move kind arguments` (`kind=rollback` takes the last change back), `advance_pass_phase`. The LLVM outline
prints each loop's header block and `vec=W`, the widest vector its header computes on; the trip count is left out.

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
becomes an OpenMP loop.

### GPU kernels

A kernel the program places on the GPU takes device pointers: Tactics' placement inserts the copies before and
after the kernel (`Session.transfers`), so the MLIR route needs no copies of its own. The agent starts from the same
CPF C++ unit a CPU kernel has, and the MLIR phase gains three moves:

- `tile-parallel` `tile_sizes`: `scf-parallel-loop-tiling{parallel-loop-tile-sizes=}`. The outer loops of a tiled
  `par` become blocks, the inner loops threads.
- `map-to-gpu`: `gpu-map-parallel-loops`, `convert-parallel-loops-to-gpu`, `gpu-kernel-outlining`. Each mapped `par`
  nest becomes one kernel launch. The move is refused when the kernel is placed on the CPU, and a GPU kernel with no
  mapped loop is refused at `advance_phase`, because it would run on one GPU thread.
- `gpu-barriers`: `gpu-eliminate-barriers`, after `copy-to-buffer` put tiles in shared memory.

The fixed lowering of a GPU kernel adds `nvvm-attach-target{chip=<the target's sm> O=3}`, `convert-gpu-to-nvvm`,
`gpu-to-llvm` and `gpu-module-to-binary{format=fatbin}`. The device code is compiled there, so the LLVM phase sees
the host code only, and its moves are refused on a GPU kernel except `cleanup` and `inline`. The host code calls
the `mgpu*` functions of `libmlir_cuda_runtime.so`, which the kernel's library links. The bench image builds MLIR
without it today: request B16 in [bench-requests.md](bench-requests.md).

### Answers to the draft's open questions

| # | Question | Answer |
|---|---|---|
| 1 | Targeting one loop | Whole-function passes. A CPF kernel unit holds one scope's nest, so a function-wide pass already acts on one nest. Per-loop transform scripts wait until a kernel shows two nests that need different arguments |
| 2 | Cleanup after MLIR moves | Append `canonicalize,cse` to every MLIR move, as the draft does: the outline stays readable and no move depends on debris |
| 3 | Aliasing lost in cgeist | Re-attach `noalias` in the lowering: DaCe's units mark every array parameter `__restrict__`, and the entry's parameters are known, so the harness sets `llvm.noalias` on the `llvm.func` arguments before `mlir-translate` |
| 4 | Fortran | Not a question here: every kernel reaches the agent as Tactics' C++ unit, whatever language the program came from |
| 7 | A raw pipeline call | No: the two baselines already offer raw strings (`set_mlir_passes`, `set_llvm_passes`). This agent offers named moves only, so its column measures the move interface |
| 8 | The reference | Tactics' default build of the kernel, as for the two baselines; `default<O3>` is not a move |
| 9 | linalg and the transform dialect | Later: cgeist emits no linalg from loop code |
| 10 | One agent or two | One agent, two phases, on Tactics' kernel units |

| 5 | GPU data | None to move: a GPU kernel's arguments are device pointers, placed by Tactics (owner, 2026-11-24: GPU in the first version) |
| 6 | The CUDA runtime of MLIR | Request B16: the bench image builds MLIR with the CUDA runtime |

## Tests

- `tests/test_mlir_llvm_moves.py`, each on a small kernel (`jacobi_1d`, a 2D stencil, a matrix product):
  1. the outline of an ingested kernel lists its loops with their bounds;
  2. `tile` on the 2D stencil gives four `for` rows, and the program still matches its oracle;
  3. a move that changes nothing answers `no change` and records nothing;
  4. `rollback` after two moves gives back the IR and labels after the first;
  5. an LLVM `vectorize` with width 4 shows `vec=4` in the outline;
  6. a pass that fails (an option out of range) answers `failed` with the tool's message and changes nothing;
  7. two LLVM moves with different knobs keep their own knobs (two `opt` calls);
  8. on a GPU target, `tile-parallel` and `map-to-gpu` on `jacobi_2d` give one kernel launch per nest, and the
     program matches its oracle (`gpu` mark, skipped without a GPU);
  9. `map-to-gpu` on a kernel placed on the CPU is refused, and `advance_phase` refuses a GPU kernel with no mapped
     loop.
- The tests need `cgeist`, `polygeist-opt` and `mlir-translate`; they skip with that reason where the machine lacks
  them, as `build_mlir_pipeline` refuses today. They run in the bench image, which builds Polygeist on LLVM 22.1.8.

## Done

The plugin offers the agent as a packet beside the two baselines, the skill (`SKILL.md`, updated to these calls) is
in `plugin/tactics/skills/`, the tests above are green in CI, and one episode on `jacobi_2d` reaches `submit` on the
CPU and one on an NVIDIA GPU.

## Decisions

Taken by the owner on 2026-11-24:

1. The first version covers CPU and GPU kernels (draft questions 5 and 6).
2. The agent is a study column on `scicomp40` only, beside the LLVM baseline (B13).
