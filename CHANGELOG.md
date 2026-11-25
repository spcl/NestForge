# Changelog

## 0.1.1 (unreleased)

### Added

- **Pass-move agent** (`nestforge-passes`, [design](docs/design/mlir-llvm-agent.md)): MLIR moves on the affine loops
  Polygeist raises from each CPU kernel, then LLVM moves on the IR the fixed lowering writes, one named pass at a
  time (cgeist at -O2, so the unit's index helpers are inlined and its accesses affine); `view_ir` shows the loops
  after each move, a move that changes nothing or fails is refused, and rollback takes
  a move back. GPU kernels wait on the bench image's MLIR CUDA runtime (bench request B16). On a kernel whose sizes
  are runtime arguments, cgeist multiplies each index by the row length, which MLIR's analyses cannot see through:
  tiling and unrolling apply there, `parallelize` and most other MLIR moves change nothing.

### Fixed

- `set_scope` on a solver's step state left a scalar the kernel takes read by an interstate edge (#111).
- `set-schedule-gpu` with a block shorter than the doall's rank pads it with ones, as CUDA reads `dim3` (#112).
- `run-on-gpu-persistent` sets the megakernel's block size on the loops it keeps, so the render no longer conflicts
  with their thread-block maps (#113).
- `list_moves` and `list_scope_moves` inside a kernel whose body holds an `if` no longer raise `KeyError: None`:
  the walk over regions skips a conditional, whose branches are regions of their own (#110, thanks to Nicolas
  Pereira Ribeiro, #115).
- The MLIR baseline builds again in the judge image: cgeist gets the compiler's builtin headers (`stddef.h`), and
  the example pipelines nest function passes under `func.func`.
- `srad`'s default program fails to build with DaCe extended 1.0.0 (CPF declares no `tmp61` in its loop kernel);
  DaCe extended 1.0.1 fixes it, and tactics requires it (#114).

### Changed

- `list_moves` copies only the containers a candidate's scope touches when it probes snapshot reads: the CloudSC
  listing takes about a fifth of its former time.
- The cleanup after a move leaves out DaCe extended 1.0.1's `ForwardCopyMaps`, which would undo the snapshot copies
  a move makes; canonicalization keeps it.
- CI runs in about half the time: duplicate and low-value tests merged or removed, end-to-end emitter checks cut to
  the kernels that cover every emitter branch.

## 0.1.0 (2026-11-23)

The first release, published on PyPI as `tactics` (formerly NestForge). Tactics lets an agent optimize a whole DaCe
program only through legal, checked moves: a tactic is a transformation rule, a move applies one tactic to labeled
rows of the program, and the ordered moves of an episode are its recipe, which replays the result.

### What it contains

- **Session API** (`tactics.Session`): one object over every stage, with typed, frozen answers.
- **Stages**, each with a deterministic default: canonicalization into the Canonical Parallel Form, moves on loop
  nests (tiling, interchange, fusion and fission, vectorization, library expansion, layout), kernel scopes,
  placement on CPU or GPU, kernel builds and compiler-variant sweeps, performance measurement, and feedback that names
  the tree label behind each hot kernel and failed build.
- **Program tree**: stable row labels that only grow, expansions that persist across moves, and move answers that
  return only the rewritten block of the tree.
- **Correctness**: every kernel is checked against its Python oracle, with a tolerance for reordered accumulations
  matching the benchmark's grading, and the whole program against the benchmark's reference.
- **GPU**: CUDA and HIP kernels, and mixed CPU and GPU programs, built into one library the judge grades.
- **Communication**: MPI library nodes whose start and wait move independently, so a program can overlap an exchange
  with compute.
- **HPCAgent-Bench plugin** (`plugin/tactics`, registered as `nestforge`): agent tools, the judge route, prompts,
  and the packets `nestforge`, `nestforge-full`, `nestforge-subagents`, `nestforge-requests`, and the LLVM and MLIR
  pass baselines (`nestforge-llvm`, `nestforge-mlir`).
- **Replay**: `python -m tactics.replay` replays a benchmark's reference-solution script next to the default.
- **Scripts**: token accounting of a run's answers (`scripts/answer_tokens.py`) and per-run statistics
  (`scripts/run_statistics.py`).

### Install

```bash
pip install tactics          # DaCe extended comes as dace-extended (import dace)
```

From a checkout of https://github.com/spcl/tactics, every dependency comes at its pinned commit:

```bash
uv sync --group dev
```

The agent plugin (`plugin/tactics`) ships with the checkout, not the wheel: point `HPCAGENT_BENCH_USER_PLUGIN_DIRS` at
it. The relayout move's SDFG cast needs FP-Arena: `pip install "tactics[fparena]"`.
Tactics requires Python 3.12 or newer and Linux.
