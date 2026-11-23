# Changelog

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
