# NestForge

NestForge optimizes whole DaCe programs on CPU and GPU. It canonicalizes a program, reshapes its loop nests with
legal moves, cuts it into kernels, places each kernel on a device, builds and sweeps each kernel, and reports what
to change next. Every kernel is checked against its Python oracle, the whole program against the benchmark's
reference.

Every stage has a deterministic default, so a run works with no agent, one agent or a human driving the same
`Session` API. Each call returns a typed record (`nestforge/records.py`, frozen pydantic models).

| Stage | Decides | Default | Call |
|---|---|---|---|
| [1 Canonicalize](docs/stages.md#stage-1-canonicalize) | canonical parallel form (CPF) | DaCe canonicalization up to fusion | |
| [2 Moves](docs/stages.md#stage-2-moves) | fusion, fission, interchange | canonicalization's fusion stage | `apply_move` |
| [3 Scopes](docs/stages.md#stage-3-scopes) | which regions become kernels | parallel maps; sequential and mixed loops whole; large maps split | `set_scope`, `unset_scope` |
| [4 Placement](docs/stages.md#stage-4-placement) | device per kernel | parallel on GPU, sequential on CPU | `place` |
| [5 Kernels](docs/stages.md#stage-5-kernels) | each kernel's code | CPF C++ (CPU) or CUDA (GPU) | `set_kernel_source` |
| [6 Variants](docs/stages.md#stage-6-variants) | compiler, FP mode, cost model | fastest variant that matches the oracle | `sweep` |
| [7 Feedback](docs/stages.md#stage-7-feedback) | what to try next | hints, whole-program and kernel times, offloading | `performance` |

## Agents: the HPCAgent-Bench plugin

[plugin/nestforge](plugin/nestforge) is an HPCAgent-Bench plugin. Point `HPCAGENT_BENCH_USER_PLUGIN_DIRS` at it and
run the `nestforge` packet; it sets the library input mode itself (`finalize` hands the judge a library). The `nestforge` packet
exposes the moves phase; scopes, placement and kernel builds run their defaults in `get_performance` and
`finalize`. The `nestforge-full` packet exposes every phase: the agent may also set scopes and write and tune kernels.
The skill page ([skills/nestforge](plugin/nestforge/skills/nestforge/SKILL.md)) explains the tree and the moves. Each
tool is listed with MCP annotations: reads are read-only, `write_kernel` and `unset_scope` overwrite earlier work.

| Phase | Tools | Packets |
|---|---|---|
| structured (stages 1-2) | `get_cpf`, `view_tree`, `list_move_kinds`, `list_moves`, `apply_move`, `run_defaults` | both |
| any | `get_performance`, `finalize` | both |
| scopes, kernels, codegen | `set_scope`, `unset_scope`, `list_kernels`, `read_kernel`, `write_kernel`, `tune_kernel` | `nestforge-full` |

`apply_move`, `run_defaults`, `write_kernel`, `tune_kernel` and `get_performance` answer with
`Session.performance()`: the whole program's time and
correctness, each kernel's device and the bytes moved between memories, and each kernel's time and share. Stages not
reached yet run with their defaults on a copy. The answer is reused until the program changes. `finalize` builds the
program in the bench C ABI, each kernel linked from its own build, into the shared folder; submit it as
`library=<library>` with its `workspace_bytes`.

## Quick start

```bash
python examples/quickstart.py --device cpu --out quickstart_out
python examples/quickstart.py --device gpu --kernel jacobi_1d --out quickstart_jacobi
```

It runs every default on an HPCAgent-Bench kernel at preset S and saves each stage's SDFG and tree, the kernel
DAG, each kernel's unit and library, the program's code, the winning variant and the feedback. On `fuse_diamond`
canonicalization turns four sequential loops into one parallel map. A benchmark's reference solution is replayed
with `python -m nestforge.replay reference_solutions/<kernel>.py` ([reference solutions](reference_solutions/README.md)).
On `fuse_diamond`:

```
SDFG 'hpcagent_bench_benchmarks_loop_level_reasoning_fuse_diamond_fuse_diamond_dace_fuse_diamond'
`- state0
   `- doall0  [_loop_it_0=0:LEN_1D]  reads=['a'] writes=['out']
```

## Install and test

```bash
sudo bash scripts/setup_apt.sh                                          # compilers, libomp, BLAS, binutils (Ubuntu)
pip install -e ".[dev]"                                                 # or: uv pip install -e ".[dev]"
pre-commit install                                                      # ruff check, ruff format, headers
pytest -m "not integration and not cloudsc and not gpu and not vendor"  # unit set, as CI runs it
pytest -m integration                                                   # compiles and runs; bench OpenMP env (CI)
pytest -m cloudsc                                                       # every phase on cloudsc_monolith vs the bench (CI)
```

HPCAgent-Bench (`main`) is pinned to a commit SHA in `pyproject.toml`, and DaCe and relayout to the commits that bench
commit pins (the bench owns them); `python scripts/bump_deps.py [bench-branch]` moves all three. NestForge is a user
plugin of the bench (`plugin/nestforge`, through `HPCAGENT_BENCH_USER_PLUGIN_DIRS`); the bench never imports it.
NestForge assumes Linux.
A pip install of the bench does not ship the KernelBench submodule, so an ML kernel (`conv2d_*`) has no PyTorch
reference and its outputs go unchecked (`correct: null`). Point `HPCAGENT_BENCH_ML_KERNELBENCH_DIR` at an
initialized KernelBench checkout to check them.

## Layout

```
nestforge/
  session.py   the one API over all stages
  replay.py    replay a benchmark's reference-solution script next to the default, timed and oracle-checked
  records.py   the typed answers and their status enums
  stages/      canonicalize, moves, scopes, placement, kernel, variants, feedback, performance
  ir/          extraction, the Python oracle, the kernel DAG, structure trees
  build/       compile and link, compilers on PATH, FP flags, validate-and-time, fork isolation
  corpus/      HPCAgent-Bench kernels and the oracle bridge
plugin/nestforge/  the HPCAgent-Bench plugin: tools, judge route, method prompt
reference_solutions/  one reference-solution script per benchmark, replayed by nestforge/replay.py
```

More: [Python oracle](docs/oracle.md), [kernel DAG](docs/depends.md), [build and runtime linking](docs/build.md),
[FP modes and cost models](docs/fp-and-vectorization.md). Contributors follow [AGENTS.md](AGENTS.md).

## References

- Stage 1 builds on *The Canonical Parallel Form as a Substrate for Parallelizing Compilers and Agentic
  Optimizers*, which defines CPF.
- The benchmark kernels and references come from *HPCAgent-Bench*.
- Stage 6 is the variant search of *The Data Must Flow (To Vector Processors): Searching Program Variants to
  Improve Compiler Auto-Vectorization Capabilities* (ICS'26).
