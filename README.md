# NestForge

NestForge optimizes whole DaCe programs on CPU and GPU. It canonicalizes a program, reshapes its loop nests with
legal moves, cuts it into kernels, places each kernel on a device, builds and sweeps each kernel, and reports what
to change next. Every kernel is checked against its Python oracle, the whole program against the benchmark's
reference.

Every stage has a deterministic default, so a run works with no agent, one agent or a human driving the same
`Session` API. Each call returns a typed record (`nestforge/records.py`, frozen pydantic models).

| Stage | Decides | Default | Call |
|---|---|---|---|
| [1 Canonicalize](docs/stages/1-canonicalize.md) | canonical parallel form (CPF) | DaCe canonicalization up to fusion | |
| [2 Moves](docs/stages/2-moves.md) | fusion, fission, interchange | canonicalization's fusion stage | `apply_move` |
| [3 Scopes](docs/stages/3-scopes.md) | which regions become kernels | one per parallel top-level map | `set_scope`, `unset_scope` |
| [4 Placement](docs/stages/4-placement.md) | device per kernel | parallel on GPU, sequential on CPU | `place` |
| [5 Kernels](docs/stages/5-kernels.md) | each kernel's code | CPF C++ (CPU) or CUDA (GPU) | `set_kernel_source` |
| [6 Variants](docs/stages/6-variants.md) | compiler, FP mode, cost model | fastest variant that matches the oracle | `sweep` |
| [7 Feedback](docs/stages/7-feedback.md) | what to try next | hints, whole-program and kernel times, offloading | `performance` |

## Agents: the HPCAgent-Bench plugin

[plugin/nestforge](plugin/nestforge) is an HPCAgent-Bench plugin. Point `HPCAGENT_BENCH_USER_PLUGIN_DIRS` at it and
run the `nestforge` packet. Its tools cover three phases; only the structured phase is served today, the others
answer 501:

| Phase | Tools |
|---|---|
| structured (stages 1-4) | `get_cpf`, `view_tree`, `list_moves`, `apply_move`, `run_defaults`, `set_scope`, `unset_scope` |
| unstructured (stage 5) | `list_kernels`, `read_kernel`, `write_kernel` |
| codegen (stage 6) | `tune_kernel` |
| any | `get_performance`, `finalize` |

`apply_move`, `run_defaults` and `get_performance` answer with `Session.performance()`: the whole program's time and
correctness, each kernel's device and the bytes moved between memories, and each kernel's time and share. Stages not
reached yet run with their defaults on a copy. The answer is reused until the program changes.

## Quick start

```bash
python examples/quickstart.py --device cpu --out quickstart_out
python examples/quickstart.py --device gpu --kernel jacobi_1d --out quickstart_jacobi
```

It runs every default on an HPCAgent-Bench kernel at preset S and saves each stage's SDFG and tree, the kernel
DAG, each kernel's unit and library, the program's code, the winning variant and the feedback. On `fuse_diamond`
canonicalization turns four sequential loops into one parallel map:

```
SDFG 'hpcagent_bench_benchmarks_loop_level_reasoning_fuse_diamond_fuse_diamond_dace_fuse_diamond'  epoch=1
`- state0_0
   `- kernel1_0  [_loop_it_0=0:LEN_1D]  reads=['a'] writes=['out']
```

## Install and test

```bash
sudo bash scripts/setup_apt.sh                           # compilers, libomp, BLAS/LAPACK, binutils (Ubuntu)
pip install -e ".[dev]"                                  # or: uv pip install -e ".[dev]"
pre-commit install                                       # ruff check, ruff format, headers
pytest -m "not integration and not gpu and not vendor"   # unit set, as CI runs it
pytest -m integration                                    # compiles and runs; needs the bench's OpenMP env (CI)
```

DaCe (`extended`) and HPCAgent-Bench (`main`) are pinned to commit SHAs in `pyproject.toml`;
`python scripts/bump_deps.py` moves both to their branch tips. NestForge assumes Linux.

## Layout

```
nestforge/
  session.py   the one API over all stages
  records.py   the typed answers and their status enums
  stages/      canonicalize, moves, scopes, placement, kernel, variants, feedback, performance
  ir/          extraction, the Python oracle, the kernel DAG, structure trees
  build/       compile and link, compilers on PATH, FP flags, validate-and-time, fork isolation
  corpus/      HPCAgent-Bench kernels and the oracle bridge
plugin/nestforge/  the HPCAgent-Bench plugin: tools, judge route, method prompt
```

More: [Python oracle](docs/oracle.md), [kernel DAG](docs/depends.md), [build and runtime linking](docs/build.md),
[FP modes and cost models](docs/fp-and-vectorization.md). Contributors follow [AGENTS.md](AGENTS.md).

## References

- Stage 1 builds on *The Canonical Parallel Form as a Substrate for Parallelizing Compilers and Agentic
  Optimizers*, which defines CPF.
- The benchmark kernels and references come from *HPCAgent-Bench*.
- Stage 6 is the variant search of *The Data Must Flow (To Vector Processors): Searching Program Variants to
  Improve Compiler Auto-Vectorization Capabilities* (ICS'26).
