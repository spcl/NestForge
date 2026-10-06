# NestForge

NestForge is an agentic optimization harness for whole DaCe programs on CPU and GPU. It canonicalizes a program,
lets a small agent reshape its loop nests with legal transformations, cuts the program into kernels, places each
kernel on a device, builds and sweeps each kernel, and reports back what to change. Every result is checked against
the kernel's Python oracle.

Each stage makes one decision and ships a deterministic default, so a run works with no agent, one agent, or a
human driving the same `Session` API.

| Stage | Decides | Default | Agent |
|---|---|---|---|
| [1 Canonicalize](docs/stages/1-canonicalize.md) | canonical parallel form (CPF) | DaCe canonicalization up to fusion | none |
| [2 Moves](docs/stages/2-moves.md) | fusion, fission, interchange | canonicalization's fusion stage | `apply_move` |
| [3 Scopes](docs/stages/3-scopes.md) | which regions become kernels | one per parallel top-level map | `define_scope` |
| [4 Placement](docs/stages/4-placement.md) | device per kernel | parallel on GPU, sequential on CPU | `place` |
| [5 Kernels](docs/stages/5-kernels.md) | each kernel's code | CPF C++ (CPU) or CUDA (GPU) | `set_kernel_source` |
| [6 Variants](docs/stages/6-variants.md) | compiler, FP mode, cost model | fastest variant that matches the oracle | none |
| [7 Feedback](docs/stages/7-feedback.md) | what to try next | ranked hints from times and compiler remarks | optional analyst |

The built-in [agent loop](docs/agent.md) drives stages 2 to 5 with the OpenAI or Anthropic API, rebuilding a
compact prompt every turn from the program view and the latest feedback.

## Quick start

```bash
python examples/quickstart.py --device cpu --out quickstart_out
python examples/quickstart.py --device gpu --out quickstart_out
python examples/quickstart.py --device cpu --kernel jacobi_1d --out quickstart_jacobi
python examples/quickstart.py --device cpu --agent anthropic --model claude-opus-5-5 --out quickstart_agent
```

The first runs every default on HPCAgent-Bench's `fuse_diamond` at preset S (`LEN_1D=512`): four loops where
`t = a*a` feeds `u = t + 1` and `v = t - 1`, then `out = u*v`. It prints the structure tree before stage 1:

```
SDFG 'hpcagent_bench_benchmarks_loop_level_reasoning_fuse_diamond_fuse_diamond_dace_fuse_diamond'  epoch=0
|- for0_0  i=0:LEN_1D
|  `- state1_0
|- for0_1  i=0:LEN_1D
|  `- state1_1
|- for0_2  i=0:LEN_1D
|  `- state1_2
`- for0_3  i=0:LEN_1D
   `- state1_3
```

and after it, where the four sequential loops are one parallel map:

```
SDFG 'hpcagent_bench_benchmarks_loop_level_reasoning_fuse_diamond_fuse_diamond_dace_fuse_diamond'  epoch=1
`- state0_0
   `- kernel1_0  [_loop_it_0=0:LEN_1D]  reads=['a'] writes=['out']
```

Stage 2 finds nothing left to fuse, stage 3 makes the map one kernel `extcall_0`, stage 4 keeps it on the CPU,
stage 5 renders it as one CPF C++ unit with `extern "C" extcall_0(a, out, LEN_1D)`, and stage 6 times 15 CPU
variants. Stage 7 reports the kernel's time and any loop the compiler did not vectorize.

`--kernel jacobi_1d` keeps two maps in the time loop, since each reads the other's neighbours:

```
`- for0_0  _loop_it_0=1:TSTEPS
   `- state1_0
      |- kernel2_0  [_loop_it_1=0:N - 2]  reads=['A'] writes=['B']
      `- kernel2_1  [_loop_it_4=0:N - 2]  reads=['B'] writes=['A']
```

Stage 3 makes two kernels. The [kernel DAG](docs/depends.md) shows the time loop carrying `A` from the second
kernel back to the first; `program` is the value before the first step:

```
extcall_0: A <- extcall_1.A | program [carried: for0_0], N <- program
extcall_1: B <- extcall_0.B, N <- program
exit: A <- extcall_1.A | program, B <- extcall_0.B | program
```

and stage 7 reports both kernels' times. The script saves one `.sdfg` per stage, the
structure trees, the kernel DAG, each kernel's unit and `lib<kernel>.a`, the program's generated code, the winning
stage 6 configuration as JSON and the feedback report.

## Install and test

```bash
sudo bash scripts/setup_apt.sh                           # compilers, libomp, BLAS/LAPACK, binutils (Ubuntu)
pip install -e ".[dev,agent]"                            # or: uv pip install -e ".[dev,agent]"
pre-commit install                                       # ruff check, ruff format, headers
pytest -m "not integration and not gpu and not vendor"   # unit set, as CI runs it
pytest -m integration                                    # compiles and runs kernels
```

DaCe (`extended`) and HPCAgent-Bench (`main`) are pinned to commit SHAs in `pyproject.toml`;
`python scripts/bump_deps.py` moves both to their branch tips. NestForge assumes Linux. HPCAgent-Bench drives
NestForge, so `import nestforge` never loads HPCAgent-Bench.

## Layout

```
nestforge/
  session.py   the one API over all stages
  stages/      canonicalize, moves, scopes, placement, kernel, variants, feedback
  agent/       the minimal OpenAI / Anthropic agent loop
  ir/          extraction, the Python oracle, the kernel DAG, the ExternalCall node, structure trees
  build/       compile and link, compilers on PATH, FP flags, validate-and-time, fork isolation
  corpus/      HPCAgent-Bench kernels and the translator bridge
```

More: [Python oracle](docs/oracle.md), [kernel DAG](docs/depends.md), [build and runtime linking](docs/build.md),
[FP modes and cost models](docs/fp-and-vectorization.md). Contributors and agents follow [AGENTS.md](AGENTS.md).

## References

- Stage 1 builds on *The Canonical Parallel Form as a Substrate for Parallelizing Compilers and Agentic
  Optimizers*, which defines CPF.
- The benchmark kernels and translators come from *HPCAgent-Bench*.
- Stage 6 is the variant search of *The Data Must Flow (To Vector Processors): Searching Program Variants to
  Improve Compiler Auto-Vectorization Capabilities* (ICS'26).
