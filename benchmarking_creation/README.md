# benchmarking_creation

Benchmark kernels that live in this repository and not in `hpcagent_bench`'s corpus, so a new benchmark can be written, run and tuned here before (or without) a pull request to HPCAgent-Bench. NestForge loads them next to the corpus: `examples/quickstart.py --kernel <name>` lists them, and `nestforge.corpus.bench.iter_dace_kernels("benchmarking_creation")` yields them.

## Layout

One folder per kernel, with the same files as an HPCAgent-Bench kernel:

```
benchmarking_creation/<name>/
  <name>_numpy.py     NumPy oracle; the entry function is <name>(...)
  <name>.yaml         manifest: name, level, parameters (presets S/M/L/XL), init, output_args
  <name>_dace.py      @dace.program, generated from <name>_numpy.py, reviewed and committed
  <name>.md           explanation and measured reference solution (see AGENTS.md)
```

Anything else in the folder (a plain-C reference, a validation script) is ignored by the loader. The registry key is `benchmarking_creation/<name>/<name>`. Agents that add a kernel follow [AGENTS.md](AGENTS.md) in this folder.

## Manifest

Besides the usual fields, the manifest sets three fields that hpcagent_bench derives from the folder inside its own corpus and cannot derive here: `relative_path: benchmarking_creation/<name>`, `func_name: <name>`, and `input_args`, the positional parameter names of the oracle function in order.

## Adding a kernel

1. Write `<name>_numpy.py` and `<name>.yaml`.
2. Generate the DaCe program: `python scripts/emit_local_dace.py <name>`. NestForge never regenerates it when it loads a kernel, so review and commit it. Deleting its first line (the `hpcagent_bench-autogen` marker) turns it into a hand-written override.
3. `python examples/quickstart.py --kernel <name> --device cpu` runs the seven stages.

`benchmarking_creation/` is excluded from ruff (`pyproject.toml`), because generated and reference code follows hpcagent_bench's layout.
