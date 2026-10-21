# AGENTS.md: writing a benchmark

Read this before adding or changing anything under `benchmarking_creation/`. A kernel that is missing a file below is not a benchmark. The loader ignores the explanation and the reference; they are still required.

These kernels follow the HPCAgent-Bench layout, so the same oracle, manifest and generated DaCe program can later move into that corpus. They live here first, loaded by `examples/quickstart.py --kernel <name>` and `nestforge.corpus.bench.iter_dace_kernels("benchmarking_creation")`.

## Files

One folder per kernel, registry key `benchmarking_creation/<name>/<name>`:

```
benchmarking_creation/<name>/
  <name>_numpy.py     NumPy oracle. The entry function is <name>(...).
  <name>.yaml         Manifest, same fields as an HPCAgent-Bench kernel.
  <name>_dace.py      @dace.program generated from the oracle, reviewed and committed.
  <name>.md           Explanation, including the reference solution. Required.
```

Anything else in the folder (a plain-C reference, a validation script, an input generator) is ignored by the loader. An input generator is an `initialize()` in a separate `<name>.py`; the loader finds it there, and the manifest does not name it.

### Oracle and manifest

Write `<name>_numpy.py` and `<name>.yaml` first. The manifest follows HPCAgent-Bench's schema (from `e63f9f1`): set `name`, `level`, `parameters` (presets S, M, L, XL), `args`, and `chain_length` when an output is a scan or a carry. `args` declares every array and scalar of the oracle: a shape string (`aa: "(N, M)"`), or a mapping with `shape`, `out: true` for a graded output, `dtype`, `domain: [low, high]`. Quote a shape inside a `{...}` mapping, or YAML splits it at the comma. Do not set `relative_path`, `func_name`, `input_args`, `output_args` or `init.func_name`: they are derived (the entry is the oracle's only top-level function, or the one named `<name>`; its parameters give the order), and the schema refuses them. `nestforge.corpus.bench.local_spec` does the derivation for a folder outside HPCAgent-Bench's corpus. Preset S is for correctness. Preset M is the size the gap is measured at, and its leading dimension is not a multiple of a block size. No `bool` arrays: there is no extern-C spelling for them. Use `int32` masks. Do not draw unstructured uniform random inputs for a kernel that divides by a difference of inputs or that needs a monotone pressure or a positive temperature. Put that generator in `initialize()` in `<name>.py`.

Generate the DaCe program and commit it. NestForge does not regenerate it on load.

```bash
python scripts/emit_local_dace.py <name>
```

Review `<name>_dace.py`. Deleting its first line (the `hpcagent_bench-autogen` marker) makes it a hand-written override. Then:

```bash
python examples/quickstart.py --kernel <name> --device cpu
python examples/quickstart.py --kernel <name> --device gpu
```

## What a benchmark is

A fixed program, a NumPy oracle, and one known better schedule that stage 2 can reach. The program starts suboptimal. The gain comes from scheduling only: fusion, fission, interchange, and the scopes those moves make legal. A different algorithm, a runtime specialization, tiling, or a hand-written kernel is out of scope.

The agent under test calls `list_moves` and `apply_move`. It does not edit SDFG nodes or memlets. Stages 1 and 6 stay on their deterministic defaults. Variants differ by the schedule, not by a compiler-flag search, unless the note says the flags are part of the question.

## The explanation

`<name>.md` is the record of the benchmark. Fill every section. A number that was not measured is marked as a guess, with what was measured instead. Record the DaCe `extended` SHA, the NestForge SHA, the HPCAgent-Bench SHA, the device, the thread count, and the preset next to every time.

### What this benchmark is

The program in one paragraph: the arrays, the carried dimensions, which axes are parallel, and the scheduling problem. Say which HPCAgent-Bench or CloudSC source it comes from, or say that it is synthetic.

### What the reference is

Two programs, both checked against the oracle.

- **Baseline.** Stage 1, then `default_moves()`. The tree `describe()` prints.
- **Reference.** The schedule the benchmark claims is the one to reach. The tree after the reference moves and `finish_moves()`.

State which device and thread count this pair is for. If CPU and GPU want different schedules, each pair is its own reference.

### How it was created

The source lines or the kernels it recombines, the edit that creates the gap, and the probes that failed on the way (a move that was illegal, a body stage 1 already solved, a build that died). Link the experiment note when there is one.

### What the solution is

The ordered stage 2 moves, each named by kind and by what the row reads and writes. Labels from `describe()` are recorded from the run that measured the time, and they are not stable across versions. Say which move is illegal until an earlier one applies. Paste the tree after the last move.

### How to optimize it

The moves that close the gap, the order they have to be applied in, and the moves that look useful and are traps (the inverse interchange, a fission that does not unlock anything, a fusion `default_moves()` already took). Say what `list_moves` offers right after stage 1, and what `default_moves()` already does, so the reference is not the default.

### What the gains are

Preset M, device, threads, flags. Baseline and reference: median, min, max, number of repeats, relative error against the oracle. The ratio. The same row at preset S, which is usually launch noise and is not the gap. If a searched schedule beat the reference, say so: the reference is then a designed schedule, not a minimum.

### CPU and GPU

What stage 1 leaves on each device, which moves remain legal, and which device the gap exists on. A kernel whose gap is only on one device is marked that way. A gap that appears only at many threads is reported at 1 thread and at the thread count where it appears. The time is whole-program host time unless the note says it is kernel time only; host-device copies then sit inside the number.

### What else matters

- **Legality.** The refusal text of every move the design expected and did not get.
- **Contiguity and stride.** Which axis is contiguous, and whether the good schedule walks it innermost.
- **Launches against stride.** An interchange that removes launches can lose on a serial CPU and win on a GPU, or the reverse.
- **Vectorization.** Whether the reference body actually vectorized. A schedule that does not vectorize is not evidence for a SIMD claim.
- **Live values and carries.** Scalars carried inside a map, and whether fission had to promote them to arrays before an interchange became legal.
- **Decoys.** A legal move that undoes the gain, and its measured time. A decoy that `finish_moves()` silently undoes is not a measured decoy.
- **Oracle.** The inputs, the relative error, and any limiter or mask that must actually bind.
- **Pins.** DaCe, NestForge, HPCAgent-Bench. A result from another pin is not this benchmark's result.

## Reference solution

Put this block at the end of `<name>.md`, filled from the run, not from the design. One block per device. `time_ms` is the median. `moves` is the sequence that was applied, in order, including `finish_moves` when the build needs it. `tree` is the `describe()` output of the program that was timed.

```yaml
device: gpu
threads: 1
preset: M
dace: <sha>
nestforge: <sha>
hpcagent_bench: <sha>
baseline:
  moves: [default_moves]
  time_ms: 0.0
  time_min_ms: 0.0
  time_max_ms: 0.0
  repeats: 5
  oracle_rel_error: 0.0
  tree: |
    <describe() after stage 1 and default_moves>
reference:
  moves:
    - kind: interchange-loop-map
      on: the row that writes outE
    - kind: interchange-loop-map
      on: the row that writes outW
    - finish_moves: true
  time_ms: 0.0
  time_min_ms: 0.0
  time_max_ms: 0.0
  repeats: 5
  oracle_rel_error: 0.0
  tree: |
    <describe() of the timed program>
```

A benchmark is not finished until this block has measured numbers, the oracle error is filled, and the reference moves were applied on that pin and produced that tree.

## Backtest

Score a later run by the oracle and by preset M time, not by whether its move list equals the reference. Matching the moves is a check that the author can replay the solution. It is not the score.

Keep, next to the reference block, the times of `default_moves()`, of applying every offered move once, and of a few random legal sequences. A benchmark that has only the reference time cannot show that a search found it rather than stumbled into it.

The explanation and the reference stay out of the workspace of an agent whose job is to find the schedule. That agent sees the oracle, the manifest, the DaCe program, and the device. The author and the backtest see `<name>.md`.

## Before calling it done

- [ ] `<name>_numpy.py`, `<name>.yaml`, `<name>_dace.py` and `<name>.md` are in the folder.
- [ ] Preset S matches the oracle on the device the benchmark claims. The size is not a multiple of a block.
- [ ] After stage 1, `list_moves` still offers the move the solution needs. `default_moves()` has not already taken it.
- [ ] The reference sequence builds through stages 3 to 6 and matches the oracle.
- [ ] Preset M baseline-to-reference ratio is at least 2× on the stated device and thread count, with min and max from at least 5 repeats.
- [ ] The reference block in `<name>.md` is filled from that run.
- [ ] CPU and GPU are both described, including the device on which there is no gap.
- [ ] DaCe, NestForge and HPCAgent-Bench SHAs are in the reference block.
