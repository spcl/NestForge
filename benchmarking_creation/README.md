# benchmarking_creation

Benchmarks built on NestForge. Each one keeps its reference solution as a Python script that `nestforge replay` runs, not as a hand-filled YAML block.

## Reference solutions

```
benchmarking_creation/<name>/reference_solution/
  solution.py     KERNEL, CONFIGS, solve(script, ctx), optional DECOYS
  replay/         one JSON record per configuration, written by replay (moves, tree, time, oracle, pins)
  results.md      the table of every record, written by replay
```

`solve` drives stage 2 through a `nestforge.replay.Script`. It picks each move by kind and by the containers its rows read and write, never by label: `s.apply(s.pick("interchange-loop-map", reads=["bb"], writes=["aa"]))`, then `s.finish()`. It may branch on the configuration (`ctx.device`, `ctx.threads`, `ctx.preset`), so one script covers every device and thread count. A pick that matches no move, or several, fails with the offered moves and the tree. That is how a solution that no longer applies on a new pin shows up.

```
python -m nestforge.replay benchmarking_creation/<name>/reference_solution/solution.py [--config cpu:72:M] [--md <name>.md]
```

For each configuration, replay starts the kernel's canonicalized session and runs the baseline (`default_moves()`), the reference and each decoy on separate copies. Each copy then goes through the default stages 3 to 5. Replay times the whole program (the mean of `--reps` calls) and checks it against the bench oracle. Each configuration runs in its own process with `OMP_NUM_THREADS`, `OMP_PROC_BIND=close` and `OMP_PLACES=cores` set. `--md` writes the table between `<!-- replay:begin -->` and `<!-- replay:end -->` in a benchmark note, so the prose stays hand-written and the numbers are generated.

A run is scored by the oracle and the time, not by whether its moves equal the reference's.
