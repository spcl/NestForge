# The `nestforge_default` flavor

[Designs](README.md)

Status: design, not implemented. Written 2026-11-16 against NestForge `bf068ae`.

## Goal

A column that runs NestForge's default moves on every kernel, with no model and no reference solution. It is the
floor an agent's run is compared with: what NestForge gives for free.

## What exists today

`nestforge_reference` (`nestforge_bench.ReferenceOptimizer`) replays a kernel's reference solution and falls back to
the default moves when the kernel has none. On a kernel without a reference solution, the two columns would be the
same run under two names, and on a kernel with one, no column shows the defaults.

## Design

1. `ReferenceOptimizer` gets a class attribute `references: bool = True`. `tune` reads the solution only when it is
   set. A subclass `DefaultOptimizer` sets it to `False` and changes nothing else, so the two columns share every
   line after the moves (scopes, placement, code generation, finalize).
2. `plugin.toml` declares the flavor beside the reference one.

   ```toml
   # The floor of every comparison: default moves, then the same automatic scopes, placement and code generation.
   [flavors.nestforge_default]
   name = "NestForge default"
   full_name = "NestForge (default moves)"
   optimizer = "nestforge_bench:DefaultOptimizer"
   extends = ["dace_cpu_canon", "dace_gpu_canon"]
   ```
3. `ReferenceOptimizer` stops falling back silently. On a kernel without a reference solution it declines the
   kernel as unsupported, so the reference column reports the covered kernels only and never repeats the default
   column. The bench has this outcome for a hand-written column without the kernel's file (bench #184). A tuner
   flavor needs the same outcome through the plugin API ([bench request B8](bench-requests.md)). Until the bench has
   it, the fallback stays and the plugin record marks the run with `reference: false`.
4. `[ci].optimizers` gains `dace_cpu_canon_nestforge_default`.

## Tests

1. On a kernel with a reference solution, the default column applies the default moves and the reference column the
   solution's: their `applied_moves` differ.
2. On a kernel without one, the reference column is unsupported, and the default column finalizes.
3. The plugin check lists both flavors.

## Done means

Both flavors run in the bench's user-plugin CI on `jacobi_1d` and `tsvc_2_s231`.

## Decision for the owner

Point 3 changes what `nestforge_reference` does on every kernel without a solution (20 kernels have one today).
Recommended: point 3, since a column should mean one thing.
