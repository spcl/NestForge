# Externalize + mega-kernel (TODO)

**Externalize is the foundation.** Mega-kernel is a second consumer of the same nesting step.

## Externalize mechanism

Any region -> put into a nested SDFG -> then either:
- nested SDFG -> `ExternalCall` (normal offload), or
- nested SDFG -> wrapped in a **mega-kernel** (below).

DaCe already provides the nesting primitives (no new dace code to nest):
- `helpers.nest_sdfg_subgraph(sdfg, subgraph, start)` -- a line-graph of control-flow blocks
  (SDFGStates + interstate edges, a LoopRegion, a ConditionalBlock) -> one NestedSDFG.
- `helpers.nest_state_subgraph(sdfg, state, subgraph)` -> dataflow nodes within a state -> one NestedSDFG.

Then the existing `NestedSDFG -> ExternalCall` lowering (`pass_lower.replace_nsdfg_with_external`) takes over.

## Granularity allowed (externalize + mega-kernel)

1. **multiple-nodes** (line graph of consecutive blocks) -- `nest_sdfg_subgraph`
2. **single node**: LoopRegion | ConditionalBlock | SDFGState -- `nest_sdfg_subgraph`
3. **single map** -- `nest_state_subgraph` (the map scope's nodes)

Today `extract_nest_to_sdfg` handles only MapEntry + LoopRegion. Extend to all of the above through the
two dace helpers. Many CI tests: externalize each granularity, validate + numerically check.

## Mega-kernel

Take a nested SDFG (from externalize) and rewrite it as ONE persistent kernel:
- GPU device kernel, or CPU persistent (multicore).
- **CPU codegen must emit a multi-dimensional OpenMP parallel scope just like a GPU kernel** (assess:
  does the readable/experimental CPU codegen already do multi-dim parallel scopes? if not, that is the
  one codegen change).
- Assign thread ids inside the scope; every kernel launch (GPU device) / cpu-kernel flag becomes a
  **grid-strided loop**.

### GPU model

Fixed cores (e.g. 2000). Persistent threads, grid-stride over the work.
- map M,N then map 2M: launch mega-kernel once, distribute threads over M*N as a grid-stride loop
  (each thread ~1000 elements; a 2D/grid distribution also allowed), **global sync**, then the 2nd map.
- Tiling: if M,N tiled so a thread computes 4x4, the grid-stride loop is emitted over `M/4, N/4`.
  Assess whether this needs NO codegen change (grid-stride + tile factor composable at the map level).

### Global sync

- Emitted via `megakernelify`.
- The global-sync tasklet lives in **its own state**.

### jacobi2d example

time for-loop wrapping 2 maps -> the WHOLE for-loop becomes one mega-kernel.

## Invariant

Externalize BEFORE deciding offload. A canonicalize-lifted known nest (Memset/Copy/BLAS) stays a
libnode and is **not** offloaded.
