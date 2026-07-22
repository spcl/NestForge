# Externalize + mega-kernel (TODO)

**Externalize is the foundation.** Mega-kernel is a second consumer of the same nesting step.

Any region -> nested SDFG -> either `ExternalCall` (normal offload, via
`pass_lower.replace_nsdfg_with_external`) or a mega-kernel. DaCe supplies both nesting primitives:
`helpers.nest_sdfg_subgraph` (a line graph of control-flow blocks) and `helpers.nest_state_subgraph`
(dataflow nodes within a state).

## Open — externalize granularity

`extract_nest_to_sdfg` dispatches MapEntry, LoopRegion and SDFGState. Still missing:

1. **Multiple nodes** — a line graph of consecutive blocks through `nest_sdfg_subgraph`.
   `extract_loop_nest` only ever passes a single block. (`split_unsupported.region_to_standalone`
   does handle multi-state regions, but by deepcopy-and-prune into a *standalone* SDFG, not by
   nesting — a different mechanism with different boundary rules.)
2. **`ConditionalBlock`** as a single-node granularity — currently a `TypeError`.
3. **Numeric checks per granularity.** `tests/test_offload_units.py` asserts SDFG validity only;
   the oracle comparison exists just for the default granularity
   (`tests/test_offload_granularity.py`).
4. **The libnode invariant is unenforced.** A canonicalize-lifted known nest (Memset/Copy/BLAS) is
   supposed to stay a libnode and not be offloaded, but `whole_program.default_offloadable` returns
   `True` unconditionally and `offload.state_has_compute` counts a `LibraryNode` AS compute — so the
   `state` unit will externalize a BLAS state today.

## Open — mega-kernel (nothing built)

Take a nested SDFG and rewrite it as ONE persistent kernel: GPU device kernel, or CPU persistent
multicore.

- CPU codegen must emit a multi-dimensional OpenMP parallel scope just like a GPU kernel. Assess
  first whether the readable/experimental CPU codegen already does multi-dim parallel scopes; if not,
  that is the one codegen change.
- Thread ids are assigned inside the scope; every kernel launch becomes a **grid-strided loop**.
- GPU model: fixed cores, persistent threads. `map M,N` then `map 2M` launches once, distributes over
  `M*N` as a grid-stride loop, **global-syncs**, then runs the second map. With tiling to 4x4 per
  thread the grid-stride loop is over `M/4, N/4` — assess whether that needs no codegen change at all
  (grid-stride and tile factor may compose at the map level).
- Global sync lives in its own state.
- jacobi2d is the worked example: a time for-loop wrapping two maps becomes one mega-kernel.

## Invariant

Externalize BEFORE deciding offload.
