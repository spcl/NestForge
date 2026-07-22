# PLAN — the `optimize` entry contract

One entry point takes *whatever the user has* and returns the best-performing build, by compiling and
measuring variants rather than trusting language- or compiler-defined semantics.

Status (2026-07-22): the CONTRACT is built and unit-tested in `nestforge/entry.py` +
`tests/test_entry_contract.py`. Nothing executes it yet.

## Built

Input classification (`classify_input`), the two search spaces, budgeted axis selection
(`plan_search`, `broad_codegen_axes`, `VARIANT_BUDGET`), the pinned-vs-uncertain codegen knobs
(verified against dace `extended`'s `config_schema.yml`), `AgentVariant`/`AgentMode`, and
`lower_to_sdfg` for `.sdfg`/`.sdfgz` + Fortran.

Two spaces, not three: the original plan had a third "no agent, exhaustive" case. It was dropped —
an agent CONTRIBUTES variants rather than choosing the space, so steered and unsteered runs sweep
identical ground and their difference measures the agent.

## Open

1. **`optimize_program(source, *, agent=None, ...) -> Report`** — the executing entry point. Does not
   exist. `plan_search` returns a `SearchPlan` that nothing hands to `run_arena`.
2. **Case A needs a way into the arena.** `run_arena` assumes a `Prepared` built from numpy emission;
   a provided C/C++/Fortran source has no such object. Smallest adapter, not a parallel path.
3. **`lower_to_sdfg` for `InputKind.NUMPY`** raises `NotImplementedError`. The plan still reports
   `needs_parse`, so a caller sees the gap rather than a wrong answer.
4. **`FLAG_AXES['vectorize']`** names `none`/`cheap`/`auto` but nothing maps them to per-compiler
   flags. `perf/flags.py::cost_flags` is the model to reuse.
5. **ccache on the arena's compile path.** `arena.compile_object` calls `run_tool` with no ccache
   prefix; `build.py` already has `BuildOptions.use_ccache`. Measured 2.57x on repeat compiles.

The `e2e` marker exists and `ci.yml` runs `-m "not integration"`, which already includes it — so no
separate CI lane is needed until an e2e test wants a budget the unit phase cannot give it.

## Non-negotiable

Every variant validates bit-exact against the numpy oracle before its timing counts. A variant that
does not validate is reported failed, never silently dropped. That is what makes "measure, don't
trust semantics" sound rather than reckless.
