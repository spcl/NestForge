# Kernel DAG

[Overview](../README.md) · related: [3 Scopes](stages/3-scopes.md), [4 Placement](stages/4-placement.md)

`kernel_dependencies(sdfg)` (`nestforge/ir/depends.py`) answers, for every `ExternalCall` argument, which producers
can reach it. It is the kernel-only view the agent reads, and stage 4 uses it to list the transfers a placement
implies. It reads the program and never changes it.

```
extcall_0: A <- extcall_1.A | program [carried: for0_0], N <- program
extcall_1: B <- extcall_0.B, N <- program
exit: A <- extcall_1.A | program, B <- extcall_0.B | program
```

- **Producers.** `program` (a program input or free symbol not written before), `extcall_i.arg` (a kernel
  output), or `host:<state>` (any other writer).
- **Whole containers.** A read reads all of it, a write replaces all of it. A view stands for its root array.
- **Control flow.** `if` without `else` keeps the incoming producers; a loop runs to a fixpoint and tags what
  crossed its back edge `[carried: <loop>]`; the incoming producers survive a loop unless it provably runs once.
- **Refused** with `UnsupportedProgram`: `Reference` containers, a view that binds nothing, a kernel inside a
  nested SDFG or without a manifest, and unmodeled control flow.

`Session.kernel_graph()` computes it once per epoch; `list_kernels()` and `describe(deps=True)` show it.
