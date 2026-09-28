# Python oracle

[Overview](../README.md) · related: [3 Scopes](stages/3-scopes.md), [5 Kernels](stages/5-kernels.md)

Every kernel carries a plain Python function that computes what its SDFG computes. It is the correctness oracle of
stages 5 and 6, and, with its YAML manifest, the input HPCAgent-Bench's translators turn into C, C++ or Fortran.

DaCe tasklets and interstate edges are already Python, so `nestforge/ir/emit_python.py` stays small. It lowers a
copy of the kernel first:

1. Library nodes expand to their `pure` implementation, which is plain SDFG components.
2. Nested SDFGs are inlined.
3. DaCe's `InlineTaskletConnectors` rewrites connectors into direct array accesses, as the CPF code generator does.
4. `unique_connectors` names the connectors left, since a Python function has one flat scope.

Then maps become nested `for` loops marked `# parallel`, loops become `while` loops, conditionals `if`/`elif`,
write-conflict resolutions combine statements and copies slice assignments. A guard `if cond: abort()` calls an
`abort()` the module defines to raise.

## Contract

- **Caller allocates.** Inputs, outputs and scratch arrays are buffer parameters written in place; only scalars
  are locals. A scratch buffer is sized at its loop's extreme value, as a function of size symbols only.
- **Python tasklets only.** A tasklet in another language is refused, never translated.
- **Loud refusals.** A construct the emitter does not model raises `UnsupportedNest`; it never emits wrong code.
- **One signature.** The function and the manifest list the same arguments in the same order.
