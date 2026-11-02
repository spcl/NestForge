# Python oracle

[Overview](../README.md) · related: [3 Scopes](stages.md#stage-3-scopes), [5 Kernels](stages.md#stage-5-kernels)

Every kernel carries a plain Python function that computes what its SDFG computes: the correctness oracle of stages
5 and 6 (`nestforge/corpus/translate.py`, `prepare`). Its parameter order is the kernel's ABI order.

DaCe tasklets and interstate edges are already Python, so `nestforge/ir/emit_python.py` stays small. It lowers a
copy of the kernel first:

1. A library call stays one: `Cholesky`, `MatMul`, `Gemm`, `Transpose`, `Reduce`, `FillLibraryNode`, `CopyLibraryNode`,
   `FFT` and `IFFT` are spelled as the NumPy call that runs the BLAS, LAPACK or FFT the reference runs (`LIBRARY_CALLS`),
   so the oracle of a call costs what the call costs. A node with no such spelling (`Scan`, `FindFirst`, `ArgReduce`,
   `ScatterConflictCheck`: linear, without a vendor library) expands in the oracle's copy only, to `pure` loops. An
   `expand-library` move on a Fourier transform leaves DaCe's C++ `dft` tasklet, which has no oracle: such a kernel is
   refused with that reason, and `set-library-implementation` is the way to choose its implementation.
2. Nested SDFGs are inlined.
3. DaCe's `InlineTaskletConnectors` rewrites connectors into direct array accesses, as the CPF code generator does.
4. `unique_connectors` names the connectors left, since a Python function has one flat scope.

Then maps become nested `for` loops marked `# parallel`, loops become `while` loops, conditionals `if`/`elif`,
write-conflict resolutions combine statements and copies slice assignments. A guard `if cond: abort()` calls an
`abort()` the module defines to raise.

## Cost

A session's oracle runs in a forked child under `ORACLE_TIMEOUT_S`: first at the `S` rung, then at the graded sizes if
that first run says it would finish in time. A kernel it cannot reach at the graded sizes is compared at `S` and timed at
the graded sizes, and the whole program is still checked there against the bench's reference. A factorization's input
is made positive definite (`make_inputs`), so both sides of the comparison factor the same matrix.

## Contract

- **Caller allocates.** Inputs, outputs and scratch arrays are buffer parameters written in place; only scalars
  are locals. A scratch buffer is sized at its loop's extreme value, as a function of size symbols only.
- **Python tasklets only.** A tasklet in another language is refused, never translated.
- **Loud refusals.** A construct the emitter does not model raises `UnsupportedNest`; it never emits wrong code.
- **One signature.** The function's parameters are the kernel's `extern "C"` arguments, in order.
