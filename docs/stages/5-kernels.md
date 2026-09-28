# Stage 5: Kernels

prev: [4 Placement](4-placement.md) · next: [6 Variants](6-variants.md)

Each kernel becomes a static library `lib<kernel>.a` with one `extern "C"` entry, linked into the program.

The default renders the kernel with DaCe's CPF code generator (`cpf.render`): one standalone unit with no DaCe
header or runtime, C++ with OpenMP on the CPU, CUDA on the GPU. The entry takes arrays by pointer and read-only
scalars by value. The compiler vectorizes; stage 6 sweeps its flags.

An agent reads `kernel_source(name)` and replaces it with `set_kernel_source(name, source, language)`. NestForge
builds the new source, checks it against the kernel's [Python oracle](../oracle.md) and links it only when it
matches; the status is `ok`, `refused`, `build-failed` or `wrong`.

| | |
|---|---|
| default | `optimize_kernel(name)` |
| agent | `set_kernel_source(name, source, language)` |
| code | `nestforge/stages/kernel.py`, `nestforge/build/` |
