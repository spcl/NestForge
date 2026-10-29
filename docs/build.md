# Build

[Overview](../README.md) · related: [5 Kernels](stages.md#stage-5-kernels),
[6 Variants](stages.md#stage-6-variants)

`nestforge/build/` compiles with one chosen compiler, flag set and OpenMP runtime, and calls kernels
through ctypes, so timings compare generated code, free of `CompiledSDFG` marshaling.

## Program builds

`compile_linked_program` builds a whole program that links its kernel libraries, through DaCe's CMake with libomp
as the one OpenMP runtime. `BuiltSDFG` binds the three C entries of an SDFG `N` (`__dace_init_N`, `__program_N`,
`__dace_exit_N`); `Session.performance()` times the program through it in an isolated child.

## Kernel libraries

`build_archive` compiles a kernel's CPF unit into `lib<kernel>.a` and links a shared twin with
`--whole-archive` for ctypes validation and timing. The program links the archive through
`ExternalCall`. Every kernel gets its own archive, because DaCe sorts the program's link flags and a
shared archive would lose members.

## Register arrays

On the CPU every array headed for registers is pinned to `Register(force=True)` (`force_register_arrays`, on every
program change and on each CPU kernel): it stays on the stack, a symbolic size as a variable-length array, instead
of a heap allocation per scope. That covers unforced register arrays, scope-lived arrays inside a map, and a
kernel's scope-lived arrays, which codegen gives registers once the kernel expands. GPU builds and GPU kernel
renders pin DaCe's new CUDA codegen (`compiler.cuda.implementation = experimental`), whose
`MoveArrayOutOfKernel` hoists a per-thread array of symbolic size out of its kernel into global memory.

## Fork isolation

`run_isolated` (`isolation.py`) runs fresh code in a forked child under a wall-clock timeout, so a
crash or hang in generated code becomes a recorded result. The child answers with a pydantic `Result` model,
validated as it crosses back; an exception, crash, hang or malformed answer is a `Failure(error=...)`. `os.fork()` copies only the calling
thread, so `pause_openmp_pools` shuts down loaded OpenMP thread pools first. A CUDA context does not
survive a fork, so `run_spawned` measures device kernels in a freshly spawned interpreter instead.

## Runtime libraries

NestForge assumes Linux. Every link passes `-Wl,--as-needed`, and every library and program links the
runtimes it needs by name instead of relying on the host process having them loaded.

- **OpenMP.** LLVM libomp is the process's one runtime. It serves g++ code through its `GOMP_*` entry
  points and clang++ or icpx code through `__kmpc_*`, so every compiler shares one thread pool.
  `openmp_compile_flags` and `openmp_link_flags` select it per compiler. The program link swaps DaCe's default
  runtime for libomp, for that compile only.
- **CUDA.** A GPU kernel links cudart from the nvcc that built it.

`ExternalCall` carries these link items, and the program links them after its objects.
