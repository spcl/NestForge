Optimize the program with the NestForge tools.

1. `get_cpf`: the program in canonical parallel form, as a tree of labeled rows headed by its epoch.
2. `view_tree`: expand states and maps to see their kernels and bodies; collapse them again.
3. `list_moves` / `apply_move`: fuse, fission or interchange. Labels and the epoch come from the latest tree; a
   stale epoch is refused.
4. `set_scope`: group regions into one kernel. `run_defaults` finishes any stage you leave (moves, finish,
   scopes, placement).
5. `get_performance`: whole-program time, which kernel runs where with the bytes moved between memories, and
   each kernel's time and share. `apply_move` and `run_defaults` return it too. Stages you have not reached run
   with their defaults on a copy.
6. `list_kernels` / `read_kernel` / `write_kernel`: once scopes exist, read a kernel's C++ unit, rewrite it and
   write it back. The entry keeps its name and parameters; a write is built, checked against the kernel's oracle
   and linked only when it matches, and returns performance.
7. `tune_kernel`: sweep compilers, FP modes and vectorizer cost models for one kernel; the fastest correct build
   is linked.
8. `finalize`: the program in the bench C ABI, each kernel linked from its own build, as a library in the shared
   folder. Submit it with `score` / `submit` as `library=<library>` and `workspace_bytes=<workspace_bytes>`
   from its answer; `source` shows the entry it was built from.

Keep a move when the whole-program time drops. Prefer moves on the kernel with the largest share.

{{HINTS}}
