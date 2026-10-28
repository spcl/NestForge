## The NestForge tools, every phase

Structured phase (the program as a tree):
- `get_cpf`, `view_tree`: the program in canonical parallel form, a tree of labeled rows headed by its epoch; open
  and close states and doalls.
- `list_move_kinds`, `list_moves`, `apply_move`: fuse, fission, interchange, tile, expand a library call, or set its
  implementation. Labels and the epoch come from the latest tree; a stale epoch is refused. `run_defaults` applies a
  stage's default (`moves`, `finish`, `scopes`, `placement`).

Scopes (which rows become one kernel):
- `set_scope`: one kernel of several top-level doalls of one state, or of a straight run of blocks. By default
  every top-level doall (a library call among them) is a kernel of its own.
- `unset_scope`: inline a kernel back; its source and tuning go with it.

Kernels (the code of each scope):
- `list_kernels`: every kernel, its device, arguments and what produces them.
- `read_kernel`, `write_kernel`: a kernel's source; a written source is built and checked against the kernel's
  oracle, and linked only when it matches.
- `tune_kernel`: sweep compilers and flags for one kernel and link the fastest correct build.

Any phase:
- `get_performance`: the whole-program time, which kernel runs where with the bytes moved between memories, and each
  kernel's time and share. `apply_move`, `run_defaults`, `write_kernel` and `tune_kernel` answer with it too.
- `finalize`: the program in the bench C ABI as a library in the shared folder, with the `library` and
  `workspace_bytes` to send to `score` and `submit`.

Work top-down: moves first, then scopes, then the kernels with the largest share. A move or a new scope drops the
kernel sources written below it, so settle the structure before you write code.
