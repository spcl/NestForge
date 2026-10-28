## The NestForge tools

- `get_cpf`: the program in canonical parallel form, a tree of labeled rows headed by its epoch.
- `view_tree`: expand states and maps to see their maps and bodies; collapse them again.
- `list_move_kinds`, `list_moves`, `apply_move`: fuse, fission or interchange. Labels and the epoch come from the
  latest tree; a stale epoch is refused. `run_defaults` applies the default fusion (`moves`) or finishes after your
  own moves (`finish`).
- `get_performance`: the whole-program time, which kernel runs where with the bytes moved between memories, and each
  kernel's time and share. `apply_move` and `run_defaults` answer with it too.
- `finalize`: the program in the bench C ABI as a library in the shared folder, with the `library` and
  `workspace_bytes` to send to `score` and `submit`; `source` shows the entry it was built from.

Scopes, placement and code generation run on their defaults, each kernel built with the fastest correct compiler
configuration: you choose the moves. Keep a move when the whole-program time drops, and prefer moves on the kernel
with the largest share.
