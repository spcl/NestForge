Optimize the program with the NestForge tools; the `nestforge` skill explains the tree and the moves.

1. `get_cpf`: the program in canonical parallel form, as a tree of labeled rows headed by its epoch.
2. `view_tree`: expand states and maps to see their maps and bodies; collapse them again.
3. `list_move_kinds` once, then `list_moves` / `apply_move`: fuse, fission or interchange. Labels and the epoch
   come from the latest tree; a stale epoch is refused. `run_defaults` applies the default fusion (`moves`) or
   finishes after your own moves (`finish`).
4. `get_performance`: whole-program time, which kernel runs where with the bytes moved between memories, and
   each kernel's time and share. `apply_move` and `run_defaults` return it too. Scopes, placement and code
   generation run automatically on their defaults (each kernel built with the fastest correct compiler
   configuration); you choose the moves.
5. `finalize`: the program in the bench C ABI as a library in the shared folder. Submit it with `score` / `submit`
   as `library=<library>` and `workspace_bytes=<workspace_bytes>` from its answer; `source` shows the entry it was
   built from.

Keep a move when the whole-program time drops. Prefer moves on the kernel with the largest share.

{{HINTS}}
