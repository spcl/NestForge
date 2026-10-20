Optimize the program with the NestForge tools. Only the structured phase is open: the other tools answer
"not implemented".

1. `get_cpf`: the program in canonical parallel form, as a tree of labeled rows headed by its epoch.
2. `view_tree`: expand states and maps to see their kernels and bodies; collapse them again.
3. `list_moves` / `apply_move`: fuse, fission or interchange. Labels and the epoch come from the latest tree; a
   stale epoch is refused.
4. `set_scope`: group regions into one kernel. `run_defaults` finishes any stage you leave (moves, finish,
   scopes, placement).
5. `get_performance`: whole-program time, which kernel runs where with the bytes moved between memories, and
   each kernel's time and share. `apply_move` and `run_defaults` return it too. Stages you have not reached run
   with their defaults on a copy.

Keep a move when the whole-program time drops. Prefer moves on the kernel with the largest share.

{{HINTS}}
