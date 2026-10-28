## End to end

1. `get_cpf`, then `view_tree` to open the states and doalls with the largest `work=`. Read the `nestforge` skill
   page before your first move.
2. `finalize` the program as it is and `score` its library: a correct version is recorded before you change
   anything.
3. Moves: `list_move_kinds` once, then `list_moves` and `apply_move`; keep a move when the whole-program time drops,
   else apply its inverse.
4. Scopes: `set_scope` doalls that share data into one kernel when `get_performance` shows traffic between them;
   `unset_scope` to undo.
5. Kernels: `list_kernels`, then `read_kernel` and `write_kernel` the kernels with the largest share; `tune_kernel`
   each kernel you keep. After each change you keep, `finalize` and `score` the library.
{{MODE:closing}}
