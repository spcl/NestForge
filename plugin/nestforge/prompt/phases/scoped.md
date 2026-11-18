## Phase: scoped
Moves inside the kernels: `list_moves` and `apply_move` with `scope` (the kernel) take the rows of its body:
`vectorize-doall` (with `isa`), `stage-loads-shared` (GPU), `set-library-implementation` (an implementation the
kernel's device runs), `set-fp-mode` (the kernel's floating-point flags), and the structured moves on the kernel's
body. Nothing runs unless you ask: `get_performance` after a move or a few. `advance_phase` goes on to the kernel code;
`undo_phase` takes this phase back.
