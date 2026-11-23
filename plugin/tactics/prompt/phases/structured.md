## Phase: structured
Moves on the whole program: `list_moves` and `apply_move` with labels from the tree (fusion, fission, interchange,
tiling, hoisting, unrolling, library moves, `relayout` of arrays, `change-precision`). A move answers at once, without
a time: nothing runs unless you ask. Call `get_performance` after a move or a few that belong together; a correct
program more than 3% slower than the fastest one measured is rolled back to it. Undo with `apply_move rollback []`,
never the inverse move. `tune_move` times up to four sizes and keeps the fastest only if it beats the program.
`apply_moves` applies several moves at once, all or nothing, and checks the program's numbers after each at a small
size (no time): the answer names the first move that failed and why. `run_defaults stage=moves` applies the default
moves. `advance_phase` goes on to the scopes.
