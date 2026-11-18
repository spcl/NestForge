`list_moves` and `apply_move` with labels from the tree: fusion, fission, interchange, tiling, hoisting, unrolling,
`inline-library`, `set-library-implementation`, `lib-before` / `lib-after` / `lib-into`, `relayout`, `vectorize-doall`. `apply_move` answers at once, without a time and without the
tree: open only the rows you need next. Call `get_performance` after a move, or after a few that belong together (it
waits for a device slot): a correct program more than 3% slower than the fastest one measured is rolled back to it.
Undo otherwise with `apply_move rollback []`, never the inverse move: `rollback` restores the program and its labels
exactly. To pick a tile size, `tune_move` times up to four in one call and
keeps the fastest only if it beats the program.
`run_defaults stage=moves` applies the default moves, `stage=finish` the stages after your own. The `nestforge` skill
page says which move pays when.
