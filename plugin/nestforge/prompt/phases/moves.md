`list_move_kinds` once, then `list_moves` and `apply_move` with labels from the latest tree: fusion, fission,
interchange, tiling, unrolling, `expand-library`, `set-library-implementation` and `propagate-config`. `apply_move`
answers at once with the outcome, not with a time: call `get_performance` after a move, or after a few that belong
together. It waits for a device slot, so do not ask after every small step. Keep a move when the time drops, else
apply the move `rollback` (labels `[]`). `run_defaults stage=moves` applies the default fusion, `stage=finish` finishes after your own moves. The
`nestforge` skill page says which move pays when.
