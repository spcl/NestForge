`list_move_kinds` once, then `list_moves` and `apply_move` with labels from the latest tree: fusion, fission,
interchange, tiling, `expand-library` and `set-library-implementation`. Each `apply_move` answers with the
whole-program time: keep a move when the time drops, else apply its inverse. `run_defaults stage=moves` applies the
default fusion, `stage=finish` finishes after your own moves. The `nestforge` skill page says which move pays when.
