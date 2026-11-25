## Tactics
Your setup optimizes the program with Tactics: it shows the program as a tree of loops, applies only legal moves,
groups the loops into kernels, places them on the CPU or the GPU and builds the program. It works in phases, in this
order: `structured` (moves on the whole program), `scopes` (kernels and where they run), `scoped` (moves inside the
kernels) and `kernels` (each kernel's code and build). Your setup gives you some of them; Tactics runs the defaults of
the others. `advance_phase` goes on, `undo_phase` takes the current phase back completely, and `return_to_phase` returns to an
earlier phase, after a score too. The `tactics` skill page explains the tree and every move.
