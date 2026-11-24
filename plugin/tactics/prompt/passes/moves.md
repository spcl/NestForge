## MLIR and LLVM pass moves
Your setup builds the program with NestForge, and your controls are compiler passes, applied one at a time as named
moves to every kernel. You work in two phases. In the MLIR phase, each kernel's C++ is lowered by cgeist to affine
MLIR and your moves run on its loops; `advance_pass_phase` then lowers it to LLVM IR with a fixed lowering, and your
LLVM moves run on that IR before the back end. NestForge applies no move of its own.

1. `get_cpf` and `view_tree` show the program; `get_performance` times it with NestForge's default build.
2. `score` the program as it is.
3. `view_ir` prints one row per loop of every kernel; `list_pass_moves` lists the moves of your phase and their
   arguments.
4. `apply_pass_move kind=... arguments={...}` appends one move. A move that leaves every kernel's IR as it was, or
   whose tool fails, is refused and records nothing. `view_ir` shows the loops after it; `apply_pass_move
   kind=rollback` takes the last move back.
5. `get_performance` checks and times the program: a wrong result names itself, and a correct program more than 3%
   slower than the fastest one measured is rolled back to it.
6. `advance_pass_phase` when the MLIR moves are done; the MLIR moves are closed after it.
7. `finalize`, `score` and `submit` the fastest correct program.
