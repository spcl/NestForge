## MLIR passes
Your setup builds the program with NestForge, and your one control is the MLIR pass pipeline every kernel goes
through. `set_mlir_passes pipeline="..."` takes `polygeist-opt --pass-pipeline=` syntax
(`builtin.module(affine-loop-tile{tile-size=32},affine-parallelize,canonicalize)`): each kernel's C++ is lowered
by cgeist to affine MLIR, your pipeline runs, then a fixed lowering to LLVM. NestForge applies no move of its own.

1. `get_cpf` and `view_tree` show the program; `get_performance` times it with NestForge's default build.
2. `score` the program as it is.
3. `set_mlir_passes`, then `get_performance`: a wrong result or a failed pass names itself. Calling it again replaces the
   pipeline. A correct program more than 3% slower than the fastest one measured is rolled back to it.
4. `finalize`, `score` and `submit` the fastest correct program.
