## LLVM passes
Your setup builds the program with Tactics, and your one control is the LLVM pass pipeline every kernel goes
through. `set_llvm_passes passes="..."` takes `opt -passes=` syntax (`default<O3>`,
`function(loop-mssa(licm),loop-unroll<O3>),loop-vectorize`): each kernel's C++ is lowered to IR without
optimization, your pipeline runs, and only the back end runs after it. Tactics applies no move of its own.

1. `get_cpf` and `view_tree` show the program; `get_performance` times it with Tactics' default build.
2. `score` the program as it is.
3. `set_llvm_passes`, then `get_performance`: a wrong result or a failed pass names itself. Calling it again replaces the
   pipeline. A correct program more than 3% slower than the fastest one measured is rolled back to it.
4. `finalize`, `score` and `submit` the fastest correct program.
