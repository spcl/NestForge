## NestForge
Your setup optimizes the program with NestForge. NestForge shows the program in its canonical parallel form, a tree of
for loops (sequential), doall loops (proven parallel), states and library calls (BLAS, FFT, reductions, scans; each a
black-box doall); applies the fusion, fission, interchange, tiling, library-expansion and library-implementation moves
you choose, each only when it is legal; groups top-level doalls into kernels (by default one per doall, or the scopes
you set); runs each kernel on the CPU with threads or on the GPU; and builds it. You may rewrite any kernel's source:
NestForge checks it against the kernel's oracle and links it only when it matches. The `nestforge` skill page explains
the tree and every move.
