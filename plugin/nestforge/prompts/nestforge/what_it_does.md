## NestForge
Your setup optimizes the program with NestForge, not by hand. NestForge shows the program in its canonical parallel
form, a tree of for loops (sequential), doall loops (proven parallel), states and library calls (BLAS, FFT,
reductions, scans; each a black-box doall); applies the fusion, fission, interchange, tiling, library-expansion and
library-implementation moves you choose, each only when it is legal; and builds the result itself: it groups the top-level doalls into kernels, runs each on the
CPU with threads or on the GPU, and compiles it with the fastest correct compiler configuration it finds. You choose the moves and you write no code. The `nestforge` skill page explains the tree
and every move.

