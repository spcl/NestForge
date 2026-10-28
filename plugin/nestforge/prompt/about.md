## NestForge
Your setup optimizes the program with NestForge. NestForge shows the program in its canonical parallel form: a tree of
for loops (sequential), doall loops (proven parallel), states and library calls (BLAS, FFT, reductions, scans; each a
black-box doall). It applies a move only when the move is legal, groups the top-level doalls into kernels, places each
kernel on the CPU or the GPU, and builds the program itself. The workflow says which phases are yours. The `nestforge`
skill page explains the tree and every move.
