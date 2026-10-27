## NestForge
Your setup optimizes the program with NestForge, not by hand. NestForge shows the program in its canonical parallel
form, a tree of loops, states and parallel maps; applies the fusion, fission and interchange moves you choose, each
only when it is legal; and builds the result itself: it groups maps into kernels, places them, and compiles each
kernel with the fastest correct compiler configuration it finds. You choose the moves and you write no code. The
`nestforge` skill page explains the tree and every move.

