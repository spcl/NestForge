## Keep improving

{{BUDGET}}

Most of the time goes to the kernels with the largest share in `get_performance`: try moves on their doalls first.
Fusion removes the memory traffic between a producer and its consumer, fission splits a doall whose parts want
different loop orders, interchange puts the contiguous index innermost, and tiling keeps a tile's data in cache. A library call (`lib_blas`, `lib_fft`,
`lib_reduce`, ...) runs as a vendor call: `expand-library` it when fusing it with its neighbors saves memory
traffic, and keep it when the vendor call is faster. Do not stop after the first move that helps: try the other moves
`list_moves` offers. Stop when no move lowers the time, then `submit` the best library you measured.
