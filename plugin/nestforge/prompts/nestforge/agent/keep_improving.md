## Keep improving

{{BUDGET}}

Most of the time goes to the kernels with the largest share in `get_performance`: try moves there first. Fusion
removes the memory traffic between a producer and its consumer, fission splits a map whose parts want different
loop orders, and interchange puts the contiguous index innermost. A library call (`kernel_blas`, `kernel_fft`,
`kernel_reduce`, ...) runs as a vendor call: `expand-library` it when fusing it with its neighbors saves memory
traffic, and keep it when the vendor call is faster. Do not stop after the first move that helps: try the other moves
`list_moves` offers. Stop when no move lowers the time, then `submit` the best library you measured.
