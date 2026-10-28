`list_move_kinds` once, then `list_moves` and `apply_move`: fuse, fission, interchange, tile, expand a library call, or
set its implementation. Labels and the epoch come from the latest tree; a stale epoch is refused. Each `apply_move`
answers with the whole-program time: keep a move when the time drops, else apply its inverse. `run_defaults
stage=moves` applies the default fusion, and `stage=finish` finishes after your own moves.

Fusion removes the memory traffic between a producer and its consumer, fission splits a doall whose parts want
different loop orders, interchange puts the contiguous index innermost, and tiling keeps a tile's data in cache. A
library call (`lib_blas`, `lib_fft`, `lib_reduce`, ...) runs as a vendor call: `expand-library` it when fusing it with
its neighbors saves memory traffic, keep it when the vendor call is faster, and `set-library-implementation` to try
another library or plain loops.
