# tsvc_2_s231

TSVC `s231`: `for i: for j in 1..LEN_2D: aa[j, i] = aa[j-1, i] + bb[j, i]` on row-major `(LEN_2D, LEN_2D)` arrays. The kernel comes from the HPCAgent-Bench corpus (`loop_level_reasoning/tsvc_2_s231`), unedited.

## How to optimize it

The recurrence runs along `j`, the strided axis. `i`, the contiguous axis, is independent. After stage 1 on the CPU the program is `for j { map i }`, one parallel region per `j`, and `default_moves()` keeps it. The one offered move, `interchange-loop-map`, gives `map i { for j }`: one parallel region, with each thread walking a column.

- **Threaded CPU:** take the interchange. Fewer parallel regions beat the strided column walk.
- **One thread:** leave it. The interchanged column walk is about 17x slower.
- **GPU:** `default_moves()` already takes the interchange. Skipping it (`no-move`) is the decoy.

The reference is [reference_solution/solution.py](reference_solution/solution.py). Its measured replay is in [reference_solution/results.md](reference_solution/results.md): 2.4x at 72 threads, preset M, on a GH200 node, at DaCe `3469167e5` and HPCAgent-Bench `0cc69ba`.
