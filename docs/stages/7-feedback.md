# Stage 7: Feedback

prev: [6 Variants](6-variants.md) · [Agent loop](../agent.md)

`feedback()` turns raw evidence into a few ranked hints the agent reads next turn: kernel times, oracle checks,
compiler remarks from one extra compile of the current variant (`-fopt-info-vec-missed`,
`-Rpass-missed=loop-vectorize`, `nvcc -Xptxas -v`), and operational intensity. A fixed rule table decides:

| evidence | hint |
|---|---|
| build or run failed | fix the kernel |
| wrong result | check bounds, races, reduction order |
| kernel is half the kernel time or more | optimize it first |
| registers spilled | loop body too big, try fission |
| OI below 1 flop/B | memory-bound: fuse with its producer or consumer |
| loop not vectorized | the reason, mapped to a move (fission, interchange, hoisting an `if`, `__restrict__`) |

Failures rank first; the report keeps at most eight hints. With `analyst=True` one more model call rewrites the
report into guidance.

| | |
|---|---|
| default | `feedback()` |
| code | `nestforge/stages/feedback.py` |
