# Stage 7: Feedback

prev: [6 Variants](6-variants.md)

`feedback()` turns raw evidence into a few ranked hints the agent reads next turn: kernel times, oracle checks,
compiler remarks from one extra compile of the current variant (`-fopt-info-vec-missed`,
`-Rpass-missed=loop-vectorize`, `nvcc -Xptxas -v`). A fixed rule table decides:

| evidence | hint |
|---|---|
| build or run failed | fix the kernel |
| wrong result | check bounds, races, reduction order |
| kernel is half the kernel time or more | optimize it first |
| registers spilled | loop body too big, try fission |
| loop not vectorized | the reason, mapped to a move (fission, interchange, hoisting an `if`, `__restrict__`) |

Failures rank first; the report keeps at most eight hints.

`performance(inputs, reference)` gives every phase's feedback in one record, on a copy completed with the default
scopes and placement; kernels keep the libraries already linked, the rest get the default build:

- `structured`: the whole program's time, and whether its outputs match `reference` (the bench's own reference and
  grading tolerance, when the plugin has one);
- `offloading`: each kernel's device and every transfer between memories, with bytes;
- `kernels`: each kernel's correctness, time and share, and the hints above.

It is cached until the program or a kernel library changes.

| | |
|---|---|
| default | `feedback()`, `performance(inputs, reference)` |
| code | `nestforge/stages/feedback.py`, `nestforge/stages/performance.py` |
