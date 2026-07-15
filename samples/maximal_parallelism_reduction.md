# Maximal parallelism: the reduction that only *looks* serial

A dot product `s = Σ a[i]·b[i]` is the smallest example that carries the whole thesis of this project:
the source-language spelling imposes a serialization that the *mathematics* does not. Lift that
spelling into a semantics-explicit form and the same computation exposes SIMD-, thread-, and
GPU-level parallelism — at a factor that dwarfs any single-flag tuning.

## Before — one accumulator, one dependency chain

```python
def dot(a, b, n):
    s = 0.0
    for i in range(n):
        s += a[i] * b[i]      # every add waits for the previous value of s
    return s
```

`s += …` is a **loop-carried dependency on `s`**: iteration `i+1` cannot start its add until
iteration `i` has written `s`. The critical path is `n` additions end to end — a single latency-bound
chain. A CPU has 16 cores and 8–16 fp64 SIMD lanes sitting idle, but the code has told it, one value
at a time, *do these in this exact order*. That ordering is a **language artifact**, not a real data
dependency: `+` is associative, so the sum may be accumulated in any grouping.

## After — say it is a reduction, and the grouping is free

```python
# The same result, decomposed so the associativity is explicit: P independent
# partial sums over contiguous blocks, combined by a (tree) reduction at the end.
def dot(a, b, n):
    return (a * b).sum()      # elementwise map, then an associative reduce
    # a @ b  is the same reduction handed to a blocked, multi-threaded BLAS kernel
```

Nothing about the numbers changed — only the claim about *ordering*. `(a*b)` is a pure map (every
lane independent); `.sum()` is a reduction the runtime is now free to do as a balanced tree. On real
hardware that tree becomes: W SIMD lanes accumulating W partial sums at once, C cores each owning a
block, or thousands of GPU threads folding in `log` steps.

## Performance difference

Measured on this box — 16-core CPU, fp64, `n = 20,000,000`, best of 5 (the serial number is the
2M-element loop scaled linearly, since the full loop takes seconds):

| form | what runs | time | speedup |
|---|---|---:|---:|
| `s += a[i]*b[i]` (serial) | one accumulator, no SIMD, no threads | 3481 ms | 1× |
| `(a*b).sum()` | SIMD map + pairwise reduce, **one core** | 94.8 ms | **37×** |
| `a @ b` | blocked, **multi-threaded** BLAS reduce | 9.0 ms | **386×** |

Two independent factors stack: ~37× from exposing the per-element data parallelism to one core's
vector units (plus removing interpreter overhead), then ~10× more from spreading the blocks across
cores. Neither is reachable while the accumulator chain is held serial.

## Why it works — and when it is not free

**Critical path.** Serial accumulation is an `O(n)`-depth dependency chain. The tree reduction is
`O(log n)` depth over `n/W` independent lanes. The hardware parallelism (SIMD width × cores × the
memory system) can only be spent once that depth collapses.

**The catch is floating point.** Reassociating the sum changes the rounding. This is usually *better*
— pairwise/tree summation has `O(log n · u)` error growth versus serial's `O(n · u)` — but for an
ill-conditioned reduction (catastrophic cancellation, a near-zero pivot downstream) the reordered sum
can diverge by orders of magnitude. That is exactly the gramschmidt case in
[`../examples/demo_gramschmidt_fma.py`](../examples/demo_gramschmidt_fma.py): well-conditioned inputs
agree to machine epsilon, ill-conditioned inputs blow up under reassociation. So the transform is
**admissible, not unconditional** — licensed when the reduction is associative-in-intent *and* the
inputs are numerically safe (or the caller explicitly accepts the FP mode).

**Where nest-forge comes in.** In C, Fortran, or Python the `+=` accumulator is a serialization the
*source language* wrote down; a plain compiler will not reassociate it without `-ffast-math` (which
reassociates *everything*, safe or not). A semantics-explicit IR (the DaCe SDFG) records that this is
a **write-conflict-resolution reduction** with an associative operator — so it can emit the parallel
tree directly, choose the SIMD width, and pick a **per-reduction** FP license instead of a global
fast-math switch. The arena then measures *which* compiler × flags × vector length actually realises
the 386× on this hardware. Lifting the language-dependent serialization is the optimization; the
arena is how we prove which lowering wins.
