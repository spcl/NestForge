# nest-forge — parallel-region handling, linking, and stability under parallelism

An extracted nest is not just a computation; it carries a **schedule**. When the arena compiles a nest
and links the winners into one program, the parallelism of each nest and of its call context decides
what is legal to emit, whether the linked program even runs, and whether the result is reproducible.
This document specifies (1) the parallelism descriptor a nest must carry, (2) the compile-intent it
implies, (3) what happens when several parallel static libraries are linked into one driver, and (4)
how parallelism enters the numerical-stability metric. It is the CPU analogue of the host/device
schedule cut already noted in the plan, and it is deliberately co-designed with `docs/FP_RISK.md`,
because a parallel reduction is a reassociation event.

## 1. The parallelism descriptor

A DaCe `MapEntry` is parallel *by definition* — its iterations are independent (a reduction is carried
by a WCR, not by a loop-carried scalar) — unless its `schedule` is explicitly `Sequential`. A
`LoopRegion` is sequential by definition (it has a loop-carried induction variable and may carry
arbitrary dependences). So the nest's own parallelism is read directly off the extracted node:

- `MapEntry`, `schedule != Sequential`  -> **parallel** (independent iterations; may have a WCR reduction).
- `MapEntry`, `schedule == Sequential`  -> **sequential**.
- `LoopRegion`                          -> **sequential**.

But the nest's own parallelism is not enough. What matters for codegen is whether the nest is extracted
*inside* something already parallel. Walk the ancestors of the extraction point:

- `parent_is_parallel` = any enclosing scope (an outer `MapEntry` we did not extract, or — at
  link time — the driver's own `#pragma omp parallel`) is a parallel region.

The `Boundary` therefore gains two fields, both computed at extraction time from the schedule and the
ancestor walk:

```
Boundary.nest_parallelism  : "parallel" | "sequential"     # the extracted node's own schedule
Boundary.parent_is_parallel: bool                          # is the call site already inside a parallel region
```

For the default `outer` strategy the extracted nest is outermost, so `parent_is_parallel` is `False`
and the nest owns whatever parallelism it has. The field only becomes `True` for `innermost` /
device-cut strategies that extract a nest sitting inside an outer parallel map.

## 2. Compile intent

The two fields give a three-way decision on how the arena is allowed to compile the nest. The rule is:
**exactly one level owns the thread-parallelism; everything below it is thread-sequential (SIMD only).**

| `parent_is_parallel` | `nest_parallelism` | compile intent | what the emitter/translator requests |
|---|---|---|---|
| `False` | `parallel`   | **nest owns the threads** | emit `#pragma omp parallel for` on the top loop; arena sweeps `OMP_NUM_THREADS` ∈ {1,2,4,…} as a flag axis and picks a winner; reductions become `reduction(+:s)` (flagged in `fp_risk`) |
| `False` | `sequential` | **single-thread** | no `omp parallel`; `#pragma omp simd` / auto-vec only; thread count is not an axis |
| `True`  | *any*        | **single-thread, per-thread body** | the caller already owns the threads; emit **no** `omp parallel` (nested parallelism — see §3); SIMD only. The nest is the body each outer thread runs |

The third row is the case the user flagged: "if the parent-map is a parallel region we must compile
single-thread aim." Emitting a second `omp parallel` inside an already-parallel region is nested
parallelism, which by default runs serially anyway (wasted pragma) and, if nesting is enabled,
oversubscribes. So a nest with `parent_is_parallel` is compiled with thread-parallelism off, keeping
only vectorization — which is always safe to stack under an outer parallel region.

This is a schedule-domain cut, the same shape as the host-wrapper / GPU-device cut: the strategy
decides *where* the parallel boundary is, and every nest below it is compiled as a sequential
per-thread kernel.

## 3. Linking several parallel static libraries into one driver

The whole-program step links each nest's winner `.a` into one driver. If two (or more) of those `.a`
contain OpenMP parallel regions, the behaviour of the linked program depends on three independent
things. This is the "two `.a` with OpenMP, one driver" question, answered in full.

### 3.1 Which OpenMP runtime each library was built with — the dominant factor

OpenMP is not one runtime. GCC's `-fopenmp` links **libgomp**; LLVM/Clang links **libomp**; Intel
links **libiomp5** (ABI-compatible with libomp). A static `.a` compiled with `-fopenmp` contains
unresolved references to that runtime's entry points (`GOMP_parallel`, `GOMP_loop_*` for libgomp;
`__kmpc_*` for libomp/libiomp5).

**Decision: nest-forge mandates a single global OpenMP runtime — `libomp` — for every node library and
the driver.** The runtime is a configuration knob (default `libomp`; the value is recorded so a build
is reproducible), but within one linked program it is fixed: every node library and the driver are
compiled and linked against the *same* configured runtime. Mixing is not "discouraged", it is rejected
at link time.

- **Single mandated runtime** (all `libomp`): the driver links **one** `-fopenmp` against the configured
  runtime, the process holds **one** runtime with **one** thread pool. Sequential driver calls into
  lib A then lib B reuse the same pool (the runtime keeps the team alive between regions), so there is
  no repeated spawn cost and no oversubscription. This is the only configuration nest-forge builds.
  `libomp` is the default because it is the most portable choice (LLVM/Clang native, ABI-compatible
  with Intel's `libiomp5`, and linkable from GCC-compiled objects via `-fopenmp=libomp` on Clang or by
  linking `libomp` explicitly), so a mixed-compiler set of node libraries can still share one runtime.
- **Different runtimes** (the rejected case: lib A libgomp, lib B libiomp5/libomp): **two runtimes load
  into one process.** Two failure modes: (a) the runtime detects the duplicate and aborts with
  `OMP: Error #15: Initializing libomp, but found libiomp5 already initialized`, or (b) with
  `KMP_DUPLICATE_LIB_OK=TRUE` it proceeds but each runtime keeps its **own** thread pool, so
  `OMP_NUM_THREADS=8` spawns 8 threads in lib A **and** 8 in lib B → up to 16 software threads on 8
  cores → oversubscription and cache thrash. nest-forge prevents this by construction (single mandated
  runtime), not by hoping the libraries agree.

This runtime knob is implemented as `nestforge.build.OpenMPRuntime` (a SEPARATE flag axis, not folded
into the base flags): it maps the one configured runtime to the right per-compiler flags, so a set of
node libraries built with different compilers all target it. LLVM compilers select by name
(`-fopenmp=libomp` — clang/clang++/flang/icx), gcc emits `GOMP_*` calls and links the runtime
explicitly (`-fopenmp` compile, `-lomp` link — libomp's GOMP-compat ABI resolves them). Ready presets
cover the four popular runtimes: `LIBOMP` (LLVM, default), `LIBGOMP` (GNU), `LIBIOMP5` (Intel;
ABI-compatible with libomp), and `LIBNVOMP` (NVIDIA HPC, only via nvc/nvfortran `-mp`, not
interchangeable with the other three). A gcc-compiled kernel linking + running against libomp is tested
end-to-end (`tests/test_build.py`).

**One global thread count.** A single `OMP_NUM_THREADS` (or, equivalently, one `omp_set_num_threads`
call in the driver, §3.4) is passed program-wide and governs every node library. There is no per-library
thread count through the environment (§3.3), so the whole program runs at one, recorded degree of
parallelism — which is also the degree the stability metric (§4) is evaluated at.

### 3.2 Sequential vs nested calls from the driver

Given one runtime:

- The driver calls the libraries **one after another** (the normal case): each library's parallel
  region grabs the shared pool, runs, and returns it. Threads are reused across calls. Benign.
- The driver calls a library **from inside its own `#pragma omp parallel`**: the library's region is
  now **nested**. OpenMP's default is `OMP_MAX_ACTIVE_LEVELS=1` (`OMP_NESTED=false`), so the inner
  region silently runs on **one** thread — correct result, no speed-up, and a surprising slowdown if the
  library assumed it owned the machine. If nesting is enabled, outer `T` × inner `T` threads are
  created → explosion. This is §2's third row seen from the link side: a library that may be called
  from a parallel driver must be built thread-sequential.

### 3.3 Link mechanics and process-global state

- The driver's link line **must** include `-fopenmp` even though the driver itself may contain no
  pragmas; otherwise the libraries' `GOMP_*` / `__kmpc_*` references are undefined symbols at link
  time. One `-fopenmp` resolves all same-runtime libraries.
- Thread affinity and count are **process-global**: `OMP_NUM_THREADS`, `OMP_PROC_BIND`,
  `GOMP_CPU_AFFINITY` are read once and shared by every library. Two libraries cannot independently
  pick thread counts through the environment; if they need different degrees of parallelism, that must
  be set per-region in code (`num_threads(...)`) rather than via the environment.
- Static linking of `.a` does not duplicate the runtime as long as the runtime itself
  (`libgomp.so`/`libomp.so`) is the shared object the driver links once. The duplication hazard in §3.1
  is about linking **two different** runtimes, not about static vs shared.

### 3.4 The driver owns OpenMP initialization

The runtime is initialized **once, by the driver, in its init code, before any node library is called.**
Node libraries never initialize the runtime themselves and never read the environment to decide a thread
count; they only *use* the already-initialized global runtime. Concretely the driver's init sequence,
run before the first library call, is:

1. `omp_set_num_threads(N)` with the one global thread count `N` (from config / the arena cell), so the
   count does not depend on each library's first-touch or on `OMP_NUM_THREADS` being set in the
   environment at run time.
2. set the binding policy once (`omp_set_proc_bind` / `OMP_PROC_BIND`) — process-global, shared by all
   libraries (§3.3).
3. warm the pool with a trivial parallel region so the team exists before the first timed library call
   (keeps thread-spawn cost out of the measured region and guarantees a consistent pool for every
   library).

Because the driver establishes the runtime state up front, every node library inherits one consistent,
already-initialized global runtime — one pool, one thread count, one binding — regardless of the order
in which the libraries are called or which of them would otherwise have triggered lazy initialization.
This is what makes the single-runtime guarantee (§3.1) observable at run time and not just at link time.

### 3.5 What the arena must therefore do

1. Compile every node library **and the driver** against the single configured OpenMP runtime
   (default `libomp`); reject a link set that mixes runtimes rather than relying on the libraries to
   agree.
2. Add `-fopenmp` (against the configured runtime) to the driver link line whenever any node library is
   parallel; emit the driver init sequence of §3.4 before the first library call.
3. Carry each nest's compile intent (§2) so that a nest which can be called from a parallel context is
   built thread-sequential — no nested `omp parallel`.
4. Pass one global `OMP_NUM_THREADS` / `omp_set_num_threads(N)` for the whole program, and record `N` as
   part of the configuration of every parallel cell (it changes both time and, per §4, the result).

## 4. Numerical stability under parallelism

"Numerically stable" here means: **the relative error of a variant, measured against the ieee-strict
*sequential* baseline, does not explode.** The baseline is sequential on purpose — sequential
ieee-strict is the reproducible reference, and every source of divergence is measured as departure from
it.

Thread-parallelism is such a source, in the same class as `-ffast-math`:

- A parallel reduction (`#pragma omp parallel for reduction(+:s)`, or a WCR lowered to a per-thread
  partial-sum tree) **reorders the summation**. By the summation theory in `docs/FP_RISK.md` §2, that
  changes the rounding error, and — crucially — the reordering **depends on the thread count**, so the
  result is a function of `OMP_NUM_THREADS`. Two threads and eight threads give different sums.
- Therefore a parallel reduction is a **reassociation event**, and `fp_risk` treats it exactly like
  `-fassociative-math`: risk scales with the trip count and the summation condition number
  `κ = Σ|xᵢ|/|Σxᵢ|`. It is benign when the addends are same-sign (`κ≈1`) and dangerous when they
  cancel. gramschmidt is the worked example (`tests/test_gramschmidt_fma.py`): the dot-product
  reductions are stable to machine epsilon when the input is well-conditioned and diverge by ~1e-3 when
  the input is ill-conditioned — the same behaviour a thread-count change would produce.
- The whole-program consequence of §3: if two linked libraries each contain a parallel reduction, the
  end-to-end result is `OMP_NUM_THREADS`-dependent across both. The arena must record the thread count
  with each result and gate acceptance on the max-diff-vs-sequential-ieee metric, not silently trade
  reproducibility for cores.

The practical rule: a parallel reduction is offered as a variant only if its max-diff vs the sequential
ieee-strict baseline stays under the mode's tolerance for the thread counts swept; `fp_risk` supplies
the static prior (same-sign reduction → green-light parallelisation; cancelling reduction → keep it
sequential or compensated).

## 5. Sequential vs parallel loop-nest optimization

The same nest can be optimized two ways, and only one of them perturbs the numbers:

- **Order-preserving** (safe): unrolling, tiling/blocking that keeps the reduction sequential within a
  tile, scalar replacement, and SIMD that uses a *strict-order* reduction. These do not change the
  accumulation order, so they are stable by construction and can be applied under any FP mode. A nest
  with `parent_is_parallel` is restricted to exactly these.
- **Order-changing** (reassociation): thread-parallel reductions, SIMD reductions that use multiple
  partial-sum lanes (which the compiler only forms under `-fassociative-math`/`-ffast-math`), and
  pairwise/blocked summation. These change the result as in §4 and are gated by `fp_risk` + the
  differential.

So "seq vs par loop-nest opt" maps onto the FP axis: sequential opts are order-preserving and always
allowed; parallel opts are order-changing and travel with the same reassociation risk and the same
thread-count-dependence. The arena sweeps both and the report shows the accuracy/speed trade-off per
(FP-mode × thread-count) cell rather than collapsing it.

## 6. Summary

- Every extracted nest carries `nest_parallelism` (from its schedule) and `parent_is_parallel` (from
  the ancestor walk).
- One level owns the threads; everything below is thread-sequential, SIMD only. A nest that can be
  called from a parallel context is compiled without its own `omp parallel`.
- Mandate **one** configured OpenMP runtime (default `libomp`) for every node library and the driver;
  reject mixed-runtime link sets (duplicate-runtime abort / oversubscription). The driver initializes
  the runtime once in its init code before any library call (set the one global thread count, bind, warm
  the pool); libraries only use the already-initialized global runtime. Sequential driver calls share the
  one thread pool; calls from inside a parallel driver are nested (serial by default, explosion if
  nesting is on).
- A parallel reduction is a reassociation event: the result depends on the thread count, so stability is
  measured vs the sequential ieee-strict baseline and gated by `fp_risk` + the empirical differential.
