# The kernel surface — what an agent is handed, and in which language

Design. Not built yet. Supersedes the ad-hoc body rendering in `introspect.kernel_body`.

## The one idea

**A kernel is a pair: an iteration domain and a body that is a pure function of a point in it.**

```
kernel2_0   domain (i0=0:N, i1=0:M)
            body(i0, i1):
                t3 = A[i0, i1] * 2.0
                B[i0, i1] = t3
```

Everything below follows from taking that literally. The domain is already canonical (`0:trip:1`,
`normalize.NormalizeLoopsAndMaps`). What is missing is that the **body** is not a well-defined object
today — it is whatever statements happen to sit under a `MapEntry`, reachable only by re-emitting and
string-slicing off the `for` headers.

## Why it is not well-defined today

`introspect.kernel_body` calls `emit_numpy.map_lines`, then drops the first `len(params)` lines and
dedents by `4 * len(params)`. That is string surgery standing in for an API. It breaks the moment a
body is not a flat statement list, and it cannot answer "give me this kernel in Fortran".

## Fix 1 — normalize the reductions, not the body shape

Two passes become part of `normalize_for_tree`:

| pass | why |
|---|---|
| `NormalizeWCR` | a masked in-body reduction becomes the ordinary seeded-transient + boundary-WCR shape |
| `NormalizeWCRSource` | every WCR edge sources from an `AccessNode`, so a reduction has ONE form |

That is what makes a reduction *recognizable*: after them, every WCR is the canonical
`AccessNode -[wcr]-> MapExit` chain, `detect_reduction_type` reads the op off it, and the emitter can
say what is being reduced rather than emitting whatever the frontend happened to build.

`NestInnermostMapBodyIntoNSDFG` is NOT in this set. It is a good pass and it would make the body a
single node with a signature, but emitting does not need that — the body is reachable from the scope
tree either way, and requiring it would drag a vectorization-pipeline pass with its own preconditions
into the path every tree render takes. Keep it optional, for the phases that want it.

`ExpandNestedSDFGInputs` is already in the pipeline; it stays.

Cost: these run once per normalize. Measure before landing — the current pipeline is 3.5 ms warm on
cavity_flow and this must not turn that into a second.

## Fix 2 — two forms, one waist

The body renders two ways, and only one of them is a lowering source.

**Point form** (canonical). Scalar, indexed by the domain variables, no slices:

```python
def body(i0, i1):
    t3 = A[i0, i1] * 2.0
    B[i0, i1] = t3
```

**Slice form** (derived). Whole-array numpy, no explicit index:

```python
B[:, :] = A[:, :] * 2.0
```

Slice form is for READING and for numpy-level optimization. Point form is what lowers. A slice form is
turned back into point form before any language emit — `numpyto_common.numpy_desugar` already does
exactly this desugaring, so this is a wiring decision, not new machinery.

```
        slice form ──desugar──┐
                              ├──> POINT FORM ──> C / C++ / Fortran   (numpyto)
        SDFG map body ────────┘
```

One waist. A backend is added by teaching numpyto a language, never by teaching the tree a second
lowering path.

## Fix 3 — one table for intrinsics AND reductions

These look like two problems and are one: **the numpy surface has no vocabulary for things the target
languages have.** `min` is one (numpy spells it `np.minimum`, C `fmin`, Fortran `MIN`, DaCe
`dace::math::min`). A tree reduction is the other, and a worse one, because numpy has no way to say it
at all.

The unification: **a reduction is an intrinsic applied over axes.** One table, one namespace `nf`, one
extra column, and reductions fall out of it.

| column | math op | reducible op |
|---|---|---|
| `name` | `nf_min` | `nf_add` |
| `arity` | 2 | 2 |
| `numpy` | `np.minimum` | `np.add` |
| `c` / `cpp` / `fortran` | `fmin` / `nf::min` / `MIN` | `+` |
| `identity` | — | `0` |
| `reducible` | no | **yes** |

`emit_numpy._MATH_INTRINSICS` is already two thirds of the math half; the table **replaces** it rather
than sitting beside it, or the two drift and the C++ says `fmin` while the oracle says `np.minimum`.

The bridge from the SDFG is `detect_reduction_type(wcr)` -> `ReductionType` -> the table row.
`ReductionType.Custom` is refused BY NAME, exactly like an unknown math op — guessing a per-language
spelling is how a kernel silently computes something else.

Shipped runtime, under `nestforge/runtime/`, all four generated from that one table:

| | file | form |
|---|---|---|
| C++ | `nf_math.hpp` | `template <class T> constexpr T nf_min(T, T)` — C++20, `constexpr` where the op allows |
| C | `nf_math.h` | `static inline` per type + `_Generic` dispatch |
| Fortran | `nf_math.f90` | module with a generic interface per op |
| Python | `nf_math.py` | thin numpy aliases, so the point form runs as the oracle unchanged |

The agent writes `nf_min(a, b)` in every language: one spelling, four expansions. And
`build.include_flags` gains `-I<nestforge runtime>` while keeping the dace include **only** for
DaCe-generated frames — an agent-authored kernel builds without it, which is the check that the
`dace::` leak is actually closed.

## Fix 4 — tree reductions, without lying to the agent

Every map WCR is a tree reduction: the map says the iterations are independent, so the accumulation
order is unspecified and an implementation may fold it as a tree. That is exactly what a real backend
does — an OpenMP `reduction(+:acc)` clause, a per-lane accumulator, a `#pragma omp simd reduction`.

Numpy cannot say any of that. Left alone, the slice projection of

```
for i, j:  C[i] += A[i, j] * B[j]
```

is `C[:] = np.sum(A * B[None, :], axis=1)` — which **materializes the whole `A * B` product** and then
reduces it. A backend would fuse the two into one loop with a register accumulator and no buffer at
all. Show the agent the numpy and it optimizes against a cost model that does not exist.

Four rules keep it honest.

**1. The numpy form is the ORACLE, not the performance model.** This is the sentence that matters.
`nf.reduce` under numpy is allowed to materialize a temporary; the lowered C/C++/Fortran is required
not to. The agent is told this in the phase skill, in those words, so it never reads "there is a
buffer here" out of a `np.sum`.

**2. `nf.reduce`'s argument is lowered as an EXPRESSION, never as a buffer.**

```python
C[:] = nf.reduce(nf_add, A * B[None, :], axis=1)
```

numpyto's desugar already turns a slice statement into a loop nest; a recognized `nf.reduce` becomes
the accumulate at the bottom of that nest rather than a second pass over a temporary. Where the
expression genuinely cannot be fused — it is consumed twice, or it is a library call — the emitter
**says so** instead of quietly materializing.

**3. The reduction is on the KERNEL LINE, not only in the body.** The structural fact — which axes
collapse, under which op, into what — belongs where the agent is already looking:

```
kernel2_0  [i0=0:N, i1=0:M]  reduce=(+ over i1 -> C)  reads=['A','B'] writes=['C']
```

Reading the body then becomes optional rather than required, which is the whole point of the tree.

**4. Reassociation is a RECORDED decision, never a silent one.** Floating-point `+` is not
associative: a tree fold and a sequential fold differ in the last bits. So each reduction carries
`reassociable`, defaulted from where it came from — a WCR the frontend built from a parallel map is
already order-unspecified, so it is reassociable; one lifted out of a sequential loop is not until
someone says so. Validation then compares against the SAME order the lowered code uses, so
bit-exactness stays the gate. A reduction is never the excuse to swap the oracle for a norm.

## The API

```python
Session.kernel_body(nest_id, form="point", lang="python") -> str
```

- `form`: `"point"` | `"slice"`
- `lang`: `"python"` | `"c"` | `"cpp"` | `"fortran"` — anything but `python` implies `form="point"`
  (a slice has no meaning in C) and goes through numpyto.
- Refuses with a reason, never a partial body. A kernel the projection cannot express is a fact the
  agent needs, and `describe(bodies=True)` already reports it as `<not emitted: ...>`.

`describe(bodies=True)` renders `form="point", lang="python"`.

## Order to build

1. **The table**, with both halves and the `identity` / `reducible` columns, replacing
   `_MATH_INTRINSICS`. Everything else reads it.
2. `NormalizeWCR` + `NormalizeWCRSource` into `normalize_for_tree`, with the timing check, and
   `reduce=(op over axes -> target)` on the kernel line. This is the part that stops the tree hiding
   a reduction, and it is useful before any of the language work.
3. The four runtime files + the refusal path + `build.include_flags`. Closes the `dace::` leak for
   the paths that exist today.
4. `kernel_body` re-cut against the scope tree — deletes the `lines[headers:]` string surgery.
5. `form="slice"`, with `nf.reduce` and the no-materialize invariant.
6. `lang=` through numpyto.

## Open

- **A reduction in point form is not a pure function of the point.** `acc += ...` breaks the framing
  the whole design opens with. Either the carried value joins the body signature, or the domain splits
  into reduced axes and free axes and the body is a function of the free ones returning a value to
  fold. The second is closer to what a backend does. Undecided.
- **`Min_Location` / `Max_Location` / `Exchange`.** Reductions in dace's enum that have no clean numpy
  spelling and no obvious `nf` signature. Probably refused at first, but they are real (argmin shows
  up in the corpus).
- **Slice form for a body with control flow.** An `if` inside a map body has no slice rendering short
  of `np.where`. Probably: slice form is offered only for a straight-line body, with the reason given.
- **Where the "cannot fuse this reduction" check lives.** It has to see the whole expression tree, so
  it is a numpyto-side analysis, not something the tree can decide alone.
