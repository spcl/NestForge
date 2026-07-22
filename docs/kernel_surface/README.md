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

## Fix 1 — make the body one object, by normalization

Three passes, all of which already exist in dace, become part of `normalize_for_tree`:

| pass | why |
|---|---|
| `NormalizeWCR` | a masked in-body reduction becomes the ordinary seeded-transient + boundary-WCR shape |
| `NormalizeWCRSource` | every WCR edge sources from an `AccessNode`, so a reduction has one form |
| `NestInnermostMapBodyIntoNSDFG` | **every innermost map body becomes exactly one `NestedSDFG`** |

The third is the load-bearing one: after it, "the body of kernel K" is a single node with a signature
(its connectors) and a graph — an object, not a slice of a line range. The first two matter because a
reduction is the one construct whose body shape varies for reasons the agent should never see.

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

## Fix 3 — the agent never sees `dace::`

Today a generated nest compiles against `-I<dace runtime include>` (`build.include_flags`). That drags
`dace::math::min`, `dace::math::ifloor` and the rest into anything an agent writes. An agent asked for
C++ should not have to know DaCe exists.

Ship a small runtime under `nestforge/runtime/`, one file per language, all generated from **one
table**:

| | file | form |
|---|---|---|
| C++ | `nf_math.hpp` | `template <class T> constexpr T nf_min(T, T)` — C++20, `constexpr` where the op allows |
| C | `nf_math.h` | `static inline` per type + `_Generic` dispatch |
| Fortran | `nf_math.f90` | module with a generic interface per op |
| Python | `nf_math.py` | thin numpy aliases, so the point form runs as the oracle unchanged |

The table is one row per op: `(name, arity, c, cpp, fortran, numpy)`. `emit_numpy._MATH_INTRINSICS` is
already two thirds of it — the table replaces it rather than sitting beside it, or the two drift and
the C++ says `fmin` while the oracle says `np.minimum`.

Rules:

- The agent writes `nf_min(a, b)` in every language. One spelling, four expansions.
- An op that is NOT in the table is refused at emit time with its name. Guessing a spelling per
  language is how a kernel silently computes something else.
- `build.include_flags` gains `-I<nestforge runtime>`. It keeps the dace include **only** for
  DaCe-generated frames; an agent-authored kernel is built without it, which is the check that the
  leak is actually closed.

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

1. The three passes into `normalize_for_tree`, with the timing check. Nothing else works without a
   well-defined body.
2. `kernel_body` re-cut against the body `NestedSDFG` — deletes the string surgery.
3. The math table + the four runtime files + the refusal path. Independently useful: it closes the
   `dace::` leak for the paths that exist today.
4. `form="slice"`.
5. `lang=` through numpyto.

## Open

- **Does every map body survive `NestInnermostMapBodyIntoNSDFG`?** It is a vectorization-pipeline pass
  with its own preconditions. Corpus-wide check before it becomes a default.
- **Slice form for a body with control flow.** An `if` inside a map body has no slice rendering short
  of `np.where`. Probably: slice form is offered only when the body is straight-line, and the agent is
  told why not otherwise.
- **Reduction in point form.** `acc += ...` across the domain is not a pure function of the point. It
  needs an explicit carried-value in the signature, or the domain is split into the reduced axes and
  the rest. Undecided, and it is the case most likely to break the "pure function" framing.
