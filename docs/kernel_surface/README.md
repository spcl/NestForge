# The kernel surface — what an agent is handed, and in which language

Design. Not built yet. Supersedes the ad-hoc body rendering in `introspect.kernel_body`.

## The one idea

**A kernel is a pair: an iteration domain and a body that is a pure function of a point in it.**

A reduction is the one case where that is not literally true — `acc += ...` carries state across
points. It is handled by splitting the domain into FREE axes and REDUCED axes: the body stays a pure
function of a free point, and returns the value to fold over the reduced ones. See Fix 4.

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

## Fix 3 — intrinsics are numpyto's job, not ours

Scrapped: the `nf` namespace, the per-language columns, the four shipped headers. numpyto already owns
all of it, and duplicating it in nest-forge would have been a second table to drift.

What it already has, read from the source:

- `numpyto_common/operators.py` — `BINOP` / `CMPOP` / `BOOLOP` per target, `FORTRAN_INTRINSICS`
  (`sqrt` -> `SQRT`, `fmin` -> `MIN`, `ceil` -> `CEILING`), and `FORTRAN_FN_EXPR` for the ops Fortran
  has no intrinsic for (`cbrt`, `log2`, `expm1`, ...).
- `numpyto_common/lowering.py` — `_NP_FUNC_ALIASES` normalizing numpy synonyms (`amax` -> `max`).
- The generated C carries its own correctness notes we would only have got wrong: `min`/`max` macros
  that PROPAGATE NaN like numpy rather than like libm, `__npb_fmin` / `__npb_fmax` single-evaluation
  helpers because libm's SUPPRESS NaN, `__npb_sign` because `(x>0)-(x<0)` gives 0 for NaN and
  double-evaluates.
- `FloorDiv` never reaches the operator table — it is intercepted and emitted through an `int_floor`
  macro, which is the exact trap a hand-rolled header would have fallen into.

**So nest-forge's rule is one line: emit correct `np.<op>`, and never invent a name.** The allowed set
is whatever numpyto lowers — not a table we maintain — and when numpyto refuses, that refusal is the
error the agent sees. `emit_numpy._MATH_INTRINSICS` keeps its existing and only job, DaCe tasklet code
-> numpy; it gains no language columns.

`int_floor` / `int_ceil` come from numpyto too, through an API it exposes for exactly this:

```python
numpyto_c.arith_header_source(lang)      # the text, include-guarded  (lang: "c" | "cpp")
numpyto_c.write_arith_header(out_dir, lang)   # ... written next to the kernel, returns the path
numpyto_c.ARITH_HEADER_NAME               # {"c": "npb_arith.h", "cpp": "npb_arith.hpp"}
```

It is byte-identical to what the emitter inlines above a kernel body, so an agent-authored kernel
that includes it computes what the numpy reference does: `min`/`max` propagate NaN as numpy's do
(libm's suppress it), `int_floor`/`int_ceil` round toward -inf/+inf for BOTH signs (C's `/` truncates
toward zero, so `-7 / 2` is -3, not -4), and `python_mod` takes the sign of the divisor. Each
dispatches on operand type -- `_Generic` in C, `if constexpr` in C++ -- so the caller never spells a
width.

That is the whole reason not to hand-roll this: every one of those is a silent wrong answer, not a
compile error.

**Correction:** an earlier revision of this document claimed `build.include_flags` drags the dace
runtime include into anything an agent writes. That is wrong, and checking it costs one grep.
`include_flags` has exactly one caller, `build.compile`, which is only ever handed a frame from
`generate_program_folder(sdfg, ...)` -- a DaCe-generated frame, which legitimately needs those
headers. A translated or agent-authored source compiles through `arena.compile_object`, whose command
line is the compiler, `FP_MODES[fp_mode]`, `-c`, the source and the output. No dace include, no
`dace::` anything. So there is nothing to fix here, and the backlog item is dropped.

## Fix 4 — the reduction gap is real, and it is measured

Every map WCR is a tree reduction: the map says the iterations are independent, so the fold order is
unspecified and a backend may use a register accumulator or an OpenMP `reduction` clause.

The question was whether the numpy projection throws that away. It does. Translating
`C[:] = np.sum(A * B[None, :], axis=1)` through numpyto gives (Fortran, abridged):

```fortran
do si0 ...; do si1 ...
    x_cb1(si1+1, si0+1) = A(si1+1) * B(si1+1)     ! a FULL temporary for A*B
do x_ax0 ...
    x_cb2(x_ax0+1) = 0.0_c_double
    do x_rd0 ...
        x_cb2(x_ax0+1) = x_cb2(x_ax0+1) + x_cb1(x_rd0+1, x_ax0+1)   ! then reduce it
```

Buffer, then reduce — two passes. numpyto's slice fusion is real (`A[1:N-1] = (B[:N-2] + B[2:]) / 3`
becomes one loop nest, not four temporaries), but it does not reach across the reduction.

The **folded** form, same kernel, lowers to what you would write by hand:

```fortran
do i = 0, N-1
    acc = 0.0_c_double
    do j = 0, N-1
        acc = acc + A(j+1, i+1) * B(j+1)
    end do
    C(i+1) = acc
end do
```

One loop nest, register accumulator, no buffer.

**So `folded` is the default**, and that inverts what this document said a revision ago. The earlier
reasoning was that a WCR on a parallel map is order-unspecified so it should render `declared` — true
about the semantics, and wrong about the consequence, because `declared` is the form that costs a
buffer today. Defaulting to it would have handed the agent the slower shape and called it the
canonical one.

The two representations stand, and the agent still picks:

| | reads as | lowers to | order |
|---|---|---|---|
| `folded` (default) | an explicit loop with the table's identity as the seed | one nest, no buffer | pinned |
| `declared` | `np.sum(...)` — shorter, and the numpy a human would write | buffer + reduce, today | unspecified |

`folded` pinning the order does not forbid a tree reduction downstream: the freedom is re-expressed
where it belongs, as an OpenMP `reduction` clause or a vectorizer decision on the loop, not as a
string in the numpy. And the emitter's hard invariant is unchanged — whatever it emits is valid numpy
that reproduces the SDFG.

The kernel line still names the reduction, since that is cheap and structural:

```
kernel2_0  [i0=0:N, i1=0:M]  reduce=(+ over i1 -> C)  reads=['A','B'] writes=['C']
```

If numpyto later fuses a reduction into its slice-fusion nest, `declared` becomes free and the default
should be revisited. That is a numpyto change, not a nest-forge one.

## The representation is pure, runnable numpy -- or it does not count

A kernel handed to an agent is **a complete numpy module it can paste in a file and run**, or **C /
C++ generated from that same numpy by the translator**. Nothing else is a representation.

Two things failed that bar and are fixed:

- **A body without its headers is a fragment.** `kernel_body` returns the statements the tree prints
  under a line, and they reference loop variables that exist only inside the headers it stripped.
  Useful as an excerpt, not runnable. `kernel_source` is the representation: preamble, `def` with a
  real signature, the loop nest.
- **The emitted numpy had no imports.** `np`, `int_floor` and `int_ceil` were supplied by
  `EMITTED_BUILTINS` at load time, so the source only ran inside `load_emitted`'s namespace -- and a
  translator reading it would meet three undefined names. `STANDALONE_PREAMBLE` now emits them as
  source, verified to match the injected helpers on every sign combination of both divisions.

Being runnable is also what makes "correct" checkable rather than asserted: emit the kernel, `exec` it
in a BARE dict, run it, and compare against what the SDFG computes. A representation nothing can
execute cannot be shown to be correct. That test exists.

And C/C++ are not a second emitter. They are this numpy, lowered by numpyto -- one waist, so a
backend can never disagree with the oracle about what the kernel means.

## The API

```python
Session.kernel_source(nest_id, lang="python") -> str   # the REPRESENTATION: a runnable module
Session.kernel_body(nest_id, form="point") -> List[str]  # the excerpt the tree prints
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
4. `kernel_body` re-cut against the scope tree — deletes the `lines[headers:]` string surgery. **done**
5. `form="slice"`, with `nf.reduce` and the no-materialize invariant.
6. `lang=` through numpyto. **done** — `kernel_source(lang="c"|"cpp"|"fortran")` extracts the nest on a
   detached copy (a projection, never a mutation), reuses `prepare` + `emit_sources`. C and C++ are the
   `.c` and `.cpp` of ONE `--target c` emit; Fortran is `--target fortran`. numpyto spells every
   intrinsic — nest-forge only hands it `np.<op>`.

## Open

- **`Min_Location` / `Max_Location` / `Exchange`.** Reductions in dace's enum that have no clean numpy
  spelling and no obvious `nf` signature. Probably refused at first, but they are real (argmin shows
  up in the corpus).
- **Slice form for a body with control flow.** An `if` inside a map body has no slice rendering short
  of `np.where`. Probably: slice form is offered only for a straight-line body, with the reason given.
- **Where the "cannot fuse this reduction" check lives.** It has to see the whole expression tree, so
  it is a numpyto-side analysis, not something the tree can decide alone.


## Re-normalizing after a move — measured, and what dirty-tracking would actually buy

The agent re-normalizes after every fusion. On npbench cavity_flow (99 kernels):

| | cost |
|---|---|
| normalize, cold | ~1000 ms (once) |
| re-normalize, nothing changed | 3.3 ms |
| re-normalize after one fusion, dense renumbering | 120 ms |
| re-normalize after one fusion, **stable names** | **37 ms** |
| the fusion move itself, for scale | 12–22 ms |

The 120 -> 37 ms came from making a canonical name KEEP its index instead of renumbering densely from
zero. Dense renumbering meant dropping one transient shifted every later name, so a single move
renamed 21 arrays. The correctness half of that matters more than the speed: an id the agent read off
the tree must still mean the same array after a move, or the tree is not a vocabulary.

What is left is one call. `sdfg.replace_dict` walks the whole SDFG once regardless of how many names
it carries, so renaming the single stale transient a fusion leaves behind (`__map_fusion_t57`) costs
32 ms. Per-stage after a move: rename 32 ms, labels 1.6, reductions 0.6, domains 0.6, everything else
under 0.3.

**So dirty/clean region tracking would optimize the part that is already cheap.** A move dirties one
CFG and one state, and skipping the scans over the clean rest saves the ~3 ms those scans cost — not
the 32 ms, which is one global walk inside dace for one name. Worth building when the scans grow
(a program much larger than cavity_flow, or a normalize with more passes), not for the current shape.

The measured alternative for the 32 ms is a state-scoped `state.replace` (9.1 ms — 4.3x), valid when
the name is referenced in exactly one state and in no interstate edge, nested SDFG or symbol mapping.
Deliberately NOT taken: it is a hand-rolled rename that has to get every reference site right, and
getting it wrong is a silent wrong answer rather than an error. 23 ms per move does not buy that risk.
The right home for it is `replace_dict` itself.
