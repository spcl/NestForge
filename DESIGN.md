# nest-forge emitter — contract & open findings

Covers `nestforge/emit_numpy.py`, `emit_libnode.py`, `emit_yaml.py`: the SDFG → numpy source path
whose output the translator turns into C/Fortran and the arena compiles.

## 1. Contract

- **C-style allocation.** The kernel allocates nothing. Inputs, outputs, `__return` and scratch
  transients are all caller-pre-allocated buffer *parameters* written in place. Only true scalars are
  python locals.
- **Sizable buffers only.** Every buffer shape must be a static function of the kernel's size-symbols.
  A shape that reads array data (a CSR span) violates this and must be refused loudly, never emitted.
  (`BACKLOG.md` §B relaxes the first rule for scratch whose extent is constant or names only
  entry-bound symbols: allocate it in `__dace_init` instead of pushing it onto the caller. The
  refusal above stays exactly as-is for a loop-dependent extent.)
- **Read-only emission.** Emission never mutates the caller's SDFG — widening, inlining and `replace`
  all run on a deep copy.
- **Semantics-preserving.** Bit-exact vs numpy wherever fp associativity allows.
- **Signature/manifest parity.** `emit_yaml.array_names` and the numpy signature are the same
  positional list, or native calls pass mismatched pointers.

## 2. Invariants

**Access rendering.** A `(name, subset)` becomes a python string by one rule: `scalar_local` → bare
`name`; otherwise `name[index_str(subset)]`. `access()` is the oracle. `copy_side` and `reshape_side`
still re-implement it with their own squeeze/rank policy — see finding #6.

**Scratch sizing.** `maxsize_loop_scratch` widens a loop-sized transient to its caller-sizable maximum
by monotone lo/hi-corner substitution over `symbol_ranges`. Rigorous for the affine dims it accepts;
it just does not yet check that the bound is data-independent (finding #1).

**Copy direction.** `m.subset` indexes `m.data`, a DaCe invariant. Direction must come from that plus
the edge's subset ranks, not from which `.data` name matched first (finding #2).

**Read-only-ness.** `expand_nested_sdfg_inputs`, `maxsize_loop_scratch` and `emit_nested_sdfg` all deep
copy before any `replace` / `apply_transformations` / descriptor write. Correct; leave alone.

## 3. Open findings

Ranked most-severe first. Closed findings are deleted, not archived — `git log` has them.

1. **Scratch sized by a data-dependent span (SILENT wrong / unallocatable).**
   `symbol_ranges` (`emit_numpy.py:861`) folds every interstate assignment as a size bound, gated only
   by `try pystr_to_symbolic except: continue`. But `pystr_to_symbolic("A_indptr[i]")` *succeeds* — it
   yields a `Subscript` — so spmv ingests `start_0 = A_indptr[0]` and shapes a buffer
   `A_indptr[M+1] - A_indptr[0]`. `reject_unsizable_scratch` (`emit_numpy.py:955`) then accepts it:
   `free_symbols - known` is empty because `A_indptr` is the sympy `Function` head, not a free symbol.
   **Fix:** a `sizable(expr, kernel_symbols)` predicate that tree-walks for a `Subscript`/`Indexed` or
   any atom named in `sdfg.arrays`, gating both sites. Never `free_symbols` — it is structurally blind
   to indexed data.

2. **Same-array in-place copy reverses direction (SILENT wrong).**
   `copy_lines` (`emit_numpy.py:420`) tests `m.data == e.dst.data` first, so when source and
   destination are the same buffer it treats `m.subset` — the source per the DaCe invariant — as the
   destination. `A[i] = A[j]` emits `A[j] = A[i]`.
   **Fix:** resolve direction structurally, disambiguating the same-name case.

3. **Reduce keepdims compares buffer descriptors, not memlet subsets (SILENT wrong).**
   `emit_reduce` (`emit_libnode.py:507-509`) reads `sdfg.arrays[...].shape` on both sides, but the
   emitted call operates on *subsets*. A reduce from whole `A:(N,M)` into the slice `B[i,:]` of
   `B:(K,M)` gets `keepdims=True` and broadcasts `(1,M)` into `(M,)` — ValueError.
   **Fix:** compare `edge.data.subset` ranks.

4. **Descriptor sync overwrites a legitimate inner shape (SILENT wrong).**
   `emit_nested_sdfg` (`emit_numpy.py:543`) blanket-replaces the inner descriptor's whole shape with
   the outer's. An under-offset multi-dim connector then emits `Z[j]` (a row) where `Z[j,k]` was meant.
   `reject_underranked_codeblock_index` guards the interstate-code case but not dataflow memlets.
   **Fix:** narrow to the scalar ↔ size-1 reconciliation actually intended.

5. **Negative-step end-inclusive `+1` truncates (SILENT wrong).**
   `map_lines` (`emit_numpy.py:630`) renders `range(beg, end + 1, step)` regardless of step sign, so a
   reverse map emits `range(N-1, 1, -1)` and drops element 0.
   **Fix:** `end + sign(step)`.

6. **Three copies of the access rule.** `access` (52), `copy_side` (384) and `reshape_side` (469) each
   re-decide scalar-local-vs-indexed with their own squeeze policy. #2 and the closed `A_index`
   regression both came from a site that hand-rolled the rule.
   **Fix:** one function parameterised by squeeze policy; the other two become calls.

7. **Sequential `symbol_mapping` (SILENT wrong, unlikely).**
   `emit_nested_sdfg` (`emit_numpy.py:561`) emits the mapping as ordered assignments, so a swap
   `{i: j, j: i}` clobbers.
   **Fix:** simultaneous binding via temps.

8. **Reordered unconditional branch (SILENT wrong, unlikely).**
   `emit_conditional` (`emit_numpy.py:737-746`) moves the unconditional branch to a trailing `else`.
   A keyed branch *stored after* it is dead in DaCe (first match wins) but becomes live here; two
   unconditional branches emit two `else:` — a SyntaxError.
   **Fix:** preserve branch order, stop at the first unconditional.

## 4. Do not touch

The CFG/dataflow walk (`emit_region`, `state_body`, `map_lines`, `emit_loop`, `emit_conditional`,
`render`), the libnode registry (`LIBNODE_EMITTERS`), `normalize_casts`, and the read-only threading.
All general and correct.

`strip_scalar_local_subscript` (`emit_numpy.py:750`) is regex surgery over raw DaCe strings and wants
deleting — but only once size-1 transients are front-normalized to `Scalar` on the working copy, which
is a larger change than any finding above. Until then it is load-bearing.
