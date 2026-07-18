# nest-forge emitter — audit & refinement

## 1. Design contract

- **C-style allocation.** The kernel allocates nothing. Every array — inputs, outputs, `__return`, and scratch transients — is a caller-pre-allocated buffer **parameter** written in place. Only true scalars are python locals.
- **Sizable buffers only.** Every buffer shape must be a static function of kernel size-symbols so the caller can pre-allocate. A shape that reads array data (a CSR span) violates the contract and must be refused loudly, never emitted.
- **Read-only emission.** Emission never mutates the caller SDFG. All widening/inlining/replace happens on a deep copy threaded through both entry points.
- **Semantics-preserving.** Bit-exact vs numpy wherever fp associativity allows.
- **Signature/manifest parity.** The emitted numpy signature and the YAML manifest's `input_args` must be the *same* positional list, or native calls pass mismatched pointers.

## 2. The general model — orthogonal concerns, one abstraction each

Four cross-cutting concerns, each owned by exactly **one** function. Today three are re-decided at multiple sites with divergent mechanisms — that divergence is the entire live bug surface.

**(a) Access rendering — the size-1-array vs scalar duality.**
A `(name, subset)` becomes a python string in one canonical place:

```
access(sdfg, name, subset, *, write) -> str
```

decides in order: `scalar_local` → bare `name`; `None`/`covers_whole` → `name` / `name[:]`; rank-collapsing reshape → point-index the higher-rank side; else `name[index_str(subset)]`. **Every** producer — tasklet connectors (`_access`), copies (both `_copy_lines` branches), libnode operands (`read_expr`/`write_lhs`), interstate element refs — calls it. This replaces the current six sites and three mechanisms (a `scalar_local` guard here, a regex strip there, a raw f-string in the reshape branch). The regression is precisely the one site (reshape branch) that hand-rolled `f"{data}[{index_str}]"` and skipped the oracle.

**(b) Scratch sizing — one sizability predicate.**
A shape dimension is *sizable* **iff**, after every symbol resolves to a monotone expression over kernel-bounded ranges, its only atoms are kernel size-symbols. **Any reference to array data disqualifies it outright** — judged by walking the expression tree for a sympy `Subscript`/`Indexed` or any atom named in `sdfg.arrays`, **never** by `free_symbols` (sympy hides an indexed array's base as the `Function` head, so every `free_symbols` test is structurally blind to data). One predicate

```
sizable(expr, kernel_symbols) -> bool
```

gates both "is this a valid size bound" (`_symbol_ranges`) and "is this scratch allocatable" (`_reject_unsizable_scratch`). The monotone lo/hi-corner substitution itself is rigorous for the affine dims it accepts and is worth keeping; it just must fire only on sound bounds.

**(c) Copy direction & rank — structural, not textual.**
Source/dest subsets and the reshape decision derive from the memlet + descriptors, not from string matching. `m.subset` indexes `m.data` (a DaCe invariant); resolve src/dst from that and from `edge.data.subset` ranks — not by "which `.data` name matched first," which mis-handles same-array in-place copies. Rank-reduction keepdims must compare **memlet-subset** ranks, not whole-buffer descriptor ranks.

**(d) Read-only-ness — mutate only a copy.**
Already correct and consistent. `_expand_nested_sdfg_inputs`, `_maxsize_loop_scratch`, `_emit_nested_sdfg` all deep-copy before any `replace`/`apply_transformations`/descriptor write. Keep; spend no refactor budget here.

## 3. Overfit verdict

| Subsystem (function) | Verdict | Action |
|---|---|---|
| CFG/dataflow walk (`_emit_region`, `_state_body`, `_map_lines`, `_emit_loop`, `_emit_conditional`, `_render`) | GENERAL | keep |
| `_tasklet_lines` + WCR→augmented-assign (`_WCR_BINOP`) | GENERAL | keep |
| libnode registry (`LIBNODE_EMITTERS`, per-node emitters) | GENERAL | keep |
| `_normalize_casts` / `_DACE_DTYPES` / `_MATH_INTRINSICS` | GENERAL | keep |
| `scalar_local`, `covers_whole`, `is_scalar`, `read_expr`, `write_lhs` | GENERAL (canonical oracles) | keep; make them *the only* access path |
| `_access`, `memlet_expr`, `memlet_lhs` | GENERAL but duplicative | refactor → fold into `access()` |
| `_copy_lines` reshape branch | **BUGGY** | refactor → route through `access()` |
| `_copy_lines` direction resolution | OVERFIT (string-match) | refactor → structural from `m.data`/subsets |
| `_symbol_ranges` (config-symbol fold) | **BUGGY** (admits data `Subscript`) | gate behind `sizable()` |
| `_reject_unsizable_scratch` | **BUGGY** (`free_symbols`-blind) | gate behind `sizable()` (tree-walk) |
| `_max_over_loops` / `_maxsize_loop_scratch` | GENERAL algo, UNSOUND acceptance | keep widening; gate residual check on `sizable()` |
| `_strip_scalar_local_subscript` | OVERFIT (regex over raw strings) | delete once size-1 transients front-normalized |
| `_emit_nested_sdfg` descriptor deep-copy sync | OVERFIT (wholesale shape swap) | narrow to scalar↔size-1 reconciliation |
| `_reject_underranked_codeblock_index` | BANDAID (per-DaCe-pass gap) | keep as loud guard; don't grow more |
| `emit_yaml._arg_order` / `manifest_dict` | **BUGGY** (drops scratch) | refactor → include `_scratch_arrays` |

## 4. Ranked findings (confirmed correctness bugs, most-severe first)

1. **Manifest omits scratch → native call passes misaligned args (SILENT corruption).** `emit_yaml.py::_arg_order` builds `input_args = inputs + extra-outputs + symbols`, but `nest_to_numpy` emits `inputs + extra-outputs + scratch + symbols`. `arena.py::_argtypes`/`_call_native` iterate `input_args` only, so for any scratch-bearing kernel the native ctypes call drops every scratch pointer and slides trailing size-symbols into pointer slots. *Input:* lu/trisolv/syrk — C expects `double*` scratch where arena supplies a `c_int64` size → dereferenced write → segfault or silent wrong maxdiff. Oracle path escapes (keyword call + `make_inputs` allocates scratch), so it is native-only and silent. **Fix:** add `_scratch_arrays` to `_arg_order`/`manifest_dict` in the exact position `nest_to_numpy` uses.

2. **`A_index` NameError regression (CRASH).** `emit_numpy.py::_copy_lines` reshape branch (176–181) builds `lhs = f"{e.dst.data}[{index_str(dst_sub)}]"` instead of routing through `write_lhs`/`read_expr`, bypassing `scalar_local`. A size-1 **transient** written from a higher-rank element takes this branch (ranks differ, both subsets non-None) and emits `A_index[0] = A[j,j]`, while every read spells it bare `A_index`. The transient is excluded from `_scratch_arrays` (`not is_scalar`), so it is never a parameter and never bound → `NameError`. *Input:* lu (`A[j,j]`), trisolv (`L[i,i]`), syrk (`A[i,k]`), nbody (pairwise). **Fix:** guard each side on `scalar_local` and route through `write_lhs`/`read_expr` (i.e. `access()`); force an explicit rank-collapsing index only for the non-scalar reshape case.

3. **spmv scratch sized by a data-dependent CSR span (SILENT wrong / unallocatable).** `_symbol_ranges` (451–458) folds *every* inter-state assignment as a size bound, gated only by `try pystr_to_symbolic except: continue`. But `pystr_to_symbolic("A_indptr[i]")` **succeeds** (yields `Subscript`), so `start_0 = A_indptr[0]`, `stop_0 = A_indptr[M+1]` are ingested; `vals`/`__tmp0` shape `-start_0+stop_0` widens to `A_indptr[M+1]-A_indptr[0]`. `_reject_unsizable_scratch` checks `free_symbols - known = {M}-{M} = ∅` and accepts it — blind because `A_indptr` is the `Function` head, not a free symbol. **Fix:** `sizable()` tree-walk rejecting any `Subscript`/`sdfg.arrays` atom, gating both `_symbol_ranges` acceptance and `_reject_unsizable_scratch`.

4. **Loop `init`/`cond`/`update` skip `_normalize_casts` (CRASH, plausible).** `_emit_loop` (325–327) emits the three strings raw while `_emit_conditional` and `_interstate_lines` normalize. *Input:* `while i < dace.int64(N)` or a scalar-local size-1 bound → `NameError: 'dace'` or bare-vs-`[0]` disagreement. **Fix:** wrap all three in `_normalize_casts` (+ the access oracle for scalar-local reads).

5. **Reduce keepdims compares buffer descriptors, not memlet subsets (SILENT wrong, plausible).** `emit_libnode.py::_emit_reduce` (182–184) tests whole-array ranks; the emitted call operates on subsets. *Input:* reduce reading whole `A:(N,M)` → slice `B[i,:]` of `B:(K,M)` emits `keepdims=True` → broadcast `(1,M)` into `(M,)` ValueError. **Fix:** compare `edge.data.subset` ranks.

6. **Same-array in-place copy reverses direction (SILENT wrong, plausible).** `_copy_lines` (164–167) resolves direction by matching `m.data==e.dst.data` first; when src and dst are the same buffer, `m.subset` (the source per DaCe invariant) is treated as dst. *Input:* `A[i]=A[j]` permutation emits `A[j]=A[i]`. **Fix:** structural direction from the memlet, disambiguate same-name copies.

7. **Descriptor sync overwrites legitimate inner shape (SILENT wrong, plausible).** `_emit_nested_sdfg` (239–240) blanket-replaces the inner descriptor's whole shape with the outer's; an under-offset multi-dim connector then emits `Z[j]` (a row) for `Z[j,k]`. Guarded for interstate code by `_reject_underranked_codeblock_index` but **not** for dataflow memlets. **Fix:** narrow to the scalar↔size-1 reconciliation actually intended.

8. **Negative-step end-inclusive `+1` truncates (SILENT wrong, plausible).** `index_str`/`_map_lines` add `end+1` regardless of step sign; `range(N-1,-1,-1)` emits `range(N-1,1,-1)`, dropping element 0. **Fix:** `end + sign(step)`.

9. **Sequential symbol_mapping / unconditional-branch ordering (SILENT wrong, plausible).** `_emit_nested_sdfg` (258–260) emits `symbol_mapping` as ordered assigns (swap `{i:j,j:i}` clobbers); `_emit_conditional` (359–368) force-moves any unconditional branch to a trailing `else`, making DaCe-dead later branches live (and two unconditionals → double `else:` syntax error). **Fix:** simultaneous binding via temps; preserve branch order, stop at first unconditional.

## 5. Refinement plan (ordered, minimal, prioritized)

1. **Fix the manifest/signature divergence.** Add `_scratch_arrays(sdfg)` to `emit_yaml._arg_order` and `manifest_dict` in the same `inputs → extra-outputs → scratch → symbols` order `nest_to_numpy` emits. *Effect:* closes finding #1 — native validation for lu/trisolv/syrk stops corrupting memory. Highest severity, smallest edit.

2. **Fix the `A_index` regression at the reshape branch.** In `_copy_lines` (176–181), guard each side on `scalar_local(sdfg, data)` and emit via `write_lhs`/`read_expr`; force an explicit rank-collapsing index only when neither side is scalar-local. *Effect:* lu, trisolv, nbody, syrk return to bit-exact GREEN (finding #2). Minimal, local.

3. **Add `sizable(expr, kernel_symbols)` and gate sizing on it.** New predicate does a sympy tree-walk rejecting any `Subscript`/`Indexed`/`sdfg.arrays`-named atom, else requires all remaining atoms ⊆ kernel size-symbols. Call it in `_symbol_ranges` (before folding an inter-state RHS) and in `_reject_unsizable_scratch` / `_max_over_loops` residual (replacing the `free_symbols - known` tests). *Effect:* spmv refuses loudly (`UnsupportedNest`) instead of emitting a CSR-span buffer (finding #3); mlp `N2=S0/S1/S2` still accepted (no Subscript).

4. **Introduce the canonical `access(sdfg, name, subset, *, write)` and route all sites through it.** Fold `_access`, `read_expr`, `write_lhs`, `memlet_expr`, `memlet_lhs`, and both `_copy_lines` branches into calls. *Effect:* the size-1 duality is decided in exactly one place; eliminates the class that produced #2 and the `_access`-vs-libnode `A[0:N]` vs `A` inconsistency. No census change if steps 2 done; hygiene + regression-proofing.

5. **Front-normalize size-1 transients to `Scalar` on the working copy** (a read-only pass mirroring `_maxsize_loop_scratch`'s copy-then-rewrite). *Effect:* `scalar_local` becomes structurally true everywhere; **delete `_strip_scalar_local_subscript`** (the regex string-surgery). Interstate `A_index[0]=…` never arises. Larger; do after 1–4 are green.

6. **De-overfit the cheap remainders.** Wrap `_emit_loop` init/cond/update in `_normalize_casts` (#4); compare memlet-subset ranks in `_emit_reduce` (#5); derive copy direction structurally, disambiguating same-array copies (#6); narrow `_emit_nested_sdfg` descriptor sync to scalar↔size-1 (#7); use `end+sign(step)` in `index_str`/`_map_lines` (#8); simultaneous-bind symbol_mapping and preserve branch order (#9). *Effect:* closes the plausible silent-wrong findings; add one corpus kernel per fix (reverse-loop, in-place-copy, sliced-reduce) to lock them.

*Do not touch:* the CFG/dataflow walk, the libnode registry, `_normalize_casts`, or the read-only threading — all GENERAL and correct.
