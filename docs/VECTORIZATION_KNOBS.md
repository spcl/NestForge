# Vectorization knobs — which ones are a real search axis

Screen before you measure. A knob whose flip emits **byte-identical C++** cannot change a timing, so it
is settled at codegen for free; only what survives that screen deserves a compile+run cell.

Evidence: a structural census over all 259 TSVC nests + a codegen-hash probe over 7 nests
(s000, s271, s2275, s275, s1161, s441, masked_store_const, s115, s231), AVX512 host, `simplify-parallel`
opt-mode, `skip-taskloops` strategy.

## Verdict

| knob | live? | why |
|---|---|---|
| `remainder_strategy` (+`assume_even`) | **yes — top axis** | 4 arms, all distinct codegen; 127/130 innermost extents are symbolic so the split is always emitted |
| `widths` (K=1 ladder) | **yes** | distinct codegen at every rung on all 5 tiling nests |
| `target_isa` | **yes, off-box** | this host offers AVX512 + SCALAR only; the interesting spread (AVX2/NEON/SVE) is a CI/other-machine question |
| `fuse_multiply_add` | **yes, nest-gated** | distinct on 4/7 — only where an `a*b + c` pattern exists. Costs 1 ULP |
| `branch_mode` merge vs fp_factor | **NO on fp64** | `LowerITEToFpFactor` blends INTEGER `ITE` only; a float arm keeps its `ITE` and lowers to a real select. Byte-identical codegen on 7/7, including nests that DO have a same-write-set data conditional |
| `loop_to_map_permissive` | untested where it counts | no effect on 7/7 already-mapped nests; only the 131 map-less nests could move |
| `widths` K>=2 | **never applied** | needs a >=2-param map; every probed nest tiles a 1-param map, so `MarkTileDims` raises `NotImplementedError` |
| `expand_tile_nodes`, `validate`, `validate_all`, `device` | no | build-time / target selection, not a code axis |

## The remainder axis in full

Four arms, four distinct emissions (`vectorize_multi_dim.py` dispatch):

- `full_mask` — no split. ONE `0:N:W` map, mask computed and applied on **every** tile.
- `masked_tail` (default) — divisible `__tile_main` interior (mask-free) + W-strided masked boundary slabs.
- `scalar_postamble` + `scalar_remainder_emit="scalar"` — interior + a plain step-1 loop tail.
- `scalar_postamble` + `scalar_remainder_emit="tile_k1"` — interior + a single-lane tile-op tail.
- `assume_even` — asserts divisibility, emits interior only, no remainder at all.

Why it is the axis that should move with the backend: `full_mask` pays a mask per tile, and that price is
ISA-dependent — AVX512 `k`-registers make it near-free, AVX2 has no mask registers so it blends, SVE is
predicated by nature. Same knob, opposite winner. That is C1 stated at the remainder level.

Why it is live even at divisible sizes: the extent reaches the vectorizer as the **symbol** `LEN_1D`, not a
constant (127/130 innermost maps), so divisibility is unprovable and the split is always emitted. The
runtime value decides only the tail's trip count, never whether the tail exists.

## Gaps in what NestForge searches today

`vectorize_variants.py` covers 2 of the 4 remainder arms, and the two entry points disagree about which 2:

| arm | `enumerate_vec_configs` | `descent_axes` |
|---|---|---|
| `masked_tail` | yes | yes |
| `full_mask` | **no** | **no** |
| postamble + scalar | **no** | yes |
| postamble + tile_k1 | yes | **no** |
| `assume_even` | yes | yes |

Also missing: the `w=64` rung, any K>=2 cell, and `loop_to_map_permissive`.

## Bugs this screen turned up

1. **`has_same_write_set_branch` cannot see the branches it is looking for.** It walks
   `all_control_flow_regions()`, whose `recursive` defaults to `False`, so a conditional inside a map
   body's NestedSDFG is invisible. Measured: 30 nests detected, **all 30 have zero maps**; the 26 nests
   that have both a map and a conditional (the whole `s27x` family, `s44x`, `s1161`, `s1279`, `s253`,
   `vif`, ...) are missed. So `fp_factor` is offered only where the tile path cannot run.
   The fix is **not** `recursive=True` — that would add 26 provably no-op cells. Drop `branch_mode` from
   the fp64 enumeration and keep it for integer-output conditionals only.
2. **`assume_even` is asserted, never checked.** It claims every tiled extent is divisible by W, but
   validation runs at `validate_cap(preset)` sizes and timing at `profile_preset` sizes — different
   values. Today's presets are all multiples of 32 for `LEN_1D`/`LEN_2D` so it is accidentally safe, but
   `LEN_3D` M=48 with `w=32`, and any `--random-sizes` run, are not. Gate the cell on the TIMING extent.
3. **The K>=2 paths in `resolved_key` / `variant_name` have never executed.** Nothing enumerates a
   multi-width config, and every probed nest would raise if one did.
4. **An inert nest is measured anyway.** On `s275_n0` and `s231_n0` every knob emits identical code — the
   vectorizer does not apply — yet the lane still spends its full descent (~66 `measure()` calls worst
   case) on them. One codegen-hash comparison rules the whole axis out.

## Next

1. Cheap corpus-wide screen (CI, codegen only, no compile): hash base vs one flipped knob per nest; report
   how many of the 259 nests are inert and how many tile at all.
2. Add `full_mask` and the missing postamble arm; make both entry points agree on the arm set.
3. Time the 4-arm remainder axis per ISA — the C1 heatmap, and the reason this axis was worth the screen.
