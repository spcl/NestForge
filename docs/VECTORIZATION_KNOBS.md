# Vectorization knobs — which ones are a real search axis

Screen before you measure. A knob whose flip emits **byte-identical C++** cannot change a timing, so it
is settled at codegen for free; only what survives that screen deserves a compile+run cell.

Evidence: a structural census over all 259 TSVC nests + a codegen-hash probe over 7 nests
(s000, s271, s2275, s275, s1161, s441, masked_store_const, s115, s231), AVX512 host, `simplify-parallel`
opt-mode, `skip-taskloops` strategy.

## Verdict — three searched axes

| knob | verdict | why |
|---|---|---|
| `remainder_strategy` | **searched — top axis** | 4 arms, all distinct codegen; 127/130 innermost extents are symbolic so the split is always emitted |
| `widths` (K=1) | **searched** | distinct codegen at every rung on all 5 tiling nests |
| `branch_mode` merge vs fp_factor | **searched, integer-gated** | `LowerITEToFpFactor` blends INTEGER `ITE` only; a float arm keeps its `ITE` and lowers to a real select. Byte-identical codegen on 7/7 fp64 nests, including ones that DO have a same-write-set data conditional — so it is offered only where a conditional writes an integer |
| `target_isa` | **decided: `AUTO`** | `AUTO` resolves to the host's best ISA at expansion. Searching it measured the host's own detection against itself; the SCALAR floor is what the plain (non-vectorized) DaCe lane already is |
| `fuse_multiply_add` | **decided: follows the FP rung** | contraction is what every rung above `strict-ieee` already grants the compiler. Denying it would make the tile path stricter than the baseline it is divided by, and no config differs *only* in FMA |
| `assume_even` | **dropped** | it ASSERTS divisibility instead of emitting a remainder, so it does not answer "how is the remainder emitted" — and nothing can establish the assertion here: the extent is a symbol, and the validating size and the timing size differ |
| `widths` K>=2 | **dropped: CPU SIMD is 1-D** | a lane count, not a blocking factor. Needs a >=2-param map to apply at all (`MarkTileDims` raises on every probed nest) and makes `target_isa` dead on top |
| `loop_to_map_permissive` | not offered | no effect on 7/7 already-mapped nests; only the 131 map-less nests could move, and permissive `LoopToMap` is the unsound direction |
| `expand_tile_nodes`, `validate`, `validate_all`, `device` | not axes | build-time / target selection, not a code axis |

The search is a coordinate descent over those three axes, so it costs `sum of axis sizes` per round —
3 widths + 4 arms (+2 branch modes where live), from three seeds — not the `widths x arms x branch`
product. There is no cap and nothing is truncated, so no cell is dropped silently.

## The remainder axis in full

Four arms, four distinct emissions (`vectorize_multi_dim.py` dispatch):

- `masked` (default) — divisible `__tile_main` interior (mask-free) + W-strided masked boundary slabs.
- `fullmask` — no split. ONE `0:N:W` map, mask computed and applied on **every** tile.
- `posttail` — `scalar_postamble` + `scalar_remainder_emit="scalar"`: interior + a plain step-1 loop tail.
- `k1tail` — `scalar_postamble` + `scalar_remainder_emit="tile_k1"`: interior + a single-lane tile-op tail.

Each arm sets **both** fields it owns. A descent that moved only `remainder_strategy` would carry the
previous arm's `scalar_remainder_emit` along and land on a config the orchestrator rejects outright
(`tile_k1` is valid only under `scalar_postamble`).

Why it is the axis that should move with the backend: `full_mask` pays a mask per tile, and that price is
ISA-dependent — AVX512 `k`-registers make it near-free, AVX2 has no mask registers so it blends, SVE is
predicated by nature. Same knob, opposite winner. That is C1 stated at the remainder level.

Why it is live even at divisible sizes: the extent reaches the vectorizer as the **symbol** `LEN_1D`, not a
constant (127/130 innermost maps), so divisibility is unprovable and the split is always emitted. The
runtime value decides only the tail's trip count, never whether the tail exists.

## The coverage hole this closed

`vectorize_variants.py` covered 2 of the 4 remainder arms, and its two entry points disagreed about which 2
— so `full_mask` was searched by neither and each covered an arm the other never saw:

| arm | deep-sweep enumeration | `descent_axes` |
|---|---|---|
| `masked` | yes | yes |
| `fullmask` | **no** | **no** |
| `posttail` | **no** | yes |
| `k1tail` | yes | **no** |

There is now one entry point and one `REMAINDER_ARMS` table. The deep-sweep enumeration is gone: it had
never had a caller, so the "two entry points" were one real search and one that only tests ran.

## Bugs this screen turned up

1. **`has_same_write_set_branch` cannot see the branches it is looking for.** It walks
   `all_control_flow_regions()`, whose `recursive` defaults to `False`, so a conditional inside a map
   body's NestedSDFG is invisible. Measured: 30 nests detected, **all 30 have zero maps**; the 26 nests
   that have both a map and a conditional (the whole `s27x` family, `s44x`, `s1161`, `s1279`, `s253`,
   `vif`, ...) are missed. So `fp_factor` is offered only where the tile path cannot run.
   FIXED: `conditional_blocks` walks recursively, and `branch_mode` is gated on
   `has_integer_conditional_write` rather than on any conditional — `recursive=True` alone would only have
   added 26 provably no-op cells.
2. **`assume_even` was asserted, never checked.** It claims every tiled extent is divisible by W, but
   validation runs at `validate_cap(preset)` sizes and timing at `profile_preset` sizes — different
   values. Today's presets are all multiples of 32 for `LEN_1D`/`LEN_2D` so it was accidentally safe, but
   `LEN_3D` M=48 with `w=32`, and any `--random-sizes` run, are not. FIXED by dropping the cell: nothing in
   this sweep can establish the assertion.
3. **The K>=2 paths in `resolved_key` / `variant_name` had never executed.** FIXED by removing them —
   every cell is K=1.
4. **An inert nest was measured anyway.** On `s275_n0` and `s231_n0` every knob emits identical code — the
   vectorizer does not apply — yet the lane spent its whole descent on them.
   FIXED: the lane now screens on the emitted SOURCE before it compiles. Measured on an inert nest,
   18 compiles collapse to 1.

## The two screens, in the order they fire

| screen | key | cost per cell | what only it can see |
|---|---|---|---|
| source | `dedup.cpp_body_key` on the emitted C++ | one codegen | a knob the vectorizer ignores on THIS nest — before a compile is spent |
| artifact | `dedup.variant_key` on the WHOLE object | codegen + compile | two DIFFERENT sources the compiler folds to the same code |

The artifact screen keys **every** function body, not `__program_<name>`. That entry point is a
three-instruction trampoline into `__program_<name>_internal`, so it disassembles identically for every
config: keying on it collapsed the whole descent onto cell 1 and the lane reported a winner it never
timed (measured: one asm hash `ba7b81640a` across widths 8/16/32, while `cpp_body_key` differed). Init and
exit stay IN the key — init allocates the persistent storage, so a config that allocates differently is a
different build. Over-separating costs one extra measurement; over-collapsing deletes the axis.

Both screens also RECORD on the collapse path. The artifact screen used to return without writing the
source cache, so that cache held one entry for an entire lane.

Neither subsumes the other, so both stay. The source screen is what closes finding 4; the artifact screen
is the 42%-collapse one already measured over the flag axis.

Measured, three descent seeds per nest, real DaCe emissions (`tests/test_dedup.py::test_the_codegen_screen_*`):

| nest | distinct sources from 3 seeds | |
|---|---|---|
| `s275_n0`, `s231_n0` | 1 | inert — collapses, no compile spent |
| `s000_n0`, `s1161_n0` | 3 | tiles — stays distinct, the width axis survives |

Both directions are asserted. A screen that over-collapsed would silently delete the axis it exists to skip,
which is the failure mode worth guarding: it would look like a faster sweep, not like a lost search.

The descent additionally memoizes on `resolved_key` across all three seeds and both rounds
(`vectorize_variants.memoized`): the seeds meet in the middle of the width ladder, and the raw descent
re-generated the same config 1.5×–3.75× (measured 28–35 calls for 8–11 distinct configs). A repeat is not
free — it pays a deepcopy, the vectorizer, DaCe codegen and a clang-format run before the source screen
can notice it has seen that code before.

## Next

1. Cheap corpus-wide screen (CI, codegen only, no compile): report how many of the 259 nests are inert and
   how many tile at all. The per-nest machinery for this now exists; only the sweep driver is missing.
2. Time the 4-arm remainder axis per target — the C1 heatmap, and the reason this axis was worth the screen.
