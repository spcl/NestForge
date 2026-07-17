# Known failures

**Status: none.** The five failures + one skip recorded here against `1630c61` are all resolved (2026-07-17).
Kept as a record of what they were and what actually caused them.

## Resolved

### 1. `sample_sizes` raised on a leaked outer index -- a regression the review fix-wave introduced

    ValueError: tsvc2 kernel 's1115': boundary symbol 'i' is read by the nest but is not a registered
    parameter, a corpus symbol (['S']) or a shape symbol, so nothing here knows its value.

Broke `tests/test_tsvc_arena.py::test_crosslang_2d_fortran_multiline_signature` and
`tests/test_tsvc_full.py::test_dace_baseline_validates_for_2d_inner_nest`.

The fix-wave replaced a blanket `else: sizes[sym] = 0` with proof / allowance / raise, where the allowance
was "the nest ASSIGNS this symbol" (a leaked induction start). s1115 peels to an inner nest that only
CONSUMES its outer index `i` (`aa[i, j]`, `cc[j, i]`) -- so it matched neither branch and raised.

Fixed by replacing the allowance with the property that actually matters: **can the symbol change the
trip count?** `extract.trip_count_symbols` answers it from loop init/condition/update statements, map
ranges and inter-state-edge CONDITIONS (assignments excluded -- `j = j + 1` carries a value, it does not
decide whether the iteration happens). A symbol absent from that set and from every array shape only
selects WHICH element the nest touches, never HOW MUCH work it does, so binding it to 0 keeps the
iteration space and buffers full-size and hands both oracle and candidate the same value.

This subsumes both shapes (the consumed outer index and the induction start) as one proof, and is
*stricter* where it matters: a symbol that is nest-assigned AND sizes work now raises instead of silently
zeroing. Corpus-wide at `simplify-parallel`, both strategies: **296 nests size OK, 8 raise, 21 leaked
indices bind to 0** -- unchanged raise set, so no kernel was lost.

Not claimed: that 0 is the corpus's real starting value (s123 shifts to `a[0..]` rather than `a[-1..]`).
Resolving the true incoming value from `boundary.parent_sdfg` is the proper follow-up.

### 2. A regression test added by the fix-wave was itself broken

`tests/test_wcr_emit.py::test_tasklet_wcr_symbolic_index_target_is_normalized` --
`TypeError: pairsum() got an unexpected keyword argument 'M'`. `M` sizes `out` and nothing else, so DaCe's
`arglist()` omits it and the emitted signature is `(A, out, N)`: the passed array already carries its own
shape. The test passed `M=8`. The `emit_numpy` normalize_casts fix it guards was fine.

### 3. Two timeouts -- the box, not the code

`tests/test_tsvc_full.py::test_run_kernel_all_lanes_s000` and `::test_omp_emit_lane_runs_for_parallel_nest_s000`
both hit `Failed: Timeout (>1200.0s)`. The gate had run on a loaded box (a second session compiling, load
~5.4, 10/12 GB used). Both pass in **21 seconds** together on an idle box. No code change.

### 4. The nbody skip

`tests/test_corpus_emit.py::test_nbody_nested_where_emits_and_computes` skipped on a stock-DaCe frontend
gap (an `IndexError` out of `to_sdfg`). CI runs the unit set under `NESTFORGE_CI_NO_SKIP=1`, so a skip
fails the session -- and it reads as "nothing to see here" for what is a known upstream gap.

Now `pytest.xfail`, raised imperatively inside the `except` around the BUILD rather than as a decorator on
the test. A decorator would mark the whole test expected-to-fail, so an emitter regression further down
would land in the same green bucket -- which is exactly what
`test_nbody_xfail_covers_the_dace_build_only_not_an_emitter_indexerror` exists to prevent. Raised where it
is, it fires only for the build gap, and the day DaCe can build nbody the test simply runs and validates.

## A note on counting

`1630c61`'s commit body says "2 known-open failures". That count was wrong: it was taken from a
still-running gate by reading pytest's per-test progress characters, and three more failures appeared
after it was read. **Do not report a count from a gate that has not printed its FAILURES section.**
