# Known failures (as of 1630c61)

The review fix-wave was committed deliberately red (5 unit failures + 1 skip). `1630c61`'s commit body
says "2 known-open failures" -- **that count is wrong**: it was taken from a still-running gate by reading
pytest's per-test progress characters, and three more failures appeared after it was read. The full gate
result is below. Do not trust a count taken from a gate that has not printed its FAILURES section.

## 1. REGRESSION introduced by the fix-wave -- `sample_sizes` raises on a leaked outer index

    ValueError: tsvc2 kernel 's1115': boundary symbol 'i' is read by the nest but is not a registered
    parameter, a corpus symbol (['S']) or a shape symbol, so nothing here knows its value.

* `tests/test_tsvc_arena.py::test_crosslang_2d_fortran_multiline_signature`
* `tests/test_tsvc_full.py::test_dace_baseline_validates_for_2d_inner_nest`

The fix-wave replaced `sample_sizes`' blanket `else: sizes[sym] = 0` with proof / allowance / else-raise.
Under the `skip-taskloops` strategy the splitter peels s1115 to an INNER nest and leaks the outer index
`i`, which is in `arglist()` but is NOT nest-assigned -- so it matches neither the proof (`not in
arglist`) nor the allowance (`in nest_defined`) and hits the raise. The old blanket else zeroed it: the
"leaked loop-carried index" case that motivated the branch. (Under the `outer` strategy s1115's nest is
`free=['LEN_2D']` with no `i`, which is why the narrower tests pass.)

The allowance is too NARROW: a leaked index the nest CONSUMES rather than assigns is the original shape.
Fix by widening the allowance to that case (documented as the same arbitrary-but-shared start), or -- the
proper fix -- resolve the real incoming value from the parent SDFG, which removes the allowance entirely.
Do not simply restore the blanket else: the vacuous-validation hole it hid is real (a genuinely
unclassified symbol silently sized 0 makes oracle and candidate agree on a degenerate result).

## 2. A regression test added by the fix-wave is itself broken

* `tests/test_wcr_emit.py::test_tasklet_wcr_symbolic_index_target_is_normalized`
  `TypeError: pairsum() got an unexpected keyword argument 'M'` -- the test calls its own `@dace.program`
  with a symbol it never declares. The `emit_numpy` normalize_casts fix it guards is probably fine.

## 3. Probably the box, not the code

* `tests/test_tsvc_full.py::test_run_kernel_all_lanes_s000`
* `tests/test_tsvc_full.py::test_omp_emit_lane_runs_for_parallel_nest_s000`
  Both `Failed: Timeout (>1200.0s)`. The gate ran on a loaded box (a second session compiling, load ~5.4,
  10/12 GB used) and these compile many cells each. Re-run on a quiet box before treating as real.

## 4. Skip -- a CI failure in its own right

* `tests/test_corpus_emit.py::test_nbody_nested_where_emits_and_computes` SKIPS.
  CI runs the unit set under `NESTFORGE_CI_NO_SKIP=1`, so any skip fails the session. A skip on a healthy
  box hides a gap: it needs an assert (if the premise cannot fail) or a strict xfail (if it is a real
  upstream gap), not a skip.
