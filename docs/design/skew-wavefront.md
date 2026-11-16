# The `skew-wavefront` move, and a canonical form without wavefronts

[Designs](README.md) · [Structure tree](../ir.md) · [Stages](../stages.md)

Status: design, not implemented. Written 2026-11-16 against NestForge `ac7033c`, DaCe extended `561defb2a6`, and
HPCAgent-Bench `aca38d362`.

## Goal

The agent decides where a loop nest becomes a wavefront, and canonicalization no longer does it on its own. Two
results follow.

1. `skew-wavefront` is a move. The agent names a sequential loop, and the move skews the nest below it so that each
   diagonal runs as a parallel doall, or, where the dependences allow, tiles the nest along the diagonal.
2. The canonical form has a variant without wavefronts. The NestForge agent and the LLVM-pass agent
   ([B13](bench-requests.md)) start from that same program. The comparison is then between two catalogues of
   transformations on one starting point, and canonicalization does neither agent's work.

User decision, 2026-11-16: both agents start from the canonical form without wavefronts, and `skew-wavefront` is
designed now.

## What exists today

| Where | Today |
|---|---|
| DaCe `passes/parallelization/parallelize.py`, `Parallelize.__init__` | runs `ParallelizeLoops`, then `WavefrontSkew(target=target)` on the loops that `ParallelizeLoops` refused for a carried dependence, or held back as a sibling chain (`hold_sibling_chains`) |
| DaCe `passes/canonicalize/pipeline.py` (line 995) | the canonicalization runs `Parallelize(target=target, hold_sibling_chains=False)`, so every canonical form holds the wavefronts that `WavefrontSkew` finds |
| DaCe `passes/parallelization/wavefront/skew.py`, `WavefrontSkew` | `plan(outer, sdfg, occurrences)` returns a `Plan` or `None`, and `commit(plan, sdfg)` rewrites. A plan either skews (`t = a*u + b*(v + s)`, one doall per diagonal) or tiles (a sequential tile diagonal over a doall of tile columns), with tile sizes `tile_i` and `tile_j` (64 on CPU and on GPU). It returns `None` at about ten points, with no reason |
| DaCe `canonicalize(...)` | takes `unroll_limit`, `peel_limit`, and `target`; there is no switch for the wavefront |
| NestForge `stages/canonicalize.py`, `canonicalize` | calls DaCe's canonicalization in two parts (before the fusion stage, then from it) |
| NestForge `stages/expand.py` | canonicalizes the SDFG of an expanded library node with the same function |
| NestForge `stages/moves.py` | has `tile-for`, `tile-doall`, `interchange-*`, `hoist-invariant-*`, and others. No move skews |

The canonical form does not tile, except inside a wavefront: the wavefront's tiled form is the only tiling that
canonicalization makes. Turning the wavefront off therefore removes both the skew and that tiling, which is the
"no wavefront, no tiling" starting point the user asked for.

## Design

### 1. DaCe: a canonicalization without wavefronts (request D3)

A change on DaCe extended, with one regression test, as for D1 and D2 in [comm-nodes.md](comm-nodes.md):

- `Parallelize.__init__` gains `wavefront: bool = True`. When it is false, the pipeline is `ParallelizeLoops`
  without `WavefrontSkew`, and `hold_sibling_chains` is forced false, since nothing would rescue a chain held back.
- `canonicalize(...)` and `canonicalize_under_authority(...)` gain `wavefront: bool = True` and pass it to the
  `Parallelize` of line 995.
- The default stays `True`, so DaCe's behavior, CPF's output, and the bench's `dace_cpu_canon` column do not change.
- Regression test: the nest of `tests/canonicalize/canonicalize_wavefront_stays_sequential_test.py` (row `i` reads row `i-1`, as in tsvc_2_5 `wf_diff_skew`),
  canonicalized with `wavefront=False`, has no diagonal loop, and its outer loop stays a sequential loop, with the same
  output as with `wavefront=True`.

A second request makes the move's refusals readable:

- `WavefrontSkew.plan` gains a variant `plan_or_reason(outer, sdfg, occurrences) -> Plan | str` that returns, at each
  of its `return None` points, one sentence saying why (for example "the inner axes are already parallel at fixed
  u", "a time-step loop: its iterator indexes nothing", "a dependence lands on an earlier diagonal"). `plan` stays
  as it is and calls the variant.

### 2. NestForge: the starting point

- `Targets` gains `wavefront: bool`, false by default for a session. `stages/canonicalize.canonicalize` and
  `stages/expand.py` pass it to DaCe.
- `run_defaults(stage="moves")` applies `skew-wavefront` to every loop that the move lists, innermost first, the
  order `WavefrontSkew.skew_all` uses. After the defaults, a program is the one canonicalization makes today. The
  `nestforge_default` column ([default-flavor.md](default-flavor.md)) therefore keeps today's wavefronts.
- `nestforge_reference` replays the reference solutions on the new starting point. A reference that relied on the
  canonical wavefront gains one `skew-wavefront` line. The replay test finds these: it fails on a reference whose
  result changes, and the fix is the added line, never a weaker test.

### 3. The move

| | |
|---|---|
| Name | `skew-wavefront` |
| Labels | one `for<n>` row: the outer loop `u` of the nest |
| Arguments | `tile_sizes: list[int] \| None`, the two tile extents along `u` and along the diagonal. `None` takes `WavefrontSkew`'s defaults. Ignored, with a note, when the plan skews without tiling |
| Pass | `WavefrontSkew(target=session target)`, with `tile_i` and `tile_j` set from `tile_sizes`, then `plan_or_reason(loop, sdfg, occurrence_index(sdfg))` and `commit(plan, sdfg)` |
| Legal | when `plan_or_reason` returns a plan. Legality is DaCe's: every exact dependence lands on a later diagonal, or on the same diagonal with its source first |
| Refused | with DaCe's sentence as the reason, under the refusal class `ANALYSIS` of [run-extractions.md](run-extractions.md) |
| Rows after | the loop `u` and the loops of its chain are retired. The new sequential diagonal loop is a fresh `for<n>`, and each statement at a fixed diagonal is a fresh `doall<n>`, or the tile loops when it tiles (`Session.bump`) |
| Listed | `list_moves` lists the move on every `for` row whose plan exists. The plan runs isl, so the listing is computed once per program version and cached, like every listing that analyzes |
| Family | a new family `wavefront` in [move-families.md](move-families.md), so the all-or-none switch covers it |

The tree shows the result as it shows any loop and doall today. One field is added on the diagonal loop:
`wavefront=(skew a=<a> b=<b>)` or `wavefront=(tiles <ti>x<tj>)`, so the agent sees what the move made.

The text for the agent, in the moves prompt: "`skew-wavefront` on a sequential loop whose iterations depend on each
other along a diagonal (Gauss-Seidel, LU, dynamic programming) runs each diagonal in parallel. It applies only where
DaCe proves the diagonals independent."

### 4. The columns without an LLM

The user asked for the gain of the canonical form to be measured on its own, without an agent. The bench and
NestForge already have the first two columns, and the third is a variant.

| Column | Program | Exists |
|---|---|---|
| reference | the task's reference C++ at the bench's flags | yes (the bench's baseline) |
| `dace_cpu_canon` | the canonical form with wavefronts (today's CPF) | yes |
| `nestforge_cpf_nowave` | the canonical form without wavefronts, with NestForge's default scopes, placement, and code generation, but no moves | **new**: a flavor in `plugin/nestforge/plugin.toml` with `wavefront` off and the moves stage skipped |

`nestforge_default` minus `nestforge_cpf_nowave` is then the gain of the default moves, the wavefront included, and
`dace_cpu_canon` minus the reference is the gain of the canonical form.

## Work packages

| # | Package | Where | Depends on |
|---|---|---|---|
| 1 | D3: `wavefront` switch in `Parallelize` and `canonicalize`, and `plan_or_reason` | DaCe extended, `parallelize.py`, `pipeline.py`, `wavefront/skew.py`, one test | none |
| 2 | `Targets.wavefront`, the two canonicalization calls | `nestforge/stages/canonicalize.py`, `nestforge/stages/expand.py` | 1 |
| 3 | the move, its listing, its tree field, and its family | `nestforge/stages/moves.py`, `nestforge/ir/introspect.py`, `docs/ir.md`, `docs/stages.md` | 1, 2 |
| 4 | the default moves apply it, and the reference solutions are replayed | `nestforge/session.py`, `reference_solutions/` | 3 |
| 5 | the flavor `nestforge_cpf_nowave` | `plugin/nestforge/plugin.toml`, `nestforge_bench/` | 2 |

## Tests

1. With `wavefront=False`, the DaCe fixture of package 1 canonicalizes with a sequential outer loop and no diagonal loop. With the
   default, the diagonal loop is there (DaCe test, package 1).
2. `skew-wavefront` on that fixture's outer loop applies, and the tree shows a fresh `for` row with the `wavefront=`
   field above fresh doall rows. The output equals the reference (marker `numeric`).
3. `skew-wavefront` with `tile_sizes=[32, 32]` on a nest that tiles gives tile loops of those extents.
4. On a loop whose inner axes are already parallel, the move is refused with DaCe's sentence, and the program's hash
   does not change.
5. On a time-step loop, it is refused with the time-step sentence.
6. `run_defaults(stage="moves")` on every program of the replay corpus gives the same program as today's
   canonicalization with wavefronts. This is the test that keeps `nestforge_default` unchanged.
7. Every reference solution replays to a correct program on the new starting point.
8. `rollback` after `skew-wavefront` restores the program and its labels.

## Done means

1. The eight tests pass in CI, and DaCe extended carries D3 with its test.
2. `nestforge_default` gives the same numbers as before on the replay corpus.
3. The study's two agents start from the program without wavefronts, and the paper states it.

## Decisions

Taken by the owner on 2026-11-16:

1. Both the NestForge agent and the LLVM-pass agent start from the canonical form without wavefronts.
2. `skew-wavefront` is a move.
3. CPF's own gain is measured by columns without an LLM.

Open, with a recommendation:

1. Whether the move also takes the skew factors `a` and `b`. Recommended: no. `WavefrontSkew` derives them from the
   dependences, and a factor the agent picks would need its own legality proof.
2. Whether the LLVM-pass agent's program is rendered with OpenMP pragmas on its doalls (the parallel form) or as
   sequential C++. Recommended: with them, since that is what the canonical form says, and Polly then has less to
   find.
