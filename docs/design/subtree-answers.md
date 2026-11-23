# Subtree answers: a move answers with what it rewrote

Decided 2026-11-22 (owner). Code: `nestforge/ir/subtree.py`, `Session.changed_tree`, `Session.view(root=...)`,
`server.with_tree`, `answers.move_answer`.

## Problem

The agent sees the program as the tree `get_cpf` and `view_tree` print. Before this change it read the whole tree
again after every move, although most moves rewrite one loop or one state. On a program of a few hundred rows that
is most of the tokens of a run.

## What the agent gets

- `apply_move` and `apply_moves` answer, once applied, with a `tree` field: the **rewritten subtree** and the rows
  around it (one line each, outermost first), headed `Rows outside <root> are unchanged; <root> now reads, under the
  rows around it:`. The agent replaces that row's lines, and the lines of the rows around it, in the tree it holds.
- When the change reaches the top level, or the agent never read a whole tree, the field is `The whole tree:` and the
  tree. When nothing printed changed, it is `The tree is unchanged.`
- `view_tree(root=<label>)` prints one row's subtree. The whole tree stays one call away.

## How the root is found

The session keeps the tree the agent last read whole (`Session.shown`: set by `get_cpf`, `view_tree` without
`root`, and every move answer). After a move it prints the new tree and compares the two line by line:

1. A row's subtree is its line and every line after it indented deeper (bodies and notes included).
2. The root is the deepest row present in both trees, with the same rows around it, whose subtree holds every
   changed line of the old and of the new tree, besides the lines of the rows around it (a loop's `num_states` or a
   state's `reads=` may change with what is inside).
3. No such row: the whole tree.

The comparison is on the printed text only, so it holds for every move kind, rollback included, without a rule per
kind, and the agent's copy is never stale: whatever changed lies inside what the answer carries.

## What stays put between answers (owner 2026-11-23)

A subtree answer only saves tokens when the agent's copy of the rest stays valid, so three things persist:

1. **Labels only grow.** A row keeps its label while it lives. A label is never handed out again once its row is
   gone, whether a move consumed the row, a `rollback` took away the move that made it, or `rewind` went back to
   before the kernels (`Session.undo`, `Session.namer_back`). A crashed move or a refused batch puts the namer back
   as it was: the agent never read what it named, and a session rebuilt from the journal (which leaves refusals
   out) must name the next rows as the live one does.
2. **The view follows the rows.** What the agent expanded, or set a language for, stays with the row while it lives.
   The rows a move makes in place of an open row open with it (`Session.inherit_view`), so tiling an expanded doall
   shows the tiled rows open. A `rollback` keeps what the agent opened since, on the rows that are back.
3. **The held tree** (`Session.shown`) is the last tree the agent read whole or was brought up to by an answer,
   and a move answers in the print it was read in (bodies, dependencies), so the answer stays a subtree. Costs
   are never recomputed for an answer (up to a minute a tree): after a read with costs, the next answer is whole.
4. **Old journals.** A session rebuilt from its own journal repeats the rollbacks, so it names the same labels. A
   journal of plugin-record version 1, written before labels only grew, replays with
   `Session.reissue_dropped_labels` ([plugin-record.md](../plugin-record.md)).

**A move that may invalidate a region returns that region whole.** The answer is never a patch: it is the full
text of the smallest row whose subtree holds every printed line the move changed, in the old and the new tree.
Any changed line outside it widens the region, up to the whole tree, so nothing the agent holds goes stale.

Tests: `tests/test_subtree_answers.py` (every listed move, its rollback and pairs of moves as one batch on time-loop
programs: the answer brings the held tree up to date, and is a subtree whenever the change stays inside one
top-level row; labels a rollback drops or a chain of moves removes never return; expansion survives a move).

## Not done

- No line diff (`+`/`-`): a subtree is easier for the model to apply than a patch.
- `view_tree(root=...)` does not update `Session.shown`; the next move then compares against the older whole tree,
  which only makes its answer larger, never wrong.
