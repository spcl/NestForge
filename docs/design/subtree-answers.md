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

## Not done

- No line diff (`+`/`-`): a subtree is easier for the model to apply than a patch.
- `view_tree(root=...)` does not update `Session.shown`; the next move then compares against the older whole tree,
  which only makes its answer larger, never wrong.
