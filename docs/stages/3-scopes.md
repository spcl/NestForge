# Stage 3: Scopes

prev: [2 Moves](2-moves.md) · next: [4 Placement](4-placement.md)

A scope is the atomic unit later stages optimize. Each scope is outlined into a standalone SDFG and replaced by an
`ExternalCall` library node (`extcall_N`) that carries the kernel's Python oracle and boundary, and a `parallel`
flag. `unset_scope(name, epoch)` inlines a kernel back (its standalone SDFG, kernel-local containers transient again),
dropping any placement first.

The default (`default_units` in `nestforge/stages/scopes.py`) walks the program's top-level blocks:

| top-level block | scopes |
|---|---|
| state | each parallel map; sequential work stays in the program |
| loop holding no parallel map | the loop, one sequential kernel (placement keeps it on the CPU) |
| loop whose body holds only parallel maps and loops | each map; each loop in it opened the same way one level deeper (its maps), or the loop when it cannot open |
| loop whose body mixes in other work | the loop |
| conditional | each branch like a loop body; the conditional when a branch cannot open |

Before that, a parallel map holding more than `Targets.max_statements` statements (tasklets, default 200) is split by
`subgraph-fission`, at most twice, at the cut between its body blocks that leaves each piece a quarter of the
statements and passes the fewest containers from one piece to the other. Loops and conditionals are not split.
`set_scope(labels, epoch)` makes one scope of several top-level maps of one state or of a straight run of
blocks of one control-flow region, at any depth (a loop and the state after it inside a time loop): the way to fuse
what no stage 2 move can, by writing the fused kernel in stage 5. The labels are one list, as the tree prints them.
It is refused, and nothing changes, when the blocks sit at different levels, when another block or a conditional edge
breaks the run, or when something outside the group runs between its parts.

Scalar inputs cross the boundary by value; a host length-1 array input is refused.

A scope inside a sequential loop takes the loop's iterator as an argument (`_loop_it_N`). The loop defines it and
nothing declares it, so `default_scopes` declares it on the program, and stages 5 and 6 check the kernel at the middle
of the loop's range unless `sizes` gives a value.

| | |
|---|---|
| default | `default_scopes()` |
| agent | `set_scope(labels, epoch)`, `unset_scope(name, epoch)` |
| code | `nestforge/stages/scopes.py`, `nestforge/ir/extract.py`, `nestforge/ir/loops.py`, DaCe's `ExternalCall` (`dace/libraries/standard/nodes/external_call.py`) |
