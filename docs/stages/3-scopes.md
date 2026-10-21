# Stage 3: Scopes

prev: [2 Moves](2-moves.md) · next: [4 Placement](4-placement.md)

A scope is the atomic unit later stages optimize. Each scope is outlined into a standalone SDFG and replaced by an
`ExternalCall` library node (`extcall_N`) that carries the kernel's Python oracle and boundary, and a `parallel`
flag. `unset_scope(name, epoch)` inlines a kernel back (its standalone SDFG, kernel-local containers transient again),
dropping any placement first.

The default makes one scope per parallel top-level map, time loops included; a purely sequential nest makes none.
`define_scope(labels, epoch)` makes one scope of several top-level maps of one state or of a straight run of
blocks of one control-flow region, at any depth (a loop and the state after it inside a time loop): the way to fuse
what no stage 2 move can, by writing the fused kernel in stage 5. The labels are one list, as the tree prints them.
It is refused, and nothing changes, when the blocks sit at different levels, when another block or a conditional edge
breaks the run, or when something outside the group runs between its parts.

Scalar inputs cross the boundary by value; a host length-1 array input is refused.

A scope inside a sequential loop takes the loop's iterator as an argument (`_loop_it_N`). The loop defines it and
nothing declares it, so `define_scopes` declares it on the program, and stages 5 and 6 check the kernel at the middle
of the loop's range unless `sizes` gives a value.

| | |
|---|---|
| default | `define_scopes()` |
| agent | `define_scope(labels, epoch)`, `unset_scope(name, epoch)` |
| code | `nestforge/stages/scopes.py`, `nestforge/ir/extract.py`, `nestforge/ir/loops.py`, DaCe's `ExternalCall` (`dace/libraries/standard/nodes/external_call.py`) |
