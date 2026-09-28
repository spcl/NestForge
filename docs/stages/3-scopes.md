# Stage 3: Scopes

prev: [2 Moves](2-moves.md) · next: [4 Placement](4-placement.md)

A scope is the atomic unit later stages optimize. Each scope is outlined into a standalone SDFG and replaced by an
`ExternalCall` library node (`extcall_N`) that carries the kernel's Python oracle, manifest and boundary, and a
`parallel` flag.

The default makes one scope per parallel top-level map, time loops included; a purely sequential nest makes none.
`define_scope(labels, epoch)` makes one scope of several top-level maps of one state or of a straight run of
top-level blocks: the way to fuse what no stage 2 move can, by writing the fused kernel in stage 5. It is refused,
and nothing changes, when something outside the group runs between its parts.

Scalar inputs cross the boundary by value; a host length-1 array input is refused.

| | |
|---|---|
| default | `define_scopes()` |
| agent | `define_scope(labels, epoch)` |
| code | `nestforge/stages/scopes.py`, `nestforge/ir/extract.py`, `nestforge/ir/libnode.py` |
