# Phase 2: Define Scopes

prev: [1 Shape Kernels](1-shape-kernels.md) · next: [3 Offload](3-offload.md)

Phase 2 decides which scopes leave the program as external kernels. Each scope is outlined into a
standalone SDFG and replaced by an `ExternalCall` library node that carries the kernel's NumPy
reference, manifest and boundary. Until phase 4 binds a compiled library, the node runs through
DaCe's reference expansion.

The default makes one kernel per parallel top-level map, wherever the map sits in the control flow,
time loops included. A purely sequential nest yields no kernel.

An agent may instead choose a scope that no move could create. `define_scope(labels, epoch)` makes one
kernel of several top-level maps of one state, or of a straight run of top-level blocks. This is the way
out when the scheduling moves cannot fuse two regions that should run as one: the kernel written in
phase 4 fuses them. The call is refused, and nothing changes, when something outside the group runs
between its parts or the blocks do not follow each other; `define_scopes` then lowers the maps left.
A single region is a valid group, including a purely sequential loop nest the default skips. A refused
phase 1 move names the call to make in `MoveResult.fallback`; the offer and `define_scope` decide by the same
`plan_scope` check, so an offered call is accepted.

Scalar inputs cross the boundary by value. Lowering refuses a host length-1 array input; only a
length-1 GPU array, a device pointer, may stand in for a scalar.

| | |
|---|---|
| default | `lower_nests_to_external_call(sdfg)` |
| agent | `define_scope(labels, epoch)` over maps or blocks, returning a `MoveResult` whose `reason` is the kernel id |
| preview | `offload_candidates(sdfg)` lists the parallel top-level maps without mutating |
| code | `nestforge/phases/scopes.py`, `nestforge/ir/extract.py`, `nestforge/ir/libnode.py` |

The [placement analysis](feedback.md) can send the program back here when kernels need different
boundaries.
