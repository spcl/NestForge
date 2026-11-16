This phase decides which code becomes a kernel and where each kernel runs. Moves close at the first kernel, so settle
them first. `set_scope` makes one kernel of top-level doalls of one state (library calls among them) or of a straight
run of blocks, named by the latest tree; `unset_scope` inlines one back. `list_scope_moves` and `apply_scope_move` run
the moves inside one kernel's body.
With a GPU target, `place devices={...}` runs each kernel on `cpu` or `gpu`; kernels you leave out keep their default
device, which `list_kernels` shows, and each call starts again from the program before the first placement. Once
placed, `set_scope` and `unset_scope` are refused until `unplace`, which drops the placement, its copies and any
device move, and keeps the kernels and the moves inside them. A GPU kernel copies its arguments inside its call until
`insert_copies` moves the copies into the program; `next_phase` inserts them otherwise. A structured move inside a
placed kernel places it again on the same devices, copies where they were. `stage-loads-shared`, a device move on a
GPU kernel, rewrites the body as placed, so every later `place`, `insert_copies`, `unplace` or structured move inside
the kernel drops it. Without a GPU target every kernel runs on the CPU, and `place`, `insert_copies` and `unplace`
change nothing and say so.
`get_performance` reports, under `offloading`, the copies the timed program makes (`copies`, `copy_bytes`) and the GPU
kernels still copying inside their call (`copies_in_call`). At `next_phase`, a program you did not place gets the
default scope for what no scope covers and the default device for every kernel; a program you placed keeps your
devices, and code no scope covers stays outside the kernels. Either way the copies a GPU kernel still makes inside its
call are inserted. The kernels are then fixed until a new round, and `unplace` is refused once `focus_kernel` has run.
