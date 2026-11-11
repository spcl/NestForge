`set_scope` makes one kernel of doalls or blocks named by the latest tree; `unset_scope` inlines one back. Moves are
closed once the program has a kernel, so settle them first. What no scope covers gets its default scope at `next_phase`,
and from there the kernels are fixed.
`list_scope_moves` and `apply_scope_move` run the moves inside one kernel's body. `place devices={...}` runs each
kernel on the CPU or the GPU and fixes the scopes; `insert_copies` moves a GPU kernel's host/device copies out of its
call into the program, and `next_phase` inserts them when you did not.
