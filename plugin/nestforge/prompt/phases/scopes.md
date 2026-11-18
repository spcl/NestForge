## Phase: scopes
Which parts of the program become kernels, and where they run. `set_scope` makes one kernel of doalls or blocks named
by the latest tree; `unset_scope` inlines one back. `list_scope_moves` and `apply_scope_move` choose a kernel's
schedule (`set-schedule-cpu`, `set-schedule-gpu`) and its persistent runs (`persistent-kernel`). `place` runs each
kernel on the CPU or the GPU; `insert_copies` moves a GPU kernel's host/device copies into the program. Ask for
`get_performance` only once the scopes are placed and the copies inserted: before that it times a program that is
not the one you are building. `advance_phase` places what you did not; `undo_phase` takes this phase back.
