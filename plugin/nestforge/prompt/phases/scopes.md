## Phase: scopes
Which parts of the program become kernels, and where they run. `set_scope` makes one kernel of doalls or blocks named
by the latest tree; `unset_scope` inlines one back. `list_scope_moves` and `apply_scope_move` choose a kernel's
schedule (`set-schedule-cpu`, `set-schedule-gpu`) and its persistent runs (`persistent-kernel`), and where kernels
run: `run-on-cpu` and `run-on-gpu` with the kernels as labels, `run-on-cpu-persistent` and `run-on-gpu-persistent`
to run them (consecutive ones merged into one) as one persistent kernel; `rollback` takes one back. `finalize_scope`
places what you did not and moves the GPU kernels' host/device copies into the program. Ask for `get_performance`
only after `finalize_scope`: before, it times a program that is not the one you are building. `advance_phase` places
what you did not; `undo_phase` takes this phase back.
