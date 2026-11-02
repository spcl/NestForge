`set_scope` makes one kernel of doalls or blocks named by the latest tree; `unset_scope` inlines one back. Moves are
closed once the program has a kernel, so settle them first. What no scope covers gets its default scope at `next_phase`,
and from there the kernels are fixed.
