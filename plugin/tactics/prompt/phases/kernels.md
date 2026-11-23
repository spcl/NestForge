## Phase: kernels
Each kernel's code and build. Work one kernel at a time, largest share in `get_performance` first: `focus_kernel` on
it, `read_kernel`, then `write_kernel` with the whole unit. A source is linked only when it matches the kernel's oracle,
and the answer gives the new times; `unchecked` means it has no oracle, so only the whole program's check tells.
`tune_kernel` sweeps compilers and flags for the kernel in focus and links the fastest correct build. `list_kernels`
names each kernel's arguments. When done, `finalize` and `score`; `return_to_phase` returns to an earlier phase.
