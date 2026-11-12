The kernels are fixed: moves and scopes stay closed until the next round. Work one kernel at a time, largest share in
`get_performance` first: `focus_kernel` on it, `read_kernel`, then `write_kernel` with the whole unit; `write_kernel`
and `tune_kernel` take only the kernel in focus. A source is linked only when it matches
the kernel's oracle, and the answer gives the new times; status `unchecked` means it has no oracle, so it is linked and
only the whole program's check tells. Stay on a kernel until it stops improving, then focus the next.
`list_kernels` names each kernel's arguments and what reaches them.
