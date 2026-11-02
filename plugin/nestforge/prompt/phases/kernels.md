The kernels are fixed: moves and scopes stay closed until the next round. Work one kernel at a time, largest share in
`get_performance` first: `read_kernel`, then `write_kernel` with the whole unit. A source is linked only when it matches
the kernel's oracle, and the answer gives the new times. Stay on a kernel until it stops improving, then take the next.
`list_kernels` names each kernel's arguments and what reaches them.
