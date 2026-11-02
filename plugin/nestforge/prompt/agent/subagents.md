## Kernel subagents
In the kernels phase, hand each kernel to a `kernel-optimizer` subagent, one at a time, largest share in
`get_performance` first. Its prompt starts empty, so give it the kernel's name, its share and what you know of the
kernel. Compare `get_performance` after each one, and call `next_phase` yourself: a subagent stays in its phase.
