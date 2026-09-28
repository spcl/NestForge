# Stage 4: Placement

prev: [3 Scopes](3-scopes.md) · next: [5 Kernels](5-kernels.md)

Stage 4 picks each kernel's device. Without a GPU target every kernel runs on the CPU. With one, the default puts
parallel kernels on the GPU and sequential ones on the CPU. DaCe's `OffloadToAccelerator` schedules the GPU
kernels and inserts the host/device copies; kernels left on the CPU keep their operands in host memory.

`place(devices, epoch)` takes `{kernel: "cpu" | "gpu"}` and returns each kernel's device, the copies, and the
transfers the [kernel DAG](../depends.md) implies: every argument whose producer and consumer sit on different
devices. Calling it again starts from the program before the first placement.

| | |
|---|---|
| default | `place()` |
| agent | `place(devices, epoch)` |
| code | `nestforge/stages/placement.py` |
