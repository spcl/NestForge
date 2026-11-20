# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Programs that communicate between MPI ranks, the fixtures of docs/design/comm-nodes.md: a halo exchange with
nonblocking calls, a residual reduced across ranks every time step, a ring of blocking calls in a loop, and a loop of
collectives, one per row. Parsed and canonicalized only; no test here launches ranks."""

import dace
from mpi4py import MPI

N = dace.symbol("N", dtype=dace.int64)
T = dace.symbol("T", dtype=dace.int64)
K = dace.symbol("K", dtype=dace.int64)


@dace.program
def halo_exchange(a: dace.float64[N], ghost: dace.float64[2], left: dace.int32, right: dace.int32) -> None:
    """One step of a 1-D halo exchange: each end goes to a neighbor, each ghost cell comes from one."""
    req = dace.define_local([4], dtype=MPI.Request)
    MPI.COMM_WORLD.Isend(a[0:1], left, 0, req[0])
    MPI.COMM_WORLD.Isend(a[N - 1 : N], right, 1, req[1])
    MPI.COMM_WORLD.Irecv(ghost[0:1], left, 1, req[2])
    MPI.COMM_WORLD.Irecv(ghost[1:2], right, 0, req[3])
    MPI.Request.Waitall(req)


@dace.program
def residual_in_time_loop(a: dace.float64[N], r: dace.float64[1]) -> None:
    """A relaxation whose residual every rank reduces each step."""
    for t in range(T):
        for i in dace.map[0:N]:
            a[i] = a[i] * 0.5
        r[0] = 0.0
        for i in dace.map[0:N]:
            r[0] += a[i]
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, r, op=MPI.SUM)


@dace.program
def ring(a: dace.float64[K, 4], b: dace.float64[K, 4], left: dace.int32, right: dace.int32) -> None:
    """Each row goes to the right neighbor while the left neighbor's arrives, row by row."""
    for i in range(K):
        MPI.COMM_WORLD.Send(a[i], right, 0)
        MPI.COMM_WORLD.Recv(b[i], left, 0)


@dace.program
def per_row_allreduce(x: dace.float64[K, 4]) -> None:
    """One collective per row, in row order."""
    for i in range(K):
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, x[i], op=MPI.SUM)


#: Every fixture, by name.
PROGRAMS = {
    "halo_exchange": halo_exchange,
    "residual_in_time_loop": residual_in_time_loop,
    "ring": ring,
    "per_row_allreduce": per_row_allreduce,
}
