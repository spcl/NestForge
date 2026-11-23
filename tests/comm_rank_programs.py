# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Programs that communicate and compute, run on MPI ranks by ``tests/test_comm_ranks.py``: each has compute that a
communication row may cross and compute it may not, so the legal moves have somewhere to go. Every fixture gives its
inputs for a rank of a communicator of a size, and the outputs every rank compares."""

from collections.abc import Callable
from dataclasses import dataclass

import dace
import numpy as np
from dace.frontend.python.parser import DaceProgram
from mpi4py import MPI

N = dace.symbol("N", dtype=dace.int64)
T = dace.symbol("T", dtype=dace.int64)
K = dace.symbol("K", dtype=dace.int64)
P = dace.symbol("P", dtype=dace.int64)
KL = dace.symbol("KL", dtype=dace.int64)
EL = dace.symbol("EL", dtype=dace.int64)
W = dace.symbol("W", dtype=dace.int64)
Q = dace.symbol("Q", dtype=dace.int64)

#: The length of each rank's vector.
LENGTH = 16
#: Time steps of the looped fixtures.
STEPS = 3
#: Rows of the row-wise fixtures.
ROWS = 4
#: Momenta and energies each rank owns in the redistribution fixture, and its stencil width and summed axis.
LOCAL_MOMENTA, LOCAL_ENERGIES, WIDTH, SUMMED = 2, 3, 2, 3


@dace.program
def halo_overlap(
    a: dace.float64[N],
    b: dace.float64[N],
    c: dace.float64[N],
    ghost: dace.float64[2],
    left: dace.int32,
    right: dace.int32,
) -> None:
    """A halo exchange with compute between its starts and its wait that touches nothing in flight (``b``), and
    compute after the wait that reads what arrived (``c``'s ends) or not (``c``'s interior)."""
    req = dace.define_local([4], dtype=MPI.Request)
    MPI.COMM_WORLD.Isend(a[0:1], left, 0, req[0])
    MPI.COMM_WORLD.Isend(a[N - 1 : N], right, 1, req[1])
    MPI.COMM_WORLD.Irecv(ghost[0:1], left, 1, req[2])
    MPI.COMM_WORLD.Irecv(ghost[1:2], right, 0, req[3])
    for i in dace.map[0:N]:
        b[i] = a[i] * 2.0
    MPI.Request.Waitall(req)
    for i in dace.map[1 : N - 1]:
        c[i] = b[i] + 1.0
    c[0] = ghost[0] + b[0]
    c[N - 1] = ghost[1] + b[N - 1]


@dace.program
def halo_in_place(ap: dace.float64[N + 2], b: dace.float64[N + 2], left: dace.int32, right: dace.int32) -> None:
    """A halo exchange into the ghost cells of the array it sends from (``ap[0]``, ``ap[N + 1]``): the interior update
    reads ``ap[1:N+1]``, which nothing received touches, so only a comparison by subset lets the wait cross it."""
    req = dace.define_local([4], dtype=MPI.Request)
    MPI.COMM_WORLD.Isend(ap[1:2], left, 0, req[0])
    MPI.COMM_WORLD.Isend(ap[N : N + 1], right, 1, req[1])
    MPI.COMM_WORLD.Irecv(ap[0:1], left, 1, req[2])
    MPI.COMM_WORLD.Irecv(ap[N + 1 : N + 2], right, 0, req[3])
    for i in dace.map[2:N]:
        b[i] = ap[i - 1] + ap[i] + ap[i + 1]
    MPI.Request.Waitall(req)
    b[1] = ap[0] + ap[1] + ap[2]
    b[N] = ap[N - 1] + ap[N] + ap[N + 1]


@dace.program
def halo_stencil(ap: dace.float64[N + 2], b: dace.float64[N + 2], left: dace.int32, right: dace.int32) -> None:
    """A halo exchange, then one stencil over ``ap[1:N+1]`` that reads the ghost cells at its ends: the overlap of
    package 16 splits the stencil (``split-doall``) and moves the wait after its interior part (``lib-after``)."""
    req = dace.define_local([4], dtype=MPI.Request)
    MPI.COMM_WORLD.Isend(ap[1:2], left, 0, req[0])
    MPI.COMM_WORLD.Isend(ap[N : N + 1], right, 1, req[1])
    MPI.COMM_WORLD.Irecv(ap[0:1], left, 1, req[2])
    MPI.COMM_WORLD.Irecv(ap[N + 1 : N + 2], right, 0, req[3])
    MPI.Request.Waitall(req)
    for i in dace.map[1 : N + 1]:
        b[i] = ap[i - 1] + ap[i] + ap[i + 1]


@dace.program
def halo_rows(ap: dace.float64[K + 2, N], b: dace.float64[K + 2, N], left: dace.int32, right: dace.int32) -> None:
    """The row exchange of the bench's ``jacobi_2d``: each rank sends its first and its last row and receives its ghost
    rows as ``(1, N)`` slices, then one stencil over rows ``1..K`` reads a ghost row at each end."""
    req = dace.define_local([4], dtype=MPI.Request)
    MPI.COMM_WORLD.Isend(ap[1:2, :], left, 0, req[0])
    MPI.COMM_WORLD.Irecv(ap[K + 1 : K + 2, :], right, 0, req[1])
    MPI.COMM_WORLD.Isend(ap[K : K + 1, :], right, 1, req[2])
    MPI.COMM_WORLD.Irecv(ap[0:1, :], left, 1, req[3])
    MPI.Request.Waitall(req)
    for i, j in dace.map[1 : K + 1, 1 : N - 1]:
        b[i, j] = ap[i - 1, j] + ap[i + 1, j] + ap[i, j - 1] + ap[i, j + 1]


@dace.program
def halo_waits(
    a: dace.float64[N],
    b: dace.float64[N],
    c: dace.float64[N],
    ghost: dace.float64[2],
    left: dace.int32,
    right: dace.int32,
) -> None:
    """A halo exchange with one ``Wait`` per request: each ghost cell is used as soon as its receive completes, and
    the sends complete last, after compute that only reads what they send."""
    req = dace.define_local([4], dtype=MPI.Request)
    MPI.COMM_WORLD.Isend(a[0:1], left, 0, req[0])
    MPI.COMM_WORLD.Isend(a[N - 1 : N], right, 1, req[1])
    MPI.COMM_WORLD.Irecv(ghost[0:1], left, 1, req[2])
    MPI.COMM_WORLD.Irecv(ghost[1:2], right, 0, req[3])
    for i in dace.map[0:N]:
        b[i] = a[i] * 2.0
    dace.comm.Wait(req[2])
    c[0] = ghost[0] + b[0]
    dace.comm.Wait(req[3])
    c[N - 1] = ghost[1] + b[N - 1]
    for i in dace.map[1 : N - 1]:
        c[i] = b[i] + a[i]
    dace.comm.Wait(req[0])
    dace.comm.Wait(req[1])


@dace.program
def two_exchanges(
    a: dace.float64[N],
    x: dace.float64[N],
    ghost_a: dace.float64[2],
    ghost_x: dace.float64[2],
    b: dace.float64[4],
    left: dace.int32,
    right: dace.int32,
) -> None:
    """Two exchanges in flight at once whose waits complete in the opposite order to their starts: ``x`` goes left
    and is used first, ``a`` goes right and is used second."""
    req = dace.define_local([4], dtype=MPI.Request)
    MPI.COMM_WORLD.Isend(a[0:2], right, 0, req[0])
    MPI.COMM_WORLD.Irecv(ghost_a, left, 0, req[1])
    MPI.COMM_WORLD.Isend(x[0:2], left, 1, req[2])
    MPI.COMM_WORLD.Irecv(ghost_x, right, 1, req[3])
    MPI.Request.Waitall(req[2:4])
    for i in dace.map[0:2]:
        b[i] = ghost_x[i] * 2.0
    MPI.Request.Waitall(req[0:2])
    for i in dace.map[0:2]:
        b[i + 2] = ghost_a[i] + a[i]


@dace.program
def halo_steps(
    ap: dace.float64[N + 2], b: dace.float64[N + 2], c: dace.float64[N], left: dace.int32, right: dace.int32
) -> None:
    """A halo exchange every time step: the ghost cells of ``ap`` arrive, a smoothing stencil reads them, and the
    result is copied back; ``c`` is scaled each step with nothing exchanged, so the starts have room to move."""
    req = dace.define_local([4], dtype=MPI.Request)
    for t in range(T):
        for i in dace.map[0:N]:
            c[i] = c[i] * 0.5
        MPI.COMM_WORLD.Isend(ap[1:2], left, 0, req[0])
        MPI.COMM_WORLD.Isend(ap[N : N + 1], right, 1, req[1])
        MPI.COMM_WORLD.Irecv(ap[0:1], left, 1, req[2])
        MPI.COMM_WORLD.Irecv(ap[N + 1 : N + 2], right, 0, req[3])
        MPI.Request.Waitall(req)
        for i in dace.map[1 : N + 1]:
            b[i] = (ap[i - 1] + ap[i] + ap[i + 1]) * 0.25
        for i in dace.map[1 : N + 1]:
            ap[i] = b[i]


@dace.program
def residual_loop(a: dace.float64[N], r: dace.float64[T], s: dace.float64[N]) -> None:
    """A relaxation whose residual every rank reduces each step, with an update independent of the residual. The
    reduction reads the step's slice ``r[t : t + 1]`` in place: canonicalization leaves no view in front of it."""
    for t in range(T):
        for i in dace.map[0:N]:
            a[i] = a[i] * 0.5
        r[t] = 0.0
        for i in dace.map[0:N]:
            r[t] += a[i]
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, r[t : t + 1], op=MPI.SUM)
        for i in dace.map[0:N]:
            s[i] = s[i] + a[i]


@dace.program
def ring(
    a: dace.float64[K, 4], b: dace.float64[K, 4], d: dace.float64[K, 4], left: dace.int32, right: dace.int32
) -> None:
    """Each row goes right while the left neighbor's arrives, with a row-wise update the receive may cross, then an
    update of what arrived. The calls read and write the rows ``a[i]`` and ``b[i]`` in place, with no view."""
    for i in range(K):
        MPI.COMM_WORLD.Send(a[i], right, 0)
        MPI.COMM_WORLD.Recv(b[i], left, 0)
        for j in dace.map[0:4]:
            d[i, j] = a[i, j] * 2.0
    for i, j in dace.map[0:K, 0:4]:
        d[i, j] = b[i, j] + d[i, j]


@dace.program
def redistribute(
    Gl: dace.float64[P * KL, EL],
    D: dace.float64[Q, W],
    Sigma: dace.float64[KL, P * EL],
) -> None:
    """OMEN's SSE exchange in small (``scattering_self_energies``, SC19 section 6): G arrives energy-distributed, a
    local transpose packs it per destination, one Alltoall gives each rank its momenta over every energy, the unpack
    transposes back, and the stencil over energies runs. ``D`` summed over ``Q`` needs nothing from the exchange."""
    send = dace.define_local([P, KL, EL], dace.float64)
    recv = dace.define_local([P, KL, EL], dace.float64)
    Gk = dace.define_local([KL, P * EL], dace.float64)
    Ds = dace.define_local([W], dace.float64)
    for d, kl, el in dace.map[0:P, 0:KL, 0:EL]:
        send[d, kl, el] = Gl[d * KL + kl, el]
    dace.comm.Alltoall(send, recv)
    for s, kl, el in dace.map[0:P, 0:KL, 0:EL]:
        Gk[kl, s * EL + el] = recv[s, kl, el]
    for w in dace.map[0:W]:
        Ds[w] = 0.0
    for q, w in dace.map[0:Q, 0:W]:
        Ds[w] += D[q, w]
    for k, e in dace.map[0:KL, 0 : P * EL]:
        Sigma[k, e] = 0.0
    for k, e, w in dace.map[0:KL, 0 : P * EL, 0:W]:
        if e - w >= 0:
            Sigma[k, e] += Gk[k, e - w] * Ds[w]


@dataclass(frozen=True, slots=True)
class Fixture:
    """A program, the inputs of one rank of ``size`` ranks (arguments and symbols), and what every rank compares."""

    program: DaceProgram
    inputs: Callable[[int, int], dict[str, object]]
    outputs: tuple[str, ...]


def neighbors(rank: int, size: int) -> dict[str, object]:
    return {"left": np.int32((rank - 1) % size), "right": np.int32((rank + 1) % size)}


def seeded(rank: int, *shape: int) -> np.ndarray:
    return np.random.default_rng(1000 + rank).random(shape)


def halo_inputs(rank: int, size: int) -> dict[str, object]:
    zeros = np.zeros(LENGTH)
    return {"a": seeded(rank, LENGTH), "b": zeros.copy(), "c": zeros.copy(), "ghost": np.zeros(2), "N": LENGTH} | (
        neighbors(rank, size)
    )


def halo_in_place_inputs(rank: int, size: int) -> dict[str, object]:
    return {"ap": seeded(rank, LENGTH + 2), "b": np.zeros(LENGTH + 2), "N": LENGTH} | neighbors(rank, size)


def halo_rows_inputs(rank: int, size: int) -> dict[str, object]:
    return {"ap": seeded(rank, ROWS + 2, LENGTH), "b": np.zeros((ROWS + 2, LENGTH)), "K": ROWS, "N": LENGTH} | (
        neighbors(rank, size)
    )


def two_exchanges_inputs(rank: int, size: int) -> dict[str, object]:
    return {
        "a": seeded(rank, LENGTH),
        "x": seeded(500 + rank, LENGTH),
        "ghost_a": np.zeros(2),
        "ghost_x": np.zeros(2),
        "b": np.zeros(4),
        "N": LENGTH,
    } | neighbors(rank, size)


def halo_steps_inputs(rank: int, size: int) -> dict[str, object]:
    return {"c": seeded(700 + rank, LENGTH), "T": STEPS} | halo_in_place_inputs(rank, size)


def residual_inputs(rank: int, size: int) -> dict[str, object]:
    return {"a": seeded(rank, LENGTH), "r": np.zeros(STEPS), "s": np.zeros(LENGTH), "N": LENGTH, "T": STEPS}


def ring_inputs(rank: int, size: int) -> dict[str, object]:
    return {
        "a": seeded(rank, ROWS, 4),
        "b": np.zeros((ROWS, 4)),
        "d": np.zeros((ROWS, 4)),
        "K": ROWS,
    } | neighbors(rank, size)


def redistribute_inputs(rank: int, size: int) -> dict[str, object]:
    return {
        "Gl": seeded(rank, size * LOCAL_MOMENTA, LOCAL_ENERGIES),
        "D": seeded(99, SUMMED, WIDTH),
        "Sigma": np.zeros((LOCAL_MOMENTA, size * LOCAL_ENERGIES)),
        "P": size,
        "KL": LOCAL_MOMENTA,
        "EL": LOCAL_ENERGIES,
        "W": WIDTH,
        "Q": SUMMED,
    }


#: Every fixture, by name.
FIXTURES = {
    "halo_overlap": Fixture(halo_overlap, halo_inputs, ("b", "c", "ghost")),
    "halo_in_place": Fixture(halo_in_place, halo_in_place_inputs, ("ap", "b")),
    "halo_stencil": Fixture(halo_stencil, halo_in_place_inputs, ("ap", "b")),
    "halo_rows": Fixture(halo_rows, halo_rows_inputs, ("ap", "b")),
    "halo_waits": Fixture(halo_waits, halo_inputs, ("b", "c", "ghost")),
    "two_exchanges": Fixture(two_exchanges, two_exchanges_inputs, ("ghost_a", "ghost_x", "b")),
    "halo_steps": Fixture(halo_steps, halo_steps_inputs, ("ap", "b", "c")),
    "residual_loop": Fixture(residual_loop, residual_inputs, ("a", "r", "s")),
    "ring": Fixture(ring, ring_inputs, ("b", "d")),
    "redistribute": Fixture(redistribute, redistribute_inputs, ("Sigma",)),
}
