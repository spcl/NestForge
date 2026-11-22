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
    "residual_loop": Fixture(residual_loop, residual_inputs, ("a", "r", "s")),
    "ring": Fixture(ring, ring_inputs, ("b", "d")),
    "redistribute": Fixture(redistribute, redistribute_inputs, ("Sigma",)),
}
