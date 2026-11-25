# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A communication row is never a kernel and never part of one (``docs/design/comm-nodes.md``, section 5): the default
scopes cut at it, ``set_scope`` refuses it, a kernel source may not call MPI, NCCL or RCCL, a program that
communicates runs on the CPU, and the row is not inlined and runs on MPI. A kernel that held a row would rewrite the
exchange in the kernel phase, where nothing checks what the ranks send."""

from comm_programs import halo_exchange, residual_in_time_loop
from dace.frontend.python.parser import DaceProgram
from dace.libraries.comm.nodes.node import CommNode

from tactics.ir.comm import comm_class
from tactics.ir.names import top_level_doalls
from tactics.records import KernelStatus, MoveStatus
from tactics.session import Session
from tactics.stages.expand import implementations
from tactics.stages.scopes import node_boundary


def canonical(program: DaceProgram) -> Session:
    session = Session(program.to_sdfg(simplify=True))
    session.canonicalize()
    return session


def test_a_communication_row_is_no_doall() -> None:
    sdfg = canonical(residual_in_time_loop).sdfg
    assert not [n for s in sdfg.states() for n in top_level_doalls(s) if isinstance(n, CommNode)]


def test_set_scope_refuses_a_row_and_a_block_that_holds_one() -> None:
    session = canonical(residual_in_time_loop)

    for labels in (["lib_comm0"], ["for0"]):
        result = session.set_scope(labels)
        assert result.status is MoveStatus.ILLEGAL, result
        assert "communication stays outside kernels" in str(result.reason) and "lib_comm0" in str(result.reason)
    assert not session.kernels()


def test_the_default_scopes_cut_at_the_row_so_the_compute_around_it_forms_kernels() -> None:
    """The time loop holds two doalls and an Allreduce: taken whole it would be one kernel holding the exchange."""
    session = canonical(residual_in_time_loop)

    session.default_scopes()

    kernels = session.kernels()
    assert len(kernels) == 2, sorted(kernels)
    for ext in kernels.values():
        body = node_boundary(ext).standalone_sdfg
        assert not [n for n, _ in body.all_nodes_recursive() if comm_class(n) is not None], ext.name
    assert [n for n, _ in session.sdfg.all_nodes_recursive() if isinstance(n, CommNode)], "the row stays"


def test_a_program_that_communicates_runs_on_the_cpu() -> None:
    session = canonical(residual_in_time_loop)
    session.default_scopes()
    first = min(session.kernels())

    result = session.place({first: "gpu"})

    assert result.status is MoveStatus.ILLEGAL
    assert result.reason == f"a program that communicates runs on the CPU in this version; place {first} on the cpu."


def test_a_kernel_source_that_calls_mpi_is_refused_before_it_is_built() -> None:
    session = canonical(residual_in_time_loop)
    source = '#include <mpi.h>\nextern "C" void k0(double* a) { MPI_Barrier(MPI_COMM_WORLD); }\n'

    outcome = session.set_kernel_source("k0", source, "cpp")

    assert outcome.status is KernelStatus.REFUSED and "does not communicate" in outcome.reason, outcome


def test_a_row_is_not_inlined_and_offers_mpi_only() -> None:
    session = canonical(halo_exchange)
    rows = [n for n, _ in session.sdfg.all_nodes_recursive() if isinstance(n, CommNode)]

    result = session.apply_move("inline-library", [rows[0].label])

    assert result.status is MoveStatus.ILLEGAL and "communicates" in str(result.reason), result
    assert all(set(implementations(row, gpu=True)) <= {"MPI"} for row in rows)
