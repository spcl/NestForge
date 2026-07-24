# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The auto-par team must be sized by the cores THIS process may use -- not by the node.

``-ftree-parallelize-loops=N`` bakes N at compile time, so an N read off the whole machine is wrong
everywhere it matters. One rank of a 4-rank job on a 288-CPU node owns 72 CPUs; ``os.cpu_count()``
says 288, the team is sized 4x its cores, and the four ranks fight each other for the same node. And
a hyperthread is not a core: two threads sharing one core's execution units contend rather than
compute, so the logical count oversubscribes by ~2x again.
"""
import os

import pytest

from nestforge.perf.tsvc_full import default_threads, physical_cores


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """These are read from the environment, and a real OMP_NUM_THREADS/SLURM_* on the dev box or a CI
    runner would otherwise decide the answer -- the test must mean the same thing wherever it runs."""
    for var in ("OMP_NUM_THREADS", "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        monkeypatch.delenv(var, raising=False)


def test_a_ranks_slurm_share_beats_the_node_cpu_count(monkeypatch):
    """THE case: 288 CPUs, 4 ranks, one per Grace socket -> this rank owns 72. os.cpu_count() reports the
    node (288) and cannot see the split, so SLURM's answer must win."""
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "72")
    assert default_threads() == 72


def test_slurm_cpus_on_node_is_the_fallback(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "36")
    assert default_threads() == 36


def test_explicit_omp_num_threads_wins_over_slurm(monkeypatch):
    """Explicit intent outranks an inferred share: a user who sets OMP_NUM_THREADS means it."""
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "72")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    assert default_threads() == 8


def test_nested_omp_num_threads_takes_the_outer_level(monkeypatch):
    """REGRESSION: OMP_NUM_THREADS is a per-level LIST for nested parallelism ("72,8" = 72 outer, 8
    inner). int("72,8") raises, and the old code caught that and fell back to the node's CPU count --
    turning the most explicit possible request (72) into the worst possible answer (the whole node).
    The outer level is the one an auto-par loop gets."""
    monkeypatch.setenv("OMP_NUM_THREADS", "72,8")
    assert default_threads() == 72


def test_an_unparseable_thread_count_falls_through_instead_of_crashing(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "not-a-number")
    n = default_threads()
    assert n >= 1  # falls through to the machine; never raises, never 0


def test_the_default_is_physical_cores_not_hyperthreads():
    """With no hints, size by PHYSICAL cores among the CPUs we are allowed to use. A hyperthread shares
    its core's execution units, so counting logical CPUs oversubscribes the team ~2x."""
    n = default_threads()
    allowed = os.sched_getaffinity(0)
    assert n == physical_cores(allowed)
    assert 1 <= n <= len(allowed), f"{n} threads for {len(allowed)} usable CPUs"


def test_physical_cores_collapses_smt_siblings():
    """Every logical CPU belongs to exactly one physical core, so the collapse can only shrink the count
    -- never grow it, and never to zero."""
    allowed = os.sched_getaffinity(0)
    n = physical_cores(allowed)
    assert 1 <= n <= len(allowed)
    assert physical_cores({next(iter(allowed))}) == 1  # one CPU is one core


def test_physical_cores_survives_an_unknown_cpu_id():
    """A kernel without topology/thread_siblings_list (or a bogus id) must degrade to the logical count,
    not raise -- sizing a team is not worth an exception."""
    assert physical_cores({99999}) == 1
