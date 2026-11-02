# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 2 static score: region entries and cache lines of a program, moves ranked on copies, and the score-guided
stage 2 that applies the best move while it lowers the score."""

import dace
import numpy as np
import pytest

from nestforge.session import Session
from nestforge.stages import score
from nestforge.stages.canonicalize import Targets
from nestforge.stages.moves import EXPAND_LIBRARY, Rewrite
from nestforge.stages.score import CACHE_BYTES, LINE_BYTES, UNSCORED, WEIGHTS, build_name, program_terms, rank_moves

N = dace.symbol("N", dtype=dace.int64)
M = dace.symbol("M", dtype=dace.int64)
#: Rows and columns of the test matrices: 8 MiB of doubles, larger than the cache rule's 1 MiB.
SIDE = 1024
#: Doubles in one cache line.
PER_LINE = int(LINE_BYTES) // 8


@dace.program
def chain(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    t = np.empty_like(a)
    for i in dace.map[0:N]:
        t[i] = a[i] + b[i]
    for i in dace.map[0:N]:
        c[i] = t[i] * 2.0


@dace.program
def rows(a: dace.float64[N, M], b: dace.float64[N, M]):
    for i, j in dace.map[0:N, 0:M]:
        b[i, j] = a[i, j] + 1.0


@dace.program
def columns(a: dace.float64[N, M], b: dace.float64[N, M]):
    for j, i in dace.map[0:M, 0:N]:
        b[i, j] = a[i, j] + 1.0


@dace.program
def map_in_loop(a: dace.float64[N, M]):
    for k in range(1, N):
        for j in dace.map[0:M]:
            a[k, j] = a[k - 1, j] * 0.5


@dace.program
def product(a: dace.float64[N, N], b: dace.float64[N, N], c: dace.float64[N, N]):
    c[:] = a @ b
    for i in dace.map[0:N]:
        c[i, i] = c[i, i] + 1.0


def session_of(program, tmp_path, sizes: dict[str, int]) -> Session:
    session = Session(program.to_sdfg(simplify=True), targets=Targets(gpu=False), work_dir=str(tmp_path), sizes=sizes)
    session.canonicalize()
    return session


def test_entries_count_every_top_level_map_times_the_loop_trips_around_it(tmp_path):
    session = session_of(map_in_loop, tmp_path, {"N": SIDE, "M": SIDE})

    terms = program_terms(session.sdfg, session.sizes)

    assert terms.entries == SIDE - 1
    assert terms.unresolved == 0


def test_unit_stride_touches_an_eighth_of_a_line_per_access_and_a_large_stride_a_whole_line(tmp_path):
    sizes = {"N": SIDE, "M": SIDE}
    accesses = 2 * SIDE * SIDE  # one read and one write per element

    unit = program_terms(session_of(rows, tmp_path / "rows", sizes).sdfg, sizes)
    strided = program_terms(session_of(columns, tmp_path / "columns", sizes).sdfg, sizes)

    assert unit.lines == pytest.approx(accesses / PER_LINE)
    assert strided.lines == pytest.approx(accesses)
    assert unit.entries == strided.entries == 1


def test_an_array_that_fits_the_cache_costs_its_lines_once(tmp_path):
    side = 64  # 32 KiB per matrix
    assert side * side * 8 <= CACHE_BYTES
    sizes = {"N": side, "M": side}

    terms = program_terms(session_of(columns, tmp_path, sizes).sdfg, sizes)

    assert terms.lines == pytest.approx(2 * side * side * 8 / LINE_BYTES)


def test_ranking_scores_moves_on_copies_and_leaves_the_program_unchanged(tmp_path):
    session = session_of(chain, tmp_path, {"N": SIDE * SIDE})
    before = session.sdfg.to_json()

    ranked = session.score_moves("cpu-omp")

    assert ranked, "two maps in a row offer a fusion"
    assert [m.d_score for m in ranked] == sorted(m.d_score for m in ranked)
    best = ranked[0]
    assert best.kind == "doall-fusion" and best.d_entries == -1 and best.d_score < 0
    assert session.sdfg.to_json() == before


def test_guided_moves_apply_the_best_move_while_it_lowers_the_score(tmp_path):
    session = session_of(chain, tmp_path, {"N": SIDE * SIDE})
    start = session.score_terms()

    applied = session.guided_moves("cpu-omp")

    assert [m.kind for m in applied] == ["doall-fusion"]
    after = session.score_terms()
    assert after.entries == start.entries - 1
    weights = WEIGHTS["cpu-omp"]
    assert after.score(weights) - start.score(weights) == pytest.approx(applied[0].d_score)
    assert all(m.d_score >= 0 for m in session.score_moves("cpu-omp"))
    session.sdfg.validate()


def test_the_score_ranks_no_move_that_takes_a_choice_or_that_the_session_finishes(tmp_path):
    session = session_of(product, tmp_path, {"N": SIDE})
    assert {EXPAND_LIBRARY, "tile-doall"} <= {m.kind for m in session.list_moves()}

    assert not {m.kind for m in session.score_moves("cpu-omp")} & UNSCORED


def test_a_move_dace_fails_to_apply_leaves_the_ranking_of_the_others(tmp_path, monkeypatch):
    session = session_of(chain, tmp_path, {"N": SIDE * SIDE})

    def fails() -> None:
        raise ValueError("memlet propagation")

    monkeypatch.setattr(score, "plan_move", lambda *_: Rewrite("fails", fails))

    assert rank_moves(session.sdfg, session.sizes, WEIGHTS["cpu-omp"]) == []


def test_a_bound_of_an_integer_function_evaluates_at_integer_sizes():
    sdfg = dace.SDFG("sqrt_bound")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_state().add_mapped_tasklet(
        "fill", {"i": "0:Max(int_sqrt(N - 1) + 1, 1)"}, {}, "o = 1.0", {"o": dace.Memlet("a[i]")}, external_edges=True
    )

    terms = program_terms(sdfg, {"N": SIDE * SIDE})

    assert terms.unresolved == 0 and terms.entries == 1


def test_a_session_without_sizes_refuses_to_score():
    session = Session(chain.to_sdfg(simplify=True), targets=Targets(gpu=False))
    session.canonicalize()

    with pytest.raises(ValueError, match="no sizes"):
        session.score_moves("cpu-omp")


def test_build_name_picks_the_weights_of_the_target_and_thread_count():
    assert build_name(True, 1) == "gpu"
    assert build_name(False, 1) == "cpu-serial"
    assert build_name(False, 32) == "cpu-omp"
    assert set(WEIGHTS) == {"gpu", "cpu-serial", "cpu-omp"}
