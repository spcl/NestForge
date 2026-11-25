# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""tactics.replay: label-free picks, the script guard rails, the s231 reference at preset S, the Markdown splice."""

from __future__ import annotations

from pathlib import Path

import pytest
from dace.sdfg.state import LoopRegion
from hpcagent_bench.flags import openmp_launch_env

from tactics import session as session_module
from tactics.corpus.bench import corpus_kernel, preset_sizes
from tactics.paths import REPO_ROOT
from tactics.records import MoveStatus
from tactics.replay import (
    Context,
    ReplayError,
    Script,
    load_solution,
    markdown,
    markers,
    openmp_env,
    replay_started,
    splice,
    started,
)
from tactics.session import Session
from tactics.stages.canonicalize import Targets
from tactics.stages.moves import ROLLBACK
from tactics_bench import solution_paths

# a solution with CI_SKIP cannot load on the bench pin; its CI_SKIP says why
SOLUTIONS = [path for path in solution_paths() if not load_solution(path).ci_skip]
S231 = REPO_ROOT / "reference_solutions" / "tsvc_2_s231.py"
#: Where CI replays every solution: correctness only, so the smallest preset on one CPU thread.
CI_CONTEXT = Context("cpu", 1, "S")
#: Timed calls per variant in the end-to-end test; the time is not asserted.
QUICK_REPS = 1


def s231_session(gpu: bool = False) -> Session:
    kernel = corpus_kernel("tsvc_2_s231")
    assert kernel is not None
    session = Session(kernel.to_sdfg(), Targets(gpu=gpu), sizes=preset_sizes(kernel, "S"))
    session.canonicalize()
    return session


def top_loops(session: Session) -> list[LoopRegion]:
    return [block for block in session.sdfg.nodes() if isinstance(block, LoopRegion)]


def test_context_parse_round_trip() -> None:
    ctx = Context.parse("cpu:72:M")
    assert ctx == Context("cpu", 72, "M")
    assert Context.parse(ctx.spelling) == ctx
    assert ctx.key == "cpu-72-M"
    with pytest.raises(ValueError):
        Context.parse("tpu:1:M")


def test_pick_finds_the_one_interchange() -> None:
    session = s231_session()
    move = Script(session).pick("interchange-for-doall", reads=["bb"], writes=["aa"])
    assert move.kind == "interchange-for-doall"
    assert set(move.labels) <= set(session.row_index())
    assert len(move.labels) == 2


def test_pick_refuses_a_filter_nothing_matches() -> None:
    script = Script(s231_session())
    with pytest.raises(ReplayError, match="no move of interchange-for-doall"):
        script.pick("interchange-for-doall", writes=["bb"])


def test_apply_refuses_a_move_whose_rows_a_move_retired() -> None:
    script = Script(s231_session())
    move = script.pick("interchange-for-doall")
    script.apply(move)
    with pytest.raises(ReplayError, match="not_found"):
        script.apply(move)
    assert [m.kind for m in script.applied] == ["interchange-for-doall"]


def test_apply_after_finish_is_refused() -> None:
    script = Script(s231_session())
    move = script.pick("interchange-for-doall")
    script.finish()
    with pytest.raises(ReplayError, match="after finish"):
        script.apply(move)


def test_s231_reference_puts_the_loop_inside_the_map() -> None:
    solution = load_solution(S231)
    assert solution.kernel == "tsvc_2_s231"
    session = s231_session()
    assert top_loops(session), "stage 1 leaves for j { map i } on the CPU"
    reference = Script(session.trial())
    solution.solve(reference, Context("cpu", 72, "S"))
    reference.finish()
    assert [m.kind for m in reference.applied] == ["interchange-for-doall"]
    assert not top_loops(reference.session), "the j loop moved into the map"


def column_scan_script() -> Script:
    kernel = corpus_kernel("column_scan_nlev_nproma")
    assert kernel is not None
    session = Session(kernel.to_sdfg(), Targets(gpu=False), sizes=preset_sizes(kernel, "S"))
    session.canonicalize()
    return Script(session)


def test_scope_after_the_default_makes_the_step_loop_one_kernel() -> None:
    script = column_scan_script()
    script.defaults()
    done = script.scope(where=lambda rows: isinstance(rows[0][0], LoopRegion))
    assert done.kind == "define-scope"
    assert len(done.labels) == 1
    assert not top_loops(script.session), "the step loop is outlined: no loop is left at the top"
    assert [m.kind for m in script.applied] == ["define-scope"]


# Reproducer: replay could not pick a move inside a kernel (0.1.1)
def test_a_schedule_move_inside_a_kernel_replays_with_its_scope() -> None:
    script = column_scan_script()
    script.defaults()
    script.scope(where=lambda rows: isinstance(rows[0][0], LoopRegion))
    (kernel,) = script.session.kernels()

    listed = script.pick("set-schedule-cpu", writes=["_out_y"], scope=kernel)

    done = script.apply(listed.model_copy(update={"schedule": "dynamic", "chunk": 16}), scope=kernel)

    assert (done.kind, done.scope) == ("set-schedule-cpu", kernel)
    assert script.session.history, "the schedule is a change rollback can take back"


def test_scope_refuses_a_filter_nothing_matches() -> None:
    script = column_scan_script()
    script.defaults()
    with pytest.raises(ReplayError, match="no top-level block"):
        script.scope(writes=["no_such_container"])


@pytest.mark.integration
# icon_aes_graupel runs stage 2 three times after a long canonicalization: 850-900 s on a CI runner, so the
# suite-wide 900 s cut it off mid-run (run 45573051650); 1800 s still catches a hang
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("path", SOLUTIONS, ids=[p.stem for p in SOLUTIONS])
def test_every_solution_replays_correctly_and_rolls_back_to_the_canonical_program(
    path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every reference solution still applies on these pins, and every variant matches the oracle at preset S. Rolling
    back every change the solution made, one ``rollback`` at a time, gives back the canonical program and its tree;
    halfway back, the program is a valid SDFG. One test for both, as starting a big kernel costs minutes."""
    solution = load_solution(path)
    monkeypatch.setattr(session_module, "UNDO_DEPTH", 10_000)  # every change of the solution stays in reach
    canonical = started(solution, CI_CONTEXT)

    report = replay_started(solution, CI_CONTEXT, canonical, reps=QUICK_REPS)

    names = [v.name for v in report.variants]
    assert names == ["baseline", "reference"]
    for variant in report.variants:
        assert variant.error == "" and variant.correct is True, variant
        assert variant.time_ms is not None and variant.speedup is not None
    assert report.variants[0].speedup == pytest.approx(1.0)
    assert set(report.pins) == {"tactics", "dace", "hpcagent_bench"}
    assert markdown([report]).count("| cpu, 1 thread(s), preset S |") == len(names)

    sut = canonical[0].trial()
    before = (sut.sdfg.hash_sdfg(), sut.describe())
    solution.solve(Script(sut), CI_CONTEXT)
    changes = len(sut.history)
    halfway = [sut.apply_move(ROLLBACK, []) for _ in range(changes // 2)]
    sut.sdfg.validate()
    rest = [sut.apply_move(ROLLBACK, []) for _ in range(changes - changes // 2)]

    assert all(result.status is MoveStatus.APPLIED for result in halfway + rest)
    assert (sut.sdfg.hash_sdfg(), sut.describe()) == before
    assert sut.history == []


#: Solutions whose ``KERNEL`` is no bench kernel. cloudsc_nproma came with HPCAgent-Bench#124, which was reverted.
NOT_IN_BENCH = frozenset({"cloudsc_nproma"})


def test_every_solution_names_its_file_and_a_bench_kernel() -> None:
    for path in SOLUTIONS:
        kernel = load_solution(path).kernel
        assert path.stem == kernel, f"{path.name} solves {kernel}"
        resolves, exempt = corpus_kernel(kernel) is not None, kernel in NOT_IN_BENCH
        assert resolves != exempt, f"{kernel}: bench kernel {resolves}, exempt {exempt}"


def test_splice_replaces_only_its_kernels_section() -> None:
    begin, end = markers("tsvc_2_s231")
    other_begin, other_end = markers("tsvc_2_s233")
    doc = f"# solutions\n\n{begin}\nold table\n{end}\n\n{other_begin}\nkept\n{other_end}\n"
    out = splice(doc, "tsvc_2_s231", "new table\n")
    assert "old table" not in out and "new table" in out and "kept" in out
    assert splice(out, "tsvc_2_s231", "new table\n") == out
    assert splice("# solutions\n", "tsvc_2_s231", "t\n") == f"# solutions\n\n{begin}\nt\n{end}\n"


def test_openmp_env_carries_the_bench_launch_env() -> None:
    env = openmp_env(Context("cpu", 72, "M"))
    assert env["OMP_NUM_THREADS"] == "72" and env["OMP_PROC_BIND"] == "close"
    assert openmp_launch_env().items() <= env.items()
