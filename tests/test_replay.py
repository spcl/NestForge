# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""nestforge.replay: label-free picks, the script guard rails, the s231 reference at preset S, the Markdown splice."""

from __future__ import annotations

from pathlib import Path

import pytest
from dace.sdfg.state import LoopRegion
from hpcagent_bench.flags import openmp_launch_env

from nestforge.corpus.bench import corpus_kernel, preset_sizes, with_program
from nestforge.paths import REPO_ROOT
from nestforge.replay import (
    Context,
    ReplayError,
    Script,
    load_solution,
    markdown,
    markers,
    openmp_env,
    replay,
    short,
    splice,
)
from nestforge.session import Session
from nestforge.stages.canonicalize import Targets

SOLUTIONS = sorted((REPO_ROOT / "reference_solutions").glob("*.py"))
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
    assert move.epoch == session.epoch
    assert len(move.labels) == 2


def test_pick_refuses_a_filter_nothing_matches() -> None:
    script = Script(s231_session())
    with pytest.raises(ReplayError, match="no move of interchange-for-doall"):
        script.pick("interchange-for-doall", writes=["bb"])


def test_apply_refuses_a_stale_move() -> None:
    script = Script(s231_session())
    move = script.pick("interchange-for-doall")
    script.apply(move)
    with pytest.raises(ReplayError, match="stale"):
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


@pytest.mark.integration
@pytest.mark.parametrize("path", SOLUTIONS, ids=[p.stem for p in SOLUTIONS])
def test_every_solution_replays_correctly(path: Path) -> None:
    """Every reference solution still applies on these pins, and every variant matches the oracle at preset S."""
    solution = load_solution(path)
    if solution.ci_skip:
        pytest.skip(solution.ci_skip)
    report = replay(solution, CI_CONTEXT, reps=QUICK_REPS)
    names = [v.name for v in report.variants]
    assert names == ["baseline", "reference", *solution.decoys]
    for variant in report.variants:
        assert variant.error == "" and variant.correct is True, variant
        assert variant.time_ms is not None and variant.speedup is not None
    assert report.variants[0].speedup == pytest.approx(1.0)
    assert set(report.pins) == {"nestforge", "dace", "hpcagent_bench"}
    assert markdown([report]).count("| cpu, 1 thread(s), preset S |") == len(names)


def test_a_solution_states_why_ci_skips_it() -> None:
    """``CI_SKIP`` reaches the loaded solution; a script without it is replayed."""
    assert "MapFission" in load_solution(REPO_ROOT / "reference_solutions" / "cloudsc_nproma.py").ci_skip
    assert load_solution(S231).ci_skip == ""


def test_a_solution_can_start_from_a_rewritten_program() -> None:
    """``PROGRAM`` swaps the kernel's loop nest and keeps its manifest: conv3d_box's output-stationary nest, graded on
    conv_3d's inputs."""
    solution = load_solution(REPO_ROOT / "reference_solutions" / "conv3d_box.py")
    assert solution.kernel == "conv_3d"
    assert solution.program == REPO_ROOT / "reference_solutions" / "programs" / "conv3d_box_dace.py"
    corpus = corpus_kernel(solution.kernel)
    assert corpus is not None and solution.program is not None
    rewritten = with_program(corpus, solution.program)
    assert rewritten.spec is corpus.spec and rewritten.dace_file == solution.program
    assert load_solution(S231).program is None and load_solution(S231).params == {}


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


def test_short_cuts_only_full_shas() -> None:
    assert short("3469167e508742b3df6fb9feba4d4c72d9e9f9e8") == "3469167e5"
    assert short("0aa3ec0+replay-uncommitted") == "0aa3ec0+replay-uncommitted"
