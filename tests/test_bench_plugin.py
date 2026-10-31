# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The NestForge bench plugin: loaded through HPCAgent-Bench's own plugin loader, every tool registered, the
structured and kernel phases served with performance feedback, and ``finalize``'s library graded by the judge."""

import contextlib
import gc
import inspect
import multiprocessing
import pathlib
import sys
import threading
import tomllib
from collections.abc import Iterator
from http import HTTPMethod, HTTPStatus
from types import SimpleNamespace

import pytest
from dace.codegen.common import get_gpu_backend
from hpcagent_agent.plugins import Plugin
from hpcagent_agent.tools import registry
from hpcagent_bench import config, plugins
from hpcagent_bench.api import RunConfig
from hpcagent_bench.categories import Baseline, InputMode, Language, Oracle, Preset, SourceMode
from hpcagent_bench.columns import FRAMEWORKS
from hpcagent_bench.config import SubmitFeedback
from hpcagent_bench.harness import tools
from hpcagent_bench.harness.envelope import Submission
from hpcagent_bench.harness.judge_wire import GradeAnswer, JudgeAnswer
from hpcagent_bench.harness.routes import ROUTES, RouteRequest, next_phase
from hpcagent_bench.harness.service import JUDGE_FAULT_CAUSE, ServiceConfig, make_server
from hpcagent_bench.harness.task import Task
from hpcagent_bench.skill_frontmatter import split_frontmatter
from hpcagent_bench.vocabulary import PACKETS
from stub_session import Stub, install

import nestforge_bench
from nestforge.paths import REPO_ROOT
from nestforge.server import CALLS, Reply, Setup
from nestforge_bench import ReferenceOptimizer

PLUGIN = REPO_ROOT / "plugin" / "nestforge"
PAGE = PLUGIN / "skills" / "nestforge" / "SKILL.md"
KERNEL = "loop_level_reasoning/fuse_diamond/fuse_diamond"
#: Seconds a spawned judge has to import the bench and bind its port.
JUDGE_START_TIMEOUT_S = 300.0


class FakeJudge:
    """The part of ``service.JudgeHandler`` a route uses; keeps the last answer."""

    def __init__(self) -> None:
        self.cfg = RunConfig(preset=Preset.S)
        self.code = 0
        self.answer: dict[str, object] = {}

    def reply(self, code: int, payload: JudgeAnswer) -> None:
        self.code, self.answer = code, payload.to_json()

    @contextlib.contextmanager
    def device_slot(self) -> Iterator[object]:
        yield object()


@pytest.fixture(scope="module")
def loaded(nestforge_plugin: tuple[Plugin, ...]) -> Iterator[None]:
    assert [plugin.name for plugin in nestforge_plugin if plugin.root == PLUGIN] == ["nestforge"]
    yield
    # every session process this module started, so no other test inherits a live child
    episodes = sys.modules[ROUTES["nestforge"].handler.__module__].EPISODES
    for _, (episode, _) in list(episodes.states.items()):
        for session in episode.sessions.values():
            session.close()


def post(tool: str, episode: str = "episode-cpu", kernel: str = KERNEL, **body: object) -> FakeJudge:
    judge = FakeJudge()
    request = RouteRequest(HTTPMethod.POST, ["nestforge", tool], {}, {"kernel": kernel, **body}, episode)
    ROUTES["nestforge"].handler(judge, request)
    return judge


@pytest.fixture
def stub(loaded: None, monkeypatch: pytest.MonkeyPatch) -> Iterator[Stub]:
    """Every session the route starts answers from a :class:`Stub`; its episodes are forgotten afterwards."""
    episodes = sys.modules[ROUTES["nestforge"].handler.__module__].EPISODES
    before = set(episodes.states)
    yield install(monkeypatch, Stub())
    for episode in set(episodes.states) - before:
        episodes.drop(episode)


def test_every_phase_registers_its_tools(loaded: None) -> None:
    names = {"get_cpf", "apply_move", "set_scope", "list_kernels", "write_kernel", "tune_kernel", "get_performance"}
    assert names <= registry.REGISTRY.keys()
    assert all(not registry.REGISTRY[name].core for name in names)


def test_a_scope_set_through_the_route_can_be_unset_again(loaded: None) -> None:
    rows = [
        line.lstrip("|`- ").split()[0]
        for line in post("get_cpf", episode="episode-scope").answer["result"].splitlines()[1:]
    ]
    states = [row for row in rows if row.startswith("state")]
    tree = post("view_tree", episode="episode-scope", expand=states).answer["result"]
    maps = [line.lstrip("|`- ").split()[0] for line in tree.splitlines() if line.lstrip("|`- ").startswith("doall")]

    scoped = post("set_scope", episode="episode-scope", labels=maps[:2]).answer["result"]
    unset = post("unset_scope", episode="episode-scope", name=scoped["reason"]).answer["result"]

    assert (scoped["status"], unset["status"]) == ("applied", "applied"), (scoped, unset)


def test_each_manifest_phase_serves_the_tools_of_its_wire_phase(loaded: None) -> None:
    from hpcagent_bench_plugins.nestforge.wire import TOOLS, Phase

    manifest = tomllib.loads((PLUGIN / "plugin.toml").read_text())

    for name, phase in manifest["phases"].items():
        assert sorted(phase.get("tools", [])) == sorted(t for t, d in TOOLS.items() if d.phase == name), name
    assert sorted(manifest["packets"]["nestforge"]["tools"]) == sorted(
        (*(t for t, d in TOOLS.items() if d.phase == Phase.ANY), "score", "submit")
    )


def test_the_packet_exposes_the_structured_phase_and_its_page_and_every_tool_is_served(loaded: None) -> None:
    from hpcagent_bench_plugins.nestforge.wire import TOOLS, Phase

    packet = PACKETS["nestforge"].tools

    assert packet[-1] == plugins.NEXT_PHASE
    assert {TOOLS[tool].phase for tool in packet[:-1] if tool in TOOLS} == {Phase.MOVES, Phase.ANY}
    assert {"score", "submit"} <= registry.OVERRIDES.keys() and {"score", "submit"} <= set(packet)
    assert "set_scope" not in packet and "list_move_kinds" in packet
    assert split_frontmatter(PAGE.read_text())[0].name == "nestforge"
    assert PACKETS["nestforge"].skills == ("nestforge",)
    assert TOOLS.keys() == CALLS.keys()
    for tool in TOOLS:
        params = inspect.signature(CALLS[tool]).parameters
        missing = TOOLS[tool].args.model_fields.keys() - params.keys()
        assert not missing, f"{tool}: the session call takes no {sorted(missing)}"


def test_the_full_packet_serves_every_tool_and_lists_reads_as_read_only(loaded: None) -> None:
    from hpcagent_bench_plugins.nestforge.wire import TOOLS

    packet = PACKETS["nestforge-full"].tools

    assert sorted(packet) == sorted((*TOOLS, "score", "submit", plugins.NEXT_PHASE))
    specs = {name: registry.REGISTRY[name].spec for name in TOOLS}
    hints = {name: spec.hints for name, spec in specs.items() if isinstance(spec, registry.Spec)}
    assert hints.keys() == TOOLS.keys()
    assert {name for name, hint in hints.items() if hint.read_only} == {
        "get_cpf",
        "view_tree",
        "list_move_kinds",
        "list_moves",
        "list_kernels",
        "read_kernel",
        "get_performance",
    }
    assert hints["write_kernel"].destructive and not hints["apply_move"].destructive


def test_a_session_that_dies_is_the_judges_fault_and_the_next_call_starts_a_fresh_one(stub: Stub) -> None:
    stub.dies_on = "apply_move"

    died = post("apply_move", "episode-crash", kind="fuse", labels=["map_0", "map_1"])
    again = post("get_cpf", "episode-crash")

    assert (died.code, died.answer["cause"]) == (HTTPStatus.INTERNAL_SERVER_ERROR, JUDGE_FAULT_CAUSE), died.answer
    assert "starts a fresh session" in died.answer["error"]
    assert again.code == HTTPStatus.OK and len(stub.started) == 2


def test_a_session_that_cannot_start_is_the_judges_fault(loaded: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def refused(setup: object) -> None:
        raise LookupError("canonicalize: RecursionError")

    monkeypatch.setattr(sys.modules[ROUTES["nestforge"].handler.__module__], "SessionProcess", refused)

    judge = post("get_cpf", "episode-unstartable")

    assert (judge.code, judge.answer["cause"]) == (HTTPStatus.INTERNAL_SERVER_ERROR, JUDGE_FAULT_CAUSE), judge.answer


def test_next_phase_runs_the_default_scopes_and_placement_and_returns_to_the_moves(
    stub: Stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(next_phase, "run_phases", lambda: PACKETS["nestforge"].phases)
    judge = FakeJudge()
    request = RouteRequest(HTTPMethod.POST, ["next_phase"], {}, {"kernel": KERNEL}, "episode-round")

    next_phase.serve(judge, request)

    assert judge.code == HTTPStatus.OK, judge.answer
    assert (judge.answer["phase"], judge.answer["round"]) == ("moves", 2)
    assert [(run["phase"], run["status"]) for run in judge.answer["ran"]] == [("scopes", 200), ("placement", 200)]
    assert [call for call in stub.calls if call[0] == "run_defaults"] == [
        ("run_defaults", {"stage": "scopes"}),
        ("run_defaults", {"stage": "placement"}),
    ]


def test_a_tool_that_refuses_the_agents_call_is_still_a_bad_request(stub: Stub) -> None:
    stub.replies["apply_move"] = Reply(error="ValueError: doall-fusion takes 2 label(s); got 1")

    judge = post("apply_move", "episode-stale", kind="fuse", labels=["map_0"])

    assert judge.code == HTTPStatus.BAD_REQUEST and "takes 2 label" in judge.answer["error"], judge.answer


def test_concurrent_calls_of_one_episode_start_one_session_and_reach_it_one_at_a_time(stub: Stub) -> None:
    calls = [threading.Thread(target=post, args=("get_cpf", "episode-race")) for _ in range(4)]
    for call in calls:
        call.start()
    for call in calls:
        call.join()

    assert len(stub.started) == 1, "a second session of the same kernel would leak until garbage collection"
    assert len(stub.calls) == 4 and stub.most_in_flight == 1, "two calls on one pipe read each other's answers"


def test_an_unknown_argument_is_refused(loaded: None) -> None:
    assert post("list_moves", depth=3).code == HTTPStatus.BAD_REQUEST


@pytest.mark.integration
def test_the_structured_phase_returns_the_tree_and_a_move_returns_performance(loaded: None) -> None:
    tree = post("get_cpf")
    moves = post("list_moves")
    first = moves.answer["result"][0]

    moved = post("apply_move", **first)

    assert tree.code == HTTPStatus.OK and "doall" in str(tree.answer["result"])
    assert moved.code == HTTPStatus.OK and moved.answer["result"]["status"] == "applied"
    performance = moved.answer["performance"]
    structured = performance["structured"]
    assert "error" not in structured and structured["time_us"] > 0
    assert structured["correct"] is True, structured
    assert set(performance["offloading"]["devices"].values()) == {"cpu"}
    assert performance["kernels"] and all(kernel["correct"] for kernel in performance["kernels"])
    assert post("get_performance").answer["performance"] == performance, "the unchanged program is measured once"


def kernel_round(episode: str, kernel: str = KERNEL) -> tuple[str, str]:
    """Default scopes on ``episode``, then the first kernel's name and source."""
    assert post("run_defaults", episode=episode, kernel=kernel, stage="scopes").code == HTTPStatus.OK
    name = post("list_kernels", episode=episode, kernel=kernel).answer["result"][0]["name"]
    return name, post("read_kernel", episode=episode, kernel=kernel, name=name).answer["result"]


@pytest.mark.integration
def test_a_kernel_written_back_unchanged_is_linked_and_a_wrong_entry_is_refused(loaded: None) -> None:
    name, source = kernel_round("episode-kernel")

    written = post("write_kernel", episode="episode-kernel", name=name, source=source, language="cpp")
    renamed = source.replace(f"void {name}(", f"void {name}_other(", 1)
    refused = post("write_kernel", episode="episode-kernel", name=name, source=renamed, language="cpp")

    assert written.answer["result"]["status"] == "ok", written.answer
    assert written.answer["performance"]["structured"]["correct"] is True
    assert refused.answer["result"]["status"] == "refused", refused.answer


@pytest.mark.integration
def test_a_tuned_kernel_reports_the_winning_configuration(loaded: None) -> None:
    name, _ = kernel_round("episode-tune")

    tuned = post("tune_kernel", episode="episode-tune", name=name, compilers=["gcc"]).answer

    assert tuned["result"]["cells"] > 1 and tuned["result"]["winner"] is not None, tuned
    assert tuned["performance"]["structured"]["correct"] is True


def serve_judge(ports: "multiprocessing.Queue[int]") -> None:
    """A judge taking library submissions at S, answering the whole grade, until terminated."""
    with (
        config.overridden("fuzz.hidden_correctness_presets", ["S"] * 5),
        config.overridden("service.submit_feedback", SubmitFeedback.FULL),
    ):
        server = make_server(
            "127.0.0.1", 0, ServiceConfig(baseline=Baseline.C, oracle=Oracle.AUTO, input_mode=InputMode.ANY, repeat=3)
        )
        ports.put(server.server_address[1])
        server.serve_forever()


@pytest.fixture
def judge_url(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> Iterator[str]:
    """A judge in a spawned process, grading at S as the bench's own suite does. Not a thread of this one: a GPU
    grade initializes the device runtime in the judge's process, and a later test forking this one dies with it."""
    # The judge's shared folder, and the one the session process a test starts writes its library into.
    monkeypatch.setenv("HPCAGENT_BENCH_SHARED_DIR", str(tmp_path))
    monkeypatch.setenv("HPCAGENT_BENCH_FUZZ_SIZE_CAP", "4096")
    monkeypatch.setenv("HPCAGENT_BENCH_MEASUREMENT_TIMING_BACKEND", "min_of_k")
    monkeypatch.setenv("HPCAGENT_BENCH_SERVICE_PRESET", "S")
    monkeypatch.setenv("HPCAGENT_BENCH_FUZZ_ANCHOR", "S")
    spawn = multiprocessing.get_context("spawn")
    ports: multiprocessing.Queue[int] = spawn.Queue()
    judge = spawn.Process(target=serve_judge, args=(ports,))  # not daemonic: the judge forks its grades
    judge.start()
    try:
        yield f"http://127.0.0.1:{ports.get(timeout=JUDGE_START_TIMEOUT_S)}"
    finally:
        judge.terminate()
        judge.join()


def graded(judge_url: str, kernel: str, final: dict[str, object], language: Language = Language.CPP) -> GradeAnswer:
    """``final``'s library as a ``language`` submission, graded by the judge at ``judge_url``."""
    submission = Submission(
        language=language, library=str(final["library"]), workspace_bytes=str(final["workspace_bytes"])
    )
    grade = tools.JudgeClient(judge_url).submit(submission, kernel)
    assert isinstance(grade, GradeAnswer), grade
    return grade


def test_the_reference_column_starts_on_the_preset_the_token_resolves_to(monkeypatch: pytest.MonkeyPatch) -> None:
    class Started(Exception):
        pass

    presets: list[str] = []

    def recording_start(setup: Setup) -> None:
        presets.append(setup.preset)
        raise Started

    monkeypatch.setattr(nestforge_bench, "start", recording_start)
    sut = ReferenceOptimizer("dace_cpu_canon_nestforge_reference")

    with config.scoped_environment({"HPCAGENT_BENCH_SERVICE_PRESET": "XL+fuzz"}), pytest.raises(Started):
        sut.tune(Task(KERNEL, SourceMode.ANY, Language.CPP), "canon_cpu")

    assert presets == ["fuzzed"]


def test_the_reference_column_leaves_an_episodes_library_in_the_episodes_folder(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FinalizedSession:
        def finalize(self, out: pathlib.Path) -> SimpleNamespace:
            library = out / "libkernel.so"
            library.touch()
            return SimpleNamespace(library=library, workspace_bytes="0")

    monkeypatch.setattr(nestforge_bench, "start", lambda setup: (FinalizedSession(), None, None))
    monkeypatch.setattr(nestforge_bench, "baseline", lambda script, ctx: None)
    monkeypatch.setattr(nestforge_bench, "reference_solution", lambda kernel: None)
    sut = ReferenceOptimizer("dace_cpu_canon_nestforge_reference", workdir=tmp_path)

    library = pathlib.Path(sut.tune(Task(KERNEL, SourceMode.ANY, Language.CPP), "canon_cpu").library)
    gc.collect()

    assert library.parent.parent == tmp_path
    assert library.is_file(), "a deferred /submit still needs it after the submission is gone"


# s231 has a reference solution; the diamond has none and runs the default moves
@pytest.mark.judge
@pytest.mark.parametrize("kernel", ["tsvc_2_s231", KERNEL])
def test_the_reference_column_submits_a_library_the_judge_grades_correct(
    loaded: None, judge_url: str, kernel: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    column = "dace_cpu_canon_nestforge_reference"
    monkeypatch.setenv("HPCAGENT_BENCH_USER_PLUGIN_DIRS", str(PLUGIN))
    if column not in FRAMEWORKS.entries:
        plugins.register_flavors.cache_clear()
        plugins.register_flavors()
    sut = ReferenceOptimizer(column)

    submission = sut.tune(Task(kernel, SourceMode.ANY, Language.CPP), "canon_cpu")

    assert FRAMEWORKS.entries[column].pipelines == ("canon_cpu",), "the manifest registers the column"
    grade = tools.JudgeClient(judge_url).submit(submission, kernel)
    assert isinstance(grade, GradeAnswer), grade
    assert grade.build_ok is True and grade.correct is True, grade.detail


# a straight diamond of maps; maps inside a time loop
@pytest.mark.judge
@pytest.mark.parametrize("kernel", [KERNEL, "jacobi_1d"])
def test_a_finalized_program_is_a_library_the_judge_grades_correct(
    loaded: None, judge_url: str, kernel: str, tmp_path: pathlib.Path
) -> None:
    episode = f"episode-final-{kernel}"
    name, _ = kernel_round(episode, kernel)
    post("tune_kernel", episode=episode, kernel=kernel, name=name, compilers=["gcc"])

    final = post("finalize", episode=episode, kernel=kernel).answer["result"]

    assert pathlib.Path(final["library"]).is_relative_to(tmp_path), final
    assert f"{name}(" in final["source"] and final["kernels"], final
    grade = graded(judge_url, kernel, final)
    assert grade.build_ok is True and grade.correct is True, grade.detail


@pytest.mark.judge
def test_a_program_finalized_after_the_structured_phase_gets_the_default_kernels(loaded: None, judge_url: str) -> None:
    episode = "episode-structured"
    post("run_defaults", episode=episode, stage="moves")

    final = post("finalize", episode=episode).answer["result"]

    assert final["kernels"], final
    grade = graded(judge_url, KERNEL, final)
    assert grade.build_ok is True and grade.correct is True, grade.detail


@pytest.mark.judge
def test_each_library_product_is_a_kernel_and_the_program_asks_for_its_scratch(loaded: None, judge_url: str) -> None:
    # atax's two products are library nodes, each its own default kernel; tmp = A @ x lives in the judge's scratch

    final = post("finalize", episode="episode-final-atax", kernel="atax").answer["result"]

    assert len(final["kernels"]) == 2 and final["workspace_bytes"] != "0", final
    grade = graded(judge_url, "atax", final)
    assert grade.build_ok is True and grade.correct is True, grade.detail


@pytest.mark.gpu
def test_a_gpu_target_reports_its_offloading(loaded: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGUAGE", "cuda")

    judge = post("run_defaults", episode="episode-gpu", stage="placement")

    assert judge.code == HTTPStatus.OK
    offloading = judge.answer["performance"]["offloading"]
    assert "gpu" in offloading["devices"].values()
    assert offloading["transfer_bytes"] > 0
    assert "error" not in judge.answer["performance"]["structured"]


@pytest.mark.gpu
def test_a_finalized_gpu_program_is_a_two_unit_library_the_judge_grades_correct(
    loaded: None, judge_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = get_gpu_backend()
    monkeypatch.setenv("LANGUAGE", backend)

    judge = post("finalize", episode="episode-final-gpu")  # never placed: the default placement

    assert judge.code == HTTPStatus.OK, judge.answer
    final = judge.answer["result"]

    grade = graded(judge_url, KERNEL, final, Language[backend.upper()])
    assert grade.build_ok is True and grade.correct is True, grade.detail
