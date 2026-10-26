# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The NestForge bench plugin: loaded through HPCAgent-Bench's own plugin loader, every tool registered, the
structured and kernel phases served with performance feedback, and ``finalize``'s library graded by the judge."""

import contextlib
import inspect
import multiprocessing
import pathlib
import sys
import tomllib
from collections.abc import Iterator
from http import HTTPMethod, HTTPStatus

import pytest
from dace.codegen.common import get_gpu_backend
from hpcagent_agent.tools import registry
from hpcagent_bench import config, plugins
from hpcagent_bench.api import RunConfig
from hpcagent_bench.categories import Baseline, InputMode, Language, Oracle, Preset
from hpcagent_bench.config import SubmitFeedback
from hpcagent_bench.harness import tools
from hpcagent_bench.harness.envelope import Submission
from hpcagent_bench.harness.judge_wire import GradeAnswer, JudgeAnswer
from hpcagent_bench.harness.routes import ROUTES, RouteRequest
from hpcagent_bench.harness.service import ServiceConfig, make_server

from nestforge.paths import REPO_ROOT
from nestforge.server import CALLS

PLUGIN = REPO_ROOT / "plugin" / "nestforge"
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
def loaded() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HPCAGENT_BENCH_USER_PLUGIN_DIRS", str(PLUGIN))
        plugins.register.cache_clear()
        plugins.load.cache_clear()
        found = plugins.load()
        from hpcagent_agent import plugins as agent_plugins

        agent_plugins.import_tools()
    assert [plugin.name for plugin in found if plugin.root == PLUGIN] == ["nestforge"]
    yield
    # every session process this module started, so no other test inherits a live child
    episodes = sys.modules[ROUTES["nestforge"].handler.__module__].EPISODES
    for _, (kernels, _) in list(episodes.states.items()):
        for session in kernels.values():
            session.close()


def post(tool: str, episode: str = "episode-cpu", kernel: str = KERNEL, **body: object) -> FakeJudge:
    judge = FakeJudge()
    request = RouteRequest(HTTPMethod.POST, ["nestforge", tool], {}, {"kernel": kernel, **body}, episode)
    ROUTES["nestforge"].handler(judge, request)
    return judge


def test_every_phase_registers_its_tools(loaded: None) -> None:
    names = {"get_cpf", "apply_move", "set_scope", "list_kernels", "write_kernel", "tune_kernel", "get_performance"}
    assert names <= registry.REGISTRY.keys()
    assert all(not registry.REGISTRY[name].core for name in names)


def test_a_scope_set_through_the_route_can_be_unset_again(loaded: None) -> None:
    epoch = 1  # get_cpf canonicalizes once
    tree = post("view_tree", episode="episode-scope", expand=["state0_0"], epoch=epoch).answer["result"]
    maps = [line.split()[1] for line in tree.splitlines() if line.lstrip("|`- ").startswith("kernel")]

    scoped = post("set_scope", episode="episode-scope", labels=maps[:2], epoch=epoch).answer["result"]
    unset = post("unset_scope", episode="episode-scope", name=scoped["reason"], epoch=epoch + 1).answer["result"]

    assert (scoped["status"], unset["status"]) == ("applied", "applied"), (scoped, unset)


def test_every_packet_tool_is_served_with_the_arguments_its_session_call_takes(loaded: None) -> None:
    from hpcagent_bench_plugins.nestforge.wire import TOOLS

    packet = tomllib.loads((PLUGIN / "plugin.toml").read_text())["packets"]["nestforge"]["tools"]

    assert set(packet) == CALLS.keys()
    for tool in packet:
        params = inspect.signature(CALLS[tool]).parameters
        missing = TOOLS[tool].args.model_fields.keys() - params.keys()
        assert not missing, f"{tool}: the session call takes no {sorted(missing)}"


def test_an_unknown_argument_is_refused(loaded: None) -> None:
    assert post("list_moves", depth=3).code == HTTPStatus.BAD_REQUEST


@pytest.mark.integration
def test_the_structured_phase_returns_the_tree_and_a_move_returns_performance(loaded: None) -> None:
    tree = post("get_cpf")
    moves = post("list_moves")
    first = moves.answer["result"][0]

    moved = post("apply_move", **first)

    assert tree.code == HTTPStatus.OK and "epoch" in str(tree.answer["result"])
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
def test_a_program_without_kernels_is_one_entry_that_asks_for_its_scratch(loaded: None, judge_url: str) -> None:
    # atax's two products are library nodes, so no scope is made; tmp = A @ x lives in the judge's scratch

    final = post("finalize", episode="episode-final-atax", kernel="atax").answer["result"]

    assert final["kernels"] == {} and final["workspace_bytes"] != "0", final
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
