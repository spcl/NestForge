# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The NestForge bench plugin: loaded through HPCAgent-Bench's own plugin loader, every tool registered, the
structured and kernel phases served with performance feedback, ``finalize`` refused as not implemented."""

import contextlib
import inspect
import sys
import tomllib
from collections.abc import Iterator
from http import HTTPMethod, HTTPStatus

import pytest
from hpcagent_agent.tools import registry
from hpcagent_bench import plugins
from hpcagent_bench.api import RunConfig
from hpcagent_bench.categories import Preset
from hpcagent_bench.harness.judge_wire import JudgeAnswer
from hpcagent_bench.harness.routes import ROUTES, RouteRequest

from nestforge.paths import REPO_ROOT
from nestforge.server import CALLS

PLUGIN = REPO_ROOT / "plugin" / "nestforge"
KERNEL = "loop_level_reasoning/fuse_diamond/fuse_diamond"


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


def post(tool: str, episode: str = "episode-cpu", **body: object) -> FakeJudge:
    judge = FakeJudge()
    request = RouteRequest(HTTPMethod.POST, ["nestforge", tool], {}, {"kernel": KERNEL, **body}, episode)
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


@pytest.mark.parametrize("tool", ["finalize"])
def test_a_later_phase_tool_is_not_implemented(loaded: None, tool: str) -> None:
    judge = post(tool)

    assert judge.code == HTTPStatus.NOT_IMPLEMENTED
    assert judge.answer["cause"] == "not-implemented"


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


def kernel_round(episode: str) -> tuple[str, str]:
    """Default scopes on ``episode``, then the first kernel's name and source."""
    assert post("run_defaults", episode=episode, stage="scopes").code == HTTPStatus.OK
    name = post("list_kernels", episode=episode).answer["result"][0]["name"]
    return name, post("read_kernel", episode=episode, name=name).answer["result"]


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


@pytest.mark.gpu
def test_a_gpu_target_reports_its_offloading(loaded: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGUAGE", "cuda")

    judge = post("run_defaults", episode="episode-gpu", stage="placement")

    assert judge.code == HTTPStatus.OK
    offloading = judge.answer["performance"]["offloading"]
    assert "gpu" in offloading["devices"].values()
    assert offloading["transfer_bytes"] > 0
    assert "error" not in judge.answer["performance"]["structured"]
