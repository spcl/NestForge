# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The agent loop over stages 2 to 5, driven by a scripted model (no network), and the one model call it makes
per turn, checked against hand-written stand-ins for the OpenAI and Anthropic SDKs."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest

import dace

from nestforge.agent import llm, loop
from nestforge.agent.llm import Reply, Tool, ToolCall
from nestforge.session import Session

N = dace.symbol("N", dtype=dace.int64)


@dace.program
def two_outputs(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i] + 1.0
    for i in dace.map[0:N]:
        c[i] = a[i] * 2.0


@dataclass(slots=True)
class ScriptedModel:
    """Answers each call with the next scripted reply (no call once the script runs out) and records every
    prompt it was sent."""

    script: list[Reply]
    prompts: list[tuple[str, str, list[str]]] = field(default_factory=list)

    def __call__(self, system: str, prompt: str, tools: list[Tool]) -> Reply:
        self.prompts.append((system, prompt, [t.name for t in tools]))
        return self.script.pop(0) if self.script else Reply("done", [])


def session(tmp_path) -> Session:
    sut = Session(two_outputs.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": 64})
    sut.canonicalize()
    return sut


def map_labels(sut: Session) -> list[str]:
    return [label for label, (obj, _) in sut.row_index().items() if isinstance(obj, dace.nodes.MapEntry)]


def stage_prompts(model: ScriptedModel, stage: str) -> list[str]:
    return [prompt for system, prompt, _ in model.prompts if system == loop.SYSTEM and f"Stage {stage}:" in prompt]


@pytest.mark.e2e
def test_a_silent_model_leaves_every_stage_to_its_default_and_the_program_still_computes(tmp_path):
    sut = session(tmp_path)
    model = ScriptedModel([])

    log = loop.run(sut, model, "make it fast", turns=2)

    assert log == []
    assert [tools for _, _, tools in model.prompts] == [
        ["list_moves", "apply_move", "metrics"],
        ["define_scope"],
        ["place"],
        ["set_kernel_source", "metrics"],
    ]
    names = [k["name"] for k in sut.list_kernels()]
    assert names and sorted(sut.builds) == sorted(names), "stage 5's default builds every kernel"
    a, b, c = np.arange(64.0), np.zeros(64), np.zeros(64)
    sut.sdfg(a=a, b=b, c=c, N=64)
    np.testing.assert_array_equal(b, a + 1.0)
    np.testing.assert_array_equal(c, a * 2.0)


@pytest.mark.e2e
def test_each_prompt_is_rebuilt_from_the_log_and_only_the_last_turns_results(tmp_path):
    sut = session(tmp_path)
    model = ScriptedModel(
        [
            Reply("", [ToolCall("list_moves", {"kind": "map-fusion"})]),
            Reply("", [ToolCall("metrics", {"label": "nothing_0"})]),
        ]
    )

    log = loop.run(sut, model, "make it fast", turns=3)

    first, second, third = stage_prompts(model, "moves")
    assert all(p.startswith("Task: make it fast") and "Latest feedback:\nnone yet" in p for p in (first, second))
    assert "Actions so far:\nnone" in first
    assert log[0].startswith("[moves] list_moves(kind=map-fusion) -> ")
    assert log[0] in second and "Results of your last calls:\nlist_moves(kind=map-fusion):" in second
    assert log[1] == "[moves] metrics(label=nothing_0) -> no tree row is labeled nothing_0."
    # the list_moves output was the last turn's result in the second prompt, and is gone from the third
    assert "list_moves(kind=map-fusion):" not in third and log[1] in third


@pytest.mark.e2e
def fission_then_refused_calls(sut: Session) -> ScriptedModel:
    labels = map_labels(sut)
    assert len(labels) == 1, "canonicalization fuses the two maps, else this tests nothing"
    return ScriptedModel(
        [
            Reply("", [ToolCall("apply_move", {"kind": "map-fission", "labels": labels, "epoch": sut.epoch})]),
            Reply("", [ToolCall("apply_move", {"kind": "map-fission", "labels": labels, "epoch": 0})]),
            Reply("", [ToolCall("frobnicate", {})]),
        ]
    )


@pytest.mark.e2e
def test_an_applied_move_and_the_refused_calls_after_it_are_logged_not_raised(tmp_path):
    sut = session(tmp_path)

    log = loop.run(sut, fission_then_refused_calls(sut), "make it fast", turns=3)

    assert log[0].endswith("-> applied: MapFission")
    assert log[1].endswith("-> stale: labels read at epoch 0; the program is at epoch 2. Describe again.")
    assert log[2] == "[moves] frobnicate() -> error: no tool frobnicate"


@pytest.mark.e2e
@pytest.mark.xfail(
    strict=True,
    reason="DaCe canonicalize's 'end' stage runs FuseMaps, so the post-fusion stages fuse a hand fission back",
)
def test_finishing_hand_chosen_moves_keeps_their_granularity(tmp_path):
    sut = session(tmp_path)

    loop.run(sut, fission_then_refused_calls(sut), "make it fast", turns=3)

    assert len(sut.list_kernels()) >= 2


@pytest.mark.e2e
def test_the_analyst_rewrites_the_report_the_kernel_stage_sees(tmp_path):
    sut = session(tmp_path)
    model = ScriptedModel([Reply("", [])] * 3 + [Reply("fuse extcall_0 into extcall_1", [])])

    loop.run(sut, model, "make it fast", turns=1, analyst=True)

    analyst_calls = [prompt for system, prompt, _ in model.prompts if system == loop.ANALYST]
    assert len(analyst_calls) == 1 and analyst_calls[0].startswith("kernel times: ")
    (kernels_prompt,) = stage_prompts(model, "kernels")
    assert "Latest feedback:\nfuse extcall_0 into extcall_1" in kernels_prompt


TOOL = Tool("metrics", "Work of a row.", {"type": "object", "properties": {"label": {"type": "string"}}})


@dataclass(slots=True)
class FakeOpenAI:
    """Stands in for ``openai.OpenAI``: records the request, answers with one tool call."""

    requests: list[dict] = field(default_factory=list)
    chat: SimpleNamespace = field(init=False)

    def __post_init__(self) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request) -> SimpleNamespace:
        self.requests.append(request)
        call = SimpleNamespace(function=SimpleNamespace(name="metrics", arguments='{"label": "kernel1_0"}'))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[call]))])


@dataclass(slots=True)
class FakeAnthropic:
    """Stands in for ``anthropic.Anthropic``: records the request, answers with text and one tool use."""

    requests: list[dict] = field(default_factory=list)
    messages: SimpleNamespace = field(init=False)

    def __post_init__(self) -> None:
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **request) -> SimpleNamespace:
        self.requests.append(request)
        text = SimpleNamespace(type="text", text="ok")
        use = SimpleNamespace(type="tool_use", name="metrics", input={"label": "kernel1_0"})
        return SimpleNamespace(content=[text, use])


def test_the_openai_call_sends_system_and_user_messages_and_function_tools():
    sdk = FakeOpenAI()

    reply = llm.chat(llm.Client("openai", "some-model", sdk=sdk), "sys", "user", [TOOL])

    assert reply == Reply("ok", [ToolCall("metrics", {"label": "kernel1_0"})])
    (request,) = sdk.requests
    assert request["model"] == "some-model"
    assert request["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "user"}]
    function = {"name": "metrics", "description": "Work of a row.", "parameters": TOOL.parameters}
    assert request["tools"] == [{"type": "function", "function": function}]


def test_the_anthropic_call_sends_a_system_prompt_one_user_message_and_input_schemas():
    sdk = FakeAnthropic()

    reply = llm.chat(llm.Client("anthropic", "some-model", sdk=sdk), "sys", "user", [TOOL])

    assert reply == Reply("ok", [ToolCall("metrics", {"label": "kernel1_0"})])
    (request,) = sdk.requests
    assert (request["system"], request["messages"]) == ("sys", [{"role": "user", "content": "user"}])
    assert request["tools"] == [{"name": "metrics", "description": "Work of a row.", "input_schema": TOOL.parameters}]


@pytest.mark.parametrize("sdk", [FakeOpenAI(), FakeAnthropic()], ids=["openai", "anthropic"])
def test_a_call_without_tools_sends_no_tool_list(sdk):
    provider = "openai" if isinstance(sdk, FakeOpenAI) else "anthropic"

    llm.chat(llm.Client(provider, "some-model", sdk=sdk), "sys", "user", [])

    assert "tools" not in sdk.requests[-1]
