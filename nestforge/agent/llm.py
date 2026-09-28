# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""One chat-with-tools call over the OpenAI API (and OpenAI-compatible servers such as vLLM or SGLang) or the
Anthropic API. Each call is a single system + user message: the loop rebuilds the prompt every turn."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

Provider = Literal["openai", "anthropic"]


@dataclass(frozen=True, slots=True)
class Tool:
    """A callable the model may request, with a JSON-schema of its arguments."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Reply:
    text: str
    calls: list[ToolCall]


@dataclass(slots=True)
class Client:
    """Which API and model to ask; ``base_url`` points the OpenAI client at a local server."""

    provider: Provider
    model: str
    base_url: str | None = None
    max_tokens: int = 4096
    sdk: Any = field(default=None, repr=False)  # created on the first call

    def connect(self) -> Any:
        if self.sdk is None:
            # deferred: the agent dependencies are optional
            if self.provider == "openai":
                import openai

                self.sdk = openai.OpenAI(base_url=self.base_url)
            else:
                import anthropic

                self.sdk = anthropic.Anthropic(base_url=self.base_url)
        return self.sdk


def chat(client: Client, system: str, prompt: str, tools: list[Tool]) -> Reply:
    """Ask the model once; its text and the tool calls it requested."""
    sdk = client.connect()
    if client.provider == "openai":
        specs = [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in tools
        ]
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        # an empty tool list is left out: some OpenAI-compatible servers refuse one
        response = sdk.chat.completions.create(
            model=client.model, messages=messages, **({"tools": specs} if specs else {})
        )
        message = response.choices[0].message
        calls = [ToolCall(c.function.name, json.loads(c.function.arguments or "{}")) for c in message.tool_calls or []]
        return Reply(message.content or "", calls)
    specs = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]
    response = sdk.messages.create(
        model=client.model,
        max_tokens=client.max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        **({"tools": specs} if specs else {}),
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return Reply(text, [ToolCall(b.name, dict(b.input)) for b in response.content if b.type == "tool_use"])
