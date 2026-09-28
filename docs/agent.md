# Agent loop

[Overview](../README.md) · feedback: [7 Feedback](stages/7-feedback.md)

`nestforge/agent/` is a minimal loop that lets one model drive stages 2 to 5 through the `Session`. It is built on
one principle: what the model reads between attempts, a concise analyzed report and a compact prompt, matters more
than more tools, skills or history.

- **One call per turn.** Every turn is one system + user message. The prompt is rebuilt from the task, the stage
  goal and epoch, the current view (tree, kernels with devices and dependencies, or kernel sources), the latest
  feedback report, one line per action tried so far, and the full results of the last calls. No chat history grows.
- **Few tools.** Stage 2: `list_moves`, `apply_move`, `metrics`. Stage 3: `define_scope`. Stage 4: `place`.
  Stage 5: `set_kernel_source`, `metrics`. A bad call returns an error line for the next turn.
- **Defaults finish the stage.** When the model stops calling tools or runs out of turns, the stage's
  deterministic default completes what it left.
- **Feedback.** Stage 5 starts from the [feedback report](stages/7-feedback.md) and gets a new one after each
  kernel change. `analyst=True` adds one model call that rewrites each report into guidance.

```python
import functools
from nestforge.agent import llm, loop

client = llm.Client("anthropic", "claude-opus-5-5")          # or llm.Client("openai", model, base_url)
log = loop.run(session, functools.partial(llm.chat, client), task, turns=loop.TURNS, analyst=True)
```

`llm.Client("openai", model, base_url=...)` also serves OpenAI-compatible servers such as vLLM and SGLang. The
SDKs are the optional `agent` extra: `pip install -e ".[agent]"`.
