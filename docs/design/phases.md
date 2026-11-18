# Phases owned by NestForge

[Designs](README.md) · [Stages](../stages.md)

Status: built (2026-11-18), owner's spec of 2026-11-18 (items P1 and P2 of `notes/phases-plan.md` in the workspace).

## Goal

The agent works on a program in phases. NestForge owns them: nothing of NestForge lives in HPCAgent-Bench, whose
generic `[phases]` feature NestForge no longer uses. The canonical parallel form runs once when the session starts;
then:

| Phase | What the agent does | Default when the agent does not run it |
|---|---|---|
| `structured` | moves on the whole program: fusion, tiling, layouts, array precision | the default moves |
| `scopes` | which parts become kernels, where they run, their schedules | the default scopes |
| `scoped` | moves inside the kernels: vectorization, staging, library implementations, floating-point flags | nothing |
| `kernels` | each kernel's code and its compiler configuration | nothing (finalize builds every kernel) |

Leaving `scopes`, whoever runs it, places every kernel: what no scope covers gets its default scope first.

## Tools

- `advance_phase`: leave the phase for the agent's next one; the phases between run their defaults. The answer names
  the new phase, what ran on the way, and carries the new phase's guide.
- `undo_phase`: take the current phase back completely, with the defaults run on the way into it; the agent is in the
  previous phase with the program as it left it. Refused in the agent's first phase (the canonical form runs once).
- `return_to_phase phase=<p>`: `undo_phase` until phase `p`, for instance after a score.

The names differ from the bench's own `next_phase` tool, which the agent's tool registry already holds.

## Moves of the phases (P5, P6)

- `apply_moves moves=[...]`: several moves in order, all or nothing; after each, the program is built and run at the
  smallest preset against the reference. The answer names the first move refused or wrong, its substep (`apply` or
  `check`) and why; each applied move stays its own change for `rollback`.
- In `scopes`, `apply_scope_move` places kernels: `run-on-cpu`, `run-on-gpu` (labels: the kernels), and
  `run-on-cpu-persistent`, `run-on-gpu-persistent`, which merge consecutive kernels into one (their scopes unset and
  set again as one) and run its longest run of blocks with doalls as one persistent kernel (MegakernelLoops). Each is
  one change for `rollback`. The `place` tool is gone; leaving `scopes` still places what the agent did not.

## Mechanism

- The session keeps the phase and, for every phase the agent left, the program as it left it (`Session.saved()`,
  the checkpoint `rollback` uses), taken before the defaults of the way out run. `undo_phase` restores it. Rollback
  takes back changes of the current phase only: leaving or undoing a phase clears its history.
- `NESTFORGE_AGENT_PHASES` in a packet's `env` names the phases the agent runs (unset: all); every call carries it,
  and the route fixes it at the episode's first call, as it does for the switched-off move families.
- The server refuses a tool outside its phases (`server.TOOL_PHASES`, the same table `wire.py` labels each tool with):
  moves on the program in `structured`, moves with a `scope` in `scoped`, placement and schedule kinds in `scopes`.
  A journal's repeat is not gated: it passed the gate when the agent made it.
- Each packet's prompt carries the guide of its agent's first phase (`prompt/phase-<phase>.yaml`); the phase tools'
  answers carry the guide of the phase they reach (`prompt/phases/<phase>.md`).

## Later (P3)

Context management: what the agent keeps of a phase it left. The exit checkpoints are the seam: a summary of each phase
(the tree it left, its kernels) can be built from them.
