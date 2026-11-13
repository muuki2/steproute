# Outcome labelling

Collectors record raw signals in `Outcome.signals`. `steproute.core.labels` turns them into the
single `Outcome.success` label with one documented precedence, so every collector labels the
same way. Weak signals do not masquerade as strong ones, and absence of evidence stays
`None`: a step nobody checked is unlabelled, not failed.

```python
from steproute.core import label, apply_label

success, confidence, reason = label({"tests_passed": False, "judge_score": 0.9})
# (False, 0.95, "task_check:tests_passed")
```

## Precedence

Strongest first. **Only the highest tier that has usable evidence decides.** A lower tier never
overrides it, even when it disagrees.

| # | Tier (`reason` prefix) | Signals | Can say | Default confidence |
|---|---|---|---|---|
| 1 | `task_check` | `tests_passed`, `schema_valid` | success or failure | 0.95 |
| 2 | `error` | `tool_error`, `compile_error` | failure only | 0.85 |
| 3 | `user` | `user_rejected`, `user_rewrote`, `user_approved` | success or failure | 0.70 |
| 4 | `retry` | `retried`, `retry_count` | failure only | 0.40 |
| 5 | `judge` | `judge_score` | success or failure | up to 0.30 |
| - | none | nothing usable | `None` | 0.0 (`reason` is `no_signals`) |

`reason` is `"<tier>:<signal>"`, for example `task_check:tests_passed`, or `no_signals`.

### Confidence is a placeholder

The confidence numbers are **untuned placeholders that only encode an ordering**. They decrease
down the precedence list and nothing more. They are not measured, not calibrated probabilities,
and must not be used as weights without calibrating them against real data. They live in the
frozen dataclass `LabelConfig` together with `judge_threshold`. For the judge tier the
confidence scales linearly with the distance of the score from the threshold: 0 at the
threshold, the tier maximum (0.30) at the nearer extreme (0 or 1).

## Signal vocabulary

Keys of `Outcome.signals`. Unknown keys are ignored. A value of the wrong type counts as
absent for that signal: it never raises and never yields `False`. Note that `bool` is a
subclass of `int` in Python, so `1` is not a valid flag and `True` is not a valid count or
score.

| Key | Type | Meaning |
|---|---|---|
| `tests_passed` | bool | Test suite result for the step. |
| `schema_valid` | bool | Output validated against a schema. |
| `tool_error` | bool | A tool call errored. True is failure; False or absent is no evidence. |
| `compile_error` | bool | Build or compile failed. True is failure; False or absent is no evidence. |
| `user_rejected` | bool | The user rejected the edit or action. True is failure. |
| `user_rewrote` | bool | The user rewrote the output. True is failure. |
| `user_approved` | bool | The user approved. True is success. |
| `retried` | bool | The same step was retried. True is failure (weak). |
| `retry_count` | int | Number of retries; above 0 means the same as `retried=True`. |
| `judge_score` | float in [0, 1] | LLM judge score; success when at or above the threshold (default 0.5). NaN, infinity and out-of-range values are absent. |

Within-tier rules:

- Task check: if any explicit check is False the label is failure; otherwise if any is True it
  is success.
- Error: only True counts (failure). A False `tool_error` does not mean success, and a lower
  tier may still decide.
- User: rejection or rewrite wins over approval.
- Retry: `retried=True` or `retry_count > 0` is failure. `retried=False` or a zero count is no
  evidence.

## Raw signals are kept

`label` only reads. `apply_label(outcome)` returns a new `Outcome` with `success` set and
`signals` copied with the same keys and values; no label metadata is added to `signals`.

When `success` is already set, `apply_label` leaves it alone by default (a collector may
have set it from information the signals do not carry). `apply_label(outcome, overwrite=True)`
recomputes it from the signals; if the signals give no evidence, the existing value is kept
rather than erased.

## Task-level versus step-level signals

A step-level signal describes one step: a tool error on this call, a user rejecting this edit.
A task-level signal describes a whole task: a replay run executes a multi-step task and then
runs the tests once, giving a single `tests_passed` for the task.

Recording the task result on every step of the task as `tests_passed` would label each step as
if the tests had checked that step alone, and would mark steps that were fine as failures.
The labeller cannot tell the difference, so the collector must decide where the signal goes:
attach `tests_passed` or `schema_valid` to the step it actually verifies (typically the final
step of the task), and keep the task-level result in the task record, not copied onto earlier
steps. Earlier steps then fall through to their own step-level signals or stay `None`.
Crediting earlier steps from a task-level result is a separate modelling choice, out of scope
here.

## Which collector supplies what (planned)

Planned, not implemented: the collectors do not exist yet. This table is derived from issues
12 to 15 and 18 and is subject to the integration spike (issue 10), which may change what each
harness can expose. "Late" means reported after the step through `POST /report` or
`report_outcome`.

| Signal | OpenAI proxy (12) | MCP server (13) | Claude Code hooks (14) | dsh plugin (15) | Replay runner (18) |
|---|---|---|---|---|---|
| `tests_passed` | late, caller-supplied | late, caller-supplied | yes: the last test command a subagent ran (Bash exit status) | from session log | yes, task-level (final step) |
| `schema_valid` | yes, cascade check on response | late, caller-supplied | no | no | yes, if the task defines a schema |
| `tool_error` | from tool-call parse failure | late, caller-supplied | yes: any tool call of the subagent's run errored | yes, from session log | yes |
| `compile_error` | no | late, caller-supplied | no (not implemented) | from session log | yes, if the task builds |
| `user_rejected` / `user_rewrote` / `user_approved` | no (no UI) | late, caller-supplied | no: denials are counted (`tool_denied_count`) but who denied is not known | yes, from approvals and rejections | no (no user) |
| `retried` / `retry_count` | yes, repeated request in a session | late, caller-supplied | no | yes, from session log | no |
| `judge_score` | optional, late | optional, late | no | optional | optional |

The MCP server only receives what the calling agent reports through `report_outcome`, so every
signal there is caller-supplied. The Claude Code column is what the plugin of issue 14 reports
(`docs/CLAUDE-CODE.md`); one outcome covers a whole subagent run.
