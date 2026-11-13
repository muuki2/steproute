# Trace schema

Schema version: `0.1` (`steproute.core.SCHEMA_VERSION`). Implemented in `src/steproute/core/schema.py`.

Two append-only record types, joined by `decision_id`: a **Decision** (logged before the step runs) and an **Outcome** (logged after). Both are plain stdlib dataclasses with `to_dict()` / `from_dict()`.

Conventions: money is USD, prices are per million tokens, time is milliseconds, timestamps are Unix seconds (float).

## Serialisation rules

- `to_dict` emits every field. `None` becomes JSON `null`; tuples become lists. Output is JSON-native, so `from_dict(json.loads(json.dumps(to_dict(x)))) == x`.
- `from_dict` ignores unknown keys and emits a `steproute.core.UnknownFieldWarning` (a `UserWarning` subclass) so a newer writer does not break an older reader. Missing required keys raise `ValueError`. Wrongly typed values raise `TypeError`.
- Free-form dicts (`features`, `signals`, `meta`) must hold JSON-native values. Tuples inside them come back as lists.
- The JSON Schema does not forbid additional properties, for the same forward-compatibility reason.
- Fields marked "required" below have no default; every other field may be omitted on read.

## Effort vocabulary

`none`, `low`, `medium`, `high`, `xhigh`, `max`, weakest to strongest (`steproute.core.EFFORTS`; `effort_rank(e)` gives the position). Providers without effort levels use `medium` only.

This differs from the original issue text, which listed `none`, `low`, `medium`, `high`, `max`. `xhigh` was added on purpose because Claude models expose it.

## Action

A routing choice. Frozen and hashable.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `model` | string | required | Model id. May contain `:` `.` `@` `/`. |
| `effort` | string | `"medium"` | One of the effort vocabulary. Anything else raises `ValueError`. |
| `escalate_to` | Action or null | `null` | Action tried only if this one fails. Chains nest to any depth. |

`chain()` returns the single-step actions in order, each with `escalate_to=None`. `label()` returns a compact string such as `a (high) -> b (low)` for display and grouping; do not parse it back.

## ModelProfile

| Field | Type | Default | Meaning |
|---|---|---|---|
| `id` | string | required | Model id, as used in `Action.model`. |
| `provider` | string | required | Provider name. |
| `price_in` | number | required | USD per million input tokens. |
| `price_out` | number | required | USD per million output tokens. |
| `context_window` | integer | required | Context window in tokens. |
| `local` | boolean | `false` | Runs locally (no per-token charge). |
| `cache_write_mult` | number | `1.0` | Prompt-cache write price as a multiple of `price_in`. |
| `cache_read_mult` | number | `1.0` | Prompt-cache read price as a multiple of `price_in`. |
| `efforts` | array of string | `["medium"]` | Effort levels the model supports. |
| `effort_output_mult` | object of number | `{}` | Output-token multiplier per effort level. |
| `meta` | object | `{}` | Free-form extras. |

## StepState

Harness-neutral description of one routable step.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `harness` | string | required | Harness name (for example `claude-code`, `proxy`, `dsh`). |
| `session_id` | string | required | Session id. |
| `step_idx` | integer | required | Step index within the session. |
| `kind` | string | required | Open string. Known values: `main`, `subagent`, `tool_followup`, `plan`. |
| `prompt_tokens` | integer | required | Prompt size in tokens. |
| `task_id` | string or null | `null` | Task id, when the harness has one. |
| `cached_prefix_tokens` | integer | `0` | Tokens of the prompt already in the session cache. |
| `current_model` | string or null | `null` | Model that owns the session cache. |
| `tools` | array of string | `[]` | Tools in scope. |
| `sensitive` | boolean | `false` | Step must not leave the machine. |
| `source` | string | `"personal"` | `replay` or `personal`; anything else raises `ValueError`. |
| `task_text` | string or null | `null` | Optional, so traces can be shared without content. |
| `features` | object | `{}` | Free-form extra features. |

## Decision

| Field | Type | Default | Meaning |
|---|---|---|---|
| `decision_id` | string | required | Join key. `new_decision_id()` makes one. |
| `ts` | number | required | Unix seconds. |
| `state` | StepState | required | The state the policy saw. |
| `candidates` | array of Action | required | Actions the policy chose among. |
| `action` | Action | required | The chosen action. |
| `policy` | string | required | Policy name. |
| `propensity` | number | required | Probability the policy gave the chosen action. |
| `predicted_success` | number or null | `null` | Predicted success probability of the chosen action. |
| `predicted_cost` | number or null | `null` | Predicted cost in USD. |

## Outcome

| Field | Type | Default | Meaning |
|---|---|---|---|
| `decision_id` | string | required | The decision this belongs to. |
| `ts` | number | required | Unix seconds. |
| `success` | boolean or null | required | Overall success; `null` if unknown. |
| `signals` | object | `{}` | Raw signals (tests passed, tool error, retry, user rejection, judge score). |
| `final_model` | string or null | `null` | Model that produced the final answer. |
| `escalated` | boolean | `false` | Whether the cascade escalated. |
| `input_tokens` | integer | `0` | Uncached input tokens. |
| `output_tokens` | integer | `0` | Output tokens. |
| `cache_read_tokens` | integer | `0` | Tokens read from the prompt cache. |
| `cache_write_tokens` | integer | `0` | Tokens written to the prompt cache. |
| `cost_usd` | number or null | `null` | Cost in USD; `null` means the provider reported none. |
| `latency_ms` | number or null | `null` | Latency in milliseconds. |

## JSON Schema

The generated schema is `src/steproute/core/trace.schema.json` and ships in the package. Load it with `steproute.core.schema_gen.load_json_schema()`. It has `$defs` for `Action`, `ModelProfile`, `StepState`, `Decision` and `Outcome`. A test fails if it drifts from the dataclasses. After changing a dataclass, regenerate it:

```
uv run python -m steproute.core.schema_gen --write
```

## Examples

A decision with a two-step cascade:

```json example=decision
{
  "decision_id": "3f2b8c1e9a7d4e5f8b6a1c2d3e4f5a6b",
  "ts": 1767225600.25,
  "state": {
    "harness": "claude-code",
    "session_id": "sess-42",
    "step_idx": 7,
    "kind": "tool_followup",
    "prompt_tokens": 18500,
    "task_id": "task-9",
    "cached_prefix_tokens": 17000,
    "current_model": "claude-sonnet-4-5",
    "tools": ["Bash", "Edit"],
    "sensitive": false,
    "source": "personal",
    "task_text": null,
    "features": {"repo_lang": "python"}
  },
  "candidates": [
    {"model": "claude-sonnet-4-5", "effort": "medium", "escalate_to": null},
    {
      "model": "claude-sonnet-4-5",
      "effort": "low",
      "escalate_to": {"model": "claude-opus-4-1", "effort": "low", "escalate_to": null}
    }
  ],
  "action": {
    "model": "claude-sonnet-4-5",
    "effort": "low",
    "escalate_to": {"model": "claude-opus-4-1", "effort": "low", "escalate_to": null}
  },
  "policy": "rule-baseline",
  "propensity": 0.8,
  "predicted_success": 0.91,
  "predicted_cost": 0.0123
}
```

The matching outcome:

```json example=outcome
{
  "decision_id": "3f2b8c1e9a7d4e5f8b6a1c2d3e4f5a6b",
  "ts": 1767225612.75,
  "success": true,
  "signals": {"tests_passed": true, "retries": 0},
  "final_model": "claude-sonnet-4-5",
  "escalated": false,
  "input_tokens": 1500,
  "output_tokens": 420,
  "cache_read_tokens": 17000,
  "cache_write_tokens": 0,
  "cost_usd": 0.0118,
  "latency_ms": 8400.0
}
```
