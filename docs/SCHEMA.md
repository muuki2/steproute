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
| `meta` | object | `{}` | Free-form extras. Documented keys are listed in `docs/CATALOG.md`. |
| `tier` | integer | `0` | Cascade strength order across the whole pool; higher is stronger, ties allowed. An editable assumption, not a measurement. |
| `tools` | boolean | `true` | `false` means the model failed the tool-call check and is excluded from agent tasks. |
| `holdout` | boolean | `false` | Reserved for the cold-start experiment; never used for training or normal routing. |
| `verified` | boolean | `false` | Price, cache multipliers, context window and effort levels were checked against provider docs. |
| `price_tiers` | array of PriceTier | `[]` | Prompt-size price tiers. `price_in` / `price_out` apply up to and including the first threshold. |
| `effort_map` | object of string | `{}` | Shared effort level to the provider's native reasoning value. A supported effort absent from the map means "send no reasoning parameter". |
| `effort_change_keeps_cache` | boolean | `false` | Whether changing effort between two requests keeps the provider's prompt cache. |
| `cache_min_tokens` | integer | `0` | Shortest prompt the provider caches; below it nothing is read or written (512 on Anthropic). |
| `latency_base_ms` | number or null | `null` | Fixed per-request latency estimate, in ms. Set together with the next two, or not at all. |
| `latency_ms_per_input_token` | number or null | `null` | Prefill latency per prompt token not read from the cache, in ms. |
| `latency_ms_per_output_token` | number or null | `null` | Decode latency per output token, in ms. |
| `latency_switch_ms` | number or null | `null` | Extra latency when the step's model is not the session's current model (for example a local model reload), in ms. `null` means 0. |

No profile in the default catalog carries a latency estimate; nothing has been measured. The cost model (`docs/COST-MODEL.md`) returns no latency for a profile without one.

`ModelProfile.prices_for(prompt_tokens)` returns the `(price_in, price_out)` in force. The prompt length counts all input tokens, cache reads and writes included, and a request over a threshold pays the higher rate on the whole request. Exactly the threshold is still the lower tier.

## PriceTier

One prompt-size price tier of a `ModelProfile`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `above_tokens` | integer | required | Prompt-token threshold; the tier applies to prompts strictly longer than this. |
| `price_in` | number | required | USD per million input tokens above the threshold. |
| `price_out` | number | required | USD per million output tokens above the threshold. |

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
| `current_effort` | string or null | `null` | Effort the cache owner ran at; the cache owner is the pair (`current_model`, `current_effort`). `null` means unknown, which the cost model treats as "cache lost on this model" unless the model supports a single effort or has `effort_change_keeps_cache`. Must be in the effort vocabulary. |
| `failed_attempts` | array of Action | `[]` | Single-step actions already tried and failed for this step, oldest first. Empty for a fresh step. Set in the state a cascade's later step runs in, so a predictor can condition on the failure. |
| `tools` | array of string | `[]` | Tools in scope. |
| `sensitive` | boolean | `false` | Step must not leave the machine. |
| `source` | string | `"personal"` | `replay` or `personal`; anything else raises `ValueError`. |
| `task_text` | string or null | `null` | Optional, so traces can be shared without content. |
| `features` | object | `{}` | Free-form extra features. Documented keys: `paths` (list of strings, the file paths the step touches; read by sensitive detection, any other type ignored, dropped by `--scrub` exports like every list), `est_output_tokens` (number, overrides the policy's output-token estimate, `docs/POLICY.md`), and `task_text_len` / `task_text_hash` (written by the store when text is not captured, `docs/PRIVACY.md`). |

## Decision

| Field | Type | Default | Meaning |
|---|---|---|---|
| `decision_id` | string | required | Join key. `new_decision_id()` makes one. |
| `ts` | number | required | Unix seconds. |
| `state` | StepState | required | The state the policy saw. |
| `candidates` | array of Action | required | Actions the policy chose among. May be empty when `candidate_set` is set: the list was left out to keep the record small (see `CandidateSet`). |
| `action` | Action | required | The chosen action. |
| `policy` | string | required | Policy name. |
| `propensity` | number | required | Probability the policy gave the chosen action. |
| `predicted_success` | number or null | `null` | Predicted success probability of the chosen action. |
| `predicted_cost` | number or null | `null` | Predicted cost in USD. |
| `candidate_set` | CandidateSet or null | `null` | Compact, verifiable definition of the candidate set. `null` means `candidates` is the complete explicit list (records written before this field existed, and policies such as `FixedPolicy`). |

Reading the candidate set: `steproute.core.rebuild_candidates(decision, catalog)` returns `candidates` when `candidate_set` is `null`, and otherwise rebuilds the list from `state` and the catalog and verifies it (fingerprint, count, and the explicit list when present), raising `CandidateSetMismatch` if anything disagrees. Readers must not treat an empty `candidates` as "no candidates" when `candidate_set` is set.

## CandidateSet

Added in schema `0.1` as a defaulted field of `Decision`, so older records stay valid. Frozen. Details in `docs/POLICY.md`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `fingerprint` | string | required | `v1:` plus 32 hex characters: SHA-256 over the catalog fields that shape the set (per model, in file order: id, tier, efforts, local, context window, tools, holdout), the parameters below, and the enumeration version. Prices and `meta` are not covered. |
| `count` | integer | required | Number of candidates (the feasible set the policy chose among). |
| `max_depth` | integer | required | `enumerate_actions` depth. |
| `include_holdout` | boolean | `false` | Whether holdout models were allowed. |
| `allow` | array of string or null | `null` | `filter_feasible` allow list, sorted; `null` means no allow list. |
| `deny` | array of string | `[]` | `filter_feasible` deny list, sorted. |

The set is `filter_feasible(state, enumerate_actions(catalog, max_depth, include_holdout=...), catalog, allow=..., deny=..., include_holdout=...)` for the decision's own `state`, in enumeration order. Rebuilding it therefore needs the logged state to be the one the policy saw: run `mark_sensitive` before `decide`, not only on the write path, or the rebuilt set can differ (and verification then fails rather than returning a wrong list).

A compact decision (state abbreviated):

```json
{
  "decision_id": "9c1d0e2f3a4b5c6d7e8f9a0b1c2d3e4f",
  "ts": 1767225600.25,
  "state": {"harness": "proxy", "session_id": "sess-7", "step_idx": 3, "kind": "main", "prompt_tokens": 20000},
  "candidates": [],
  "action": {"model": "claude-haiku-5-5", "effort": "low", "escalate_to": null},
  "policy": "scored",
  "propensity": 0.9500900900900901,
  "predicted_success": 0.91,
  "predicted_cost": 0.0021,
  "candidate_set": {
    "fingerprint": "v1:57e9bb21b743682d97bb2933ebc37b5c",
    "count": 555,
    "max_depth": 2,
    "include_holdout": false,
    "allow": null,
    "deny": []
  }
}
```

## Outcome

| Field | Type | Default | Meaning |
|---|---|---|---|
| `decision_id` | string | required | The decision this belongs to. |
| `ts` | number | required | Unix seconds. |
| `success` | boolean or null | required | Overall success; `null` if unknown. |
| `signals` | object | `{}` | Raw signals (tests passed, tool error, retry, user rejection, judge score). |
| `final_model` | string or null | `null` | Model that produced the final answer. |
| `escalated` | boolean | `false` | Whether the cascade escalated. |
| `input_tokens` | integer | `0` | Prompt tokens neither read from the cache nor billed as cache writes (billed at `price_in`). |
| `output_tokens` | integer | `0` | Output tokens, reasoning included. |
| `cache_read_tokens` | integer | `0` | Prompt tokens read from the prompt cache. |
| `cache_write_tokens` | integer | `0` | Prompt tokens billed as cache writes; `0` on providers without a write charge. |
| `cost_usd` | number or null | `null` | Cost in USD; `null` means the provider reported none. For an escalated outcome, the sum of each attempt priced separately. |
| `latency_ms` | number or null | `null` | Latency in milliseconds. |

### Token fields

One convention, whatever the provider; collectors normalise to it. `input_tokens`, `cache_read_tokens` and `cache_write_tokens` are disjoint, and their sum is the full prompt length (which selects a price tier).

- Anthropic: `input_tokens` (already excludes cache tokens), `cache_read_input_tokens`, `cache_creation_input_tokens`, used as they are.
- DeepSeek, LiteLLM and other OpenAI-compatible servers (Ollama `/v1` too), whose `prompt_tokens` includes cached tokens: `input_tokens = prompt_tokens - cached_tokens`, `cache_read_tokens = cached_tokens`, `cache_write_tokens = 0`.
- dsh: `inputTokens` and `cacheReadTokens`, with `cache_write_tokens = 0`.

For an escalated outcome the token fields are sums over attempts and serve statistics only. `steproute.core.realized_cost` refuses to price them; see `docs/COST-MODEL.md`.

## JSON Schema

The generated schema is `src/steproute/core/trace.schema.json` and ships in the package. Load it with `steproute.core.schema_gen.load_json_schema()`. It has `$defs` for `Action`, `PriceTier`, `ModelProfile`, `StepState`, `CandidateSet`, `Decision` and `Outcome`. A test fails if it drifts from the dataclasses. After changing a dataclass, regenerate it:

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
    "current_effort": "medium",
    "failed_attempts": [],
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
  "predicted_cost": 0.0123,
  "candidate_set": null
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
