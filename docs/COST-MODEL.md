# Cost model

`steproute.core.cost` prices routing actions. Standard library only. Cost is computed, not learned (DESIGN section 1): everything here is arithmetic over catalog prices, token counts and who owns the prompt cache. The router chooses `argmin expected_cost subject to P(success) >= tau`, so any optimism in this file about a switch or a cascade would make the router over-choose it. Every rule below is written to never understate a cost when an input is unknown.

```python
from steproute.core import Action, StepState, expected_cost, realized_cost
from steproute.core.catalog import load_default_catalog

catalog = load_default_catalog()
state = StepState(
    "proxy",
    "s1",
    7,
    "main",
    prompt_tokens=50_000,
    cached_prefix_tokens=45_000,
    current_model="claude-opus-5-5",
    current_effort="medium",
)
b = expected_cost(
    state, Action("claude-opus-5-5", "medium"), catalog, est_output_tokens=2_000, p_success=0.9
)
b.total_usd  # 0.074
```

Units: USD; prices per million tokens; latency in milliseconds.

## What is assumed and what is measured

- **Prices, cache multipliers, the Haiku tier and the 512-token Anthropic cache minimum** come from provider docs (`docs/INTEGRATION-NOTES.md` section 7, checked 2026-11-13).
- **`effort_output_mult` is 1.0 for every effort in the default catalog. That is a placeholder, not a measurement.** Until the pilot replay (issue 18) measures it, higher effort costs the same as lower effort in this model, apart from cache effects.
- **`est_output_tokens` is supplied by the caller.** The cost model does not predict output length. The worked examples use round numbers.
- **`p_success` is supplied by the caller** (a predictor, issue 7). The cost model only combines the probabilities.
- **Latency.** No profile in the default catalog carries a latency estimate, because nothing has been measured, so `latency_ms` is `None` for the default catalog. Example 6 uses illustrative numbers.

## Token convention

One convention for `Outcome` and for the planned counts (`AttemptUsage`). Collectors normalise to it.

| Field | Meaning | Billed at |
|---|---|---|
| `input_tokens` | Prompt tokens neither read from the cache nor billed as a cache write | `price_in` |
| `cache_read_tokens` | Prompt tokens read from the cache | `price_in * cache_read_mult` |
| `cache_write_tokens` | Prompt tokens billed as cache writes | `price_in * cache_write_mult` |
| `output_tokens` | Output tokens, reasoning included | `price_out` |

The three prompt fields are disjoint, and their sum is the full prompt length, which selects the price tier.

How each provider maps to it:

| Provider usage | `input_tokens` | `cache_read_tokens` | `cache_write_tokens` |
|---|---|---|---|
| Anthropic Messages API (`input_tokens` already excludes cache tokens) | `input_tokens` | `cache_read_input_tokens` | `cache_creation_input_tokens` |
| DeepSeek, LiteLLM, any OpenAI-compatible server incl. Ollama `/v1` (`prompt_tokens` includes cached tokens) | `prompt_tokens - prompt_tokens_details.cached_tokens` | `prompt_tokens_details.cached_tokens` | `0` |
| dsh session log | `inputTokens` | `cacheReadTokens` | `0` |

Providers without a write charge report 0 writes: a miss there is billed at `price_in`, and since their `cache_write_mult` is 1.0 it makes no difference whether a miss is counted as input or as a write.

## Pricing one request

`price_usage(profile, usage) -> UsageCost` is the only pricing function. With `(price_in, price_out) = profile.prices_for(prompt)` and `prompt = input + read + write`:

```
input_usd       = (input_tokens + cache_write_tokens) * price_in
cache_read_usd  = cache_read_tokens  * price_in * cache_read_mult
cache_write_usd = cache_write_tokens * price_in * (cache_write_mult - 1)     # the premium only
output_usd      = output_tokens * price_out
total_usd       = input_usd + cache_read_usd + cache_write_usd + output_usd  # all / 1e6
```

So `input_usd` is every prompt token not served from the cache at the base price, and `cache_write_usd` is only the premium for writing. On DeepSeek and Ollama the premium is 0. A `cache_write_mult` below 1 is refused (`CostModelError`): no provider has a write discount, and allowing one would make the premium negative.

**Price tiers.** The tier is chosen by the full prompt, cache reads and writes included, and the tier's prices apply to the whole request, output included (Anthropic pricing footnote). Exactly 100,000 tokens on Haiku 5.5 is still the lower tier.

## One attempt

`planned_usage(state, step, profile, est_output_tokens)` predicts the counts of running `step = (model, effort)` in `state`. `P = state.prompt_tokens`, `C = state.cached_prefix_tokens`.

1. **Below the cache minimum** (`P < profile.cache_min_tokens`, 512 on Anthropic): nothing is read or written. `input = P`.
2. **The step owns the cache**: `read = C`, `write = P - C`.
3. **Otherwise**: `write = P`. This is the switch tax.
4. Output: `est_output_tokens * effort_output_mult[effort]`.

**Cache ownership is per (model, effort).** The owner is `(state.current_model, state.current_effort)`. A step owns the cache if it runs the same model and either

- the same effort, or
- a profile with `effort_change_keeps_cache = true` (for example an Anthropic proxy that changes effort with the per-message beta).

An **unknown owner effort** (`current_effort = None`) counts as a match only for a model that supports a single effort, since there the effort cannot have differed. Otherwise the cache is treated as lost. That can overstate the cost of staying, but never understates the cost of an effort change. Collectors know the effort they sent and should fill `current_effort`.

**The switch tax by provider shape.** No provider name appears in the code; the multipliers carry the difference. Compared with staying warm, a switch to an equally priced model costs `C * price_in * (cache_write_mult - cache_read_mult)` more:

- **Anthropic** (write 1.25, read 0.025 to 0.1): the lost read discount plus the write premium on the cached prefix.
- **DeepSeek** (write 1.0, read 0.02 or 0.033): the lost read discount only.
- **Ollama** (price 0): nothing in money. The switch cost is latency: the reload (`latency_switch_ms`) plus prefilling the whole prompt.

## After a failed attempt

`state_after_failure(state, step, profile)` is the state the next step of a cascade runs in:

- `current_model = step.model`, `current_effort = step.effort`;
- `cached_prefix_tokens = P`, or 0 if `P < profile.cache_min_tokens`;
- `failed_attempts` gains `step`;
- everything else, including `prompt_tokens`, is unchanged.

The next attempt therefore resends the same prompt: the failed output is not appended.

## Cascades

For `action.chain() = (s1, ..., sn)`, with `c_i` the cost of `s_i` in the state the failed attempts before it leave behind and `p_i` its success probability:

```
total = c1 + (1 - p1) * (c2 + (1 - p2) * (c3 + ...))
max   = c1 + c2 + ... + cn                # every step runs
P(success) = 1 - prod(1 - p_i)
```

`c1 <= total <= max` for any `p_i` in [0, 1], and the code folds both sums in the same order, so the bound also holds exactly in floating point. A same-model step at a higher effort is cheap only if the cache survives the effort change. A step to another model always pays the full switch tax, but only in the failure branch.

**What `p_i` means.** `p_i` is the probability that step `i` succeeds **given that it runs**, which means given that every earlier step failed. With conditional `p_i`, `1 - prod(1 - p_i)` is exact. Failures are correlated: a task that defeats one model is harder for the next. So plugging in each model's unconditional success rate as if the steps were independent **overstates** cascade success and understates expected cost. The interface lets a predictor condition on the failure:

- `p_success` may be a callable `(state, step) -> float`. It is called with the state each step actually runs in, so for step 2 `state.failed_attempts == [s1]` and `state.current_model == s1.model`. A predictor can use these as features.
- `p_success` may be a sequence with one probability per step, which the caller states as conditional.
- A single number is accepted only for single-step actions. For a cascade it raises, because one number cannot say which probability it is.

`est_output_tokens` may also be one number or one per step.

## CostBreakdown

| Field | Meaning |
|---|---|
| `input_usd`, `cache_read_usd`, `cache_write_usd`, `output_usd` | The **first attempt**'s components. It is always paid. `cache_write_usd` is the write premium only. |
| `cache_usd` (property) | `cache_read_usd + cache_write_usd` |
| `first_attempt_usd` | Sum of the four |
| `escalation_usd` | **Expectation over the failure branch**: `(1 - p1) * (c2 + (1 - p2) * (...))` |
| `total_usd` | `first_attempt_usd + escalation_usd` |
| `max_usd` | Cost if every step runs; use it for spend caps |
| `p_success` | `1 - prod(1 - p_i)` |
| `first_attempt_latency_ms`, `latency_ms` | First-step latency and expected latency (the same fold as cost). `None` unless every profile involved has a latency estimate |
| `attempts` | Per step: `step`, `warm`, `usage`, `cost` (a `UsageCost`), `latency_ms`, `p_success`, `reach_prob` (probability the step runs). Each `cost` is conditional on the step running |

## Latency

Optional `ModelProfile` fields, all absent in the default catalog:

- `latency_base_ms`, `latency_ms_per_input_token` and `latency_ms_per_output_token` are set together or not at all (the catalog rejects a partial set).
- `latency_switch_ms` is optional and defaults to 0.

```
latency = latency_base_ms
        + latency_ms_per_input_token  * (input_tokens + cache_write_tokens)    # prefill of uncached tokens
        + latency_ms_per_output_token * output_tokens
        + latency_switch_ms   if step.model != state.current_model             # e.g. Ollama reload
```

`latency_switch_ms` is how a local reload is expressed, and for Ollama it is the whole switch cost. It applies when the model changes, including when no model is known to be loaded (`current_model = None`). An effort change on the same model is not a reload: gpt-oss changing `think` rewrites the system header, which the prefill term already charges as a full miss.

## Local models and compute price

Ollama profiles have `price_in = price_out = 0`, so their money cost is 0. To charge for local compute, set non-zero `price_in` and `price_out` on the local profile. Nothing else changes, because the formulas never look at `local`. A KV-cache hit skips prefill compute, so a compute-priced local profile should also set `cache_read_mult` near 0. The default 1.0 would charge cached tokens as if they were recomputed. Example 6 shows this end to end.

## Realized cost

`realized_cost(outcome, profile)` prices a **single-attempt** `Outcome` with `price_usage`. Use it when the provider reports no cost, and always for LiteLLM, whose price map lacks Fable 5.2 and ignores DeepSeek off-peak pricing.

- It **refuses an escalated outcome** (`CostModelError`). The token fields of an escalated outcome may sum attempts on different models, and summing prompts across attempts would also pick the wrong tier (two 60K Haiku attempts are not one 120K request).
- Collectors that run a cascade price each attempt with `price_usage(profile_i, AttemptUsage(...))` and write the sum into `Outcome.cost_usd`. The token fields hold the sum over attempts and are for statistics only.
- It also refuses a profile whose id differs from `outcome.final_model`.

When the realized token counts equal the planned ones, `realized_cost` equals the first attempt's `total_usd`, because both call `price_usage`.

## Errors

`CostModelError` (a `ValueError`) is raised for:

- a model not in the catalog, or an effort the model does not list;
- a partly filled `effort_output_mult` that lacks the effort being priced. An empty table means 1.0 for every effort (unmeasured); a table that lists some efforts but not this one is treated as a mistake, and the catalog loader rejects it up front;
- a probability outside [0, 1] or NaN, from a number, a sequence or a callable;
- a sequence of the wrong length, or a single `p_success` for a cascade;
- negative or non-finite token counts, prices or multipliers, or `cache_write_mult < 1`;
- `cached_prefix_tokens > prompt_tokens`;
- `realized_cost` on an escalated outcome or on a mismatched profile.

A zero prompt is valid: only output is charged. Context-window fit is not checked here; it is a hard constraint, applied by `filter_feasible`.

## Worked examples

All prices come from the default catalog, with Anthropic 5-minute cache writes and DeepSeek peak prices. `tests/test_cost.py` recomputes every number in the result blocks and fails if any differs by more than 1e-12 USD. The arithmetic shows `x * price` with prices per million tokens.

### Example 1: warm stay versus switch on Anthropic

The session is on Opus 5.5 at `medium`. Prompt 50,000 tokens, of which 45,000 are cached. Estimated output 2,000 tokens.

- **Stay on Opus 5.5 (medium), warm:**
  - read 45,000 × 4 × 0.05 = $0.009
  - input 5,000 × 4 = $0.020
  - write premium 5,000 × 4 × 0.25 = $0.005
  - output 2,000 × 20 = $0.040
  - total **$0.074**
- **Switch to Sonnet 5.5 (medium), cold:**
  - input 50,000 × 2 = $0.100
  - write premium 50,000 × 2 × 0.25 = $0.025
  - output 2,000 × 10 = $0.020
  - total **$0.145**

Sonnet's prices are half of Opus's, yet switching costs almost twice as much as staying. The switch tax is the lost read discount and the write premium on the 45,000 cached tokens. Switching to a model with Opus's prices would cost $0.29; the tests check this with a twin profile (acceptance criterion 2).

```text cost-example=anthropic-stay-vs-switch
stay.input_usd = 0.02
stay.cache_read_usd = 0.009
stay.cache_write_usd = 0.005
stay.output_usd = 0.04
stay.total_usd = 0.074
switch.input_usd = 0.1
switch.cache_read_usd = 0.0
switch.cache_write_usd = 0.025
switch.output_usd = 0.02
switch.total_usd = 0.145
```

### Example 2: the same on DeepSeek, which has no write premium

The session is on `deepseek-flash` at `high`. Same token counts as example 1.

- **Stay on flash, warm:**
  - read 45,000 × 0.30 × 0.02 = $0.00027
  - input 5,000 × 0.30 = $0.0015
  - write premium $0
  - output 2,000 × 1.20 = $0.0024
  - total **$0.00417**
- **Switch to `deepseek-v4-pro`, cold:**
  - input 50,000 × 1.32 = $0.066
  - output 2,000 × 3.96 = $0.00792
  - total **$0.07392**

Against an equally priced model, the switch tax here is only the lost hit discount: 45,000 × 0.30 × (1 − 0.02) = $0.01323, with no premium term.

```text cost-example=deepseek-stay-vs-switch
stay.input_usd = 0.0015
stay.cache_read_usd = 0.00027
stay.cache_write_usd = 0.0
stay.output_usd = 0.0024
stay.total_usd = 0.00417
switch.input_usd = 0.066
switch.cache_write_usd = 0.0
switch.output_usd = 0.00792
switch.total_usd = 0.07392
flash_switch_tax_usd = 0.01323
```

### Example 3: the Haiku 5.5 100K boundary

Haiku 5.5 at `medium` is warm, with 90,000 tokens cached and 1,000 output tokens. The two cases differ by one prompt token.

- **Prompt 100,000 (lower tier, $0.10 / $0.50):**
  - read 90,000 × 0.10 × 0.1 = $0.0009
  - input 10,000 × 0.10 = $0.001
  - premium 10,000 × 0.10 × 0.25 = $0.00025
  - output 1,000 × 0.50 = $0.0005
  - total **$0.00265**
- **Prompt 100,001 (upper tier, $0.50 / $2.50 on the whole request):**
  - read 90,000 × 0.50 × 0.1 = $0.0045
  - input 10,001 × 0.50 = $0.0050005
  - premium 10,001 × 0.50 × 0.25 = $0.001250125
  - output 1,000 × 2.50 = $0.0025
  - total **$0.013250625**

The 90,000 cache-read tokens count toward the threshold, and the upper tier reprices them too. One extra token makes the request 5 times as expensive.

```text cost-example=haiku-tier-boundary
at_100000.total_usd = 0.00265
at_100000.price_in = 0.1
at_100001.cache_read_usd = 0.0045
at_100001.input_usd = 0.0050005
at_100001.cache_write_usd = 0.001250125
at_100001.output_usd = 0.0025
at_100001.total_usd = 0.013250625
at_100001.price_in = 0.5
```

### Example 4: same model, higher effort, with and without `effort_change_keeps_cache`

The action is Sonnet 5.5 `low`, then Sonnet 5.5 `high`. The session is on Sonnet `low`. Prompt 40,000 tokens with 36,000 cached, output 1,500 per attempt. The conditional probabilities are p1 = 0.7 and p2 = 0.6.

- **First attempt (warm):**
  - read 36,000 × 2 × 0.05 = $0.0036
  - input 4,000 × 2 = $0.008
  - premium 4,000 × 2 × 0.25 = $0.002
  - output 1,500 × 10 = $0.015
  - c1 = **$0.0286**

  Afterwards the cache holds all 40,000 tokens and is owned by (Sonnet, `low`).
- **Default catalog (`effort_change_keeps_cache = false`).** The effort change loses the cache, so the second attempt is cold:
  - input 40,000 × 2 = $0.08
  - premium $0.02
  - output $0.015
  - c2 = $0.115
  - total = 0.0286 + 0.3 × 0.115 = **$0.0631**; max $0.1436
- **With `effort_change_keeps_cache = true`.** The second attempt reads all 40,000 tokens:
  - read 40,000 × 2 × 0.05 = $0.004
  - output $0.015
  - c2 = $0.019
  - total = 0.0286 + 0.3 × 0.019 = **$0.0343**; max $0.0476

Overall success is 1 − 0.3 × 0.4 = 0.88 in both cases. The flag decides whether the effort ladder costs 2.2 times the first attempt or 1.2 times it.

```text cost-example=effort-cascade
no_keep.first_attempt_usd = 0.0286
no_keep.c2_usd = 0.115
no_keep.escalation_usd = 0.0345
no_keep.total_usd = 0.0631
no_keep.max_usd = 0.1436
keep.c2_usd = 0.019
keep.escalation_usd = 0.0057
keep.total_usd = 0.0343
keep.max_usd = 0.0476
p_success = 0.88
```

### Example 5: a cross-model cascade

The action is Haiku 5.5 `medium`, then Opus 5.5 `medium`, at a subagent's first request: prompt 20,000 tokens, nothing cached, no owner (INTEGRATION-NOTES, issue 14 item 5). Output is 1,000 per attempt, with p1 = 0.6 and p2 = 0.85.

- **Haiku, cold:**
  - input 20,000 × 0.10 = $0.002
  - premium $0.0005
  - output 1,000 × 0.50 = $0.0005
  - c1 = **$0.003**
- **Opus, cold (the cache is Haiku's):**
  - input 20,000 × 4 = $0.08
  - premium $0.02
  - output 1,000 × 20 = $0.02
  - c2 = $0.12
- **Cascade:**
  - total = 0.003 + 0.4 × 0.12 = **$0.051**; max $0.123
  - P(success) = 1 − 0.4 × 0.15 = 0.94

Opus alone costs $0.12 with p = 0.85. The cascade is cheaper and more likely to succeed, but only if 0.85 really is Opus's success rate *on the tasks Haiku failed*. If failures are correlated, the true p2 is lower, and so is the cascade's success.

```text cost-example=cross-model-cascade
first_attempt_usd = 0.003
c2_usd = 0.12
escalation_usd = 0.048
total_usd = 0.051
max_usd = 0.123
p_success = 0.94
opus_alone_total_usd = 0.12
```

### Example 6: a local model with a compute price and reload latency (illustrative numbers)

This example uses `qwen3-coder:30b` with **made-up, unmeasured** values, to show the mechanics end to end:

- `price_in = 0.02`, `price_out = 0.08`, `cache_read_mult = 0.0`;
- `latency_base_ms = 200`, `latency_ms_per_input_token = 0.5`, `latency_ms_per_output_token = 25`, `latency_switch_ms = 8000`.

Prompt 12,000 tokens with 10,000 cached, output 800.

- **Switch from gpt-oss:20b:**
  - cost: 12,000 × 0.02 + 800 × 0.08 = **$0.000304**
  - latency: 200 + 0.5 × 12,000 + 25 × 800 + 8,000 = **34,200 ms**
- **Already loaded** (owner `qwen3-coder:30b` with `current_effort = None`, which counts as owning the cache because the model supports a single effort):
  - cost: 0 + 2,000 × 0.02 + 800 × 0.08 = **$0.000104**
  - latency: 200 + 0.5 × 2,000 + 25 × 800 = **21,200 ms**

With the default price of 0, both costs are $0 and only the latency differs.

```text cost-example=local-compute-price
switch.total_usd = 0.000304
switch.latency_ms = 34200.0
stay.total_usd = 0.000104
stay.latency_ms = 21200.0
```

## Design decisions

1. **The uncached remainder of a warm prompt is billed at the write rate.** After the attempt the whole prompt is cached (the cascade's next state says so), and on Anthropic that write costs `price_in * cache_write_mult`. *Rejected:* billing the remainder at plain `price_in`, as the issue text says. That gives the staying model a free cache write, so the model would contradict its own after-state. Staying still beats switching under either rule (criterion 2): `C * (read - write) < 0` whenever `C > 0` and `read < write`.
2. **The cache owner is (model, effort)**, with `effort_change_keeps_cache` as the escape hatch, and an unknown owner effort counts only for single-effort models. *Rejected:* treating an unknown effort as a match, which is optimistic about effort changes on Anthropic. Also rejected: defaulting it to `meta.default_effort`, which not every profile has.
3. **Conditional success probabilities**, with a callable that sees `failed_attempts`. *Rejected:* one independent probability per model, which overstates cascade success.
4. **The cache minimum is modelled** (`cache_min_tokens`, 512 on Anthropic). Below it nothing is cached, so there is no write premium, and the next state has nothing cached. *Rejected:* ignoring it, which would charge small prompts a premium they never pay and credit cascades with a cache that never exists.
5. **`realized_cost` is single-attempt only**, and `price_usage` is the per-attempt building block. *Rejected:* pricing summed token fields with the final model's profile, which is wrong whenever the attempts used different models or the summed prompt crosses a tier.
6. **The write premium is reported separately** from base input, so `cache_write_usd` is 0 on DeepSeek and the components agree whichever way a collector classifies DeepSeek misses.

## Differences from DESIGN section 5

| DESIGN section 5 / issue 6 says | This module does | Why |
|---|---|---|
| The cache is owned by the model | Owned by (model, effort), unless `effort_change_keeps_cache` | Anthropic top-level effort and thinking changes invalidate the message cache, and a gpt-oss level change rewrites the system header (notes 7a, 7c) |
| On a switch, the whole prompt is billed at the cache-write rate | It is, but the rate is `price_in * cache_write_mult` from the profile, which is 1.0 on DeepSeek and the price is 0 on Ollama | The write premium exists only for Anthropic with caching on (notes, issue 6). Ollama's switch cost is latency (`latency_switch_ms`) |
| The uncached remainder of a warm prompt is billed at the input rate | At the write rate | It is written to the cache (decision 1) |
| The prompt size picks the price tier (implicit) | The full prompt, cache reads and writes included, picks the tier, and the tier applies to the whole request | Anthropic pricing footnote (notes 7a) |
| After a failed attempt: `current_model` = first model, `cached_prefix_tokens = prompt_tokens` | Also `current_effort` and `failed_attempts`, and nothing is cached below `cache_min_tokens` | Ownership is per effort, and predictors need to condition on the failure |
| Overall success `1 - prod(1 - p_i)` | The same, stated as exact only for conditional `p_i` | Correlated failures |
| A latency term can be set in the profile | Four optional fields, all absent by default; `latency_ms = None` without them | Nothing has been measured |

## Limitations

- **Effort output multipliers are placeholders (1.0).** Until they are measured, higher effort looks no more expensive in output.
- **Output length is the caller's estimate.** Reasoning tokens are part of it.
- **One cache owner.** `StepState` tracks only the last owner. A provider may still hold an older model's cache (Anthropic caches per model for 5 minutes), so switching back to it may be cheaper than this model says.
- **Cache writes are assumed to cover the whole prompt**, meaning a breakpoint at the end of the prompt. This is the usual agent-harness setup, and the issue 12 proxy should place one there; it is not verified per harness. A harness that caches only a fixed prefix pays less than predicted for the write, and has less cached afterwards.
- **Cache expiry is not modelled.** The collector must set `cached_prefix_tokens = 0` once the TTL (5 minutes on Anthropic, best effort on DeepSeek, `keep_alive` on Ollama) may have passed.
- **Only the 5-minute Anthropic write is priced.** A collector that uses the 1-hour TTL must price with a profile whose `cache_write_mult` is 2.0 (`meta.cache_write_1h_mult`).
- **DeepSeek is priced at peak rates.** Off-peak requests cost half, so `realized_cost` overstates them unless the collector passes an off-peak profile (prices in `meta`).
- **DeepSeek cache scope and the effect of toggling thinking are undocumented.** The conservative default `effort_change_keeps_cache = false` treats an effort change as a miss. Its minimum cacheable length is unknown, so it is 0.
- **Partial-prefix minimums.** `cache_min_tokens` is compared with the whole prompt. The finer Anthropic rule (a minimum per breakpoint) is not modelled.
- **Cost is not monotone in prompt length at the cache minimum.** Crossing 512 tokens turns on caching, which can make a cascade's second step cheaper. This matches the provider's behaviour.
- **A cascade retry resends the same prompt.** If a harness appends the failed attempt's output, the next prompt is longer than modelled.
- **The latency model is linear** and ignores queueing, rate limits and network variance.
