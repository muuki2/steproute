# Implementation issues: open, outcome-trained, graph-based step router

*Version 0.4, 13 Nov 2026. All human decisions resolved (see table). Read `docs/DESIGN.md` first; this file turns it into GitHub issues.*

## How to use this file

**Audience:** a Claude Code orchestrator session (Fable 5.2) that delegates each issue to a subagent running Opus 5.5 or Sonnet 5.5.

1. Create the labels and milestones listed below, then one GitHub issue per `## Issue` block (title = heading, body = everything under it).
2. Work in dependency order. Issues with no unmet `Depends on` can run in parallel.
3. Give each issue to **one fresh subagent** on the model named in its `Model` line. Do not switch model inside an issue; if the assigned model fails twice, first add a failing test that pins the problem, then raise effort one level, then move the issue up a tier.
4. An issue is done only when every acceptance criterion passes in CI. One PR per issue, referencing it with `Closes #N`.
5. Anything marked **VERIFY** is a claim this document could not confirm. Check it against current upstream docs or source before writing code that depends on it, and record what you found in `docs/INTEGRATION-NOTES.md`.
6. The **Resolved decisions** table is binding. Two approval gates remain and are marked **HUMAN GATE**: stop and ask there; do not guess.
7. Priority is tool first: if work must be sequenced, M1 collectors come before M2 to M4. M4 still runs, subject to the go/no-go in issue 23.

**Model assignment rule used below:** Sonnet 5.5 for well-scoped, spec-driven work with a clear test. Opus 5.5 for design judgment, unfamiliar integration surfaces, ML and evaluation logic.

## Conventions

- Python 3.11+, `uv`, `src/` layout, `pytest`, `ruff`, `mypy --strict` on `src/steproute/core`.
- Package and repo name: `steproute`. Licence: MIT.
- Core (`steproute.core`) has **no third-party runtime dependencies**. Heavier parts are extras: `[proxy]`, `[mcp]`, `[ml]`, `[graph]`.
- Money is USD; prices are per million tokens; time is milliseconds; timestamps are Unix seconds (float).
- No network calls in unit tests. Provider calls are mocked or recorded.
- Never log prompt or completion text unless `capture_text=true` is set explicitly.

### Labels

`area:core` `area:collector` `area:data` `area:eval` `area:ml` `area:graph` `area:docs` `type:feature` `type:spike` `type:experiment` `model:opus` `model:sonnet`

### Milestones

| Milestone | Outcome |
|---|---|
| M0 Foundation | Schema, store, cost model, policy interface, rule baseline |
| M1 Collectors | Real traffic produces traces from proxy, Claude Code, dsh |
| M2 Data | Dense outcome matrix over (task, model, effort); published dataset |
| M3 Baselines | Evaluator plus rule, LightGBM and external baselines |
| M4 Graph | GNN predictor and the cold-start experiment |
| M5 Online | Online updates, drift, ablations |

### Dependency overview

```
1 ─┬─ 2 ─┬─ 3 ─────────────┐
   │     ├─ 4 ─ 5 ─ 6 ─ 7 ─┼─ 8
   │     └─ 9              │
   └─ 10 (spike)           │
7,3 ─ 11 ─┬─ 12 (proxy)    │
          ├─ 13 (MCP) ─ 14 (Claude Code, needs 10)
          └─ 15 (dsh, needs 10)
2 ─ 16 (privacy)
12,17 ─ 18 ─ 19
3 ─ 20 ─┬─ 21 ─┐
        └─ 22 ─┼─ 23 ─ 24 ─ 25 ─ 26
               └─ 27 ─ 28
```

## Resolved decisions

| ID | Decision | Resolution | Affects |
|---|---|---|---|
| D1 | Name and licence | `steproute`, MIT | 1 |
| D2 | Local stack | Ollama on the maintainer's Mac: **M4 Pro, 48 GB unified memory**. Local pool: `qwen3.5:9b` (weak tier), `gpt-oss:20b` (mid tier, adjustable reasoning mapped to effort levels), `qwen3-coder:30b` (strong tier). `devstral:24b` is reserved as the held-out model for the cold-start experiment and must not appear in training data. Load one local model at a time and cap context (start at 32K) so memory stays under about 36 GB. **VERIFY** each tag still exists on ollama.com and passes the tool-call smoke test in issue 4; if one fails, substitute the nearest model of the same tier and record it. vLLM is out of scope for now; keep the proxy generic to any OpenAI-compatible server | 4, 12, 18, 26 |
| D3 | Task set and spend | A SWE-bench-style subset of roughly 100 to 200 multi-step coding tasks with tests. Replay budget is **per provider**: Anthropic API spend is hard-capped at **$80 per calendar month** (the maintainer has $100 of API credits a month; the rest is buffer). DeepSeek is a required part of the pool, capped at $20 for the pilot on a separate DeepSeek API key; if the key is missing, stop and ask the maintainer instead of skipping. Ollama runs are free. The pilot must fit in one month; the full grid needs approval and is spread over later months using resume | 17, 18 |
| D4 | Cloud pool | Claude (Haiku 5.5, Sonnet 5.5, Opus 5.5, Fable 5.2) plus DeepSeek API models. No OpenAI models | 4, 12, 18, 22 |
| D5 | Trace publication | Only traces from benchmark replays are published. Traces from personal Claude Code or dsh usage stay local and may be used for training only | 16, 19 |
| D7 | Billing | Claude Code implementation work runs on the maintainer's subscription usage limits, not on API credits. API credits are reserved for replays. Never point replay runs at the subscription, and never use the API key for implementation work | all |
| D6 | Project target | Both, tool first. Collectors and the cost-aware router ship first; the GNN and experiments follow and may become a paper | ordering, 24 onward |

**Remaining human gates**
- **Gate A (issue 17):** approve the chosen task subset and the estimated cost of the full replay grid.
- **Gate B (issue 23):** go or no-go for M4 after the baseline report.

---

# M0 Foundation

## Issue 1: Repository scaffold and CI

**Labels:** `area:core` `type:feature` `model:sonnet` · **Milestone:** M0 · **Depends on:** none · **Model:** Sonnet 5.5, low effort

**Tasks**
- `pyproject.toml` (uv, src layout, extras `proxy`, `mcp`, `ml`, `graph`, `dev`).
- Package skeleton: `steproute/core`, `steproute/collectors`, `steproute/eval`, `steproute/ml`, `steproute/graph`.
- GitHub Actions: ruff, mypy on core, pytest on 3.11 and 3.12.
- `README.md` with a one-paragraph pitch and a link to `docs/DESIGN.md`; MIT `LICENSE`; `CONTRIBUTING.md`.

**Acceptance criteria**
- `uv sync --extra dev && uv run pytest` passes on a clean clone.
- CI is green on the scaffold PR.
- `import steproute.core` pulls in no third-party module (test asserts this via `sys.modules`).

## Issue 2: Trace schema

**Labels:** `area:core` `type:feature` `model:sonnet` · **Milestone:** M0 · **Depends on:** 1 · **Model:** Sonnet 5.5, medium

Implement `steproute/core/schema.py` with stdlib dataclasses and explicit `to_dict` / `from_dict`. `SCHEMA_VERSION = "0.1"`.

| Type | Fields |
|---|---|
| `Action` (frozen, hashable) | `model: str`, `effort: str = "medium"`, `escalate_to: Action \| None`. Methods `chain()`, `label()`. |
| `ModelProfile` | `id`, `provider`, `local: bool`, `price_in`, `price_out`, `cache_write_mult`, `cache_read_mult`, `context_window`, `efforts: tuple[str, ...]`, `effort_output_mult: dict[str, float]`, `meta: dict` |
| `StepState` | `harness`, `session_id`, `step_idx`, `task_id`, `kind` (`main`, `subagent`, `tool_followup`, `plan`, ...), `prompt_tokens`, `cached_prefix_tokens`, `current_model` (owner of the session cache), `tools: list[str]`, `sensitive: bool`, `source: str` (`replay` or `personal`, default `personal`), `task_text: str \| None`, `features: dict` |
| `Decision` | `decision_id`, `ts`, `state`, `candidates: list[Action]`, `action`, `policy: str`, `propensity: float`, `predicted_success`, `predicted_cost` |
| `Outcome` | `decision_id`, `ts`, `success: bool \| None`, `signals: dict`, `final_model`, `escalated: bool`, `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens`, `cost_usd`, `latency_ms` |

Effort vocabulary: `none`, `low`, `medium`, `high`, `max`. Providers without effort levels use `medium` only.

**Acceptance criteria**
- Round-trip `from_dict(to_dict(x)) == x` for every type, including a three-level cascade.
- Unknown keys on read are ignored with a warning, not an error (forward compatibility).
- `docs/SCHEMA.md` documents every field and gives one example decision and outcome as JSON.
- A JSON Schema file is generated and checked in; a test fails if it drifts from the dataclasses.

## Issue 3: Append-only trace store

**Labels:** `area:core` `type:feature` `model:sonnet` · **Milestone:** M0 · **Depends on:** 2 · **Model:** Sonnet 5.5, low

`TraceStore(path)` writing JSONL records `{"type": "decision" | "outcome", "v": ...}`.

**Tasks**
- `log_decision`, `log_outcome`, `records()`, `joined()` yielding `(Decision, Outcome | None)`; the last outcome per decision wins.
- Safe concurrent appends from several processes (file lock or `O_APPEND` single-write lines).
- Daily file rotation; `joined()` reads across rotated files.
- CLI `steproute traces stats <path>`: counts, join rate, cost total, success rate per model and effort.

**Acceptance criteria**
- Test with 8 processes each appending 1,000 records produces 8,000 valid lines.
- A truncated final line is skipped and reported, not fatal.
- Stats CLI output is covered by a snapshot test.

## Issue 4: Model catalog

**Labels:** `area:core` `type:feature` `model:sonnet` · **Milestone:** M0 · **Depends on:** 2 · **Model:** Sonnet 5.5, low

**Tasks**
- Load `ModelProfile`s from a TOML catalog (stdlib `tomllib`); validate and reject unknown efforts and negative prices.
- Ship `catalogs/default.toml` with the resolved pool: the four Claude tiers, the current DeepSeek API models, and the Ollama models from decision D2 (`qwen3.5:9b`, `gpt-oss:20b`, `qwen3-coder:30b`, plus `devstral:24b` flagged `holdout = true`). **VERIFY** DeepSeek model ids, prices, cache pricing and reasoning controls against DeepSeek's docs, and the Ollama tags, sizes and context limits against ollama.com; record what you found in `docs/INTEGRATION-NOTES.md`.
- Tool-call smoke test per model (`steproute catalog check`): send one request with a tool defined and assert the response carries a structured tool call, not the call as plain text. A model that fails is marked `tools = false` and excluded from agent tasks. Run against Ollama locally; skipped in CI. Starting prices, taken from Anthropic's model-selection post on 13 Nov 2026: Haiku 5.5 $0.10 / $0.50 (prompts up to 100K tokens; $0.50 / $2.50 above), Sonnet 5.5 $2 / $10, Opus 5.5 $4 / $20, Fable 5.2 $8 / $40.
- Support tiered pricing by prompt size (needed for Haiku 5.5).
- Map each provider's reasoning control onto the shared effort vocabulary (Claude effort levels; DeepSeek thinking on/off or equivalent; Ollama models usually `medium` only).
- **VERIFY** current prices, cache read/write multipliers and supported effort levels per model against provider docs. Until verified, mark the values `verified = false` in the catalog and print a warning on load.

**Acceptance criteria**
- Catalog loads, validates and round-trips.
- A test covers the tiered price boundary at exactly 100,000 prompt tokens.
- Loading an unverified catalog emits exactly one warning.

## Issue 5: Action enumeration and hard constraints

**Labels:** `area:core` `type:feature` `model:sonnet` · **Milestone:** M0 · **Depends on:** 4 · **Model:** Sonnet 5.5, medium

**Tasks**
- `enumerate_actions(profiles, max_depth=2)`: all `(model, effort)` singles plus cascades.
- Cascade rules: an escalation step must be strictly stronger than the step before it, meaning the same model at higher effort or a model later in the catalog's `tier` order. No cycles; depth capped.
- `filter_feasible(state, actions, profiles)` applying hard constraints: `sensitive` means local models only at every step of the chain; `prompt_tokens` must fit each model's context window; optional allow and deny lists.
- Raise a typed `NoFeasibleAction` error when the filter leaves nothing.

**Acceptance criteria**
- For a 4-model, 3-effort catalog the count of depth-1 and depth-2 actions matches a hand-computed number stated in the test.
- No enumerated cascade ever escalates to an equal or weaker step (property test).
- A sensitive state never yields an action containing a cloud model.

## Issue 6: Cache-aware cost model

**Labels:** `area:core` `type:feature` `model:opus` · **Milestone:** M0 · **Depends on:** 5 · **Model:** Opus 5.5, medium

`expected_cost(state, action, profiles, est_output_tokens, p_success) -> CostBreakdown`.

**Single attempt**
- Action model owns the cache (`state.current_model == action.model`): `cached_prefix_tokens` billed at `price_in * cache_read_mult`, the remaining prompt at `price_in`.
- Otherwise: the whole prompt billed at `price_in * cache_write_mult`. This is the switch tax.
- Output: `est_output_tokens * effort_output_mult[effort] * price_out`.

**Cascade**
- `cost = c1 + (1 - p1) * c2`, with `c2` evaluated in the state the first attempt leaves behind (`current_model = first model`, `cached_prefix_tokens = prompt_tokens`). Recursive for deeper chains.
- Overall success: `1 - prod(1 - p_i)`.

**Also**
- `CostBreakdown` reports input, cache, output and expected-escalation components separately, plus expected latency if profiles carry a latency estimate.
- `realized_cost(outcome, profile)` computing actual cost from an `Outcome`'s token fields, for use when the provider does not return a cost.
- Local models: price 0 by default, optional per-token compute price in the profile.

**Acceptance criteria**
- Worked examples in `docs/COST-MODEL.md` are reproduced exactly by tests.
- Test: with a warm cache, staying on the current model is cheaper than switching to a model with an identical price.
- Test: a cascade's expected cost lies between the cost of its first step and the sum of all steps, for any `p` in [0, 1] (property test).
- Test: `realized_cost` agrees with `expected_cost` when predictions equal the realized values.

## Issue 7: Policy interface and constrained-argmin policy

**Labels:** `area:core` `type:feature` `model:opus` · **Milestone:** M0 · **Depends on:** 6 · **Model:** Opus 5.5, medium

**Tasks**
- `Predictor` protocol: `predict(state, action_step) -> float` (success probability for a single `(model, effort)` step), plus optional `update(decision, outcome)`.
- `Policy` protocol: `decide(state) -> Decision`, `report(outcome) -> None`.
- `ScoredPolicy(predictor, catalog, tau, epsilon)`: among feasible actions with predicted overall success at least `tau`, pick minimum expected cost. If none reaches `tau`, pick the highest predicted success. With probability `epsilon`, explore uniformly over feasible actions.
- Record the true propensity of the chosen action, accounting for both the greedy and the exploration branch.
- Deterministic under a seed.

**Acceptance criteria**
- With a stub predictor, the chosen action matches a hand-worked example.
- Empirical choice frequencies over 20,000 seeded decisions match logged propensities within tolerance.
- Raising `tau` never lowers the predicted success of the chosen action (property test).

## Issue 8: Rule baseline policy

**Labels:** `area:core` `type:feature` `model:sonnet` · **Milestone:** M0 · **Depends on:** 7 · **Model:** Sonnet 5.5, low

A no-learning policy encoding published model-selection guidance, used as the floor every learned policy must beat.

**Rules**
- Default: mid-tier model at `medium`.
- `kind == "subagent"` with read-only tools (search, read, grep, test run): cheapest tier.
- Cascade shape: same model one effort level up, then next tier at a lower effort.
- Stay on `current_model` unless the rule changes tier (avoids needless switch tax).
- All thresholds in a config file.

**Acceptance criteria**
- Table-driven test of at least 12 states mapping to expected actions.
- Rules live in config; changing a threshold needs no code change.

## Issue 9: Outcome labelling

**Labels:** `area:core` `type:feature` `model:sonnet` · **Milestone:** M0 · **Depends on:** 2 · **Model:** Sonnet 5.5, medium

Turn raw signals into `Outcome.success` with a documented precedence.

**Tasks**
- `label(signals) -> (success | None, confidence, reason)`.
- Precedence: explicit task check (tests, schema validation) > tool or compile error > user rejection or rewrite > retry on the same step > judge score > none.
- Keep raw signals; never overwrite them with the label.
- `docs/LABELS.md` states which signals each collector can supply.

**Acceptance criteria**
- Table-driven tests for every precedence pair.
- A step with no signals yields `success=None`, never `False`.

---

# M1 Collectors

## Issue 10: Spike: verify integration points

**Labels:** `area:collector` `type:spike` `model:opus` · **Milestone:** M1 · **Depends on:** 1 · **Model:** Opus 5.5, high. Research only, no production code.

Write `docs/INTEGRATION-NOTES.md` answering, with links to current upstream docs or source and the date checked:

1. **Claude Code:** which hook events exist and what payload each carries (model, token usage, tool results); how a subagent's model is set (definition frontmatter, per-call parameter, environment variable); whether a hook or MCP tool can change the model of a spawn; whether subagents inherit the parent session's MCP servers (a community README claims they do not).
2. **Claude Code:** where per-request usage including cache read/write tokens can be read (transcript files, hooks, OpenTelemetry).
3. **DeepSeek Harness (dsh):** exact seam for LLM calls (`ctx.llm` is the name in our notes), whether a plugin can wrap or intercept it per step, session-log event schema, fork API, pinned version to target.
4. **LiteLLM:** current custom-callback and custom-router hooks; whether a callback can rewrite the target model and reasoning effort before dispatch; how cache token counts surface.
5. **LLMRouter:** the custom-router plugin contract (`MetaRouter`, `route_single`, `route_batch`) and data format, to judge how thin our adapter can be.
6. **ParetoBandit and OpenSquilla:** licence, install path, and the interface needed to run each as a baseline.

**Acceptance criteria**
- Every question has an answer, a source link and a confidence note, or an explicit "could not determine" with what was tried.
- A short "consequences for issues 12 to 15" section lists any design change required.

## Issue 11: Core service (local HTTP API)

**Labels:** `area:core` `type:feature` `model:sonnet` · **Milestone:** M1 · **Depends on:** 3, 7 · **Model:** Sonnet 5.5, medium

Collectors in other languages need a process to talk to.

**Tasks**
- `steproute serve`: `POST /decide` (StepState in, Decision out), `POST /report` (Outcome in), `GET /models`, `GET /healthz`.
- Policy and catalog chosen by config; traces written through `TraceStore`.
- Binds to localhost by default; optional bearer token.
- Fail-open: if the policy raises or exceeds a time budget (default 50 ms), return the configured default action and log the failure.

**Acceptance criteria**
- Contract tests for each endpoint against the JSON Schema from issue 2.
- A policy that sleeps past the budget triggers the default action.
- p95 `/decide` latency under 10 ms with the rule baseline on a laptop-class machine (benchmark script checked in).

## Issue 12: OpenAI-compatible proxy collector

**Labels:** `area:collector` `type:feature` `model:opus` · **Milestone:** M1 · **Depends on:** 10, 11 · **Model:** Opus 5.5, medium

Route and log for any client that speaks the OpenAI chat API. Upstreams in scope: Ollama (local), Anthropic and DeepSeek (cloud). Keep the upstream layer generic so another OpenAI-compatible server can be added by config.

**Tasks**
- Implement on LiteLLM using the hook identified in issue 10.
- Build `StepState` from the request: session id from a header (fallback: hash of the first messages), step index, prompt tokens, tools, `current_model` and cached prefix tracked per session.
- Apply the decision by rewriting model and reasoning effort; execute cascades when the response fails a configurable check (tool-call parse failure, schema validation, empty output).
- Emit `Outcome` from the response usage; accept late signals through `POST /report`.
- Streaming must pass through without buffering the full response.
- Shadow mode: decide and log, but do not change the request.

**Acceptance criteria**
- Integration test against a mock upstream covering: route, cascade on failure, shadow mode, streaming.
- With Ollama running locally, a documented smoke test produces joined traces. The test is skipped, not failed, when Ollama is absent (CI).
- Session cache state is tracked correctly across a 5-step session that switches model once.

## Issue 13: MCP server

**Labels:** `area:collector` `type:feature` `model:sonnet` · **Milestone:** M1 · **Depends on:** 11 · **Model:** Sonnet 5.5, medium

**Tools**
- `route_task(description, kind, tools, est_size)` returns recommended model, effort, cascade and a one-line reason, plus a `decision_id`.
- `report_outcome(decision_id, success, signals)`.
- `list_models()`.

**Notes**
- The server only advises; it cannot change the host's model. State this in each tool description so the calling model knows to apply the recommendation itself.
- Tool descriptions should tell the agent when to call `route_task` (before spawning a subagent or starting a new task), and to always call `report_outcome`.

**Acceptance criteria**
- Works over stdio with the MCP inspector; a recorded session is checked in as a test fixture.
- Every `route_task` call produces a logged decision; unreported decisions older than a timeout are closed with `success=None`.

## Issue 14: Claude Code integration

**Labels:** `area:collector` `type:feature` `model:opus` · **Milestone:** M1 · **Depends on:** 10, 13 · **Model:** Opus 5.5, high

Design depends on issue 10's findings. Intended shape:

- A Claude Code plugin bundling the MCP server, hook scripts, and subagent definitions per tier.
- Hooks capture state before a subagent spawn and outcomes after tool use and at stop (test results, tool errors, token usage including cache fields).
- The routing decision is applied at subagent-spawn boundaries only, never mid-session on the main thread.
- Shadow mode by default on first install.

**Acceptance criteria**
- Install instructions work on a clean machine in under five minutes.
- A scripted session with three subagent spawns yields three joined decision and outcome pairs with non-zero token counts.
- If issue 10 shows a planned mechanism is impossible, the issue documents the fallback chosen and why.

## Issue 15: dsh plugin

**Labels:** `area:collector` `type:feature` `model:opus` · **Milestone:** M1 · **Depends on:** 10, 11 · **Model:** Opus 5.5, high

TypeScript plugin for DeepSeek Harness, pinned to the version recorded in issue 10.

**Tasks**
- Wrap the LLM seam to call `/decide` per step and apply the model and effort.
- Derive outcomes from the session log: tool errors, test results, approvals and rejections, retries.
- Fail-open to the user's configured model if the core service is unreachable.
- Shadow mode.

**Acceptance criteria**
- Plugin loads in the pinned dsh version; a scripted session yields joined traces.
- Killing the core service mid-session does not interrupt the session.
- **VERIFY** all seam names against dsh source; do not rely on the names in this document.

## Issue 16: Privacy and redaction

**Labels:** `area:core` `type:feature` `model:sonnet` · **Milestone:** M1 · **Depends on:** 2 · **Model:** Sonnet 5.5, medium

**Tasks**
- `capture_text` off by default; when off, `task_text` is never stored, only length and an embedding or hash if configured.
- Sensitive detection: path globs, regexes for secrets and keys, and a user-supplied callback. Sets `StepState.sensitive`.
- Every trace carries a `source` field: `replay` or `personal`. `steproute traces export` includes only `source == "replay"`; there is no flag to export personal traces in this version (decision D5).
- `--scrub` on export: drops text, hashes session and task ids with a salt, buckets timestamps.

**Acceptance criteria**
- A test corpus of seeded secrets is fully removed by scrub.
- A sensitive state is never routed to a cloud model in an end-to-end test through the proxy.
- A test proves a store containing both sources exports zero personal records.

---

# M2 Data

## Issue 17: Task set with automatic checks

**Labels:** `area:data` `type:feature` `model:opus` · **Milestone:** M2 · **Depends on:** 1 · **Model:** Opus 5.5, high

Select the SWE-bench-style subset for the bootstrap dataset (decision D3): roughly 100 to 200 tasks for the full set and a 15 to 25 task pilot slice drawn from it. **VERIFY** which current benchmark variant to use, its licence, and that its harness runs on the maintainer's machine or in containers.

**Requirements**
- Every task has an automatic pass/fail check.
- A spread of difficulty; include tasks the cheapest model solves and tasks only the strongest solves, since the hard tail dominates spend.
- Multi-step agent tasks, not single prompts, so step-level and cache effects exist.
- Licence permits redistribution of tasks and traces.

**Acceptance criteria**
- `docs/TASKSET.md` lists candidates, the choice and the reasons, with a cost estimate for the full replay grid.
- A loader runs one task end to end and returns pass or fail.
- A pilot slice is defined so that the Anthropic share of the pilot grid fits the $80 monthly cap with margin, based on a per-task cost estimate.
- **HUMAN GATE A:** the maintainer approves the subset and the full-grid spend estimate. The pilot may run before this approval; the full grid may not.

## Issue 18: Replay runner

**Labels:** `area:data` `type:feature` `model:opus` · **Milestone:** M2 · **Depends on:** 12, 17 · **Model:** Opus 5.5, medium

Run each task under each `(model, effort)` configuration from the resolved pool (Claude tiers, DeepSeek, Ollama models) to get a dense outcome matrix. All traces written here carry `source = "replay"`.

**Tasks**
- Grid runner with resume, per-run isolation (fresh session, no shared cache between configurations), concurrency limits and a hard spend cap. Caps are per provider and per calendar month: default **Anthropic $80**, **DeepSeek $20**; raising either requires an explicit config value recorded after Gate A. The runner persists month-to-date spend so a resumed run cannot exceed the monthly cap.
- Run order is cheapest first: Ollama, then Haiku, Sonnet and DeepSeek, then Opus, then Fable; within a model, lower effort first. A cap hit then still leaves a usable partial matrix.
- DeepSeek configurations are required. If no DeepSeek key is configured, the runner exits with a clear message before spending anything; ask the maintainer for the key.
- Local runs: one Ollama model loaded at a time, sequential, with the context cap from D2. Models flagged `holdout` are run in a separate pass and written to a separate split.
- Run the pilot slice first and write `docs/results/pilot.md`: realized cost per configuration, success rates, and a revised estimate for the full grid.
- Records per-step traces through the proxy plus the task-level result.
- A dry-run mode that prints the plan and estimated cost.

**Acceptance criteria**
- Interrupting and resuming does not duplicate or lose runs.
- The spend cap stops the run within one task of the limit, counting realized provider cost, and the pilot finishes under the per-provider caps.
- A test simulates a month rollover: spend resets, and a resumed run continues without repeating finished runs.
- A 3-task, 2-configuration smoke grid completes in CI against mock models.

## Issue 19: Dataset release

**Labels:** `area:data` `area:docs` `type:feature` `model:sonnet` · **Milestone:** M2 · **Depends on:** 16, 18 · **Model:** Sonnet 5.5, medium

**Tasks**
- Export scrubbed replay traces only (decision D5) as Parquet plus the JSON Schema.
- Datasheet: collection method, models and versions with dates, known biases, label definitions, licence.
- Fixed train, validation and test splits by task, plus a held-out-model split for the cold-start experiment.

**Acceptance criteria**
- A loader test reads the release and reproduces the summary table in the datasheet.
- No task appears in more than one split.

---

# M3 Baselines and evaluation

## Issue 20: Offline evaluator

**Labels:** `area:eval` `type:feature` `model:opus` · **Milestone:** M3 · **Depends on:** 3, 6 · **Model:** Opus 5.5, high

**Tasks**
- Metrics: cost per solved task, success rate, quality retention versus the strongest single configuration, switch count, p50 and p95 added latency.
- Full-information evaluation on the dense matrix.
- Off-policy evaluation on logged bandit data: inverse propensity scoring and doubly robust, with confidence intervals and a warning when effective sample size is low.
- The single-configuration frontier: every `(model, effort)` alone, plotted as score against spend. Report whether a policy lies beyond it.
- `steproute eval --policy ... --data ...` producing a JSON report and a plot.

**Acceptance criteria**
- On synthetic data with a known optimal policy, the evaluator ranks policies correctly and the off-policy estimates cover the true value at the stated confidence level.
- The report states plainly whether the policy beats every single configuration.

## Issue 21: Tabular predictor baseline

**Labels:** `area:ml` `type:feature` `model:sonnet` · **Milestone:** M3 · **Depends on:** 20 · **Model:** Sonnet 5.5, medium

**Tasks**
- Feature builder from `StepState` and `ModelProfile` (including profile and probe features, so the comparison with the GNN is fair).
- LightGBM `Predictor` with probability calibration.
- Train and evaluate scripts; model card.

**Acceptance criteria**
- Reproducible training from a seed.
- Calibration error reported; evaluator report generated and checked in.

## Issue 22: External baselines

**Labels:** `area:eval` `type:feature` `model:opus` · **Milestone:** M3 · **Depends on:** 10, 20 · **Model:** Opus 5.5, medium

Adapters so these run under our evaluator on our data and our model pool (Claude, DeepSeek, Ollama): ParetoBandit, LLMRouter `graphrouter`, RouteLLM (two-model setting), and OpenSquilla's open LightGBM router if its features can be mapped.

**Acceptance criteria**
- Each baseline either produces an evaluator report or has a written note explaining why it could not be run fairly.
- Versions and commits pinned.

## Issue 23: Baseline report

**Labels:** `area:eval` `area:docs` `type:experiment` `model:opus` · **Milestone:** M3 · **Depends on:** 8, 21, 22 · **Model:** Opus 5.5, medium

`docs/results/baselines.md`: all baselines on the same splits, with the single-configuration frontier.

**Acceptance criteria**
- One table, one frontier plot, reproducible from one command.
- A stated go or no-go: does any learned baseline beat every single configuration?
- **HUMAN GATE B:** report the result to the maintainer and wait for approval before starting M4.

---

# M4 Graph

## Issue 24: Graph construction

**Labels:** `area:graph` `type:feature` `model:opus` · **Milestone:** M4 · **Depends on:** 23 (Gate B) · **Model:** Opus 5.5, high

**Tasks**
- Heterogeneous graph (PyTorch Geometric): model-config nodes (profile and probe features), step nodes (state features, optional text embedding), tool nodes, harness nodes.
- Edges: step to model-config with outcome, step to step in sequence, step to tool, step to harness.
- Time-decayed edge weights.
- Inductive insertion: add a new model node from a profile and probe results without rebuilding.

**Acceptance criteria**
- Graph builds from the released dataset; node and edge counts match trace statistics.
- A test adds an unseen model node and obtains predictions without retraining.

## Issue 25: GNN edge predictor

**Labels:** `area:graph` `area:ml` `type:feature` `model:opus` · **Milestone:** M4 · **Depends on:** 24 · **Model:** Opus 5.5, high

**Tasks**
- Edge-scoring model for `P(success | step, model-config)` trained with edge masking; implements the `Predictor` protocol.
- Calibration; uncertainty estimate.
- Ablations: without tool nodes, without harness nodes, without probe features.

**Acceptance criteria**
- Evaluator report on the standard splits alongside issue 21's baseline.
- Training is reproducible from a seed and fits on a single consumer GPU or CPU within a documented time.

## Issue 26: Cold-start experiment

**Labels:** `area:graph` `type:experiment` `model:opus` · **Milestone:** M4 · **Depends on:** 22, 25 · **Model:** Opus 5.5, high

The experiment that justifies the GNN.

**Protocol**
- Hold out one model entirely from training. Introduce it with its profile and results on a probe set of size k (k in 0, 5, 20, 50). The primary held-out model is `devstral:24b` (never seen in training, per D2); the leave-one-out repeats below use the other pool models.
- Stream requests; measure cumulative regret and requests until regret per step falls below a threshold.
- Compare: GNN, LightGBM with the same profile features, ParetoBandit.
- Repeat with each model held out; report mean and spread.

**Acceptance criteria**
- `docs/results/cold-start.md` with the regret curves and a plain verdict.
- If the GNN does not beat LightGBM with the same features, say so and recommend dropping it.

## Issue 27: Distillation and serving latency

**Labels:** `area:graph` `area:ml` `type:feature` `model:sonnet` · **Milestone:** M4 · **Depends on:** 25 · **Model:** Sonnet 5.5, medium

**Tasks**
- Distil the GNN to an MLP over precomputed node embeddings; refresh embeddings offline.
- Benchmark `/decide` latency for GNN, distilled model and LightGBM.

**Acceptance criteria**
- Distilled model within a stated tolerance of the GNN on the evaluator.
- p95 `/decide` under 10 ms on CPU.

---

# M5 Online learning and ablations

## Issue 28: Online updates and drift

**Labels:** `area:ml` `type:feature` `model:opus` · **Milestone:** M5 · **Depends on:** 27 · **Model:** Opus 5.5, high

**Tasks**
- Incremental updates from reported outcomes; exploration schedule; propensities kept correct.
- Drift detection per model (success rate shift) with faster decay when triggered.
- Simulated drift test: degrade one model mid-stream.

**Acceptance criteria**
- Under simulated degradation the policy shifts traffic away within a stated number of requests.
- Off-policy estimates remain valid after online updates (checked on synthetic data).

## Issue 29: Ablation experiments

**Labels:** `area:eval` `type:experiment` `model:opus` · **Milestone:** M5 · **Depends on:** 26 · **Model:** Opus 5.5, medium

1. Action space: model only, versus model and effort, versus model, effort and cascade.
2. Cache awareness: switch tax on versus off, in multi-step sessions.
3. Cross-harness transfer: train on one harness's traces, evaluate on another's.

**Acceptance criteria**
- `docs/results/ablations.md` with one table per ablation and a one-line conclusion each.

## Issue 30: User documentation and release

**Labels:** `area:docs` `type:feature` `model:sonnet` · **Milestone:** M5 · **Depends on:** 14, 15, 23 · **Model:** Sonnet 5.5, medium

Quick starts for the proxy, Claude Code and dsh; configuration reference; how to contribute traces; versioned release with changelog.

**Acceptance criteria**
- Each quick start is exercised by a scripted test or a recorded walkthrough.
- A new user reaches shadow-mode traces in under ten minutes following the docs.
