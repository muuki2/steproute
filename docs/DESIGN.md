# Design: an open, outcome-trained, graph-based step router

*Draft 0.4, 14 Nov 2026. Package name `steproute`, MIT licence. The companion note `gnn-llm-router-findings.md` named in earlier drafts is not in this repository.*

This is the design overview: what was decided and why. The detail lives in the per-area documents named in each section, and in the code under `src/steproute/`. `docs/ISSUES.md` holds the issue list, the milestones M0 to M5, the resolved decisions D1 to D7 and the two human gates.

## 0. Status

Three labels are used in this document.

- **Built and tested.** The code exists and its tests pass. The tests use mocks, loopback servers and synthetic data. No test calls a paid provider.
- **Built, not verified on real paid runs.** The code exists, but the behaviour that matters (real prices, real cache behaviour, real step counts) has not been seen from a provider. The pilot replay is what verifies it. It has not run (14 Nov 2026).
- **Planned.** No code, or only a stub.

| Part | Status | Detail |
|---|---|---|
| Trace schema, append-only store, privacy and export, outcome labelling | Built and tested | `docs/SCHEMA.md`, `docs/PRIVACY.md`, `docs/LABELS.md` |
| Model catalog, action enumeration, cost model (cache-aware, time-of-day prices) | Built and tested. Effort output multipliers are placeholders (1.0), not measured | `docs/CATALOG.md`, `docs/ACTIONS.md`, `docs/COST-MODEL.md` |
| Policy interface, constrained-argmin policy, rule baseline | Built and tested. The rule defaults are hand-written, not measured | `docs/POLICY.md`, `docs/RULES.md` |
| Local service, OpenAI-compatible proxy, MCP server | Built and tested against mocks | `docs/SERVICE.md`, `docs/PROXY.md`, `docs/MCP.md` |
| Claude Code plugin, dsh plugin | Built and tested. Run in real Claude Code sessions (2.1.323) and on the pinned dsh (0.2.2-rc.3) with a mock upstream | `docs/CLAUDE-CODE.md`, `docs/DSH.md` |
| Task set (150 tasks, 20-task pilot), task checks | Built and tested. Images were validated with the gold and baseline patches on the maintainer's Mac; five tasks whose gold patch fails were replaced (section 8) | `docs/TASKSET.md` |
| Offline evaluator (frontier verdict, off-policy estimators) | Built and tested on synthetic data | `docs/EVALUATION.md` |
| Tabular predictor (LightGBM) | Built and tested on synthetic data only. No real result | `docs/MODEL-CARD-TABULAR.md` |
| Replay runner: free half (Docker, harness, proxy, Ollama) | Built and run on real infrastructure with local models | `docs/REPLAY.md` |
| Replay runner: paid path, spend caps, ledger, cache canary | Built, not verified on real paid runs | `docs/REPLAY.md` |
| Pilot replay, dataset release, baseline report (Gate B) | Planned | `docs/ISSUES.md` issues 18, 19, 23 |
| External baselines (LLMRouter graphrouter, ParetoBandit, OpenSquilla, RouteLLM) | Planned. No code | `docs/INTEGRATION-NOTES.md` sections 5 and 6 (reading notes only) |
| Graph predictor and experiment 1 (M4) | Planned. `steproute.graph` holds only a docstring | issues 24 to 27 |
| Online updates, drift handling, experiments 2 to 4 (M5) | Planned | issues 28, 29 |

## 1. Goal

One router that covers five gaps found in the open-source market, built so each part is useful on its own:

| # | Gap | Where it lives in this design |
|---|---|---|
| 1 | Open step-level outcome data | Trace schema + collectors (section 3) |
| 2 | Learned (model, effort, cascade) action space | `Action` type + expected-cost model (sections 4, 5) |
| 3 | Inductive cold start for new models | Graph predictor over model profiles (section 6, planned) |
| 4 | Cache-aware decisions | Cache ownership inside the cost model (section 5) |
| 5 | Cross-harness transfer | Harness-neutral state + harness as a graph node (sections 3, 6) |

The five aggregate cleanly because of one split: **cost is computed, quality is learned.**

- Cost of an action in a given state is arithmetic over prices, token counts and cache ownership. Gaps 2 and 4 are mostly solved here, with no training.
- Probability of success for (state, action) is the only learned quantity. Gaps 3 and 5 are properties of *how* that predictor is built. Gap 1 is the data it trains on.

So the system is: `decide = argmin expected_cost  subject to  P(success) >= tau`, with a pluggable `P(success)` predictor (`docs/POLICY.md`). Rule baseline, LightGBM, bandit and GNN all plug into the same slot and are compared on the same traces. Only the rule baseline and the LightGBM predictor exist today.

## 1a. Resolved decisions

The binding table is in `docs/ISSUES.md` (D1 to D7). Summary of what this design rests on:

- **Target:** tool first, research second. Collectors and the cost-aware router ship before the GNN work.
- **Model pool:** the pool is the default catalog, `src/steproute/catalogs/default.toml`, with prices checked on 13 Nov 2026.
  - Claude: `claude-haiku-5-5`, `claude-sonnet-5-5`, `claude-opus-5-5`, `claude-fable-5-2`. Efforts `low`, `medium`, `high`, `xhigh`, `max` for each.
  - DeepSeek: `deepseek-flash` and `deepseek-v4-pro`. Efforts `none`, `low`, `high`, `max`.
  - Local models served by Ollama on an M4 Pro Mac with 48 GB: `qwen3.5:9b` (`none`, `medium`), `gpt-oss:20b` (`low`, `medium`, `high`), `qwen3-coder:30b` (`medium` only). `devstral:24b` is held out for the cold-start experiment. Local context is capped at 32,768 tokens (D2).
  - That is 35 configurations: 20 Claude, 8 DeepSeek, 7 local (the held-out model included).
- **Bootstrap data:** a 150-task subset of SWE-bench Verified with arm64 images and a 20-task pilot drawn from it (`docs/TASKSET.md`). The maintainer approved the subset and the sparse grid (option B) at Gate A on 13 Nov 2026. No run beyond the pilot is approved.
- **Spend caps (D3):** Anthropic API spend is capped at $80 per calendar month. DeepSeek has $10 in total, funded once, and runs only in its off-peak hours (the first draft said $20 for DeepSeek). The runner enforces both (`docs/REPLAY.md`).
- **Publication:** only benchmark replay traces are published. Personal usage traces stay local.
- **Licence:** MIT.

## 2. What we reuse instead of building

- **Transport:** the LiteLLM SDK, pinned exactly (`litellm==1.109.1`). The proxy is our own small server on top of it, not LiteLLM's proxy, because the proxy must own the retry loop for cascades (`docs/PROXY.md`, decision 1). LiteLLM's cost estimates are not used: the catalog prices every call.
- **Benchmark harness:** the official SWE-bench harness, run per task on arm64 images (`docs/TASKSET.md`).
- **Replay scaffold:** our own bash-only loop with mini-SWE-agent's step shape, not mini-SWE-agent itself (`docs/REPLAY.md`, decision 1; the reasons are checkable there).
- **Baselines (planned, not built):** ParetoBandit (online bandit with budgets), OpenSquilla's open LightGBM router, RouteLLM, and LLMRouter's `graphrouter`. `docs/INTEGRATION-NOTES.md` sections 5 and 6 record what their code and data formats are.

We write: the trace schema, the collectors, the cost model, the evaluator, the replay runner and the graph predictor.

## 3. Trace schema (gap 1, gap 5) - built and tested

Two append-only record types, joined by `decision_id` (`docs/SCHEMA.md`, schema version 0.1):

- **Decision**: harness-neutral `StepState`, the candidate actions, the chosen action, the policy name and the **propensity** (probability the policy gave the chosen action). Propensity is what makes the logs usable for off-policy training and evaluation.
- **Outcome**: success signal, raw signals (tests passed, tool error, retry, user rejection, judge score), token usage including cache read and write, cost, latency, whether the cascade escalated.

`StepState` carries only what every harness can supply: harness name, session, task and step ids, step kind, prompt tokens, cached prefix tokens, which model and effort own the cache (`current_model`, `current_effort`), the actions that already failed in this step (`failed_attempts`), tools in scope, a `sensitive` flag, a `source` (`replay` or `personal`) and a free `features` dict. Task text is optional and dropped by default, so traces can be shared without content (`docs/PRIVACY.md`).

Success labels follow one documented precedence (`docs/LABELS.md`). A step nobody checked is unlabelled, not failed.

Collectors, each a thin adapter emitting the same records:

1. **OpenAI-compatible proxy** (`steproute proxy`, `docs/PROXY.md`). Modes `route` and `shadow`. In shadow mode the request runs unchanged and the router's decision is only logged. Built and tested against loopback mocks. No live provider has been called.
2. **Claude Code plugin** (`plugins/claude-code/`, `docs/CLAUDE-CODE.md`) and an **MCP server** (`steproute mcp`, `route_task` and `report_outcome`, `docs/MCP.md`). The MCP server only advises: it cannot change the host's model. The plugin is a Claude Code mod that sets the model of each subagent in `route` mode; the default is `shadow`.
3. **dsh plugin** (`plugins/dsh/`, `docs/DSH.md`), pinned to dsh 0.2.2-rc.3, which has no stable release.

All three call one local decision service (`steproute serve`, `docs/SERVICE.md`), or embed it. The service fails open: if the policy fails, it answers with a configured fallback and logs why.

## 4. Action space (gap 2) - built and tested

`Action = (model, effort, escalate_to)`, where `escalate_to` is another `Action` tried only if the first attempt fails (`docs/ACTIONS.md`). This expresses the ladder from Anthropic's model-selection guidance: same model at higher effort first, then the next model up at lower effort.

Overall success of a cascade is `1 - prod(1 - p_i)` only when each `p_i` is the probability that step `i` succeeds given that the earlier steps failed. Failures are correlated: a task that defeats one model is harder for the next. Plugging in unconditional rates overstates cascade success. The predictor interface therefore sees the failed attempts (`StepState.failed_attempts`), and a single probability is refused for a cascade (`docs/COST-MODEL.md`, "Cascades").

Hard constraints are applied before any scoring: a sensitive state gets local models only, the prompt must fit the context window, tools need tool support, and held-out models are excluded.

## 5. Cost model (gap 2, gap 4) - built and tested

Full rules, worked examples and limitations: `docs/COST-MODEL.md`. The rule that shapes everything: where an input is unknown, the model must never understate a cost, because the router would over-choose the option it flatters.

For a single attempt:

- **Cache ownership is per (model, effort), not per model.** A step reads the cache only if it runs the same model at the same effort, unless the profile sets `effort_change_keeps_cache`. The default catalog sets it to false for every model. An unknown owner effort counts as a match only for a model with a single effort. This replaces the first draft's "the model owns the cache".
- **Warm step:** cached prefix tokens are billed at `price_in * cache_read_mult`, the uncached rest at the input rate plus the write premium (the rest is written to the cache after the call).
- **Cold step (the switch tax):** the whole prompt is billed at `price_in * cache_write_mult`. The multipliers come from the profile: Anthropic writes at 1.25 times the input price and reads at 0.025 to 0.1 times (per model); DeepSeek has no write premium (1.0) and reads at 0.02 or 0.033 times; Ollama prices at 0, so its switch cost is latency (model reload plus prefill), which is an optional profile term that is unset in the default catalog.
- **Cache minimum:** below `cache_min_tokens` (512 on Anthropic) nothing is cached, so no premium is charged.
- **Price tiers:** the full prompt, cache reads and writes included, selects the tier, and the tier applies to the whole request, output included. Haiku 5.5 above 100,000 tokens pays the higher rates.
- **Output tokens** are the caller's estimate, scaled by an effort multiplier. **The multiplier is 1.0 for every effort in the default catalog. That is a placeholder, not a measurement**, so higher effort costs no more output than lower effort until the pilot measures it.
- **Time of day:** DeepSeek charges half outside its peak windows. The windows are catalog data (`meta.peak_hours`: 01:00-04:00 and 06:00-10:00 UTC, Monday to Friday). The replay runner prices each DeepSeek call by when it ran and records the billed cost and the list cost. The router's `expected_cost` does not use time of day and prices at the catalog (peak) price. Chinese public holidays are treated as ordinary weekdays, because we cannot verify that calendar here.

For a cascade: `cost = c1 + (1 - p1) * (c2 + ...)`, where each `c_i` is evaluated in the state the failed attempts before it leave behind (the cache now belongs to the previous step). A cascade that changes model pays the switch tax only in the failure branch. The same fold gives `max_usd`, the cost if every step runs, which spend caps use.

Local models have price 0 by default. A per-token compute price and a latency term can be set in the profile.

**Not verified:** all prices, cache multipliers and cache rules come from provider documentation (`docs/INTEGRATION-NOTES.md` section 7, checked 13 Nov 2026), not from a bill. Effort multipliers and latencies are unmeasured. DeepSeek's cache scope per model, the effect of toggling thinking, and how it bills a call that crosses a peak boundary are undocumented; the model assumes the costly case in each. Cache expiry is not modelled, and the model tracks one cache owner per session.

## 6. Graph predictor (gap 3, gap 5) - planned

Nothing in this section is built. `steproute.graph` is a stub, and the `graph` extra (`torch`, `torch-geometric`) is declared but unused.

Heterogeneous graph:

- **Model-config nodes** (model x effort): features from the profile (provider, local/cloud, size, quantization, price, context window) plus results on a small fixed probe set.
- **Step nodes**: embedded state.
- **Tool nodes** and **harness nodes**.
- **Edges**: step-model with observed outcome; step-step sequence; step-tool; step-harness.

Prediction is edge scoring: `P(success | step, model-config)`.

Why a graph and not a classifier: a new model is a new node described by its profile and probe results, connected to existing structure, so it gets a usable prediction before any real traffic. A bandit must explore it from zero. Harness as a node rather than a separate model per harness is what gives cross-harness transfer a chance.

Serving: distil to an MLP if GNN inference exceeds the latency budget (single-digit ms).

**The comparison model exists.** The LightGBM predictor (`steproute.ml`, `docs/MODEL-CARD-TABULAR.md`) uses the same kind of inputs: profile and probe features, and never the model id, so it can say nothing specific about an unseen model by memorising it. It is the model the graph has to beat. It has only been fitted on synthetic data, so its card carries the banner "SYNTHETIC DATA. NOT A RESULT."

## 7. Experiments that decide whether this is worth it

None of these has run on real data.

1. **Cold start (the GNN claim):** add an unseen local model (`devstral:24b`); measure requests until regret falls below a threshold. Graph vs. ParetoBandit vs. LightGBM-with-profile-features. Needs M4.
2. **Action space:** cost per solved task with (model) only vs. (model, effort) vs. (model, effort, cascade). The replay grid runs one configuration per task, so it can score task-level cascades but not switching inside a task (`docs/EVALUATION.md` section 1).
3. **Cache awareness:** same policy with cache ownership on and off, in multi-step sessions. The grid cannot show this at all (no run switches model inside a task). It needs logged step-level traces, and the evaluator's logged-data estimates are one-step values (`docs/EVALUATION.md` section 12).
4. **Transfer:** train on harness A traces, evaluate on harness B.
5. **Bar to clear:** the routed setup must lie beyond the frontier of single configurations. The evaluator defines this precisely (`docs/EVALUATION.md` section 5):
   - The frontier is the upper concave envelope of every single configuration's (spend, success rate) point, with the origin. It is the best success reachable at each spend by mixing single configurations at random.
   - A policy beats it only when the lower end of the 95% interval of its margin above the envelope is above a chosen minimum (default 0). The interval is a paired cluster bootstrap.
   - **No verdict is given on fewer than 100 tasks**, or when the effective sample size is below 50, or when the data cannot score the policy somewhere. The answer is then "not shown". The 20-task pilot cannot produce a verdict.
   - Logged data is scored with off-policy estimators (IPS, SNIPS and doubly robust; doubly robust is the headline), which are only as right as the logged propensities.

   This replaces the first draft's wording, "beat every single-model, single-effort configuration on cost per solved task". That wording is degenerate: a free local model has cost per solved task 0, so nothing could beat it. If the bar is not cleared, there is no product.

If the GNN does not beat LightGBM with the same profile features in experiment 1, drop it and ship the rest. Gaps 1, 2 and 4 stand without it.

## 8. Milestones

Status as of 14 Nov 2026.

- **M0 Foundation:** schema, JSONL store, action enumeration, cache-aware cost model, policy interface, rule baseline, tests. Built and tested.
- **M1 Collectors:** proxy, Claude Code (plugin and MCP server), dsh. Built and tested against mocks and in real Claude Code and dsh sessions. No paid provider has been called.
- **M2 Data:** replay agent tasks across models and efforts to get an outcome matrix, and publish it. The runner, the task set and the spend ledger are built. The data does not exist.
  - Task set: 150 SWE-bench Verified tasks with native arm64 evaluation images, and a 20-task pilot. A task whose gold patch fails on its image can never be solved, so it is replaced. Five `psf/requests` tasks were replaced on 14 Nov 2026, one of them in the pilot (`docs/TASKSET.md` section 8.3).
  - Grid: option B, the sparse grid. Every local configuration and each paid model at its default effort on all 150 tasks, plus the other paid efforts on 30 tasks. The cost estimate is to be redone from the pilot's measured costs before month 2.
  - Pilot: 20 tasks over the pilot configurations. Estimated at $41 of Anthropic spend (range $23 to $179) and $8.79 of DeepSeek spend at off-peak prices (range $4.70 to $44.43) against the $10 total. These are estimates from assumed step counts and prices, not measurements.
  - Spend control: a runner that cannot pass the caps. Every paid call writes its worst case to an append-only ledger before the call and the realised cost after it. A crash over-counts and never under-counts. DeepSeek calls wait for off-peak hours. A per-run cap ends a run as a labelled failure. This is tested against mocks only; whether the pilot finishes under the caps is untested.
- **M3 Baselines:** the evaluator and the LightGBM predictor are built; the report on real data (issue 23, then Gate B) is not. ParetoBandit, LLMRouter's `graphrouter`, OpenSquilla and RouteLLM baselines are planned.
- **M4 Graph:** planned. Graph construction, GNN edge predictor, cold-start experiment, distillation. Starts only after Gate B.
- **M5 Online:** planned. Online updates, drift handling, ablations, experiments 2 to 4.

## 9. Risks

- **Scope.** Five gaps is a lot. The order above means stopping after any milestone still leaves something usable.
- **Cold-start data cost.** M2 needs real spend on replays. The sparse grid is estimated at about 6 months at the Anthropic cap (range 4 to 21) before the pilot replaces the assumptions (`docs/TASKSET.md`).
- **Model drift.** Model snapshots change during a multi-month grid, and a deprecation breaks the matrix.
- **Outcome noise.** Implicit signals (retries, rejections) are weak labels; test-backed tasks anchor the dataset. Benchmark tasks can themselves be broken: 5 of the 7 `psf/requests` images in the 150-task draw failed with the gold patch (`docs/TASKSET.md` section 8.3), and a task nobody solves may be broken rather than hard.
- **Grid is task-level.** The replay data cannot measure step-level routing, the switch tax or switching inside a task. Those need logged traces from real use.
- **Unverified provider behaviour.** Cache behaviour, usage reporting and DeepSeek off-peak billing have not been seen on a real bill. The replay watches for the first (a cache canary) and the pilot report compares the ledger to the provider consoles.
- **Unstable integration points.** dsh has no stable release and the plugin is pinned to one pre-release. Claude Code mods need version 2.1.287 or later.
- **Privacy.** Traces can contain code and prompts. Text fields are optional and off by default, and a `sensitive` flag forces local-only routing (`docs/PRIVACY.md`).
