# Design: an open, outcome-trained, graph-based step router

*Draft 0.3, 13 Nov 2026. Package name `steproute`, MIT licence. Companion to `gnn-llm-router-findings.md`.*

## 1. Goal

One router that covers five gaps found in the open-source market, built so each part is useful on its own:

| # | Gap | Where it lives in this design |
|---|---|---|
| 1 | Open step-level outcome data | Trace schema + collectors (section 3) |
| 2 | Learned (model, effort, cascade) action space | `Action` type + expected-cost model (sections 4, 5) |
| 3 | Inductive cold start for new models | Graph predictor over model profiles (section 6) |
| 4 | Cache-aware decisions | Switch tax inside the cost model (section 5) |
| 5 | Cross-harness transfer | Harness-neutral state + harness as a graph node (sections 3, 6) |

The five aggregate cleanly because of one split: **cost is computed, quality is learned.**

- Cost of an action in a given state is arithmetic over prices, token counts and cache ownership. Gaps 2 and 4 are mostly solved here, with no training.
- Probability of success for (state, action) is the only learned quantity. Gaps 3 and 5 are properties of *how* that predictor is built. Gap 1 is the data it trains on.

So the system is: `decide = argmin expected_cost  subject to  P(success) >= tau`, with a pluggable `P(success)` predictor. Rule baseline, LightGBM, bandit and GNN all plug into the same slot and are compared on the same traces.

## 1a. Resolved decisions

- **Target:** tool first, research second. Collectors and the cost-aware router ship before the GNN work.
- **Model pool:** Claude (Haiku 5.5, Sonnet 5.5, Opus 5.5, Fable 5.2), DeepSeek API models, and local models served by Ollama on an M4 Pro Mac with 48 GB (`qwen3.5:9b`, `gpt-oss:20b`, `qwen3-coder:30b`; `devstral:24b` held out for the cold-start experiment).
- **Bootstrap data:** a SWE-bench-style subset of about 100 to 200 tasks; a pilot replay first, capped at $80 of Anthropic API spend per calendar month (plus $20 DeepSeek); the full grid only after approval, spread over later months.
- **Publication:** only benchmark replay traces are published. Personal usage traces stay local.
- **Licence:** MIT.

## 2. What we reuse instead of building

- **Transport and failover:** LiteLLM (proxy and SDK).
- **Router zoo and benchmark harness:** LLMRouter (custom-router plugin; its `graphrouter` is a baseline).
- **Baselines:** ParetoBandit (online bandit with budgets), OpenSquilla's open LightGBM router, RouteLLM.

We write: the trace schema, the collectors, the cost model, and the graph predictor.

## 3. Trace schema (gap 1, gap 5)

Two append-only record types, joined by `decision_id`:

- **Decision**: harness-neutral `StepState`, the candidate actions, the chosen action, the policy name and the **propensity** (probability the policy gave the chosen action). Propensity is what makes the logs usable for off-policy training and evaluation later.
- **Outcome**: success signal, raw signals (tests passed, tool error, retry, user rejection, judge score), token usage including cache read/write, cost, latency, whether the cascade escalated.

`StepState` carries only what every harness can supply: harness name, session/task/step ids, step kind, prompt tokens, cached prefix tokens, which model owns the cache, tools in scope, a `sensitive` flag, and a free `features` dict. Task text is optional so traces can be shared without content.

Collectors (milestone M1), each a thin adapter emitting the same records:

1. OpenAI-compatible proxy (LiteLLM callback) for local stacks and any harness.
2. Claude Code hooks plus an MCP `route_task` / `report_outcome` pair.
3. dsh plugin reading the session log.

## 4. Action space (gap 2)

`Action = (model, effort, escalate_to)`, where `escalate_to` is another `Action` tried only if the first attempt fails. This expresses the ladder from Anthropic's model-selection guidance directly: same model at higher effort first, then the next model up at lower effort.

Overall success of a cascade: `1 - (1 - p1)(1 - p2)`.

## 5. Cost model (gap 2, gap 4)

For a single attempt:

- If the action's model owns the session cache: cached prefix tokens are billed at the cache-read rate, the rest at the input rate.
- If it does not: the whole prompt is billed at the cache-write rate. This is the **switch tax**.
- Output tokens are estimated and scaled by an effort multiplier.

For a cascade: `cost = c1 + (1 - p1) * c2`, where `c2` is evaluated in the state the first attempt leaves behind (cache now owned by the first model). A cascade that changes model therefore pays the switch tax only in the failure branch, which is exactly the trade the router should weigh.

Local models have price 0 by default; a per-token compute price or a latency term can be set in the profile.

**Assumptions to verify:** cache read/write multipliers and effort-to-output multipliers are placeholders in `ModelProfile` and must be measured per provider.

## 6. Graph predictor (gap 3, gap 5)

Heterogeneous graph:

- **Model-config nodes** (model x effort): features from the profile (provider, local/cloud, size, quantization, price, context window) plus results on a small fixed probe set.
- **Step nodes**: embedded state.
- **Tool nodes** and **harness nodes**.
- **Edges**: step-model with observed outcome; step-step sequence; step-tool; step-harness.

Prediction is edge scoring: `P(success | step, model-config)`.

Why a graph and not a classifier: a new model is a new node described by its profile and probe results, connected to existing structure, so it gets a usable prediction before any real traffic. A bandit must explore it from zero. Harness as a node rather than a separate model per harness is what gives cross-harness transfer a chance.

Serving: distil to an MLP if GNN inference exceeds the latency budget (single-digit ms).

## 7. Experiments that decide whether this is worth it

1. **Cold start (the GNN claim):** add an unseen local model; measure requests until regret falls below a threshold. Graph vs. ParetoBandit vs. LightGBM-with-profile-features.
2. **Action space:** cost per solved task with (model) only vs. (model, effort) vs. (model, effort, cascade).
3. **Cache awareness:** same policy with the switch tax on and off, in multi-step sessions.
4. **Transfer:** train on harness A traces, evaluate on harness B.
5. **Bar to clear:** the routed setup must beat every single-model, single-effort configuration on cost per solved task. If it does not, there is no product.

If the GNN does not beat LightGBM with the same profile features in experiment 1, drop it and ship the rest. Gaps 1, 2 and 4 stand without it.

## 8. Milestones

- **M0:** schema, JSONL store, action enumeration, cache-aware cost model, policy interface, rule baseline, tests.
- **M1:** collectors (proxy first, then Claude Code, then dsh).
- **M2:** bootstrap dataset. Replay a few hundred agent tasks across models and efforts to get a dense outcome matrix; publish it.
- **M3:** baselines on that data (rules, LightGBM, ParetoBandit, LLMRouter graphrouter).
- **M4:** graph predictor and experiment 1.
- **M5:** online updates, drift handling, experiments 2 to 4.

## 9. Risks

- **Scope.** Five gaps is a lot. The order above means stopping after any milestone still leaves something usable.
- **Cold-start data cost.** M2 needs real spend on replays.
- **Outcome noise.** Implicit signals (retries, rejections) are weak labels; test-backed tasks should anchor the dataset.
- **Unverified integration points.** Claude Code subagent/MCP behaviour and dsh seam names need checking against current docs before M1.
- **Privacy.** Traces can contain code and prompts. Text fields are optional and a `sensitive` flag forces local-only routing.
