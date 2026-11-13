# Rule baseline policy

`steproute.core.rules` holds `RulePolicy`: a policy with no learning that encodes common model-selection guidance as a fixed baseline. It is the floor every learned policy must beat (issue 23 compares against it) and the policy the HTTP service runs first (issue 11). Standard library only, `mypy --strict` clean, deterministic (propensity 1.0).

The guidance it encodes is ordinary rule-of-thumb practice: use a mid-tier model by default, use the cheap tier for read-only helper work, and escalate when a step fails. The defaults below are a hand-written starting point, **not a measured optimum**.

```python
from steproute.core import RulePolicy, load_rule_config
from steproute.core.catalog import load_default_catalog

policy = RulePolicy(load_default_catalog())  # shipped rules.toml
policy = RulePolicy(catalog, load_rule_config("my_rules.toml"))
decision = policy.decide(state)
```

## Rules

The rules speak in **tiers** (the catalog's `tier` field), not model names, so they survive catalog changes. With the default catalog, tier 5 is the mid tier (`claude-sonnet-5-5`, `deepseek-v4-pro`) and tier 4 the cheap tier (`claude-haiku-5-5`, `deepseek-flash`).

| Rule | When | Target |
|---|---|---|
| `readonly_subagent` | `kind` is a subagent kind, `state.tools` is non-empty, and every tool is in `read_only_tools` (case-insensitive) | `cheap.tier` at `cheap.effort` (default tier 4, `low`) |
| `default` | anything else, including a subagent with any other tool or with no tools | `default.tier` at `default.effort` (default tier 5, `medium`) |

`decide` then runs these steps in order:

1. `mark_before_routing(state)`; the marked state is the one logged in the `Decision`.
2. Hard filter: `filter_feasible` over one probe action per non-holdout model. Sensitive means local only, the prompt must fit the context window, tools need `tools = true`, holdout models are out. If nothing is feasible, `NoFeasibleAction` propagates.
3. Choose the tier: the target if it has a feasible model; otherwise the **fallback**: the lowest feasible tier above the target, then the next higher ones, then the feasible tiers below the target from the nearest downwards. (A sensitive default state therefore lands on tier 3, the strongest local tier.)
4. Choose the model in that tier: `current_model` if feasible and in the tier (when `models.prefer_current`), else the first feasible model by `models.provider_order` (unlisted providers last), then catalog order.
5. Stay rule: if the chosen model is `current_model` and `stay.enabled`, it is a "stay". The effort is `state.current_effort` when `stay.keep_effort`, it is set, and the model supports it; otherwise the rule's effort, moved to the nearest effort the model supports (ties go lower; DeepSeek has no `medium`, so `medium` becomes `low`).
6. Cascade (when `cascade.enabled`): the first step, then (a) the same model one supported effort above (`cascade.effort_up`), then (b) the next higher feasible tier's preferred model at `cascade.next_tier_effort`, nearest supported (`cascade.next_tier`). A rung that does not exist (top effort, top tier) is skipped. Every rung is a feasible model, and every consecutive pair is strictly stronger per `docs/ACTIONS.md` (same model higher effort, or a higher tier), so the chain never cycles.

Example, default catalog, plain main step: `claude-sonnet-5-5 medium -> claude-sonnet-5-5 high -> claude-opus-5-5 low`.

### Cost consequence of staying

The cache owner is the pair (model, effort) (`docs/COST-MODEL.md`), so changing effort on the same model drops the cached prefix unless the profile sets `effort_change_keeps_cache`. With `stay.keep_effort = true` (default) a stay on a long-running session keeps the cache the session already has, at the price of not applying the rule's own effort (a session at `high` stays at `high` even though the default rule says `medium`). Set it to `false` to apply the rule's effort strictly and accept the cache loss. Changing model costs the switch tax regardless; the rules only change model when the tier differs or the model is infeasible.

### Where the reason is recorded

`Decision` has no reason field and the schema is unchanged. The policy records it in `decision.state.features["rules"]` (a new dict; the caller's state is not mutated):
`{"rule": "default" | "readonly_subagent", "target_tier": int, "tier": int, "fallback": bool, "stayed": bool}`.

## Config reference

`steproute/catalogs/rules.toml` ships as package data; `load_rule_config()` reads it, `load_rule_config(path_or_mapping)` reads yours. Missing keys keep the `RuleConfig` defaults, which equal the shipped file (tested). **Unknown sections or keys are an error** (`RuleConfigError`), as are unknown efforts and wrong types. `provider_order` naming a provider absent from the catalog is an error at `RulePolicy` construction.

| Section.key | Default | Meaning |
|---|---|---|
| `default.tier`, `default.effort` | `5`, `"medium"` | default rule |
| `cheap.tier`, `cheap.effort` | `4`, `"low"` | read-only subagent rule |
| `cheap.subagent_kinds` | `["subagent"]` | `kind` values counted as subagent work |
| `cheap.read_only_tools` | search, read, grep, glob, ls, list, find, cat, websearch, webfetch, notebookread, test, tests, run_tests, test_run, pytest | read-only tool names |
| `stay.enabled` | `true` | keep `current_model` when it is in the chosen tier |
| `stay.keep_effort` | `true` | keep `current_effort` while staying |
| `cascade.enabled` | `true` | build a cascade at all |
| `cascade.effort_up` | `true` | same model, one supported effort up |
| `cascade.next_tier` | `true` | next higher tier, preferred model |
| `cascade.next_tier_effort` | `"low"` | effort used on that rung |
| `models.prefer_current` | `true` | prefer the model in use among same-tier models |
| `models.provider_order` | anthropic, deepseek, ollama | tie-break between providers in a tier |

## Limitations

- Nothing is measured. Tier numbers are the catalog's starting guesses; the rules do not look at the task, only `kind`, tool names, size and the privacy flag.
- Tool names are matched by exact (case-folded) string, so harness-specific names must be added to `read_only_tools`. A tool that reads but is not listed makes a subagent step default-rule.
- A "read-only" tool list says nothing about what a tool really does; a mislabelled tool is the config author's error.
- No allow or deny lists, no cost cap, no use of `failed_attempts` or `features`: after a failure the caller walks the cascade it was given.
- The cheap rule only looks at the tier. If the catalog has no model in a tier, the fallback silently picks a neighbour (visible in `fallback`).
- Latency is dominated by the sensitive-text scan in `mark_before_routing`; see `scripts/bench_policy.py`.
