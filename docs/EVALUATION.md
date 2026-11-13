# Offline evaluation

`steproute.eval` measures routing policies against every single `(model, effort)` configuration. It is the instrument behind the baseline report (issue 23), the cold-start experiment (issue 26) and the ablations (issue 29), so it is built to refuse a number rather than give an optimistic one. Standard library only (`mypy --strict`); metrics version `1`.

```
steproute eval --policy rules --policy fixed:medium:claude-sonnet-5-5 \
    --data grid.jsonl --split test --out report.json        # writes report.json and report.svg
```

```python
from steproute.eval import EvalSettings, as_target, evaluate_grid, load_grid

ev = evaluate_grid(load_grid("grid.jsonl"), [as_target(policy)], EvalSettings(), split="test")
print(ev.policies[0].verdict.sentence)
```

`python -m steproute.eval.synthetic --out DIR` writes a synthetic grid, trace store, catalog and rules file to try it on.

## 1. Two kinds of data, and what each can show

| | Replay grid (issue 18) | Logged traces (trace store) |
|---|---|---|
| Level | **task**: one fixed `(model, effort)` for a whole task | **step**: one action per step |
| Feedback | full information: every task under every configuration | bandit: only the action taken, with its propensity |
| Evaluator | `evaluate_grid`, exact expectations | `evaluate_logged`, off-policy estimates |
| Can evaluate | policies that pick one configuration or one cascade **per task** | any policy whose distribution `pi(a | state)` is known |
| Cannot evaluate | switching inside a task, the switch tax, within-task switch counts | the effect of a choice on later steps (see section 10) |

The boundary is enforced in code, not just stated. `evaluate_grid` asks a policy for its decision once, in the task's start state, and applies it to the whole task; there is no way to hand it per-step choices. Every grid report carries the scope sentence "No run in the grid switches model within a task, so step-level routing, the switch tax and within-task switching are NOT measured here". A step-level router evaluated on the grid is evaluated as a task-level router, and the report says so.

## 2. Data contracts

### 2.1 The replay grid (`load_grid`, `write_grid`)

JSON Lines in the trace store's line shape, `{"type": ..., "v": "0.1", "data": {...}}`, readable with the standard library. Readers accept any `0.x`, ignore unknown keys with a warning, skip and count unreadable lines, and reject a file with no run lines or a duplicated run.

| `type` | How many | Purpose |
|---|---|---|
| `header` | at most one | `{"format": "steproute-grid", ...}` plus free provenance (runner version, catalog, dates). |
| `task` | optional, one per task | Split, routing state, join features. |
| `run` | one per run | One task under one fixed configuration for the whole task. |

**Run** (`GridRun`):

| Field | Type | Required | Meaning |
|---|---|---|---|
| `task_id` | string | yes | Task id; the join key to `StepState.task_id` and to issue 19's splits. |
| `model`, `effort` | string | yes | The configuration, fixed for the whole run. `effort` is in the effort vocabulary. |
| `success` | bool or null | yes | The task check's result. `null` = no usable label (harness error, timeout before the check). Never a failure. |
| `cost_usd` | number or null | yes | Realized cost of the whole run. `null` = unknown. |
| `latency_ms` | number or null | no | Wall-clock time of the run. |
| `run_id` | string or null | no | Unique per run. |
| `session_id` | string or null | no | The run's session in the trace store, so per-step traces join to the run (`StepState.session_id`). |
| `repeat` | int | no (0) | Repetition index of the same task and configuration. |
| `ts` | number or null | no | Unix seconds. |
| `meta` | object | no | Free (runner version, spend-cap state, ...). |

**Task** (`GridTask`): `task_id`; `split` (`train`, `validation`, `test`, or a held-out-model split name; issue 19); `state`, the `StepState` a router sees at task start (when absent, `task_state` builds a minimal replay state from `features`, with `features["prompt_tokens"]` if present); `features`, passed through untouched for issue 21's tabular and issue 24's graph predictor to join on (repository, language, task size, probe results, ...).

```json
{"type":"header","v":"0.1","data":{"format":"steproute-grid","runner":"replay 0.1"}}
{"type":"task","v":"0.1","data":{"task_id":"django-1234","split":"test","state":null,"features":{"repo":"django","prompt_tokens":18000}}}
{"type":"run","v":"0.1","data":{"task_id":"django-1234","model":"claude-haiku-5-5","effort":"low","success":false,"cost_usd":0.021,"latency_ms":91000,"run_id":"r-17","session_id":"s-17","repeat":0,"ts":1767225600.0,"meta":{}}}
```

A **cell** is all runs of one task under one configuration. Its success is the mean of its labelled runs, its cost and latency the means of the runs that report them. It is **complete** when every run is labelled and costed.

### 2.2 Logged traces (`load_logged`)

The existing trace store directory or a `steproute traces export` file; nothing new is written. The evaluator reads, per decision: the logged `state`, `action`, `propensity`, the candidate set (explicit, or rebuilt from `candidate_set` with `rebuild_candidates` and the catalog), and from the last outcome `success`, `cost_usd` (or, for a single-step outcome without a reported cost, the price of its token counts from the catalog). A decision without an outcome is kept, unlabelled and uncosted, and counted. The **resampling cluster** is `task_id` when the state has one, else `session_id`.

## 3. Policies

Everything is reduced to "what probability does the policy give each action in this state" (`TargetPolicy.distribution`). `as_target` wraps: objects with `action_probabilities` (`ScoredPolicy`); objects with only `decide` (`RulePolicy`, `FixedPolicy`), **assumed deterministic**; and plain callables. An empty distribution (`NoFeasibleAction`) means the policy cannot act in that state.

CLI specifications: `fixed:EFFORT:MODEL`; `cascade:EFFORT:MODEL,EFFORT:MODEL[,...]` (effort first because Ollama tags contain `:`); `rules` or `rules:PATH`; `oracle` (grid only; the per-task best configuration in hindsight, flagged `hindsight: true`, never ranked, an upper reference rather than a policy); `python:MODULE:ATTR` (a policy object, or a factory called with the catalog: the hook for issues 21 and 22).

## 4. Metric definitions (version 1)

"Unit" is a task on the grid and a step in logged data.

| Metric | Definition | Awkward cases |
|---|---|---|
| `success_rate` | Mean success over units; for a stochastic policy the exact expectation on the grid. | Unlabelled outcomes are excluded and bounded (section 9), never counted as failures. |
| `cost_per_task` / `cost_per_step` | Mean realized USD per unit, failed attempts included. | |
| `cost_per_solved` | Total spend / units solved = mean cost / success rate. | Nothing solved: value `null` with `"unbounded": true` and note "no task solved". An interval reaching infinity is flagged `ci_unbounded`. |
| `quality_retention` | Success rate / success rate of the **strongest single configuration** (highest success on the same units, ties to the cheaper). | **Not capped**: above 1 means the policy solves more than the strongest configuration (a cascade can). |
| `strongest_solved_retained` | Grid: of the tasks the strongest configuration solves, the share the policy also solves. | At most 1 by construction; logged data: `null` (no paired outcomes per step). |
| `switches_per_task`, `config_changes_per_task` | Grid: expected number of model changes (resp. `(model, effort)` changes) between attempts of a cascade. 0 for a single configuration. | |
| `switch_rate_per_step` | Logged: probability that the policy's first step uses a model other than the state's cache owner, over the logged states. | One-step quantity; see section 10. |
| `added_latency_p50_ms`, `_p95_ms` | Grid: lower quantile over tasks of the latency of cascade attempts **before** the one that produced the result: latency added by routing on top of the attempt that answered. | Routing-decision time is not in the grid and not included (the service benchmarks it separately, issue 11). Logged data: `null`, because outcomes do not split latency by attempt. |
| `total_latency_p50_ms`, `_p95_ms` | Grid: quantiles of end-to-end latency of the policy's runs. | Reported so a fast cheap model is not hidden by "0 added latency". |
| `frontier_margin` | Success rate minus the frontier's success rate at the policy's spend (section 5). | |
| `cost_saving_at_equal_success` | Grid: the frontier's least spend reaching the policy's success rate, minus the policy's spend. | Above the frontier's top: `null`, flagged unbounded. |
| `utility` | `success - cost_weight * cost`, only with `--cost-weight`. | Ranking switches to it when given. |

Rejected: "cost of solved tasks only" for cost per solved (ignores the money spent on failures, which is exactly what routing wastes); capping retention at 1 (hides the one case where a router is clearly useful); added latency relative to the strongest configuration (can be negative and mixes model speed with routing overhead).

## 5. Beyond the frontier: the verdict

**The frontier** is the upper concave envelope of every single configuration's `(spend, success)` point together with the origin `(0, 0)`. It is the best success rate reachable at each spend by sending units at random to single configurations, or to none, and it is flat beyond the configuration with the highest success. A policy **lies beyond the frontier** when its success is above the envelope at its spend; the **margin** is that vertical gap in success-rate points.

Why this shape:

- A random mix of two configurations is a trivial router with no intelligence. Its point lies on the chord between them, so beating only the configurations' step-shaped Pareto frontier would credit it. The concave envelope does not.
- The origin makes the comparison defined at every spend. The segment from the origin is the best cost-per-solved line, so beating the envelope implies beating every configuration's cost per solved task at the same or higher success rate. That is DESIGN section 7's bar, made precise.
- The bar as literally written ("beat every single configuration on cost per solved task") is degenerate: a free local model has cost per solved 0, so nothing can beat it, and a model that solves 1% of tasks almost for free would win. The envelope compares at equal spend instead.

**The verdict** uses the margin's confidence interval (section 7), paired across policies and configurations:

| Status | When | Sentence starts |
|---|---|---|
| `beats_frontier` | lower end of the interval > `min_margin` (default 0) | "P beats every single configuration: at $x per task it succeeds on y% ..." |
| `does_not_beat` | upper end <= `min_margin` | "P does not beat every single configuration: ... C alone costs ..." |
| `not_shown` | the interval contains `min_margin`, or any gate applies | "Not shown: this data cannot tell whether P beats every single configuration (reasons)" |

Ties: a policy that *is* a frontier configuration has margin exactly 0 and does not beat it. Noise: nothing is decided on a point estimate. `--min-margin` sets the smallest gain worth acting on. On the grid a stochastic policy is scored in exact expectation and the comparison is paired, so a real gain of a tenth of a point can be "beats". The margin is always printed, so read it. Gates that force `not_shown`: fewer than `min_clusters` (100) tasks or clusters; any unit the data cannot score the policy on; effective sample size below `min_ess`; support violations; invalid or mismatched propensities; a frontier configuration that cannot be estimated from logs; a verdict that flips when unlabelled outcomes are counted as all failures or all successes.

The verdict is in the report twice: machine-readable (`verdicts[].status`, `beats_every_single_configuration`) and as a sentence (`verdicts[].sentence`, collected in `summary`). Examples from `tests/test_eval_coverage.py::test_verdict_examples` (synthetic grid, 200 tasks):

- "optimal beats every single configuration: at $0.04022 per task it succeeds on 66.5% of tasks, where the best any single configuration or random mix of them reaches at that spend is 49.9% (margin +16.6 points, 95% interval [+13.4, +21.1] points)."
- "inverted does not beat every single configuration: at $0.4541 per task it succeeds on 66.5% of tasks, where the best any single configuration or random mix of them reaches at that spend is 71.8% (margin -5.3 points, 95% interval [-10.5, -0.6] points). cloud-mid (medium) alone costs $0.3617 per task and succeeds on 71.0%."
- "Not shown: this data cannot tell whether local>big-high beats every single configuration; at $2.055 per task it succeeds on 84.5% of tasks, where the best any single configuration or random mix of them reaches at that spend is 81.4% (margin +3.1 points, 95% interval [-0.5, +6.4] points), and the interval contains 0." (Its true margin is +3.1 points: real, too small for 200 tasks.)
- "Not shown: this data cannot tell whether optimal beats every single configuration (only 40 tasks could be evaluated; at least 100 needed)."

Rejected: the step-shaped Pareto frontier (credits random mixing); cost per solved against each configuration alone (degenerate, see above); "dominates every configuration" (impossible: no policy is both cheaper than the cheapest and better than the best); a verdict on point estimates or on overlapping per-policy intervals (unpaired intervals overlap far more often than the paired difference is uncertain).

## 6. Off-policy estimators

With `w_i = pi(a_i | s_i) / mu_i` (`mu_i` the logged propensity), for success `r` and, identically, cost `c`:

- **IPS**: `sum(w r) / n`. Unbiased under correct propensities and support; unbounded, high variance.
- **SNIPS**: `sum(w r) / sum(w)`. Self-normalised: bounded, invariant to adding a constant to rewards, lower variance, `O(1/n)` bias.
- **DR**: `sum(sum_a pi(a | s_i) q(s_i, a) + w (r_i - q(s_i, a_i))) / n`. Unbiased under correct propensities **whatever** `q` is; `q` only changes the variance.

Cost per solved is the ratio of each estimator's cost and success estimates.

**Headline: DR.** It keeps IPS's guarantee (correct propensities give no bias, however wrong the model) and uses the model to cut variance, which matters for deterministic targets that agree with the logging policy on few steps. `tests/test_eval_ope.py::test_dr_stays_unbiased_with_a_misspecified_reward_model` gives it an absurd model (always 0.99, $5) and checks that DR stays centred on the truth while the model alone is off by more than 15 points. All three estimators are always in the report (`estimators`), so a disagreement between them is visible. `--headline snips` is the conservative alternative.

**The reward model** is either fitted by the evaluator or supplied. Anything with `fit(rows) -> model` is **cross-fitted**: clusters are split into 5 seeded folds, and each row's predictions come from a model fitted on the other folds, so no row is predicted by a model that saw it (`test_reward_model_is_cross_fitted_out_of_fold` checks that with a model that refuses to predict its training rows). A model with only `predict(state, action) -> (p_success, cost)` is used as given, and the report states that it must not have been trained on these logs; `--reward-model MODULE:ATTR` passes one. The built-in `CellMeanModel` is mean success and cost per (action, step kind), shrunk toward the action and global means: crude on purpose, because DR does not need it to be right. Issue 21's predictor can be plugged in the same way. The model is not refitted inside the bootstrap (noted in the report).

**Clipping** is off by default. With `--clip M`, weights above `M` are capped. The report gives the clipped rows and weight mass, a plug-in bias bound (`sum((w - M)+) / n` times the reward range: 1 for success, the largest observed cost for cost), and **widens every interval by that bound on both sides**. Clipping can then only make an estimate less confident, never more.

Rejected: SNIPS as the headline (its bias is small but has no correction term, and it cannot use a model); clipping by default (silent downward bias); a model fitted on all rows (in-sample fit makes DR look better than it is); the direct method alone (its error is the model's error, unmeasured).

## 7. Uncertainty

Intervals are **BCa cluster bootstrap** intervals (bias-corrected and accelerated) at `--confidence` (0.95), with 2000 replicates by default.

- **Unit of resampling:** whole clusters. On the grid a cluster is a task. In logs it is the task id when present, else the session: steps within a session share the task and the cache, and sessions of one task share its difficulty, so neither is independent.
- **Paired:** one set of resamples serves every policy and configuration of a dataset. The margin's interval therefore accounts for the frontier's own noise and the policy's correlation with it.
- **BCa** rather than percentile: the percentile interval undercovers skewed statistics such as mean cost and ratios. The acceleration comes from a grouped delete-one-cluster jackknife (at most 100 groups). It falls back to the percentile interval when undefined.
- Every statistic, point and replicates alike, comes from one function of the cluster multiplicities, so points and intervals cannot disagree on definitions. Seeds derive from `--seed`, so a report is reproducible byte for byte apart from its timestamp (`test_cli_is_deterministic`).
- **Effective sample size** (Kish, `(sum w)^2 / sum w^2`, on labelled rows and on costed rows) is reported for every estimate. Below `--min-ess` (50) the headline metrics are `null` with "withheld: effective sample size ..." and the verdict is `not_shown`; the raw estimators stay in `estimators` for diagnosis. The threshold comes from the coverage measurements in section 8: at ESS about 20 the 95% intervals for cost covered 85 to 87%; at about 50, 90 to 92%; at 70 and above, 91 to 96%.
- **`--min-clusters` (100)** is the fewest clusters for a verdict. The margin's interval is measured to undercover on small grids (section 8). Below 100 tasks the frontier, a maximum over noisy points, is biased upward, and misses fall mostly on one side.

Rejected: a normal approximation with a cluster-robust variance (poor for ratios, for the envelope and at small n); resampling steps or sessions independently (understates the variance); Bayesian intervals (need a model of the outcome process we do not have).

## 8. Synthetic ground truth and coverage (acceptance criterion 1)

`steproute.eval.synthetic` is a small discrete world, so every true value is an exact finite sum, not a simulation:

- A task has a hidden difficulty level (5 levels). Policies see a hint equal to it with probability 0.8. A configuration succeeds with probability `sigmoid(ability - difficulty - step offset)`. Every model faces the same difficulty, so a task that defeats one model is harder for the others: failures are correlated across models.
- Costs span three orders of magnitude, and one configuration is a free local model. Cost grows with difficulty and carries log-normal noise of mean 1.
- Logged sessions have 4 steps with the previous step's model as cache owner, a 25% switch tax, two sessions per task, and an epsilon-greedy logging policy (`SynthLogging`, epsilon 0.3) whose greedy choice depends on the cache owner. The logged state distribution therefore depends on the logging policy, as in real sessions.
- `optimal_task_policy` is the known optimum of `success - 2 * cost` given the hint. `true_task_value` and `true_logged_value` compute exact values; the step-level truth propagates the logging policy's cache-owner distribution step by step.

**Ranking.** `test_grid_ranks_policies_correctly` (1,500 tasks) and `test_off_policy_ranks_policies_correctly` (1,500 tasks of logs) check that every pair of policies whose true values differ by more than 2 points is ordered correctly, by utility and by frontier margin, and that the optimum is ranked first. They also check that the optimum beats the frontier, while a uniform random mix of configurations and a single configuration do not.

**Coverage.** Fixed seeds, 200 repetitions each, 95% BCa intervals from 200 bootstrap replicates (the default is 2000). The grid used 150 tasks per repetition. The logs used 200 tasks, two sessions each, 1,600 steps, and the ESS was above 50 in every repetition, so this is coverage where the report gives a headline (the test asserts it).

| Data | Policy | Metric | Coverage |
|---|---|---|---|
| grid | optimal | success / cost / cost per solved / margin | 0.935 / 0.910 / 0.890 / 0.920 |
| grid | cascade local -> mid | success / cost / cost per solved / margin | 0.960 / 0.950 / 0.945 / 0.960 |
| logged | by-hint (deterministic) | IPS success / cost | 0.950 / 0.935 |
| | | SNIPS success / cost | 0.940 / 0.955 |
| | | DR success / cost | 0.945 / 0.930 |
| logged | mix (stochastic) | IPS success / cost | 0.955 / 0.920 |
| | | SNIPS success / cost | 0.935 / 0.915 |
| | | DR success / cost | 0.940 / 0.950 |

With 200 repetitions a single figure has a standard error of about 1.6 points. A longer run (800 repetitions, grid, 150 tasks) put the true coverage at 0.93 for cost-type metrics, 0.94 for the margin and 0.95 for success. So the evaluator's 95% intervals are about 2 points narrow on skewed costs at this size: a small, known optimism, stated here and in the test.

**Tolerance**, justified in the test's docstring: per metric, at least `0.93 - 3 * sqrt(0.93 * 0.07 / 200) = 0.876` (nominal minus the measured 2-point shortfall, minus 3 Monte Carlo standard errors), and at most 0.995 (intervals too wide to be useful are a failure too); the mean over each evaluator's metrics, at least 0.91. The bounds come from the documented shortfall, not from the seeds.

Smaller grids (same world, 400 repetitions): at 25 tasks the margin's interval covered 82 to 91%, with the true value above the interval 6 to 13% of the time, because the estimated frontier is too high. At 50 tasks it covered 86 to 89%, with misses on both sides. At 100 tasks, 92 to 93%, balanced. Success-rate intervals stayed at 94 to 96% throughout. Hence `min_clusters = 100`.

**Failure modes documented by tests.** Propensities logged wrong (`test_wrong_propensities_bias_the_estimate_silently`): a collector that logs `1/K` instead of the true epsilon-greedy probability moves IPS more than 15 points off the truth, and at least 15 of 20 intervals exclude the truth. The report cannot know: `propensity_mismatches` is `null`, unchecked. Only `--logging-policy`, which recomputes every propensity, exposes it. Support (`test_support_violations_are_detected_and_rows_kept` and neighbours): a target on actions the logger never took, a deterministic logger, and a compact candidate set without a catalog are each detected and stated, and no row is dropped.

## 9. Support, propensities and unlabelled outcomes

**Support.** A decision logs the propensity of the chosen action only, not the logging distribution. The evaluator takes the logging policy's support at a row to be, in order of preference:

1. `{a : mu(a | s) > 0}` from `--logging-policy` (also checks every logged propensity, tolerance 1e-6);
2. the logged action alone when the propensity is 1 (a deterministic logger: `RulePolicy`, `FixedPolicy`, the service's `fail_open:*` decisions, counted as `fail_open_rows`);
3. the logged candidate set when the propensity is below 1, which assumes the logger gave every candidate some probability (true for `ScoredPolicy` with `epsilon > 0`);
4. otherwise **unknown**.

Per target, the report gives `unsupported_mass`, the mean probability the target puts on actions outside that support, and `unknown_support_mass`, along with the number of affected rows. Above `support_tolerance` (0.1%) the verdict is `not_shown`, with the share named in the reason. A target that cannot act at a state (`NoFeasibleAction`) counts as fully unsupported there. Rows are never dropped for support: IPS and SNIPS would silently estimate a different policy, and DR would silently extrapolate the reward model.

**Propensities.** A propensity outside `(0, 1]` or missing is invalid. Those rows are excluded from estimation, counted (`invalid_propensity`), and gate the verdict.

**Unlabelled outcomes** (`success = null`) are not failures. On the grid the headline uses the tasks where every configuration's runs are labelled and costed, the same task set for every policy and configuration. Excluded tasks are counted by reason. In logs, success estimates use labelled steps, which assumes labels are missing at random given state and action; cost estimates use costed steps. In both, `unlabelled_sensitivity` recomputes the margin with every unlabelled outcome counted as a failure and then as a success. If the verdict's direction does not survive both, it becomes `not_shown`.

Rejected: imputing unlabelled outcomes as failures in the headline (biases against configurations that time out more); dropping only the unlabelled cell (compares policies on different task sets).

## 10. Reading the report

```
format, metrics_version, steproute_version, generated_at, command, settings, definitions
verdicts[]      data, kind, level, policy, hindsight, status, beats_every_single_configuration, sentence
summary[]       the sentences
evaluations[]   one per --data
  kind, level, scope                      what this data can and cannot show
  provenance                              files, sha256, counts: runs/decisions, unlabelled, uncosted,
                                          excluded tasks by reason, unsupported, candidate-set issues,
                                          reward model and how it was fitted
  frontier      configurations[], envelope[], excluded{}   each point with interval, n, ESS
  policies[]    metrics{}                 headline: value, ci, n, ess, note, unbounded flags
                verdict{status, reasons, sentence}
                diagnostics{}             ESS, support, clipping, unlabelled sensitivity, action mix
                estimators{ips,snips,dr}  logged data only
  ranking[]     by frontier margin, or utility with --cost-weight; hindsight references excluded
  notes[]       cascade simulation, latency, split, repeats, propensity assumptions
```

Read in this order: the `scope`, the provenance counts (how much was excluded and why), the verdict and its reasons, then the margin and its interval. A `not_shown` verdict with a large point margin means "collect more or better data", not "probably beats".

**The plot** (`--plot`, default next to `--out`, one SVG per dataset) shows score (success rate) against spend (mean cost per unit). Configurations are neutral circles with direct labels. The frontier is a dashed line. Each policy has its own marker shape, a direct label, interval whiskers and a legend entry stating its verdict, so identity never depends on colour alone. Light and dark themes follow the viewer, and every mark has a tooltip. It is written with the standard library, so the JSON report and the plot are always produced.

## 11. Dependencies

The evaluator is standard library only, like `steproute.core`: JSON Lines in, JSON and SVG out, `statistics.NormalDist` for BCa. Nothing is needed beyond `pip install steproute`, and `import steproute.eval` loads no third-party module (`test_eval_import_is_stdlib_only`). Rejected: a plotting extra (matplotlib). It would make the plot optional and untested in CI, and the lockfile would need a new resolution. A scatter plot with an envelope line is small enough to write directly. Rejected too: numpy for the bootstrap. The pure-Python bootstrap evaluates a 150-task grid with several policies at 2000 replicates in a few seconds.

## 12. Limitations

Blunt, because the project's go/no-go depends on them.

- **The grid cannot see step-level routing.** It measures task-level policies: one configuration or cascade per task. A router that switches models mid-task has never been run in the grid, so any grid number for it describes a different policy. The switch tax, within-task cache effects and within-task switch counts are absent.
- **Simulated cascades are optimistic about independence and pessimistic about cost.** A cascade is scored from independent runs: the second step pays a full fresh run, with no cache reuse, and does not see why the first failed. A real cascade could be cheaper (warm context) or better (learns from the failure), and its failures could be more or less correlated than independent runs given the task. The synthetic world has the same independence given difficulty, which flatters cascades: every cascade in it beats the frontier. Do not read the synthetic results as evidence for cascades.
- **Logged estimates are one-step values.** Each estimate is the value of the target routing one step in a state the *logging* policy produced. It includes that step's switch tax, but not the target's effect on later steps' cache owner or prompts, so it is not the value of deploying the target for whole sessions. A trajectory estimator (product of weights over a session) is unbiased for that, but its effective sample size collapses with session length. It is not implemented.
- **Logged data rarely supports a verdict.** Comparing against every single configuration needs each one to be explored. With `ScoredPolicy` at depth 2 (555 candidates) and `epsilon = 0.05`, an explored action has propensity about 1e-4. Fixed single configurations then get ESS near 0, the frontier is incomplete, and the verdict is `not_shown`. That is the correct answer: logs from a nearly greedy router cannot say whether it beats the configurations it never tries. Use the grid for the frontier.
- **Propensities are trusted.** The estimators are only as right as the logged propensities. Wrong propensities give confident wrong answers (section 8), and only `--logging-policy` can catch them. Policies with randomness that `action_probabilities` does not describe (a Thompson-sampling predictor, issue 28) log wrong propensities by construction (docs/POLICY.md).
- **Support is inferred, not logged.** Without `--logging-policy` the evaluator assumes a stochastic logger covered its whole candidate set. A logger with probability-0 candidates breaks that silently.
- **Missing labels are assumed missing at random** for the headline. The sensitivity check bounds the effect on the verdict, but not on the estimates themselves.
- **Intervals run about 2 points narrow** on skewed costs at 150 tasks, and the margin is unreliable below 100 tasks. The 15 to 25 task pilot (issue 17) cannot produce a verdict; it is gated.
- **Selection.** Evaluating many policies on one task set and reporting the best inflates the winner. The intervals are per policy, not simultaneous. Evaluate tuned policies on a held-out split (`--split test`); the report warns when no split is used.
- **Added latency** excludes routing-decision time. On logged data it is not estimated at all.
- **The built-in reward model is crude.** DR stays unbiased, but its variance reduction is modest until issue 21's predictor is plugged in.
