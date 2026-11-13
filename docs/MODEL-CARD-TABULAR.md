# SYNTHETIC DATA. NOT A RESULT.

> **SYNTHETIC DATA. NOT A RESULT.** Every number in this card comes from the evaluator's synthetic generator (`steproute.eval.synthetic`), because no real outcome data exists yet (the paid replay of issue 18 has not run). They test the code. They say nothing about any real model, about whether routing pays, or about how LightGBM will do on real tasks. Issue 23's report on real data is a different document in a different place (`docs/results/`); this card and its outputs live under `docs/MODEL-CARD-TABULAR.md` and `docs/synthetic/tabular-predictor/` so the two cannot be confused.

# Model card: the tabular predictor (LightGBM)

`steproute.ml` (issue 21). The first learned `P(success | state, model step)`, and the baseline every later model has to beat: the graph model (issues 24 to 26) is dropped if it does not beat this one given the same profile and probe features (`docs/DESIGN.md` section 7).

## 1. Intended use

- **Plugged into `ScoredPolicy`** (`docs/POLICY.md`) as its `Predictor`: it answers "does this `(model, effort)` step succeed in this state", and the policy turns that into a cost-versus-success decision. It also offers `predict_many`, which the policy uses (section 4.6).
- **As the comparison point** for issue 23 (the baseline report), issue 26 (cold start: same profile and probe features, no graph) and issue 27 (serving latency).
- **Not** a step-level model of the switch tax or the cache (section 4.3 says why the data cannot support one), not a text model (it never sees `task_text`), and not for any decision about real spend until it has been retrained and re-checked on real data (section 7).

## 2. Data

Read through the evaluator's grid contract only (`docs/EVALUATION.md` section 2.1, `load_grid`); there is no second format. Logged traces are not used (section 4.3).

| | |
|---|---|
| Source | `python -m steproute.eval.synthetic --out DIR --tasks 1000 --seed 0`: 1,000 tasks, every task run once under each of 6 configurations (4 models), one task feature (`hint`, 80% accurate) |
| Why it is not real | A task has one of 5 hidden difficulty levels; a configuration succeeds with `sigmoid(ability - difficulty)`. Failures are correlated across models through the task, and runs are independent given the task. There is no text, no repository, no tool use |
| Roles (by task, seed 0) | 50 probe, 523 train, 95 tune, 142 calibration, 190 test, 0 excluded |
| Rows | 6,587 train, 1,204 tune, 1,758 calibration (fresh and cascade rows, section 4.3); 2,353 test (1,140 first step, 1,213 after one failure) |
| Fingerprint | `fb8f9faaa41067992eca579ab24006b96ee49136f324abf58e12a79827b3499d` (data, roles, probes, feature spec, settings; stored in the model's `meta.json`) |

A **row** is one `(task, configuration)` cell whose first labelled run gives the 0/1 label (repeats beyond the first are not used in v1; an unlabelled run gives no row, it is not a failure). A **cascade row** is a later step after earlier steps failed, labelled with that step's own outcome on the same task (section 4.3).

## 3. Features

All columns are numbers; missing is `NaN` (LightGBM routes it down a learned branch). `FeatureSpec.names` is the authoritative list, saved with every model.

| Block | Columns |
|---|---|
| State | `kind_main`, `kind_subagent`, `kind_tool_followup`, `kind_plan`, `kind_other`; `log_prompt_tokens`, `cached_frac`, `log_cached_tokens`, `n_tools`, `step_idx`, `sensitive`; `task_<key>` for each numeric or boolean key of `state.features` seen in training (synthetic: `task_hint`); `n_failed`, `failed_max_tier`, `failed_max_effort_rank`, `failed_max_probe_rate` |
| Configuration (profile) | `tier`, `local`, `log_price_in`, `log_price_out`, `log_context_window`, `effort_rank`, `effort_output_mult`, `cache_read_mult`, `supports_tools`, `log_size_gb` (from `meta.size_gb`, NaN if absent); `provider_anthropic`, `_openai`, `_google`, `_deepseek`, `_ollama`, `_other` |
| Configuration (probe) | `probe_n`, `probe_rate`, `probe_log_cost` |
| Relation | `owns_cache`, `same_effort_as_current`, `model_already_failed`, `tier_minus_failed_max`, `probe_rate_minus_failed_max` |

40 columns for the synthetic spec. Prices use `log1p`, so a free local model is 0. Tiered prices (`price_tiers`) are not applied: the base price stands in.

## 4. The eight design decisions

### 4.1 The feature set, and how categoricals are encoded

**Decision.** A configuration is described only by what it is (its profile) and what it did (probes). The model id, and anything derived from it, is never a feature. Ids are used to look a profile or a probe result up and for nothing else. The provider is a one-hot over a fixed family vocabulary with an `other` bucket, and the step kind likewise, so the vocabularies are in the code and in the feature spec, not learned from whatever the training file contained. Training iterates configurations and tasks in an order that does not depend on ids (profile attributes first, the id only as a last tie-break).

**Why.** A tree that can split on "this is model X" memorises X and has nothing to say about a model it has not seen; the whole point of the comparison with the graph model is the unseen model. `test_renaming_every_model_gives_the_same_predictions` renames every model (reversing their sort order) and checks identical predictions twice: from the same trained model with a renamed catalog, probe table and states, and from a retrained model on the renamed grid.

**Consequence, stated plainly.** Two models with identical profiles and probe results are indistinguishable to this predictor. That is intended. `tier` is a hand-set starting guess in the catalog, not a measurement, and the trees may lean on it; an ablation without it belongs to issue 26.

**Rejected.** The model id (or a LightGBM categorical on it, or a hash, or a name embedding): memorisation. The provider as a free string categorical: it is an attribute shared by many models, but an unseen provider would have no learned meaning, so it falls into `other`. High-cardinality or text task features: only numeric and boolean keys of `state.features` are passed through.

### 4.2 Probe features and cold start

**What a probe set is.** A fixed, ordered list of tasks, drawn once from the training pool by the seeded splitter (50 by default). Every configuration's result on each of them (its first labelled run) is the probe table. "k probes" is the first k of the list, so the sets for k = 0, 5, 20, 50 are nested and identical for every configuration. The table is saved with the model (`probes.json`) and can be replaced by another one to describe a new model from its own probe runs.

**How results become features.** For every row of a configuration: `probe_n` (how many of the first k probes it has a labelled result for), `probe_rate` (success rate on those), and `probe_log_cost` (mean `log1p(1000 * cost_usd)`, milli-dollars). `failed_max_probe_rate` and `probe_rate_minus_failed_max` carry the same for the earlier failed steps of a cascade.

**Cold start.** With no probe results `probe_n = 0` and the two statistics are `NaN`, never 0 (zero would say "failed everything"). A configuration with a few probes has a small `probe_n` and a noisy rate; the model sees `probe_n`, so it can discount the rate. To make the trees learn those branches, each **training** row draws its probe count from (0, 5, 20, all) at random (probe dropout), seeded. At prediction time `n_probes` picks the count (the cold-start experiment uses 0, 5, 20, 50).

**Leakage.** A probe task is in no other role: not train, tune, calibration or test (`test_roles_are_by_task_and_probes_stay_out_of_every_other_role` builds every role's rows and checks they never contain a probe task and never share a task). The probe table only ever contains probe tasks. If the grid has groups (a repository, a stratum), probe groups are drawn whole.

**Fairness to the graph model.** This baseline gets the full profile and the full probe table. If the graph model (issue 24) is also given inputs this one is not (text embeddings, tool or harness nodes), the report must say so: the comparison is then not "same features".

**Rejected.** Per-probe columns (50 sparse columns that cannot represent k = 5 and invite memorising the probe set). Weighting probes by how hard the existing pool found them (an IRT-style ability): the pool contains the configuration being described for old models but not for a new one, so training and serving would see different features. Allowing probe tasks to also be test tasks: that leaks the label of the rows being scored.

### 4.3 Conditional probabilities for cascades

**What the data can support.** The grid has every task under several configurations, so for a pair `A -> C` it has, on the tasks where `A` failed, `C`'s own outcome. That is the conditional probability the evaluator's cascade simulation realises (a later step is a fresh, independent run; a cascade succeeds if any step's run did). So the model is trained on **cascade rows**: the task's start state after `A` failed (built by `state_after_failure`, exactly as `ScoredPolicy` builds it), a stronger configuration `C` (the `enumerate_actions` rule), labelled with `C`'s outcome, only on tasks where `A` failed. `failed_attempts` becomes `n_failed`, the strongest failed tier, effort and probe rate, whether the model already failed, and how much stronger `C` is. In the synthetic data the model learns the right direction (`test_a_failed_earlier_attempt_lowers_the_next_step_in_the_right_direction`); the observed success rate of a later step is 55% against 67% for a first step.

**What it cannot support.**

- The effect of *seeing* the failure (a warm context, the error text, a shorter second run). The grid's second step is a fresh run. A real cascade could be better or cheaper than the data says.
- The cache and the switch tax. Grid runs never switch model inside a task, so `owns_cache`, `cached_frac` and the cache fields are almost constant within each kind of training row; the model cannot tell "a step failed" from "the cache is warm" and cannot learn what a warm cache does to success. Those fields are there for logged data.
- Histories longer than `max_history` (default 1, matching the default candidate depth of 2). A deeper policy extrapolates; retrain with `--max-history 2`.
- Logged traces. Their rows are per step with a step-level label, a different quantity from "the task was solved by this configuration". They are not mixed in. A later version can add them as rows with a level indicator.

**Rejected.** Ignoring `failed_attempts` (the unconditional probability, which makes every cascade look better than it is, because failures are correlated across models: `docs/POLICY.md`). A hand-set correlation discount (there is nothing to fit it on except these same pairs, so fit the pairs directly). Training on whole-task cascade outcomes (the policy asks about one step at a time).

### 4.4 Calibration

**Method.** `auto` (the default): `none`, Platt scaling (`sigmoid(a * margin + b)` on the booster's log-odds) and isotonic regression (pool-adjacent-violators, values kept off 0 and 1) are compared by 5-fold cross-validated log loss on the calibration tasks, folds by task, and a method replaces the identity only if it lowers the pooled held-out log loss by more than 0.001. Calibration is verified, not assumed. In this run `none` won (CV log loss none 0.578, Platt 0.586, isotonic 0.632), so the calibrated and uncalibrated models are the same model; `--calibration platt` and `isotonic` force a method and are compared in section 5.

**Fitted on.** The calibration tasks only: disjoint from train and tune (the trees have not seen them; the tune tasks chose the stopping round, so they are not used either) and from test. Probe-dropout is applied to those rows as to training rows, so one calibrator serves every probe count.

**Metrics** (`steproute ml evaluate`, `calibration.md`): expected calibration error with **10 equal-width bins** of the predicted probability (`sum_b n_b/n * |observed_b - mean predicted_b|`); Brier score; log loss; AUC; accuracy at 0.5; a reliability table; and, because the policy compares the prediction with a threshold `tau`, the gap *observed minus predicted* over rows within 0.05 of `tau` for tau 0.5, 0.7, 0.8, 0.9, with its count. Everything has a 95% **percentile cluster bootstrap** interval (300 resamples of whole tasks), paired between calibrated and uncalibrated for the differences. Rows from one task are correlated, so resampling rows would understate the width.

**Reading ECE.** It is biased upward at small n. For a perfectly calibrated predictor with 1,100 independent rows its median is 0.026 and its 95th percentile 0.043 (simulated); clustered rows push that higher. The reported 0.06 to 0.07 is above that, and the reliability table shows why: in the first step observed success is 4 to 8 points above the prediction in the 0.6 to 0.9 bins and below it in the 0.3 to 0.5 bins, so the predictions are a little compressed toward the middle.

**Rejected.** Calibrating on the tune split or on training rows (optimistic). Isotonic as the default (ties break the policy's tie-breaks, small samples overfit; in the section 5 comparison it is the worst). `CalibratedClassifierCV` (refits internally and stores a pickle). One calibrator per probe count (too few rows each).

### 4.5 Reproducibility

**Meaning.** The same grid, catalog, `TrainConfig` (seed included) and library versions give the same model: a byte-identical `model.txt` and identical predictions. **How.** One thread, `force_row_wise`, `deterministic=True`, every random stream seeded from the seed (`bagging_seed`, `feature_fraction_seed`, `data_random_seed`), row order independent of ids, tasks and rows drawn from `random.Random` seeded with strings (not Python's randomised `hash`), the booster trimmed to the best iteration and reloaded so the live model and the loaded one are the same object. **Test.** `test_same_seed_gives_the_same_model_in_separate_processes` trains twice with seed 11 and once with seed 12, each in a fresh process through the CLI, then compares: `model.txt` bytes equal, predictions on a fixed input set (fresh and cascade states) exactly equal, the different seed different. **Limits.** Not claimed across machines, LightGBM versions or thread counts other than one (the loader warns on a version difference). Versions used here: Python 3.12.13, LightGBM 4.7.0, NumPy 2.5.4 (macOS, arm64), recorded in every saved model.

### 4.6 Serving cost

`ScoredPolicy` calls `predict` once per distinct chain prefix: 555 times per decision on the default catalog at depth 2. The per-call path costs about 22 microseconds per prefix in total, which is too slow.

**Decision.** An optional `BatchPredictor` protocol (`predict_many(pairs)`), used by `ScoredPolicy` when the predictor has it (`docs/POLICY.md`); `Predictor` is unchanged. `TabularPredictor.predict` is `predict_many` of one pair, and `tests/test_policy_batch.py` and `test_predictor_works_inside_scored_policy_and_batch_equals_per_call` check that the batch path returns exactly what the per-call path returns (`==` on every candidate's cost and success).

**Measured** (`scripts/bench_tabular.py`; `decide`, default catalog, depth 2, 555 candidates, a warm-cache 20,000-token state with tools, no probe results so every configuration is cold-start; Apple M4 Pro, Python 3.12; 100 trees is about what early stopping keeps on real-sized data, 300 is the cap):

| | per-call `predict` | batch `predict_many` |
|---|---|---|
| 100 trees | p50 12.1 ms, p95 12.3 ms | p50 1.7 ms, p95 1.7 ms |
| 300 trees | p50 13.9 ms, p95 14.7 ms | p50 3.3 ms, p95 3.3 ms |

The batch path fits the few-millisecond budget; the per-call path does not. The batch time is about 0.6 ms of policy work, a feature matrix built once per decision, and one booster call. The synthetic model has 23 kept rounds; the benchmark times 100 and 300 trees to bracket real-data sizes. No test asserts on timing.

### 4.7 The saved model format

A directory, no pickle: `model.txt` (LightGBM text, trimmed to the best iteration), `meta.json` (format and version, the feature spec with every column name, the calibrator, the LightGBM parameters, the seed, library versions, the data fingerprint, the SHA-256 of `model.txt`, and the synthetic flag and stamp), `probes.json` and `roles.json`. Only file names, never directories, go into any of them.

`TabularModel.load` refuses, with a named error: an unknown format or format version; a `model.txt` whose hash differs from `meta.json`; a feature version other than the one this code builds; a saved column list that differs from the columns this code builds for the same spec (`FeatureSpecMismatch`); and a booster whose own column names differ from the spec. It warns on a LightGBM or NumPy version difference. Tests edit each of these in a saved model and check the refusal.

**Rejected.** Pickle or joblib (code execution on load; fragile across versions). LightGBM's own file alone (no feature spec, calibrator or provenance to check against). A model file in the repository: none is checked in; `models/` is ignored.

### 4.8 Scripts and CLI

`steproute ml train` and `steproute ml evaluate` (`steproute.ml.cli`; the arguments are stdlib-only, so `steproute --version` and `import steproute.cli` work without the extra, tested with `lightgbm` and `numpy` made unimportable; the command names the missing extra). `evaluate` writes `report.json` and `report.svg` (the evaluator's report: the predictor, calibrated and not, inside `ScoredPolicy` at each `--tau`, the rule baseline and the hindsight oracle, against the single-configuration frontier, scored only on the model's test tasks) and `calibration.json` and `calibration.md`. A custom catalog needs `--rules`; without it the rule baseline is skipped and the report says so. `scripts/synthetic_tabular_report.sh` regenerates the checked-in files in one command. End to end on a 500-task synthetic grid, train plus evaluate takes well under five seconds.

**Stamping.** Whether an output is stamped follows the data: the grid header names the generator, the model's metadata records it, and every output made from such a model or grid carries `SYNTHETIC DATA. NOT A RESULT.`: the JSON report (a `stamp` and `notice` field, and the start of every verdict and summary sentence), the SVG (title, diagonal watermark and a bottom line), the Markdown (heading, each section, last line), the model's `meta.json`, and the console output. A grid from a real run gets none (`test_the_stamp_follows_the_data_not_a_flag`). A test checks the checked-in files.

**Rejected.** A `python:MODULE:ATTR` hook for the stock `steproute eval` (it cannot carry the model path or the model's own test split, and would not stamp). Loose scripts outside the CLI. `scikit-learn` is in the `ml` extra but the code does not import it (the metrics, calibrators and fold splits are small and take lists, which also keeps the saved calibrator plain data); the maintainer may drop it from the extra.

## 5. Metrics (synthetic data, 190 test tasks)

Default model: seed 0, 23 boosting rounds kept, calibration `auto` (chose `none`). 95% task-cluster bootstrap intervals in brackets.

| | first step (1,140 rows, 190 tasks) | after one failure (1,213 rows, 134 tasks) |
|---|---|---|
| Observed success rate | 0.665 | 0.551 |
| Brier of the constant base rate | 0.2228 | 0.2474 |
| ECE (10 bins) | 0.068 [0.047, 0.098] | 0.062 [0.047, 0.112] |
| Brier | 0.162 [0.149, 0.177] | 0.197 [0.178, 0.216] |
| Log loss | 0.500 [0.470, 0.533] | 0.581 [0.538, 0.624] |
| AUC | 0.815 | 0.771 |
| Accuracy at 0.5 | 0.787 | 0.728 |

"Before" and "after" calibration are identical for the default model, because `auto` kept the identity. What forcing a method does to the same test rows (change = calibrated minus uncalibrated, negative is better, with the paired interval):

| Method | Group | ECE | Brier | Log loss |
|---|---|---|---|---|
| Platt | first step | 0.068 to 0.063 (-0.006 [-0.023, +0.028]) | 0.1618 to 0.1617 (-0.0002 [-0.0044, +0.0038]) | 0.5001 to 0.4996 (-0.0005 [-0.0102, +0.0086]) |
| Platt | after one failure | 0.062 to 0.033 (-0.028 [-0.044, +0.012]) | 0.1968 to 0.1917 (-0.0051 [-0.0102, +0.0007]) | 0.5811 to 0.5690 (-0.0122 [-0.0235, +0.0006]) |
| Isotonic | first step | 0.068 to 0.066 (-0.003 [-0.029, +0.034]) | 0.1618 to 0.1636 (+0.0017 [-0.0037, +0.0071]) | 0.5001 to 0.5031 (+0.0030 [-0.0129, +0.0199]) |
| Isotonic | after one failure | 0.062 to 0.053 (-0.008 [-0.043, +0.029]) | 0.1968 to 0.1950 (-0.0018 [-0.0096, +0.0064]) | 0.5811 to 0.6058 (+0.0247 [-0.0131, +0.0819]) |

Every interval contains 0: with 190 test tasks nothing here separates the three options, which is why `auto` keeps the identity unless cross-validation shows a gain. Platt moves the right way on every metric and is the only forced method that is not worse anywhere; isotonic is worse in three of the six cells. The tests check that the default and Platt are not worse than uncalibrated on a 900-task held-out set by ECE, Brier and log loss (within 0.002), and that `auto` picks a calibrator when the scores are made overconfident.

**Near the thresholds** (first step, observed minus predicted success; rows in the window): tau 0.5: -0.062 [-0.152, +0.031] (106); tau 0.7: +0.059 [-0.027, +0.146] (95); tau 0.8: +0.061 [+0.025, +0.099] (567); tau 0.9: no rows within 0.05. After one failure: tau 0.5: -0.085 (174); tau 0.7: -0.019 (263); tau 0.8: -0.012 (327), intervals about plus or minus 0.09. Near 0.8 the model is a little under-confident in the first step (observed above predicted by 6 points, interval above zero) and no single step is predicted above 0.9 in the test rows (the highest bin used is 0.8 to 0.9), so at `tau = 0.9` only cascades can qualify.

**The policy** (evaluator, grid, the same 190 test tasks, `report.json`; policy rows are the same with and without calibration here):

| Policy | Success | Spend per task | Margin over the frontier [95% interval] | Verdict |
|---|---|---|---|---|
| tabular, tau 0.7 | 80.5% | $1.101 | +3.2 pts [-1.7, +8.6] | not shown (interval contains 0) |
| tabular, tau 0.8 | 84.2% | $1.364 | +5.2 pts [+1.4, +10.4] | beats the frontier |
| tabular, tau 0.9 | 88.4% | $1.618 | +8.0 pts [+5.2, +12.0] | beats the frontier |
| rules | 86.3% | $1.057 | +9.2 pts [+7.0, +13.7] | beats the frontier |
| oracle (hindsight, not a policy) | 94.2% | $0.487 | +20.7 pts | reference |

The rule baseline does better than the learned predictor here. In this synthetic world the rules key on the hint through the catalog's tiers and the costs are artificial, and the world flatters any cascade (failures are independent given difficulty). Do not read this as evidence about either method. That the tabular policy "beats the frontier" is a statement about the generator.

**A held-out model** (sanity check of section 4.1, one seed, not issue 26's experiment): trained with `cloud-mid` given no training rows (`--holdout-model cloud-mid`), its first-step rows on the test tasks, calibrated model:

| Probes describing `cloud-mid` | Mean predicted | Observed | Brier | AUC within the configuration |
|---|---|---|---|---|
| 0 (cold start) | 0.535 | 0.726 | 0.194 | 0.799 |
| 5 | 0.562 | 0.726 | 0.185 | 0.799 |
| 20 | 0.562 | 0.726 | 0.185 | 0.799 |
| 50 | 0.560 | 0.726 | 0.186 | 0.799 |

The model ranks tasks well for a model it never saw (AUC 0.80, which comes from the task feature), but it under-predicts the model's level by about 16 points, and probes recover only about 2 points of it. With six configurations in training there are six distinct probe rates for the trees to learn from, which is far too few to learn a smooth map from probe rate to success. This is the weakness the graph model is supposed to fix, and also a warning that this baseline's cold-start numbers on a pool of 10 real models will be weak for a structural reason.

## 6. Limitations

- Everything above is synthetic. The only task feature is a noisy difficulty hint; the real grid has text, repositories, tools, languages. Nothing here tests those.
- The label is the first labelled run of a cell; repeats are unused. Unlabelled runs give no row (missing at random is assumed).
- Probe results describe a configuration only through a fixed set of tasks drawn from the training pool; a probe set that is unrepresentative of the test tasks is a bias this baseline cannot detect.
- The calibrator is fitted on the training mixture of probe counts. Calibration at a particular count (5 probes, say) is not verified separately.
- Cache and switch effects are not learned (section 4.3), prices ignore `price_tiers`, and `tier` is a human prior.
- Hyperparameters (learning rate 0.05, 15 leaves, minimum 40 rows per leaf, bagging 0.8, 300 rounds with early stopping after 30) were fixed before any evaluation. A sweep of seven settings over three seeds, read on test tasks, differed by less than the noise, so nothing was selected from it.
- Results are for one seed. The intervals are over test tasks, not over training randomness.
- No external baseline (issue 22), no graph model, no real data (issue 23).

## 7. What must be redone when real data arrives

1. Retrain on the real grid (`steproute ml train --data grid.jsonl`), with the real splits of issue 19 (`train`, `validation`, `test` are respected when present) and `--group-feature` for the stratum or repository so groups stay whole.
2. Decide the probe set on purpose rather than by a seeded draw, run it for every configuration, and check that it is representative. Every method compared in issue 23 must score the same test tasks, with the probe tasks excluded for all of them.
3. Rebuild the feature spec on the real task features, and check each of them: which exist at routing time, which leak the label, which are strings to encode.
4. Re-check calibration with the real tier of data (a few hundred tasks): the `auto` choice, the threshold windows at the `tau` the product uses, and the calibration at 0, 5, 20 and 50 probes.
5. Re-measure `decide` latency with the real feature set and tree count (`scripts/bench_tabular.py`).
6. Re-examine section 4.3 against real cascades: if the replay includes real switching runs, add them as rows so the cache and the effect of a seen failure can be learned.
7. Regenerate or delete everything under `docs/synthetic/tabular-predictor/`, and report on real data only in `docs/results/` (issue 23), unstamped because it is not synthetic.

## 8. Reproduce

```
uv sync --locked --extra ml
scripts/synthetic_tabular_report.sh                     # regenerates docs/synthetic/tabular-predictor/
uv run --extra ml python scripts/bench_tabular.py       # the latency table
```

Or by hand: `steproute ml train --data grid.jsonl --catalog catalog.toml --out models/tabular --seed 0`, then `steproute ml evaluate --model models/tabular --data grid.jsonl --catalog catalog.toml --rules rules.toml --out-dir OUT`. The report is reproducible apart from its `generated_at` field.

**SYNTHETIC DATA. NOT A RESULT.**
