# External baselines (issue 22)

*Checked 14 Nov 2026. No real replay data existed yet, so every report linked here comes from the evaluator's synthetic world and is stamped **SYNTHETIC DATA. NOT A RESULT.** The comparison on real data is issue 23.*

Four published routers were to run under our evaluator, on our data and our model pool. Three run as adapters. The fourth gets the note in section 6.

| Baseline | Here | Upstream, pinned | Licence | Main fairness caveat |
|---|---|---|---|---|
| ParetoBandit | adapter, `steproute.baselines.paretobandit` | `ParetoBandit/ParetoBandit` `72ccbefe`; PyPI `paretobandit==0.1.1` (tag `v0.1.1`, `e20a5979`; `src/` identical) | Apache-2.0 (dependency, not vendored) | Linear in our numeric state, not in a prompt embedding. Its budget pacer is not used. |
| LLMRouter `graphrouter` | adapter, `steproute.baselines.graphrouter` | `ulab-uiuc/LLMRouter` `c65ee012` (`llmrouter-lib` 0.4.0); package not installed | MIT (GNN file adapted, notice kept) | Query nodes carry numeric state instead of Longformer text embeddings, and LLM nodes carry our profile and probe features. |
| RouteLLM (`mf`, two models) | adapter, `steproute.baselines.routellm` | `lm-sys/RouteLLM` `0b64fdaf` (`routellm` 0.2.0); package not installed | Apache-2.0 (MF code adapted, notice and licence kept) | It can only pick between two configurations out of the pool. It is trained on battles built from task success, with no OpenAI embeddings. |
| OpenSquilla LightGBM router | **note**, section 6 | `TokenRhythm/opensquilla` `fdc3898a` (v0.5.8 + 9) | Apache-2.0 | Not run. It is frozen and reads text only, and we have no prompt text. Its training code was removed upstream. |

Every adapter was **retrained from scratch on our training tasks**. No pretrained weights, shipped priors, PCA artefacts, text encoders or paid APIs are used anywhere.

## 1. Running them

```bash
uv sync --locked --extra dev --extra ml --extra baselines   # project-local, not in scripts/ci.sh
scripts/synthetic_baselines_report.sh                       # regenerates docs/synthetic/baselines/
uv run --extra ml --extra baselines python -m steproute.baselines all \
  --data grid.jsonl --catalog catalog.toml --rules rules.toml --out-dir out/
```

`python -m steproute.baselines {paretobandit,graphrouter,routellm,all}` trains on the grid and writes `out/<baseline>/report.json` and `report.svg`. Each report holds the baseline's policies (one per point of its own cost knob), the rule baseline and the hindsight oracle, scored against every single configuration on the `test` tasks. Other options:

- `--seed`: split, training and bootstrap seed. Default 0.
- `--probes`: probe-set size, as for the tabular model. Default 50.
- `--bootstrap`: evaluator replicates.
- `--exclude-config EFFORT:MODEL`, repeatable: drops a configuration from the arms and the frontier (section 7).
- `--pair WEAK,STRONG`: RouteLLM only.

A synthetic grid gives stamped output. On the synthetic grid of 1,000 tasks, all three take about two minutes on a laptop CPU. Two runs give the same files except `generated_at` (checked on this machine).

## 2. What every baseline gets, and what it does not

**Same split as ours.** `BaselineData` draws task roles with `steproute.ml.splits.assign_roles`, using the same seed, probe count and fractions as `steproute ml train`. A baseline and the tabular predictor trained with the same seed are therefore scored on the same `test` tasks. The baselines fit on `train`. They use `tune` where the method has a validation step: GraphRouter for its best epoch, and RouteLLM only to report pairwise accuracy. `calibration` and `test` are never read in training. Probe tasks only describe configurations (GraphRouter's LLM nodes). A task is used only when every arm's cell is complete, which is also what the evaluator's headline requires.

**Inputs: numeric state, no text.** All three published routers read prompt text through an encoder:

| Router | Upstream encoder |
|---|---|
| ParetoBandit | MiniLM + PCA |
| GraphRouter | Longformer |
| RouteLLM | OpenAI `text-embedding-3-small`, a paid call |

Our grid publishes no prompt text (`docs/PRIVACY.md`, `docs/TASKSET.md`), and the synthetic world has none. `ContextEncoder` is the substitute every adapter uses:

- It takes the tabular predictor's state block (`FeatureSpec.state_names`): step kind, prompt and cache sizes, tools, step index, sensitivity, each numeric `state.features` key, and the failed-attempt summary.
- Columns that are constant on the training tasks are dropped. The rest are z-scored on the training tasks.
- Each column holding at most 8 distinct integers also gets a one-hot copy, so a linear or factorised model can use a difficulty hint non-monotonically.
- It never reads `task_text`, `task_id` or `session_id` (tested).

The baselines get the same task information as the tabular model. Nothing more, and nothing in a form tuned to them.

**What ours gets that they do not:**

- The tabular predictor describes each configuration by its catalog profile and probe results, so it can score a configuration it never trained on (cold start). ParetoBandit and RouteLLM learn one parameter vector per arm. They cannot generalise to a new configuration, and they need training rows for every arm. GraphRouter gets our profile and probe features as its LLM node features, so it gets the same configuration information as ours.
- `ScoredPolicy` minimises expected cost subject to a success threshold over **cascades**, and it conditions on earlier failures. Every baseline picks **one configuration per task**, because that is what each method does.
- Upstream they are query-level routers, so "per step" here means "per routing state". Each adapter is an ordinary `TargetPolicy` and answers for any `StepState`, so the logged-trace evaluator can score it too (tested). It was trained at task level only, though. On the grid, kind and step index are constant, so they are dropped, and the step state is reduced to the task features.

**What they get that ours does not:** nothing in information terms. One structural edge: each adapter's cost knob is swept, and every point is reported as its own policy. Those knobs are λ_c for ParetoBandit, β for GraphRouter, and the strong share for RouteLLM. The tabular policy is reported at its τ values the same way. Neither side is tuned on `test`.

**Effort is routed.** Arms are `(model, effort)` configurations, not models. Each method's per-model parameters become per-configuration parameters.

## 3. ParetoBandit

*Dependency `paretobandit==0.1.1` (PyPI, wheel hash in `uv.lock`). Its installed `pareto_bandit/` is byte-identical to `src/pareto_bandit/` at `72ccbefe` (checked with `diff -r`). Tag `v0.1.1` (`e20a5979`) differs from that commit only outside `src/`. Paper: arXiv 2604.00136.*

**Method.** Disjoint LinUCB, one ridge regression per arm. The arm picked maximises `x·θ_m + α·sqrt(x' A_m⁻¹ x) − λ_c·normcost(m)`, where `normcost` is the arm's log price between two anchors.

**Mapping.**

- **Arms** are configurations. An arm's price is its **mean realised cost per task on the training tasks**. It is registered as the per-1k price, so `normcost` is the log of that mean between the cheapest paid arm and the dearest arm, which is the paper's "cost range anchored to arm cost extremes". A free local arm sits at the floor.
- **Context** is `ContextEncoder` plus the bias term the library expects last (`FeatureService.for_precomputed`).
- **Reward** is task success, 0 or 1.

**Training, two variants, both on `train`:**

- `full`: every arm is updated with its own outcome on every training task. This is the library's offline warm start, with the grid's full information.
- `online`: `route()` then `process_feedback()`, one training task at a time in a seeded order, so only the picked arm's outcome is seen. This is the bandit feedback the method was built for.

The router is then frozen, and the evaluator asks it for one arm per state through the public `route()`. Answers are cached per state, so the library's random tie-breaking cannot make the policy non-deterministic.

**Hyperparameters.**

| Parameter | Value | Why |
|---|---|---|
| α | 0.01 (`full`), 0.05 (`online`) | The library's own tuned values for warm and cold start (`BEST_K3_HPARAMS`, `BEST_K3_TABULA_RASA_HPARAMS`) |
| `init_lambda` | 1.0 | Default |
| `forgetting_factor` | 1.0 | The grid is stationary. The default 0.997 is for drifting traffic and would discount the early training tasks for no reason. |
| `priors` | `"none"` | The shipped priors are for three other models and a MiniLM-PCA space |
| λ_c sweep | 0, 0.1, 0.3, 0.5, 1.0 | 0.3 is the library default |
| Budget pacer | Not used | See below |

**Not used: the budget pacer.** It adjusts λ_c online to hit a spend target. A frozen policy scored on a fixed test set has no stream to pace, so the λ_c sweep traces the curve the pacer would move along. Pacing could be judged on logged step data with a budget, which we do not have yet.

**Caveat seen on synthetic data.** The `online` variant can lock onto one arm early, which is the usual cold-start behaviour of LinUCB with a small α. This is the method at its own tuned settings, not a bug in the adapter. On real data, with fewer training tasks, expect it more often.

## 4. LLMRouter `graphrouter`

*Adapted from `llmrouter/models/graphrouter/graph_nn.py` at `c65ee0122bc9e84f71d571fcdec4a948481b6e88` into `src/steproute/baselines/_vendor/graphrouter_gnn.py`. The MIT notice is in the file and in `_vendor/LICENSE-LLMRouter`. The data preparation of `router.py` is re-done in `steproute/baselines/graphrouter.py`. Paper: arXiv 2410.03834.*

**Why adapted, not installed.** `llmrouter-lib` pulls in gradio, litellm, transformers, datasets and peft. `import llmrouter.models` imports every router, and `route_batch` calls model APIs. Only the GNN (torch and torch-geometric) is needed.

**Changes to the GNN file.** None of them alters the model or the training step:

- The forward pass is split into encode and score. A new query is then scored against stored LLM node states. In eval mode this equals upstream `predict` with the query appended to the graph (tested: same argmax).
- The best epoch is kept in memory instead of in a file.
- The random edge masks come from a seeded generator.
- Nothing is printed.

**Mapping.**

- **LLM nodes** are configurations. Their features are the tabular model's configuration block: tier, local, prices, context window, effort rank and multiplier, provider, and probe success and cost on the probe tasks. They are min-max scaled as upstream scales its LLM embeddings. Upstream uses a text embedding of a model description, or random features.
- **Query nodes** are tasks, with `ContextEncoder` features min-max scaled **on the training queries**. Upstream refits the scaler on training plus test queries. That would make one test state's answer depend on the other test states, so it was not copied.
- **Edges** carry upstream's cost-aware performance (`benchmark_pipeline/run_pipeline.py`): `α·norm(success) − β·norm(cost)`, min-max over the training edges, with α = 1 − β. Labels are the one-hot argmax per task; ties go to the cheaper tier and effort.
- **Validation** edges are the `tune` tasks. Upstream draws 20% of its training queries at random instead.

**Hyperparameters.** Upstream defaults from `configs/model_config_train/graphrouter.yaml`: hidden 64, AdamW lr 1e-3, weight decay 1e-4, 100 epochs, 4 masked samples per step, edge mask rate 0.3. The β sweep is 0, 0.2, 0.4, 0.6, 0.8; upstream's example is α 0.6, β 0.4. Training runs on the CPU, seeded.

**Caveat (structural, holds on real data).** With 0/1 success, the training label is "the cheapest configuration that solved this task" for every β < α. Only β > α can prefer a cheap failure, so the lower half of the sweep barely moves the policy. A per-task graded outcome (tests passed, partial credit) would give β more grip; the grid has none.

## 5. RouteLLM (two-model setting)

*Adapted from `routellm/routers/matrix_factorization/{train_matrix_factorization.py,model.py}` at `0b64fdafe049e596a3f5657c219329f24af24198` into `src/steproute/baselines/_vendor/routellm_mf.py`. The Apache-2.0 notice and the list of changes are in the file, and the full licence is `_vendor/LICENSE-RouteLLM`. The upstream repository has no NOTICE file. Paper: arXiv 2406.18665.*

**Why adapted, not installed.**

- `routellm` pins `numpy<2`, which would downgrade the `ml` extra's numpy in the shared lock.
- Importing its MF model creates an OpenAI client at import time.
- The MF model embeds every prompt with OpenAI `text-embedding-3-small`, a paid call that is not allowed here.
- The training code for the `bert` and `causal_llm` routers is not in the repository.

So only `mf`, the paper's best router, is run.

**Changes.**

- The prompt table holds `ContextEncoder` vectors, and inference takes one: no network and no OpenAI client.
- The model table spans our configurations instead of the fixed 64-model Arena table.
- The training loop is seeded and runs on the CPU.

**Mapping.**

- **Battles:** on each training task, every ordered pair of configurations where one succeeded and the other failed is one battle, won by the configuration that succeeded. Both-succeeded and both-failed pairs are ties and are dropped, as upstream drops ties. Upstream trains on Chatbot Arena human preferences plus judge-labelled data. Here a "win" is task success, which is a different signal.
- **MF:** the model is trained on all configurations, as upstream trains on 64 models, and then asked about one pair.
- **The pair:** weak is the cheapest configuration by mean training cost (in our pool, usually a free local model). Strong is the configuration with the highest training success rate, ties going to the cheaper. `--pair` overrides both.
- **Threshold calibration:** as upstream calibrates on Arena prompts, each strong share s in {0.1, 0.25, 0.5, 0.75, 0.9} gets as its threshold the 1 − s quantile of the training tasks' strong win rates. One policy per share.

**Hyperparameters.** Upstream's example values: dim 128 with projection, Adam lr 3e-4, weight decay 1e-5, input noise 0.1, batch 64, 100 epochs.

**Caveat.** By construction it uses two of the pool's configurations, so it cannot reach a frontier point that needs a third. That is the setting the issue asks for. A multi-model use of the same MF scores (argmax over arms) would be a different router, and it is not presented as RouteLLM.

## 6. OpenSquilla's LightGBM router: not run fairly, and why

*Checked at `TokenRhythm/opensquilla` `fdc3898a31432b48a1731bfe663a2b6f44223e88` (v0.5.8 plus 9 commits, 2026-11-10), Apache-2.0, `requires-python >= 3.12`. Paper: arXiv 2607.11399. The repository `opensquilla/opensquilla` redirects there.*

**What it is.** The open router is a **frozen** LightGBM tier classifier. It maps a request to one of four tiers, C0 to C3, through a fixed 380-feature pipeline (`squilla_router/lightgbm_model/_vendor/features/lightgbm380.py`):

- 64 + 64 + 64 BGE-embedding PCA dimensions, for the current request, the history and the previous answer.
- 100 character TF-IDF→SVD dimensions of the request.
- 88 "numeric" features (`numeric88.v1`). Most of these are computed from text: lengths, language ratios, code blocks, keyword counts, and properties of the previous answer.

Since the 2026-11 releases, a BERT-60M ONNX intent classifier also runs beside it. The tier weights (`assets/tier/model.txt.gz`, 12.2 MB) are in Git LFS. The BGE ONNX encoder (24 MB), the intent model (60 MB) and the joblib-pickled scikit-learn transformers are listed in `assets/model_manifest.json`. `PROVENANCE.md` records that the training code (`router_training`) was removed: only the inference subset ships.

**Why it cannot be run fairly on our setting:**

1. **It needs prompt text, and we publish none.** The large majority of its 380 inputs are functions of the text. The others are route-history features: previous tier, turn index, switch counts, cache ratio, context size. They partly map to our `StepState` (`step_idx`, `failed_attempts`, `cached_prefix_tokens`, `prompt_tokens`). Feeding zeros or constants for the text blocks would put a frozen model far outside the distribution it was trained on, and score an artefact, not the router.
2. **It cannot be retrained on our data.** The training code is not public, so the only possible run is zero-shot with weights trained on OpenSquilla's own pool and traffic. Pretrained weights tuned on other models are not a fair baseline by default.
3. **Its output is a tier, not a configuration.** Using it means writing a C0–C3 → `(model, effort)` mapping for our 35 configurations by hand. Its own mapping (`opensquilla.toml [squilla_router.tiers]`) names other models. That mapping would decide most of the result.
4. **It needs Python 3.12+ and loads pickled artefacts**, so it needs a separate environment and a trust decision about unpickling downloaded files. That alone would not block it.

**What was tried.** The repository was cloned without LFS objects and its inference path, feature registry, manifests and provenance were read. Nothing was installed or executed, and no weights were downloaded. All of its artefacts together are well under 1 GB, so size was not the reason.

**What would make it possible:**

- **Text.** Our tasks are public SWE-bench Verified problem statements (`docs/TASKSET.md`), and they could be joined locally by task id to give the router its `current_request`. With the step-level text it would also see history.
- **A frozen tier mapping**, written down before any result is seen, for example by tier and effort rank.
- **A separate Python 3.12 environment** with the v0.5.8 release wheel and `git lfs pull` for the tier weights.

It would then be a clearly labelled **zero-shot, pretrained-elsewhere** reference, not a retrained baseline. The maintainer has to decide whether that is worth having (section 8).

## 7. Real data: what cannot be judged yet

- **Every number in `docs/synthetic/baselines/`.** The synthetic world has one informative task feature (a noisy difficulty hint) and six configurations, so these files test the plumbing and nothing else.
- **Coverage.** The approved grid is sparse: each paid model at its default effort on all 150 tasks, and its other efforts on 30 tasks only (`docs/TASKSET.md`, option B). A baseline needs every arm on every task it trains on. On the real grid, run it on the dense part, with `--exclude-config` for each configuration that ran on the 30-task subset only. Alternatively, run it once on the 30 dense tasks, which is too few to train on.
- **Training size.** About 55% of 150 tasks is roughly 80 training tasks. That is small for every one of these methods, and for ours. Expect wide intervals and `not_shown` verdicts.
- **Which real task features exist.** `ContextEncoder` uses whatever numeric `features` the grid's task lines carry. The pilot's grid now exists (`docs/results/pilot.md`), but its task features have not been examined: it is not known how informative they are, or whether any of them leak the label. The tabular model card asks the same question (`docs/MODEL-CARD-TABULAR.md`).
- **Whether RouteLLM's default pair is sensible.** On the real pool it will probably be a local model against a top Claude configuration.

## 8. Decisions for the maintainer

1. **OpenSquilla.** Run it as a zero-shot, text-based reference with public SWE-bench problem statements and a pre-registered tier mapping, or leave the note as the final answer. The note's recommendation is to leave it, unless the comparison is meant to include text-reading routers.
2. **Sparse grid.** For issue 23, choose between excluding the 30-task configurations (more tasks, fewer arms) and a separate run on the dense 30 tasks.
3. **ParetoBandit online variant.** Keep both variants in issue 23, or report only `full`. `online` answers "how well does it learn from bandit feedback", which our grid can only simulate.

## 9. Dependencies and pins

The `baselines` extra holds `paretobandit==0.1.1`, `torch>=2.2` and `torch-geometric>=2.5`, resolved in `uv.lock`. Torch 2.15.0 and torch-geometric 2.8.1 were already in the lock through the `graph` extra; the only new locked package is `paretobandit` 0.1.1. Installed into the project environment, the extra added about 635 MiB, almost all of it torch. It is imported lazily, mypy treats it as optional, and `scripts/ci.sh` neither installs it nor needs it. Its tests skip without it. Nothing else was downloaded at run time: no model weights, no encoders, no datasets.

| Upstream | Repository | Commit (date) | Version | Licence | Use here |
|---|---|---|---|---|---|
| ParetoBandit | https://github.com/ParetoBandit/ParetoBandit | `72ccbefe5cb858c95e00368e2b68de3c2b28a2ee` (2026-04-13) | PyPI `paretobandit` 0.1.1 = tag `v0.1.1` `e20a597963681ec0c506b6dbce05447b18fa7145` | Apache-2.0 | Dependency |
| LLMRouter | https://github.com/ulab-uiuc/LLMRouter | `c65ee0122bc9e84f71d571fcdec4a948481b6e88` (2026-10-08) | `llmrouter-lib` 0.4.0 | MIT | `graph_nn.py` adapted |
| RouteLLM | https://github.com/lm-sys/RouteLLM | `0b64fdafe049e596a3f5657c219329f24af24198` (2024-08-10) | `routellm` 0.2.0 | Apache-2.0 | MF model and loop adapted |
| OpenSquilla | https://github.com/TokenRhythm/opensquilla | `fdc3898a31432b48a1731bfe663a2b6f44223e88` (2026-11-10) | v0.5.8 + 9 | Apache-2.0 | Note only |

These commits are the upstream HEADs on 14 Nov 2026. They match the ones recorded in `docs/INTEGRATION-NOTES.md` (sections 5 and 6), so nothing moved between that research and this issue.
