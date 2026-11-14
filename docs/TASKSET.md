# Task set for the replay grid (issue 17)

*Checked 13 Nov 2026. Prepared for Gate A. Nothing in this file was run against a paid model. Section 8 records what issue 56 then ran on the real machine the same evening: the harness, the pilot images and a local model.*

## Gate A decision

Decided by the maintainer on 13 Nov 2026:

- **Task set approved:** the 150 SWE-bench Verified tasks in the manifest, with the 20-task pilot drawn from them.
- **Grid: option B, the sparse grid.** Every local configuration and each paid model at its default effort on all 150 tasks, plus the other paid efforts on 30 tasks.
- **No run is approved yet.** The pilot and every later batch need a separate go-ahead, and the option B estimate is to be redone from the pilot's measured costs before month 2.

Decided by the maintainer on 13 Nov 2026, after Gate A (issue 57):

- **DeepSeek has $10 in total**, funded once. It is not a monthly allowance. The runner enforces it as a total budget per ledger, next to a $10 monthly cap.
- **Every DeepSeek call runs off-peak** (outside 01:00-04:00 and 06:00-10:00 UTC on weekdays), at half price. The runner starts a DeepSeek run only when off-peak lasts at least the 2-hour run timeout, and never sends a DeepSeek call in peak hours.
- **No DeepSeek configuration is dropped from the pilot.** Its DeepSeek block (100 runs) is $17.57 central at peak, $8.79 off-peak: it fits in $10 at the central estimate, not at the high one ($44.43 off-peak), where the budget stops DeepSeek and the rest stays pending.

The rest of this section is the material the decision was made from.

## The decision to make

The task set is **150 SWE-bench Verified tasks** with native arm64 evaluation images, and a **20-task pilot** drawn from them (`src/steproute/tasks/manifests/swebench-verified-v1.toml`). The open question is how much of the `(task, model, effort)` grid to run on them. The pool has 35 configurations:

- **Claude:** 4 models × 5 efforts = 20.
- **DeepSeek:** 2 models × 4 efforts = 8.
- **Local:** 7 (6 in the pool plus the held-out `devstral:24b`). Local runs are free and run in every option.

Spend is estimated at three points: **low / central / high**. Months are at the D3 caps: Anthropic $80 a month, and DeepSeek $20 a month, which was an assumption past the pilot when this table was made. Since 13 Nov 2026 DeepSeek has $10 in total and runs off-peak only (decision above), so the DeepSeek months columns no longer describe a plan: beyond the pilot, DeepSeek needs new funding.

| Option | Tasks × configurations | Anthropic $ (runs) | Months at $80 | DeepSeek $ peak (off-peak ≈ half) | Months at $20 | Local compute |
|---|---|---|---|---|---|---|
| **A. Dense grid** | 150 × all 35 | 775 / **1,321** / 4,558 (3,000) | 10 / **17** / 57 | 189 / **356** / 1,641 | 10 / **18** / 83 (off-peak 5 / 9 / 42) | about 111 h |
| **B. Sparse grid** | 150 × (every local configuration, plus each paid model at its default effort), plus the other paid efforts on 30 tasks | 266 / **456** / 1,600 (1,080) | 4 / **6** / 21 | 73 / **138** / 640 | 4 / **7** / 32 (off-peak 2 / 4 / 16) | about 111 h |
| **C. Smaller dense** | 60 × all 35 | 310 / **529** / 1,823 (1,200) | 4 / **7** / 23 | 76 / **142** / 657 | 4 / **8** / 33 | about 44 h |
| **Pilot (any option)** | 20 × the pilot configurations (below) | 23 / **41** / 179 (180) | 1 / **1** / 3 | 9 / **18** / 85 | off-peak 4.70 / **8.79** / 44.43 against the $10 total | about 15 h |

`steproute tasks estimate` reproduces every number in this table (see "The cost model").

**What each option lets the later experiments conclude**

- **A. Dense grid**
  - *Allows:* everything issue 20 assumes. A full-information evaluation of every single configuration, the complete single-configuration frontier, and experiment 2 (model vs. model plus effort) at full power, on 150 tasks. Leave-one-out cold start (issue 26) over every model.
  - *Does not allow:* finishing soon enough. At the central estimate it takes 17 months, and 57 at the high one. By then the Claude and DeepSeek snapshots will have changed, so the matrix would mix model versions, and the deprecation of any pool model breaks it.
- **B. Sparse grid**
  - *Allows:*
    - A complete matrix at the model level: every model at its default effort, and every local configuration, on all 150 tasks. That is enough for the model-level frontier, the rule, LightGBM and external baselines (issues 21 and 22), and the cold-start experiment. The held-out `devstral:24b` and all local models are dense.
    - Effort curves for every paid model on 30 tasks, which is enough to measure effort multipliers and to see whether effort routing ever pays.
  - *Does not allow:* experiment 2's effort comparison with tight confidence intervals. With 30 tasks a 10-point difference in success rate is within noise. If the 30-task sweep shows effort routing matters, extend the sweep in later months.
- **C. Smaller dense**
  - *Allows:* the full-information evaluation for every configuration.
  - *Does not allow:* useful splits. 60 tasks split into train, validation and test (issue 19) leaves about 12 test tasks: too few to rank policies, train LightGBM or a GNN, or report per-stratum results. It costs more than B for less.

**Recommendation.** Approve the subset and the pilot now. Choose **option B** for the full grid, with its spend re-estimated from the pilot before month 2.

- The pilot costs about $41 of Anthropic credit (worst case: the $80 hard cap stops it) and about $8.79 of DeepSeek, which runs off-peak only, out of its $10 total.
- Option B buys the model-level dense matrix, which every downstream experiment needs, in about 6 months (4 to 21). Option A needs 17 months (10 to 57), and Option C costs more and gives worse splits.
- The ranges are wide because nobody has published runs of *these* model versions on this scaffold. The pilot exists to replace the assumptions.
- Two preconditions belong to issue 18, and the pilot should not start without them:
  1. **Anthropic prompt caching must actually reach the provider through the proxy.** Without it every option costs about 3.2 times as much: the pilot would be $157 central, which does not fit.
  2. **The scaffold must be mini-SWE-agent or equivalent.** The estimates are calibrated on it.

---

## 1. Candidate benchmarks

Checked 13 Nov 2026. Sources are linked. "Secondary" marks a claim seen only in press coverage of a primary source that I could not open.

| Benchmark | Size | Check | Licence (tasks; repos) | arm64 images | Saturation and contamination (2026) | Free difficulty signal |
|---|---|---|---|---|---|---|
| **SWE-bench Verified** | 500, 12 Python repos | FAIL_TO_PASS / PASS_TO_PASS tests in Docker; the dataset carries `eval_script` and `image` since 2026-08-16 | Harness MIT; HF card has no licence tag; repos permissive except pylint (GPL-2.0) | **Yes:** 281 official `swebench/sweb.eval.arm64.*` (Docker Hub, pushed 2025-04-20) and 420 from Epoch AI (GHCR, "best-effort … not tested") | **Saturated and contaminated.** OpenAI stopped reporting it on 2026-02-23 (flawed tests; models reproduce gold patches; secondary). Vals AI archived it near 97%. The official bash-only board tops out at 76.8% | **Yes:** human `difficulty` buckets, plus per-instance resolved flags and API-call counts for 182 submissions in SWE-bench/experiments |
| SWE-bench Verified Mini | 50 (django, sphinx) | as Verified | MIT generator; BSD repos | Yes (Epoch, 50/50) | as Verified | Built to match Verified's difficulty distribution |
| SWE-bench Lite / full | 300 / 2,294 | as Verified | as Verified | Epoch: 1,819 of 2,294 on arm64 | Older, as contaminated; Lite board stopped in September 2025 | Experiments repo (lite, test) |
| SWE-bench Multilingual | 300, 9 languages | Docker tests | MIT; repos mixed, some non-permissive (redis, terraform) | None found | 72.7% top (bash-only) | No difficulty column |
| SWE-bench Multimodal | 480 test (JavaScript) | Docker tests | MIT | None found | 36% top | None per instance |
| **SWE-bench Pro** (Scale) | 731 public (v1); v2 642 + HARD-51 (2026-09-22) | Docker tests, locked protocol | Harness MIT; **public repos chosen for strong copyleft (GPL / AGPL)** | **amd64 only** (`ghcr.io/scaleapi/...`) | Not saturated (about 61%), but OpenAI's 2026-07-08 audit flagged 27-34% of public tasks as broken (secondary) | `hard` and category columns; trajectories for about 9 models |
| **SWE-rebench** (Nebius) | Monthly splits, 860 tasks to 2026-03; latest window 111 | Docker tests (their SWE-bench fork) | CC-BY-4.0; per-instance repo licence given | **amd64 only** | Not saturated (Fable 5 64.5%); decontaminated by date | Leaderboard aggregates only; no per-task results found |
| SWE-bench-Live (Microsoft) | 1,000 test plus monthly | Docker tests | MIT; repos mixed | amd64 only found | Fresh monthly tasks | Per-instance results not found |
| Multi-SWE-bench, SWE-PolyBench | 1,632 / 2,110 | Docker tests | Apache-2.0 / MIT | Not found / amd64 only | — | — |
| Terminal-Bench 2.x / 3 / 4 | 89 / 74 / … | Container tests | Apache-2.0 | Not determined | 2.1 called saturated | Public trajectories required |
| DeepSWE (Datacurve, 2026) | 113 | Verifier programs | Apache-2.0 with a canary string (must not enter training corpora) | Not determined | New; adopted by Artificial Analysis | Not found |
| SWE-smith, SWE-Gym | 59K / 2.4K | Docker tests | MIT | amd64 | Training sets, no leaderboard | — |

**Sources**

- SWE-bench:
  - Docs: https://www.swebench.com/SWE-bench/guides/docker_setup/
  - Dataset card: https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified (revision `78f471bf`, 2026-08-16)
  - Harness 5.0.2 (PyPI, 2026-08-18): `run_evaluation.py` and `utils.make_test_spec` read the image name from the row (`image=instance["image"]`)
  - Docker Hub arm64 repositories: listed by name prefix
  - Per-instance results: https://github.com/SWE-bench/experiments at `40f164d5`
- Epoch AI: https://github.com/epoch-research/SWE-bench and https://epoch.ai/blog/swebench-docker
- Retirement and audits:
  - OpenAI, "Why we no longer evaluate SWE-bench Verified" (2026-02-23; page returned 403, so this rests on secondary coverage at https://the-decoder.com)
  - Vals AI: https://www.vals.ai/benchmarks/swebench
  - METR on unmergeable passing PRs (2026-03-10): https://metr.org/notes/2026-03-10-many-swe-bench-passing-prs-would-not-be-merged-into-main/
- SWE-bench Pro: https://scale.com/blog/swe-bench-pro and https://github.com/scaleapi/SWE-bench_Pro-os
- SWE-rebench: https://swe-rebench.com and https://huggingface.co/datasets/nebius/SWE-rebench-leaderboard
- SWE-bench-Live: https://github.com/microsoft/SWE-bench-Live
- Terminal-Bench: https://www.tbench.ai/news
- DeepSWE: https://github.com/datacurve-ai/deep-swe

### Why SWE-bench Verified, despite saturation

It is the only candidate that meets all four of the hard requirements at once:

- an automatic test-based check;
- **native arm64 images**, so it runs on the M4 Pro without emulation;
- permissive repositories, so traces containing their code can be published;
- **free per-instance results for dozens of models**, so difficulty can be stratified without spending anything.

Saturation matters less here than it does for a leaderboard:

- The router needs *spread across our pool*, and that spread exists. In public runs on the selected 150 tasks, a weak open model (Qwen2.5-Coder-32B) resolves 7%, Claude Haiku 4.5 55% and Claude Opus 4.5 75%.
- The pool's local models sit at the bottom of that range.

Contamination is the real cost. Memorised gold patches inflate the success of strong models on these old issues and compress the differences between configurations. The dataset's datasheet (issue 19) must say so. A contamination check against fresh tasks is an open question below.

The alternatives each fail a hard requirement:

- SWE-bench Pro's public repositories are copyleft by design, and its images are amd64 only.
- SWE-rebench and SWE-bench-Live are fresh but amd64 only, with no per-task results to stratify on.
- Verified Mini is a good pilot source but too narrow for the full set (two repositories).

### Licence position

- **Tasks.** We redistribute **no task data**, only instance ids and our own labels. Problem statements, patches and tests stay on Hugging Face and are fetched at run time.
- **Traces.** Traces will contain repository code and model output.
  - The selected repositories are permissive (GitHub API, checked 2026-11-13):
    - django BSD-3-Clause, astropy BSD-3-Clause, scikit-learn BSD-3-Clause, seaborn BSD-3-Clause, flask BSD-3-Clause;
    - pytest MIT;
    - requests Apache-2.0;
    - sympy BSD-3-style and sphinx BSD-2 (GitHub reports `NOASSERTION`);
    - matplotlib under its own PSF-style licence (GitHub reports none).
  - pylint (GPL-2.0) is excluded.
  - Publishing excerpts of these repositories with attribution is compatible with each licence.
  - SWE-bench-Live and Terminal-Bench *require* public trajectories, and SWE-bench/experiments publishes them for Verified. Publishing trajectories is normal practice, and no benchmark licence I read forbids it.
- **Not verified:**
  - the model providers' terms on publishing outputs (Anthropic Commercial Terms, DeepSeek terms). This is a legal reading for the maintainer, under D5.
  - the licence of the SWE-bench/experiments repository, which has none (`license: null`). We use its per-instance results only to choose tasks and never redistribute them.

## 2. How the subset is drawn

`scripts/select_taskset.py` builds the manifest from public inputs it downloads into a cache, about 10 MB:

1. **Universe.** Start from the 500 Verified tasks.
   - Keep the 420 with an arm64 image. The official `swebench/sweb.eval.arm64.*` image is preferred where it exists (281 tasks); otherwise the task uses Epoch's.
   - Drop pylint (10 tasks, GPL-2.0).
   - That leaves **410** tasks. They cover every Verified repository except xarray, which has no arm64 image.
2. **Difficulty stratum**, from public per-instance results in SWE-bench/experiments at commit `40f164d5`:
   - **easy**: at least 2 of 4 weak open-model runs resolved it (Qwen2.5-Coder-32B, gpt-oss-120b, GPT-5-nano, Qwen3-Coder-30B-A3B);
   - **medium**: otherwise, Claude Haiku 4.5 (high) resolved it;
   - **hard**: otherwise, at least one of 9 frontier runs resolved it (Opus 4.5 and 4.6, Sonnet 4.5, GPT-5.2, Gemini 3 Flash, MiniMax 2.5, GLM-5, Kimi K2.5, DeepSeek 3.2);
   - **unsolved**: nobody resolved it.
   - Devstral results are deliberately not used, because `devstral:24b` is our held-out model.
   - **This is an estimate of difficulty for our pool, not a measurement.** The pilot measures it.
3. **Context fit** (`fits32k`). The estimated final prompt is 2,000 + issue tokens + 550 × the median API-call count of the panel runs that solved the task. A task "fits" if that is at most 28,672, which is the 32K cap minus room for output. 78 of the 150 tasks fit.
4. **Sampling.**
   - Full set: easy 40, medium 45, hard 50, unsolved 15 (150).
   - Within a stratum, tasks are spread across repositories in proportion to the square root of each repository's count, so django does not dominate, then drawn with `random.Random(17)`.
   - Hard is the largest stratum because the hard tail is where routing saves money and where "only the strongest solves it" lives.
5. **Pilot** (drawn from the full set): easy 5, medium 6, hard 7, unsolved 2.
   - Easy and medium tasks are taken from those that fit 32K first, so local models get a fair try. 10 of those 11 fit.
   - Repositories are covered before any repeats.

| | Full set (150) | Pilot (20) |
|---|---|---|
| Strata | easy 40, medium 45, hard 50, unsolved 15 | 5 / 6 / 7 / 2 |
| Repos | django 48, sympy 29, sphinx 23, astropy 14, pytest 13, requests 7, scikit-learn 7, matplotlib 6, seaborn 2, flask 1 | sphinx 4, sympy 4, astropy 3, django 3, pytest 2, scikit-learn 2, flask 1, requests 1 |
| Human `difficulty` | <15 min 52, 15 min-1 h 86, 1-4 h 11, >4 h 1 | 5 / 14 / 1 / 0 |
| Fits 32K (estimate) | 78 | 12 |
| Public resolve rate: weak (Qwen2.5-Coder-32B) / Haiku 4.5 / Opus 4.5 | 7% / 55% / 75% | 0% / 55% / 70% |
| Image source | 98 official, 52 Epoch | — |

**Reproduce**:

```
uv run python scripts/select_taskset.py --cache .cache/taskset          # prints the tables above
uv run python scripts/select_taskset.py --cache .cache/taskset --write  # rewrites the manifest
```

The manifest pins:

- the dataset revision;
- the experiments commit;
- the seed.

Docker Hub and GHCR image listings are not versioned. If an image disappears, the universe changes, and the script then reports a different selection.

**Unsolved tasks are suspect.** OpenAI's audit found that most of the hard tasks it reviewed had flawed tests or descriptions (secondary source). A task nobody solves may be broken rather than hard. Gold-patch validation (below) catches broken *images*, not broken *specifications*. See open question 6.

## 3. The cost model

**Scaffold assumed: mini-SWE-agent 2.x**, bash only. It keeps a linear history with no condensing, a 250-step limit and a $3 cost limit (https://github.com/SWE-agent/mini-swe-agent, `config/benchmarks/swebench.yaml`). The choice of scaffold belongs to issue 18; section 4 explains why the estimates assume this one.

A run is modelled step by step, and each step is priced with steproute's own `price_usage`. The catalog therefore supplies prices, cache multipliers, the 512-token minimum and Haiku's 100K tier.

- The prompt at step *k* is `prompt_base + growth_per_step × k`.
- A share `cache_hit` of the previous prompt is read from the cache, and the rest is written (Anthropic) or billed as input.
- Output per step is `output_per_step` × the effort multiplier.
- Steps per run follow 20 quantiles of a public analog model's API-call counts on our 150 tasks, which keeps the heavy tail: cost grows roughly with the square of the step count.

Every assumption is in `src/steproute/tasks/cost_assumptions.toml`, tagged as measured, derived or assumed. Edit it and re-run:

```
steproute tasks estimate [--plan dense|sparse|small|pilot] [--tasks 150] [--per-config] [--no-cache]
                         [--assumptions FILE] [--catalog FILE]
```

| Assumption | Low | Central | High | Tag and source |
|---|---|---|---|---|
| Prompt at step 1 | 1,800 | 2,000 | 2,500 | measured: 1.5-2.2K in public mini-SWE-agent trajectories (s3://swe-bench-submissions, bash-only runs of Claude 4.5/4.6 and DeepSeek 3.2, 12 instances each) |
| Prompt growth per step | 450 | 550 | 650 | measured: 0.45-0.7K per step, same trajectories |
| Output per step (at "high") | 200 | 250 | 400 | measured: 110-360 for Claude 4.5/4.6; high is assumed, because newer models reason more |
| Cache hit (share of the previous prompt read) | 0.96 | 0.95 | 0.90 | measured: 84-97% of input read from cache for Claude with breakpoints |
| Steps multiplier | 0.8 | 1.0 | **2.0** | high is derived: Claude 5-generation models used 4.3-4.7M tokens per task on SWE-rebench (Sonnet 5 / Opus 5 [high]), about 3 to 5 times the Claude 4.5 analogs |
| Step counts per model | quantiles of an analog | | | derived: Haiku 5.5 from Haiku 4.5 (high), Sonnet 5.5 from Sonnet 4.5 (high), Opus 5.5 from Opus 4.5 (high), Fable 5.2 from Opus 4.6, DeepSeek from DeepSeek 3.2 (high), local from Qwen2.5-Coder-32B. **The mapping is assumed** |
| Effort output multiplier | none 0.4, low 0.5, medium 0.75, high 1, xhigh 1.5, max 2 | | | assumed. Anchors: GPT-5.2 default to high doubled output tokens; Anthropic reports that Opus 4.5 at medium used 76% fewer output tokens than Sonnet 4.5 |
| Effort steps multiplier | none and low 1.1, the rest 1.0 | | | assumed. Public Claude runs changed by under 15% across efforts |
| Cost cap per run | $3 | | | measured: mini-SWE-agent's SWE-bench config. A capped run counts as a failure |
| Local context cap | 32,768 | | | decision D2. A local run whose prompt passes it ends as a failure |
| Local throughput (prefill / generation, tokens per second) | qwen3-coder:30b 750 / 60; gpt-oss:20b 500 / 50; qwen3.5:9b 400 / 40; devstral:24b 200 / 17 | | | qwen3-coder measured (omlx.ai, M4 Pro 48 GB); gpt-oss and qwen3.5 derived from llama.cpp benchmark threads 15396 and 4167; devstral assumed. qwen3.5:9b (effort none) measured here by issue 56: 32-39 output tokens per second over whole-call time with 96% of the prompt read from Ollama's prefix cache, 2.8-3.9 s per agent step including the command |
| Container time per run | 120 s | | | assumed; the harness check alone measured 18-85 s per pilot task (median about 21 s) on this Mac (issue 56), so 120 s is conservative |
| DeepSeek prices | peak | | | catalog. Off-peak (outside 01:00-04:00 and 06:00-10:00 UTC on weekdays) is half |

**Calibration.** At Claude 4.5/4.6 prices, the central scenario reproduces the published mean cost per instance of the same mini-SWE-agent runs on all of Verified (`tests/test_tasks.py::test_central_scenario_reproduces_published_costs`):

| Run | Published | Estimated | Difference |
|---|---|---|---|
| Haiku 4.5 (high) | $0.331 | $0.360 | +9% |
| Sonnet 4.5 (high) | $0.658 | $0.649 | −1% |
| Opus 4.5 (high) | $0.754 | $0.637 | −16% |
| Opus 4.6 | $0.552 | $0.562 | +2% |

The high scenario gives 3 to 5 times the published figures. That is deliberate: it stands in for the newer and more verbose 5-generation models, for which there is no mini-SWE-agent data.

**Per-run cost in our pool** (150-task step distribution; low / central / high, USD):

| Configuration (default effort) | $/run | Steps (central) | Input tokens (sum over steps) | Output tokens |
|---|---|---|---|---|
| claude-haiku-5-5 medium | 0.021 / **0.038** / 0.356 | 70 | 1.7M | 13K |
| claude-sonnet-5-5 high | 0.209 / **0.366** / 1.633 | 49 | 0.8M | 12K |
| claude-opus-5-5 medium | 0.230 / **0.395** / 1.604 | 33 | 0.4M | 6K |
| claude-fable-5-2 high | 0.467 / **0.800** / 2.145 | 32 | 0.4M | 8K |
| deepseek-flash high | 0.054 / **0.100** / 0.506 | 92 | 2.8M | 23K |
| deepseek-v4-pro high | 0.242 / **0.454** / 2.091 | 92 | 2.8M | 23K |
| local (any) | 0 | about 33 (stopped at 32K in about 30% of runs) | | 5-12 min of wall time per run |

Notes on this table:

- **Haiku's high bound is 9 times its central value.** In the high scenario its long runs pass 100K tokens of prompt, and the whole request is then billed at 5 times the price (catalog `price_tiers`).
- **Effort costs little in this model** (Sonnet: $0.354 at low, $0.488 at max), because the analogs say Claude effort moves output more than steps. That is an assumption the pilot's effort sweep tests directly.

**Against the maintainer's rough arithmetic.** The estimate was 100 to 200 tasks × 20 Claude configurations = 2,000 to 4,000 runs at about $0.50, giving $1,000 to $2,000.

- Our central mean over the 20 Claude configurations is **$0.44 a run**, or $8.81 per task for all 20. The range is $0.26 to $1.52 a run ($5.17 to $30.39 per task).
- So the central estimate agrees: 150 tasks is $1,321, or 17 months.
- The high bound is 3.5 times worse ($4,558, or 57 months).
- The conclusion stands either way: the dense grid is infeasible at $80 a month.

**Sensitivities that matter most, in order:**

1. **Prompt caching off** (`--no-cache`) multiplies Anthropic cost by about 3.2. Sparse becomes $1,455 central and the pilot $157. Public mini-SWE-agent Claude trajectories show 84-97% cache reads, so breakpoints were set in those runs. Since issue 46 the proxy places Anthropic cache breakpoints itself, and forwards a request untouched when the client already set its own (`docs/PROXY.md`, "Prompt-cache breakpoints"). That is tested against a mock only. **VERIFY in issue 18**, on the first paid runs, that cache reads actually appear in the usage Anthropic reports.
2. **Steps per run.** Cost grows roughly with steps squared. The high scenario's doubling of steps is the main driver of the high bound.
3. **The analog mapping**, especially Fable 5.2 from Opus 4.6.

## 4. Agent scaffold candidates (the choice belongs to issue 18)

| Scaffold | Licence | OpenAI-compatible base URL (our proxy) | SWE-bench runner | Cost implications |
|---|---|---|---|---|
| **mini-SWE-agent** | MIT | Yes, via LiteLLM `api_base` | Yes, Docker per instance | Linear history means every step extends the last prompt, which maximises cache hits. Public per-instance cost and step data exist, so the estimates calibrate on it. Bash only, so tool definitions are minimal |
| SWE-agent | MIT | Yes (LiteLLM) | Yes | Collapses old observations, which breaks prefix caching. Heavier tool prompts |
| OpenHands | MIT | Yes (LiteLLM) | Yes (benchmarks repo) | A condenser rewrites history, so more cache misses. $0.75-1.82 per instance for Claude on the OpenHands Index |
| Claude Code | Proprietary (Anthropic Commercial Terms) | Anthropic API only (`ANTHROPIC_BASE_URL`) | No | $2.70-3.39 per task on SWE-rebench, 4 to 7 times mini-SWE-agent. Cannot drive Ollama or DeepSeek natively. D7: never on the subscription |
| Codex CLI | Apache-2.0 | Responses API only; chat completions removed | No | Incompatible with the proxy as built |
| opencode | MIT | Yes | No | No public SWE-bench cost data |
| Aider | Apache-2.0 | Yes | No (separate repo) | No public data |
| Harbor | Apache-2.0 | Per wrapped agent | Yes; wraps mini-SWE-agent, OpenHands and others | A runner, not an agent. Useful if the grid needs several scaffolds |

Licence checks came from the GitHub API; the cost figures are from https://swe-rebench.com and https://github.com/OpenHands/openhands-index-results (2026-11-13).

The estimates assume mini-SWE-agent. A scaffold that condenses history, or uses large tool schemas, should be re-estimated by changing `prompt_base`, `growth_per_step` and `cache_hit`.

## 5. Pilot design

**Tasks:** the 20 with `pilot = true` in the manifest (`steproute tasks list --pilot`).

**Configurations** (`PILOT_EFFORTS` in `src/steproute/tasks/estimate.py`):

| Provider | Configurations | Runs | Central $ (low-high) |
|---|---|---|---|
| Ollama | all 7: qwen3.5:9b none/medium; gpt-oss:20b low/medium/high; qwen3-coder:30b; devstral:24b (holdout, separate split) | 140 | 0; about 15 h |
| Anthropic | Haiku all 5 efforts; Sonnet low and high; Opus medium (default); Fable low | 180 | **41** (23-179) |
| DeepSeek | flash all 4 efforts; v4-pro high (default) | 100 | **8.79 off-peak** (4.70-44.43); 17.57 at peak (9.39-84.57) |

**Order**, per issue 18, cheapest first and lower effort first within a model:

1. Ollama models;
2. Haiku;
3. Sonnet and DeepSeek;
4. Opus;
5. Fable.

**Spend against the caps:**

- **Anthropic.** The central estimate is $41 against $80, so about 1.9 times headroom.
- **DeepSeek.** $10 in total, off-peak only (decision of 13 Nov 2026). The central estimate is $8.79 off-peak against $10, so about 1.14 times headroom; at peak it would be $17.57 and would not fit, which is why the runner never calls DeepSeek in peak hours. At 1.5 times the central estimate ($13.18) the budget stops DeepSeek after about 89 of the 100 runs (simulated against the mock in `tests/test_replay_offpeak.py`); unrun cells stay pending, not failed.
- **The high scenario does not fit either cap.** The runner's hard per-provider cap is what bounds the spend. Cheapest-first order then leaves a usable partial pilot, with Fable and then Opus the ones cut short.

**Stage gate** (for issue 18 to implement as a check, not a judgement call):

1. Run Ollama, the Haiku efforts and Sonnet low on all 20 tasks first. That is about $11 central.
2. Compute r = realized ÷ estimated cost for those configurations.
3. If r × the remaining central estimate would take the month above $70, cut the remaining paid configurations to 10 tasks, keeping the stratum balance (3, 3, 3, 1).

**What the pilot measures, replacing assumptions:**

1. **Cost per run, steps, tokens per step and cache-hit rate** for every model, straight from proxy usage. These replace every "assumed" row in the table in section 3.
2. **Effort multipliers**:
   - all five Haiku efforts and all four DeepSeek-flash efforts;
   - the two ends of Sonnet's range, and three gpt-oss levels.
   - Write them into the catalog's `effort_output_mult` (issue 4 placeholders).
3. **Success rate per stratum**, which tests whether the public-panel strata predict our pool.
4. **How often local runs hit the 32K cap** (the estimate is about 30%), and their wall time.
5. **Gold and baseline validation of all 20 images.** Free, Docker only. The gold patch must pass and the baseline patch (`--patch baseline`: no code change, but the tests run) must fail, which matters most for the 52 Epoch images (5 of them in the pilot). `--patch empty` is no validation: the harness skips an empty patch without running anything. Done by issue 56; the table is in section 8.
6. **Run-to-run noise**, if approved (question 12): two extra repeats of Haiku medium and qwen3-coder on the 20 tasks, about $2.

## 6. What task-level labels give, and what they do not

Each `(task, configuration)` run gets one pass or fail from the check (`tests_passed`). That label is copied to every step of the run.

**They give:**

- A full-information matrix P(pass | task, configuration) over whatever the option covers. This is what issue 20's frontier and full-information evaluation, issue 22's external baselines and issue 26's cold start need.
- Cost per solved task, per configuration, measured rather than estimated.
- Cascades that restart from scratch can be evaluated exactly from the matrix: run A, and if it fails, run B. Cost is c_A plus, on failure, c_B including the switch tax; success is A or B.

**They do not give:**

- *Step-level credit.* Which step's model choice made the run succeed is not observed. A step-level predictor trained on these labels learns a delayed, shared reward.
- *Mid-trajectory switching.* Every run uses one configuration from start to finish, so switching model inside a task, and experiment 3's cache effects at the switch, cannot be evaluated from the grid. That needs logged bandit data from the proxy (issue 20's off-policy evaluation) or a separate replay design.
- *Noise.* One run per cell is a single Bernoulli draw, so agent nondeterminism is inside every label. The pilot's repeats estimate how large it is.
- *Partial credit.* A patch that fixes 9 of 10 failing tests is a fail.

## 7. The loader

Code: `src/steproute/tasks/`, standard library only, checked with `mypy --strict`.

| Module | Contents |
|---|---|
| `manifest.py` | `load_manifest`, `load_default_manifest`. A strict TOML loader: `[meta]` plus `[[task]]` with `id`, `stratum`, `fits32k`, `pilot`, `image`; for local tasks also `repo`, `test_cmd`, `timeout_s`, `gold_patch` |
| `check.py` | `run_check(task, patch, backend) -> CheckResult`, with `status` ∈ `passed, failed, patch_failed, timeout, error`. `tests_passed` is True, False or None (None on `error`). `signals()` returns the `Outcome.signals` for `steproute.core.labels.label` |
| | `LocalBackend` copies the task directory into a fresh git workspace, applies the patch (`git apply`, then `patch --fuzz=5`, the harness's order), and runs the command in its own process group with a timeout that kills the group |
| | `SwebenchBackend` writes a one-row dataset (with the task's arm64 `image` replacing the row's x86_64 one) and a one-line predictions file, runs the official harness as an external command, and reads `report.json`. With no report it reads `run_instance.log`: "Patch Apply Failed" is `patch_failed`, a test timeout is `timeout`, an infrastructure failure is `error`. An empty patch is `failed` without running, as in the harness |
| `estimate.py`, `cost_assumptions.toml` | The estimate above |
| `cli.py` | `steproute tasks list | check | fetch | estimate` |

```
steproute tasks check tiny-add --manifest tests/fixtures/tasks/manifest.toml --patch gold
PASSED  tiny-add: exit code 0 (118 ms)

steproute tasks fetch                         # 500 Verified rows, about 9 MB, into .cache/steproute/
scripts/install_swebench.sh                   # swebench==5.0.2, pinned, into .cache/steproute/swebench-env
steproute tasks check django__django-11815 --patch gold       # PASSED, about 20 s
steproute tasks check django__django-11815 --patch baseline   # FAILED: the unfixed code fails its tests
```

Exit codes:

- `0` passed;
- `1` failed (`empty` and `baseline` should give this), patch not applied or timeout;
- `3` the check itself errored;
- `2` usage error.

**Tests.** `tests/test_tasks.py` runs a tiny self-contained task end to end: `tiny_add`, whose `add()` subtracts, and whose unittest fails. It checks:

- pass (`fix.patch`);
- fail (`wrong.patch`);
- a patch that does not apply (`bad.patch`);
- a timeout (`hang.patch` loops forever, and the process group is killed after 1 s);
- an empty patch.

The swebench backend is tested against a stand-in harness script (`tests/fixtures/tasks/fake_harness.py`). It covers:

- every report and log outcome;
- both log layouts (5.0.x `logs/run_evaluation` and the unreleased `logs/evaluation`);
- the arm64 image override.

## 8. What was verified on the real machine, and what still cannot be

### 8.1 Tested on 13 Nov 2026 (issue 56)

Docker Desktop 28.0.4 (arm64, an 8 GB VM) on the M4 Pro, the official harness `swebench==5.0.2` installed by `scripts/install_swebench.sh`, the dataset rows at revision `78f471bf` (the manifest's pin; the Hub reported the same sha). Evidence and harness output: `docs/INTEGRATION-NOTES.md` section 14.

- **The official harness runs on this Mac**, one instance at a time, against a native arm64 image named in a one-row dataset. No emulation, no harness flag.
- **Images: all 20 pilot images pulled.** 15 official from Docker Hub with `docker pull`; the 5 Epoch images from GHCR with `scripts/fetch_ghcr_image.py`, because Docker Desktop's pull got `403 Forbidden` on the GHCR token request (it presents the stored `ghcr.io` login; an anonymous token works). Compressed 0.72-1.15 GB each, 19.1 GB in all; unpacked 2.4-4.0 GB each as `docker image ls` shows them; the Docker image store grew by 15.0 GB for all 20 (`docker system df`, shared layers stored once). Pull time 105-478 s per official image and 211-673 s per Epoch image, measured while about 40 GB of Ollama models downloaded on the same line, so these are upper bounds. None failed to run on arm64.
- **Grading: 19 of 20 grade correctly, 1 does not.** For each task the gold patch and the baseline patch (`--patch baseline`: one inert new file, so the tests run on the unchanged code) went through `steproute tasks check`. `--patch empty` never reaches the harness (it skips empty patches) and is shown only for completeness.

| Task | Stratum | Image source | Size (GB) | Gold | Empty | Baseline | s (gold / baseline) |
|---|---|---|---|---|---|---|---|
| `astropy__astropy-13977` | hard | Epoch | 0.90 | passed (F2P 20/20, P2P 322/322) | failed (not run) | failed (F2P 0/20, P2P 322/322) | 79 / 76 |
| `astropy__astropy-14096` | medium | Epoch | 0.90 | passed (F2P 1/1, P2P 426/426) | failed (not run) | failed (F2P 0/1, P2P 426/426) | 64 / 85 |
| `astropy__astropy-14309` | easy | official | 1.03 | passed (F2P 1/1, P2P 141/141) | failed (not run) | failed (F2P 0/1, P2P 141/141) | 42 / 41 |
| `django__django-11728` | hard | Epoch | 0.96 | passed (F2P 2/2, P2P 44/44) | failed (not run) | failed (F2P 1/2, P2P 43/44) | 21 / 22 |
| `django__django-11815` | easy | official | 1.07 | passed (F2P 2/2, P2P 44/44) | failed (not run) | failed (F2P 0/2, P2P 44/44) | 20 / 21 |
| `django__django-14771` | medium | Epoch | 1.03 | passed (F2P 1/1, P2P 60/60) | failed (not run) | failed (F2P 0/1, P2P 60/60) | 25 / 23 |
| `pallets__flask-5014` | easy | official | 0.85 | passed (F2P 1/1, P2P 59/59) | failed (not run) | failed (F2P 0/1, P2P 59/59) | 22 / 21 |
| `psf__requests-1921` | hard | official | 0.80 | failed (F2P 6/6, P2P 104/107) **WRONG** | failed (not run) | failed (F2P 5/6, P2P 104/107) | 46 / 45 |
| `pytest-dev__pytest-5787` | hard | official | 0.82 | passed (F2P 2/2, P2P 123/123) | failed (not run) | failed (F2P 0/2, P2P 123/123) | 24 / 24 |
| `pytest-dev__pytest-7324` | medium | Epoch | 0.72 | passed (F2P 3/3, P2P 58/58) | failed (not run) | failed (F2P 0/3, P2P 58/58) | 22 / 21 |
| `scikit-learn__scikit-learn-25747` | hard | official | 1.15 | passed (F2P 1/1, P2P 14/14) | failed (not run) | failed (F2P 0/1, P2P 14/14) | 20 / 20 |
| `scikit-learn__scikit-learn-26323` | medium | official | 1.15 | passed (F2P 1/1, P2P 188/188) | failed (not run) | failed (F2P 0/1, P2P 188/188) | 21 / 19 |
| `sphinx-doc__sphinx-10614` | unsolved | official | 1.00 | passed (F2P 1/1, P2P 5/5) | failed (not run) | failed (F2P 0/1, P2P 5/5) | 21 / 21 |
| `sphinx-doc__sphinx-7889` | medium | official | 0.94 | passed (F2P 1/1, P2P 5/5) | failed (not run) | failed (F2P 0/1, P2P 5/5) | 19 / 18 |
| `sphinx-doc__sphinx-8475` | easy | official | 0.94 | passed (F2P 1/1, P2P 17/17) | failed (not run) | failed (F2P 0/1, P2P 17/17) | 21 / 21 |
| `sphinx-doc__sphinx-8638` | hard | official | 0.94 | passed (F2P 1/1, P2P 34/34) | failed (not run) | failed (F2P 0/1, P2P 34/34) | 20 / 19 |
| `sympy__sympy-13647` | easy | official | 0.99 | passed (F2P 1/1, P2P 76/76) | failed (not run) | failed (F2P 0/1, P2P 76/76) | 23 / 22 |
| `sympy__sympy-15599` | hard | official | 0.99 | passed (F2P 1/1, P2P 80/80) | failed (not run) | failed (F2P 0/1, P2P 80/80) | 23 / 22 |
| `sympy__sympy-19346` | medium | official | 0.99 | passed (F2P 1/1, P2P 40/40) | failed (not run) | failed (F2P 0/1, P2P 40/40) | 19 / 19 |
| `sympy__sympy-21596` | unsolved | official | 0.99 | passed (F2P 1/1, P2P 46/46) | failed (not run) | failed (F2P 0/1, P2P 46/46) | 23 / 23 |

  - **`psf__requests-1921` grades wrongly: the gold patch fails** (official image). Every FAIL_TO_PASS test passes, but 3 of 107 PASS_TO_PASS tests fail, so the task is "not resolved" even for the reference fix. In the harness output: `test_POSTBIN_GET_POST_FILES_WITH_DATA` fails with `TypeError: '...' object (type: <class 'str'>) must be callable`, because the image has pytest 8.3.4 (the test uses the string form of `pytest.raises`, removed in pytest 5); `test_mixed_case_scheme_acceptable` and `test_pyopenssl_redirect` fail with `SSLError: CERTIFICATE_VERIFY_FAILED` on requests to `https://httpbin.org` (the 2014 code's bundled certificates do not verify today's chain). The baseline also passes 5 of the 6 FAIL_TO_PASS tests, so only one test separates a fix from no fix. **No model can pass this task on this image**: drop it from the pilot or count it as broken in the report (a question for the maintainer).
  - `django__django-11728` (Epoch) grades correctly, but its baseline is not the dataset's: one FAIL_TO_PASS test (`test_app_not_found`) already passes and one PASS_TO_PASS test (`test_builtin_fields`) fails before the fix. The direction of the grade is right (gold passes, baseline fails).
  - Every other task shows the textbook signature: gold passes everything, baseline fails exactly the FAIL_TO_PASS tests and passes every PASS_TO_PASS test. That includes the other four Epoch images, which their authors call untested.
- **Seconds per check: 18-85 s** (median about 21 s; about 15 s of each is the harness stopping its container). The astropy images are the slowest (41-85 s). This replaces the 120 s assumed in section 3 for the check; the agent's own container time is in `docs/REPLAY.md`.
- **Local throughput of `qwen3.5:9b`** (effort `none`) in the agent loop: 32-39 output tokens per second over whole-call time, 2.8-3.9 s per step including the command, with 96% of prompt tokens read from Ollama's prefix cache. One run hit the 32K cap after 54 steps (30,657 prompt tokens), on a task estimated "long".
- **Contamination, seen directly:** in one local run `qwen3.5:9b` submitted a patch byte-identical to the gold patch of `pallets__flask-5014` (same blob hash `1aa82562`, same error message).

### 8.2 Still not verified

Money, or more time on the machine than issue 56 had:

- **That the arm64 images grade like the x86_64 ones in general.** The pilot's 20 do, except `psf__requests-1921`; the other 130 images of the full set were not pulled or validated. Validate each with `--patch gold` and `--patch baseline` before its first paid run.
- **Disk for all 150 images.** Extrapolating the pilot: about 75 GB in Docker's store (0.75 GB per image after sharing), well within the VM's 900 GB. The pre-flight's free-disk line measures the host disk at the output directory, not the Docker VM's disk.
- **Every per-run cost number for the 5.5 and 5.2 Claude models and the current DeepSeek models.** These are derived from older analogs and must be measured in the pilot.
- **That prompt caching works end to end against the live Anthropic API** through the proxy (sensitivity 1 above). The proxy's breakpoints are tested against a mock only.
- **Local throughput** of `gpt-oss:20b`, `qwen3-coder:30b` and `devstral:24b` in the agent loop: they passed the one-call tool test (issue 56; all four local models PASS) but ran no task.

## 9. Open questions for the maintainer

1. Approve **SWE-bench Verified (arm64 subset, 150 tasks)** as the task set? *yes / no*
2. Full grid: option **A, B or C**? *A / B / C* (recommended: B)
3. Approve the **pilot** (20 tasks; about $41 Anthropic central under the $80 hard cap; about $9 DeepSeek off-peak)? *yes / no*
4. DeepSeek budget **per month after the pilot**? *a dollar amount* (the table assumes $20). *Answered 13 Nov 2026 for the pilot: $10 in total; nothing after it is funded yet.*
5. Run DeepSeek **only off-peak**? *yes / no*. *Answered: yes.*
6. Keep the **15 unsolved tasks**, which are likely enriched with flawed tests? *keep / drop*
7. Later, add about 30 fresh **SWE-rebench 2026 tasks** under x86_64 emulation as a contamination check? *yes / no*
8. Assume **mini-SWE-agent** as the issue 18 scaffold? *yes / no*
9. Keep the **$3 per-run cost cap**, with a capped run counting as a failure? *yes / no*
10. Use **Epoch's untested arm64 images** for the 52 tasks without an official one, after gold validation? *yes / no*
11. Is **about 150 GB of free disk** available for images? *yes / no*
12. Spend about **$2** on repeat runs in the pilot to measure run-to-run noise? *yes / no*
