# Pilot results (issue 18)

*Runs of 14 to 16 Nov 2026 on the maintainer's M4 Pro Mac. Written 16 Nov 2026. Every measured number on this page was computed from the two run directories named in "How this report was made"; the estimates are those of `steproute tasks estimate`.*

The pilot ran 20 SWE-bench Verified tasks under 21 model and effort configurations, 420 runs in all, to measure real cost and success before the larger grid. This page gives the cost and success of each configuration, how the runs ended, what the results cannot be trusted for, and a revised estimate for the sparse grid.

## Summary

- **All 420 runs finished with a result, and 221 passed.** The 14 paid configurations passed 207 of 280 runs (74%). The 7 local configurations passed 14 of 140 (10%).
- **Paid spend was $12.85.** Anthropic took $7.58 of its $80 monthly cap. DeepSeek took $5.26 of its $10 total, with every call in off-peak hours.
- **Paid runs cost much less than estimated.** Anthropic came in at 0.18 of the task-set estimate ($7.50 against $41.28), mainly because Sonnet, Opus and Fable needed 4 to 6 steps per run where the estimate assumed 33 to 54. DeepSeek came in at 0.57 of its estimate at list price.
- **The sparse grid is re-estimated at about $70 for Anthropic** (the task-set estimate was $456) **and about $101 at list price for DeepSeek**, which is about $50 when billed off-peak. DeepSeek has $4.74 left of its $10, so its part of the sparse grid is not funded.
- **Five of the 21 results are dominated by a limit of the test setup, not by the model's ability to fix the task.** They are `claude-haiku-5-5` at `max` (the output-token cap), `qwen3.5:9b` at both efforts and `gpt-oss:20b` at `high` (the 32K local context cap), and `devstral:24b` (it does not use the tool-call interface). The caveats section explains each.
- **Seven decisions are open for the maintainer.** They are listed at the end.

## What was run

- **Tasks.** The 20 pilot tasks of the 150-task set in `docs/TASKSET.md`. By the public-panel difficulty strata they are 5 easy, 6 medium, 7 hard and 2 that no public run had solved.
- **Configurations.** A configuration is a model with an effort level. There are 9 for Anthropic, 5 for DeepSeek and 7 for local models served by Ollama. `devstral:24b` is the held-out model of decision D2, so its runs are stored in a separate split.
- **Runs.** A run is one configuration working on one task from start to finish. Each task and configuration ran once, so run-to-run noise was not measured.
- **Scaffold.** The scaffold is the small agent loop of `docs/REPLAY.md`: the model gets one `bash` tool, runs commands in the task's container, and submits a git patch. A step is one model reply. Two versions were used, `steproute-bash-1` and `steproute-bash-2`; the caveats say what changed.
- **Limits per run.** 250 steps, $3, 16,384 output tokens per call, and for local models a 32,768-token context window. The scaffold also ends a run after three replies in a row without a usable tool call.
- **Passed** means the official SWE-bench harness applied the submitted patch and the task's tests passed, both the tests the fix must make pass and the tests that must keep passing.

The runs happened in three sessions (times are CET):

| Session | Commit | What ran |
|---|---|---|
| 14 Nov 18:35 to 15 Nov 13:21 | `7ff8441` | All 280 paid runs (finished by 01:18 on 15 Nov), the 40 `qwen3.5:9b` runs, and the first `gpt-oss:20b` runs |
| 15 Nov 13:22 to 16:35 | `e0411f6` | `gpt-oss:20b` again on `steproute-bash-1`, after the first of two fixes. Not counted (see "Local models") |
| 15 Nov 21:53 to 16 Nov 06:44 | `2a456fe` | 100 local runs on `steproute-bash-2` in a second directory: `gpt-oss:20b` at three efforts, `qwen3-coder:30b` and `devstral:24b` |

## Results per configuration

Costs are the provider's reported token usage priced with the catalog. *Billed* is that usage at the price that applied when the call was made: DeepSeek's off-peak price is half its list price. *At list price* is the same usage at the catalog's peak price, which is the number to compare across providers and with the estimates. Local runs cost nothing in dollars. The scaffold column drops the `steproute-` prefix.

| Provider | Model | Effort | Runs | Passed | Success | $ per run, billed | $ total, billed | $ total at list price | Steps per run | Scaffold |
|---|---|---|---|---|---|---|---|---|---|---|
| Anthropic | `claude-haiku-5-5` | low | 20 | 10 | 50% | $0.0149 | $0.30 | $0.30 | 51.5 | bash-1 |
| Anthropic | `claude-haiku-5-5` | medium | 20 | 13 | 65% | $0.0187 | $0.37 | $0.37 | 48.4 | bash-1 |
| Anthropic | `claude-haiku-5-5` | high | 20 | 14 | 70% | $0.0084 | $0.17 | $0.17 | 15.8 | bash-1 |
| Anthropic | `claude-haiku-5-5` | xhigh | 20 | 14 | 70% | $0.0259 | $0.52 | $0.52 | 20.1 | bash-1 |
| Anthropic | `claude-haiku-5-5` | max | 20 | 8 | 40% | $0.0657 | $1.31 | $1.31 | 16.6 | bash-1 |
| Anthropic | `claude-sonnet-5-5` | low | 20 | 15 | 75% | $0.0226 | $0.45 | $0.45 | 4.6 | bash-1 |
| Anthropic | `claude-sonnet-5-5` | high | 20 | 15 | 75% | $0.0384 | $0.77 | $0.77 | 5.5 | bash-1 |
| Anthropic | `claude-opus-5-5` | medium | 20 | 18 | 90% | $0.0637 | $1.27 | $1.27 | 5.5 | bash-1 |
| Anthropic | `claude-fable-5-2` | low | 20 | 17 | 85% | $0.1166 | $2.33 | $2.33 | 3.8 | bash-1 |
| DeepSeek | `deepseek-flash` | none | 20 | 11 | 55% | $0.0200 | $0.40 | $0.80 | 65.5 | bash-1 |
| DeepSeek | `deepseek-flash` | low | 20 | 18 | 90% | $0.0126 | $0.25 | $0.51 | 29.1 | bash-1 |
| DeepSeek | `deepseek-flash` | high | 20 | 20 | 100% | $0.0160 | $0.32 | $0.64 | 31.4 | bash-1 |
| DeepSeek | `deepseek-flash` | max | 20 | 19 | 95% | $0.0147 | $0.29 | $0.59 | 27.6 | bash-1 |
| DeepSeek | `deepseek-v4-pro` | high | 20 | 15 | 75% | $0.1881 | $3.76 | $7.52 | 108.5 | bash-1 |
| local | `qwen3.5:9b` | none | 20 | 1 | 5% | $0 | $0 | $0 | 58.4 | bash-1 |
| local | `qwen3.5:9b` | medium | 20 | 2 | 10% | $0 | $0 | $0 | 64.5 | bash-1 |
| local | `gpt-oss:20b` | low | 20 | 3 | 15% | $0 | $0 | $0 | 13.9 | bash-2 |
| local | `gpt-oss:20b` | medium | 20 | 4 | 20% | $0 | $0 | $0 | 32.2 | bash-2 |
| local | `gpt-oss:20b` | high | 20 | 0 | 0% | $0 | $0 | $0 | 56.4 | bash-2 |
| local | `qwen3-coder:30b` | medium | 20 | 4 | 20% | $0 | $0 | $0 | 55.5 | bash-2 |
| local | `devstral:24b` | medium | 20 | 0 | 0% | $0 | $0 | $0 | 4.7 | bash-2 |

By provider: Anthropic 124 of 180 (69%), DeepSeek 83 of 100 (83%), local 14 of 140 (10%).

**How much a rate can be trusted.** With 20 tasks, one task is 5 points, and each configuration ran each task once. The 95% interval (Wilson) is 53% to 89% for 15 of 20, 84% to 100% for 20 of 20, and 8% to 42% for 4 of 20. Most pairs of paid configurations therefore cannot be ranked from this table.

**What stands out, with that margin in mind.**

- `deepseek-flash` at `high` passed all 20 tasks at $0.016 per run billed ($0.032 at list price). DeepSeek's thinking configurations, this one among them, ran with a known fault (caveat 3), so this is not a clean measurement.
- Every task was passed by at least one paid configuration. Five tasks were passed by all 14, and the hardest (`sympy__sympy-21596`) by 2.
- `claude-haiku-5-5` is cheapest at `high`, not at `low`. At `low` and `medium` many of its replies carried no tool call (537 of 1,029 at `low` and 374 of 967 at `medium`), so those runs took about three times the steps.
- `deepseek-v4-pro` cost about 12 times as much per run as `deepseek-flash` at the same effort and passed fewer tasks (15 against 20).

The pilot was sized to measure cost and to test the pipeline. It is too small to say whether routing beats the best single configuration.

## Spend

| Provider | Billed | At list price | Limit | Task-set estimate, central (low to high) |
|---|---|---|---|---|
| Anthropic | $7.58 | $7.58 | $80 per calendar month | $41.28 ($23.28 to $178.60) |
| DeepSeek | $5.26 | $10.53 | $10 in total, off-peak hours only | $8.79 off-peak ($4.70 to $44.43); $17.57 at list price |
| Local | $0 | $0 | none | $0 |

- The totals come from the spend ledger and include every attempt. The per-configuration totals above count only the last attempt of each run and sum to $7.50 and $5.03. The difference, $0.32, is nine attempts that were stopped and repeated (caveat 4).
- The ledger holds 3,456 Anthropic calls and 5,438 DeepSeek calls. Every DeepSeek call was billed at the off-peak price.
- No run hit the $3 per-run cap, and neither provider limit was reached, so a real cap stop was not exercised.
- The runner's stage gate passed. After the first stage it measured Anthropic cost at 0.28 of the estimate and projected $11.45 for the month against its $70 threshold, so no Anthropic run was cut.
- These amounts were not compared with the providers' own billing pages for this report.

## How runs ended

| Model | Effort | Passed | Patch did not resolve | Patch not applicable | Empty patch | No submission | Context limit | Step limit |
|---|---|---|---|---|---|---|---|---|
| `claude-haiku-5-5` | low | 10 | 5 | 0 | 0 | 3 | 0 | 2 |
| `claude-haiku-5-5` | medium | 13 | 5 | 0 | 0 | 0 | 0 | 2 |
| `claude-haiku-5-5` | high | 14 | 6 | 0 | 0 | 0 | 0 | 0 |
| `claude-haiku-5-5` | xhigh | 14 | 6 | 0 | 0 | 0 | 0 | 0 |
| `claude-haiku-5-5` | max | 8 | 0 | 0 | 0 | 12 | 0 | 0 |
| `claude-sonnet-5-5` | low | 15 | 5 | 0 | 0 | 0 | 0 | 0 |
| `claude-sonnet-5-5` | high | 15 | 4 | 1 | 0 | 0 | 0 | 0 |
| `claude-opus-5-5` | medium | 18 | 2 | 0 | 0 | 0 | 0 | 0 |
| `claude-fable-5-2` | low | 17 | 3 | 0 | 0 | 0 | 0 | 0 |
| `deepseek-flash` | none | 11 | 7 | 0 | 0 | 0 | 0 | 2 |
| `deepseek-flash` | low | 18 | 2 | 0 | 0 | 0 | 0 | 0 |
| `deepseek-flash` | high | 20 | 0 | 0 | 0 | 0 | 0 | 0 |
| `deepseek-flash` | max | 19 | 1 | 0 | 0 | 0 | 0 | 0 |
| `deepseek-v4-pro` | high | 15 | 3 | 0 | 0 | 0 | 0 | 2 |
| `qwen3.5:9b` | none | 1 | 0 | 1 | 0 | 1 | 17 | 0 |
| `qwen3.5:9b` | medium | 2 | 0 | 0 | 0 | 0 | 18 | 0 |
| `gpt-oss:20b` | low | 3 | 10 | 0 | 1 | 6 | 0 | 0 |
| `gpt-oss:20b` | medium | 4 | 5 | 1 | 0 | 8 | 2 | 0 |
| `gpt-oss:20b` | high | 0 | 1 | 0 | 0 | 8 | 11 | 0 |
| `qwen3-coder:30b` | medium | 4 | 6 | 6 | 3 | 0 | 1 | 0 |
| `devstral:24b` | medium | 0 | 0 | 0 | 0 | 20 | 0 | 0 |
| **All 21** | | **221** | **71** | **9** | **4** | **58** | **49** | **8** |

- **Patch did not resolve:** the patch applied and the tests failed.
- **Patch not applicable:** the harness could not apply what was submitted.
- **Empty patch:** the run submitted nothing.
- **No submission:** the run ended on three replies in a row without a usable tool call.
- **Context limit** (local models only): the next prompt would have passed 30,720 tokens, which is the 32,768-token window less 2,048 for the reply.
- **Step limit:** 250 steps.

Every ending other than "passed" counts as a failure in the success rate. No run ended on the $3 cap or the 2-hour time limit, and no final attempt ended on a provider or internal error.

Two notes on the paid rows:

- The 15 paid runs without a submission are 12 of `claude-haiku-5-5` at `max`, which the output-token cap explains (caveat 2), and 3 at `low`.
- The one paid "patch not applicable" is a `claude-sonnet-5-5` run that submitted only the first five lines of its patch.

## Paid runs: estimate against measurement

The estimates are those of `steproute tasks estimate --plan pilot --per-config`, at list price. *Cache-read share* is the share of prompt tokens that the provider served from its prompt cache.

| Model | Effort | $ per run at list price, estimated | measured | Measured / estimated | Steps per run, estimated | measured | Output tokens per run, estimated | measured | Cache-read share |
|---|---|---|---|---|---|---|---|---|---|
| `claude-haiku-5-5` | low | $0.0420 | $0.0149 | 0.35 | 77 | 51.5 | 9,600 | 6,897 | 99% |
| `claude-haiku-5-5` | medium | $0.0379 | $0.0187 | 0.49 | 70 | 48.4 | 13,200 | 12,873 | 98% |
| `claude-haiku-5-5` | high | $0.0401 | $0.0084 | 0.21 | 70 | 15.8 | 17,500 | 10,958 | 92% |
| `claude-haiku-5-5` | xhigh | $0.0445 | $0.0259 | 0.58 | 70 | 20.1 | 26,300 | 41,629 | 94% |
| `claude-haiku-5-5` | max | $0.0489 | $0.0657 | 1.34 | 70 | 16.6 | 35,100 | 120,779 | 93% |
| `claude-sonnet-5-5` | low | $0.3540 | $0.0226 | 0.06 | 54 | 4.6 | 6,700 | 1,078 | 77% |
| `claude-sonnet-5-5` | high | $0.3661 | $0.0384 | 0.10 | 49 | 5.5 | 12,200 | 2,085 | 78% |
| `claude-opus-5-5` | medium | $0.3954 | $0.0637 | 0.16 | 33 | 5.5 | 6,200 | 1,721 | 75% |
| `claude-fable-5-2` | low | $0.7351 | $0.1166 | 0.16 | 35 | 3.8 | 4,400 | 1,608 | 69% |
| `deepseek-flash` | none | $0.0975 | $0.0400 | 0.41 | 101 | 65.5 | 10,100 | 13,753 | 99% |
| `deepseek-flash` | low | $0.1005 | $0.0253 | 0.25 | 101 | 29.1 | 12,600 | 12,923 | 95% |
| `deepseek-flash` | high | $0.0997 | $0.0320 | 0.32 | 92 | 31.4 | 22,900 | 16,482 | 96% |
| `deepseek-flash` | max | $0.1272 | $0.0294 | 0.23 | 92 | 27.6 | 45,900 | 13,942 | 95% |
| `deepseek-v4-pro` | high | $0.4539 | $0.3761 | 0.83 | 92 | 108.5 | 22,900 | 21,085 | 99% |

- **Steps explain most of the gap.** Sonnet, Opus and Fable finished in 4 to 6 steps per run, against 33 to 54 assumed. Only `deepseek-v4-pro` used more steps than assumed.
- **Prompt caching reached both providers.** The cache-read share is 69% to 99% for Claude and 95% to 99% for DeepSeek. It is lowest for the models with the shortest runs, as expected when the first call writes the cache and only a few later calls read it.
- **One configuration cost more than estimated:** `claude-haiku-5-5` at `max`, at 1.34 times the estimate. Its replies often used the whole output-token cap (see the caveats).

**Output tokens per call by effort.** The estimate assumed output multipliers of 0.4 for `none`, 0.5 for `low`, 0.75 for `medium`, 1.5 for `xhigh` and 2 for `max`, relative to `high`. Measured, for the models that ran at `high` and at least one other effort (for `gpt-oss:20b`, only the calls with reported usage):

| Model | Effort | Calls | Output tokens per call | Relative to `high` |
|---|---|---|---|---|
| `claude-haiku-5-5` | low | 1,029 | 134 | 0.19 |
| `claude-haiku-5-5` | medium | 967 | 266 | 0.38 |
| `claude-haiku-5-5` | high | 316 | 694 | 1.00 |
| `claude-haiku-5-5` | xhigh | 403 | 2,066 | 2.98 |
| `claude-haiku-5-5` | max | 333 | 7,254 | 10.46 |
| `claude-sonnet-5-5` | low | 92 | 234 | 0.61 |
| `claude-sonnet-5-5` | high | 109 | 383 | 1.00 |
| `deepseek-flash` | none | 1,311 | 210 | 0.40 |
| `deepseek-flash` | low | 581 | 445 | 0.85 |
| `deepseek-flash` | high | 627 | 526 | 1.00 |
| `deepseek-flash` | max | 552 | 505 | 0.96 |
| `gpt-oss:20b` | low | 244 | 98 | 0.31 |
| `gpt-oss:20b` | medium | 512 | 172 | 0.55 |
| `gpt-oss:20b` | high | 863 | 314 | 1.00 |

`qwen3.5:9b` has no `high` level; it wrote 199 output tokens per call at `none` and 270 at `medium`. The catalog's `effort_output_mult` is still 1.0 for every effort. Updating it and `cost_assumptions.toml` from these values is step 8 of `docs/REPLAY.md` and is not part of this report.

## Success by difficulty stratum

The strata come from public results of other models (`docs/TASKSET.md`). Each cell is passed runs over runs, summed over the efforts of a model.

| Model | easy | medium | hard | unsolved |
|---|---|---|---|---|
| `claude-haiku-5-5` | 21/25 | 26/30 | 12/35 | 0/10 |
| `claude-sonnet-5-5` | 10/10 | 12/12 | 8/14 | 0/4 |
| `claude-opus-5-5` | 5/5 | 6/6 | 7/7 | 0/2 |
| `claude-fable-5-2` | 4/5 | 6/6 | 7/7 | 0/2 |
| `deepseek-flash` | 18/20 | 22/24 | 23/28 | 5/8 |
| `deepseek-v4-pro` | 5/5 | 5/6 | 5/7 | 0/2 |
| `qwen3.5:9b` | 3/10 | 0/12 | 0/14 | 0/4 |
| `gpt-oss:20b` | 6/15 | 1/18 | 0/21 | 0/6 |
| `qwen3-coder:30b` | 1/5 | 3/6 | 0/7 | 0/2 |
| `devstral:24b` | 0/5 | 0/6 | 0/7 | 0/2 |

- Over the paid configurations, the hard and unsolved strata were harder, as predicted: easy 63 of 70 (90%), medium 77 of 84 (92%), hard 62 of 98 (63%), unsolved 5 of 28 (18%). Easy and medium do not differ.
- The two "unsolved" tasks are not unsolvable. `deepseek-flash` passed `sphinx-doc__sphinx-10614` at three efforts and `sympy__sympy-21596` at two. No other configuration passed either.
- Local models passed 10 of 35 easy runs, 4 of 42 medium runs and no hard or unsolved run.

## Local models

Local runs cost no money but take machine time. One model is loaded at a time and runs go one after another.

| Model | Effort | Runs | Hours in total | Minutes per run, mean | median | longest | Steps per run | Runs ended on the context limit |
|---|---|---|---|---|---|---|---|---|
| `qwen3.5:9b` | none | 20 | 1.9 | 5.8 | 5.9 | 9.8 | 58.4 | 17 |
| `qwen3.5:9b` | medium | 20 | 2.8 | 8.5 | 9.2 | 12.7 | 64.5 | 18 |
| `gpt-oss:20b` | low | 20 | 0.5 | 1.4 | 1.1 | 3.5 | 13.9 | 0 |
| `gpt-oss:20b` | medium | 20 | 1.6 | 4.8 | 5.1 | 8.8 | 32.2 | 2 |
| `gpt-oss:20b` | high | 20 | 4.0 | 12.0 | 9.6 | 34.2 | 56.4 | 11 |
| `qwen3-coder:30b` | medium | 20 | 1.8 | 5.5 | 5.4 | 9.6 | 55.5 | 1 |
| `devstral:24b` | medium | 20 | 0.9 | 2.8 | 1.3 | 16.7 | 4.7 | 0 |
| **All 7** | | **140** | **13.6** | **5.8** | | | | |

The task-set estimate for these 140 runs was about 15 hours. The 100 runs of the third session took 8.8 hours of wall time, unattended.

**`qwen3.5:9b`: stopped by the context cap.** 35 of its 40 runs ended on the context limit, 3 passed, one submitted the text of a command instead of a patch, and one made no submission. These runs say little about what the model could do with more context.

**`gpt-oss:20b`: usable after two fixes, and still limited.**

- The first runs, on `steproute-bash-1`, are not counted. None of those 58 runs passed. The model edits files with an `apply_patch` command that did not exist in the workspace (all 5 such calls in the stored transcripts answered "command not found"), and it often writes tool-call arguments that are not valid JSON, which Ollama rejects with an HTTP 500. Pull request 74 made the runner treat a rejected call as the model's step instead of a provider failure. Pull request 76 quotes the rejected text back to the model and puts an `apply_patch` command in every workspace. The scaffold with the changes of pull request 76 is `steproute-bash-2`, and the three configurations were run again on it.
- After the fixes it passed 3, 4 and 0 of 20 at `low`, `medium` and `high`. The run reached a submission in 14, 10 and 1 of the 20 runs.
- Ollama still rejected many calls: 35 of 279 steps at `low` (13%), 132 of 644 at `medium` (20%) and 264 of 1,127 at `high` (23%). Where the transcript is stored, the reply after a rejected call was a valid call in 24 of 35 cases at `low`, 70 of 102 at `medium` and 83 of 109 at `high`.
- In the stored transcripts, `apply_patch` returned success in 17 of 20 calls at `low`, 22 of 32 at `medium` and 3 of 3 at `high`.
- At `high`, 11 of 20 runs ended on the context limit and 8 made no submission.

**`qwen3-coder:30b`: works with the tools, and often spoils its own submission.** It passed 4 of 20. It drew only 8 format errors in the 1,022 steps of its stored transcripts. But in 6 runs the harness could not apply what it submitted: two submissions were prose, two were patches typed by hand with a placeholder `index` line, and two had a real header but a body that was not the workspace's real `git diff`. In 3 more runs it submitted an empty patch although it had changed files.

**`devstral:24b`: the result measures an interface mismatch, not the model's coding ability.** All 20 runs ended without a submission, after a median of 3 steps. The model writes its commands as text in a fenced code block instead of calling the tool: 75 of its 93 replies carried no tool call, and 64 of those held a fenced block. 16 runs never made a tool call at all. The scaffold's rule is the same for every model, so the 0 of 20 is a fair result for this setup, but it does not show how well the model fixes code.

## Caveats

1. **Two scaffold versions.** The paid runs and `qwen3.5:9b` ran on `steproute-bash-1`; `gpt-oss:20b`, `qwen3-coder:30b` and `devstral:24b` ran on `steproute-bash-2`. Version 2 quotes arguments that are not valid JSON back to the model, and it adds an `apply_patch` command to the workspace. In the 8,676 stored steps of the 280 paid runs there is no invalid-JSON call and no `apply_patch` call, so the paid results are not expected to differ on version 2. This is an inference from the transcripts, not a rerun. For `qwen3.5:9b` only 5 of 40 transcripts exist (see caveat 8); none of its 2,457 calls was rejected by Ollama.
2. **`claude-haiku-5-5` at `max` hit the output-token cap.** Every call is limited to 16,384 output tokens. At `max` effort, 61 of this configuration's 333 replies used exactly that many, in 18 of its 20 runs. All 12 runs that ended without a submission ended on three such replies in a row, cut off before a usable tool call; in 11 of the 12 those replies were empty. The 40% success rate and the cost above the estimate are therefore a result of the cap, not a measurement of the model at `max`. The cap was reached once at `xhigh` and never in any other configuration.
3. **DeepSeek's thinking configurations ran without their earlier reasoning (issue 73).** The scaffold keeps only the text and tool calls of each reply, so `deepseek-flash` at `low`, `high` and `max` and `deepseek-v4-pro` at `high` received a blank in place of their own earlier reasoning on every later call. The size of the effect is unknown.
4. **Nine attempts were stopped by the privacy rule and repeated (issue 72).** The router refused to send a step to a cloud model, the run was recorded as an internal error, and it was started again. The issue attributes the refusals to the privacy rule by elimination; the refused decision itself is not stored. This happened in 7 runs: 2 of `claude-haiku-5-5` at `max`, 1 of `deepseek-flash` at `max` and 4 of `deepseek-v4-pro` at `high` (two of them twice). All 7 then finished, 6 passed and 1 failed, and the table counts that last attempt. The stopped attempts cost $0.32. The outcome of these 7 runs comes from a second or third try, which the other runs did not get.
5. **The 32K local context cap (decision D2).** 49 of the 140 local runs (35%) ended on it: 17 and 18 of 20 for `qwen3.5:9b`, 0, 2 and 11 of 20 for `gpt-oss:20b` at `low`, `medium` and `high`, 1 of 20 for `qwen3-coder:30b`, and 0 for `devstral:24b`. The task-set estimate assumed about 30% overall. The manifest's `fits32k` flag did not predict it: the limit ended 29 of 84 runs on tasks marked as fitting and 20 of 56 on tasks marked as not fitting.
6. **Twenty tasks, one run each.** See "How much a rate can be trusted" above. The repeats that would measure run-to-run noise were not run.
7. **Costs are priced usage, not invoices.** Each call's reported usage is priced with the catalog. The totals were not compared with the providers' billing pages for this report.
8. **A run that ends on the context limit stores no transcript.** Counts taken from transcripts (the replies after a rejected call, the `apply_patch` results, the format errors) therefore cover only the other runs: 18 of 20 for `gpt-oss:20b` at `medium`, 9 of 20 at `high`, and 19 of 20 for `qwen3-coder:30b`. Counts of rejected calls come from the journal and cover every run.

## Revised estimate for the sparse grid

The sparse grid (option B of `docs/TASKSET.md`) is every local configuration and each paid model at its default effort on all 150 tasks, plus the other paid efforts on 30 tasks: 1,080 Anthropic runs, 480 DeepSeek runs and 1,050 local runs. `steproute replay report` re-estimates it from the pilot: a configuration the pilot ran is priced at its measured cost per run, and any other at the task-set estimate times the measured ratio of the same model.

| Provider | Runs | Task-set estimate, central | Re-estimated | Runs priced from a measured configuration | Runs priced through the model's ratio |
|---|---|---|---|---|---|
| Anthropic | 1,080 | $456 | about $70 | 630 | 450 |
| DeepSeek, at list price | 480 | $138 | about $101 | 390 | 90 |
| DeepSeek, billed off-peak | 480 | about $69 | about $50 | | |

- **Anthropic: about $70.** About $36 of it is `claude-fable-5-2`, $19 `claude-opus-5-5`, $10 `claude-sonnet-5-5` and $6 `claude-haiku-5-5`. The largest single item, Fable at `high` on 150 tasks (about $19), was not run in the pilot: its price comes from Fable's ratio at `low`. The amount fits under the $80 monthly cap in a month that starts at zero. November has $72.42 left.
- **DeepSeek: about $101 at list price, about $50 off-peak.** About $94 of the $101 is `deepseek-v4-pro`, and `deepseek-v4-pro` at `high` on 150 tasks alone is about $56. DeepSeek's budget is $10 in total and $5.26 is spent, so $4.74 is left. **The gap of about $46 is an open funding question.**
- **Both figures are central values from 20 tasks.** They carry the pilot's margins, the caveats above (the Haiku `max` price is inflated by the output cap, and the DeepSeek prices come from runs with the issue 73 fault), and no allowance for repeated attempts.

**Local wall time: about 102 hours.** At the measured mean minutes per run, 150 runs of each local configuration take 14.5 and 21.3 hours for `qwen3.5:9b`, 3.4, 11.9 and 30.0 hours for `gpt-oss:20b`, 13.8 hours for `qwen3-coder:30b` and 6.9 hours for `devstral:24b`. That is 4.2 days of continuous running, or about 12 nights of the length of the third session. The task-set estimate was about 111 hours. This assumes the same Mac and Ollama settings, that the 20 pilot tasks stand for the 150, and that nothing about the setup changes. A larger context window would lengthen the `qwen3.5:9b` and `gpt-oss:20b` runs that now stop at the cap, and a scaffold that accepts `devstral:24b`'s commands would turn its 3-step runs into full ones, so any such change raises the figure.

## Decisions for the maintainer

None of these is decided here.

1. **The 32K local context window (decision D2).** It ended 35% of local runs and nearly all of `qwen3.5:9b`'s. Keep it, or raise it. Recommended: try a larger window on a few tasks first, to see what it does to memory, speed and success.
2. **How to treat `devstral:24b`.** Keep the 0 of 20 as the result for this setup, or let the scaffold accept a single fenced `bash` block as a command, which is a new scaffold version and a rerun. Recommended: keep the result in this report and open an issue for the scaffold change.
3. **Issue 72, the privacy rule in benchmark replays.** Option A: do not apply the rule to benchmark replays, and count how often it would have fired. Option B: keep it and end such a run as `blocked_sensitive` without a retry. Recommended in the issue: option A.
4. **Issue 73, and whether to rerun the DeepSeek block.** The fix is to keep and return what each provider expects between turns. A rerun of the four thinking configurations would show the size of the effect. In the pilot they cost $4.86 with their repeated attempts ($4.63 without), the whole DeepSeek block cost $5.26, and $4.74 is left. So the whole block does not fit, and the four alone have no margin, before the extra input tokens that returned reasoning would add.
5. **DeepSeek funding for the sparse grid.** About $50 off-peak is needed and $4.74 is left. Almost all of the cost is `deepseek-v4-pro`.
6. **The go-ahead for the sparse grid's spend.** The task set and option B were approved at Gate A on 13 Nov 2026, with the estimate to be redone from the pilot. This page is that estimate. No run beyond the pilot is approved.
7. **The output-token cap for `claude-haiku-5-5` at `max`.** This was found while writing this report and has not been discussed. Keep the 16,384-token cap and report the configuration as limited by it, or raise the cap for the high efforts and run the 20 tasks again (the first run cost $1.31).

## How this report was made

- **Two run directories.** `runs/pilot` holds the paid runs, `qwen3.5:9b` and the first `gpt-oss:20b` runs. `runs/pilot-local-2` holds the 100 runs of the third session. Run directories are not committed.
- **Which runs count.** Only the last attempt of each task and configuration counts. The `gpt-oss:20b` runs in `runs/pilot` are left out, because the same 60 runs exist in `runs/pilot-local-2` on the fixed scaffold.
- **Sources.** `steproute replay report` was run on each directory. It gave the cost columns, the spend, the strata table and the sparse-grid estimate. It reads one directory at a time, so the tables here were put together from its two outputs. The endings, wall time, token counts and transcript counts were computed from each directory's journal (`runs.jsonl`), step traces and stored transcripts.
- **Not published here.** This page holds aggregates only. The traces themselves are not released; that is issue 19.
