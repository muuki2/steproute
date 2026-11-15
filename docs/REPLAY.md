# Replay runner

`steproute replay` produces the project's dataset: each task run under one fixed `(model, effort)` for the whole task, through the proxy, with a task-level pass or fail, the realised cost and the per-step traces. It is also the only part of steproute that spends real money, unattended, on a small monthly budget. So the safe behaviour is the default, and every unsafe one needs an explicit, visible choice.

Code: `src/steproute/replay/` (standard library only; a run that calls a model needs the `[proxy]` extra). Issue 18; the task set and the approved sparse grid (option B) are in `docs/TASKSET.md`; the grid format is `docs/EVALUATION.md` section 2.1.

**Status (13 Nov 2026, issue 56): the free half has run on real infrastructure; no paid provider has ever been called.** Three local runs went end to end on this Mac (Docker task container, the proxy, Ollama's `/v1`, the official harness, the grid and traces), and the pilot's images were validated with the gold and baseline patches. The test suite still drives only the mock upstream on loopback. See "Tested on real infrastructure" and "What has never been tested".

## Commands

```
steproute replay plan      --plan pilot|sparse|FILE --out DIR [--config replay.toml]   # the plan, estimate, caps, month-to-date
steproute replay preflight --plan pilot --out DIR [--config ...] [--env-file F]   # what a real run is missing; changes nothing
steproute replay run       --plan pilot --out DIR [--config ...] [--env-file F]   # a DRY RUN, the same as `plan`
steproute replay run       --plan pilot --out DIR --config replay.toml --execute   # the real run
steproute replay resume    --out DIR --execute                     # continue the plan recorded in DIR
steproute replay resume    --out DIR --execute --retry-errors      # the same, and run the given-up cells again
steproute replay status    --out DIR                               # attempts, statuses, given-up cells, spend per month
steproute replay report    --out DIR [--md docs/results/pilot.md]  # the pilot report
steproute replay ledger show   [--config ...]                      # the shared spend ledger: where, spend, reservations
steproute replay ledger import --out DIR [--config ...]            # bring DIR's own ledger.jsonl into the shared ledger, once
```

`--env-file` also works on `resume`; see "Keys" below. `--retry-errors` (on `resume` only) runs again the cells that were given up: their last attempt ended as `provider_error` or `infra_error` after `max_retries` retries, so they count as finished (see "Failure taxonomy and retries"). Every command takes `--ledger PATH` to use another shared ledger (see "The spend ledger").

Exit codes of `run` and `resume`: `0` every cell finished; `4` some cells are still pending (a cap, the total budget or an out-of-funds answer stopped a provider; resume when there is budget); `3` a fatal stop (the executed configuration differed from the plan); `130` interrupted; `2` a configuration problem or a missing key (nothing ran); `1` from `preflight` when something required is missing.

## Running the pilot, step by step

1. **Prerequisites.**
   - `uv sync --extra proxy` (the runner uses the proxy's pinned LiteLLM).
   - The official harness, pinned: `scripts/install_swebench.sh` puts `swebench==5.0.2` (every dependency at an exact version, `scripts/swebench-requirements.txt`) into `.cache/steproute/swebench-env`. Run from the project root, `steproute tasks check` and the runner find it there without a `[check] harness` line.
   - Docker running, and the 20 pilot images pulled and validated: for each pilot task, `steproute tasks check <id> --patch gold` must pass and `--patch baseline` must fail (`docs/TASKSET.md` section 5, item 5). `--patch empty` is not a validation: the harness skips an empty patch without starting a container. This costs nothing but Docker time (18 to 85 s per check, median about 21 s, measured).
   - The Epoch images on `ghcr.io`: on this Mac `docker pull` of them failed with `403 Forbidden` on the token request, because Docker Desktop presents the credential stored for `ghcr.io`, which these public repositories refuse; an anonymous token works. `scripts/fetch_ghcr_image.py` fetches one anonymously and `docker load`s it. Removing the stored `ghcr.io` login (`docker logout ghcr.io`) would also fix it, but that is the maintainer's credential and was not touched.
   - The dataset rows: `steproute tasks fetch` (problem statements and tests come from them). Run from the project root, the runner finds them at `.cache/steproute/SWE-bench_Verified.jsonl` without a `[check] rows` line.
   - Ollama started with the D2 settings, and every local pilot model pulled:
     `OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_MAX_LOADED_MODELS=1 ollama serve`, then `ollama pull qwen3.5:9b gpt-oss:20b qwen3-coder:30b devstral:24b`.
   - The two API keys in environment variables (by default `ANTHROPIC_API_KEY` and `DEEPSEEK_API_KEY`). They must be **API keys** from the API console, never the Claude subscription (decision D7). DeepSeek is required: without its key the runner stops before doing anything. The keys can also come from a file with `--env-file` (below).
   - About 120 GB of free disk for the images.
2. **Write `replay.toml`** (below). Keep the default caps unless Gate A says otherwise.
3. **Pre-flight:** `steproute replay preflight --plan pilot --out runs/pilot --config replay.toml`. Fix every `FAIL`. It checks the Docker daemon, the images, the harness and rows, each key (set or not, never the value; it cannot check that a key is valid without spending), Ollama and its models, and free disk.
4. **Dry run:** `steproute replay plan --plan pilot --out runs/pilot --config replay.toml`. Read the order, the estimate per provider with its low-high range, the caps, month-to-date and all-time spend (from the shared ledger, every output directory), the ledger's path, and the worst case per run.
5. **The real run:** the same with `steproute replay run ... --execute`. It prints the same plan again, then starts. Local runs (one model at a time) and paid runs (two in parallel by default) go in two lanes at once. It can run for many hours; it needs no attention.
6. **Interrupt and resume:** Ctrl-C stops after the calls in flight and records those runs as interrupted. `steproute replay resume --out runs/pilot --execute` continues, and so does a resume after a crash, a reboot or a cap stop (next month the cap resets).
7. **The report:** `steproute replay report --out runs/pilot` writes `docs/results/pilot.md`: realised cost per configuration, success rates (overall and per stratum), the cache-read share actually observed, steps per run, run statuses, spend per provider and month, the stage-gate decision, and the sparse-grid estimate redone from the measured values.
8. **Before the sparse grid:** check the report's cache-read share for the Claude models (it must be far above zero; see the cache canary), update `src/steproute/tasks/cost_assumptions.toml` and the catalog's `effort_output_mult` from the measured values, and get the go-ahead for the next batch. The sparse plan reuses the pilot's finished cells when it runs in the same output directory.

### Running `gpt-oss:20b` again (issue 75)

The pilot's `gpt-oss:20b` runs finished on `steproute-bash-1`, which had no `apply_patch` and did not quote invalid JSON. To run its three configurations again on the pilot tasks, into a new output directory so the old runs stay as they are, write a plan file, for example `gpt-oss-rerun.toml`:

```toml
name = "gpt-oss-rerun"
description = "gpt-oss:20b again on the pilot tasks, scaffold steproute-bash-2 (issue 75)"

[[block]]
tasks = "pilot"
configs = [["gpt-oss:20b", "low"], ["gpt-oss:20b", "medium"], ["gpt-oss:20b", "high"]]
```

Then, with the same `replay.toml` as the pilot:

```
steproute replay plan      --plan gpt-oss-rerun.toml --out runs/gpt-oss-rerun --config replay.toml
steproute replay preflight --plan gpt-oss-rerun.toml --out runs/gpt-oss-rerun --config replay.toml
steproute replay run       --plan gpt-oss-rerun.toml --out runs/gpt-oss-rerun --config replay.toml --execute
steproute replay report    --out runs/gpt-oss-rerun --md runs/gpt-oss-rerun/report.md
```

The dry run (`run` without `--execute`) was checked on 15 Nov 2026: 60 runs, all local, $0, in the order low, medium, high, and nothing written. `preflight` was not run then, because it asks the Ollama server, which a pilot run was using. The real run was not started. It needs Ollama, and the go-ahead. It writes nothing to `runs/pilot`. If a pilot run is still going, do not start this until the pilot's local lane is done: both would use the one Ollama server, and the runner keeps one model loaded only within one process.

### The configuration file

Strict: unknown sections and keys are errors. Every value shown is the default; an empty file is valid.

```toml example=replay
[caps]                    # USD per provider per UTC calendar month (decision D3)
anthropic = 80.0
deepseek = 10.0
# allow_raise = true      # required to set a cap ABOVE its default (only after Gate A)

[totals]                  # USD per provider over ALL months and ALL output directories; none for Anthropic
deepseek = 10.0           # DeepSeek was funded once with $10 (13 Nov 2026)
# allow_raise = true      # required to set a total ABOVE its default

[schedule.deepseek]       # DeepSeek runs only in its off-peak hours (catalog meta.peak_hours)
offpeak_only = true
# start_margin_s = 7200   # a run starts only with this much off-peak ahead (default: run_timeout_s)
call_margin_s = 600.0     # a run pauses before a call when less off-peak than this is left

[limits]
run_cap_usd = 3.0         # per-run cap: the run ends as a labelled failure (run_cap), not retried
step_limit = 250
max_prompt_tokens = 200000   # largest prompt a paid call may carry; bounds the worst call
max_output_tokens = 16384    # max_tokens on every call
local_context_cap = 32768    # decision D2
command_timeout_s = 60.0
run_timeout_s = 7200.0
max_retries = 2              # whole-run retries after a provider or infrastructure error
call_retries = 2             # retries of one call after HTTP 429 or 5xx, inside a run
call_backoff_s = 5.0
max_observation_chars = 10000
cache_canary_runs = 3        # stop Anthropic if caching is not working (0 disables)
min_cache_read_share = 0.3
min_run_budget_usd = 0.5     # smallest reduced budget a run may start with (see "Budget")

[concurrency]
paid = 2                  # paid runs in parallel; local runs are always one at a time

[providers.anthropic]
api_key_env = "ANTHROPIC_API_KEY"   # the NAME of the variable, never the key
[providers.deepseek]
api_key_env = "DEEPSEEK_API_KEY"
[providers.ollama]
base_url = "http://127.0.0.1:11434/v1"

[environment]
kind = "docker"           # or "local" (tests: a plain directory)
docker = "docker"
manage_ollama = true      # unload the previous local model before the next

[check]                   # unset: the pinned harness and the fetched rows in the project
# harness = [".cache/steproute/swebench-env/bin/python", "-m", "swebench.harness.run_evaluation"]
# rows = ".cache/steproute/SWE-bench_Verified.jsonl"

[ledger]                  # unset: $STEPROUTE_LEDGER, else the per-user state directory
# path = "/somewhere/kept/steproute-ledger.jsonl"   # relative paths are relative to this file
```

## The safety model, in plain terms

- **Nothing runs by default.** Every command is a dry run unless it is `run` or `resume` with `--execute`. Before the first paid call a real run prints the plan, the estimated spend per provider with its range, the caps and month-to-date spend.
- **The caps cannot be passed.** Before every paid run the runner asks: month-to-date spend (from the shared ledger, every output directory) + what every run in flight could still spend, in this runner or any other on the same ledger + this run's worst case ≤ the provider's monthly cap? And, for a provider with a total budget (`[totals]`, DeepSeek $10), the same with all-time spend in the shared ledger against the total. If not, that provider stops; its cells stay pending; other providers and local runs carry on. The check is made under one lock across threads and processes, so parallel runs cannot jointly pass a cap: each admitted run holds a reservation of (its bound - what it has spent) until it ends, written to the ledger. A new output directory does not start again at $0 (issue 60); see "The spend ledger" below.
- **Budget: the whole cap is usable.** When less than a full worst case is left, but at least `min_run_budget_usd` plus one worst-case call, the run starts with a reduced budget: its reservation is exactly the headroom left, and it stops before its next call once its spend reaches (headroom - one worst-case call). That stop is `budget_stop`: not a model failure, not in the grid, not retried now, run again by a later resume when there is budget, and not counted against `max_retries`. A run with the full bound still stops at `run_cap_usd` as `run_cap`, unchanged. The default floor of $0.50 is about twice the central off-peak cost of the dearest DeepSeek run (v4-pro high, $0.23) and 1.5 times it at 1.5 times the estimate, so a run that starts on the last dollars usually finishes instead of being thrown away. What stays unusable is less than the floor plus one worst-case call: $0.58 with flash, $0.83 with v4-pro, so at least 92% of DeepSeek's $10 can be spent. In the test that simulates the pilot's DeepSeek block at 1.5 times the central estimate, $9.44 was spent and 11 of 100 cells were left pending.
- **The worst case of a run** is `run_cap_usd` plus the worst case of one call. The per-run cap is checked before every call, so a run can pass it by at most one call. One call costs at most a `max_prompt_tokens` prompt billed entirely at the highest input rate (a cache write where writes cost extra, at the highest price tier) plus `max_output_tokens` of output, priced with `price_usage` from the catalog; the prompt bound is enforced before each call from the provider's last reported prompt size plus a deliberately high estimate (2 characters per token) of what was added, and a run whose next prompt could pass it ends as `context_limit`. With the defaults: Haiku $3.17, Sonnet $3.66, Opus $4.33, Fable $5.66, DeepSeek flash $3.08, v4-pro $3.33 (all at peak, list prices: reservations never assume the off-peak discount). The cost of that safety: near the end of a budget, with two paid runs in parallel, one worker waits for the other to finish, and the last dollars are used through reduced budgets (next point). Lowering `max_prompt_tokens` or `run_cap_usd` tightens the bound. *Rejected:* checking the estimate instead of the worst case (the high scenario is 4 times the central one, so an estimate-based check could overshoot by a lot); stopping at the first call that passes the cap (that would pass the cap by a whole call every time).
- **Realised cost** is the provider's reported usage priced with `price_usage` from the catalog, per call. A call whose usage is missing (or all zero: LiteLLM fills in zeros when the provider sends none) is charged its worst case and flagged `missing_usage`, never $0. A DeepSeek call is priced at off-peak rates only if both its start and its end fall in off-peak hours (`src/steproute/core/offpeak.py`); otherwise at peak. DeepSeek does not document how a call that crosses a boundary is billed, so this is the safe reading. Each settle line records the `period` (`peak`, `offpeak` or `flat`) and `list_usd`, the same usage at peak (list) price.
- **Off-peak only (DeepSeek, default).** A DeepSeek run starts only when off-peak hours stay open for at least `start_margin_s` (default: the 2-hour run timeout), so a normal run ends before peak starts. While DeepSeek waits, the paid workers take other providers' runs in plan order; when only DeepSeek runs are left, the runner sleeps and prints until when, in UTC and local time. The backstop is in the run's client: no call is sent in peak hours or with less than `call_margin_s` of off-peak left; the run pauses (keeping its reservation) until off-peak resumes, and the pause does not count against `run_timeout_s`. Waiting runs are deferred, never dropped or failed. Chinese public holidays (off-peak at DeepSeek) are treated as working days, because we cannot verify the holiday calendar; that only costs waiting time.
- **Out of funds.** An HTTP 402 or an "insufficient balance" answer ends the run as `balance_stop` and halts that provider for the rest of the invocation. The cell stays pending; nothing is retried and nothing is labelled a failure.
- **The ledger** is the single source of truth for the caps: one shared, append-only, `fsync`-ed file for every output directory (where it lives and why: "The spend ledger" below). Each paid call writes an intent with its worst case *before* the call and a settle line with the realised cost *after*. Month-to-date = settled amounts + the worst case of every unsettled intent. A crash at any point therefore over-counts, never under-counts: killed during a call, the worst case stays charged; a torn line (killed mid-write) is closed with a newline and skipped on reading, and a torn intent means the call never started.
- **The month is the UTC calendar month** of the moment a call starts. Provider consoles report usage in UTC, and the time zone and daylight saving of the machine then do not matter. A run that crosses midnight at the end of the month charges each call to the month it started in.
- **Resume is driven only by what is durably recorded.** `runs.jsonl` holds a start line before a run's first call and an end line when it is over. A cell (task, configuration, repeat) is finished when its latest attempt ended with a terminal status. A run that was in flight when the process died has a start and no end; on resume it is closed as `interrupted`, its spend stays in the ledger, and **the cell is run again**: an interruption is the runner's fault, not the model's, so it says nothing about the task; leaving the cell empty would lose it from the matrix; and the cost of re-running is at most one run per paid slot, already bounded by the cap. Re-runs are bounded by `max_retries`; the abandoned attempt's traces stay in the store under its own session id, and the grid only ever lists the final attempt.
- **Two runners on one directory cannot both spend:** a run takes an exclusive `flock` on `DIR/.lock`; a second runner exits at once. The kernel drops the lock when a process dies. Runners on **different** directories may run at the same time: they share the ledger, its lock and each other's reservations, so together they stay under every cap and total.
- **Keys** come only from the environment variables the config names. `--env-file PATH` (on `run`, `resume` and `preflight`) loads `NAME=value` lines into the runner's own environment first: an optional leading `export`, an optional pair of single or double quotes around the value, `#` comment lines, blank lines, no final newline needed, no interpolation. A variable already set in the environment wins. Every value from the file goes into the redactor. An unreadable file or a malformed line is an error naming only the line number, never its content. The runner never prints, logs or writes them: the dry run shows only "set" or "NOT set", every message and artefact passes through a redactor that replaces the configured key values, and the agent's commands run in a workspace that sees a fixed, harmless environment (no host variables beyond `PATH`), so `env` inside a task shows no key. Only the run's own upstream gets the one key it needs. Per decision D7, the runner uses API keys only; it has no way to use the Claude subscription.
- **The configuration that ran is the one recorded.** Every call is checked: the proxy's response headers (`X-Steproute-Mode` must be `route`, `X-Steproute-Model` and `X-Steproute-Effort` must be the planned ones), the decision the proxy logged, and the outcome's `final_model`. Any difference fails the run as `config_mismatch` and stops the whole runner, because it means a bug that would corrupt every later run.
- **Isolation between runs:** a fresh session id and a fresh proxy per run (so no session or cache state carries over), and a random per-run nonce as the first line of the system prompt, so two runs of the same model cannot read each other's provider cache: Anthropic caches per model and workspace (`docs/INTEGRATION-NOTES.md` section 7), and the nonce makes every run's cached prefix unique from the system prompt on. The one-tool definition before it is identical across runs and below Anthropic's 512-token minimum, so nothing cacheable is shared.
- **Prompt caching is watched.** The task-set estimate assumes Anthropic prompt caching works through the proxy; without it every Claude run costs about 3.2 times as much. After `cache_canary_runs` finished Anthropic runs of at least 5 calls, if less than `min_cache_read_share` of their input was read from the cache, the runner stops Anthropic with a message (the "cache canary"). *(Not tested against the real API.)*

## The spend ledger

Issue 60. Until then each output directory had its own ledger, so a run in a new directory started again at $0 against the monthly caps and DeepSeek's $10 total. Now there is **one ledger for every output directory**, and every limit is counted over it.

### Where it lives, and why

| Order | Source | Example |
|---|---|---|
| 1 | `--ledger PATH` on the command line | a scratch ledger for a try by hand |
| 2 | `[ledger] path` in the replay config (relative to the config file) | the project's `replay.toml` |
| 3 | the environment variable `STEPROUTE_LEDGER` | |
| 4 | the default: `$XDG_STATE_HOME/steproute/ledger.jsonl` when `XDG_STATE_HOME` is set (and absolute), else `~/.local/state/steproute/ledger.jsonl` | |

`plan`, `run`, `status` and `ledger show` print the path in use and which of these chose it. The default is a per-user **state** directory: the XDG place for data that must survive restarts and is not a document. Not a cache (`.cache/` is meant to be clearable, and this is a record of money spent), not the repository (it would differ per checkout or worktree and could be committed or deleted with one), and not an output directory (that was the defect). Per user rather than per project, because the money is one API account: two checkouts of the project spend the same $10.

Next to the ledger the runner keeps `ledger.jsonl.lock` (the admission lock) and `ledger.jsonl.owners/` (one small lock file per live runner, see "After a crash"). The ledger is small; back it up with the output directories. It is meant for **one machine**: `flock` is not reliable on network file systems, so do not put it on a network share or in a folder synced between machines.

**Moving it.** Stop every runner, move `ledger.jsonl` (the `.lock` file and the `.owners` directory can be deleted while no runner runs), then set `[ledger] path` in the config, which `resume` picks up from the recorded `--config`; `--ledger` and `STEPROUTE_LEDGER` are not recorded, so they must be given again on every command. A runner pointed at the wrong (for example a new, empty) ledger stops before spending, because its directory's spend is unknown to that ledger (below).

### How the limits are counted

The ledger's lines, each carrying `out`, the absolute path of the run's output directory:

| Kind | Written | Meaning |
|---|---|---|
| `intent` | before each paid call | the call's worst case, charged until settled |
| `settle` | after the call | the realised cost (`usd`), its `period` and `list_usd` |
| `reserve` | when a run is admitted | the run's reservation (`usd`), its owner (a lock file) and its process id (for people only) |
| `release` | when the run ends, or when its owner is found dead | the reservation no longer counts |
| `import` | by `ledger import` | the record of an import; the imported lines are ordinary intents and settles |

Month-to-date spend is every intent of the current UTC month, every directory; all-time spend is every intent. Settled intents count their realised cost, open ones their worst case. In flight is, for each reservation not released, its amount minus what its run has spent (never below zero). A run is admitted when spend + in flight + its bound fits under the monthly cap and, for a provider with one, the total. The check takes an exclusive `flock` on `ledger.jsonl.lock` and re-reads what other runners appended first, and the reservation is written before the lock is let go, so no two runners can both take the same headroom. A runner's spend inside its reservation does not change the sum (spent goes up as in flight goes down), so a runner that reads the ledger a moment late never sees more room than there is.

A runner refused because another runner holds the headroom stops that provider for the rest of the invocation, as at a cap: its cells stay pending, the exit code is 4, and a resume continues when the other runner has finished. It waits only for its own runs in flight, not for other runners.

### The per-directory view

Every intent and settle line is also appended to the run's own `DIR/ledger.jsonl`, after the shared ledger (the shared ledger is the one the caps read, so it is written first). `report` and `status` read that file for what the directory's runs spent; `status` also prints the shared totals, and `ledger show` lists spend per directory. If a runner dies between the two writes, the next real run in that directory copies the missing lines from the shared ledger and says so.

### Bringing in a directory written before the shared ledger

A directory written before issue 60 has its spend only in its own `ledger.jsonl`. A real run there **refuses to start** (exit 2) and names the fix; a dry run, `plan`, `preflight` (check `spend ledger`) and `status` say the same. The fix:

```
steproute replay ledger import --out DIR [--config replay.toml]
```

It takes the directory's lock (it refuses while a runner is running there), adds every intent and settle line the shared ledger does not hold yet, matched by kind and intent id, tagged with `out` and `imported: true`, and writes one `import` line. Running it again adds nothing. Lines written before issue 57 (no `period`, no `list_usd`) are taken as they are and count in the period `unknown`; an intent that was never settled is imported open and stays charged at its worst case; torn lines are skipped and counted. The directory's own file is not changed. Spend is counted where it happened: the imported lines keep their timestamps and months.

**The pilot run before issue 60** (`runs/pilot`), step by step:

1. Let the pilot runner finish or stop it (Ctrl-C). The import refuses while it holds the directory.
2. Update steproute to the version with the shared ledger.
3. `steproute replay ledger import --out runs/pilot --config replay.toml` (the config the pilot used). It prints the ledger it wrote to and how many calls and settlements it added.
4. `steproute replay ledger show --config replay.toml` and `steproute replay status --out runs/pilot --config replay.toml`: the shared ledger's spend per provider and month must equal the directory's.
5. Every later run, in `runs/pilot` or any other directory, now counts the pilot's spend; `steproute replay plan` shows it as month-to-date and, for DeepSeek, as spent of the $10 total.

### After a crash

- **Killed during a paid call** (between intent and settle): the intent stays open and is charged at its worst case, in every directory's limits, for good. Nothing settles it later; that over-counts, never under-counts.
- **Killed while holding reservations:** each runner holds an exclusive `flock` on its own file `ledger.jsonl.owners/<owner id>.lock` for as long as it lives, and its `reserve` lines name that owner. The kernel drops the lock when the process dies, however it dies (crash, kill, power loss). The next admission by any runner finds the file unlocked, writes a `release` line ("its runner ended without releasing it") and prints it. What the dead runner spent stays: settled calls at their cost, open intents at their worst case. Process ids play no part, so a reused process id cannot keep or free a reservation.
- **In doubt, the money is held:** a reservation whose owner file is missing (deleted by hand) or cannot be checked stays held. To free it, make sure no runner is running, then create the empty file `ledger.jsonl.owners/<owner>.lock` with the owner id from the `reserve` line; the next admission finds it unlocked and releases the reservation.
- **Killed between the shared and the directory write:** repaired by the next run in that directory (above).
- **A torn line** (killed mid-write): closed with a newline by the next write and skipped on reading, as before.

## Design decisions

### 1. The scaffold: our own small loop with mini-SWE-agent's step shape

`BashAgent` (`agent.py`) is about 200 lines: a system message and the task message, then a linear history that only grows; one structured `bash` tool; each command in a fresh subshell with a 60 s timeout; output cut to its head and tail above 10,000 characters; the run ends on `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` (the rest of that output is the patch), at 250 steps, or after three replies in a row without a usable tool call. A reply without a usable tool call (none, an unknown tool, arguments that are not valid JSON, no `command`) is a counted step: the model gets a format-error message and the loop goes on. A tool call that the provider itself could not parse is the same step: Ollama fails the whole request (HTTP 500, `error parsing tool call: raw='...'`) instead of returning the reply, the proxy turns that into the error kind `tool_call_parse` (`docs/PROXY.md`), and the client hands the scaffold an unreadable reply. The model then gets the format error for invalid JSON, after an assistant turn with empty content and no tool call in the history: the reply itself is lost, an empty turn keeps the roles alternating for every chat template (user or tool, assistant, user), and it is the shape an empty reply already has. The call counts as a call with unknown usage (`calls_missing_usage`; charged the worst case, $0 for a local model) and is not retried. The rule is about the provider's answer, not a model name; before issue 71 such a call was a `provider_error`, which restarted the run and, after `max_retries`, left the cell without an outcome. It sits behind two small interfaces, `ChatClient` and `Workspace`, and implements `Scaffold`, so another scaffold can be swapped in.

**Invalid JSON arguments are quoted back (issue 75).** The format error for arguments that are not valid JSON shows the model what it sent and what the parser said, then one sentence with the required shape: one JSON object whose `command` is a single string. The quote is cut at 2,000 characters, with a note that says so. The rule is the same for every provider. When Ollama rejected the call, the arguments and the parser's error come from its HTTP 500 message, which the proxy returns to the client in the field `rejected_tool_call` (`docs/PROXY.md`); when a provider returns the call inside a normal reply, the arguments are the reply's and the parser error is Python's `json` error. If Ollama's message has another shape, the model gets the sentence without a quote. In both cases the call itself does not go back into the history: the assistant turn keeps the reply's text (empty for Ollama) and no tool call. A tool call with invalid JSON in the history would make the next request fail (Ollama rejects it; LiteLLM's Anthropic conversion parses the arguments and raises on JSON it cannot repair). Before issue 75 the scaffold kept such a call, answered it "not executed", and said only "the arguments are not valid JSON". The quoted text reaches only the next prompt and the run's `trajectory.json`, which holds model output anyway. It is not in a log, a trace, `runs.jsonl` or `result.json`.

**The `apply_patch` command (issue 75).** Every workspace has an `apply_patch` command on `PATH`. It applies a patch in the `*** Begin Patch` format (`*** Update File:` with optional `*** Move to:`, `*** Add File:`, `*** Delete File:`, hunks after `@@` with an optional context line, `*** End of File`, `*** End Patch`), given on standard input (`apply_patch <<'PATCH' ... PATCH`) or as one argument. It finds each hunk by its context and removed lines, searching forward from the previous hunk, and falls back to ignoring trailing whitespace. On any failure it changes nothing, exits 1 and names the file and the first line that did not match. Paths must be relative and stay inside the current directory. The script is `src/steproute/replay/apply_patch.py`: written for this project from the format's description, standard library only, and in Python 3.6 syntax because the task images go back to 3.6 (a test parses it with `feature_version=(3, 6)`). In a task container it is written to `/usr/local/bin/apply_patch`, outside `/testbed`, so it never shows in the task's `git diff`. Its `#!` line names the image's system Python (`/usr/bin/python3`, 3.10.12 in all 20 pilot images), not the `testbed` environment: the command then works the same in every image and nothing the model installs or removes in `testbed` changes it; the conda base Python and `python3` on `PATH` are fallbacks. In a local workspace it is in a private `bin` directory beside the repository, run by the runner's own Python. The task prompt does not mention it. Why: `gpt-oss:20b` edits files with `apply_patch`, the edit tool of the environment it was trained for; without the command it got `command not found`. The command is there for every model.

**Scaffold identifier.** `steproute-bash-2` since issue 75 (quoted invalid JSON, invalid calls left out of the history, `apply_patch`). It is written on every run's end line in `runs.jsonl` and in the grid's `meta.scaffold`. Runs that finished before keep `steproute-bash-1`; nothing rewrites them. Nothing in `status`, `resume` or `report` compares the identifier: on a copy of the pilot journal (15 Nov 2026, 521 finished attempts, all `steproute-bash-1`) the three commands gave the same output with and without the change, apart from the report's new column, and no finished cell became pending. A resume in such a directory runs its remaining cells on `steproute-bash-2`, so a configuration can end up with runs on both. The report's per-configuration table has a *Scaffold* column, which says `MIXED` with the counts for such a configuration, and a line under the table lists them. Neither change would have altered a step of the paid pilot runs: their trajectories contain no `apply_patch` call and no invalid JSON (issue 75).

*Rejected: mini-SWE-agent itself* (2.4.6, MIT, read from the PyPI wheel; details in `docs/INTEGRATION-NOTES.md` section 11). It reaches a custom endpoint through LiteLLM and uses structured tool calls, and its history is linear, all of which fit. What does not fit:
- its cost tracking calls LiteLLM's cost map and raises when the cost is 0, which it is for every model name behind our proxy; the switch around that (`cost_tracking: ignore_errors`) also disables its `cost_limit`, so its own money guard would silently be off;
- it retries every failed model call up to 10 times with exponential backoff (tenacity), invisible to a runner that must bound retries and count each attempt;
- its `LocalEnvironment` runs commands with the full host environment (`os.environ | config.env`), so the model could read the API keys;
- it imports `litellm` at module import (before our offline switches are guaranteed) and depends on `datasets`, `textual`, `typer`, `rich`, `openai` and more, none typed for `mypy --strict`.
Keeping it would mean wrapping or patching each of those, which is more code than the loop itself.

**What this does to the cost estimates.** The estimate's per-step shape is preserved: linear history (so prefix caching works as modelled), bash only, a one-tool schema, the same observation truncation, step and cost limits. Our system and task messages are about 1,900 characters against mini-SWE-agent's 4,800 for its SWE-bench template, so `prompt_base` is about 700 tokens lower (2,000 assumed, problem statement included). With about 550 tokens added per step, a 50-step run's mean prompt is near 15,000 tokens, so this lowers input cost by roughly 5%; `growth_per_step` and `cache_hit` should match, because the step shape is the same. Behaviour can still differ (prompt wording changes how models act), so the pilot's measured cost and steps replace the assumptions either way, and the report recomputes the sparse estimate from them.

### 2. How one configuration is applied: a private proxy per run with a fixed policy

Each run starts its own in-process proxy on a fresh loopback port (`proxyrun.RunProxy`): the decision source is the normal `Service` with a `FixedPolicy` of the run's `(model, effort)`, the only upstream serves only that model, and `on_no_decision = "error"` and `on_upstream_error = "error"` mean the proxy never fails open to another configuration or falls back to the client's model (`on_no_decision` was added to the proxy for this). Every step is an ordinary route-mode decision and outcome in the trace store, with `source = "replay"`, the run's session id and `task_id`, and the policy name `replay-fixed`. At the end of the run the task-level result is copied onto every step's outcome through the proxy's `/report` path (`replay_status`, `tests_passed`, `check_status`, and `success` set explicitly for the cap and limit failures). *Rejected:* one shared proxy with a per-session lookup table (one bug away from running a session under another configuration, and the session table and cache state would be shared); calling the providers directly (no step-level traces, and the proxy's cache breakpoints and usage normalisation would be duplicated).

### 3. The plans: built from the manifest, the catalog and the estimate code

`plan.build_named_plan` builds `pilot` and `sparse` from `steproute.tasks.estimate.build_plans`, the same code that produces every number in `docs/TASKSET.md`, so the plan and the document cannot drift: the dry run of the pilot reproduces the document's $41 central ($23-$179) for Anthropic and $18 ($9-$85) for DeepSeek, 180 and 100 runs. The sparse plan's 30-task effort sweep is stratum-balanced like the full set (8 easy, 9 medium, 10 hard, 3 unsolved) and takes the 20 pilot tasks first, so finished pilot cells are reused. The pilot's stage gate (`docs/TASKSET.md` section 5) is a check, not a judgement: after stage 1 (all local configurations, Haiku, Sonnet low), `r` = realised / estimated cost of stage 1's Anthropic runs; if month-to-date + `r` x the rest of the central estimate passes $70, every remaining Anthropic cell is cut to 10 stratum-balanced pilot tasks (3, 3, 3, 1). The gate never cuts another provider: DeepSeek keeps all 20 tasks under its own budget. The decision is written to `gate.json` and kept on resume; if stage 1 did not finish (a cap), the gate cuts. Order: holdout pass last; then stage; then local (by tier), Haiku, Sonnet and DeepSeek (by input price), Opus, Fable; lower effort first within a model; tasks in manifest order within a configuration (configuration-major). A plan file (`[[block]]` with `tasks` and `configs`) is for tests and ad-hoc batches. A plan recorded in an output directory cannot change: resuming with a different cell list is refused.

### 4. What is written, and where

| Path in `DIR` | What | Kept for issue 19? |
|---|---|---|
| `grid.jsonl` | The grid in the evaluator's contract (`header`, `task`, `run` lines), rebuilt from the journal after every session; one run line per finished cell (its last attempt), `meta.status`, `meta.reason`, `meta.attempts`, spend on earlier attempts. `cost_usd` is what was billed (off-peak calls at off-peak prices); `meta.list_cost_usd` is the same usage at peak (list) price, and `meta.calls_by_period` counts calls per period. Compare models on `list_cost_usd`: an off-peak run's `cost_usd` is not a list-price cost | **yes** |
| `grid-holdout.jsonl` | The same for holdout models (`devstral:24b`): the held-out-model split | **yes**, as its own split |
| `traces/` | The trace store of every step (pool models), `source = "replay"`, no prompt text | **yes** (export with `steproute traces export --scrub`) |
| `traces-holdout/` | The holdout model's step traces, in their own store so they can never enter training data by accident | **yes**, as its own split |
| `runs.jsonl` | The run journal: start and end of every attempt | yes (provenance) |
| `ledger.jsonl` | This directory's copy of its own lines in the shared spend ledger (intents and settles, each tagged `out`); what `report` and `status` read for the directory's spend. The caps read the shared ledger | yes (provenance; no prompt content) |
| `plan.json`, `gate.json` | The plan's identity and inputs; the stage-gate decision | yes |
| `runs/<run_id>/result.json` | Status, reason, steps, usage, cost of one attempt | yes |
| `runs/<run_id>/patch.diff`, `check.json` | The submitted patch and the check result | yes (patches are model output on permissive repositories) |
| `runs/<run_id>/trajectory.json`, `workspace.diff` | The full conversation and the raw workspace diff | **scratch**: they contain repository code and model text; issue 19 decides, under D5, whether trajectories are published |
| `.lock` | The runner's lock | scratch |

No file contains a key: every text the runner writes passes through the redactor, and the test `test_keys_never_reach_any_output` scans every file of a run whose model tried to read the environment.

### 5. Concurrency

Two lanes run at once. The **local lane** is one thread: runs strictly one after another, models in tier order with all of one model's configurations together, and before switching model the previous one is unloaded (`keep_alive = 0`), so exactly one is loaded (with `OLLAMA_MAX_LOADED_MODELS=1` as the second line of defence). The **paid lane** has `[concurrency] paid` workers (default 2) that take cells in plan order; each admission goes through the cap guard under one lock (decision above), and a refused admission waits while other runs of that provider are in flight (their reservations may free up) before the provider is stopped. Retries of a failed cell go back to the front of the lane and through the guard again. *Rejected:* running local and paid in one sequential queue (the 15 local hours would delay every paid result without saving anything); unbounded paid concurrency (rate limits, and more reservation headroom lost at the end of a month).

### 6. Failure taxonomy and retries

| Status | What happened | `success` | Retried |
|---|---|---|---|
| `passed` / `failed` | The check ran on the submitted patch (failed includes a patch that does not apply, a test timeout, and no submission) | true / false | no |
| `run_cap`, `step_limit`, `context_limit`, `wall_time` | The run hit a per-run limit | false (a labelled failure, as in the task-set estimate) | no |
| `provider_error` | The provider answered an error (after up to `call_retries` retries of a 429 or 5xx inside the run). Not a tool call the provider could not parse (`tool_call_parse`): that is a format-error step of the model (decision 1) | null | yes, up to `max_retries` whole runs; then given up, until `resume --retry-errors` |
| `infra_error` | Container, workspace, check harness or runner error | null | yes, up to `max_retries`; then given up, until `resume --retry-errors` |
| `interrupted` | The runner stopped while the run was in flight | null | yes, on resume (counts toward `max_retries`) |
| `blocked_sensitive` | The proxy's privacy rule refused to send the step to a cloud model | null | no (it would happen again) |
| `config_mismatch` | What ran was not what was planned | null | no; the runner stops |
| `budget_stop` | The run reached a reduced budget given from the provider's last headroom | null; **pending**, not in the grid | not now; a later resume, when there is budget (not counted in `max_retries`) |
| `balance_stop` | The provider answered HTTP 402 / insufficient balance; the provider is halted | null; **pending**, not in the grid | not now; a resume after funds are added |

Only the first two rows are model outcomes. The last two leave the cell pending. The others are recorded in the grid with `success = null`, which the evaluator treats as unlabelled, never as a failure. A provider error that timed out is charged its worst case (the provider may have done the work); an error answered before any work is charged $0.

**Given-up cells.** A cell whose last attempt is a `provider_error` or `infra_error` after `max_retries` retries counts as finished: it is in the grid with `success = null`, and a resume leaves it alone. `steproute replay status` prints how many cells are in that state, and the dry run and the start of a real run say so too: their plan text counts them on their own (`420 in the plan, 327 finished, 39 given up, 54 to run`) and leaves them out of the order and the estimates, unless `--retry-errors` is given, when they are to run. `steproute replay resume --out DIR --execute --retry-errors` runs each of them again, as further attempts in the same journal (`attempt` goes on counting, the grid's `meta.attempts` counts every attempt). The first attempt of the new round has `"retry_errors": true` on its start line, and `max_retries` counts from that attempt, so the cell gets `1 + max_retries` attempts again. A cell is re-opened at most once per invocation; if the new round is given up too, the cell is finished again until the next `--retry-errors`. Without the option nothing changes.

### 7. The pilot report

`steproute replay report` (`report.py`) reads a finished pilot's directory and writes `docs/results/pilot.md`. It shows each configuration's cost billed and at list price, the scaffold its runs used (`MIXED` when they used more than one; decision 1), spend by time of day, and the cells left pending by `budget_stop` or `balance_stop`. Realised / estimated ratios and the re-estimate use list prices, because the estimates are at peak prices. The re-estimate of the sparse grid prices each sparse configuration at its measured $/run where the pilot ran it, otherwise the task-set estimate times the measured realised/estimated ratio of the same model, or of the same provider, or unchanged (the table counts which). A directory whose paid providers pointed at loopback is marked "MOCK DATA. NOT A PILOT." at the top. The test `test_pilot_report_on_mock_data_is_marked_mock` runs it on mock data; no results file is checked in.

## Tested on real infrastructure (13 Nov 2026, issue 56)

On the maintainer's M4 Pro: Docker Desktop 28.0.4 (arm64, a VM of 8 GB), Ollama 0.40.16 started with `OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_MAX_LOADED_MODELS=1`, the harness `swebench==5.0.2` from `scripts/install_swebench.sh`. No paid provider was called by any route, and no key was set in the environment of any run. The evidence is in `docs/INTEGRATION-NOTES.md` section 14.

- **The 20 pilot images and the official harness on this Mac.** All 20 pulled and ran natively on arm64; 19 grade correctly (gold passes, baseline fails). **`psf__requests-1921` fails even with the gold patch** (pytest 8 in the image, and tests that need `https://httpbin.org`), so no model can pass it: `docs/TASKSET.md` section 8.1 has the table and the harness output. It is marked broken and replaced by `django__django-13401` (issue 58, `docs/TASKSET.md` section 8.3).
- **`DockerWorkspace` on the real images.** The container starts in about 0.3 s, `/testbed` is a clean checkout with the image's own commits, and the runner's `base` commit goes on top. Checked inside the container, not taken from the code: `env` shows only the fixed set (the image's own `ENV` lines such as `HOME` and the conda variables, plus the runner's `LANG`, `PAGER`, `PYTHONDONTWRITEBYTECODE` and friends) and no host variable; `--network none` leaves only `lo` and tunnel stubs, and a DNS lookup fails ("Temporary failure in name resolution"). `BASH_ENV` activates the `testbed` conda environment (`python` is the testbed's 3.11). `git diff` after an edit is the patch; the container is removed at the end (`docker ps` empty after every run).
- **The proxy against a real Ollama `/v1`.** Structured `tool_calls` arrive for every step of `qwen3.5:9b` (effort `none`); every step is a route-mode decision and outcome with `source = "replay"`, policy `replay-fixed`, the planned model and effort, and the configuration check passed on every call. Ollama's `/v1` reports cached prompt tokens (`prompt_tokens_details.cached_tokens`): 96% of prompt tokens were prefix-cache reads.
- **Unloading** with `keep_alive = 0`: after each run `ollama ps` was empty.
- **The context cap in practice:** a run on a long task (`pytest-dev__pytest-5787`) grew to 30,657 prompt tokens in 54 steps and ended as `context_limit` before the next call ("prompt of about 31723 tokens passes the local cap", the cap being 32,768 - 2,048). Reported prompt sizes rose monotonically, so Ollama never truncated a prompt silently.
- **The check after a run:** a submitted patch went through the official harness and came back `passed` (`check.json`, grid `success = true`).
- **What a run writes:** `grid.jsonl` (one run line per cell), `runs.jsonl` (start and end), `ledger.jsonl` (empty: no paid call), `plan.json`, `runs/<id>/` with `result.json`, `trajectory.json`, `workspace.diff`, `patch.diff` and `check.json`, and `traces/` with 100% of decisions joined to outcomes and the task label copied onto every step.
- **Measured speed** (`qwen3.5:9b`, effort `none`, M4 Pro): 2.8-3.9 s per step including the command, about 32-39 output tokens per second over call time, 2.3-3.9 minutes per 40-54-step run. The harness check: 18-85 s per task, median about 21 s (about 15 s of it is the harness stopping its container).
- **Bugs found and fixed** (each with a regression test): a local-only run wrote no `ledger.jsonl`; an HTTP 500 from Ollama, retried by the runner, left that step labelled `success = false` through the proxy's `retried` signal; `--patch empty` could never show that an image's tests fail (the harness skips empty patches), so `--patch baseline` was added; the harness and rows were not found without a `[check]` section; one transient 502 from the datasets-server failed `tasks fetch`.

## What has never been tested

Everything below needs a paid provider, or more time on the real machine than issue 56 had:

- **Any paid call.** The Anthropic path runs through LiteLLM's real `anthropic` provider against the Anthropic-format mock, and DeepSeek through LiteLLM's `deepseek` provider against the OpenAI-format mock; the real APIs, their errors, rate limits and reported usage were never seen. **That the pilot finishes under the caps is untested**; what makes it true is the cap guard, the ledger and the cheapest-first order, all tested against the mock.
- **That prompt caching reaches Anthropic** through the proxy and shows up as cache reads in the reported usage (the task-set document's precondition 1). The cache canary is the guard; the first paid Claude runs are the test.
- **Local models other than `qwen3.5:9b` in the agent loop.** `gpt-oss:20b`, `qwen3-coder:30b` and `devstral:24b` passed the one-call tool smoke test (`steproute catalog check`, all four PASS) only; none of them ran a task, and their speed is not measured. Model switching in the local lane (one model unloaded before the next) never ran with two models in one plan.
- **Parallel containers under load.** Two containers (one agent, one harness) ran side by side once without trouble in the 8 GB VM; the pilot's paid lane (two runs) plus the local lane and their checks (up to four containers) was not tried.
- **DeepSeek's off-peak billing.** That DeepSeek bills a call at half price when it starts and ends off-peak, how it bills a call that crosses a boundary, and that its usage report matches our pricing: none of it has been seen on a real bill. The schedule and pricing are tested only with a fake clock against the mock. Compare the ledger's off-peak total with DeepSeek's console after the first day.
- **DeepSeek's real out-of-funds answer.** Tested with the mock's HTTP 402, passed through LiteLLM as status 402. The real body and LiteLLM's mapping of it were never seen; the text "insufficient balance" is also matched.
- **Real step counts and costs** of the 5.5 and 5.2 Claude models and the current DeepSeek models on this scaffold, which the pilot exists to measure.
- **Wall time**: hours of unattended running, and the 2-hour per-run timeout (`run_timeout_s`; no real run came near it, the longest took under 4 minutes).
- **The shared ledger under real, long runs.** Two runners in two directories on one ledger are tested with real processes against the mock (`tests/test_replay_shared_ledger.py`), as are a killed runner's reservation and a kill between intent and settle; two real runners side by side for hours, and the ledger on anything but a local disk, were not tried.
- **Ollama's unparseable tool call against the real server** (issue 71). The answer was reproduced outside the runner on Ollama 0.40.16 (`/v1/chat/completions`, non-streaming); the runner's handling of it is tested only with the loopback mock sending the same status and body through LiteLLM's real `openai` provider. Whether Ollama's streaming endpoint, or another version, fails the same way is unknown.
- **The quoted format error and `apply_patch` with a real model** (issue 75). The parsing of Ollama's message follows the bodies captured in issue 75; the whole path (rejected call, quoted message, corrected call, `apply_patch`, submission) is tested only against the mock. That `gpt-oss:20b` corrects itself after the quoted message rests on the small canned-loop sample in issue 75, not on a run. `apply_patch` was installed and used inside all 20 pilot images (a scratch file, a patch on standard input, a patch as an argument under the `testbed` Python, a hunk that does not match), not in an agent run.
- **The proxy's privacy rule on real repositories.** Test fixtures in SWE-bench repositories may contain secret-shaped strings; a step that trips the detector is sent nowhere and the run ends `blocked_sensitive`. How often that happens is unknown.
