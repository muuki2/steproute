# Claude Code plugin

The plugin in `plugins/claude-code/` makes everyday Claude Code use produce steproute traces:
one routing decision for every subagent Claude Code starts, and one outcome when that
subagent's run ends, joined by `decision_id` in the trace store. In **route** mode it also
applies the recommended model (and effort) to the subagent as it starts. **Shadow** mode,
the default, logs and changes nothing.

It is a *mod*: a plugin with a JavaScript hooks module that runs inside Claude Code
(`docs/INTEGRATION-NOTES.md` sections 1, 2 and 12). It talks to a local `steproute serve`
(`docs/SERVICE.md`) over loopback HTTP.

Issue 14. The answers to the design questions are in [Design decisions](#design-decisions).

## Install

You need:

- Claude Code **2.1.287 or later** (mods), signed in. Check with `claude --version`.
- [`uv`](https://docs.astral.sh/uv/) and `git`. Node is **not** needed to use the plugin
  (Claude Code runs the hooks module itself); it is needed only to run the plugin's tests.

```bash
git clone https://github.com/muuki2/steproute
cd steproute
uv tool install --from . steproute          # puts `steproute` on your PATH
```

Start the service in a terminal of its own and leave it running. The plugin's service config
restricts decisions to Claude models (see [Which models](#1-which-models-a-decision-may-name)):

```bash
steproute serve --config plugins/claude-code/service/steproute.toml
# steproute serve: listening on http://127.0.0.1:8787
```

Traces go to `~/.steproute/traces` (add `--store DIR` to change it).
The startup warning `no local fail-open action is configured` is expected with this config:
there is no local model to fall back to, so a sensitive task gets no decision.

**Try it for one session** without installing anything into Claude Code:

```bash
claude --plugin-dir /path/to/steproute/plugins/claude-code
```

The line under the prompt reads `steproute: shadow · 0 logged`, and `/steproute` prints the
mode, the service address and what has been logged. Ask Claude for something that uses a
subagent, then:

```bash
steproute traces stats ~/.steproute/traces
```

**Keep it** (adds this repository as a plugin marketplace in your Claude Code setup; not
exercised by the tests here):

```
/plugin install steproute --marketplace muuki2/steproute
```

or from a shell, `claude plugin install steproute --marketplace muuki2/steproute`. Remove it
again with `/plugin` (Installed tab) or `claude plugin uninstall steproute`.

What a clean machine still needs that the steps above do not cover: Claude Code itself and a
login, `uv`, `git`, and Python 3.11+ (uv fetches one if missing). Timing: see
`docs/INTEGRATION-NOTES.md` section 12.

## Modes and settings

| Mode | What happens at a subagent spawn |
|---|---|
| `shadow` (default) | `/decide` is called and the decision logged; the subagent starts exactly as Claude Code chose. The outcome is reported at its end. |
| `route` | As shadow, and the decision's model is set on the spawn (`next({ ...e, model })`) and its effort on that subagent's own requests (`turn.step`). Only for spawns nothing else chose a model for (below). |
| `off` | Nothing is called, logged or changed. |

Settings, highest precedence first, none of which touches your global Claude Code settings:

1. Environment variables for one run: `STEPROUTE_MODE=shadow|route|off`,
   `STEPROUTE_URL=http://127.0.0.1:8787`, `STEPROUTE_TOKEN` (the bearer token, if the
   service uses `[auth]`). Example: `STEPROUTE_MODE=route claude --plugin-dir ...`.
2. The plugin's options (`userConfig`): `mode`, `service_url`, `decide_timeout_ms` (200),
   `report_timeout_ms` (500), `billing` (`subscription` or `api`), `route_agent_types`
   (`general-purpose,Explore,Plan,claude`, or `*`). An installed plugin shows them in
   `/config`; for `--plugin-dir` they can be passed for one run with
   `--settings '{"pluginConfigs":{"steproute@inline":{"options":{"mode":"route"}}}}'`.
3. Defaults.

A mode that does not parse means shadow, and `/steproute` says so. **At a glance:** the status
line under the prompt always names the mode (`steproute: route · 3 logged · 3 routed`) and the
last problem, if any (`· service unreachable`). In `claude -p` there is no status line; the
same text is in the debug log (`--debug-file`).

The service config is `plugins/claude-code/service/steproute.toml` (port 8787, Anthropic
models only, the rule baseline with cascades off, a fail-open default). Keys:
`docs/SERVICE.md`.

## What it does and does not do

Hooks (`plugins/claude-code/hooks/register.mjs`), in order of a subagent's life:

| Event | What the plugin does |
|---|---|
| `session.start` | Reads the settings, registers `/steproute`, shows the status line. |
| `agent.spawn` | Builds the state, calls `/decide` (at most `decide_timeout_ms`), in route mode sets the model, then lets the spawn go on. A resumed run (`e.resume`) gets no decision and no change. Forks and teammates are left alone and not logged. |
| `turn.step` | For a tracked subagent's requests: adds up `usage` (input, output, cache read, cache creation) and the model the API reported; in route mode sets that subagent's effort. Every other request passes untouched. |
| `tool.call` | For a tracked subagent's own tools: counts calls, errors and denials, and the result of test commands (`pytest`, `npm test`, `go test`, ...). The tools that start agents (`Agent`, `SendMessage`, ...) are kept out of its matcher. |
| `turn.complete` | A subagent's run ended: the outcome is sent to `/report` (at most `report_timeout_ms`). |
| `session.end` | Runs still open are reported (`end_reason: session_end`); unsent outcomes get one more second. |

It never changes the main conversation's model or effort, never changes a subagent's prompt,
tools or type, never refuses a spawn, and never throws into Claude Code (every handler catches;
a hook that fails anyway is skipped by Claude Code and the event goes on). It adds four agent
types, `steproute:haiku`, `steproute:sonnet`, `steproute:opus`, `steproute:fable`, each pinned
to that model, for when you ask for a tier by name; their descriptions tell Claude not to use
them otherwise. Route mode leaves them alone (their type is not in `route_agent_types`).

The MCP server (`docs/MCP.md`) ships with the plugin but is **not** loaded by default (see
decision 7 below). To add it for one run: `claude --plugin-dir ... --mcp-config
/path/to/steproute/plugins/claude-code/mcp.json`.

## What is logged

A **decision** per subagent spawn (`docs/SCHEMA.md`, Decision), with `state`:

- `harness = "claude-code"`, `session_id` (Claude Code's session id), `step_idx` (spawns
  counted in the session), `kind = "subagent"`, `task_id` (the Agent tool call id),
  `source = "personal"`.
- `prompt_tokens`: the task prompt's characters / 4 (an estimate; the hook does not see the
  subagent's system prompt or tool list). `cached_prefix_tokens = 0`, no `current_model`.
- `task_text`: the task prompt, sent for sensitive detection and **dropped by the store**
  (only `features.task_text_len` is kept) unless the service sets `capture_text = true`.
- `features`: `subagent_type`, `background`, `workflow`, `nested`, `parent_model`,
  `requested_model`, `prompt_chars`, `routing_mode`, `plugin_version`,
  `claude_code_version`, and `paths` (file-like tokens found in the prompt, at most 32).

An **outcome** per run (`docs/SCHEMA.md`, Outcome): the summed token fields, `final_model`
(the model the API reported, date suffix removed), `latency_ms` (the run's duration),
`escalated = false`, `success = null` unless a labelling signal says otherwise, and
`cost_usd = 0` on a subscription. `signals` holds: `routing_mode`, `applied`, `apply_reason`
(`shadow`, `routed`, `explicit_model`, `agent_type_not_routed`, `fail_open`,
`not_claude_model`, `model_refused`), `recommended_model`/`_effort`, `executed_model(s)`,
`spawn_model`, `model_matches_recommendation`, `subagent_type`, `background`, `end_reason`
(`answer`, `aborted`, `error`, `refusal`, `session_end`, `evicted`, `spawn_denied`),
`requests`, `usage_source`, `turn_usage_matches`, `tool_call_count`, `tool_error_count`,
`tool_denied_count`, `billing`, `decide_ms`, `engine_effort`, and when they apply
`effort_rewrites`, `resumed_runs`, `fallback_not_run`, and the labelling signals
`tool_error` and `tests_passed`.

Traces from Claude Code are personal (decision D5): `steproute traces export` never exports
them.

## The privacy limit

**steproute can advise, but it cannot stop Claude Code from sending a subagent's prompt to
Anthropic.** When the service finds a task sensitive (a secret in the prompt, a path such as
`.env` or `~/.ssh/`), it has no model to offer: the plugin's catalog holds Claude models only,
and a sensitive step may only go to a local model. The service answers 503 with
`"sensitive": true`, nothing is logged, and the plugin lets the spawn go ahead exactly as
Claude Code chose; `/steproute` counts these spawns. The subagent's prompt was written by the
main conversation's model, so it is at Anthropic already, and the subagent will send it again
along with whatever files it reads. Detection sees only the task prompt, not those files, and
it is pattern matching (`docs/PRIVACY.md`). If a task must stay on your machine, do not run it
in Claude Code.

## Troubleshooting

- **Status line says `service unreachable` or `service timed out`.** `steproute serve` is not
  running, or not at `service_url`. The plugin waits 30 seconds before trying again (5
  minutes after a 401, 403, 404 or 405) and meanwhile leaves every spawn to Claude Code.
  Outcomes of decisions already made are kept (up to 64) and resent.
- **No status line, `/steproute` unknown.** The mod did not load: check `claude --version`
  (2.1.287+), run `claude plugin validate plugins/claude-code`, and start once with
  `claude --debug-file /tmp/cc.log --plugin-dir ...`; search the log for `steproute`.
  `--bare`, `--safe-mode` and `"disableAllHooks": true` turn mods off.
- **Decisions but no outcomes.** A run still open is reported at session end; if the session
  was killed, its outcomes are lost. `steproute traces stats` shows the join rate.
- **Route mode does not change the model.** Look at `apply_reason` in the outcome:
  `explicit_model` (Claude passed a model), `agent_type_not_routed` (a custom or plugin agent
  type; add it to `route_agent_types`), `fail_open`, or `model_refused` (`availableModels`).
  `CLAUDE_CODE_SUBAGENT_MODEL_FORCE` and managed settings outrank a mod.
- **The service refuses to start.** `docs/SERVICE.md`, startup checks. Port 8787 in use:
  `--port`, and set `STEPROUTE_URL` to match.

## Check it works (acceptance criterion 2)

`scripts/claude_code_session.sh shadow|route SCRATCH_DIR` runs one real non-interactive
session on the cheapest model with three trivial read-only subagent tasks, against a fresh
service and store, and checks that three decisions joined with three outcomes with non-zero
tokens (`scripts/check_claude_code_traces.py`). It uses your Claude Code login and makes real
requests. Results of the runs made for issue 14: `docs/INTEGRATION-NOTES.md` section 12.

Tests that need no Claude Code: `node --test tests/claude-code/*.test.mjs` (the hooks module
against a fake engine and a loopback fake service) and `tests/test_claude_code_plugin.py`
(the hooks module against the real service built from the plugin's config). Both run in
`scripts/ci.sh`, and are skipped with a message when Node is missing.

## Design decisions

### 1. Which models a decision may name

**Chosen: the service is configured for Claude Code.** `plugins/claude-code/service/steproute.toml`
sets `[catalog] providers = ["anthropic"]` (a service option added for this issue), so the
policy chooses among the four Claude models and the logged candidate set and propensity
describe what Claude Code can actually run. The plugin also checks: a decision naming
anything but a Claude model id or alias is logged but never applied (`not_claude_model`).
When there is no usable answer (timeout, unreachable, any error, a 503) the spawn keeps
Claude Code's own choice and nothing is applied. A sensitive task gets a 503 from this
catalog (no local model), and the plugin then does nothing; see [The privacy limit](#the-privacy-limit).

*Rejected:* filtering in the plugin alone (mapping a DeepSeek or local recommendation to a
Claude model would log one action and run another, so the propensities would be wrong for
training and evaluation); refusing a sensitive spawn with `{ deny }` (the prompt is already at
Anthropic, so it protects nothing and breaks the user's work).

### 2. What goes in the state at spawn

What the `agent.spawn` input carries, and nothing guessed: kind `subagent`; `tools = []`
because the hook does not receive the subagent's tool list (its type is in
`features.subagent_type`; the docs say only that Explore and Plan are read-only); a prompt
size of task-prompt characters / 4, labelled as such; the task text for sensitive detection,
which the store drops; file-like tokens from the prompt as `features.paths`; and no cache
(`cached_prefix_tokens = 0`, no `current_model`), because a subagent's first request never
reads the parent's cache. The real sessions showed that a subagent's first request can still
read a few thousand cached tokens of the system prompt and tool definitions that other
sessions on the same model cached; 0 overstates the cost, which is the cost model's rule for
unknowns.

*Rejected:* guessing tools per built-in agent type (not documented exactly; a wrong list would
change which rule fires); a token count from a tokenizer (no dependencies, and the unseen
system prompt dominates anyway).

### 3. Cascades

Nothing can transparently retry a subagent, so a fallback step cannot be run. **The plugin's
service runs the rule baseline with cascades off** (`service/rules.toml`), so what is logged is
what can be executed. If a decision still carries a fallback (another service config), the
plugin runs the first step only, records `escalated = false` and the unrun chain as
`signals.fallback_not_run`.

*Rejected:* telling the main model about the fallback so it can retry (changes the main
conversation and its cache); re-spawning on failure (a second subagent the user did not ask
for).

### 4. Outcome and usage accounting

Per-request `usage` from `turn.step` is summed per `agentId` (input, output, cache read, cache
creation); requests of the main conversation and of untracked agents are ignored. A run ends
at its `turn.complete` (each run of a subagent's loop is one turn), whose summed `usage` is
recorded as a cross-check (`turn_usage_matches`) and used when no `turn.step` was seen
(`usage_source`). Background subagents need nothing special: their requests carry their
`agentId` and their run ends with its own `turn.complete` (in the real sessions all three
subagents ran in the background, Claude Code's default). A resumed run (`SendMessage`) gets no
new decision; its usage is added to the same run and the outcome is reported again (the store
keeps the last outcome per decision). Runs still open at session end are reported then;
past 256 open runs the oldest is reported as `evicted`.

Signals are facts: `tool_error` when a tool call of the run errored, `tests_passed` from the
last test command it ran. Nothing else is claimed, so `success` stays `null` unless one of
those decides it (`docs/LABELS.md`). The executed model is the one the API reported for the
run's requests (`final_model`, `executed_models`), next to `recommended_model`, `spawn_model`
and `model_matches_recommendation`; in shadow mode they usually differ, and a reader must
treat the decision's action as advice, not as what ran.

*Rejected:* parsing transcripts (their format is internal); OpenTelemetry (needs the user's
collector setup); inferring success from the subagent's answer text.

### 5. Cost on a subscription

**`cost_usd = 0`** with `signals.billing = "subscription"`: there is no per-request charge, and
`steproute traces stats` then prints `total cost: $0.0000` rather than a figure someone could
read as money spent. The catalog-rate equivalent is recomputable at any time from the stored
tokens and `final_model` (`realized_cost`, `docs/COST-MODEL.md`), and the router's own cost
estimates never read `cost_usd`. With `billing = "api"` the field is left empty and the service
prices it at catalog rates, as for any API traffic.

*Rejected:* letting the service fill `cost_usd` at API rates (indistinguishable from real
spend in every report); a second, plugin-side price table (would drift from the catalog).

### 6. Failure and latency

Every call to the service is raced against a timer (`$.http.fetch` alone only gives up after
30 seconds, and time inside a `$` call does not count against a hook's budget): 200 ms for
`/decide`, 500 ms for `/report`. A connection refused quickly is retried once in the time
left. On a timeout, a refusal or a 5xx the plugin stops calling for 30 seconds (5 minutes after
401/403/404/405), so a dead service costs at most one timeout per 30 seconds. Outcomes that
could not be stored are kept (at most 64) and resent after the next successful call. Memory is
bounded: 256 open runs, 128 finished runs kept for resumes, 256 delay samples. Every handler
catches; none rejects into Claude Code.

Measured added delay per spawn in the real sessions: `/decide` took 1 to 5 ms inside the hook
(`decide_ms`), and Claude Code's debug log timed the whole `agent.spawn` hook, worker hop and
the engine's own spawn included, at 8 to 26 ms (details in `docs/INTEGRATION-NOTES.md`
section 12).

### 7. Configuration and modes

Environment variables for one run, then the plugin's `userConfig` options, then defaults (see
[Modes and settings](#modes-and-settings)); nothing is written to the user's settings. The
status line names the mode at all times, and `/steproute` gives the details. The MCP server is
shipped (`plugins/claude-code/mcp.json`) but **opt-in**: its tool descriptions tell the main
model to call `route_task` before every spawn, which changes the main conversation and would
log a second, unjoined decision for each spawn the hooks already log.

*Rejected:* a config file under `~/.claude` or the project (edits a tool's setup the user owns);
loading the MCP server by default (above).

### 8. Language and packaging

Plain JavaScript ES modules (`.mjs`), no build step, no npm dependencies, typed with JSDoc
(`// @ts-check`, `import('claude-code').Register`). The mod interface allows exactly this. One
constraint it does force: Claude Code's validator accepts `$` only at a call site
(`$.noun.event(...)`), so `$` is never passed around; each hook hands the collector a small
adapter of closures. Layout: the plugin at `plugins/claude-code/` (a marketplace file at the
repository root points there), its tests in `tests/claude-code/` (Node's built-in runner) and
`tests/test_claude_code_plugin.py`, both in `scripts/ci.sh`, which prints a skip message when
Node is missing. Users install `steproute` with `uv tool install` and the plugin with
`--plugin-dir` or the marketplace.

### 9. Older Claude Code versions

**The `PreToolUse` fallback is not shipped.** It could set a spawn's model on versions before
mods (2.1.287), but it sees no per-request usage and nothing at all for background subagents,
which are now Claude Code's default, so it cannot produce the joined outcomes with token counts
this issue is for. It would also be a second implementation in another language. It was not
tested. Before 2.1.287 Claude Code has no mods, so the plugin does nothing there (not tried on
an older version); update Claude Code.
