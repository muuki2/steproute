# dsh plugin

The plugin in `plugins/dsh/` makes DeepSeek Harness (dsh) sessions produce steproute traces:
one routing decision for **every step** of every agent (a step is one model request plus the
tool calls it asked for), and one outcome per step, joined by `decision_id` in the trace store.
In **route** mode it also applies the decision's model and effort to the step, where dsh
allows that. **Shadow** mode, the default, logs and changes nothing.

It is a cordis plugin in plain JavaScript that dsh loads as it is. It talks to a local
`steproute serve` (`docs/SERVICE.md`) over loopback HTTP and never to anything else.

Issue 15. Built for and pinned to **dsh 0.2.2-rc.3** exactly (there is no stable dsh release;
the project calls itself a developer preview with breaking changes). What was checked in dsh's
source, and how, is in `docs/INTEGRATION-NOTES.md` section 13. The answers to the design
questions are in [Design decisions](#design-decisions).

## Privacy: read this first

**dsh's default setup uploads your session log to DeepSeek, and steproute cannot stop that.**
The base bundle mounts `session-log-deepseek`, which puts the session's events (your prompts,
the model's answers, tool results, the working directory) into a `dsh_session_log` field of
every request to the DeepSeek API, incrementally, up to 8 MiB per request. Its sibling
`plugin-package-inventory-deepseek` adds the names and versions of your active plugins, this
one included. Every request through dsh's DeepSeek adapter also carries an anonymous user id
(`$DSH_HOME/.anonymous-user-id`) and the session id in headers, and that cannot be configured
away. This is dsh's behaviour, independent of steproute.

steproute's `sensitive` flag decides which *model* a step may go to. It does not see dsh's own
uploads, and when only cloud models are configured it has no safe model to offer anyway (see
[decision 2](#2-which-models-a-decision-may-name)). If a session must not reach DeepSeek, do not
run it on a DeepSeek route. To turn the uploads off, add to your profile's
`cordis.patch.yml` (`$DSH_HOME/profiles/<profile>/cordis.patch.yml`):

```yaml
- id: session-log-deepseek
  disabled: true
- id: plugin-package-inventory-deepseek
  disabled: true
```

dsh's README for `session-log-deepseek` says that turning it back on later uploads the backlog,
including what was recorded while it was off. The full list of what dsh 0.2.2-rc.3 sends out,
and the switch for each, is in `docs/INTEGRATION-NOTES.md` section 13.

The plugin itself sends the service each step's state (sizes, tool names, model ids, file
paths from tool arguments) and, for sensitive detection, the step's new text: the user's
prompt at a turn's first step, the tool results before a later step. The service drops that
text before storing (`docs/PRIVACY.md`) unless you configured `capture_text = true`. Traces
from dsh are personal (decision D5) and `steproute traces export` never exports them.

## Install

You need:

- dsh **0.2.2-rc.3** and Node **24.9 or later** (dsh's own requirement, its `engines` field;
  dsh 0.2.2-rc.3 does not start on Node 22). Checked with Node v24.22.3.
- [`uv`](https://docs.astral.sh/uv/) and `git` for steproute.

```bash
git clone https://github.com/muuki2/steproute
cd steproute
uv tool install --from . steproute                     # puts `steproute` on your PATH
steproute serve --config plugins/dsh/service/steproute.toml
# steproute serve: listening on http://127.0.0.1:8789
```

Leave the service running. Its config restricts decisions to the models dsh can run (DeepSeek
by default; see [decision 2](#2-which-models-a-decision-may-name)). The startup warning
`no local fail-open action is configured` is expected with it.

**Try it for one run** without installing anything into dsh. Write an overlay that inserts
the plugin row (the path must be absolute):

```yaml
# steproute.yml
- insert:
    - id: steproute
      name: /path/to/steproute/plugins/dsh/index.js
```

```bash
dsh headless --patch ./steproute.yml "explain what this repository does"
```

**Keep it**: install the plugin as a bundle into a profile (dsh forwards to pnpm and links the
directory; not exercised by the tests here):

```bash
dsh plugin --profile headless add /path/to/steproute/plugins/dsh
dsh --profile headless --dump-config     # shows a "# == steproute-dsh" layer
```

Remove it with `dsh plugin --profile headless remove steproute-dsh`. The bundle declares
`"engines": { "dsh": "0.2.2-rc.3" }`: on any other dsh version dsh skips it with a warning and
starts without it (`dsh plugin allow-version` overrides that, at your own risk).

## Modes and settings

| Mode | What happens at each step |
|---|---|
| `shadow` (default) | `/decide` is called and the decision logged, without the step waiting for the answer; the step runs exactly as dsh would have run it. The outcome is reported when the step is over. |
| `route` | As shadow, and between steps the decision's model and effort are selected for the next step (`agent.model.select()`). The first step of a turn cannot be changed and is logged only. |
| `off` | Nothing is called, logged or changed. |

Settings, highest precedence first:

1. Environment variables for one run: `STEPROUTE_MODE=shadow|route|off`,
   `STEPROUTE_URL=http://127.0.0.1:8789`, `STEPROUTE_TOKEN` (the bearer token, only if the
   service uses `[auth]`; it is never read from a file), `STEPROUTE_STATUS=stderr` (see
   below).
2. The plugin row's `config` in a patch layer (`plugins/dsh/cordis.patch.yml` shows every key):
   `mode`, `serviceUrl` (loopback only: `http://127.0.0.1:<port>` or `http://localhost:<port>`),
   `decideTimeoutMs` (200), `reportTimeoutMs` (500), `routeSubagents` (true),
   `statusToStderr` (false), `models` (the mapping in
   [decision 2](#2-which-models-a-decision-may-name)), and `enabled: false` as an off switch. A patch replaces a row's whole `config`, so restate the keys you keep.
3. Defaults.

A value that does not parse falls back to the safe default (an unknown mode means shadow, a
non-loopback address means the default address) and is reported; nothing in the config can
stop the plugin from loading. To remove it completely, `disabled: true` on the row.

**Which mode is active.** The plugin writes `steproute 0.1.0: <mode> mode, service <url>`
when it loads, and when dsh shuts it down a summary line: decisions, outcomes, how many were
applied and restored, the median time a step waited, and the last problem (for example
`steproute: route · 3 decisions · 1 outcomes · 1 applied · 1 restored · step wait median
1.16 ms · service unreachable`). Both go to dsh's logger, which shows nothing in a headless
run (it has no exporter there), so set `STEPROUTE_STATUS=stderr` (or `statusToStderr: true`)
to get them on stderr. It is off by default because stderr belongs to the terminal UI in
interactive profiles. `dsh --profile <name> --dump-config` shows the row and its config.
Every outcome records `signals.routing_mode`.

## What it does

The hooks, in the order of a step (`plugins/dsh/index.js`; dsh's names, each checked in the
source, `docs/INTEGRATION-NOTES.md` section 13):

| dsh event | What the plugin does |
|---|---|
| `agent/created` | Remembers the agent's model selection: your model, the one to fall back to. |
| `agent/pre-step` (waterfall, after `next()`) | Notes the selection this step was fixed with. For the first step of a turn, asks `/decide` without waiting; that decision is logged, never applied. |
| `session/event` (`step/start`, `request/header`, `assistant/message`, `tool/call`, `tool/result`, `approval/*`, `llm/retry`, `model/selection`, `step/end`, `turn/end`) | Builds the step's outcome from the session log as it is written. A `model/selection` the plugin did not make (you changed the model) becomes the new fallback. |
| `tools/post-execute` (waterfall, after `next()`) | In the last tool call of a batch: asks `/decide` for the next step. In route mode it waits for the answer (at most `decideTimeoutMs`) and applies it with `agent.model.select()`; in shadow mode it does not wait. |
| `session/flush` | When an idle session is flushed (the headless runner does this before it exits), waits up to a second for outcomes still being sent. |
| `agent/disposed` | Reports what is still open and forgets the agent. |

It never changes a prompt, a message, a tool or a tool result, never refuses a step, never
writes to the session log except through `agent.model.select()` (dsh's own `model/selection`
event), and never throws into dsh: every handler catches, and each waterfall returns exactly
what `next()` returned.

## What is logged

A **decision** per step (`docs/SCHEMA.md`, Decision), with `state`:

- `harness = "dsh"`, `session_id` (dsh's session id, one per agent), `step_idx` (decisions
  counted in the session), `source = "personal"`.
- `kind`: `main` for the first step of a turn, `tool_followup` for a step after a tool batch,
  `subagent` for every step of a subagent (its session header says `origin: subagent`).
- `prompt_tokens`: the previous request's whole prompt plus its output plus the new text's
  characters / 4 (an estimate, said so in `features.prompt_tokens_basis`).
- `current_model`, `current_effort`, `cached_prefix_tokens`: the route and effort of the
  previous request, and its whole prompt (see [decision 3](#3-the-state-sent-per-step)).
- `tools`: the tool names of the latest request header.
- `task_text`: the new text, for sensitive detection, dropped by the store.
- `features`: `step_role` (`turn_start`, `after_tools`, `unplanned`), `agent_kind`,
  `dsh_turn`, `selected_route`/`selected_effort` (the route in effect when asked),
  `cache_owner_route`, `parent_session` (subagents), `paths` (from tool arguments and the
  prompt, at most 32), `routing_mode`, `plugin_version`, `built_for_dsh`.

An **outcome** per step (`docs/SCHEMA.md`, Outcome): the step's token counts from
`assistant/message.usage` (`input_tokens` is uncached input, `cache_read_tokens` the cache
hits, `cache_write_tokens` dsh's `cacheWriteTokens`, 0 on DeepSeek), `final_model` (the
catalog id of the route that ran), `latency_ms` (step start to step end), `escalated = false`,
`success = null` unless a labelling signal says otherwise, and `cost_usd` filled by the
service from the catalog. `signals` holds `routing_mode`, `step_role`, `applied`,
`apply_reason`, `recommended_model`/`_effort`, `selected_route`/`_effort` (what the step was
fixed with), `executed_route`/`_effort` (what ran), `selection_ran`,
`model_matches_recommendation`, `effort_matches_recommendation`, `model_switched`,
`requests`, `failed_attempts`, `llm_retry_count`, `tool_calls_requested`, `tool_call_count`,
`tool_error_count`, `tool_denied_count`, `tool_aborted_count`, `approvals_*`, `test_runs`,
`reasoning_tokens`, `decide_ms`, `plugin_wait_ms`, `end_reason`, `step_ran`, and when they
apply `fallback_not_run`, `interrupted` and the labelling signals `tool_error`,
`tests_passed` and `user_rejected`.

`apply_reason` is one of: `routed`, `shadow`, `turn_start`, `no_tool_batch`, `late`,
`fail_open`, `unmapped_model`, `unmapped_effort`, `provider_not_registered`,
`subagent_not_routed`, `no_agent`, `select_failed`. Decisions that never got an answer
(timeout, unreachable, an error status, a 503) are not in the store at all.

## The proxy fallback

If a dsh upgrade breaks the plugin's hooks, the steproute proxy (`docs/PROXY.md`) still works,
because dsh can send any model route to an OpenAI-compatible endpoint through `llm-pi-ai`:

```yaml
- id: llm-pi-ai
  config:
    providers:
      steproute:
        apiKeyEnv: STEPROUTE_PROXY_KEY     # any value unless the proxy sets [server] token_env
        api: openai-completions
        baseURL: http://127.0.0.1:8788/v1
        models:
          - id: deepseek-v4-pro
            compat:
              thinkingFormat: deepseek
```

Then pick `steproute/deepseek-v4-pro` as the model. The proxy decides per request and logs its
own traces (harness `proxy`). What you lose: dsh's session log then records the model id you
picked, not the one the proxy chose, so the plugin's outcome accounting (tool errors,
approvals, tests) is gone, and the proxy sees requests, not dsh steps.

## Limits

- **The first step of a turn is never routed.** dsh reads the step's model synchronously before
  any hook it awaits ([decision 1](#1-where-the-decision-is-taken-and-applied)). In a typical
  coding turn that is one step in several; it runs on whatever was selected last.
- **A cascade's fallback is never run** ([decision 4](#4-cascades)); the service's rules for dsh
  have cascades off.
- **Outcomes in flight are lost when dsh exits while the service is down.** They are kept in
  memory (at most 64) and resent after the next successful call, not written to disk. Seen
  through the CLI: a session whose service was killed lost the outcomes due after the kill.
  With the service up, the last step's outcome (sent at turn end) did arrive before the
  headless runner exited.
- **Prompt size is an estimate**, built from the previous request's usage and the new text.
  The first step of a session does not know the system prompt's size.
- **Only the shell tools (`bash`, `pwsh`) count as test runs**, only the common runners
  (`pytest`, `npm test`, `go test`, `cargo test`, ...), and only a foreground run that
  finished: a background job or a timed-out run gives no verdict. Compile errors are not
  detected.
- **Validated against dsh 0.2.2-rc.3 only**, and through the CLI only in the headless profile
  with the plugin inserted by a `--patch` overlay (five sessions on Node v24.22.3, inside a
  sandbox that allowed loopback connections only). Loading it as a bundle with
  `dsh plugin add`, and the TUI and web profiles, were not run
  (`docs/INTEGRATION-NOTES.md` section 13).
- **The first step's prompt size is badly underestimated**: in the CLI sessions the turn-start
  state said 269 tokens (the user's text) for a request billed at 1650 (the system prompt
  and tool definitions are unseen).

## Troubleshooting

- **Nothing is logged.** Is the service running at `serviceUrl`, and does the summary line say
  `service unreachable` or `service timed out`? After such a failure the plugin waits 30
  seconds before asking again (5 minutes after 400, 401, 403, 404, 405, 411, 413, 415) and
  meanwhile leaves every step to dsh. Is the row there (`--dump-config`)? Is the bundle
  skipped for a version mismatch (dsh prints a warning naming `steproute-dsh`)?
- **Route mode does not change the model.** Look at `apply_reason`: `turn_start` (cannot be
  routed), `unmapped_model` or `unmapped_effort` (add the model or effort to `models`),
  `provider_not_registered` (the mapping names a route dsh has no adapter for), `late` (the
  service answered after `decideTimeoutMs`), `fail_open`, `subagent_not_routed`.
- **The session switched back to my model.** That is fail-open: the step after a tool batch
  had no usable decision (service away, timeout, a decision it could not apply), so the plugin
  put your selection back. If you pick a model yourself, that becomes the fallback.
- **The service refuses to start.** `docs/SERVICE.md`, startup checks. Port 8789 in use: run
  it with `--port`, and set `STEPROUTE_URL` to match.

## Check it works

Tests that need no dsh (both in `scripts/ci.sh`, skipped with a message without Node):

- `node --test tests/dsh/*.test.mjs`: the plugin against a fake dsh loop
  (`tests/dsh/fake_dsh.mjs`, which emits dsh's events in dsh's order) and a scripted service:
  shadow and route, the first step of a turn, fail-open on timeout and on every status code, a
  late answer, the cache owner after a switch, usage and signal mapping, a sensitive step,
  unmapped models, a fail-open decision, cascades, the user's own model change, subagents,
  off, bounded memory, config parsing, the entry module and the HTTP client.
- `tests/test_dsh_plugin.py`: the plugin's entry module and HTTP client against a real
  `steproute serve` built from `plugins/dsh/service/`, including a service killed with
  SIGKILL between turns.

Opt-in sessions on the real dsh, which need the pinned install (about 1 GB):

```bash
(cd tests/dsh/pinned && npm ci --ignore-scripts)
scripts/dsh_inprocess_session.sh route /tmp/dsh-check-1          # kills the service after request 2
scripts/dsh_inprocess_session.sh shadow /tmp/dsh-check-2 none   # keeps it
# the dsh CLI (macOS): put a Node 24.9+ build's bin first on PATH for this command only
PATH=/path/to/node-v24.x-darwin-arm64/bin:$PATH scripts/dsh_session.sh route /tmp/dsh-check-3
PATH=/path/to/node-v24.x-darwin-arm64/bin:$PATH scripts/dsh_session.sh shadow /tmp/dsh-check-4 none
```

- `scripts/dsh_inprocess_session.sh` composes the core of dsh-base from the real dsh
  libraries in one Node process (session, projections, agent registry, LLM runtime, system
  prompt, tools, native executor, agent loop, and the `[model changed …]` notice the headless
  runner installs), a scripted model adapter registered as route `deepseek-official`, one tool
  and the plugin. Two turns of three steps; it checks that every turn completed and the store.
  No provider, no key, nothing leaves loopback.
- `scripts/dsh_session.sh` runs the real CLI, `dsh headless`, against the repository's mock
  upstream (`scripts/dsh_mock_upstream.py`) on loopback, with HOME, `DSH_HOME` and
  `DSH_AGENTS_HOME` in the scratch directory and a patch that turns off every outbound path
  found in dsh 0.2.2-rc.3. dsh runs inside `/usr/bin/sandbox-exec` with
  `scripts/dsh-loopback-only.sb`, which blocks every outbound connection except to loopback;
  `scripts/dsh_sandbox_check.sh` proves that before each session (an external host by name
  and by address must fail, the mock must answer), and the script does not start dsh if it
  fails. The mock kills `steproute serve` with SIGKILL as model request N arrives (default 2;
  `none` keeps it). Afterwards it prints dsh's own session log per step
  (`scripts/dsh_session_log.mjs`), what the mock received, the sandbox's network denials from
  the system log, and the store check. It refuses to run on Node older than 24.9 and says how
  to put Node 24 on `PATH`.

Results: `docs/INTEGRATION-NOTES.md` section 13.

**What the pinned install pulls in.** `tests/dsh/pinned/package-lock.json` (integrity hashes
for every package, all from registry.npmjs.org) pins `@deepseek-ai/dsh@0.2.2-rc.3` and its
closure: 743 lockfile entries (674 installed on macOS arm64), 415 under `@deepseek-ai/`
(every `@deepseek-ai/dsh-*` at exactly 0.2.2-rc.3, plus `cordis` 4.1.0, `schemastery`
3.19.0, the cordis plugins, `libreoffice-kit` and `node-addon-system` with their platform
builds), and third-party SDKs and native modules: `openai`, `@anthropic-ai/sdk`,
`@google/genai`, `@aws-sdk/*`, `google-auth-library`, `@opentelemetry/*`, `undici`, `zod`,
`sharp`, `koffi`, `node-pty`, `sherpa-onnx` (speech), `protobufjs`. Five packages have install
scripts (`@deepseek-ai/dsh-subprocess-local`, `@google/genai`, `koffi`, `node-pty`,
`protobufjs`); `--ignore-scripts` skips them all, and the in-process session runs without
them. `node_modules` is in `.gitignore`. The plugin itself (`plugins/dsh/`) has no
dependencies at all.

## Design decisions

### 1. Where the decision is taken and applied

**Chosen: ask in `tools/post-execute` of the last tool call of a batch, apply there with
`agent.model.select()`; log the first step of each turn without applying it.**

dsh's loop fixes a step's model with `agent.model.current()`, synchronously, as the first thing
pre-step does (`dsh-agent-loop/src/agent.ts:294`), and nothing it awaits comes before that
read. After a step's model request, the loop awaits the tool batch (`agent.ts:596`), and inside
it each call's `tools/post-execute` waterfall runs, in model order, just before that call's
`tool/result` is committed (`dsh-tools/src/batch.ts:139-150`, `pipeline.ts:343-361, 494-497`).
So the last call's post-execute sees every other result of the batch in the session log, and
is still before the next step's read. The plugin asks `/decide` there, waits at most
`decideTimeoutMs`, and calls `select()`; the next step's pre-step then reads that selection.
In the in-process session and in the CLI sessions the `model/selection` event lands between
the batch's `tool/call` and `tool/result`. Through the CLI, dsh's own session log then shows
a new `request/header` with the selected model and effort, and the next requests reached the
DeepSeek endpoint (the mock) as `deepseek-v4-pro` with `output_config.effort: low`.

That the model which ran is the one selected is visible in the outcome: the selection the
step was fixed with comes from `agent/pre-step`'s `modelSelection` (`agent.ts:303`,
`signals.selected_route`), and what ran is `assistant/message.message.source`, set from the
request's provider and model (`agent.ts:572-583`, `signals.executed_route`). The request's
provider, model and effort are the selection's: dsh throws if anything changes them after
step preparation (`agent.ts:641-646`). `signals.selection_ran` compares the two, and
`final_model` is the catalog id of what ran.

The first step of a turn follows the user's message, and the only hooks before its read are
synchronous. Its decision is asked for in `agent/pre-step` without waiting, logged with
`apply_reason = turn_start`, and never applied. A step after steering (no tool batch before
it) is treated the same way (`no_tool_batch`).

The plugin's selection stays in force after the turn, so the next turn starts on the model
the last step ran on. A fail-open (no usable decision between steps) puts the user's own
selection back.

*Rejected:* rewriting the request in `llm/stream` (loop requests are deep-frozen there,
`dsh-llm/src/index.ts:64-76`, and a short-circuit would make the logged header and the message
source wrong); `agent/request` (deprecated, and dsh throws on a model or effort change,
`agent.ts:641-646`); calling `select()` from the `session/event` feed (an append cannot
re-enter another append's publication, `dsh-session/src/index.ts:961`); asking in the first
post-execute of a batch (the results still to come would be missing from the state); restoring
the user's model at every turn end through `agent/turn-stopping` (tried: the in-process
session then switched model twice per turn, a full cache miss each time, the opposite of what
a cost router is for); prefetching a decision for the next turn's first step at the previous
turn's end (its prompt is not known yet).

### 2. Which models a decision may name

**Chosen: the service is configured for dsh, and the plugin maps catalog ids to dsh routes.**
`plugins/dsh/service/steproute.toml` sets `[catalog] providers = ["deepseek"]`, so the policy
chooses among models dsh has a route for and the logged candidates and propensities describe
what can run. The plugin row's `models` maps each catalog id to a dsh `provider` route, the
dsh `model` id and an effort mapping (steproute's `none` is dsh-llm-deepseek's `off`,
`model-info.ts:6-31`). The default covers `deepseek-flash` and `deepseek-v4-pro` on route
`deepseek-official`, the DeepSeek API route dsh-base mounts (`dsh-base/cordis.patch.yml:90-94`,
`dsh-llm-deepseek-api-key/src/index.ts:20`). A decision is applied only if its model is mapped,
its effort is in the model's mapping, and the route has a registered adapter
(`ctx.llm.listProviders()`, `dsh-llm/src/index.ts:498`); otherwise it is logged with the
reason and the user's model runs the step.

No usable answer (timeout, unreachable, an error status, a 503, a `fail_open:` decision) means
the step runs on the user's model. A sensitive step with only cloud models configured gets a
503 with `"sensitive": true`: there is no model the service may name, so the plugin logs
nothing for it and the step runs on the user's model, which is a cloud model. Add a local
model (an `llm-pi-ai` route to Ollama, mapped in `models`, and `"ollama"` in the service's
providers) and the service can keep such steps local.

*Rejected:* mapping a DeepSeek or local decision onto whatever the user configured in the
plugin (one action logged, another run, wrong propensities); blocking a sensitive step (dsh's
own session-log upload has already sent the history, and a refused step breaks the session).

### 3. The state sent per step

**Chosen: what the session log shows, and nothing guessed.** Kind as above; tools in scope
from the latest `request/header`; the prompt as the previous request's whole prompt
(`inputTokens + cacheReadTokens + cacheWriteTokens`) plus its output plus the new text's
characters / 4; the new text (the prompt at a turn's first step, the batch's tool results
otherwise) as `task_text` for sensitive detection, dropped by the store; file paths from the
tool calls' arguments (path-like keys and shell commands) and the prompt.

The cache owner is the previous request's route and effort, and its whole prompt is the cached
prefix. Every dsh model switch is a full miss: providers cache per model, dsh injects a
`[model changed …]` notice (`dsh-agent/src/model-selection.ts:133-171`) and logs a new request
header, and the headless persona names the model in the system prompt
(`dsh-headless/cordis.patch.yml:12`). With one owner in the state, the cost model prices any
other model as a miss. Going back to a model that ran earlier may in fact hit part of its old
cache; treating it as a miss overstates the cost, which is the cost model's rule for unknowns.

*Rejected:* counting tokens with a tokenizer (no dependencies; the system prompt is unseen
anyway); a multi-owner cache state (the schema has one owner).

### 4. Cascades

A dsh step's answer is committed to the session log as it streams, and nothing can re-run a
step, so a fallback step cannot be honoured. **The dsh service's rules have cascades off**
(`plugins/dsh/service/rules.toml`), so decisions carry one step and what is logged can run. A
decision that still carries a fallback (another service config) runs its first step only, with
`escalated = false` and the unrun chain in `signals.fallback_not_run`.

*Rejected:* applying the fallback to the next step after a failure (that is another step with
another state, and logging it as this decision's escalation would be wrong); forking the session
to retry (`SessionStore.fork` exists, but it changes the user's session history).

### 5. Outcome accounting

One outcome covers one step: from `step/start` to `step/end`, its model request (with any
failed attempts and retries) and the tool calls it asked for. It is reported when the next step
starts (`end_reason = continued`) or the turn ends (`turn_<reason>`: `completed`, `error`,
`aborted`, `max-tokens`, `blocked`), so the last step of a turn carries how the turn ended.
Usage comes from `assistant/message.usage`, dsh's `TokenUsage`, whose counts are disjoint
(`dsh-llm/src/types.ts:174-196`): `inputTokens` to `input_tokens`, `cacheReadTokens` to
`cache_read_tokens`, `cacheWriteTokens` to `cache_write_tokens`, `outputTokens` (reasoning
included) to `output_tokens`, `reasoningTokens` kept as a signal.

Labelling signals are facts (`docs/LABELS.md`): `tool_error` when a `tool/result` of the step
has `isError` and is neither a denial (`APPROVAL_*` codes) nor a cancellation (`ABORTED*`);
`tests_passed` from the last test command the step's shell calls ran (dsh renders a non-zero
exit as a final `[exit code: N]`, `dsh-tool-bash/src/render.ts:17-66`, and does not mark it as
an error); `user_rejected` when an approval was answered `rejected` (dsh words the denial "the
user rejected tool …", `dsh-tools/src/pipeline.ts:468`). An `allowed-once` is counted, not
labelled: approving a command says nothing about whether the step was right. Provider retries
(`llm/retry`) are counted as `llm_retry_count`, not `retry_count`: they are transport failures
(rate limits, 5xx), not the same step being redone because it was wrong. With none of those,
`success` stays `null`; most steps are unlabelled.

*Rejected:* reading the compressed session file (the live feed is the same data, in order,
without zstd or file paths); inferring success from the answer text; a judge.

### 6. Failure and latency

Each call to the service is raced against a timer: 200 ms for `/decide`, 500 ms for `/report`.
A fast refusal is retried once in the time left. A timeout, a refusal or a 5xx stops calls for
30 seconds, a 4xx configuration problem for 5 minutes, so a dead service costs at most one
timeout per 30 seconds and refusals cost about a millisecond. The HTTP client is `node:http`
with keep-alive, not the global `fetch` dsh may route through a proxy. A decision that arrives
after the timeout is logged as `late` and never applied. Outcomes that could not be sent are
kept (at most 64) and resent after the next success. Memory is bounded: 64 agents, 128 tool
calls per step, 64 unsent outcomes, 256 delay samples. Every handler catches; none rejects
into dsh; a waterfall listener returns `next()`'s result unchanged.

Measured added delay per step in route mode (`signals.plugin_wait_ms`, the time a step waited
in post-execute): 1.01 to 1.16 ms in the CLI sessions and 0.85 to 4.18 ms in the in-process
sessions, against a local service on an Apple M-series laptop, the same as the `/decide`
round trip. The first step of a turn waits for
nothing, and in shadow mode no step waits (the decision is asked for and the step goes on).
Details: `docs/INTEGRATION-NOTES.md` section 13.

### 7. Language and packaging

**Plain JavaScript ES modules with JSDoc types, no build step, no dependencies.** dsh loads a
plugin as a module exporting `name` and `apply` (`docs/user/develop/basic/index.md:15-27` in
the dsh package); the bundle guide's own example is a plain `index.js`
(`publish.md:46-54`). A config schema must implement Standard Schema; the plugin exports a
small hand-written one, so it imports nothing from dsh (`cordis/src/fiber.ts:50-61`). The
package is `steproute-dsh` with `dsh.bundle.patch` pointing at its `cordis.patch.yml` and
`engines.dsh` pinned to `0.2.2-rc.3`. It has no `peerDependencies`, because it imports no dsh
package; with an `engines.dsh` field dsh checks that alone (`dsh-app-boot/src/profile.ts:699`).
A mismatch makes dsh skip the bundle and start without it (`profile.ts:705`), which is what
"never take a session down" needs on an upgrade.

The pinned dsh used for the opt-in sessions lives in `tests/dsh/pinned/` with its own lockfile,
outside the plugin, so linking the plugin never drags 1 GB of dsh into a profile.

*Rejected:* TypeScript with a compiler step (nothing in dsh requires it, and a build would add
a pinned toolchain and a `lib/` to keep in sync); importing `@deepseek-ai/schemastery` for the
config (a peer dependency for six fields).

### 8. Configuration

The plugin row's `config` in any patch layer, overridden for one run by `STEPROUTE_MODE`,
`STEPROUTE_URL` and `STEPROUTE_TOKEN`; the token only from the environment. The service
address must be loopback. Off is `mode: off`, `enabled: false`, or `disabled: true` on the
row. The active mode is in the load line and the shutdown summary (on stderr with
`STEPROUTE_STATUS=stderr`, since dsh's logger is silent in headless), in `--dump-config`, and
in every outcome. See [Modes and settings](#modes-and-settings).

*Rejected:* a steproute config file under `$DSH_HOME` (another file to keep in sync with the
patch layers dsh already has); a slash command or status surface (headless dsh has none, and
the web surfaces would need client plugins).
