# OpenAI-compatible proxy

`steproute proxy` is a local server that speaks the OpenAI chat-completions API. Point any client at it. For each request the proxy:

1. builds a `StepState`;
2. asks the router for a decision;
3. rewrites the model and reasoning effort;
4. calls the upstream, cascading to the next step if the response fails a check;
5. logs an `Outcome` with normalised token counts and the cost from steproute's own catalog.

It is the collector for any harness that can change its base URL, and the path the replay runner (issue 18) drives the model pool through.

Implementation: `src/steproute/collectors/proxy/` (config, session, checks, source, upstream, engine, server). The test mock is `src/steproute/collectors/mock_upstream.py`. The proxy needs the `[proxy]` extra (`litellm==1.109.1`); `import steproute` and `steproute --version` work without it.

## Quick start

```
uv sync --extra proxy
steproute proxy --config proxy.toml          # prints: steproute proxy: listening on http://127.0.0.1:8788/v1
```

Then point your client at `http://127.0.0.1:8788/v1`. Any API key works unless `[server] token_env` is set. The proxy never forwards the client's key: upstream keys come from the environment variables the config names.

```
curl -s http://127.0.0.1:8788/v1/chat/completions -H 'Content-Type: application/json' \
  -H 'X-Steproute-Session: demo' \
  -d '{"model": "qwen3-coder:30b", "messages": [{"role": "user", "content": "hello"}]}'
```

Start in `mode = "shadow"` to watch the decisions before letting them change anything.

## Configuration

One TOML file. **Unknown sections and keys are an error.** Relative paths resolve against the file's directory. Secrets are never in the file: `api_key_env`, `token_env` and `salt_env` hold the NAME of an environment variable. A value that is not a valid variable name is refused, and the refusal does not echo it. A named variable that is unset stops the proxy at start.

This example is loaded by the test suite, so it stays valid:

```toml example=proxy
[server]
host = "127.0.0.1"          # non-loopback needs token_env
port = 8788
# token_env = "STEPROUTE_PROXY_TOKEN"   # clients must then send Authorization: Bearer <token>

[proxy]
mode = "route"              # "route" applies decisions; "shadow" decides and logs only
harness = "proxy"           # StepState.harness
on_upstream_error = "client_model"   # or "error"
# extra_params = ["some_vendor_field"]  # forward more request fields (api_base etc. never)

[session]
header = "X-Steproute-Session"
max_sessions = 1024
idle_s = 3600
chars_per_token = 4.0       # starting ratio for the prompt estimate, recalibrated per session
# salt_env = "STEPROUTE_SESSION_SALT"   # key for the fallback session hash (default: random per process)

[decisions]
source = "inprocess"        # or "http" (a running `steproute serve`)
service_config = "service.toml"   # inprocess: policy, catalog, store, fail-open, privacy
# url = "http://127.0.0.1:8787"   # http
# token_env = "STEPROUTE_TOKEN"   # http
timeout_ms = 250            # http /decide client timeout
report_timeout_ms = 2000

[checks]
empty_output = true
tool_call_parse = true
schema = true
refusal = false

[privacy]
task_text = "none"          # or "last_user" (the store still drops it unless capture_text)
local_fallback = "qwen3-coder:30b"   # for sensitive requests when no decision is available

[aliases]
# "gpt-4o" = "claude-sonnet-5-5"     # client model name -> catalog id

[[upstream]]
name = "ollama"
provider = "openai"         # Ollama's OpenAI-compatible endpoint, not LiteLLM's ollama provider
base_url = "http://127.0.0.1:11434/v1"
local = true                # requires a loopback base_url
models = ["qwen3.5:9b", "gpt-oss:20b", "qwen3-coder:30b"]
cache_ttl_s = 300           # Ollama keep_alive default

[[upstream]]
name = "anthropic"
provider = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
models = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-2"]
drop_params = ["temperature", "top_p", "top_k"]
forced_tool_choice = "auto"
cache_breakpoints = "route" # the default for anthropic; "always" also marks shadow requests
cache_ttl_s = 300           # must not exceed 300 while breakpoints are on (5-minute cache)

[[upstream]]
name = "deepseek"
provider = "deepseek"
api_key_env = "DEEPSEEK_API_KEY"
models = ["deepseek-flash", "deepseek-v4-pro"]
cache_ttl_s = 3600
```

### `[[upstream]]` keys

| Key | Default | Meaning |
|---|---|---|
| `name` | required | Label for logs and `/v1/models`. |
| `provider` | required | `openai` (any OpenAI-compatible server), `anthropic` or `deepseek`; this is LiteLLM's provider prefix. |
| `models` | required | Catalog ids served. A model may be served by only one upstream. |
| `base_url` | LiteLLM's default | Required for `openai`. Must be `http(s)` and must not carry credentials. |
| `api_key_env` | none | Name of the variable holding the key. Without it the proxy sends a placeholder, which is fine for Ollama. |
| `local` | `false` | The upstream runs on this machine. It must have a loopback `base_url`, and only a local upstream may serve a catalog model with `local = true`. Sensitive steps go only to local upstreams. |
| `model_names` | `{}` | Catalog id to the name the upstream expects. |
| `reasoning_values` | `{}` | Translates a catalog `effort_map` value into the value sent as `reasoning_effort`, for a server that names levels differently. The default catalog needs none: its values are the ones each upstream's endpoint accepts. |
| `drop_params` | `[]` | Request fields removed in route mode. |
| `forced_tool_choice` | `keep` | `auto` turns `tool_choice: "required"` or a named tool into `"auto"` in route mode. |
| `cache_ttl_s` | 300 | After this many seconds without a call, the cached prefix is assumed gone. With `cache_breakpoints` on it may not exceed 300, the lifetime the proxy requests. |
| `cache_breakpoints` | `route` for `anthropic`, else `off` | Where the proxy adds Anthropic prompt-cache breakpoints: `route` (requests it rewrites), `always` (also shadow, fail-open and client-fallback requests), `off`. Only `provider = "anthropic"` may set it; DeepSeek and Ollama cache without markers. See decision 10. |
| `timeout_s` | 600 | Per-call timeout. |

Startup checks: every served model is in the catalog, and `local_fallback` is served and local. A non-holdout catalog model that no upstream serves produces a warning: if the policy picks it, that cascade step is skipped (signal `skipped: ["no_upstream"]`).

### Request headers

| Header | Meaning |
|---|---|
| `X-Steproute-Session` (name configurable) | Session id. Without it the proxy uses a keyed hash of the messages up to and including the first user message, prefixed `h:`. |
| `X-Steproute-Kind` | `StepState.kind` (default `main`; for example `subagent`). |
| `X-Steproute-Task` | `StepState.task_id`. |
| `X-Steproute-Source` | `personal` (default) or `replay`. The replay runner sets `replay`. |

### Response headers

`X-Steproute-Session`, `X-Steproute-Mode` (`route`, `shadow` or `fail_open`), `X-Steproute-Decision-Id` (the id to use with `/report`), `X-Steproute-Model` and `X-Steproute-Effort` (what actually ran), and `X-Steproute-Attempts`. The body's `model` echoes the name the client sent, so clients that check it are not confused; the real model is in the header and the trace.

### Endpoints

- `POST /v1/chat/completions`: also accepted as `/chat/completions`.
- `GET /v1/models`: the served catalog ids and aliases.
- `POST /report` with `{"decision_id", "signals", "success"?}`: late signals, for example `tests_passed`. The proxy merges them into the outcome it emitted for that id and re-reports the whole outcome, so tokens and cost are kept. The labeller then recomputes `success` unless you send one. It answers 404 if the proxy no longer holds the outcome (more than 4096 outcomes ago, or after a restart); in that case report a full `Outcome` to the service directly.
- `GET /healthz`: counters (`requests`, `source_failures`, `upstream_errors`, `privacy_blocked`, `report_failures`, `fail_open`) and the session count.

Security follows `docs/SERVICE.md`: it binds loopback by default, a non-loopback host needs a token, requests without a token must carry a loopback `Host` header, the body is capped (32 MiB), and request lines, bodies and prompt text are never logged.

## What is logged

Per request (route mode), one `Decision` written by the service, then one `Outcome`:

- `Decision.state`:
  - `harness`, `session_id`, `step_idx`, `kind`, `task_id`, `source`;
  - `prompt_tokens` (estimate) and `cached_prefix_tokens`, `current_model`, `current_effort` (session state);
  - `tools` (names only) and `sensitive`;
  - `features.proxy = {mode, client_model, stream, forced_route?, dropped_client_params?}`. `client_model` is the catalog id, or `null` when the client's name does not resolve.
  - `task_text` is `null` by default, and no text or paths are put in `features`.
- `Outcome`:
  - `final_model`: the last attempt that answered;
  - `escalated`: more than one upstream call was made;
  - token fields summed over attempts, in the `docs/SCHEMA.md` convention;
  - `cost_usd`: each attempt priced with `price_usage` from the catalog, then summed. LiteLLM's `response_cost` is never used;
  - `latency_ms`: the whole request;
  - `success = null`: the service's labeller sets it from the signals.
- `Outcome.signals`:
  - `proxy_mode` and `attempts`;
  - `failed_checks`, `upstream_errors`, `skipped`, `params_dropped`, `stream`, `stream_error`, `client_disconnected` and `fallback_to_client`, when they apply;
  - `cache_markers`: `"proxy"` or `"client"`, who placed the Anthropic cache breakpoints on the final attempt (absent when it had none);
  - the label signals of the final answer (below);
  - `retried = true`, added later to the previous step's outcome when the client re-sends an identical request.

## Design decisions

### 1. Shape: our own small server, sync, on the LiteLLM SDK

The proxy is a stdlib `ThreadingHTTPServer` (HTTP/1.1, keep-alive, one thread per connection). It exposes `/v1/chat/completions` and `/v1/models` and calls `litellm.completion` itself. That makes the cascade possible, because the proxy owns the retry loop.

A stream is forwarded with chunked transfer encoding. Each chunk LiteLLM yields is written and flushed at once, so at most one chunk is held. No dependency beyond what `litellm` brings is added.

*Rejected:*

- **A LiteLLM proxy plugin (`async_pre_call_hook`).** It can rewrite model and effort, but not cascade: fallbacks fire only on exceptions, and exceptions in success and stream hooks are swallowed (`docs/INTEGRATION-NOTES.md` section 4). It would also bring in the whole LiteLLM proxy server, with its telemetry and UI.
- **An asyncio server (aiohttp ships with LiteLLM).** It scales to more concurrent streams, but a local collector serves a handful of agents. The sync design shares its patterns and security checks with `steproute serve` and can embed the sync `Service` directly.

### 2. Decisions: one interface, two sources

`DecisionSource` has `decide(state)` and `report(outcome)`. There are two implementations:

- `InProcessSource` wraps a `Service` built from a `steproute serve` config. It has the same policy, time budget, fail-open, labelling and store, with no extra hop. This is the default.
- `HttpSource` calls a running `steproute serve` (`/decide` with a 250 ms client timeout, and `/report`).

Any failure (unreachable, timeout, non-200, a 503 from the service, a bad body) is a `SourceError`, and the proxy **fails open**: it runs the client's own request unchanged. The privacy rule overrides this. A sensitive request whose client model is not on a local upstream goes to `[privacy] local_fallback`. Without one it is refused with HTTP 503 `steproute_sensitive` and sent nowhere. Nothing is logged for a request with no decision, because there is no decision to join; it is counted in `/healthz`.

*Rejected:* HTTP only, which costs a hop and a second process for the common single-machine case; and in-process only, which prevents several collectors from sharing one policy and store.

### 3. Session state

A session (`SessionTable`, `session.py`) holds:

- the next step index;
- the **cache owner**, `(model, effort)` of the last attempt that got a response;
- `cached_tokens`, the provider's real prompt size of that attempt, or 0 if it was below the model's `cache_min_tokens`;
- the units of that prompt;
- its time, the upstream's `cache_ttl_s`, and a calibrated chars-per-token ratio.

A request is split into **units**: unit 0 is the tool definitions, then one unit per message. Each unit is identified by a hash of its canonical JSON.

- **Before the call:**
  - If the new request extends the last executed prompt (all its units are a prefix), the cached prefix is `cached_tokens`.
  - If it diverges after `k` units (history edited or compacted), the cached prefix is the character share of the common prefix times the old prompt size.
  - If the tools changed, or the TTL passed, it is 0.
- **After a call that returned usage** (whether or not the answer passed its check): the owner becomes that attempt's `(model, effort)` and `cached_tokens` becomes the real prompt size, or 0 for an Anthropic request that carried no cache breakpoint (Anthropic caches nothing then; decision 10). This is `state_after_failure` with measured numbers. A failed cascade step therefore leaves the cache owned by the failed model, which is what the next step pays for.
- **After a call that failed with no response** (an HTTP error or no connection): the provider did nothing, so the session is unchanged.
- **Shadow and fail-open calls** own the cache as `(client model, client effort)`, or `None` if the client sent no effort in the shared vocabulary.

Sessions live in an LRU of at most `max_sessions` (1024). One idle longer than `idle_s` (1 h) is dropped. A dropped session starts again at step 0. Updates take a per-session lock.

The 5-step test (`test_five_step_session_switching_model_once`) asserts the exact state at every step. The provider reports prompt sizes P = 1200, 1500, 1800, 2100, 2400:

| step | action | `current_model` | `current_effort` | `cached_prefix_tokens` | `prompt_tokens` |
|---|---|---|---|---|---|
| 0 | qwen3-coder:30b medium | `null` | `null` | 0 | estimate of the whole prompt |
| 1 | qwen3-coder:30b medium | qwen3-coder:30b | medium | 1200 | 1200 + estimate of the 2 new messages |
| 2 | **gpt-oss:20b high** (the switch) | qwen3-coder:30b | medium | 1500 | 1500 + estimate |
| 3 | gpt-oss:20b high | gpt-oss:20b | high | 1800 | 1800 + estimate |
| 4 | gpt-oss:20b high | gpt-oss:20b | high | 2100 | 2100 + estimate |

The state describes the cache before the step, so the switch appears one step later. At step 2 the cost model sees that `gpt-oss:20b` does not own the 1500 cached tokens and charges the switch.

*Rejected:*

- trusting the provider's `cached_tokens` as next step's cached prefix: it says what *this* request read, not what the next one can read;
- keeping per-model caches: `StepState` has one owner;
- no expiry: a long-lived proxy would grow without bound.

### 4. Prompt tokens before the call

When the request extends the last prompt, the estimate is the provider's real count for that prompt plus an estimate for the new units. For a fresh or diverged session it is the estimate of the whole prompt from the common prefix on.

The per-unit estimate is `ceil(chars / ratio) + 4` per message, plus 1000 per image part. `chars` is the canonical JSON length. The ratio starts at `chars_per_token` (4.0) and is recalibrated after every call from the real prompt size, clamped to [0.5, 20]. So only the new messages are ever estimated, and errors do not accumulate.

*Rejected:*

- `litellm.token_counter`: tiktoken, the wrong tokenizer for Claude, Qwen and DeepSeek, and slow on a 100K-token conversation inside the 50 ms budget;
- a fixed `chars / 4` over the whole conversation: its error grows with the session.

### 5. Sensitive detection on real requests

Each session remembers the units it has scanned. A request is scanned only from the first unit that differs, so normally only the new messages.

- **What is scanned:** the core `detect_sensitive` runs over every string in content, names, refusals and tool-call arguments (secret patterns). It also runs over paths: values of path-like keys in tool-call arguments (`path`, `file_path`, `filename`, ...), plus path-looking tokens in the text, at most 256 per scan, against the sensitive globs.
- **Sticky:** a positive verdict lasts for the session, because the secret stays in the context and the provider's cache. It also survives a compaction that removes it, which is conservative.
- **Where it goes:** the flag goes into `StepState.sensitive`. The policy then only chooses local models, and the proxy refuses any non-local upstream for the step (counted as `privacy_blocked`).
- **What is stored:** `task_text` defaults to `none`, so no prompt text is stored or sent to the service. `last_user` puts the last user message in `task_text` for a policy that reads it; the store still drops it unless `capture_text` is set. Paths are not put in `features`.
- **Shadow mode:** a sensitive request whose client model is not local runs the router's local decision. It is logged as a route-mode decision with `features.proxy.forced_route = "sensitive"`, because the client's model may not see it.

*Rejected:*

- scanning the whole conversation each step: about 1.7 ms per 20,000 characters, unbounded, inside a 50 ms budget;
- sending the text to the service as `task_text` for it to scan: the same cost, and prompt text over the wire by default.

### 6. Parameter compatibility after a rewrite

A request written for one model is kept valid on another by a per-upstream **parameter policy**, applied only when the proxy rewrites (route mode):

- the client's `reasoning_effort`, `thinking` and `reasoning` are replaced by the decision's effort, mapped through the catalog `effort_map` and then the upstream's `reasoning_values`. An effort missing from the map sends no reasoning parameter;
- the upstream's `drop_params` are removed;
- `forced_tool_choice = "auto"` downgrades a forced tool choice.

Every change is listed in `signals.params_dropped`. Shadow mode and fail-open send the client's parameters unchanged (cache breakpoints aside, decision 10).

The catalog's `effort_map` values are the ones each upstream's endpoint accepts, so no `reasoning_values` is needed with the default catalog. For `qwen3.5:9b` they are `none` and `medium` for Ollama's `/v1` endpoint (`docs/INTEGRATION-NOTES.md` section 9).

What LiteLLM 1.109.1 does on its own is checked in `docs/INTEGRATION-NOTES.md` section 8:

- It refuses `temperature != 1`, `top_p` and `top_k` client-side for all four Claude models. That is a 400 before the call, so `drop_params` is needed.
- It refuses forced tool choice for Sonnet 5.5 and Opus 5.5, but **not for Fable 5.2**, which is missing from its model map. Only our `forced_tool_choice = "auto"` covers Fable 5.2.
- For `reasoning_effort = "none"` it sends no thinking parameter, so Opus 5.5 and Fable 5.2 (which cannot switch thinking off) get their default adaptive thinking. The catalog does not list `none` for Anthropic anyway.

The proxy does not set LiteLLM's global `drop_params`, which silently drops many more parameters.

**When an upstream still rejects the request**, the attempt counts as failed (`upstream_errors`) and the cascade moves on. If every routed attempt fails without a response and `on_upstream_error = "client_model"` (the default), the proxy makes one last call with the client's original request, privacy permitting. That outcome has `fallback_to_client = true`. Its label signals are left out, so the client's model never stands in as evidence for the router's chain. With `"error"`, the client gets the last upstream status.

**A refusal** (Anthropic `stop_reason: "refusal"`, which LiteLLM maps to `finish_reason: "content_filter"`, or a non-empty `message.refusal`) is recorded as the raw signal `refusal = true`, which the labeller ignores. A refusal can be the right answer, so it is neither success nor failure. It cascades only if `[checks] refusal = true`, and is never also counted as empty output.

*Rejected:*

- `litellm.drop_params = True`: global, silent, and it misses Fable 5.2's forced tool choice;
- refusing rerouted requests that carry such parameters: it breaks the client.

### 7. Shadow mode

The client's request runs unchanged on its own model, and the router's decision is logged with `features.proxy.mode = "shadow"`. Unchanged includes cache breakpoints: an Anthropic upstream adds none in shadow mode unless it sets `cache_breakpoints = "always"` (decision 10). The outcome of the call that ran is reported under the id `shadow:<decision id>`, which is also the `X-Steproute-Decision-Id` the client gets for late reports.

No decision has that id. So joins (`TraceStore.joined`, `compute_stats` success tables, training and off-policy evaluation) leave the recommendation unjoined and never pair the client model's result with the router's action. A learning policy's `report` ignores the unknown id. The spend is still real and is counted in the store's totals, as an orphan outcome. Exports withhold orphans, so shadow outcomes never leave the machine.

*Rejected:*

- attaching the outcome to the recommendation with `final_model` set to the client's model: any code keyed on `decision.action` would learn from it;
- dropping the outcome: it loses real spend and the data for comparing the client's choice with the router's.

### 8. Cascade checks

The checks run on the first choice, non-streaming, in route mode only. A failed check moves to the next step of the decision's `escalate_to` chain. The last answer is returned even if it failed.

| Check (`[checks]`) | Fails when | Label signal (`docs/LABELS.md`) |
|---|---|---|
| `tool_call_parse` | A tool call has no function name, a name not among the request's tools, or `arguments` that are not a JSON object (empty arguments are allowed). | `tool_error = true` (error tier) |
| `empty_output` | There are no tool calls and `content` is missing or only whitespace (text parts are joined). | `schema_valid = false`: the reply breaks the minimal response contract |
| `schema` | The request set `response_format` `json_object` or `json_schema`, the reply is text, and it is not a JSON object or fails the schema (`jsonschema`). A broken schema is not held against the model. | `schema_valid = false`; a pass gives `schema_valid = true` |
| `refusal` (off by default) | See decision 6. | raw `refusal = true` only |

Label signals describe the **final** answer. Earlier failures are listed in `failed_checks`. A passing empty-output or tool-call check is not evidence of success and adds nothing.

### 9. LiteLLM's own network behaviour

Before importing `litellm`, `upstream.load_litellm` forces three variables:

- `LITELLM_OFFLINE_MODE=true`;
- `LITELLM_LOCAL_MODEL_COST_MAP=true`;
- `LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS=true`.

Without them, the pinned version fetches the model cost map from GitHub on import (plus a background retry thread), and fetches the Anthropic beta-header config on the first Anthropic call. Offline mode also stops Hugging Face tokenizer and template downloads.

The loader also:

- clears every callback list;
- turns message logging off;
- suppresses debug banners;
- leaves `drop_params` off;
- passes `max_retries=0` on every call.

Proxy telemetry, update checks and blog fetches live in LiteLLM's proxy server, which is never imported.

This is checked by two tests:

- `test_litellm_import_and_setup_make_no_network_call` runs the loader in a fresh interpreter with every non-loopback name lookup refused, and also forces the beta-header load. It asserts no lookup was attempted.
- Every integration test runs under a guard that fails the test on any non-loopback connect or lookup.

Details and line references are in `docs/INTEGRATION-NOTES.md` section 8.

### 10. Prompt-cache breakpoints (Anthropic)

Anthropic caches a prompt only up to a `cache_control` breakpoint. Without one, every step pays the full input price, and the cost model's warm prompt (the cached prefix read, the remainder written; `docs/COST-MODEL.md`) never happens. DeepSeek and Ollama cache matching prefixes on their own and take no markers. So for an Anthropic upstream the proxy adds breakpoints itself, and the session state believes exactly what it asked for. What LiteLLM does with each marker, with line references, is in `docs/INTEGRATION-NOTES.md` section 9.

**Placement.** Two breakpoints of the four Anthropic allows, the same two on every step:

1. **The static prefix:** `cache_control: {"type": "ephemeral"}` on the last text block of the leading system messages. Tools render before the system prompt, so this caches tools and system together. Without a system prompt it goes on the last tool definition. This read point survives anything that happens later in the conversation: an edited or compacted history still reads tools and system.
2. **The moving end:** Anthropic's request-level automatic breakpoint, a top-level `cache_control: {"type": "ephemeral"}`. Anthropic puts it on the last cacheable block and walks back from a block that cannot carry one. Step `n` writes the whole prompt; step `n + 1` sends that prompt byte for byte plus the new messages, its breakpoint looks back, finds step `n`'s entry and reads it, and the new messages are written. This is exactly the convention the session state and the cost model use.

The proxy never changes the bytes of the conversation to place a marker: a string system prompt gets a message-level marker, which LiteLLM copies onto the block it builds, so the rendered prefix is the same on every step. A mid-conversation `role: "system"` message is not part of the static prefix and is not marked.

*Rejected:*

- **A marker on the last message, placed by the proxy.** It has to work for every message shape (string, parts, a tool result, an assistant turn ending in tool calls), and the automatic breakpoint already does it, including walking back from blocks that cannot be marked.
- **A third marker at the end of the previous step's prompt**, to survive the 20-block lookback. A proxied agent loop adds two or three positions per step, far below 20; the extra marker would only help a client that calls the model many times without the proxy (see Limitations).
- **LiteLLM's injection (`cache_control_injection_points`, or `litellm.enable_anthropic_prompt_caching`).** It does nearly the same placement, but global or hidden: the proxy could not tell from the request it built whether markers would be sent. The loader forces the global flag off, so the environment variable `LITELLM_ENABLE_ANTHROPIC_PROMPT_CACHING` cannot inject markers behind the proxy's back either.

**Config.** `cache_breakpoints` per upstream: `route` (the default for `provider = "anthropic"`), `always`, or `off`. Any other provider may only leave it out or set `off`: DeepSeek and Ollama cache automatically, and a marker there would be noise.

**The client's own markers win.** If the request already carries a `cache_control` anywhere (a message, a content block, a tool), the proxy adds nothing and sends the markers as they are. Adding to them could exceed the four-breakpoint limit, put a 5-minute entry after a 1-hour one (a 400), or move the point the client chose. The outcome records `cache_markers = "client"`.

**Shadow mode.** Shadow mode promises not to change the request. A marker changes billing and caching but not the answer; still, the conservative default is to keep the promise: with `route`, shadow, fail-open and client-fallback requests go out unmarked, exactly as the client sent them, and the session records that nothing is cached (`cached_tokens = 0`). `always` marks them too, for a user who wants shadow mode to cost what route mode would. *Rejected:* marking shadow requests by default, which silently changes a client's bill (a 1.25x write on every step it did not ask for).

**Lifetime.** The proxy requests the default 5-minute entry (no `ttl`), which is what the catalog prices (`cache_write_mult = 1.25`) and what `cache_ttl_s = 300` assumes; the lifetime is refreshed on every read, and so is the session's timer. While breakpoints are on, a `cache_ttl_s` above 300 is a configuration error: the proxy would believe in a cache that has expired. A lower value is allowed (conservative). *Rejected:* requesting the 1-hour lifetime, which writes at 2x; the catalog has that rate only in `meta`, and nothing prices it yet.

**What Anthropic receives** (tested in `tests/test_proxy_anthropic.py` through LiteLLM's real `anthropic` provider):

```
"system": [{"type": "text", "text": "<system prompt>", "cache_control": {"type": "ephemeral"}}],
"tools": [... unmarked ...],
"messages": [... unmarked, step n's messages a prefix of step n+1's ...],
"cache_control": {"type": "ephemeral"},
"output_config": {"effort": "<decision effort>"}, "thinking": {"type": "adaptive", ...}
```

## Streaming

A streaming request gets **one attempt with a logged outcome and no check cascade**. The cascade needs the whole response, and unbuffered streaming forbids holding it (the conflict noted in `docs/INTEGRATION-NOTES.md`, issue 12 item 3).

The proxy:

- **fetches the first chunk before sending anything**, so an HTTP error or a refused connection on one step still moves on to the next cascade step (no byte has gone out yet);
- **always asks the upstream for usage** (`stream_options.include_usage`), so the outcome has token counts. It removes the usage-only chunk unless the client asked for it;
- **assembles what the checks need while chunks pass:** finish reason, tool-call arguments, and the text only when a schema check applies. It runs the checks after the stream for the signals only;
- **ends the stream with an SSE error event** if the upstream fails mid-stream (`stream_error`);
- **records `client_disconnected`** if the client goes away. Tokens are then whatever usage arrived, often 0.

The tests check both behaviours. `test_streaming_has_no_check_cascade` sends an empty stream: one call, and the outcome is labelled a failure. `test_cascade_on_empty_output` sends the same reply non-streaming: two calls.

## Ollama smoke test

Opt-in; CI skips it. It runs two steps through the proxy against a real local Ollama and checks that both decisions joined their outcomes with non-zero tokens, and that the second step saw a cached prefix.

1. Install Ollama, then start it with the D2 context cap and one model loaded at a time:
   ```
   OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_MAX_LOADED_MODELS=1 ollama serve
   ```
2. Pull a model: `ollama pull qwen3-coder:30b`. Any local catalog model works.
3. Run:
   ```
   STEPROUTE_TEST_OLLAMA=1 STEPROUTE_TEST_OLLAMA_MODEL=qwen3-coder:30b \
     uv run --extra proxy pytest -q tests/test_proxy.py -k ollama
   ```
   It is skipped, not failed, without `STEPROUTE_TEST_OLLAMA=1` or when nothing listens on 127.0.0.1:11434.
4. Or by hand: write the example config above with only the `ollama` upstream, and a `service.toml` with a `[store] path`. Run `steproute proxy --config proxy.toml`, send two requests with the same `X-Steproute-Session`, then run `steproute traces stats <store>` and check `joined: 2`.

## Limitations

- **The prompt estimate is an estimate** until the first call of a session calibrates it. Image tokens are a flat 1000 per image.
- **One cache owner per session.** A provider may still hold an older model's cache (Anthropic keeps it per model for 5 minutes), so switching back can be cheaper than the cost model thinks.
- **Client cache markers.** When the client places its own `cache_control` markers, the proxy adds none and still assumes the whole prompt is cached afterwards, which holds only if the client's last marker is at the end. A client marker asking for the 1-hour lifetime is priced at the catalog's 5-minute write rate.
- **The 20-block lookback.** The moving breakpoint finds the previous step's entry only within 20 content positions. A single step that appends more than 20 (a long sequential tool loop that never passes through the proxy) misses, and the step pays a full write the session state did not predict.
- **The fallback session id** merges two sessions that open with identical messages, and changes when the proxy restarts (random salt) unless `salt_env` is set. Send `X-Steproute-Session`.
- **Concurrent requests in one session** each get a step index, but their cache estimates read the state as it was when they started.
- **Nothing is logged** for a request without a decision (source down, no fallback), and `/report` only merges into outcomes this process emitted.
- **Rewriting is per request.** The proxy does not use Anthropic's per-message effort beta, so an effort change on Claude is treated as a cache miss (`effort_change_keeps_cache = false`).
- **Only the first choice** (`n = 1`) is checked.
- **No live provider has been called.** Every behaviour above is tested against the loopback mocks. The Anthropic path runs through LiteLLM's real `anthropic` provider against an Anthropic Messages-format mock (`tests/test_proxy_anthropic.py`), so the request shape and the usage mapping are exercised, but Anthropic's actual caching behaviour rests on its documentation. The DeepSeek path rests on LiteLLM's code at the pinned tag (`docs/INTEGRATION-NOTES.md` sections 8 and 9) until a separately approved run.
