# Local service

`steproute serve` runs one local process that makes routing decisions and records outcomes, so
collectors written in any language (the TypeScript mods, the LiteLLM hook, shell scripts) share
one policy, one catalog and one trace store. Implementation: `steproute/core/service.py`.
Standard library only (`http.server`), HTTP/1.1 with keep-alive, JSON bodies.

It sits in the request path of every LLM call, so three properties are non-negotiable:

1. **Fast.** Typical `/decide` round trip with the rule baseline is well under a millisecond of
   service time (see [Latency](#latency)).
2. **Fail-open.** If the policy raises or is too slow the service still answers, with a
   configured fallback action, and logs why.
3. **Privacy first.** A sensitive state never gets a cloud model from any path, including
   fail-open.

```
steproute serve [--config PATH] [--host HOST] [--port PORT] [--store DIR]
```

`--host`, `--port` and `--store` override the config file. `--port 0` picks a free port; the
first line on stdout is then `steproute serve: listening on http://127.0.0.1:<port>`. Logs go
to stderr through `logging`. SIGINT and SIGTERM stop the server and exit with status 0. Exit
status 2 means the configuration was refused (the reason is on stderr).

## API

All bodies are JSON (`Content-Type: application/json` is required on POST). Records follow the
checked-in JSON Schema (`steproute/core/trace.schema.json`, `docs/SCHEMA.md`). Request bodies
may contain keys the schema does not know; they are ignored. Error bodies are always
`{"error": "<code>", "message": "<text>"}`; messages never repeat request content.

### `POST /decide`

Body: a `StepState`. Response: the `Decision`, exactly as it was written to the trace store.
Because the store applies the privacy config, `state.task_text` comes back `null` (with
`features.task_text_len` set) unless the service is configured with `capture_text = true`.
`state.sensitive` is the value after sensitive detection, so it can be `true` even if you sent
`false`.

```
POST /decide
{"harness": "claude-code", "session_id": "s1", "step_idx": 3, "kind": "main",
 "prompt_tokens": 20000, "current_model": "claude-sonnet-5-5", "tools": ["Edit", "Bash"]}

200 OK
{"decision_id": "6f1c...", "ts": 1794000000.1,
 "state": {"harness": "claude-code", "session_id": "s1", "step_idx": 3, "...": "..."},
 "candidates": [{"model": "claude-sonnet-5-5", "effort": "medium", "escalate_to": {"...": "..."}}],
 "action": {"model": "claude-sonnet-5-5", "effort": "medium",
            "escalate_to": {"model": "claude-sonnet-5-5", "effort": "high", "escalate_to": null}},
 "policy": "rules", "propensity": 1.0,
 "predicted_success": null, "predicted_cost": null, "candidate_set": null}
```

Apply `action.model` and `action.effort`; `action.escalate_to` is the next step to try if the
check you configured on the response fails. Keep `decision_id` and send it back in `/report`.
A decision whose `policy` starts with `fail_open:` was not made by the policy (see below).

### `POST /report`

Body: an `Outcome`. Response: the `Outcome` as stored.

* `success` is set from `signals` (`docs/LABELS.md`) when you send `null` or omit it. A
  `success` you send is kept. With no usable signals it stays `null`.
* `cost_usd` is filled from the catalog prices (`docs/COST-MODEL.md`) when it is missing, the
  outcome is not `escalated`, and `final_model` names a catalog model. Otherwise it stays
  `null`: the service never guesses (an escalated outcome may span models; price each attempt
  yourself and send `cost_usd`).
* An outcome for a `decision_id` the store has never seen is stored anyway and answered `200`.
* The policy is then told (`policy.report`); a failure there is logged and does not change the
  answer, since the outcome is already stored.

```
POST /report
{"decision_id": "6f1c...", "ts": 1794000003.2, "success": null,
 "signals": {"tests_passed": true}, "final_model": "claude-sonnet-5-5",
 "input_tokens": 20000, "output_tokens": 800}

200 OK
{"decision_id": "6f1c...", "ts": 1794000003.2, "success": true, "...": "...", "cost_usd": 0.072}
```

### `GET /models`

`{"models": [ModelProfile, ...]}` in tier order, weakest first. Each profile is the full
`ModelProfile` record, so `holdout` (excluded from normal routing) and `local` are visible.

### `GET /healthz`

No authentication. `{"status": "ok", "version": "<package>", "schema_version": "<trace schema>",
"policy": "<policy name>", "fail_open_count": n, "abandoned_policy_calls": n,
"store_errors": n}`. The three counters are for monitoring and are since process start.
`fail_open_count` and `store_errors` only grow; `abandoned_policy_calls` is a gauge.

## Status codes and what a client should do

| Code | `error` | Meaning | Client action |
|---|---|---|---|
| 200 | | Success. | Use the body. |
| 400 | `bad_json`, `bad_request` | Body is not JSON, not an object, or fails the record schema (the message names the field, never the value). | Bug in the client. Do not retry; route with your own default. |
| 401 | `unauthorized` | Token missing or wrong (`WWW-Authenticate: Bearer`). | Configuration error. Do not retry; route with your own default. |
| 403 | `forbidden_host` | No token configured and the `Host` header is not a loopback name (anti DNS-rebinding). | Connect to `127.0.0.1` or `localhost`. |
| 404 | `not_found` | Unknown path. | Bug in the client. |
| 405 | `method_not_allowed` | Wrong method; `Allow` header names the right one. | Bug in the client. |
| 411 | `length_required` | No `Content-Length`, or chunked upload (unsupported). | Send `Content-Length`. |
| 413 | `payload_too_large` | Body over `max_body_bytes` (default 1 MiB). The connection is closed. | Do not send the whole prompt; send `task_text` only if needed, truncated. |
| 415 | `unsupported_media_type` | POST without `Content-Type: application/json`. | Set the header. |
| 500 | `internal_error` | Unexpected service bug. The connection is closed. | Treat like an unreachable service. |
| 503 | `no_action_allowed` | `/decide` only: the policy failed and neither the default action nor the local fallback is allowed for this state. Body also has `"sensitive": bool`. | Do not call a cloud model if `sensitive` is true. Run the step locally yourself or refuse it. If false, use your own default. |
| 503 | `store_unavailable` | `/report` only: the outcome could not be written. | Retry later, or keep it and re-send. |

A client must also handle "service unreachable" (refused connection, timeout, closed
keep-alive connection): the service is a helper, so a collector should keep its own default
route and a short client timeout (suggested 100 ms for `/decide`, longer for `/report`). The
service closes idle connections after 30 seconds; reconnect and retry once on a connection
reset. On an unreachable service a collector that knows a step is sensitive must still not
send it to a cloud model.

Responses that leave an unread request body behind (401, 403, 404, 405 on a POST, 411, 413,
415, 500) carry `Connection: close`; open a new connection.

## Configuration

One TOML file, `steproute serve --config service.toml`. Every key is optional; **unknown
sections and keys are an error**. Relative paths resolve against the config file's directory.

```toml
[server]
host = "127.0.0.1"        # default; non-loopback needs [auth]
port = 8787               # 0 = pick a free port
max_body_bytes = 1048576  # larger bodies get 413

[auth]
token_env = "STEPROUTE_TOKEN"   # NAME of the environment variable holding the token

[store]
path = "traces"           # default: ~/.steproute/traces

[catalog]
path = "my-catalog.toml"  # default: the built-in catalog

[policy]
kind = "rules"            # "rules" (default) or "fixed"
rules_config = "rules.toml"                          # kind = "rules": default: built-in rules
action = { model = "claude-sonnet-5-5", effort = "medium" }   # kind = "fixed" (required then)

[failopen]
budget_ms = 50            # time the policy may take
max_abandoned = 8         # see "Abandoned calls"
default = { model = "claude-sonnet-5-5", effort = "medium" }
local_fallback = { model = "qwen3-coder:30b", effort = "medium" }

[privacy]
capture_text = false      # true: task_text is stored (and echoed back)
hash_text = false         # true: store a salted hash of task_text; needs salt_env
salt_env = "STEPROUTE_SALT"     # NAME of the environment variable holding the salt
```

* `effort` defaults to `medium` in an action table.
* `token_env` and `salt_env` hold variable **names**, never the secret. A value that is not a
  valid variable name is refused (and not echoed). If the named variable is unset or empty the
  service refuses to start.
* The same `PrivacyConfig` (capture, hash, salt) goes to the policy and to the trace store.
* Command-line flags `--host`, `--port`, `--store` override `[server]` and `[store]`.

### Startup checks (the service refuses to start)

* `host` is not loopback and no token is configured.
* The catalog, rules file or any action does not load or validate: an unknown model, an effort
  the model does not list, a holdout model, a `local_fallback` that is not a local model.
* `policy.kind = "fixed"` without `[policy] action`.
* `hash_text = true` without a salt.
* The default fail-open action is a cloud model, no `local_fallback` is set, and the catalog
  has non-holdout local models. A sensitive state would then have nowhere to go when the policy
  fails; set `local_fallback`. (If the catalog has no local models at all this is allowed, and a
  sensitive state gets a 503 instead.)

Zero configuration is valid: the default fail-open action is then the strongest non-holdout
local model (so fail-open can never send anything to a cloud provider, but may be weaker than
you want). Set `[failopen] default` to what your users expect.

## Fail-open

`policy.decide` runs in a worker thread and the request waits at most `budget_ms`. The fallback
is used when:

| `Decision.policy` | cause |
|---|---|
| `fail_open:timeout` | the policy did not return within the budget |
| `fail_open:error` | it raised any exception |
| `fail_open:no_feasible` | it raised `NoFeasibleAction` (the hard constraints leave it nothing) |
| `fail_open:bad_result` | it returned something other than a `Decision` |
| `fail_open:policy_violation` | it chose a non-local model for a sensitive state |
| `fail_open:stuck` | `max_abandoned` earlier calls are still running; the policy was not called |

The fallback chooses the first of `[failopen] default`, then `local_fallback` that is
**feasible for this state** under the same hard constraints every policy obeys
(`docs/ACTIONS.md`: sensitive means local only, context window, tools, holdout). The state is
first passed through sensitive detection (`docs/PRIVACY.md`), so a secret in `task_text` or a
sensitive path keeps the step local even when the collector did not set the flag and even
when the default is a cloud model. If neither action is allowed the answer is
`503 no_action_allowed` with `"sensitive"` set; no decision is logged for it.

The fail-open decision has `propensity` 1.0 and the policy name above, so it is logged like any
other decision and can be filtered out of training and off-policy evaluation by prefix
`fail_open:`. The failure is logged through `logging` (logger `steproute.service`, level
WARNING) with the policy name and exception type only: exception messages and prompt text are
never logged.

### Abandoned calls

A Python thread cannot be killed. When the budget runs out the request stops waiting and the
policy call keeps running in its daemon thread, *abandoned*. Its result is discarded: only the
request thread logs, so a late result is never written as "the decision that was returned".
The number of abandoned calls still running is limited by `max_abandoned` (default 8); at the
limit new requests skip the policy and fail open immediately (`fail_open:stuck`) until some
finish. A policy that holds state (a learner) may still update it from an abandoned call.

`/report` calls `policy.report` on the request thread without a budget; make `report` quick.

## Security

* Binds `127.0.0.1` by default. A non-loopback `host` is refused without a token.
* With `[auth] token_env`, every endpoint except `GET /healthz` needs
  `Authorization: Bearer <token>`, compared in constant time. Unauthenticated requests learn
  nothing, not even which paths exist.
* Without a token, any local process (and any web page, via the browser) could reach the
  service. Two cheap defences are on: POST needs `Content-Type: application/json` (a browser
  cannot send that cross-origin without a preflight, which is never answered), and the `Host`
  header must be a loopback name. Use a token on shared machines.
* Request bodies are capped (`max_body_bytes`) and read only after the cap check.
  Connections idle or stalled for 30 s are closed.
* Request bodies and prompt text are never logged. Validation errors name fields, not values.
* The token travels in clear text: do not expose the service beyond loopback without a TLS
  terminator in front of it.
* Trace files are written with mode 0644 by `TraceStore`; the store directory is yours to
  protect.

## Latency

`scripts/bench_service.py` starts the service as a separate process with the rule baseline and
the default catalog, and times `/decide` round trips over one kept-alive connection (3000
requests after 200 warm-up), with and without a 20,000-character `task_text`. Run it with
`uv run python scripts/bench_service.py`. On an Apple M-series laptop with Python 3.13 it
measured:

| request | p50 | p95 | p99 |
|---|---|---|---|
| no `task_text` (typical) | 0.29 ms | 0.35 ms | 0.47 ms |
| 20,000-char `task_text`, 5 paths | 3.2 ms | 3.5 ms | 3.8 ms |

The cost with text is the sensitive-detection scan (secret patterns over the text), which runs
in the policy and again when the decision is redacted for the store. Latency includes a trace
store append per request. Numbers depend on the machine and are not asserted in tests.
