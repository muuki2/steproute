# Privacy and redaction

Traces can contain code and prompts. This page states what steproute stores, what it
withholds, and where the protections stop. Implementation: `steproute/core/privacy.py`
(rules, detection, scrub), `steproute/core/export.py` (export), and the write path in
`TraceStore.log_decision`.

## What is stored by default

`TraceStore` takes an optional `PrivacyConfig`; the default has `capture_text=False`. The
policy is applied inside `log_decision`, so a collector cannot forget to call it.

With `capture_text` off, a decision is written with `StepState.task_text = null`. If the text
was present, two things are recorded in `StepState.features`:

| key | value |
|---|---|
| `task_text_len` | `len(task_text)`, always when text was present |
| `task_text_hash` | first 32 hex characters of HMAC-SHA256(salt, text); only with `hash_text=True` (which requires a non-empty `salt`) |

The caller's `Decision` and `StepState` are never mutated; a redacted copy is written.
Everything else in the record is stored as given: session and task ids, tool names, model ids,
timestamps, and the free-form `features` and `signals` dicts. Collectors must not put prompt
text or code in those dicts; the store does not inspect them.

With `capture_text=True` the text is stored as given. This is an explicit opt-in per store.

## Sensitive detection

`detect_sensitive(text, paths, config)` returns whether a step is sensitive and the names of
the rules that fired. It never returns matched text. Three sources:

- path globs (case-insensitive, `/` and `\` both accepted);
- regexes over the text;
- an optional user callback `callback(text, paths) -> bool`. A callback that raises fails
  closed: the step is treated as sensitive (rule `callback_error`).

`mark_sensitive(state, paths, config)` returns a copy of the state with `sensitive=True` when
detection fires. Detection and the callback can only set the flag, never clear it.

**Detection runs before routing.** Every policy (`ScoredPolicy`, `FixedPolicy`, and the rule
baseline and later policies) calls `mark_before_routing(state, config)` on the incoming state
before the hard filter and scoring, and logs the marked copy in `Decision.state`. A sensitive
state only gets local models (`docs/ACTIONS.md`), so a secret in `task_text` keeps the step on
the machine even when the collector did not set the flag. That matters most for collectors in
other languages, which reach the policy through the HTTP service and cannot run detection
themselves. The policy's `PrivacyConfig` (argument `privacy`, default `PrivacyConfig()`) sets
the patterns, globs and callback.

**File paths.** A step's paths travel in `StepState.features["paths"]`
(`steproute.core.FEATURE_PATHS`), a list of strings. `mark_before_routing` and `redact_state`
pass them to detection; a value of any other type is ignored, not raised on. `--scrub` exports
drop every list value in `features`, so the paths never reach a published dataset.

**The store's check remains as a second line.** `TraceStore.log_decision` still runs
`redact_state`, which applies the same text and path rules before dropping the text. A
decision made by a policy is already marked, so this changes nothing with the same config. It
still catches records that did not come from a policy, and states routed with a laxer config
than the store's.

Default glob rules: `dotenv_files`, `dotenv_variants`, `dotenv_suffix`, `pem_files`,
`key_files`, `keystore_p12`, `keystore_pfx`, `ssh_dir`, `ssh_key_rsa`, `ssh_key_ed25519`, `ssh_key_ecdsa`, `ssh_key_dsa`, `aws_dir`,
`gcloud_dir`, `azure_dir`, `kube_config`, `netrc`, `secrets_dir`.

Default secret patterns: `aws_access_key_id`, `github_token`, `api_key_sk`, `slack_token`,
`google_api_key`, `private_key_block`, `jwt`, `bearer_token`, `secret_assignment`
(`password=`, `api_key:`, `AWS_SECRET_ACCESS_KEY=` and similar).

**Detection is best effort.** It is pattern matching: it will miss secrets in formats it has
no pattern for, secrets that are split or encoded, and anything sensitive that is not a
credential (proprietary code, personal data). It can also flag harmless text. Add your own
rules or a callback for what matters to you.

## Export

```
steproute traces export <store> --out <file> [--scrub] [--salt VALUE]
```

Decision D5: only benchmark replay traces are published. The export includes a decision only
when `state.source == "replay"`, and an outcome only when its decision was included. Orphan
outcomes (no decision in the store) are withheld because their source is unknown. A decision
id that appears with any non-replay source is withheld entirely. There is no flag, and the
core function `export_replay` has no parameter, that can include personal records. The
command prints how many records were exported and withheld. Output is JSONL in the store's
own line format.

### `--scrub`

Scrubbing is applied to each exported record:

- `task_text` is dropped.
- `session_id` and `task_id` are replaced by HMAC-SHA256(salt, id) truncated to 16 hex
  characters. The mapping is consistent within one export, so records still group.
- `decision_id` (on decisions and outcomes) is replaced by a keyed hash under its own prefix
  (32 hex characters), consistently, so outcomes still join to their decisions. It is hashed
  because a collector may have built it from the session id.
- Every `ts` is rounded down to the hour (configurable in `ScrubOptions.bucket_seconds`).
- In `features` and `signals`, only booleans, numbers and null are kept. Every string, list
  and nested object is dropped, because those can carry arbitrary text.
- The secret patterns are run over every remaining string (`harness`, `kind`, `policy`,
  `current_model`, tool names, model ids including those in `failed_attempts`, dict keys) and each match is replaced with
  `[REDACTED]`.

`PrivacyConfig.salt` and `ScrubOptions.salt` are excluded from `repr`, so logging a config does not leak them.

The salt comes from `--salt`, else `STEPROUTE_SCRUB_SALT`, else a random one generated for
that export. It is never printed or written to the output. Use the same salt across exports
only if you want ids to be linkable between them; keep it private, since anyone holding it can
test guesses of an id.

### Limits

Scrub removes the fields above. It is not a proof of anonymity. Strings that are kept
(`harness`, `kind`, `policy`, tool names, model ids) are only checked against the secret
patterns, so free text there that matches no pattern survives. Numeric values, step counts,
token counts and hour-level timing remain and can in principle be used to re-identify a
session. Check the output before you publish it.
