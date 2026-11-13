# MCP server

`steproute mcp` runs an MCP server over stdio. Any MCP client (Claude Code, dsh, others) can ask it
which model and effort to use for a task, and report how the task went. The server **only
advises**: an MCP server cannot change the host's model, so the calling model has to apply the
recommendation itself (for example by choosing the model when it spawns a subagent). A caller can
ignore the advice, and nothing here enforces it.

The code is `src/steproute/collectors/mcp_server.py`. It embeds a `Service` built with
`build_service`, so decisions, privacy rules, labelling, costing and trace storage are exactly
those of `steproute serve` (`docs/SERVICE.md`). It does not reimplement routing. It does not talk
to the HTTP service; pointing at a running `steproute serve` is not supported.

## Install and register

```bash
pip install 'steproute[mcp]'     # or: uv sync --extra mcp
steproute mcp --help
```

Without the extra, `steproute --version` still works and `steproute mcp` prints how to install it.

A project-level `.mcp.json` (Claude Code and other clients read the same shape):

```json
{
  "mcpServers": {
    "steproute": {
      "command": "steproute",
      "args": ["mcp", "--config", "steproute.toml"]
    }
  }
}
```

With `uv` and no install: `"command": "uv", "args": ["run", "--extra", "mcp", "steproute", "mcp"]`.

Options: `--config` (the same file as `steproute serve`; the `[server]` and `[auth]` sections are
ignored), `--store DIR`, `--harness NAME` (default `mcp`), `--timeout MINUTES` (default 30),
`--small-tokens/--medium-tokens/--large-tokens N`. The server writes logs to stderr only; stdout
carries the protocol.

In Claude Code, subagents inherit the parent's MCP tools, so a subagent can itself call
`route_task` and `report_outcome`. A subagent definition with a `tools` allowlist must list
`mcp__steproute__*` explicitly or it loses them (`docs/INTEGRATION-NOTES.md`, issue 13).

## What a call becomes

`route_task` builds a `StepState`: `harness` from `--harness`, one `session_id` per server process
(`mcp-<8 hex>`), `step_idx` counting `route_task` calls, `kind` and `tools` passed through,
`source = personal`, `prompt_tokens` from `est_size`, and `description` as `task_text`. The store
drops `task_text` by default (`docs/PRIVACY.md`); sensitive detection still runs on it first, so a
description containing a secret makes the task sensitive and only local models are recommended.

`est_size` is the estimated prompt size in tokens, an integer. The words `small`, `medium` and
`large` are accepted and mapped to 2000, 8000 and 32000 tokens (configurable with the flags
above). Those numbers are rough placeholders, not measurements. Omitted means `medium`.

## Tools

### `route_task(description, kind="main", tools=None, est_size=None)`

When to call: before spawning a subagent or starting a new task. Example result for a read-only
subagent:

```json
{
  "recommendation": true,
  "decision_id": "6dcfa2420af048a2bfac2c93bcac927b",
  "model": "claude-haiku-5-5",
  "effort": "low",
  "local": false,
  "cascade": [
    {"model": "claude-haiku-5-5", "effort": "medium"},
    {"model": "claude-sonnet-5-5", "effort": "low"}
  ],
  "reason": "Read-only subagent work, so a cheaper tier (4).",
  "sensitive": false,
  "note": "Advice only: apply it yourself, then call report_outcome with the decision_id when the task is finished."
}
```

`cascade` is the ordered list of fallbacks to try if the first choice fails. `reason` is built from
what the policy recorded (which rule fired, whether it fell back to another tier, whether the
router itself failed open). `sensitive: true` means local models only.

When the service has no decision to give, the result is `{"recommendation": false, "decision_id":
null, "model": null, ..., "reason": "..."}` telling the caller to use its own default model, or to
keep the work local when `sensitive` is true. Nothing is logged then. A bad `est_size` is a normal
tool error.

### `report_outcome(decision_id, success=None, signals=None)`

Call once the task is finished, with honest results including failures. `success` is `true`,
`false` or `null`. `signals` uses the vocabulary of `docs/LABELS.md` (`tests_passed`,
`tool_error`, `user_rejected`, `judge_score`, ...); unknown keys are kept. An explicit `success`
is never overridden by labelling; with `null`, the signals decide (or stay `null`).

```json
{"recorded": true, "found": true, "success": true}
```

An unknown `decision_id` (never issued by this server process) returns
`{"recorded": false, "found": false, "message": "..."}` and stores nothing.

### `list_models()`

The non-holdout models: `{"models": [{"id", "provider", "tier", "efforts", "local"}, ...]}`.

## Unreported decisions

Every `route_task` that returns a recommendation logs a decision and is tracked as pending. A
decision still unreported after the timeout (default 30 minutes) is closed by logging an outcome
with `success = null` and `signals = {"timed_out": true}`. The check runs on every tool call, on
a background timer, and at shutdown (stdin closing), which closes all remaining pending decisions
the same way. A late `report_outcome` for a closed decision is still stored and replaces the
timeout outcome (the store keeps the last outcome per decision). Decisions from an earlier server
process are not known to a new one, so reporting them returns `found: false`.

## Tool descriptions

The calling model reads the descriptions to decide when to call, so they are part of the product
(`ROUTE_TASK_DESCRIPTION` and the others in the module). Each says what the tool does, that it only
advises, and when to call it; `report_outcome` says to always call it, including for failures.

## Tests, fixture and the inspector

`tests/fixtures/mcp_session.json` is a recorded stdio session (initialize, tools/list,
`list_models`, `route_task`, `report_outcome`) captured with the SDK's own client by
`scripts/record_mcp_session.py`; `tests/test_mcp_server.py` replays the requests against a fresh
server process and compares the responses with decision ids normalised. Re-record with
`uv run --extra mcp python scripts/record_mcp_session.py`.

To try the server with the official inspector (needs Node):

```bash
npx @modelcontextprotocol/inspector --cli "$(which python)" -c \
  "import sys; from steproute.cli import main; sys.exit(main(sys.argv[1:]))" mcp \
  --method tools/list
# interactive UI: npx @modelcontextprotocol/inspector steproute mcp
```

The `--cli` mode starts a new server per call, so a `decision_id` cannot be carried between
calls there; use the interactive UI for a route-then-report round trip.

## SDK entry points used

`mcp.server.MCPServer` (registered tools with `@server.tool(name=, description=)`, a `lifespan`
for the background sweeper, `server.run("stdio")`), and for tests and the recorder
`mcp.Client`, `mcp.ClientSession`, `mcp.StdioServerParameters` and `mcp.client.stdio.stdio_client`.

## Limits

- Advice only; the caller may ignore it, and a model may forget to call the tools.
- Size words are placeholders; the rule policy ignores the task itself (`docs/RULES.md`).
- The server does not see what the caller actually used, so outcomes carry no token counts or
  cost. Issue 14's hooks can add those.
