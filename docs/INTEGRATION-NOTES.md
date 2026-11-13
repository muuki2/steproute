# Integration notes

*Started by issue 10 (spike), checked 13 Nov 2026. Later issues append their VERIFY findings as new sections at the end, in the same format.*

## Purpose

`docs/ISSUES.md` planned the collectors (issues 12 to 15) and the external baselines (issue 22) from secondhand notes. This file records what upstream docs and source actually say, with links, so implementers do not build on unverified names. It also records the provider prices, cache rules and model ids that the catalog (issue 4) and cost model (issue 6) depend on.

Rules for this file:

- Every claim has a source someone can open: a docs URL, or a file path at a named commit or tag.
- A section leads with the answer, then gives evidence, then a confidence note saying what the confidence rests on.
- "Could not determine" is a valid answer. It must list what was tried.
- Source was read, never built, installed or run. Line numbers are at the pinned commit given in each section. The exception is section 14 (issue 56), which installed and ran the SWE-bench harness, the task images and Ollama on the real machine, and says so.

## Pinned versions

| Component | Pin | Commit | Date |
|---|---|---|---|
| DeepSeek Harness (dsh) | `0.2.2-rc.3` (tag `dsh-v0.2.2-rc.3`, npm `@deepseek-ai/dsh@0.2.2-rc.3`) | `38a6a5c965aa99ab525161efb7cc5cf1fe62fced` | 2026-11-13 |
| LiteLLM | `1.109.1` (tag `v1.109.1`, PyPI) | `6c4ead4fc064d14e36b0c69d6795acc130c866bf` | 2026-11-12 |
| LLMRouter | `ulab-uiuc/LLMRouter` HEAD; PyPI `llmrouter-lib` 0.4.0 | `c65ee0122bc9e84f71d571fcdec4a948481b6e88` | 2026-10-08 |
| GraphRouter (standalone) | `ulab-uiuc/GraphRouter` HEAD | `0edfb8758d45acd93c8029354867ffb49bb8a97f` | 2025-12-30 |
| ParetoBandit | `ParetoBandit/ParetoBandit` HEAD; PyPI `paretobandit` 0.1.1 | `72ccbefe5cb858c95e00368e2b68de3c2b28a2ee` | 2026-04-13 |
| OpenSquilla | `TokenRhythm/opensquilla` v0.5.8 | `fdc3898a31432b48a1731bfe663a2b6f44223e88` | 2026-11-10 |
| RouteLLM | `lm-sys/RouteLLM` HEAD; PyPI `routellm` 0.2.0 | `0b64fdafe049e596a3f5657c219329f24af24198` | 2024-08-10 |
| Ollama (source read for cache behaviour) | `v0.40.16` | `d63b702609c4300a2ca07be2bdb0adb9834725e5` | 2026-11-07 |
| mini-SWE-agent (source read; not a dependency) | `2.4.6` (PyPI wheel, sha256 `a35463c5…cd54`) | n/a (wheel) | 2026-07-23 |
| SWE-bench harness (installed and run, issue 56) | `swebench==5.0.2` (PyPI), every dependency pinned in `scripts/swebench-requirements.txt`, project-local env by `scripts/install_swebench.sh` | n/a (PyPI) | 2026-08-18 |
| Claude Code | docs only; version gates quoted per feature (mods need v2.1.287+) | n/a | docs read 2026-11-13 |

## Summary

| # | Question | Short answer | Confidence |
|---|---|---|---|
| 1 | Claude Code hooks, subagent model, MCP inheritance | 33 hook events. Only `SessionStart` carries a `model`, and no hook carries per-request token usage. A subagent's model is set by the per-call `model` alias, then frontmatter, then `CLAUDE_CODE_SUBAGENT_MODEL`. A mod's `agent.spawn` hook can choose it. Subagents **do** inherit the parent's MCP tools, so the community README is wrong. | High (mods, inheritance); medium (PreToolUse rewrite of the Agent `model`) |
| 2 | Claude Code per-request usage incl. cache | OpenTelemetry `claude_code.api_request` events and `claude_code.token.usage` metrics carry cache read and creation tokens per request, with `agent_id` since v2.1.310. A mod's `turn.step` result carries per-request `usage`. Hooks carry only final-request usage on the Agent tool's `tool_response`. Transcript format is documented as internal. | High |
| 3 | dsh LLM seam, intercept, session log, fork, version | `ctx.llm` exists. The `llm/stream` waterfall wraps every call, but loop requests arrive frozen. Per-step model and effort are set with `agent.model.select()` before the step. The `intercept` patch action does not exist (the patch action is `insert`). The log is zstd-compressed JSONL with a documented schema. `SessionStore.fork()` exists. Pin `0.2.2-rc.3`. There is no stable release. | High (source); medium (the exact hook point for calling `select`) |
| 4 | LiteLLM hooks, model/effort rewrite, cache tokens | `async_pre_call_hook` (proxy only) runs before routing, and a rewritten `model` and `reasoning_effort` are honoured. Cascades cannot be built from callbacks and conflict with unbuffered streaming. Cache reads and writes surface for Anthropic and DeepSeek. LiteLLM's `ollama` provider drops Ollama's cached-token count. | High |
| 5 | LLMRouter custom-router contract | `MetaRouter(nn.Module, ABC)` with abstract `route_batch(batch)` / `route_single(batch)`. Plugins are found in `./custom_routers/`, `~/.llmrouter/plugins/` or `$LLMROUTER_PLUGINS`. Data is dense (query, model) rows. The benchmark pipeline hardcodes its router list. | High |
| 6 | ParetoBandit and OpenSquilla | ParetoBandit: Apache-2.0, `pip install paretobandit`, LinUCB with `route()` / `update()`, fits bandit-feedback data. OpenSquilla: Apache-2.0, repo moved to `TokenRhythm/opensquilla`, frozen LightGBM tier classifier with no training code, Python 3.12+, text-only inputs. | High |
| 7 | Prompt-cache behaviour per provider | Anthropic: writes cost 1.25x (5 min) or 2x (1 h), reads 0.1x (Haiku), 0.05x (Sonnet, Opus) or 0.025x (Fable). The minimum prefix is 512 tokens. The cache is per model and per workspace. A top-level effort or thinking change can invalidate it; the per-message effort beta `mid-conversation-output-config-2026-07-01` avoids that. DeepSeek: automatic caching, no write premium, hits at about 2 to 3 % of miss price, lifetime "hours to days". Ollama: prefix reuse while loaded, lost when the model is unloaded, reported as `prompt_eval_cached_count`. | High (Anthropic, Ollama); medium (DeepSeek scope and thinking toggle) |

---

## 1. Claude Code: hooks, subagent model, spawn control, MCP inheritance

**Answer.** Claude Code documents 33 hook events. None of them carries per-request token usage. Only `SessionStart` can carry `model`. The Agent tool's `PostToolUse` `tool_response` carries `resolvedModel`, `modelsUsed` and a `usage` object, but for the subagent's final request only.

A subagent's model resolves in this order: the per-call `model` parameter (an alias only), then frontmatter `model`, then `CLAUDE_CODE_SUBAGENT_MODEL`, then the main model. The one documented way to choose a spawn's model from a plugin is a **mod** (a plugin with a JS/TS hooks module): the `agent.spawn` hook returns `next({ ...e, model })`. A settings `PreToolUse` hook with `updatedInput` on the `Agent` tool should also work, but no example is documented. No MCP mechanism can change the model.

Subagents inherit the parent's MCP tools by default, so the community README's claim is false. Plugin-shipped subagents ignore the `mcpServers` and `hooks` frontmatter fields.

### Evidence (all checked 2026-11-13)

**Hook events.** Source: https://code.claude.com/docs/en/hooks

- The events are: SessionStart, Setup, UserPromptSubmit, UserPromptExpansion, PreToolUse, PermissionRequest, PermissionDenied, PostToolUse, PostToolUseFailure, PostToolBatch, Notification, MessageDisplay, SubagentStart, SubagentStop, TaskCreated, TaskCompleted, Stop, StopFailure, TeammateIdle, InstructionsLoaded, ConfigChange, CwdChanged, DirectoryAdded, FileChanged, WorktreeCreate, WorktreeRemove, PreCompact, PostCompact, PreModelSwitch, PostModelSwitch, Elicitation, ElicitationResult, SessionEnd.
- Every event gets the common fields `session_id`, `prompt_id`, `transcript_path`, `cwd`, `permission_mode` and `hook_event_name`. It also gets `agent_id` and `agent_type` when the hook fires inside a subagent, and an `effort` object `{level}` on events with a tool-use context.

**Model in hook payloads.** Same page.

- "Only `SessionStart` hooks can receive a `model` field, and Claude Code doesn't always include it."
- `PreModelSwitch` and `PostModelSwitch` get `from_model`, `to_model`, `prompt_cache_warm`, `estimated_cache_write_usd` and related fields. They cover session model switches only (`/model`, the picker, SDK `set_model`), not subagent spawns.
- "There is no `$CLAUDE_MODEL` environment variable."

**Tool payloads.** Same page.

- `PreToolUse` has `tool_name`, `tool_input` and `tool_use_id`, plus `mcp_server` for MCP tools.
- `PostToolUse` has `tool_input`, `tool_response` and `duration_ms`.
- `PostToolUseFailure` has `error`.
- `PostToolBatch` has `tool_calls[]`, each with its `tool_response`.

**Agent tool.** Same page.

- `tool_input` holds `prompt`, `description`, `subagent_type` and `model`. The table does not list `effort`, although the sub-agents page documents an `effort` parameter.
- The `PostToolUse` `tool_response` holds `status`, `agentId`, `content`, `resolvedModel`, `modelsUsed`, `totalTokens`, `totalDurationMs`, `totalToolUseCount` and `usage`.
- `usage` holds `input_tokens`, `output_tokens`, `cache_creation_input_tokens` and `cache_read_input_tokens`.
- The docs warn: "`totalTokens` and `usage` cover the final request only".
- For background subagents, "`tool_response` carries no usage fields". The response has only `status: "async_launched"`, `agentId`, `outputFile` and `resolvedModel`.

**SubagentStart and SubagentStop.** Same page.

- `SubagentStart` gets `agent_id` and `agent_type`. It "can't block subagent creation", but it can inject `additionalContext`.
- `SubagentStop` gets `agent_id`, `agent_type`, `agent_transcript_path` and `last_assistant_message`. It can block with `decision: "block"`.
- Neither event carries a model or token counts.

**Subagent model and effort.** Source: https://code.claude.com/docs/en/sub-agents

- Frontmatter `model` accepts "`sonnet`, `opus`, `haiku`, `fable`, a full model ID such as `claude-opus-5-5`, or `inherit`".
- The per-call parameter accepts "a model alias: `sonnet`, `opus`, `haiku`, or `fable`. Claude can't pass a full model ID or `inherit` on that call."
- Precedence: "1. The per-invocation `model` parameter 2. The subagent definition's `model` frontmatter … 3. The `CLAUDE_CODE_SUBAGENT_MODEL` environment variable … 4. The main conversation's model."
- Before v2.1.251 the environment variable came first.
- `CLAUDE_CODE_SUBAGENT_MODEL_FORCE=1` (v2.1.257+) ignores both the frontmatter and the per-call model.
- Values are checked against `availableModels`.
- "If an installed mod sets a model in its `agent.spawn` hook, Claude Code uses that model in place of the per-invocation parameter."
- Effort comes from frontmatter `effort` (`low` to `max`) or a per-call `effort` parameter (v2.1.292+). `CLAUDE_CODE_EFFORT_LEVEL` overrides both.
- "On another model, [the session effort level] doesn't apply"; see also https://code.claude.com/docs/en/model-config, "Effort level of subagents and teammates".

**Changing a spawn's model.**

- **Settings hook.** Source: https://code.claude.com/docs/en/hooks
  - `PreToolUse` matches `Agent`.
  - `updatedInput` "Replaces the entire input object, so include unchanged fields alongside modified ones. … Combine with `"allow"` to auto-approve … With no `permissionDecision`, auto mode can deny the changed call".
  - The list of tools that a hook may change in auto mode without denial (Bash, PowerShell, Monitor, Edit, Write, NotebookEdit) does not include `Agent`.
  - `resolvedModel` "can differ from the `model` value in `tool_input`", so it is the field to check afterwards.
- **Mods.** Sources: https://code.claude.com/docs/en/plugins/mods/reference, https://code.claude.com/docs/en/plugins/mods/events, https://code.claude.com/docs/en/plugins/mods/overview
  - `agent.spawn`: "`next({ ...e, model })` to choose the model of one that is starting, or `{ deny: reason }`". A changed `model` on a resumed run is ignored.
  - `turn.step` fires for every model request ("`e.agentId` is set for a subagent's request"). It can return `next({ ...e, model })` or `next({ ...e, effort })`.
  - Mods need Claude Code v2.1.287+ in the terminal. They run in the terminal, Desktop, VS Code, `claude -p` and the Agent SDK, but not in Desktop WSL sessions.
- **MCP.** https://code.claude.com/docs/en/mcp describes no way for an MCP server or tool to set the host's model.

**MCP inheritance.** Source: https://code.claude.com/docs/en/sub-agents

- "Subagents inherit the built-in tools and MCP tools available in the main conversation".
- "a background subagent keeps every MCP tool".
- `tools`: "Inherits every tool available to subagents if omitted". Only a `tools` allowlist or `disallowedTools` narrows this.
- String entries in `mcpServers` "share the parent session's connection".
- "plugin subagents don't support the `hooks`, `mcpServers`, or `permissionMode` frontmatter fields". Plugin authors ship hooks in `hooks/hooks.json` and MCP servers in `.mcp.json` instead.
- "Hooks from settings files, managed policy settings, and plugins all apply inside subagents."
- https://code.claude.com/docs/en/plugins/components lists the frontmatter fields plugin agents support: `name`, `description`, `model`, `effort`, `maxTurns`, `tools`, `disallowedTools`, `skills`, `memory`, `background`, `isolation`, …

**Plugin packaging.** Sources: https://code.claude.com/docs/en/plugins/manifest-reference, https://code.claude.com/docs/en/plugins/cli-reference

- The manifest is `.claude-plugin/plugin.json`. It is optional, and only `name` is required.
- A plugin can bundle `skills/`, `commands/`, `agents/`, `hooks/hooks.json`, `.mcp.json`, `output-styles/`, `monitors/`, `bin/` and a mod hooks module (the `modules` entry in `hooks/hooks.json`).
- Install with `claude plugin install <plugin> [-s user|project|local]` or `/plugin install name@marketplace`. Load for one session with `--plugin-dir <path>`.
- Older links: `/en/plugins` and `/en/plugins-reference` now serve the overview and manifest-reference pages.

### Confidence

- **High** for the event list, payload fields, precedence order, mods and MCP inheritance. These rest on verbatim quotes from the official pages above.
- **Medium** that a settings `PreToolUse` `updatedInput` rewrite of the Agent `model` takes effect. It is implied by the input schema and the precedence order, but no doc example shows it, and auto mode may deny it.
- The docs say nothing about an `effort` field in the Agent `tool_input` that hooks see.

## 2. Claude Code: per-request usage including cache tokens

**Answer.** Per-request usage, with cache fields, is available from two documented sources:

- **OpenTelemetry**, which is authoritative and attributes subagent requests.
- A **mod's `turn.step` hook**, whose result carries per-request `usage` and the model.

The other sources are weaker. Hooks see only the final request's usage, on the Agent `PostToolUse`. The status line, `/cost` and the transcript JSONL are either session-level, exclude subagents, or have an undocumented format.

### Evidence (checked 2026-11-13)

**OpenTelemetry.** Source: https://code.claude.com/docs/en/monitoring-usage

- Metric `claude_code.token.usage` has `type` ∈ `"input"`, `"output"`, `"cacheRead"`, `"cacheCreation"`. "The `"input"` type excludes tokens read from or written to the prompt cache."
- Its attributes include `model`, `query_source` (`main`, `subagent`, `auxiliary`), `effort`, `agent.name` and `plugin.name`.
- Event `claude_code.api_request` has `model`, `cost_usd`, `duration_ms`, `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_creation_tokens`, `cache_creation_5m_tokens`/`cache_creation_1h_tokens` (v2.1.300+), `request_id`, `query_source` and `effort`.
- The standard attributes `agent_id` and `parent_agent_id` need v2.1.310+.
- Event `claude_code.subagent_completed` has `agent_type`, `model`, `final_model`, `model_swapped` and `total_tokens` ("Not a sum across the run").
- User-defined agent names are redacted to `"custom"` unless `OTEL_LOG_TOOL_DETAILS=1` is set.
- The hooks page says Claude Code strips `OTEL_*` variables from hook subprocesses. A hook therefore cannot forward to the user's collector by inheriting its config.

**Mods.** Source: https://code.claude.com/docs/en/plugins/mods/events

- The `turn.step` result has "`result.usage` holds … `input_tokens`, `output_tokens`, `cache_read_input_tokens`, and `cache_creation_input_tokens`", plus `model`.
- `e.agentId` identifies subagent requests.
- `turn.complete` has the turn's totals.

**Transcripts.** Source: https://code.claude.com/docs/en/sessions

- Path: `~/.claude/projects/<project>/<session-id>.jsonl`. Subagent transcripts are in a nested `subagents/` folder; `SubagentStop` gives the path.
- "The entry format is internal to Claude Code and changes between versions."
- Per monitoring-usage, `request_id` is persisted as `requestId` on assistant entries.

**Status line, `/cost` and the SDK.**

- https://code.claude.com/docs/en/statusline: `context_window.current_usage` holds the main session's last-call cache fields. Its prompt-cache statistics exclude subagent requests.
- https://code.claude.com/docs/en/costs: the `/cost` prompt-cache line "covers the main conversation only, not subagents".
- https://code.claude.com/docs/en/agent-sdk/cost-tracking: `modelUsage`/`model_usage` covers the whole tree, while `usage` undercounts when subagents nest.

**Claude Code's own cache behaviour.** Source: https://code.claude.com/docs/en/prompt-caching

- "A subagent starts its own conversation with its own system prompt and tool set … Its first request doesn't read the parent's cache … The parent's cache is unaffected."
- "With an API key or a Claude subscription, changing effort keeps the cache on Fable 5.1 and later, Opus 5.5, Sonnet 5.5, and Haiku 5.5."

### Confidence

**High.** This rests on verbatim quotes from the official pages. The transcript format is deliberately undocumented, so do not parse it as a primary source.

## 3. DeepSeek Harness (dsh): LLM seam, interception, session log, fork API, version

**Answer.** The LLM seam is `ctx.llm`, an `LlmRuntime` from `@deepseek-ai/dsh-llm`. A plugin can wrap every model call through the `llm/stream` waterfall event. However, requests built by the agent loop arrive **deep-frozen**, so a listener can read them but not rewrite the model or effort.

The supported per-step control is `agent.model.select({ provider, model, reasoningEffort })`. It has to be called **before** the next step is prepared. The `agent/request` waterfall, which can change call controls, is deprecated, and it rejects changes to the provider, model or effort.

The session log is one append-only JSONL file per session, zstd-compressed by default, with a documented event catalog. Its `assistant/message` events carry usage, including `cacheReadTokens`. A session fork API exists on `ctx.sessions`.

Target `0.2.2-rc.3`. Every release so far is a prerelease, and the README promises breaking changes.

### Names from the maintainer's notes

| Name in notes | Finding |
|---|---|
| Docs at `https://deepseek.com/harness/en/` | **Wrong.** That page says "Page moved" and points to `https://deepseek.com/en/harness/`, a marketing page. Developer docs: https://deepseek-harness.github.io/deepseek-harness/ (README.md:9); plugin docs under `/en/develop/basic/`. |
| `docs/capability-seams.md` | **Confirmed.** Generated by `scripts/gen-doc-graphs.ts`. |
| Built on Cordis | **Confirmed.** Vendored as `@deepseek-ai/cordis` 4.1.0, from upstream `cordiverse/cordis` 4.0.0-rc.12 (`vendor/README.md:15`). README.md:7: "powered by Cordis". |
| Released Aug 2026 as developer preview | **Confirmed.** First npm publish of `@deepseek-ai/dsh` 0.0.1-rc.1 was on 2026-08-10. First GitHub release `dsh-v0.1.0-rc.7` was on 2026-08-17. README.md:11-13: "developer preview … THERE WILL BE COMPATIBILITY-BREAKING CHANGES." |
| LLM seam `ctx.llm` | **Confirmed.** |
| `intercept` patch action | **Wrong.** The patch action is `insert` (rows can also be overridden by `id`). `intercept` is a Cordis context method `ctx.intercept(name, config)` (`docs/cordis-api/context.md:71-88`) and a loader entry option `intercept?: Dict` (`vendor/loader/src/config/isolate.ts:7`). It attaches service configuration and does not wrap calls. |
| Session-log fork API | **Exists, under another name.** It is a session fork on `SessionStore`, not a log API (see below). |

### Evidence

All paths are in `deepseek-ai/deepseek-harness`. Line numbers are at HEAD `39163dd30f703a79079e2d2446545cff3c453738` (2026-11-13). A diff of the cited files against tag `dsh-v0.2.2-rc.3` (`38a6a5c9…`) shows only import and re-export changes; `model-selection.ts` line numbers are one lower at the tag.

**Seam.**

- `docs/capability-seams.md:180`: "`ctx.llm` | `seam` | llm | llm-deepseek, llm-pi-ai, llm-replay | agent-loop, compaction-basic".
- `packages/llm/llm/src/index.ts:58-76` declares:
  ```ts
  'llm/stream'(this: LlmRuntime, options: GenerateOptions, next: () => AsyncIterable<StreamChunk>): AsyncIterable<StreamChunk>
  ```
  Its doc comment reads: "Waterfall around every streaming model call (retry, replay, routing) … A LOOP-built request … arrives deep-frozen (mutation throws): its content is a pure function of the session log … listeners read it, never rewrite it."
- `prepareCall(config, signal?, configure?)` is at `index.ts:975`. Its `configure` callback can change `reasoningEffort` (`call-config.ts:44`), but the agent loop calls it without `configure` (`packages/core/agent-loop/src/agent.ts:650`).

**Per-step selection.** `packages/core/agent/src/runtime-types.ts:54-71`:
```ts
interface Model { current(): Readonly<ModelSelection> | undefined; select(selection: Readonly<ModelSelection>): void }
```
- The doc comment on `select` reads: "Replace the next step's complete selection."
- `ModelSelection = { provider: string; model: string; reasoningEffort?: ReasoningEffortId }` (`packages/core/agent/src/types.ts:79-86`).
- `select` appends a durable `model/selection` event (`agent/src/model-selection.ts:113-116`).
- The loop reads `this.model.current()` at the start of `preStep` (`agent-loop/src/agent.ts:294`).

**Hooks around a step.**

- `agent/pre-step` (`runtime-types.ts:351`) receives `modelSelection` as read-only.
- `agent/request` (`runtime-types.ts:374`) is marked `@deprecated`: "a replacement may adjust controls but must keep the provider, model, and reasoning effort selection … Use agent.model.select() before step preparation for model and reasoning choices." This is enforced by a throw at `agent-loop/src/agent.ts:641-646`.
- An async hook that runs between steps is `tools/post-execute` (`packages/core/tools/src/events.ts:41`).

**Effort vocabulary for DeepSeek.**

- The `dsh-llm-deepseek` efforts are `off | low | high | max`.
- `off` maps to `thinking: {type: 'disabled'}`. The others map to `thinking: {type: 'enabled'}` plus `output_config.effort` (`packages/llm/llm-deepseek/src/serialize.ts:147-156`, `model-info.ts:7-10`).
- An unsupported effort is rejected with `UNSUPPORTED_REASONING_EFFORT` (`llm/src/index.ts:943-955`).

**Model changes and the cache.** A model change makes the loop inject a "[model changed …]" user notice (`model-selection.ts:150-171`) and log a new `request/header`. Both change the prompt prefix.

**Session log.**

- **Location:** `~/.dsh/sessions/--<normalized-cwd>--/<encoded-id>/session.v4.jsonl.zstd`. The root is set at `packages/bundle/base/cordis.patch.yml:138-141`, and `DSH_HOME` defaults to `~/.dsh` (`packages/util/home-paths/src/index.ts:12,71`).
- **Compression and format version:** compression defaults to `'zstd'`; `compression: 'none'` gives plain `.jsonl` (`session-persistence-jsonl/README.md:48,57-73`). `SESSION_FORMAT_VERSION = 4` (`packages/core/session/src/types.ts:90`).
- **Envelope:** `{ type, seq, time /* epoch ms */, data, ignorable?: true }` (`types.ts:506-524`). Readers refuse unknown events that are not marked `ignorable`.
- **Event types:** 63, listed at `docs/persistence-catalog.md:25-87`. The relevant ones:
  - `request/header { header: { config: { provider, model, reasoningEffort?, … }, tools? }, reason }` is logged only when the call config changes (catalog:760-775).
  - `assistant/message { turn, step, message, usage?: TokenUsage, interrupted? }`. `message.source` is `{provider, model}` (catalog:297-321; `agent-loop/src/agent.ts:572-578`).
  - `TokenUsage { inputTokens; outputTokens; totalTokens?; cacheReadTokens?; cacheWriteTokens?; reasoningTokens? }` (`packages/llm/llm/src/types.ts:182-196`). The counts are disjoint: `inputTokens` is uncached input only. The DeepSeek adapter maps `cache_read_input_tokens` to `cacheReadTokens` (`llm-deepseek/src/translate.ts:37`).
  - `tool/call { turn, step, callId, name, arguments }` and `tool/result { turn, step, message, error?: {name, code, reason?} }`. `error` is present only on failure (catalog:1094-1185).
  - `approval/asked { id, toolName, callId?, reason? }` and `approval/decided { id, outcome }`, with `outcome` ∈ `allowed-once | rejected | cancelled | unavailable` (`user-approval/src/types.ts:32`).
  - `model/selection`, `llm/retry`, `turn/start|end` and `step/start|end`.
- **Live access:** `ctx.sessionPersistence`, `ctx.sessionQuery`, or the `session/event` feed on `ctx.sessions`.
- **Privacy:** the base bundle mounts `session-log-deepseek` by default (`cordis.patch.yml:43-44`). Its README says it uploads the session log incrementally inside official DeepSeek API requests. `apps/cli/reference/README.md:139` calls this "default-on DeepSeek session-log upload".

**Fork.**

- In process: `SessionStore.fork(source: Session | SessionId, boundary?: SessionSeq, childSessionId?: SessionId): Session` (`packages/core/session/src/index.ts:1483`). The boundary is inclusive, open steps and turns get synthetic closers, and the child header is marked `isSeeded: true`.
- Remote: `fork({ sessionId, atSeq?, allowMigration? }) → { sessionId }` (`api/session-controller/src/commands.ts:230`, `types.ts:310-321`).
- The Python SDK has no fork method.

**Plugin packaging.** Sources: `docs/user/develop/basic/index.md:17-27,89-101`, `publish.md:35-131`.

- A plugin is a Cordis module exporting `name`, optional `inject` (e.g. `['llm']`) and `apply(ctx)`.
- It ships as an npm package whose `package.json` declares `"dsh": { "bundle": { "patch": "./cordis.patch.yml" } }`, with an `- insert:` row naming the package.
- Install with `dsh plugin --profile <name> add <pkg | ./dir | github:user/repo#sha | .tgz>`.
- The only compatibility gate is `"engines": { "dsh": ">=0.2.0 <0.3.0" }`, which install and startup enforce (`publish.md:66`).
- Shared dsh packages go in both `peerDependencies` and `devDependencies` (`publish.md:105`).

**Versions.**

- There are 40 tags, `dsh-v0.1.0-rc.7` … `dsh-v0.2.2-rc.3`, and all GitHub releases are prereleases.
- npm `@deepseek-ai/dsh` dist-tags: `latest: 0.2.2-rc.1`, `next: 0.2.2-rc.3`.
- The sub-packages' `latest` tags are stale: `@deepseek-ai/dsh-llm` `latest` is 0.0.1-rc.1. Pin exact versions and never use `latest`.
- npm metadata has an empty `gitHead`, so the 0.2.2-rc.3 tarball was not tied to the tag commit.

**Providers.**

- `dsh-llm-deepseek` ships the models `deepseek-flash` and `deepseek-v4-pro`, both with 1,048,576 tokens of context (`llm-deepseek/src/models.ts:8-23`). It speaks Anthropic Messages at `https://api.deepseek.com/anthropic`, overridable with `baseURL` or `DEEPSEEK_BASE_URL` (`config.ts:116-119`).
- `dsh-llm-pi-ai` supports custom routes with `api: openai-completions | openai-responses | anthropic-messages | google-generative-ai` and a `baseURL` (`llm-pi-ai/src/provider.ts:48-53`; `docs/user/guide/providers.md:29,161`). This means dsh can point at an OpenAI-compatible local proxy.

### Confidence

- **High** for the names, signatures, log schema, fork API and versions. These were read directly from source and the repo docs.
- **Medium** for the recommended hook point (call `select()` from `tools/post-execute` between steps, or before the turn starts). It is inferred from the code and was not run.
- **Medium** for the exact `session.v4` filename. The README tree shows up to v3 and says "later versions use vN".

## 4. LiteLLM: callbacks, routing hooks, model and effort rewrite, cache tokens

**Answer.** Use a `CustomLogger` subclass whose `async_pre_call_hook` rewrites `data["model"]` and `data["reasoning_effort"]`. The proxy runs that hook before `route_request`, so the router selects on the rewritten model, and a request-level `reasoning_effort` overrides the deployment's thinking settings.

The hook is proxy-only. It runs only if the method is defined on the registered class itself; inheriting it is not enough.

Cascades cannot be built from callbacks: router fallbacks fire only on exceptions, and exceptions in success and streaming hooks are swallowed. A cascade must live in our own wrapper, and it cannot coexist with unbuffered streaming once bytes have gone out.

Cache tokens surface as follows:

- **Anthropic:** `prompt_tokens_details.cached_tokens` and `cache_creation_input_tokens`.
- **DeepSeek:** `prompt_tokens_details.cached_tokens`, plus the raw `prompt_cache_hit_tokens`.
- **Ollama:** LiteLLM's `ollama`/`ollama_chat` provider does **not** read Ollama's cached count.

### Evidence

Source is BerriAI/litellm at tag `v1.109.1`, commit `6c4ead4fc064d14e36b0c69d6795acc130c866bf`. Docs were checked 2026-11-13.

**Release.**

- 1.109.1 is the latest stable release: PyPI 2026-11-12, and a GitHub release that is not a prerelease. The LTS line is at 1.108.7.
- Advisory GHSA-2rwg-2wh4-7c4f (high, 2026-11-12) is fixed in 1.106.9, 1.108.6 and 1.109.1.
- Requires Python `>=3.10,<3.16`.

**Supply chain.**

- Never pin **1.82.7 or 1.82.8** (malicious `.pth` credential stealer, 2026-03-24; issues #24512, #24518).
- Never pin **1.107.0** (wheels built from the lookalike repo `Berri-AI/litellm`; Docker `latest`/`main-stable` were also affected; issue #47826, 2026-10-21).
- All three versions have been removed from PyPI.
- Pin by hash (`uv` lockfile hashes or `--require-hashes`).

**Hooks** (`litellm/integrations/custom_logger.py`).

Proxy only (the section is commented `#### CALL HOOKS - proxy only ####`, line 412):

- `async_pre_call_hook(self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict, call_type) -> Exception | str | dict | None` (417-426).
- `async_moderation_hook` (511). It runs in parallel with the LLM call and can reject a request but not change it.
- `async_post_call_success_hook(self, data, user_api_key_dict, response)` (495). A non-None return replaces the response (`proxy/utils.py:4419-4425`).
- `async_post_call_failure_hook` (474).
- `async_post_call_streaming_hook(self, user_api_key_dict, response: str)` (519). It receives the accumulated text so far, not the current chunk.
- `async_post_call_streaming_iterator_hook(self, user_api_key_dict, response, request_data) -> AsyncGenerator` (526).

SDK and proxy:

- `async_pre_call_deployment_hook(self, kwargs, call_type)` (280).
- `async_post_call_success_deployment_hook` (297).
- `async_post_call_failure_deployment_hook` (307).
- `async_log_success_event(self, kwargs, response_obj, start_time, end_time)` (190).

The proxy gate (`proxy/utils.py:3265`):

```python
and "async_pre_call_hook" in vars(_callback.__class__)
```

**Order** (`litellm/proxy/common_request_processing.py`).

1. Aliases (`model_alias_map`, key aliases, `model_group_alias`) are resolved at 2542-2564.
2. `self.data = await proxy_logging_obj.pre_call_hook(...)` runs at 2643.
3. `llm_call = await route_request(data=self.data, ...)` runs at 3096.
4. The router resolves the rewritten name (`route_llm_request.py:966-967`).

A returned dict replaces `data`. Returning None keeps the in-place mutation. A returned str or Exception rejects the request (`proxy/utils.py:2487-2502, 3287-3290`).

**Effort override.** `_deployment_params_with_request_reasoning_override` (`router_python_core.py:3393-3422`) drops the deployment's `thinking`, `output_config.effort` and `reasoning` when the request carries `reasoning_effort`.

**Real target and access checks.**

- The response's `model` is restored to what the client sent (3175-3180). The real target is in `standard_logging_object["model"]`, `["model_group"]` and `["model_id"]`.
- Not verified: whether key or team model-access checks are re-run on the rewritten model. Auth runs before the hook.

**Other routing extension points.**

- `CustomRoutingStrategyBase.async_get_available_deployment(model, messages=None, input=None, specific_deployment=False, request_kwargs=None)` (`types/router.py:1151`) is registered with `router.register_plugin()`. It replaces deployment selection inside one model group, so it is the wrong level for choosing between models.
- `Router.set_custom_routing_strategy` and `Router.async_pre_routing_hook` are deprecated shims (`router_deprecated.py:1-14, 801-836`).
- `CustomLogger.async_pre_routing_hook` runs only for the built-in auto, complexity, adaptive and quality strategies (`router_python_core.py:16415-16420`).
- The built-in `auto_router/complexity_router` is heuristic, and customising its classifier is an Enterprise feature (https://docs.litellm.ai/docs/proxy/auto_routing).

**`reasoning_effort` mapping per provider.**

- **Anthropic** (`llms/anthropic/chat/transformation.py:1644-1700, 2049-2057, 462-469`).
  - Models flagged `supports_adaptive_thinking` (all three Claude 5.5 entries are flagged) get `thinking={"type":"adaptive"}` plus `output_config={"effort": …}`.
  - The mapping is `minimal` → low, then low, medium, high, xhigh, max. `none` sends **no** thinking parameter at all (lines 1644-1651, 2044-2046), so the model's default applies: adaptive thinking on Opus 5.5 and Fable 5.2, which cannot turn it off. *(Corrected by issue 46; the spike said `none` → thinking off. Found by issue 12, section 8, and re-probed against the Anthropic-format mock.)*
- **DeepSeek** (`llms/deepseek/chat/transformation.py:23-85`).
  - `none` sends `thinking={"type":"disabled"}`. Any other value sends `thinking={"type":"enabled"}` and also forwards `reasoning_effort`.
  - `budget_tokens` is dropped with a warning.
- **Ollama** (`llms/ollama/common_utils.py:17-18, 304-310`).
  - Model names starting with `gpt-oss` get the string passed through as `think`.
  - All other models get a boolean: `true` only for low, medium or high. `minimal`, `xhigh` and `max` silently become `think=False`, with only a warning logged.

**Cache tokens.**

- **Anthropic** (`anthropic/chat/transformation.py:2858-2947`):
  - `prompt_tokens` = input + cache creation + cache read.
  - `prompt_tokens_details.cached_tokens` = cache read.
  - `prompt_tokens_details.cache_creation_tokens` = cache creation, with the 5 m / 1 h split in `cache_creation_token_details`.
  - The top-level `cache_creation_input_tokens` and `cache_read_input_tokens` are kept.
- **DeepSeek** (`types/utils.py:2075-2080, 2130-2133`): `prompt_cache_hit_tokens` maps to `prompt_tokens_details.cached_tokens`. The raw hit and miss fields stay on the usage object.
- **Ollama** (`ollama/chat/transformation.py:386-399, 418-427`): usage is only `prompt_eval_count` and `eval_count`. A grep of `litellm/` finds no use of Ollama's `prompt_eval_cached_count`. Ollama's own OpenAI-compatible endpoint does return `usage.prompt_tokens_details.cached_tokens` (`openai/openai.go:270-271` at Ollama `d63b7026`).
- **Logging:** `StandardLoggingPayload` has no top-level cache fields. Use `standard_logging_object["metadata"]["usage_object"]` and `cost_breakdown.cache_read_cost` / `cache_creation_cost` (`types/utils.py:3448, 3780-3781`). For streams, `async_log_success_event` fires once with the assembled response.

**Cost and the price map** (`model_prices_and_context_window.json`, inspected at the tag).

- Present and matching Anthropic's current prices: `claude-haiku-5-5` (including `*_above_100k_tokens` tiers), `claude-sonnet-5-5`, `claude-opus-5-5`, `deepseek/deepseek-flash` and `deepseek/deepseek-v4-pro`.
- **`claude-fable-5-2` is missing** (only `claude-fable-5` and `claude-fable-5-1` exist).
- DeepSeek entries carry peak prices only, so the off-peak 50 % discount is not modelled.
- When a cache rate is missing, cost falls back to the input rate (`llm_cost_calc/utils.py:871-879`).

**Cascade.**

- Router fallbacks trigger only on exceptions.
- Exceptions raised in `async_post_call_success_deployment_hook` are swallowed for ordinary loggers (`utils.py:1820-1835`).
- Guardrail interventions are excluded from fallback (`router_python_core.py:9101-9111`).
- Exceptions in the per-chunk stream hook are swallowed (`streaming_handler.py:1850-1863`).
- `MidStreamFallbackError` covers provider errors only.
- A non-streaming `async_post_call_success_hook` could call the router again and return the second response. Spend attribution for that second call was not verified.

**Licence.** Everything outside `enterprise/` is MIT (`LICENSE`). Everything listed above is in the MIT part.

**Docs.**

- https://docs.litellm.ai/docs/proxy/call_hooks (its example sets `data["model"] = "my-new-model"`)
- https://docs.litellm.ai/docs/observability/custom_callback
- https://docs.litellm.ai/docs/router/learn/strategies
- https://docs.litellm.ai/docs/proxy/logging_spec

### Confidence

- **High** for the hook signatures, call order, effort mapping, cache fields and supply-chain facts. These rest on source at the tag, the PyPI JSON and the GitHub issues.
- **Medium** that no proxy-config key exists for registering a custom routing strategy (none was found).
- **Low** for any trick that forces a fallback from a guardrail. Do not use one.

## 5. LLMRouter: custom-router plugin contract and data format

**Answer.** The repo is `ulab-uiuc/LLMRouter` (PyPI `llmrouter-lib` 0.4.0, MIT, Python ≥3.10). The contract named in the notes is real:

- `MetaRouter(nn.Module, ABC)` with abstract `route_batch(self, batch)` and `route_single(self, batch)`.
- Custom routers are discovered automatically from plugin directories.

Our adapter can be thin, about one class. Three constraints apply:

- It depends on torch.
- The xRouteBench pipeline hardcodes its router list.
- Training and evaluation data are dense rows of one per (query, model) with recorded performance, whereas our logged traces are bandit feedback.

### Evidence

Commit `c65ee0122bc9e84f71d571fcdec4a948481b6e88`, 2026-10-08.

**Contract.**

- `llmrouter/models/meta_router.py:11` declares `class MetaRouter(nn.Module, ABC)`.
- `:26` is `def __init__(self, model: nn.Module, yaml_path: str | None = None, resources=None)`. If a YAML path is given, it loads the config and attaches the data through `DataLoader.load_data`.
- `:70` and `:86` are `route_batch(self, batch)` and `route_single(self, batch)`. Both are abstract, with an untyped `batch`.
- In practice the input and output are dicts: `{"query": …}` → `{"model_name": …}` (`custom_routers/randomrouter/router.py:78-124`). Built-in routers use `route_batch(self, batch=None, task_name=None)`.

**Plugin discovery** (`llmrouter/plugin_system.py`).

- Lines 293-332 scan `./custom_routers/`, `~/.llmrouter/plugins/` and `$LLMROUTER_PLUGINS` (colon-separated).
- Each subdirectory becomes a router named after the directory in lower case (`:103`).
- The class picked is the first one whose name ends in `Router` and does not start with `Meta`.
- An optional `Trainer` class enables `llmrouter train` (`cli/router_train.py:106-114`).
- The only check is that `route_single` and `route_batch` exist (`:249`).

**Harness.**

- `benchmark_pipeline/run_pipeline.py:416` rejects names that are not in `ALL_ROUTERS` (`:47-55`).
- Its fallback calls `r.route_single({"query": row["query"]})` and reads `model_name` or `routed_model` (`:280-286`).
- Scoring joins on exact `query` + `model_name` (`:290-302`).

**Data** (`llmrouter/data/data_loader.py:41-54`). These are YAML `data_path` keys:

- `query_data_train/test` (jsonl).
- `routing_data_train/test` (jsonl, one row per query × model). Fields: `task_name, query, ground_truth, metric, choices, model_name, response, input_tokens, output_tokens, token_num, response_time, performance, embedding_id, user_id`.
- `query_embedding_data` (`.pt`).
- `llm_data` (JSON per model: `size, feature, input_price, output_price, model, service, api_endpoint`).
- `llm_embedding_data`.

Cost is computed from tokens and prices. In cost-aware runs, `performance` becomes `alpha*norm(perf) - beta*norm(cost)` (`run_pipeline.py:74-81`).

**GraphRouter.**

- The `graphrouter` baseline is re-implemented in-tree (`llmrouter/models/graphrouter/router.py:248,326`, citing arXiv 2410.03834). It does not import the standalone repo.
- The standalone `ulab-uiuc/GraphRouter` (MIT, `0edfb875`, 2025-12-30) is research scripts, not a package. It needs a dense CSV laid out task-major and query × LLM (`data_processing/construct_router_data.py:21,43`; `model/multi_task_graph_router.py:59-61`). Its README now points users to LLMRouter.
- The notes' lowercase `ulab-uiuc/graphrouter` resolves to `ulab-uiuc/GraphRouter`.

**Dependencies.** torch, torch-geometric, transformers, litellm and gradio. The CLI is `llmrouter`.

### Confidence

**High.** This was read from source at the pinned commit; the licence was read from the LICENSE file.

## 6. ParetoBandit and OpenSquilla (plus RouteLLM)

**Answer.**

- **ParetoBandit** is arXiv **2604.00136** ("ParetoBandit: Budget-Paced Adaptive Routing for Non-Stationary LLM Serving"). Code is at `ParetoBandit/ParetoBandit`, Apache-2.0, `pip install paretobandit` (0.1.1, Python ≥3.10, numpy + scikit-learn + joblib). It is a LinUCB router with a budget pacer, and its `route()` / `update()` API accepts precomputed feature vectors. It is the only external baseline built for bandit feedback and needs the thinnest adapter.
- **OpenSquilla's** paper id is confirmed: arXiv **2607.11399**, "Agentic Routing: The Harness-Native Data Flywheel". The code link `opensquilla/opensquilla` redirects to **`TokenRhythm/opensquilla`** (Apache-2.0, v0.5.8, Python ≥3.12). Its "open LightGBM router" ships as frozen weights with no training code. It takes request text and outputs one of four tiers (C0 to C3), so it can run only as a zero-shot baseline that maps tiers onto our ladder.
- **RouteLLM** is confirmed as arXiv **2406.18665** (`lm-sys/RouteLLM`, Apache-2.0, `routellm` 0.2.0). It routes between two models only, and the repo has been dormant since 2024-08.

### Evidence

**ParetoBandit** (commit `72ccbefe5cb858c95e00368e2b68de3c2b28a2ee`, 2026-04-13). All references are to `src/pareto_bandit/router.py` unless noted.

- `:749` is `BanditRouter.create(model_registry=…, context_model=…, priors="warmup", …)`. Registry entries need `input_cost_per_m` and `output_cost_per_m`.
- `:1276` is `route(self, prompt: str | np.ndarray, *, max_cost=None, max_latency=None, quality_floor=None, input_tokens=None, output_tokens=600) -> tuple[str, RoutingLog]`.
- `:1403` is `process_feedback(request_id, reward)`, with the reward clipped to [0, 1].
- `:1514` is `update(model_id, context, reward, weight=1.0, advance_time=True)`.
- `:479` is `register_model(...)`, so arms can be added at runtime.
- Precomputed vectors go through `FeatureService.for_precomputed(dimension=d)` (README:168-183).
- `budget_pacer.py:135` is `BudgetPacer(target_avg_spend_usd, …)`. The pacer sees cost estimated from registry prices, not realised cost (`router.py:1253, 1474`).

**OpenSquilla** (commit `fdc3898a31432b48a1731bfe663a2b6f44223e88`, 2026-11-10). Paths are under `src/opensquilla/squilla_router/lightgbm_model/`.

- **Input:** `input_contract.py:10-21`, `RouterInput(current_request: str, task_anchor, history_user, previous_answer, previous_usage, previous_outcome, active_route_tier, route_history, context, tool_state, attachments)`.
- **Features:** 380 (`_vendor/features/lightgbm380.py:44-53`), built from BGE-embedding PCA blocks, character TF-IDF→SVD and 88 numeric features.
- Tier inference builds features from `current_request` alone (`model.py:390, 412`).
- **API:**
  - `LightGBMRouterModel(model_root=None)`
  - `.predict_tier(router_input) -> {"tier_probabilities": {C0..C3}, "tier_decision", …}` (`model.py:383`)
  - `.predict_tier_features(float32[380])` (`:361`)
- **No training code:** `lightgbm_model/PROVENANCE.md` records that `router_training` was removed.
- **Weights:** stored in Git LFS (`assets/tier/model.txt.gz`, about 12 MB); `git lfs pull` is needed.
- **Current default policy:** a BERT-60M ONNX intent classifier now runs alongside the LightGBM tier classifier.
- **Tiers to models:** the mapping is in `opensquilla.toml` `[squilla_router.tiers.c0..c3]`. Thinking mode is derived by a rule (`controller.py:76`).
- **Install:** from the GitHub release wheel (`uv tool install --python 3.12 "opensquilla[recommended] @ https://github.com/TokenRhythm/opensquilla/releases/download/v0.5.8/..."`). The PyPI `opensquilla` 0.3.0 is stale, and it was not confirmed to be the same project.

**RouteLLM** (commit `0b64fdafe049e596a3f5657c219329f24af24198`, 2024-08-10).

- `routellm/controller.py:42` is `Controller(routers, strong_model, weak_model, config=None, api_base=None, api_key=None, progress_bar=False)`.
- `routers/routers.py:38` is the abstract `calculate_strong_win_rate(prompt)`, compared against a threshold.
- The routers are `random, mf, causal_llm, bert, sw_ranking` (`:256`).
- Retraining `mf` needs pairwise battles plus prompt embeddings, with a fixed `MODEL_IDS` table, and inference embeds prompts with OpenAI `text-embedding-3-small` (`matrix_factorization/model.py:88,113`).
- The training code for `bert` and `causal_llm` is not in this repo.

### Confidence

- **High** for all three. This rests on source at the pinned commits, the LICENSE files, PyPI and the arXiv abstract pages.
- **Medium** for how closely current OpenSquilla matches the paper's LightGBM ranker, given the added BERT intent model.

## 7. Prompt-cache behaviour per provider

**Answer.**

- **Anthropic:** cache writes cost 1.25x base input (5-minute TTL) or 2x (1-hour TTL). Reads cost 0.1x (Haiku 5.5), 0.05x (Sonnet 5.5, Opus 5.5) or 0.025x (Fable 5.2). The minimum cacheable prefix is 512 tokens on all four models. The TTL is refreshed on each hit. The cache is per model and per workspace. Changing top-level `output_config.effort` or the thinking mode can invalidate the message cache, and on some models the whole cache. The beta header `mid-conversation-output-config-2026-07-01` changes effort per message without breaking the cache.
- **DeepSeek:** caching is automatic and has no write premium; a miss is billed at the normal input price. A hit costs about 2 % (flash) or 3.3 % (pro) of the miss price. The lifetime is best-effort, "a few hours to a few days". Whether the cache is scoped per model, and the effect of toggling thinking, are not documented.
- **Ollama:** the KV cache is reused for a matching prefix while the model stays loaded, and it is lost when the model is unloaded (on `keep_alive` expiry or to make room for another model). Usage reports `prompt_eval_cached_count`.

### 7a. Anthropic

Sources (checked 2026-11-13):

- Pricing: https://platform.claude.com/docs/en/about-claude/pricing
- Prompt caching: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- Models overview: https://platform.claude.com/docs/en/about-claude/models/overview
- Effort: https://platform.claude.com/docs/en/build-with-claude/effort
- Per-message effort: https://platform.claude.com/docs/en/build-with-claude/effort-per-message
- Thinking: https://platform.claude.com/docs/en/build-with-claude/thinking
- Cache diagnostics: https://platform.claude.com/docs/en/build-with-claude/cache-diagnostics
- Messages API: https://platform.claude.com/docs/en/api/messages/create

| Model (API id) | Input $/MTok | 5 m write | 1 h write | Cache read (mult) | Output | Min prefix | Effort levels (default) |
|---|---|---|---|---|---|---|---|
| Fable 5.2 `claude-fable-5-2` | 8 | 10 | 16 | 0.20 (0.025x) | 40 | 512 | low, medium, high, xhigh, max (high) |
| Opus 5.5 `claude-opus-5-5` | 4 | 5 | 8 | 0.20 (0.05x) | 20 | 512 | same (**medium**) |
| Sonnet 5.5 `claude-sonnet-5-5` | 2 | 2.50 | 4 | 0.10 (0.05x) | 10 | 512 | same (high) |
| Haiku 5.5 `claude-haiku-5-5`, prompt ≤ 100K | 0.10 | 0.125 | 0.20 | 0.01 (0.1x) | 0.50 | 512 | same (**medium**) |
| Haiku 5.5, prompt > 100K | 0.50 | 0.625 | 1.00 | 0.05 (0.1x) | 2.50 | 512 | same |

- The pricing footnote reads: "Cache hits and refreshes on Claude Fable 5.2 … are priced at 0.025x the base input price."
- The Haiku tier boundary counts cache tokens: "A request's prompt length counts all of its input tokens, including cache reads and cache writes … a request over the threshold pays the higher prices even when part of its prompt is a cache hit."
- All four models have a 1M context and 128K max output. Batch pricing is 50 % off; `inference_geo: "us"` costs 1.1x.

**Lifetime.**

- `ttl` is `"5m"` (default) or `"1h"`. "This lifetime is refreshed each time the cached content is used."
- A request can have up to 4 breakpoints.
- An entry is readable only after the first response starts.

**Scope.**

- "Caches are isolated between organizations … Caches are also isolated by workspace within an organization."
- The cache-diagnostics page, `model_changed` row: "The cache is per-model."
- Changing `speed` (fast mode) invalidates the system and message caches.

**Invalidation.** The prompt-caching page has an invalidation table. The cache hierarchy is tools → system → messages, and a change invalidates its own level and every level after it.

- Tool definitions: invalidate everything.
- `tool_choice`, or adding or removing images: invalidate messages.
- Thinking parameters and `output_config.effort`: invalidate messages; whether tools and system survive is model-specific.
- Mid-conversation `role: "system"` messages keep the cache on all four models.

**The effort lead.** The lead's substance is confirmed, but its location is wrong. The API reference entry for `output_config.effort` says nothing about caching. The statement is in the guides:

- Effort page: "Because top-level effort shapes the rendered prompt, changing it between requests can invalidate cached prefixes from earlier turns."
- Thinking page: "Changing the effort value can invalidate message-level breakpoints on any model. On Claude Opus 5, Claude Sonnet 5, Claude Fable 5, Claude Mythos 5, and earlier models … tool and system-prompt breakpoints too. Treat any thinking or top-level effort change as invalidating the whole cache."
- Prompt-caching page: "Setting effort explicitly to the model's default is equivalent to omitting it and does not invalidate."

**Per-message effort (beta).**

- Header: `mid-conversation-output-config-2026-07-01`.
- How it works: "A per-message effort change sets a new effort level for later turns of a conversation without changing the request's top-level value, so the cached prefix still matches." The change is a `role: "system"` message with empty `content` and `output_config.effort`.
- Supported models include `claude-fable-5-2`, `claude-opus-5-5`, `claude-sonnet-5-5` and `claude-haiku-5-5`.
- Restriction: "On Claude Sonnet 5.5 with `thinking: {"type": "between_tools"}`, and on Claude Haiku 5.5 with `thinking: {"type": "disabled"}`, you can't change the effort level mid-conversation … returns a 400 error."
- https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence: "Changing effort at the top level cost 20 to 40 times what changing it in a message cost."

**Thinking modes.**

- "Switching between `adaptive`, `enabled`, and `disabled`, or changing `budget_tokens`, invalidates cache breakpoints."
- `budget_tokens` is rejected with a 400 on all four current models.
- Opus 5.5 and Fable 5.2 accept only `adaptive`.
- Sonnet 5.5 also allows `between_tools` at effort high or below.
- Haiku 5.5 also allows `disabled` at effort high or below.

**Usage.**

- `input_tokens` is "Number of input tokens that were not read from or used to create a cache".
- Total input = `cache_read_input_tokens + cache_creation_input_tokens + input_tokens`.
- `cache_creation.ephemeral_5m_input_tokens` and `ephemeral_1h_input_tokens` break down the writes, and their sum equals `cache_creation_input_tokens`.
- A prompt below the minimum is silently not cached; both cache fields are 0.

**Confidence: high.** This rests on verbatim quotes from the official pages.

### 7b. DeepSeek

Sources (checked 2026-11-13):

- Pricing: https://api-docs.deepseek.com/quick_start/pricing
- Context caching: https://api-docs.deepseek.com/guides/kv_cache
- Thinking mode: https://api-docs.deepseek.com/guides/thinking_mode
- Chat completion API: https://api-docs.deepseek.com/api/create-chat-completion
- Rate limits: https://api-docs.deepseek.com/quick_start/rate_limit
- Anthropic API: https://api-docs.deepseek.com/guides/anthropic_api
- Updates: https://api-docs.deepseek.com/updates

| | `deepseek-flash` | `deepseek-v4-pro` |
|---|---|---|
| Serves | DeepSeek-V4.1-Flash-1017 | DeepSeek-V4-Pro-0813 |
| Context / max output | 1,048,576 / 393,216 | 1,048,576 / 393,216 |
| Input, cache hit (peak / off-peak) $/MTok | 0.006 / 0.003 | 0.044 / 0.022 |
| Input, cache miss (peak / off-peak) | 0.30 / 0.15 | 1.32 / 0.66 |
| Output (peak / off-peak) | 1.20 / 0.60 | 3.96 / 1.98 |

**Model ids.**

- The API enum is `["deepseek-flash", "deepseek-v4-pro"]`.
- "The legacy names `deepseek-v4-flash` and `deepseek-v4-flash-vision-exp` are still accepted, but the corresponding models have been retired, their requests are served by the DeepSeek-V4.1-Flash-1017 model."
- The updates page (2026-04-24) says `deepseek-chat` and `deepseek-reasoner` "will be discontinued in three months (2026-07-24)". They are absent from the current enum. Whether they now return an error was not tested with a live call.

**Off-peak hours.** "Off-peak rates are half of the peak rates. Peak hours are 01:00 - 04:00 and 06:00 - 10:00 UTC, Monday through Friday, excluding Chinese public holidays."

**Caching.**

- Caching is automatic: "enabled by default for all users".
- No cache-write price is listed; a miss is billed at the miss rate.
- Each cached prefix is "an independent, complete unit". A request hits only if it fully matches a stored unit.
- Units are stored at the end of the user input, at the end of the model output, on common prefixes, and "at fixed token intervals" (the interval is not stated).
- Lifetime: "automatically cleared, usually within a few hours to a few days", best-effort.

**Reasoning controls.**

- Chat Completions takes `thinking: {"type": "enabled" | "disabled"}` and `reasoning_effort` ∈ `none | low | high | max`.
- Thinking is on by default at `high`. The aliases map `minimal` → low, `medium` and `xhigh` → high, and `ultra` → max.
- In thinking mode, `temperature` and the penalty parameters are ignored.
- The Anthropic-format endpoint uses `thinking` plus `output_config.effort`.

**Usage.**

- `prompt_tokens` "equals prompt_cache_hit_tokens + prompt_cache_miss_tokens".
- `prompt_tokens_details.cached_tokens` equals the hit count.
- `completion_tokens_details.reasoning_tokens` is reported.

**Anthropic-compatible endpoint.** It exists at `https://api.deepseek.com/anthropic`. There, `cache_control` is ignored and `cache_creation_input_tokens` is "currently `0`". `claude-opus*` model names map to `deepseek-v4-pro`, and other `claude*` names map to `deepseek-flash`.

**Could not determine:**

- The minimum cacheable length or unit. The current page no longer states the 64-token unit from older notes.
- Whether the cache is scoped per model. The rate-limit page says `user_id` "isolate[s] KVCache for users on your business side", and concurrency is per account.
- Whether toggling thinking invalidates the prefix. The thinking guide says that without `tools`, earlier `reasoning_content` "will not be concatenated into the context", which suggests turn-end units may not match. This is inference only.

Tried: the pricing, kv_cache, thinking_mode, rate_limit, list-models and chat-completion pages, and the news and updates pages.

**Confidence:**

- **High** for the ids, prices and usage fields.
- **Medium** for the cache lifetime (the docs call it best-effort).
- **Low** for the scope and the thinking-toggle effect.

### 7c. Ollama

Sources (checked 2026-11-13):

- Library pages: https://ollama.com/library/qwen3.5:9b, https://ollama.com/library/gpt-oss:20b, https://ollama.com/library/qwen3-coder:30b, https://ollama.com/library/devstral:24b (context lengths from each family's tag table)
- Docs: https://docs.ollama.com/faq, https://docs.ollama.com/context-length, https://docs.ollama.com/api/usage, https://docs.ollama.com/capabilities/thinking, https://docs.ollama.com/api/openai-compatibility
- Source: `ollama/ollama` at `v0.40.16` (`d63b7026…`)

**Decision D2 tags.** All four exist.

| Tag | Size | Context | Quant / params | Capabilities | Reasoning control |
|---|---|---|---|---|---|
| `qwen3.5:9b` (= `latest`) | 6.6 GB (llama.cpp), 7.6 GB (mlx) | 256K | Q4_K_M, 8.95B, plus a 922 MB vision projector | vision, tools, thinking | `think` on or off |
| `gpt-oss:20b` | 14 GB | 128K | MXFP4, 20.9B | tools, thinking | `low` / `medium` (default) / `high`; cannot be disabled |
| `qwen3-coder:30b` | 19 GB | 256K | Q4_K_M, 30.5B MoE (3.3B active) | tools | none |
| `devstral:24b` | 14 GB | 128K | Q4_K_M, 23.6B | tools | none |

- The thinking docs say `false` means "Models that cannot disable thinking use their lowest supported level instead".
- How `/v1` maps `reasoning_effort` for GPT-OSS: the code (`openai/openai.go:587-618`) first aliases `minimal` to `low` and `xhigh` / `ultra` to `max`, and sends `none` as `think = false`; GPT-OSS gets no model metadata on this path (`server/model_thinking.go:97-103`, Harmony is excluded), so those values go on to the GPT-OSS handling. The compatibility page at the tag (`docs/api/openai-compatibility.mdx:253`) says the end result for GPT-OSS is `minimal` → `low`, `xhigh` / `ultra` → `high`, and `none` → no thinking output. *(Corrected by issue 46; the spike said `none` → `low` and that the aliasing to `high` happens directly. The catalog never sends `none`, `xhigh` or `ultra` to gpt-oss, so nothing depends on it.)*
- The gpt-oss template puts `Reasoning: {{ .ThinkLevel }}` and `Current date: …` in the system header (https://ollama.com/library/gpt-oss:20b/blobs/fa6710a93d78).

**KV reuse.**

- On the llama.cpp path, Ollama launches upstream `llama-server` with `-c NumCtx*numParallel` and sends `"cache_prompt": true` on every request (`llm/llama_server.go`). The prefix matching itself happens inside upstream llama-server; that code was not inspected.
- The MLX path keeps "a compressed prefix trie" shared across conversations, with snapshots paged out up to 8 GiB (`mlxrunner/prefix_cache.go`).
- The lead's `runner/ollamarunner/cache.go` no longer exists at this tag.

**Defaults** (`envconfig/config.go`, FAQ).

| Setting | Default | Note |
|---|---|---|
| `keep_alive` | 5 minutes | `0` unloads at once; a negative value keeps the model loaded |
| `OLLAMA_NUM_PARALLEL` | 1 | Parallel slots multiply the context: "a 2K context with 4 parallel requests will result in an 8K context" |
| `OLLAMA_MAX_LOADED_MODELS` | 0 (auto) | Auto means 3 per GPU |

**Loading another model.** When the model limit or memory is reached, `server/sched.go` `findRunnerToUnload` unloads an idle runner. The FAQ: "As prior models become idle, one or more will be unloaded to make room for the new model." Unloading ends the runner process, so its KV cache is lost.

**Reloads.**

- `needsReload` (`server/sched.go`, about lines 1416-1467) reloads the model when runner options change: `num_ctx`, `num_batch`, `num_gpu`, `main_gpu`, `use_mmap`, `num_thread` or `draft_num_predict`.
- Changing `think` does not force a reload, because it is not a runner option. For gpt-oss, though, it rewrites the system header, so the effect is close to a full prefix miss. The daily `Current date` line has the same effect once a day.

**Usage.**

- `prompt_eval_cached_count` is "How many prompt tokens were read from the cache" (`/api/usage`).
- `prompt_eval_count` includes the cached tokens (`llm/llama_server.go`, `cache_n + prompt_n`).
- The OpenAI-compatible `/v1` endpoint sends `usage.prompt_tokens_details.cached_tokens` when the count is available (`openai/openai.go:270-271`).

**Context default.** The docs contradict each other.

- FAQ: "By default, Ollama uses a context window size of 4096 tokens."
- https://docs.ollama.com/context-length: 4k below 24 GiB of VRAM, 32k for 24-48 GiB, 256k at 48 GiB and above.
- Which one the code implements on a 48 GB Mac was not determined.
- `num_ctx` cannot be set through the OpenAI-compatible API. Use `OLLAMA_CONTEXT_LENGTH`, or a Modelfile `PARAMETER num_ctx`.

**Confidence:**

- **High** for tags, sizes, usage fields and eviction.
- **Medium** for the details of llama.cpp prefix matching (upstream code not read).
- **Low** for which default context applies on the maintainer's machine. Set it explicitly.

---

## Consequences for issues 12 to 15

### Issue 12 (LiteLLM proxy)

1. **Pin `litellm==1.109.1` with hashes** in the `[proxy]` extra. Add a comment that bans 1.82.7, 1.82.8 and 1.107.0. Do not use the `latest`/`main-stable` Docker tags.
2. **Hook.** Use a `CustomLogger` subclass that defines `async_pre_call_hook` **directly on the registered class**, not in a shared base class (`proxy/utils.py:3265`).
   - It calls `/decide`, then sets `data["model"]` to a configured `model_name` and sets `data["reasoning_effort"]`.
   - Shadow mode returns `data` unchanged.
   - Log the real target from `standard_logging_object["model_id"]`, because the response's `model` echoes the client's value.
3. **Cascade is not possible inside LiteLLM callbacks.** Implement it in our own layer: a thin route or wrapper that calls `litellm.Router.acompletion` itself and re-issues the call when the check fails. **The spec conflicts with itself.** "Execute cascades when the response fails a check" needs the full response, and "streaming must pass through without buffering" forbids holding it. Proposed resolution:
   - Cascades apply to non-streaming requests only.
   - Streaming requests get one attempt plus a logged outcome, or an opt-in buffered mode.
   - Update the acceptance criteria to match.
4. **Ollama upstream.** Configure Ollama through its OpenAI-compatible endpoint (`api_base: http://localhost:11434/v1`) rather than LiteLLM's `ollama`/`ollama_chat` provider. The native provider drops `prompt_eval_cached_count` and silently turns `reasoning_effort` `minimal`/`xhigh`/`max` into `think=false` for non-gpt-oss models. **VERIFY in issue 12:** that LiteLLM's `openai/` (or `hosted_vllm/`) provider forwards `reasoning_effort` unchanged and surfaces `prompt_tokens_details.cached_tokens`.
5. **Cache accounting.** Read cache tokens from `usage.prompt_tokens_details.cached_tokens` for all three providers, and from `cache_creation_input_tokens` for Anthropic. Do not trust LiteLLM's `response_cost`:
   - `claude-fable-5-2` is missing from the 1.109.1 price map.
   - DeepSeek's off-peak 50 % discount is not modelled.
   Compute cost with our own `realized_cost` (issue 6) from the catalog.
6. **Effort vocabulary per upstream.** For DeepSeek, send `none|low|high|max` (`medium` is aliased to `high`). For gpt-oss, `low|medium|high`, with no "off". For `qwen3-coder:30b` and `devstral:24b`, send nothing.
7. **Context cap (D2).** It cannot be set per request over `/v1`. Document `OLLAMA_CONTEXT_LENGTH=32768` (and `OLLAMA_MAX_LOADED_MODELS=1`) in the smoke-test setup, because the default on 48 GB is ambiguous in the docs and may be 256k.

### Issue 13 (MCP server)

1. The "advisory only" note stays correct: no MCP mechanism changes the host's model.
2. Subagents inherit the parent's MCP tools by default, so a subagent can itself call `route_task` and `report_outcome`. A subagent definition with a `tools` allowlist must list `mcp__steproute__*` explicitly, or it loses them.
3. Plugin-shipped subagents ignore `mcpServers`. Ship the server in the plugin's `.mcp.json`.

### Issue 14 (Claude Code plugin)

1. **Primary mechanism: a mod**, meaning a plugin with a JS/TS hooks module (Claude Code ≥ v2.1.287).
   - `agent.spawn` applies the decision with `next({ ...e, model })`. Skip events where `e.resume` is set, because a model change there is ignored.
   - `turn.step` reads per-request `result.usage` (input, output, `cache_read_input_tokens`, `cache_creation_input_tokens`) and `model`, keyed by `e.agentId`. This is the per-request outcome source.
   - Effort per spawn: `turn.step` can set `next({ ...e, effort })` on requests carrying that `agentId`. Alternatively, the per-call Agent `effort` parameter (v2.1.292+) can be used.
   - The hooks module is TypeScript or JavaScript, so this collector is not pure Python. It calls `/decide` and `/report` over HTTP.
2. **Fallback for older versions:** a settings or plugin `PreToolUse` hook on `Agent` returns `updatedInput` (the whole input, with `model` changed) **together with `permissionDecision: "allow"`**. Without it, auto mode may deny the call.
   - Limitations: the per-call model accepts only the aliases `sonnet|opus|haiku|fable`, not full ids, and `availableModels` or `CLAUDE_CODE_SUBAGENT_MODEL_FORCE` can override it.
   - Always check `tool_response.resolvedModel` in `PostToolUse`.
   - This path is not shown in the docs; issue 14 must test it and record the result here.
3. **Token usage without a mod is weak.**
   - The Agent `PostToolUse` `usage` covers the final request only, and background subagents report none.
   - The OpenTelemetry route needs the user to set `OTEL_*` to point at a collector. Hooks cannot see those variables, and core would need an OTLP receiver, so this is optional, not the default.
   - Do not parse transcript JSONL; its format is internal.
4. **Per-tier subagent definitions** in the plugin may set `model` (full ids allowed in frontmatter) and `effort`, but not `hooks`, `mcpServers` or `permissionMode`. Put hooks in `hooks/hooks.json`.
5. **Cost model for spawns.** A subagent's first request never reads the parent's cache. For spawn decisions, `cached_prefix_tokens = 0` and no model owns a cache, so switching model at a spawn boundary carries no switch tax. This supports the "spawn boundaries only" rule.
6. Remove the "subagents do not inherit MCP" assumption wherever it appears.

### Issue 15 (dsh plugin)

1. **Pin `@deepseek-ai/dsh*` to exactly `0.2.2-rc.3`** (tag commit `38a6a5c9`) as `peerDependencies` and `devDependencies`. Set `"engines": { "dsh": ">=0.2.2-rc.3 <0.3.0" }`. Never rely on npm `latest`, because the sub-package `latest` tags are stale. Expect breaking changes, since the project is a developer preview.
2. **Do not rewrite inside `llm/stream`.** Loop requests are deep-frozen there. Short-circuiting with a different model would make the logged `request/header` and `assistant/message.source.model` wrong.
3. **Apply decisions with `agent.model.select({ provider, model, reasoningEffort })`** before the next step is prepared. The async `tools/post-execute` waterfall is a candidate place to await `/decide` between steps. At turn start the selection read is synchronous, so the decision must already be in hand.
   - `agent/pre-step` sees `modelSelection` read-only.
   - `agent/request` is deprecated and throws on model or effort changes.
   - Shadow mode skips `select()`.
   - Fail-open: if `/decide` times out, do not call `select()`.
4. A model change makes dsh inject a "[model changed …]" user notice and a new `request/header`. Both alter the prefix, so the cost model must treat every dsh model switch as a full cache miss.
5. **Outcomes from the session log.** Subscribe to the `session/event` feed on `ctx.sessions`, or read the file with zstd decompression (`session.v4.jsonl.zstd`).
   - Map `assistant/message.usage`: `inputTokens` is uncached input and `cacheReadTokens` is cache hits.
   - Map tool errors from `tool/result.error`, and approvals and rejections from `approval/decided.outcome`.
   - Map retries from `llm/retry`, and the model and effort from `request/header.config` and `model/selection`.
   - Any event the plugin writes must set `ignorable: true`.
6. Use `ctx.sessions.fork(source, boundary?)` for replay branching if issue 18 needs it. It is not needed for the collector.
7. **Privacy (issue 16 and decision D5):** dsh's default bundle mounts `session-log-deepseek`, which uploads the session log to DeepSeek inside API requests. The plugin README must tell users this, independent of steproute. `sensitive` routing cannot prevent dsh's own upload.
8. **Fallback design.** If the plugin route breaks on a dsh upgrade, point a `dsh-llm-pi-ai` route at the issue 12 proxy (`api: openai-completions`, `baseURL: http://127.0.0.1:<port>/v1`, `compat.thinkingFormat: deepseek`). The session log would then record the proxy's model id, not the routed one.

### Notes for other issues

**Issue 4 (catalog).**

- DeepSeek ids are `deepseek-flash` and `deepseek-v4-pro`. `deepseek-chat` and `deepseek-reasoner` were discontinued on 2026-07-24.
- DeepSeek's reasoning control has four levels (`none|low|high|max`), not on/off.
- Off-peak pricing is time-of-day dependent, so the catalog needs a peak/off-peak price pair or a documented choice of peak prices only.
- Anthropic cache-read multipliers differ per model (0.025, 0.05, 0.05, 0.1). Store absolute cache prices, not one shared multiplier.
- Store separate 5 m and 1 h write prices.
- Effort defaults differ: Opus 5.5 and Haiku 5.5 default to `medium`, Sonnet 5.5 and Fable 5.2 to `high`.
- Ollama: `qwen3-coder:30b` and `devstral:24b` have no reasoning control. `gpt-oss:20b` has `low|medium|high` and cannot be turned off. `qwen3.5:9b` has thinking on or off. All four D2 tags exist, so no substitution is needed.
- The tool-call smoke test is still required.
- The starting prices in issue 4 for the four Claude tiers match the official pricing page, including the Haiku 100K tier.

**Issue 6 (cost model).** Three corrections to `docs/DESIGN.md` section 5.

- **Switch tax only with caching on.** The "whole prompt billed at the cache-write rate" rule holds only for Anthropic requests that set `cache_control`. Otherwise the miss is billed at base input. DeepSeek has no write premium (`cache_write_mult = 1.0`), so its switch tax is the lost hit discount. Ollama's switch cost is latency (reload plus full prefill), not money.
- **Cache ownership.** For Anthropic it is per model, and it also depends on the effort and thinking setting. A top-level effort or thinking change can invalidate the message cache, so the "same model at higher effort" cascade step is not free unless the proxy uses the per-message effort beta (`mid-conversation-output-config-2026-07-01`). That beta is unavailable for Sonnet 5.5 with `between_tools` and Haiku 5.5 with thinking `disabled`. On gpt-oss, a reasoning-level change rewrites the system header, which is effectively a full miss. Model the cache owner as (model, effort) by default, with a per-provider flag for "effort change preserves cache".
- **Haiku's 100K tier** counts cache reads and writes toward the threshold.

Two smaller points:

- Anthropic's `input_tokens` excludes cache tokens, and so does dsh's `inputTokens`. DeepSeek's and LiteLLM's `prompt_tokens` include them. `realized_cost` must normalise these.
- The minimum cacheable prefix is 512 tokens for Anthropic.

**Issue 18 (replay runner).**

- Fresh-session isolation is consistent with per-model and per-workspace caching. Sharing one Anthropic workspace across concurrent configurations does not leak cache between different models, but it does between runs on the same model with identical prefixes. Use a per-run nonce in the system prompt if strict isolation matters.
- Ollama: load one model at a time (`OLLAMA_MAX_LOADED_MODELS=1`) and set the context explicitly. Expect a cold KV cache at every model change.
- DeepSeek off-peak hours halve the cost, so scheduling DeepSeek runs off-peak stretches the $20 cap.

**Issue 22 (external baselines).**

- LLMRouter, GraphRouter and RouteLLM train on full-information data (every model's outcome per query), and LLMRouter scores by looking up the outcome of the model it picks. Our step-level traces are bandit feedback. Run these baselines on issue 18's dense **task × configuration** matrix, at task level. Do not run them on step-level traces.
- **ParetoBandit** is the one baseline that fits step-level bandit data. Use `route()` with precomputed features, replay evaluation that keeps only steps where the pick matches the logged action, and `update()`. Its budget pacer uses estimated cost, so track realised cost outside the library.
- **LLMRouter:** adding our router needs a `MetaRouter` subclass (which brings in a torch dependency, so keep it in the `[ml]` extra or outside `src/`). It also needs a small patch to the `ALL_ROUTERS` list in `run_pipeline.py`, or the `llmrouter train/infer` CLI instead. Encode (model, effort) as `model_name = "<model>|<effort>"`.
- **OpenSquilla:** frozen, text-only, Python ≥ 3.12, weights in Git LFS, no training code. Run it as a zero-shot tier baseline with a written tier → (model, effort) mapping, in a separate Python 3.12 environment. Our numeric step state cannot be fed in.
- **RouteLLM:** two arms only. Retraining `mf` needs pairwise labels and calls OpenAI embeddings, which conflicts with D4 if it means an OpenAI key. Either swap the embedding function or write the note explaining why it could not be run fairly.
- Licences: ParetoBandit, OpenSquilla and RouteLLM are Apache-2.0, and LLMRouter and GraphRouter are MIT. Use them as external dependencies, not vendored code, so the MIT package needs no notices.

### Contradictions with `docs/ISSUES.md` and `docs/DESIGN.md`

| Where | Says | Finding |
|---|---|---|
| ISSUES issue 10, Q1 | Community claim that subagents do not inherit MCP servers | False: they inherit MCP tools by default (section 1). |
| ISSUES issue 15, DESIGN section 3 | dsh plugin "wraps the LLM seam" | The seam's waterfall cannot rewrite loop requests. Use `agent.model.select()` instead (section 3). |
| Maintainer notes (dsh) | docs at `deepseek.com/harness/en/`; `intercept` patch action | The docs URL is wrong, and there is no `intercept` patch action (section 3). |
| ISSUES issue 12 | Cascades on response checks and unbuffered streaming | These cannot both hold for one request (issue 12, item 3). |
| ISSUES issue 4 | DeepSeek "thinking on/off" | Four effort levels, plus time-of-day pricing (section 7b). |
| DESIGN section 5, ISSUES issue 6 | Cache owned by model; switch billed at the cache-write rate | Ownership is per (model, effort) for Anthropic unless the per-message beta is used. The write premium exists only for Anthropic with `cache_control` (issue 6 notes). |
| ISSUES issue 14 | Hooks capture "token usage including cache fields" | Hooks give final-request usage only. Use a mod's `turn.step` hook or OpenTelemetry (section 2). |

---

## 8. Issue 12: LiteLLM's OpenAI-compatible path, parameter handling and network behaviour

*Added by issue 12, checked 2026-11-13.* Unlike sections 1 to 7, these findings come from the installed wheel of `litellm==1.109.1` (from `uv.lock`, hash-checked). Its `litellm/` tree matches tag `v1.109.1`. Line numbers are in that installed package.

Each finding was also exercised against the loopback mock (`src/steproute/collectors/mock_upstream.py`), or through LiteLLM's request transformation called directly. No provider was called.

**Answer.**

- **Reasoning effort is forwarded unchanged on the OpenAI-compatible path.** For `openai/<model>`, any model name not in OpenAI's own catalog gets `OpenAIUnknownModelConfig`, which adds `reasoning_effort` to the supported parameters. That covers every Ollama tag, `gpt-oss:20b` included. The value is passed through unvalidated: `none`, `low`, `medium`, `max` and even `true` arrived at the mock verbatim. `temperature` passes through too.
- **`cached_tokens` surfaces on the same path.** `prompt_tokens_details.cached_tokens` comes through in both the non-streaming usage and the final streaming usage chunk (`stream_options.include_usage`).
- **So Ollama should go through `provider = "openai"` and `base_url = http://127.0.0.1:11434/v1`.** At issue 12 the catalog's `effort_map` for `qwen3.5:9b` held the native-API values `"true"`/`"false"`, which are not valid on Ollama's `/v1`, and the proxy translated them per upstream (`reasoning_values`). Issue 46 put the `/v1` values in the catalog (section 9).
- **The pinned LiteLLM makes network calls of its own unless switched off.** On import it fetches the model cost map from GitHub, with a background retry thread. On the first Anthropic call it fetches the beta-header config. `LITELLM_OFFLINE_MODE=true` stops both, and the Hugging Face tokenizer download. Telemetry exists only in the LiteLLM proxy server and is opt-in there.
- **Claude parameter checks are client-side, and incomplete for Fable 5.2:**
  - All four Claude models: `temperature != 1`, `top_p` and `top_k` raise `UnsupportedParamsError` (status 400) before any request unless `drop_params` is set.
  - Sonnet 5.5 and Opus 5.5: forced `tool_choice` raises the same error.
  - Fable 5.2 is missing from the model map, so its forced `tool_choice` (`any`/`tool`) is **forwarded** and would get Anthropic's 400.
  - An Anthropic refusal becomes `finish_reason = "content_filter"`.

**Evidence.**

*OpenAI-compatible path:*

- `utils.py:9876-9877`: `ProviderConfigManager.get_provider_chat_config` returns `OpenAIUnknownModelConfig()` when `not OpenAIGPTConfig.is_openai_catalog_model(model)`.
- `llms/openai/chat/gpt_transformation.py:997-1002`: `OpenAIUnknownModelConfig` adds `reasoning_effort` to the supported parameters.
- `llms/openai/openai.py:229-232`: the same choice on the handler side.
- The o-series check needs `o<digit>` and catalog membership (`llms/openai/chat/o_series_transformation.py:125-132`), so `gpt-oss` does not match it.
- `litellm_core_utils/llm_response_utils/convert_dict_to_response.py:883`: `litellm.Usage(**response_object["usage"])`.
- `types/utils.py:2063-2073`: keeps the provider's `prompt_tokens_details` as given.
- Probe result for `openai/qwen3-coder:30b`: `Usage(prompt_tokens=100, prompt_tokens_details=…cached_tokens=40)`.
- Streaming probe: the final chunk was `{"choices": [], "usage": {"prompt_tokens": 100, …, "prompt_tokens_details": {"cached_tokens": 40}}}`.

*Ollama's side* (`openai/openai.go` at `d63b7026`):

- `FromChatRequest` takes `reasoning.effort` or `reasoning_effort`.
- `ThinkingFromReasoningEffort` maps `""` to no setting and `"none"` to `think=false`, and aliases `minimal` to `low` and `xhigh`/`ultra` to `max`. For a model without thinking metadata it rejects unrecognised values with an error. *(Issue 46: for a model with metadata, such as one on the `qwen3.5` renderer, an unrecognised value is not rejected but passed on as a name and resolved to the model's default, thinking on; section 9.)* Either way the native `"true"`/`"false"` must not be sent.
- `ToUsage` sets `prompt_tokens_details.cached_tokens` from `PromptEvalCachedCount` when present. This confirms section 7c.

*Network at import and first use:*

- `__init__.py:411-413` (`model_cost_map_url`, a raw GitHub URL), `__init__.py:554` (`model_cost = get_model_cost_map(url=…)`, run at import) and `litellm_core_utils/get_model_cost_map.py:746`: the local backup is used only if `is_offline_mode()` or `LITELLM_LOCAL_MODEL_COST_MAP` is set. Otherwise it fetches, and on a transient failure starts the thread `litellm-model-cost-map-retry` (line 771).
- `__init__.py:419-421` and `anthropic_beta_headers_manager.py:153`: the beta-header config is fetched lazily, unless offline or `LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS`.
- `litellm_core_utils/offline_mode.py:4-7`: `LITELLM_OFFLINE_MODE=true` is the documented switch for fetches LiteLLM makes "on its own". It also gates the tokenizer download (`utils.py:2909`).
- Confirmed by probe: with every non-loopback lookup refused, a plain `import litellm` attempted `raw.githubusercontent.com` once (and logged a retry). Loading the beta headers attempted it again. With the three variables set, neither did.
- Telemetry settings use the `LITELLM_TELEMETRY_` prefix and live under `proxy/telemetry/` (`settings.py:12`), which the SDK path does not import.

*Claude parameter handling:*

- `llms/anthropic/common_utils.py:859-884` (`_supports_sampling_params`, flag `supports_sampling_params` with a name fallback that matches `fable`) and `887-907` (`_apply_sampling_param`: only `temperature == 1` passes without `drop_params`).
- `common_utils.py:924-945`: `forced_tool_use_unsupported` is `supports_forced_tool_use is False`, so a model missing from the map (`claude-fable-5-2`) is not protected.
- Map flags in `model_prices_and_context_window_backup.json`:
  - `claude-haiku-5-5`: `supports_sampling_params: false`, `supports_forced_tool_use: true`.
  - `claude-sonnet-5-5` and `claude-opus-5-5`: both `false`.
  - `claude-fable-5-2`: absent.
- Probe of `AnthropicConfig().map_openai_params`: Fable 5.2 with `tool_choice: "required"` gave `{"tool_choice": {"type": "any"}}`; Sonnet and Opus raised.
- `llms/anthropic/chat/transformation.py:1644-1651`: `reasoning_effort = "none"` maps to no thinking parameter at all (lines 2044-2046 pop `thinking` and `output_config`). The model's default applies, which on Opus 5.5 and Fable 5.2 is adaptive thinking.
- `_is_adaptive_thinking_model` (`common_utils.py:1098-1107`) uses fallback generalisation rules, so Fable 5.2 still gets `thinking: adaptive` plus `output_config.effort`. This was probed.
- `litellm_core_utils/core_helpers.py:312`: `"refusal": "content_filter"`.

**Consequences, implemented in issue 12** (`docs/PROXY.md`):

- `load_litellm` forces `LITELLM_OFFLINE_MODE`, `LITELLM_LOCAL_MODEL_COST_MAP` and `LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS` before import. A test checks that no lookup happens.
- The Anthropic upstream uses `drop_params = ["temperature", "top_p", "top_k"]` and `forced_tool_choice = "auto"`.
- The Ollama upstream set `reasoning_values = {"true" = "medium", "false" = "none"}`. Issue 46 removed the need: the catalog now holds `none` / `medium`.
- Refusals are recorded as the raw `refusal` signal.

**Contradictions with earlier sections.**

| Where | Says | Finding |
|---|---|---|
| Section 4, Anthropic effort mapping (corrected in place by issue 46) | `none` → thinking off | At 1.109.1, `none` sends **no** thinking parameter, so the model default applies (adaptive on Opus 5.5 and Fable 5.2, which cannot turn it off). |
| Catalog `qwen3.5:9b` `effort_map` (fixed by issue 46) | `none = "false"`, `medium = "true"` | Those are native `think` values. Over `/v1` they are invalid and need translating to `none` / `medium`. |
| Section 7c (from the compat docs; corrected in place by issue 46) | GPT-OSS maps `xhigh`/`ultra` to `high` | The `/v1` code aliases them to `max` before model-specific validation. The catalog never sends them for gpt-oss, so this has no effect. |
| Brief for issue 12 | Haiku 5.5 rejects only non-default sampling values | LiteLLM refuses any `temperature != 1` and any `top_p`/`top_k` for Haiku 5.5 client-side (map flag). `drop_params` makes that moot. |

**Confidence.**

- **High** for the OpenAI-compatible path, `cached_tokens` and the network fetches: source plus probes against loopback.
- **High** for the parameter checks as LiteLLM implements them.
- **Not verified:** that Anthropic's API returns 400 for Fable 5.2 forced tool choice. That rests on the brief and on LiteLLM's own comment for Fable 5.1; no call was made.

## 9. Issue 46: Anthropic cache control through LiteLLM, and Ollama `/v1` reasoning values

*Added by issue 46, checked 2026-11-13.* As in section 8, line numbers are in the installed wheel of `litellm==1.109.1` (from `uv.lock`), whose `litellm/` tree matches tag `v1.109.1`. Ollama is the source tree at `v0.40.16` (`d63b7026…`). Each LiteLLM finding was also exercised by calling `litellm.completion` with `anthropic/<model>` against an Anthropic Messages-format mock on loopback (`src/steproute/collectors/mock_upstream.py`, `POST /v1/messages`) and reading the raw request it received. No provider was called and Ollama was not run.

**Answer.**

- **Block-level markers pass through.** An OpenAI-format request reaches Anthropic's Messages API with every `cache_control` it carries: on a system message (message level for string content, block level for parts), on a user or tool message, on a content block, and on a tool definition.
- **Request-level automatic caching passes through.** `cache_control` is a supported OpenAI-side parameter for Anthropic and is copied to the top of the Messages request unchanged. This is the breakpoint the proxy uses for the moving end of the conversation.
- **LiteLLM has its own injection, off by default.** `cache_control_injection_points` (a per-call LiteLLM-only argument) places markers at named positions. Separately, the global `litellm.enable_anthropic_prompt_caching`, which the environment variable `LITELLM_ENABLE_ANTHROPIC_PROMPT_CACHING=true` switches on at import, synthesises two default breakpoints (the system prompt and the last message) unless the request already carries a marker. The proxy forwards neither argument from clients and forces the global flag off, so markers reach Anthropic only when the proxy (or the client) wrote them.
- **Usage maps as section 4 says**, confirmed against the mock: Anthropic's `input_tokens`, `cache_read_input_tokens` and `cache_creation_input_tokens` become `prompt_tokens` = their sum, `prompt_tokens_details.cached_tokens` = the read, and `cache_creation_input_tokens` (top level) and `prompt_tokens_details.cache_creation_tokens` = the write, in both the non-streaming response and the final stream chunk. `normalise_usage` gives back the three counts exactly.
- **Ollama `/v1` reasoning values for `qwen3.5:9b` are `none` (thinking off) and `medium` (thinking on)**, now in the catalog. The native `true`/`false` are wrong on `/v1`: with the model's thinking metadata an unknown name silently resolves to the model's default (thinking on, so `false` turned thinking on), and without metadata it is a 400.

**Evidence: LiteLLM.**

- `llms/anthropic/chat/transformation.py:831`: `"cache_control"` is in Anthropic's supported OpenAI parameters; `2077-2079`: "Pass through top-level cache_control for automatic prompt caching" (`optional_params["cache_control"] = value`).
- `transformation.py:2160-2161` (string system content: the message-level `cache_control` is copied onto the text block) and `2180-2181` (list content: per block).
- `transformation.py:1177-1190`: a tool's `cache_control`, at the top level of the tool or inside `function`, is copied to the Anthropic tool (except tool-search tools).
- `litellm_core_utils/prompt_templates/factory.py:3119-3130` (user string content: message-level marker onto the text block), `1877-1885` (text parts), `1934` and `1962-1963` (tool results), `2239-2252` (`add_cache_control_to_content`).
- `integrations/anthropic_cache_control_hook.py:70`: `MAX_CACHE_CONTROL_BLOCKS = 4`. `1241-1287` (`get_default_injection_points`): returns `[]` unless `litellm.enable_anthropic_prompt_caching` or the per-request `enable_prompt_caching` is true, or when the request already has a marker; otherwise one point on the system message and one on message `-1`. `1338-1391` (`maybe_seed_default_injection_points`), called from `main.py:756` (`acompletion`) and `main.py:5831` (`completion`) on every call. `__init__.py:325`: `enable_anthropic_prompt_caching` is read from `LITELLM_ENABLE_ANTHROPIC_PROMPT_CACHING` at import. `utils.py:5976`: `cache_control_injection_points` is a LiteLLM-only parameter.
- Probe, `anthropic/claude-haiku-5-5` with a system message carrying a message-level marker, a tool call and its result, `cache_control={"type": "ephemeral"}` and `reasoning_effort="low"`: the request went to `/v1/messages` with `"system": [{"type": "text", "text": …, "cache_control": {"type": "ephemeral"}}]`, top-level `"cache_control": {"type": "ephemeral"}`, unmarked messages and tools, `"thinking": {"type": "adaptive", "display": "summarized"}`, `"output_config": {"effort": "low"}` and `"max_tokens": 128000`. The mock's usage `{input 50, cache_creation 1000, cache_read 2000}` came back as `prompt_tokens=3050, cached_tokens=2000, cache_creation_tokens=1000`.
- Probe, `anthropic/claude-sonnet-5-5` with `reasoning_effort="none"`: no `thinking` and no `output_config` in the request (confirms the section 4 correction).

**Evidence: Ollama** (`ollama/ollama` at `v0.40.16`).

- `openai/openai.go:587-618` (`ThinkingFromReasoningEffort`): `"none"` → `think = false`. Otherwise the value is aliased (`minimal` → `low`, `xhigh`/`ultra` → `max`) and checked by `api.ValidateLegacyThinking` (`api/types.go:1254-1264`: `low`, `medium`, `high`, `max`). With valid thinking metadata that supports `true` and has no named levels, a valid value becomes `think = true`; an invalid one is passed on unchanged as a name. Without metadata an invalid value is an error (`openai.go:614-616`), which the middleware returns as 400 (`middleware/openai.go:465-469`).
- `model/renderers/renderer.go:73-75` and `model/renderers/qwen35.go:363-368`: the `qwen3.5` renderer's thinking values are `[false, true]`. `model/renderers/thinking.go:30-45` (`ResolveThinking`): an unsupported name resolves to the default, which `server/model_thinking.go:80-82` sets to `true`.
- `types/model/thinking.go:100-115`: a named level such as `"medium"` counts as thinking on (`Bool()` is `true` for any string), so `medium` turns thinking on also when the model has no metadata.
- `docs/api/openai-compatibility.mdx:253`: "Boolean-only models that support enabling thinking map recognized compatibility efforts to `true`, while `"none"` requests `false`."

**Consequences, implemented in issue 46** (`docs/PROXY.md`, decision 10):

- The proxy adds two breakpoints to Anthropic requests it rewrites: the static prefix (last system block, or last tool) and the request-level automatic breakpoint. Configurable per upstream (`cache_breakpoints`), off in shadow mode by default, and never added when the client already placed markers.
- The loader forces `litellm.enable_anthropic_prompt_caching = False`.
- A request to Anthropic without any marker records nothing cached in the session state.
- While breakpoints are on, `cache_ttl_s` may not exceed 300 (the 5-minute entry the proxy requests).
- The catalog's `qwen3.5:9b` `effort_map` is `none = "none"`, `medium = "medium"`; the example config and the test harness no longer set `reasoning_values`.

**Found on the way, not changed.**

- Without a client `max_tokens`, LiteLLM fills in the model's maximum (`transformation.py:581-583`, 128000 for the Claude 5.5 models) even on a non-streaming request. Anthropic's SDKs refuse such non-streaming requests client-side; whether the API itself does was not checked. Clients of the proxy should send `max_tokens`, or stream.
- LiteLLM adds `thinking.display = "summarized"` to adaptive thinking on its own. It changes what the response shows, not the cache key between steps (it is the same on every step).

**Confidence.**

- **High** for what LiteLLM sends and how usage maps back: source plus probes through the real `anthropic` provider.
- **High** that the breakpoint placement follows Anthropic's documented rules (section 7a and the prompt-caching guide: tools → system → messages, automatic caching on the last cacheable block, 20-block lookback, 4 breakpoints). **Not verified** against the live API: that the automatic breakpoint finds the previous step's entry in practice, and the exact counts Anthropic reports.
- **High** for the Ollama values from source and docs. **Not verified** by a run: Ollama is not installed here.

## 10. Issue 17: SWE-bench harness, arm64 images, public per-instance results

*Added by issue 17, checked 2026-11-13.* These findings come from the harness source on GitHub (`SWE-bench/SWE-bench` main at `02e7a74f`, PyPI `swebench` 5.0.2 of 2026-08-18), the Hugging Face dataset API, Docker Hub and GHCR listings, and the `SWE-bench/experiments` repository at `40f164d5`. Nothing was pulled, built or run, and no Docker daemon was started. The task-set rationale is in `docs/TASKSET.md`.

**Answer.**

- **The harness takes the image from the dataset row.** `make_test_spec` sets `image=instance["image"]`, and `--dataset_name` accepts a local `.json` file. A one-row dataset with `image` rewritten therefore makes the harness use a native arm64 image without any harness flag. `SwebenchBackend` does this.
- **Verified rows are self-contained since 2026-08-16.** HF `SWE-bench/SWE-bench_Verified` revision `78f471bf` carries `image` (always `swebench/sweb.eval.x86_64.*`), `eval_script`, `log_parser` and `eval_type`. Harness 5.0 no longer needs a Dockerfile checkout to evaluate.
- **arm64 images exist for 420 of 500 Verified tasks.**
  - 281 are official, `swebench/sweb.eval.arm64.<owner>_1776_<repo>-<n>`, pushed 2025-04-20 (about 1.2 GB compressed for the one sampled).
  - 420 are Epoch AI's, `ghcr.io/epoch-research/swe-bench.eval.arm64.<instance_id>:latest`. Epoch's README says: "provided on a best-effort basis, and have not been tested".
  - The official set is a subset of Epoch's. Neither has xarray. The harness README calls arm64 support "experimental".
- **Report layout differs by version.** Per-instance `report.json` is under `logs/run_evaluation/<run_id>/<model>/<id>/` in 5.0.x, and under `logs/evaluation/...` on unreleased main. The backend reads both.
- **No report means no grade.** On a patch that does not apply, or a test timeout, the harness writes no `report.json`. It only logs `>>>>> Patch Apply Failed` or `Test timed out after N seconds.` to `run_instance.log`. Reports carry `infra_failure` since 5.0.
- **The harness skips empty patches**, counting them as unresolved.
- **Public per-instance results are free to read.** `evaluation/verified/<submission>/per_instance_details.json` gives `{id: {resolved, api_calls, cost}}` for recent submissions. Older ones have `results/results.json` lists. Public trajectories with per-call usage, cache reads and writes included, are at `https://swe-bench-submissions.s3.amazonaws.com/bash-only/<submission>/trajs/<id>/<id>.traj.json`, about 0.5 MB each. The experiments repository has no licence file.
- **mini-SWE-agent's SWE-bench config** has `step_limit: 250`, `cost_limit: 3.` and a 60 s timeout per command, with Docker environments. The config does not say how Anthropic cache breakpoints are set. Public Claude trajectories nevertheless show 84-97% of input read from the cache.

**Evidence.**

- `swebench/harness/utils.py` `make_test_spec` (image from the row) and `load_swebench_dataset` (`.json` / `.jsonl` / `.parquet` paths).
- `swebench/harness/run_evaluation.py`: `GIT_APPLY_CMDS` (`git apply --verbose`, `--3way`, `--reject`, then `patch --batch --forward --fuzz=5 -p1 -i`), the `EvaluationError` on patch failure and timeout, and `report.json` written only after grading.
- `swebench/harness/constants/__init__.py`: `APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"`, `LOG_REPORT`, `LOG_INSTANCE`, `RUN_EVALUATION_LOG_DIR = Path("logs/evaluation")` on main; the CHANGELOG lists that rename as unreleased.
- `swebench/image_builder/image_spec.py`: `arch` is `amd64` or `arm64` (`linux/arm64/v8`).
- Docker Hub `hub.docker.com/v2/repositories/swebench/?name=sweb.eval.arm64.<prefix>`. Anonymous paging stops after 100, so the listing was split by prefix: 281 in all.
- GHCR: an anonymous token, then `HEAD /v2/epoch-research/swe-bench.eval.arm64.<id>/manifests/latest` for each of the 500 ids: 420 returned 200. The check is reproduced by `scripts/select_taskset.py`.
- `config/benchmarks/swebench.yaml` in `SWE-agent/mini-swe-agent` main.

**Consequences for issue 18.**

- The agent also runs inside the task container (mini-SWE-agent's Docker environment). Docker is therefore needed for the agent loop, not only for the check, and the arm64 image choice applies to both.
- Verify that Anthropic cache breakpoints reach the provider through the proxy before the pilot. Without them the estimate triples (`steproute tasks estimate --no-cache`). LiteLLM's `cache_control_injection_points` is a fallback (`docs/PROXY.md`, Limitations).
- Configuration-major, cheapest-first order touches every task image in every month. Either keep about 150 GB of images or pull per batch.
- Validate each image with the gold patch, which must pass, and the empty patch, which must fail, before any paid run on it: `steproute tasks check <id> --patch gold|empty`.

**Confidence.**

- **High** for the harness behaviour: source read at a named commit.
- **High** for the image counts: listed directly.
- **Low** for whether the April 2025 official arm64 images and Epoch's untested builds grade like the x86_64 images with the August 2026 `eval_script`. Nothing was run.

*Tested on 13 Nov 2026 (issue 56, section 14):* the harness 5.0.2 runs on this Mac against the arm64 image named in a one-row dataset, as described above; the per-instance report is under `logs/run_evaluation/` (5.0.x layout); the empty patch is indeed skipped without a container. For the 20 pilot tasks, 19 images grade correctly with the August 2026 `eval_script` (gold passes, a no-change patch fails), including all 5 Epoch images; `psf__requests-1921` fails even with the gold patch.

## 11. Issue 18: mini-SWE-agent as the replay scaffold, and two proxy findings

*Added by issue 18, checked 2026-11-13.* mini-SWE-agent was read from the PyPI wheel `mini_swe_agent-2.4.6-py3-none-any.whl` (uploaded 2026-07-23; sha256 `a35463c553ac825c7773b03cfa69cd44958e3af20155dcc5711fdf9e4c67cd54`), unpacked into a scratch directory and never installed or run. Line references are to files in that wheel. The two proxy findings were reproduced against the loopback mock with the pinned `litellm==1.109.1`. No provider was called.

**Answer.**

- **mini-SWE-agent fits the step shape but not the runner**, so the replay runner uses its own loop with the same shape (`docs/REPLAY.md`, decision 1). Licence MIT (PyPI metadata; copyright line names the two authors). Not added as a dependency.
- **LiteLLM 1.109.1 invents zero usage.** For an OpenAI-compatible response without a `usage` object, `litellm.completion` returns `usage = {prompt_tokens: 0, completion_tokens: 0, total_tokens: 0}`. Before this issue the proxy priced that at $0. Fixed: `normalise_usage` treats a 0-token prompt as no usage, the outcome gets `cost_usd = null` and `signals.usage_missing = true`, and the service no longer fills in a $0 cost for it.
- **The proxy's path scan backtracked exponentially.** `_PATH_TOKEN` in `collectors/proxy/session.py`, `(?:~|\.{0,2})?(?:/?[\w.@+-]+)*/[\w.@+-]+|...`, splits a long word without a slash in exponentially many ways: 22 characters took 0.6 s, 24 took 2.5 s, and the 37-character `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` in the scaffold's instructions hung the proxy indefinitely. Any real coding conversation has such identifiers. Fixed with an equivalent unambiguous pattern (every segment after the first follows a `/`), which matched exactly the same tokens on the test strings; regression test `test_path_scan_is_linear_on_long_identifiers`.

**Evidence: mini-SWE-agent 2.4.6.**

- *Endpoint.* `models/litellm_model.py`: `LitellmModel._query` calls `litellm.completion(model=..., messages=..., tools=[BASH_TOOL], **model_kwargs)`, so a custom OpenAI-compatible endpoint is `model_kwargs.api_base`. The module does `import litellm` at import time.
- *Actions.* Structured tool calls by default: `models/utils/actions_toolcall.py` defines one `bash` tool and parses `tool_calls` (unknown tool, bad JSON or missing `command` is a `FormatError`). A text-parsing variant exists (`litellm_textbased_model.py`, a regex over fenced blocks).
- *Commands.* `environments/local.py`: `subprocess.Popen(command, shell=True, cwd=..., env=os.environ | self.config.env)`, so a local run passes the **whole host environment**, API keys included, to the model's commands. `environments/docker.py`: `docker run -d` then `docker exec`, forwarding only `env` and `forward_env`.
- *Submission and limits.* `LocalEnvironment._check_finished`: the first output line `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` with exit code 0 ends the run, the rest is the submission. `agents/default.py`: `step_limit`, `cost_limit` (checked before each query, "exceeding (!)"), `max_consecutive_format_errors = 3`. `config/benchmarks/swebench.yaml`: step limit 250, cost limit 3, command timeout 60 s, observations above 10,000 characters cut to 5,000 head and 5,000 tail.
- *Cost tracking.* `LitellmModel._calculate_cost` uses `litellm.cost_calculator.completion_cost` and raises `RuntimeError` when the cost is not positive unless `cost_tracking = "ignore_errors"`, which then records 0. Behind our proxy every model name is unknown to LiteLLM's map, so its `cost_limit` would be inert.
- *Retries.* `models/utils/retry.py`: tenacity, `stop_after_attempt(MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT or 10)`, `wait_exponential(min=4, max=60)`, on every exception except authentication, not-found, permission, context-window, unsupported-parameter errors and `KeyboardInterrupt`.
- *Cache markers.* `models/utils/cache_control.py`: with `set_cache_control: "default_end"` a marker goes on the last message only; the default is `None` (no markers). Unset, our proxy places its own breakpoints (`docs/PROXY.md`, decision 10); set, the proxy would leave the client's single marker alone.
- *History.* `DefaultAgent.run` and `add_messages`: a linear message list that only grows; nothing is condensed.
- *Dependencies* (wheel metadata): `pyyaml`, `requests`, `jinja2`, `pydantic>=2`, `litellm>=1.75.5` (excluding 1.82.7 and 1.82.8), `tenacity`, `rich`, `python-dotenv`, `typer`, `platformdirs`, `textual`, `prompt_toolkit`, `datasets`, `openai`.

**Consequences, implemented in issue 18.** `steproute.replay.agent.BashAgent` keeps the step shape (linear history, one `bash` tool, 60 s commands, 10,000-character observations, the submission marker, 250 steps, three format errors); the runner owns cost, limits and retries; the workspace passes no host variables to commands; the proxy places the cache breakpoints. The proxy gained `[proxy] on_no_decision` (`client_model` or `error`), which the runner sets to `error`.

**Confidence.**

- **High** for what mini-SWE-agent 2.4.6 does: source read from the published wheel.
- **High** for the two proxy findings: reproduced and covered by tests.
- **Not verified:** how the shorter task message of our loop changes model behaviour; the pilot measures steps and cost directly.

*Tested on 13 Nov 2026 (issue 56, section 14):* the loop ran three times on `qwen3.5:9b` through the proxy to Ollama's `/v1`, with commands in the real task container: one run solved its task (graded `passed` by the harness), one hit the step limit, one hit the 32K cap. Structured tool calls arrived on every step; the model never needed the text fallback.


## 12. Issue 14: the Claude Code mod, checked against the docs and in real sessions

*Added by issue 14, checked 2026-11-13.* Claude Code **2.1.323** (`claude --version`). The mod type declarations that ship with that build (`claude-code.d.ts`) were used alongside the docs. Docs read the same day: https://code.claude.com/docs/en/plugins/mods/overview, `/plugins/mods/events`, `/plugins/mods/reference`, `/plugins/components`, `/plugins/manifest-reference`, `/sub-agents`. Three real sessions were run (below). The plugin is `plugins/claude-code`, documented in `docs/CLAUDE-CODE.md`.

**Answer.**

- **Sections 1 and 2 hold for 2.1.323.** `agent.spawn` with `next({ ...e, model })` sets a starting subagent's model, and a changed model on a resumed run (`e.resume`) is ignored. `turn.step` carries `e.agentId` for subagent requests, and its result's `usage` has the four token fields plus the answering `model`. Mods need v2.1.287+ and run in `claude -p`. Subagents inherit MCP tools; plugin agents ignore `hooks`, `mcpServers`, `permissionMode` (and `initialPrompt`).
- **Observed in route mode: the recommended model ran.** Claude Code's own `--output-format json` `modelUsage` listed `claude-sonnet-5-5` for exactly the three subagents' tokens while the main session ran on Haiku, and the plugin's per-subagent sums matched it to the token.
- **New facts the spike did not record**, each used by the plugin:
  - `agent.spawn`'s result is `{ model, agentId }`, the model as core resolved it, and the subagent's end is its own `turn.complete` with `agentId` and the run's summed `usage`: a second, per-run usage source next to `turn.step`. In all three sessions the two agreed exactly.
  - "With nothing configured, a subagent that Claude spawns from your main conversation starts in the background" (sub-agents page). All nine subagents in the sessions ran in the background; their `turn.step` and `turn.complete` events arrived as for foreground ones.
  - The mod validator (`claude plugin validate`) refuses a module that passes `$` as a value: "$ is always spelled $.noun.event(...) at the call site". `$.env.get` takes literal names only.
  - `$.http.fetch` gives up only after 30 seconds, and a hook's 10-second budget does not count time inside `$` calls (reference, Limits), so a client must race its own timer.
  - Per the declarations, "the hooks an agent's spawn was nested in do not hear its start, as at its `turn.step`s". The plugin's `tool.call` hook therefore excludes `Agent`, `Task`, `SendMessage`, `Workflow` and `TeamCreate` by matcher, so no spawn happens inside it. With that, the plugin's `turn.step`, `tool.call` and `turn.complete` hooks all saw the subagents' events.
- **Correction to section 1's consequence 5 ("no model owns a cache").** A subagent's first request does not read the *parent's* cache, but it can read the cached system prompt and tool definitions that earlier subagents or sessions wrote on the same model: the first subagent request read 4,284 cached tokens (Haiku, run 1) and 4,256 and 7,922 (Sonnet, runs 2 and 3). The plugin still sends `cached_prefix_tokens = 0`, which can only overstate a spawn's cost.
- **Effort.** In both route runs the engine's effort for the Sonnet subagents' requests was already `medium` (`signals.engine_effort`), the decision's effort, so the `turn.step` effort rewrite never had to act; it is covered by unit tests only.

**Real sessions.** Command shape (`scripts/claude_code_session.sh`), run in a fresh scratch directory outside the repository holding three one-line text files, with a fresh `steproute serve --config plugins/claude-code/service/steproute.toml --port 0 --store <scratch>/traces`:

```
env -i HOME=... USER=... PATH=... STEPROUTE_MODE=<mode> STEPROUTE_URL=<service> \
  claude -p "<start three general-purpose subagents, one per file, no model parameter; each reads its file and summarises it in one sentence>" \
  --plugin-dir <scratch>/plugin --model haiku --tools Agent,Read --allowedTools Agent,Read \
  --strict-mcp-config --no-session-persistence --output-format json --debug-file <scratch>/debug.log
```

No API key was in the environment; the sessions ran on the subscription login. Each exited 0.

| Run | Mode | Decisions / outcomes / joined | Recommended | Ran (API-reported) | Tokens per subagent (in / out / cache read / cache write) | Claude Code `modelUsage` |
|---|---|---|---|---|---|---|
| 1 | shadow | 3 / 3 / 3 | `claude-sonnet-5-5` medium | `claude-haiku-5-5` (not applied) | 4/158/4284/5447, 4/158/7984/1747, 4/159/7984/1742 | haiku only |
| 2 | route | 3 / 3 / 3 | `claude-sonnet-5-5` medium | `claude-sonnet-5-5` | 4/160/4256/5394, 4/159/7933/1717, 4/150/7933/1712 | sonnet 12/469/20122/8823 (= the three sums); haiku for the main session |
| 3 | route (committed code) | 3 / 3 / 3 | `claude-sonnet-5-5` medium | `claude-sonnet-5-5` | 4/160/7922/1706, 4/159/7922/1706, 4/150/7922/1701 | sonnet 12/469/23766/5113 (= the sums) |

Every outcome: two requests per subagent counted via `turn.step`, `turn_usage_matches = true`, one `Read` call, no tool errors, `success = null` (nothing was checked), `cost_usd = 0`, `end_reason = answer`. `steproute traces stats`: `decisions: 3, outcomes: 3, joined: 3, join rate: 100.0%, total cost: $0.0000` for each run. Acceptance criterion 2 passed in all three runs.

**Delay per spawn.** `/decide` measured inside the hook (`signals.decide_ms`): 1 to 5 ms over the nine spawns; Claude Code's debug log timed the `$.http.fetch` calls at 2 to 3 ms. The engine's timing of the whole `agent.spawn` hook ("settled in", worker hop and the engine's own spawn included) was 7.8 to 25.6 ms, the first spawn of a session the slowest. `turn.complete` with the `/report` call: 2 to 21 ms.

**Install timing (criterion 1).** From the committed tree unpacked into an empty scratch directory, with an empty uv cache and uv's tool directories pointed into the scratch directory: `uv tool install --from . steproute` 1 s (the core has no dependencies), `claude plugin validate` 1 s, the service up in under 3 s; under 10 s in all, plus the clone. Not covered, because it would change this machine's setup: a network clone, installing Claude Code and `uv`, signing in, and the marketplace install (`/plugin install steproute --marketplace muuki2/steproute`), which was validated (`claude plugin validate .` passes) but not run.

**Not done or not verified.**

- The `PreToolUse` `updatedInput` fallback (section 1, consequence 2) was neither shipped nor tested (`docs/CLAUDE-CODE.md`, decision 9); its confidence stays medium.
- The plugin was not run on a Claude Code older than 2.1.323, in the Desktop app or in VS Code, and its status line and `/steproute` were seen only in the debug log (`$.ui.status ... no status row in a headless session; kept here`), since all runs were headless.
- Resumed runs, aborted runs, test commands, tool errors and the effort rewrite are exercised by unit tests, not by a real session.

**Confidence.** High for the mechanism (docs, this build's declarations, and three sessions whose token counts match Claude Code's own report). Medium for anything that depends on the agent type list and the background default, which the docs say change between versions.

## 13. Issue 15: the dsh plugin, checked against the pinned source and run in-process

*Added by issue 15, checked 2026-11-13; the CLI sessions were added the same day, once a local Node 24 was approved.* Source read in the npm packages of `@deepseek-ai/dsh@0.2.2-rc.3` and its exact-version `@deepseek-ai/dsh-*` closure, installed with `npm ci --ignore-scripts` into `tests/dsh/pinned/` (lockfile there; tarball integrity `sha512-S+4oUYVK…qRg==` for `@deepseek-ai/dsh`). Every package ships its `src/`; the paths below are `<package>/src/<file>:<line>` in those tarballs, which correspond to `packages/**/<package>/src/` at tag `dsh-v0.2.2-rc.3`. Where section 3 cites the same lines at HEAD, the tarball agrees with its "one lower at the tag" note (`dsh-agent/src/model-selection.ts:112-114` for `select`). The plugin is `plugins/dsh`, documented in `docs/DSH.md`.

**Answer.**

- **Section 3's mechanism holds, and the hook point is now settled.** `agent.model.select()` called inside `tools/post-execute` changes the model and effort of the next step. In a session on the real dsh libraries (below) the `model/selection` event landed between the batch's `tool/call` and `tool/result`, the next request reached the adapter with the selected provider, model and effort, the `[model changed …]` notice was injected, and the logged `assistant/message.message.source` named the selected route.
- **The first step of a turn cannot be routed.** `preStep` reads `this.model.current()` as its first act (`dsh-agent-loop/src/agent.ts:294`), and the turn reaches it from `turn/start` with nothing awaited in between (`agent.ts:333-353`). The plugin logs a decision for that step and never applies it.
- **Only the last call of a tool batch should ask.** Calls of a batch commit in model order; each call's post-execute runs just before its own `tool/result` is appended (`dsh-tools/src/batch.ts:139-150`, `pipeline.ts:343-361`), and calls that fail before dispatch skip post-execute altogether (`pipeline.ts:166-178, 230-232`). So the last call's post-execute is the one point that sees every result of the batch and still precedes the next step's read.
- **The `session/event` feed cannot call `select()`.** A listener runs inside the append's publication and a nested append throws "session append cannot reenter while another append is being published" (`dsh-session/src/index.ts:957-962`); the feed is fire-and-forget (`index.ts:66-79`). The plugin only observes it.
- **The plugin writes no session event of its own.** Its only write is `select()`, which appends dsh's own `model/selection`. (Plugin records, if ever needed, are `plugin:<name>` types that dsh appends with `ignorable: true` itself, `dsh-session/src/index.ts:929-946`.)
- **New facts the spike did not record**, each used by the plugin:
  - The DeepSeek API route is provider **`deepseek-official`** (`dsh-llm-deepseek-api-key/src/index.ts:20`), and the default selection is `deepseek-official`/`deepseek-flash` (`dsh-base/cordis.patch.yml:90-94`). Its efforts are `off|low|high|max` (`dsh-llm-deepseek/src/model-info.ts:6-31`); steproute's `none` maps to `off`.
  - **dsh 0.2.2-rc.3 requires Node 24.9.0 or later** (`engines.node` of `@deepseek-ai/dsh`), and its launcher runs only under `import.meta.main` (`dsh/src/bin.ts:76`), which Node 22.15 lacks. The CLI sessions below ran on Node v24.22.3.
  - **dsh's logger shows nothing in a headless run.** No exporter is mounted there, so a plugin's `ctx.logger(...).info` lines are not on stderr and no log file is written under `DSH_HOME` (seen in CLI session 1). The plugin therefore has an opt-in, `STEPROUTE_STATUS=stderr` or `statusToStderr: true`, that also writes its load line and its shutdown summary to stderr; sessions 2 to 5 show both lines.
  - The headless persona names the model: "You are a coding agent powered by the {{model}} model." (`dsh-headless/cordis.patch.yml:12`), and the headless runner installs the model-switch notice (`dsh-headless/src/index.ts:383`). A switch therefore changes the system prompt as well as adding the notice.
  - `engines.dsh` decides compatibility alone when present; a mismatched bundle is skipped and dsh starts without it (`dsh-app-boot/src/profile.ts:699-705`). The plugin pins it to exactly `0.2.2-rc.3`.
  - A plugin needs no build and no dsh import: a module exporting `name` and `apply` loads (`dsh/docs/user/develop/basic/publish.md:46-54`), and a config schema is any Standard Schema object (`cordis/src/fiber.ts:50-61`).
  - `ctx.llm.listProviders()` lists the routes that have an adapter (`dsh-llm/src/index.ts:493-500`). The plugin applies no decision to an unlisted route, since a step on a route with no adapter fails.
  - Shell tool results: a non-zero exit is not an error. It ends the model-facing text with `[exit code: N]`, and stopped, timed-out or backgrounded runs end with other bracketed markers (`dsh-tool-bash/src/render.ts:17-66`). Tool names in the default bundle: `read`, `write`, `edit`, `apply_patch`, `read_image` (`dsh-tool-fs`), `grep`, `glob` (`dsh-tool-fs-search`), `bash`, `pwsh`, `web_search`, `web_fetch`.
  - An approval answered `rejected` becomes a tool error with code `APPROVAL_REJECTED` and the text "the user rejected tool …" (`dsh-tools/src/pipeline.ts:466-468`); cancellations and missing answerers have their own codes (`dsh-tools/src/error-codes.ts`).
  - The non-interactive entry is `dsh headless "<task>"`, short for `dsh --profile headless`; launcher flags follow the profile name (`dsh/src/args.ts:90-99, 205-208`). There is no `-p`.
  - A subagent's session header has `origin: 'subagent'` and `parentSession` (`dsh-tool-subagent/src/model-selection-settings.ts:63`).

**Seams the plugin relies on.**

| Seam | Kind | Where (0.2.2-rc.3) |
|---|---|---|
| `agent/created` `{ agent, source }` | serial | `dsh-agent/src/runtime-types.ts:291` |
| `agent/disposed` `{ agent }` | emit | `runtime-types.ts:300` |
| `agent/pre-step` `{ agent, messages, turn, step, signal, modelSelection }` | waterfall | `runtime-types.ts:351`; called at `dsh-agent-loop/src/agent.ts:302-305` |
| `agent.model.current()` / `agent.model.select({ provider, model, reasoningEffort })` | method | `runtime-types.ts:61,71`; `dsh-agent/src/model-selection.ts:105-115` |
| `tools/post-execute` `(exec, result, next)` | waterfall | `dsh-tools/src/events.ts:41`; run at `pipeline.ts:494-497` |
| `session/event` `(session, event)` | emit | `dsh-session/src/index.ts:79` |
| `session/flush` `(session)` | parallel, awaited | `index.ts:87`; awaited by the headless runner before exit, `dsh-headless/src/index.ts:435` |
| `ctx.llm.listProviders()` | method | `dsh-llm/src/index.ts:498` |
| events `turn/start`, `turn/end`, `step/start`, `step/end`, `assistant/message`, `assistant/attempt`, `tool/call`, `tool/result`, `request/header` | log | `dsh-session/src/types.ts:294,303,305,307,349,363,369,383,398` |
| `approval/asked`, `approval/decided` (`allowed-once`, `rejected`, `cancelled`, `unavailable`) | log | `dsh-user-approval/src/types.ts:32,44,55` |
| `llm/retry`, `model/selection` | log | `dsh-llm-retry/src/types.ts:9`; `dsh-agent/src/types.ts:121` |
| `TokenUsage` (disjoint counts) | type | `dsh-llm/src/types.ts:174-196` |

Not used, and why: `llm/stream` (requests deep-frozen, `dsh-llm/src/index.ts:64-76`); `agent/request` (deprecated, throws on a model or effort change, `runtime-types.ts:363-374`, `agent.ts:641-646`); `agent/turn-stopping` (restoring the user's model there was tried and doubled the switches per turn); the compressed log file (the live feed carries the same events).

**Outbound paths in the headless profile, and how the test sessions switch them off.** Read in the same tarballs; each is on by default unless said otherwise. The switches are in the patch `scripts/dsh_session.sh` writes, plus the environment it sets. In the CLI sessions they were not relied on alone: dsh ran inside `sandbox-exec` with a profile that blocks every outbound connection except to loopback (below), and no request the mock received carried `dsh_session_log` or `dsh_plugin_packages`.

| Path | Where | Switched off by |
|---|---|---|
| Model requests of the DeepSeek route (`POST <baseURL>/v1/messages`, images to `/v1/files`) | `dsh-llm-deepseek/src/adapter.ts:142-153`, `config.ts:116-119, 311` | `llm-deepseek` row `baseURL` set to the mock upstream on loopback; the key is read from `STEPROUTE_MOCK_UPSTREAM_KEY`, a placeholder |
| Headers on those requests: anonymous user id, session id, user agent | `adapter.ts:145-152`, `dsh-llm/src/attribution.ts:55-90` | cannot be switched off; they go to the loopback mock, and the id file lives in the scratch `DSH_HOME` |
| Session log in each request body (`dsh_session_log`) | `dsh-base/cordis.patch.yml:43-44`, `dsh-session-log-deepseek/src/index.ts`, merged at `dsh-llm-deepseek/src/request-extensions.ts:42` | `session-log-deepseek: disabled: true` |
| Plugin names and versions in each request body (`dsh_plugin_packages`) | `cordis.patch.yml:85-86`, `dsh-plugin-package-inventory-deepseek/src/index.ts:33,205` | `plugin-package-inventory-deepseek: disabled: true` |
| `/models` listing fetch | `dsh-llm-deepseek-api-key/src/index.ts:47-52` | made only for the official origin; the mock's origin is not |
| Session title from a second model call | `cordis.patch.yml:67-77` | `session-title-llm: disabled: true` |
| OTLP session telemetry to `dsh-otel-collector.deepseeksvc.com` (on feedback events only) | `cordis.patch.yml:205-232` | `session-telemetry-otel: disabled: true`, `DSH_TELEMETRY_DISABLED=1`, `DSH_TELEMETRY_MODE=DISABLED`, and `DSH_TELEMETRY_OTLP_URL` pointed at a dead loopback port |
| `web_search` (the query and the user's question) and `web_fetch` (any URL) | `cordis.patch.yml:484-505` (rows at 490-501), `dsh-web-deepseek-api/src/index.ts:73` | `tool-web`, `web-search-deepseek`, `web-fetch-http`, `web-deepseek-api`: `disabled: true` |
| Shell tool (the model could run `curl`; the sandbox limits writes, not network) | `cordis.patch.yml:281-283` | `tool-bash`, `tool-jobs`: `disabled: true` (the scripted model never asks for a shell) |
| `llm-pi-ai` routes, the account route, the plugin manager (pnpm, registry) | `cordis.patch.yml:16-22, 120-136, 549-550` | `llm-pi-ai`, `llm-deepseek-account`, `plugin-manager`: `disabled: true` (headless already disables `deepseek-account`) |
| Compaction (a summary model call on the session's own route) | `cordis.patch.yml:355-361` | left on: it would call the loopback mock, and the session stays far below its threshold |

No update check, remote config or feature-flag fetch was found at startup. Not mounted in headless: product telemetry and analytics (desktop only), the web server, the MCP client, and the optional bundles (translation, speech model download, plugin market, forge, liveshare, auto-review). HOME, `DSH_HOME` and `DSH_AGENTS_HOME` (default `~/.agents`, read for skills and `AGENTS.md`/`CLAUDE.md`) point into the scratch directory (`dsh-home-paths/src/index.ts:12-27`); dsh refuses `DSH_*` names from `.env` files (`dsh-app-boot/src/index.ts:158`).

**Sessions run.**

- **Through the `dsh` command line: five sessions**, after one refused attempt on Node 22.15 (`dsh 0.2.2-rc.3 needs Node 24.9 or later (found v22.15.0); not running`, exit 3). They are described under "CLI sessions" below. Acceptance criteria 1 and 2 pass through the CLI.
- **In-process sessions on the real dsh libraries: eight.** Five ran with `scripts/dsh_inprocess_session.sh` and three by hand with the same harness, all on Node 22.15.0, below dsh's stated minimum; the core libraries loaded and ran. The harness (`tests/dsh/pinned/inprocess_session.mjs`) mounts `dsh-session`, `dsh-session-projection`, `dsh-agent`, `dsh-llm`, `dsh-system-prompt` (with the headless persona), `dsh-tools`, `dsh-tool-executor-native` and `dsh-agent-loop` from the pinned install. Next to them it mounts a scripted adapter registered as `deepseek-official` (efforts `off|low|high|max`), one `read_note` tool, and `plugins/dsh/index.js`. It creates the agent with `installModelSelection`, as the headless runner does, runs two turns of three steps (two single-call tool batches, then an answer), and flushes. Nothing but the plugin's loopback calls leaves the process, and no key exists. The first hand run failed at the first request because the scripted adapter declared no efforts ("does not support reasoning effort \"high\""); the plugin was not involved. Command shape:

  ```
  scripts/dsh_inprocess_session.sh shadow|route <scratch> [KILL_AFTER_REQUEST|none]
  # steproute serve --config plugins/dsh/service/steproute.toml --port 0 --store <scratch>/traces
  # (cd tests/dsh/pinned && env -i HOME=<scratch>/home PATH=... node inprocess_session.mjs <url> <mode> [--kill-after-step N --pid PID])
  ```

  The three final runs used the committed code, except for shadow mode's no-wait change, which was made after them and is covered by unit tests:

  | Run | Mode | Service | Ran per request (model, effort) | Decisions / outcomes / joined | Turns |
  |---|---|---|---|---|---|
  | A | route | killed (SIGKILL) as request 3 started | flash high, v4-pro low, v4-pro low, v4-pro low, flash high, flash high | 3 / 1 / 1 | completed, completed |
  | B | route | up | flash high, then v4-pro low ×5 | 6 / 6 / 6, join rate 100 % | completed, completed |
  | C | shadow | up | flash high ×6 | 6 / 6 / 6, join rate 100 % | completed, completed |

  Every decision was `deepseek-v4-pro` `low`: the rule baseline's default tier 5, with `medium` moved to DeepSeek's nearest effort, `low`.

  In B, the two turn-start steps were logged `turn_start` and not applied. The four steps after a tool batch were applied (`routed`, `selection_ran = true`, `final_model = deepseek-v4-pro`). One `model/selection` was written, and one notice was injected: `[model changed: assistant turns above this point were generated by deepseek-flash; the session continues with deepseek-v4-pro]`. Tokens per outcome are the adapter's scripted counts. For example, step 2 of B has `input_tokens 1000`, `output_tokens 31`, `cache_read_tokens 800`, and the state for it has `prompt_tokens 937`, `cached_prefix_tokens 900`, owner `deepseek-flash`/`high`. The service priced the outcomes from the catalog ($0.0091 for B); the tokens are scripted, so no money was spent.

  In A, the service died during the third request, whose step already had its decision. The fourth step, a turn start, ran on the last selection. At the fifth step the plugin found the service refusing and put the user's `deepseek-flash`/`high` back. Both turns completed. The two outcomes due after the kill were never stored: they were kept in memory and lost at exit.

  Runs 4 and 5 of the eight used an earlier version that restored the user's model at every turn end. It switched model twice per turn, which is why it was dropped (`docs/DSH.md`, decision 1).

**Node 24 used.** `node-v24.22.3-darwin-arm64.tar.gz` from `https://nodejs.org/dist/v24.22.3/` (the current `latest-v24.x`), SHA-256 `ff6e17d4874b76d272212b908cc020d6e815fba5a6c7eaaa06add725840307fd`, which matches the line for that file in the release's `SHASUMS256.txt` (`shasum -a 256 -c`: OK; the file was the same at `latest-v24.x/`). The signature of `SHASUMS256.txt` itself was not checked. Unpacked into a scratch folder and used only by putting its `bin` first on `PATH` for the session commands; the system Node 22 was not touched. With it, `npm ci --ignore-scripts` in `tests/dsh/pinned` (npm 11.19.1, cache and config in the scratch area) installed the 674 locked packages and left the lockfile unchanged.

**The sandbox.** `scripts/dsh-loopback-only.sb`: `(allow default) (deny network-outbound) (allow network-outbound (remote ip "localhost:*"))`. Unix-domain sockets count as outbound too, so the system resolver cannot be asked to look up a name on the sandboxed process's behalf. `scripts/dsh_sandbox_check.sh` proves it before every session, with the same profile and the Node 24 on `PATH`. The first run, by hand, against the mock upstream:

```
ok    curl https://nodejs.org: blocked (curl: (6) Could not resolve host: nodejs.org)
ok    node fetch https://nodejs.org: blocked (ENOTFOUND)
ok    curl http://104.16.213.131 (by address): blocked (curl: (7) Failed to connect to 104.16.213.131 port 80 after 0 ms: Couldn't connect to server)
ok    node fetch http://104.16.213.131 (by address): blocked (EPERM)
ok    curl http://127.0.0.1:49254/v1/messages: reached ({"id": "msg_mock", "type": "message", ...)
ok    node fetch http://127.0.0.1:49254/v1/messages: reached (200 {"id": "msg_mock", ...)
sandbox check: PASSED
control, outside the sandbox: curl http://104.16.213.131/ (Host nodejs.org) -> http 308; curl https://nodejs.org/ -> 307
```

The system log recorded the kernel's denials for the two curl attempts (`Sandbox: curl(…) deny(1) network-outbound /private/var/run/mDNSResponder` and `… network-outbound remote:*:80`), but not for Node's attempts, so an empty denial log is weaker evidence than a failed connection. `scripts/dsh_session.sh` runs the check again before each session (by name and by an address resolved outside the sandbox) and refuses to start dsh if it does not pass; it did pass before all five sessions.

**CLI sessions.** Command shape (`scripts/dsh_session.sh shadow|route <scratch> [KILL_AT_REQUEST|none]`, with the Node 24 `bin` first on `PATH`):

```
cd <scratch>/work && env -i HOME=<scratch>/home DSH_HOME=<scratch>/dsh-home DSH_AGENTS_HOME=<scratch>/agents \
  PATH=<node24>/bin:... TMPDIR=<scratch>/home DSH_TELEMETRY_DISABLED=1 DSH_TELEMETRY_MODE=DISABLED \
  DSH_TELEMETRY_OTLP_URL=http://127.0.0.1:9/v1/logs STEPROUTE_MOCK_UPSTREAM_KEY=placeholder-for-the-mock STEPROUTE_STATUS=stderr \
  /usr/bin/sandbox-exec -f scripts/dsh-loopback-only.sb \
  node tests/dsh/pinned/node_modules/@deepseek-ai/dsh/lib/bin.js headless --patch <scratch>/session.patch.yml --json "<task>"
```

The plugin is loaded through dsh's own loading path: a `- insert:` row in the `--patch` overlay naming `plugins/dsh/index.js` (`--dump-config`, run once in the sandbox beforehand, showed the row as `file:///…/plugins/dsh/index.js` in the overlay's layer). `steproute serve` (with `plugins/dsh/service/steproute.toml`) and the mock upstream (`scripts/dsh_mock_upstream.py`) ran outside the sandbox on loopback. The task asks for `a.txt`, then `b.txt`, then a summary; the scripted model answers with one `read` call per step, then text. That makes one turn of three steps. Each session exited 0 with `turn/end` `completed`.

| # | Mode | Kill | Decisions / outcomes / joined | Requests the mock received (model, `output_config.effort`) | Notes |
|---|---|---|---|---|---|
| 1 | shadow | none | 3 / 3 / 3 | flash high ×3 | The mock's `read` calls used `path` instead of dsh's `file_path`: both failed with `INVALID_ARGS`, and the plugin labelled those steps `tool_error` (`success = false`). Mock fixed after this session. dsh's logger lines were nowhere. |
| 2 | route | none | 3 / 3 / 3 | flash high, v4-pro low, v4-pro low | Steps 2 and 3 routed (`applied = true`, `selection_ran = true`). stderr: `steproute 0.1.0: route mode, …` and `steproute: route · 3 decisions · 3 outcomes · 2 applied · 0 restored · step wait median 1.06 ms`. |
| 3 | route | as request 2 arrived, polled every 10 ms | 3 / 1 / 1 | flash high, v4-pro low, v4-pro low | The poll was too slow: the kill landed after the step-3 decision, so no decision failed; the two later outcomes were lost. The kill was then moved into the mock. |
| 4 | route | in the mock, before answering request 2 | 2 / 1 / 1 | flash high, v4-pro low, **flash high** | The decision for step 3 found the service gone, so the plugin put the user's selection back before step 3 (`1 restored`, `service unreachable`). |
| 5 | shadow | none | 3 / 3 / 3 | flash high ×3 | Both reads `completed`; no labelling signal, `success = null`. |

- **Criterion 1 (the plugin loads; decisions and outcomes join; tokens through the real adapter).** Sessions 1, 2 and 5 joined 3 of 3. The token counts came through `dsh-llm-deepseek`'s real Anthropic Messages path: the mock reported `input_tokens = prompt - cached`, `cache_read_input_tokens`, `output_tokens`. dsh logged them as `inputTokens`, `cacheReadTokens` and `cacheWriteTokens: 0` (with `totalTokens`), and the plugin stored them unchanged. Session 5, steps 1 to 3: in 650/800/950, cache read 1000 each, out 25/25/15 (service-priced $0.0008 at catalog rates, scripted tokens, nothing spent). The requests arrived streamed, with `thinking: {type: enabled}` and `output_config.effort`.
- **Route mode, from dsh's own log** (session 2's `session.v4.jsonl.zstd`, read with `scripts/dsh_session_log.mjs`):
  - Step 1 ran `deepseek-official`/`deepseek-flash` under a `request/header` with effort `high`. The one `model/selection` (`deepseek-v4-pro`, `low`) was appended during step 1, between its `tool/call` and `tool/result`.
  - Steps 2 and 3 ran under a new `request/header` `deepseek-v4-pro`/`low`; their `assistant/message.source` is `deepseek-official`/`deepseek-v4-pro`.
  - The log has the notice `[model changed: assistant turns above this point were generated by deepseek-flash; the session continues with deepseek-v4-pro]`.
  - The wire agrees: requests 2 and 3 asked for `deepseek-v4-pro` with `output_config.effort: low`.
- **Criterion 2 (killing the service does not interrupt the session).** Sessions 3 and 4 exited 0 with the turn `completed` and the same final answer as the others. In session 4 the log shows two selections: the plugin's `deepseek-v4-pro`/`low`, then the restored `deepseek-official`/`deepseek-flash` with no effort, the user's selection as `agent.model.current()` gave it at creation. dsh resolved that effort to its adapter default, `high`. The log also has both "[model changed …]" notices.
- **Outcomes due after the kill are lost at exit**, as in-process: they are kept in memory and the process ends. Session 3 lost 2 of its 3 outcomes; session 4 lost 1 of 2 (the step-3 decision was never made, so no outcome was due for it).
- **The flush before the headless exit works with the real runner.** In sessions 2 and 5 the last step's outcome is sent at `turn/end`, moments before the runner flushes and exits, and it was stored both times (3 of 3). The shutdown summary line, printed from the plugin's dispose, counted 3 outcomes.
- **No outbound attempt by dsh was seen.** dsh's stderr held only the plugin's two lines. The JSON run stream showed no errors. The system log's network denials during each session were only the sandbox check's two curl probes. No request reached the mock with the session-log or plugin-inventory field.
- **Not in the in-process runs:**
  - dsh's logger is silent in headless.
  - The read tool's argument is `file_path`.
  - The user's selection at creation can carry no effort (the default model row has none), so a fail-open restore selects the adapter default.
  - A turn-start state underestimates the prompt badly: 269 tokens estimated (the user's text) against 1650 billed, because the system prompt is unseen.
  - The runner writes a local `session/title` (fallback) even with `session-title-llm` off.

**Delay per step.** `signals.plugin_wait_ms` is the time a routed step waited in post-execute. In the CLI sessions it was 1.06 and 1.01 ms (session 2), and the median was 1.16 ms in session 4 (from its summary line); the first `/decide` of a session, at a turn start, took about 14 ms and is not waited for. In run B it was 4.18, 0.85, 2.13 and 1.19 ms, against `decide_ms` of 4.15, 0.95, 2.27 and 1.41 ms. The first `/decide` of a process took about 10 ms (connection and service warm-up), and a turn-start step waits for nothing. Measured on an Apple M-series laptop with the service on loopback. In shadow mode no step waits.

**Not done or not verified.**

- Loading through `dsh plugin add` (a pnpm link into a profile) was not run: it needs pnpm and the plugin manager, which the test patch turns off. Only the `--patch` overlay was used.
- The outbound-path patch was not tested on its own: every session ran inside the sandbox as well, and there no blocked attempt by dsh showed up. Which of the patch's switches matter was therefore not measured.
- Only the headless profile was run; the TUI and web profiles (and the stderr opt-in under a terminal UI) were not.
- Subagents (`dsh-tool-subagent`), approvals, shell test runs, retries and plan mode are covered by unit tests on a fake loop, not by a real dsh run.
- Whether `web_fetch` can reach loopback or private hosts was not settled (the base patch comment and `dsh-base/README.md` disagree); the test patch turns it off.
- `cacheWriteTokens` from the DeepSeek adapter, and the cache effect of an effort change on DeepSeek, were not observed.

**Confidence.** High for the seam names, signatures, event payloads and the ordering facts above: they were read in the tarball sources, exercised in the in-process sessions, and seen again in dsh's own session log in the CLI sessions. High for the headless CLI path with the `--patch` overlay (five sessions). Medium for loading as a bundle through `dsh plugin add`, and for other profiles, which were read, not run.

## 14. Issue 56: the replay path on real infrastructure (harness, images, Docker, Ollama)

*Added by issue 56, run on 13-14 Nov 2026 on the maintainer's M4 Pro.* Unlike every section above, this one installed and ran things: Docker Desktop 28.0.4 (server arm64, 14 CPUs, a VM of about 8 GB), Ollama 0.40.16 started with `OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_MAX_LOADED_MODELS=1`, the SWE-bench harness `swebench==5.0.2` in a project-local environment, and the 20 pilot images. **No paid provider was called by any route**: no key was set in the environment of any command, every plan that ran contained only `ollama` configurations (checked with a dry run first), and every run's `ledger.jsonl` is empty.

**Answer.**

- **The harness works on Apple Silicon**, with native arm64 images, one instance at a time, through `SwebenchBackend` exactly as section 10 describes (a one-row dataset whose `image` is the arm64 one). Installed by `scripts/install_swebench.sh`: Python 3.12 venv at `.cache/steproute/swebench-env`, `swebench==5.0.2` and 79 dependencies at the exact versions in `scripts/swebench-requirements.txt` (`uv pip compile` for `aarch64-apple-darwin`). `steproute tasks check` and the replay config now find it, and the fetched rows, from the project root.
- **19 of the 20 pilot images grade correctly; `psf__requests-1921` does not** (the gold patch fails). The table is in `docs/TASKSET.md` section 8.1. All 5 Epoch images ("not tested" by their authors) grade correctly.
- **An empty patch proves nothing about an image**: the harness skips it without starting a container. Validation needs a patch that changes no code but is not empty; `--patch baseline` adds one inert file.
- **GHCR pulls fail through Docker Desktop on this Mac** with `403 Forbidden` on `https://ghcr.io/token?scope=repository:epoch-research/...:pull`, also with an empty `DOCKER_CONFIG`; an anonymous token request with `curl` returns 200 and the manifests are readable. Docker Desktop routes pulls through its own proxy (`docker info`: `HTTP Proxy: http.docker.internal:3128`) and presents the `ghcr.io` login it has stored, which these public repositories refuse. `scripts/fetch_ghcr_image.py` fetches anonymously and `docker load`s; the stored login was left alone.
- **`DockerWorkspace` behaves as documented**, checked inside the container: a fixed environment and no network.
- **The proxy talks to Ollama's `/v1`, and the local models make structured tool calls.** Ollama's `/v1` reports cached prompt tokens, so the trace store records Ollama prefix-cache reads (unlike LiteLLM's `ollama` provider, summary row 4).
- **Ollama answered one HTTP 500** in 135 agent calls. The runner's call retry handled it, but the proxy then labelled the failed step `success = false` (fixed, below).

**Evidence: the harness and images.**

- `steproute tasks fetch`: the first attempt got `HTTP Error 502: Bad Gateway` from `datasets-server.huggingface.co`, the second wrote 500 rows (9.4 MB). `https://huggingface.co/api/datasets/SWE-bench/SWE-bench_Verified` reported sha `78f471bf655a3137b2e8a75af1501690ec009ec3`, the manifest's pin.
- Run log of `pallets__flask-5014` with the gold patch (`--keep`): `eval.sh` activates the `testbed` conda environment, runs `python -m pip install -e .`, checks out the test files at the base commit and applies the test patch; `report.json` says `resolved: true`, FAIL_TO_PASS 1/1, PASS_TO_PASS 59/59; about 15 s of the 20 s is "Attempting to stop container" until "stopped successfully".
- `psf__requests-1921`, gold: FAIL_TO_PASS 6/6, PASS_TO_PASS 104/107. `test_output.txt`: `platform linux -- Python 3.9.21, pytest-8.3.4`; `test_POSTBIN_GET_POST_FILES_WITH_DATA` fails with `TypeError: 'requests.post(url, data=...)' object (type: <class 'str'>) must be callable` (the string form of `pytest.raises`, gone since pytest 5); `test_mixed_case_scheme_acceptable` and `test_pyopenssl_redirect` fail with `requests.exceptions.SSLError: [SSL: CERTIFICATE_VERIFY_FAILED]` against `https://httpbin.org`. The baseline passes 5 of 6 FAIL_TO_PASS tests (network tests that pass either way).
- `django__django-11728` (Epoch), baseline: FAIL_TO_PASS 1/2 (`test_app_not_found` passes), PASS_TO_PASS 43/44 (`test_builtin_fields` fails). Gold: 2/2 and 44/44.
- Sizes and times: compressed 0.72-1.15 GB per image (`docker image inspect`), 19.1 GB for the 20; `docker system df` grew from 37.55 GB to 52.51 GB of images (15.0 GB). Official pulls took 105-478 s each and Epoch fetches 211-673 s, while the Ollama models downloaded on the same line. One check took 18-85 s.

**Evidence: the task container** (`DockerWorkspace` on `swebench/sweb.eval.arm64.pallets_1776_flask-5014`):

- Started in 0.3 s. `env | sort` printed `BASH_ENV`, the image's conda variables (`CONDA_DEFAULT_ENV=testbed`, `CONDA_PREFIX`, ...), `HOME=/root`, `HOSTNAME`, `TZ=Etc/UTC`, `PATH` (the testbed's), and the runner's fixed set (`LANG`, `LC_ALL`, `PAGER`, `MANPAGER`, `GIT_PAGER`, `PIP_PROGRESS_BAR`, `TQDM_DISABLE`, `PYTHONDONTWRITEBYTECODE`). No host variable, no key.
- Network: `urllib.request.urlopen("https://pypi.org")` raised `URLError: [Errno -3] Temporary failure in name resolution`; `getent hosts pypi.org` found nothing; `/proc/net/dev` lists `lo` and kernel tunnel stubs only, no `eth0`.
- `git log`: the runner's `base` commit on top of the image's `SWE-bench` commit and the task's base commit; after `echo hello > newfile.txt`, `git diff` showed exactly that file. `docker ps` was empty after `close()` and after every run.

**Evidence: tool calls** (`steproute catalog check --model <id>` against `http://localhost:11434/v1`):

| Model | Result | Time (cold load included) |
|---|---|---|
| `qwen3.5:9b` | PASS: `get_weather({"city": "Paris"})` | 7.9 s (runner `mlx`, context 32768 per `ollama ps`) |
| `gpt-oss:20b` | PASS: `get_weather({"city": "Paris"})` | 7.3 s (runner `llamacpp`) |
| `qwen3-coder:30b` | PASS: `get_weather({"city": "Paris"})` | 8.5 s |
| `devstral:24b` | PASS: `get_weather({"city": "Paris"})` | 15.9 s (runner `llamacpp`) |

A final `steproute catalog check` (the catalog's default selection: all four local models) printed PASS for each. No model needed `tools = false`; the catalog is unchanged.

**Evidence: three end-to-end runs** (`steproute replay run --plan <file> --config <file> --execute`, plan files with one block, `configs = [["qwen3.5:9b", "none"]]`; config `max_retries = 0`, `run_timeout_s = 1200`, `[environment] kind = "docker"`, no `[check]`; the dry run listed only `ollama ... free (local)` and no keys):

| Run | Task | Step limit | Result | Steps | Wall time | Prompt tokens (first / last) | Output tokens | Cached share |
|---|---|---|---|---|---|---|---|---|
| 1 | `pallets__flask-5014` | 40 | `step_limit` (the fix was in the workspace, not yet submitted) | 40 | 138 s | 815 / 16,667 | 5,087 | 96% |
| 2 | `pallets__flask-5014` | 100 | `passed` (harness: resolved) | 40 | 149 s | 814 / 17,565 | 3,599 | 96% |
| 3 | `pytest-dev__pytest-5787` (estimated "long") | 250 | `context_limit` ("prompt of about 31723 tokens passes the local cap") | 54 | 232 s | 1,670 / 30,657 | 6,643 | 97% |

- Every run wrote `grid.jsonl` (one run line, `meta.status` as above, `cost_usd = 0.0`), `runs.jsonl` (start and end), `plan.json`, `runs/<id>/result.json`, `trajectory.json`, `workspace.diff` (and `patch.diff`, `check.json` for run 2), and `traces/`: 40, 41 and 54 decisions, all joined (`steproute traces stats`: join rate 100%), every decision `source = "replay"`, policy `replay-fixed`, action `qwen3.5:9b none`, the task label copied onto every step (`replay_status`, `success`). The configuration check (`X-Steproute-Mode/Model/Effort`) passed on every call.
- Nothing paid: `ledger.jsonl` empty in runs 2 and 3 (absent in run 1, the bug below); every outcome's `final_model` is `qwen3.5:9b` and `cost_usd` 0; the only upstream each run's proxy had was `ollama` on `127.0.0.1:11434`.
- Speed: 2.8-3.9 s per step on average including the command, call latency median 2.0-3.3 s, 32-39 output tokens per second over whole-call time.
- Unloading: `ollama ps` was empty after each run (`keep_alive = 0`).
- The context cap: prompt sizes rose monotonically to 30,657 tokens and the run stopped before a call that could pass 32,768 - 2,048; Ollama never truncated a prompt.
- Run 2's patch is byte-identical to the gold patch (blob `1aa82562`): memorisation of the public fix, the contamination `docs/TASKSET.md` section 1 warns about.
- Run 2 printed `upstream 'ollama' failed: 500 InternalServerError` once; the call was retried after 5 s and succeeded.

**Bugs found and fixed** (each with a test that fails without the fix):

1. A local-only run wrote no `ledger.jsonl`, so its directory could not show that nothing was spent. The runner now always creates it. `tests/test_replay.py::test_local_only_run_leaves_an_empty_ledger`.
2. After the Ollama 500 and the runner's call retry, the proxy's identical-request detection marked the failed step `retried = true`, labelled `success = false`: an infrastructure error in the training data as the model's failure. The client now records the decisions of calls that got an error answer, and the runner relabels them unlabelled (`replay_call_failed = true`, `retried = false`, no `tests_passed`). `tests/test_replay.py::test_a_call_retried_after_an_upstream_error_is_not_labelled_a_failure`.
3. `--patch empty` could not validate an image (the harness never runs it); `--patch baseline` added. `tests/test_tasks.py::test_baseline_patch_changes_no_code_but_is_not_empty` and `test_cli_check_exit_codes`.
4. The harness and rows were not found unless `[check]` named them (the pre-flight failed on both after a fetch). `find_harness`, `find_rows`; `tests/test_tasks.py::test_find_harness_and_rows_in_the_project`, `test_replay_config_finds_the_project_harness_and_rows`.
5. One transient 502 failed `tasks fetch`; pages are now retried on 429, 5xx or a dropped connection. `tests/test_tasks.py::test_fetch_rows_retries_a_transient_server_error`.

**The final pre-flight** (`steproute replay preflight --plan pilot`, run from the project root with no config file): `ok` for the Docker daemon (server 28.0.4), the task images ("all 20 present"), the dataset rows, the harness (`.cache/steproute/swebench-env/bin/python -m swebench.harness.run_evaluation`), Ollama, the Ollama models ("all pulled: devstral:24b, gpt-oss:20b, qwen3-coder:30b, qwen3.5:9b") and free disk; `FAIL` only for `ANTHROPIC_API_KEY` and `DEEPSEEK_API_KEY` (not set, on purpose). The rows and the harness live in the checkout's `.cache/`, which git ignores: another checkout needs `steproute tasks fetch` and `scripts/install_swebench.sh` once.

**Found, not changed.**

- `psf__requests-1921` cannot be passed on its arm64 image; whether to drop it is the maintainer's call (`docs/TASKSET.md` section 8.1).
- The pre-flight's free-disk line measures the host disk at the output directory; images live in the Docker Desktop VM's disk, which the pre-flight does not see.
- The SWE-bench images run `pip install -e .` in `eval.sh`; with `--network none` the agent cannot install anything in its container, by design.

**Confidence.**

- **High** that the free half of the replay path works on this machine for `qwen3.5:9b`: three real runs, each step and artefact inspected.
- **High** for the grading table: each check's `report.json` and `test_output.txt` were read.
- **Medium** for the speeds: measured once per configuration while large downloads ran on the same machine.
- **Not verified:** any paid provider; the other local models in the agent loop; two models in one local lane; four containers at once; hours of unattended running.
