// @ts-check
// Pure helpers of the steproute Claude Code collector: configuration, the StepState sent to
// /decide, and the small parsers the outcome uses. No `$`, no I/O; docs/CLAUDE-CODE.md.

export const PLUGIN_VERSION = '0.1.0'
export const HARNESS = 'claude-code'
export const MODES = /** @type {const} */ (['shadow', 'route', 'off'])
export const BILLINGS = /** @type {const} */ (['subscription', 'api'])
export const DEFAULT_URL = 'http://127.0.0.1:8787'
export const DEFAULT_ROUTE_TYPES = ['general-purpose', 'Explore', 'Plan', 'claude']

/** Longest task text sent for sensitive detection; the service drops it before storing. */
export const MAX_TASK_TEXT = 100_000
/** Most file paths sent in `features.paths`, and the longest one kept. */
export const MAX_PATHS = 32
export const MAX_PATH_CHARS = 256
/** Effort levels a `turn.step` rewrite may carry (the public builds refuse numbers). */
export const STEP_EFFORTS = ['low', 'medium', 'high', 'xhigh', 'max']

/**
 * @typedef {'shadow' | 'route' | 'off'} Mode
 * @typedef {{
 *   mode: Mode,
 *   url: string,
 *   token: string | null,
 *   decideTimeoutMs: number,
 *   reportTimeoutMs: number,
 *   billing: 'subscription' | 'api',
 *   routeTypes: string[] | '*',
 *   warnings: string[],
 * }} Config
 * @typedef {{ mode?: unknown, url?: unknown, token?: unknown }} EnvOverrides
 */

/**
 * The effective configuration: the environment overrides the plugin's `userConfig`
 * options, which override the defaults. A value that does not parse falls back to the
 * safer default (an unknown mode means shadow) and leaves a warning for `/steproute`.
 *
 * @param {Record<string, unknown> | undefined} options
 * @param {EnvOverrides} [env]
 * @returns {Config}
 */
export function resolveConfig(options, env = {}) {
  const o = options ?? {}
  /** @type {string[]} */
  const warnings = []
  const modeRaw = nonEmpty(env.mode) ?? nonEmpty(o.mode) ?? 'shadow'
  const modeText = modeRaw.trim().toLowerCase()
  /** @type {Mode} */
  let mode = 'shadow'
  if (/** @type {readonly string[]} */ (MODES).includes(modeText)) {
    mode = /** @type {Mode} */ (modeText)
  } else {
    warnings.push(`unknown mode ${JSON.stringify(modeRaw.slice(0, 20))}; using shadow`)
  }
  let url = (nonEmpty(env.url) ?? nonEmpty(o.service_url) ?? DEFAULT_URL).trim().replace(/\/+$/, '')
  if (!/^https?:\/\/[^\s/]+$/.test(url)) {
    warnings.push('service address is not an http(s) base URL; using the default')
    url = DEFAULT_URL
  }
  const billingRaw = nonEmpty(o.billing) ?? 'subscription'
  const billing = billingRaw === 'api' ? 'api' : 'subscription'
  const typesRaw = nonEmpty(o.route_agent_types)
  /** @type {string[] | '*'} */
  let routeTypes = DEFAULT_ROUTE_TYPES
  if (typesRaw !== undefined) {
    const list = typesRaw.split(',').map((t) => t.trim()).filter(Boolean)
    routeTypes = list.includes('*') ? '*' : list
  }
  return {
    mode,
    url,
    token: nonEmpty(env.token) ?? null,
    decideTimeoutMs: bounded(o.decide_timeout_ms, 200, 20, 2000),
    reportTimeoutMs: bounded(o.report_timeout_ms, 500, 50, 5000),
    billing,
    routeTypes,
    warnings,
  }
}

/** @param {unknown} v */
function nonEmpty(v) {
  return typeof v === 'string' && v.trim() !== '' ? v : undefined
}

/**
 * @param {unknown} v
 * @param {number} dflt
 * @param {number} lo
 * @param {number} hi
 */
function bounded(v, dflt, lo, hi) {
  const n = typeof v === 'number' ? v : typeof v === 'string' ? Number(v) : NaN
  if (!Number.isFinite(n)) return dflt
  return Math.min(hi, Math.max(lo, n))
}

/**
 * The `agent.spawn` fields the state is built from (a subset of `AgentSpawnInput`).
 * @typedef {{
 *   tool_use_id?: string,
 *   prompt?: string,
 *   description?: string,
 *   subagentType?: string,
 *   model?: string,
 *   parentModel?: string,
 *   parentAgentId?: string,
 *   background?: boolean,
 *   fork?: boolean,
 *   isTeammate?: true,
 *   workflow?: unknown,
 *   resume?: { agentId: string },
 * }} SpawnInput
 */

/**
 * The `StepState` for one subagent spawn (docs/SCHEMA.md).
 *
 * A subagent's first request never reads the parent's prompt cache, so the state has no
 * cache owner and nothing cached. The hook sees the task prompt but not the subagent's
 * system prompt or tool list, so `prompt_tokens` is the task prompt's characters divided by
 * four (an estimate, said so in `features.prompt_tokens_basis`) and `tools` is empty.
 *
 * @param {SpawnInput} e
 * @param {{ sessionId: string | null, ccVersion: string | null }} ctx
 * @param {number} stepIdx
 * @param {Mode} mode
 */
export function buildState(e, ctx, stepIdx, mode) {
  const prompt = typeof e.prompt === 'string' ? e.prompt : ''
  /** @type {Record<string, unknown>} */
  const features = {
    subagent_type: typeof e.subagentType === 'string' ? e.subagentType : null,
    background: e.background === true,
    workflow: e.workflow !== undefined && e.workflow !== null,
    nested: typeof e.parentAgentId === 'string',
    parent_model: typeof e.parentModel === 'string' ? e.parentModel : null,
    requested_model: typeof e.model === 'string' ? e.model : null,
    prompt_chars: prompt.length,
    prompt_tokens_basis: 'task_prompt_chars_div_4',
    routing_mode: mode,
    plugin_version: PLUGIN_VERSION,
    claude_code_version: ctx.ccVersion,
  }
  if (prompt.length > MAX_TASK_TEXT) features.task_text_truncated = true
  const paths = extractPaths(prompt)
  if (paths.length) features.paths = paths
  return {
    harness: HARNESS,
    session_id: ctx.sessionId ?? 'unknown',
    step_idx: stepIdx,
    kind: 'subagent',
    prompt_tokens: Math.ceil(prompt.length / 4),
    task_id: typeof e.tool_use_id === 'string' && e.tool_use_id ? e.tool_use_id : null,
    cached_prefix_tokens: 0,
    current_model: null,
    current_effort: null,
    tools: [],
    sensitive: false,
    source: 'personal',
    task_text: prompt ? prompt.slice(0, MAX_TASK_TEXT) : null,
    features,
  }
}

const SPLIT = /[\s"'`()<>[\]{},;|]+/
const HAS_EXTENSION = /^[\w.@+-]*\w\.[A-Za-z0-9]{2,8}$/

/**
 * File-like tokens in a prompt, for the service's path rules (`.env`, `*.pem`, `.ssh/`, ...).
 * Keeps tokens with a `/`, dotfiles and names with an extension; drops URLs. At most
 * MAX_PATHS, each at most MAX_PATH_CHARS long, first occurrence order.
 *
 * @param {string} text
 * @returns {string[]}
 */
export function extractPaths(text) {
  /** @type {string[]} */
  const out = []
  if (!text) return out
  const seen = new Set()
  for (const raw of text.split(SPLIT)) {
    const t = raw.replace(/[.:!?]+$/, '')
    if (t.length < 2 || t.length > MAX_PATH_CHARS || t.includes('://') || seen.has(t)) continue
    const pathy =
      (t.includes('/') && /\w/.test(t)) || /^\.[A-Za-z0-9_]/.test(t) || HAS_EXTENSION.test(t)
    if (!pathy) continue
    seen.add(t)
    out.push(t)
    if (out.length >= MAX_PATHS) break
  }
  return out
}

const CLAUDE_ALIASES = new Set(['haiku', 'sonnet', 'opus', 'fable'])

/**
 * Whether Claude Code can run `model` as a subagent: a Claude model id or alias.
 * @param {unknown} model
 */
export function isClaudeModel(model) {
  return (
    typeof model === 'string' && (CLAUDE_ALIASES.has(model) || /^claude-[a-z0-9][a-z0-9.-]*$/.test(model))
  )
}

/**
 * A model id as the API reports it, mapped to the catalog's form: a trailing date
 * (`-20261101`) and a bracketed option (`[1m]`) are removed.
 * @param {unknown} model
 * @returns {string | null}
 */
export function normaliseModel(model) {
  if (typeof model !== 'string' || !model) return null
  return model.replace(/\[[^\]]*\]$/, '').replace(/-\d{8}$/, '')
}

const TEST_COMMAND =
  /(?:^|[\s;&|(])(?:pytest|py\.test|tox|nox|jest|vitest|mocha|rspec|phpunit|ctest)(?:\s|$)|\b(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?test\b|\bgo\s+test\b|\bcargo\s+(?:test|nextest)\b|\bmake\s+(?:test|check)\b|\bpython3?\s+-m\s+(?:pytest|unittest)\b|\bgradlew?\s+test\b|\bmvn\s+test\b|\bdotnet\s+test\b/

/**
 * Whether a Bash command runs a test suite (the common runners only).
 * @param {unknown} command
 */
export function isTestCommand(command) {
  return typeof command === 'string' && TEST_COMMAND.test(command)
}

/**
 * Compact label of an action chain, as `Action.label()` prints it: `a (high) -> b (low)`.
 * @param {unknown} action
 * @returns {string | null}
 */
export function actionLabel(action) {
  /** @type {string[]} */
  const parts = []
  let a = /** @type {any} */ (action)
  for (let i = 0; a && typeof a === 'object' && i < 8; i++) {
    parts.push(`${a.model} (${a.effort})`)
    a = a.escalate_to
  }
  return parts.length ? parts.join(' -> ') : null
}
