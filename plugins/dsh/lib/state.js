// @ts-check
// Pure helpers: reading dsh session-log event payloads and building the StepState sent to
// /decide. No I/O. Field names are those of dsh 0.2.2-rc.3 (docs/persistence-catalog.md in
// the @deepseek-ai/dsh package; docs/INTEGRATION-NOTES.md section 13).

/** Longest text sent for sensitive detection; the service drops it before storing. */
export const MAX_TASK_TEXT = 100_000
/** Most file paths sent in `features.paths`, and the longest one kept. */
export const MAX_PATHS = 32
export const MAX_PATH_CHARS = 256

/** @param {unknown} n */
export const count = (n) => (typeof n === 'number' && Number.isFinite(n) && n > 0 ? Math.floor(n) : 0)

/**
 * The text blocks of a message's content (`{type: 'text', text}` blocks; other block types,
 * reasoning included, are skipped).
 * @param {unknown} content
 * @returns {string}
 */
export function textOf(content) {
  if (!Array.isArray(content)) return ''
  let out = ''
  for (const b of content) {
    if (b && typeof b === 'object' && b.type === 'text' && typeof b.text === 'string') {
      out += (out ? '\n' : '') + b.text
      if (out.length > MAX_TASK_TEXT) return out.slice(0, MAX_TASK_TEXT)
    }
  }
  return out
}

/**
 * Number of tool-call blocks in an assistant message.
 * @param {unknown} message
 */
export function toolCallCount(message) {
  const content = message && typeof message === 'object' ? /** @type {any} */ (message).content : null
  return Array.isArray(content) ? content.filter((b) => b && b.type === 'tool-call').length : 0
}

/**
 * `assistant/message.usage` (dsh-llm TokenUsage; the counts are disjoint, `inputTokens` is
 * uncached input only, dsh-llm/src/types.ts:174-196) as the four Outcome token fields.
 * @param {any} u
 */
export function usageOf(u) {
  if (!u || typeof u !== 'object') return null
  return {
    input: count(u.inputTokens),
    output: count(u.outputTokens),
    cacheRead: count(u.cacheReadTokens),
    cacheWrite: count(u.cacheWriteTokens),
    reasoning: count(u.reasoningTokens),
  }
}

const TEST_COMMAND =
  /(?:^|[\s;&|(])(?:pytest|py\.test|tox|nox|jest|vitest|mocha|rspec|phpunit|ctest)(?:\s|$)|\b(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?test\b|\bgo\s+test\b|\bcargo\s+(?:test|nextest)\b|\bmake\s+(?:test|check)\b|\bpython3?\s+-m\s+(?:pytest|unittest)\b|\bgradlew?\s+test\b|\bmvn\s+test\b|\bdotnet\s+test\b|\bnode\s+--test\b/

/** Whether a shell command runs a test suite (the common runners only). @param {unknown} command */
export function isTestCommand(command) {
  return typeof command === 'string' && TEST_COMMAND.test(command)
}

/** The shell tools of the default bundle (dsh-tool-bash/src/index.ts:393, dsh-tool-pwsh/src/index.ts:409). */
export const SHELL_TOOLS = new Set(['bash', 'pwsh'])

/**
 * The verdict of a finished foreground shell run, from the model-facing text dsh-tool-bash
 * renders (dsh-tool-bash/src/render.ts:17-66): a non-zero exit ends the text with
 * `[exit code: N]`, and a run that was stopped, timed out, killed or moved to the background
 * ends with another bracketed marker. Exit 0 adds no marker.
 * @param {string} text
 * @returns {boolean | null} true for exit 0, false for a non-zero exit, null when unknown
 */
export function shellVerdict(text) {
  const last = text.trimEnd().split('\n').pop() ?? ''
  if (/^\[exit code: -?\d+\]$/.test(last)) return false
  if (/^\[(still running|timed out|killed by signal|stopped|sandbox|some output)/.test(last)) return null
  return true
}

const SPLIT = /[\s"'`()<>[\]{},;|]+/
const HAS_EXTENSION = /^[\w.@+-]*\w\.[A-Za-z0-9]{2,8}$/
const PATH_KEYS = /^(path|paths|file|file_path|filepath|filename|dir|directory|workdir|cwd|target|source|destination)$/i

/**
 * File-like tokens in free text, for the service's path rules (`.env`, `*.pem`, `.ssh/`...).
 * @param {string} text
 * @param {string[]} [out]
 * @returns {string[]}
 */
export function extractPaths(text, out = []) {
  if (!text) return out
  const seen = new Set(out)
  for (const raw of text.split(SPLIT)) {
    if (out.length >= MAX_PATHS) break
    const t = raw.replace(/[.:!?]+$/, '')
    if (t.length < 2 || t.length > MAX_PATH_CHARS || t.includes('://') || seen.has(t)) continue
    const pathy = (t.includes('/') && /\w/.test(t)) || /^\.[A-Za-z0-9_]/.test(t) || HAS_EXTENSION.test(t)
    if (!pathy) continue
    seen.add(t)
    out.push(t)
  }
  return out
}

/**
 * Paths in a tool call's raw `arguments` JSON string (`tool/call.arguments`): values of
 * path-like keys, and file-like tokens of a `command`.
 * @param {unknown} argsJson
 * @param {string[]} out
 */
export function pathsOfToolArgs(argsJson, out) {
  if (typeof argsJson !== 'string' || out.length >= MAX_PATHS) return out
  let args
  try {
    args = JSON.parse(argsJson)
  } catch {
    return out
  }
  if (!args || typeof args !== 'object') return out
  for (const [k, v] of Object.entries(args)) {
    if (out.length >= MAX_PATHS) break
    if (PATH_KEYS.test(k)) {
      for (const p of Array.isArray(v) ? v : [v]) {
        if (typeof p === 'string' && p && p.length <= MAX_PATH_CHARS && !out.includes(p) && out.length < MAX_PATHS) out.push(p)
      }
    } else if (k === 'command' && typeof v === 'string') {
      extractPaths(v, out)
    }
  }
  return out
}

/**
 * The `command` of a shell tool call's raw arguments, or null.
 * @param {unknown} argsJson
 * @returns {{ command: string | null, background: boolean }}
 */
export function shellCommandOf(argsJson) {
  try {
    const a = typeof argsJson === 'string' ? JSON.parse(argsJson) : null
    return {
      command: a && typeof a.command === 'string' ? a.command : null,
      background: Boolean(a && a.run_in_background === true),
    }
  } catch {
    return { command: null, background: false }
  }
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

/** @param {{ provider?: string, model?: string } | null | undefined} r */
export function routeLabel(r) {
  return r && r.model ? `${r.provider ?? '?'}/${r.model}` : null
}

/**
 * Two dsh model selections are the same choice (provider, model and effort, an absent
 * effort being its own value), as dsh-agent/src/model-selection.ts:52-55 compares them.
 * @param {any} a
 * @param {any} b
 */
export function sameSelection(a, b) {
  if (!a || !b) return a === b
  return a.provider === b.provider && a.model === b.model && (a.reasoningEffort ?? null) === (b.reasoningEffort ?? null)
}
