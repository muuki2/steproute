// @ts-check
// Configuration of the steproute dsh plugin: the plugin row's `config` in a cordis patch,
// then three environment variables, then defaults. Pure, no I/O. docs/DSH.md, "Configuration".

export const PLUGIN_VERSION = '0.1.0'
/** The one dsh release this plugin was written and checked against (package.json engines.dsh). */
export const BUILT_FOR_DSH = '0.2.2-rc.3'
export const MODES = /** @type {const} */ (['shadow', 'route', 'off'])
export const DEFAULT_URL = 'http://127.0.0.1:8789'

/** dsh-llm-deepseek's efforts are off|low|high|max; steproute calls "off" `none`. */
const DEEPSEEK_EFFORTS = { none: 'off', low: 'low', high: 'high', max: 'max' }

/**
 * Catalog model id -> the dsh route that runs it. The default covers the models of the
 * DeepSeek API key route that dsh-base mounts (provider `deepseek-official`,
 * dsh-base/cordis.patch.yml:90-94, dsh-llm-deepseek-api-key/src/index.ts:20). Anything else
 * (a local Ollama model behind an `llm-pi-ai` route, say) is added in the row config.
 * @type {Record<string, { provider: string, model: string, efforts: Record<string, string> }>}
 */
export const DEFAULT_MODELS = {
  'deepseek-flash': { provider: 'deepseek-official', model: 'deepseek-flash', efforts: { ...DEEPSEEK_EFFORTS } },
  'deepseek-v4-pro': { provider: 'deepseek-official', model: 'deepseek-v4-pro', efforts: { ...DEEPSEEK_EFFORTS } },
}

/**
 * @typedef {'shadow' | 'route' | 'off'} Mode
 * @typedef {{ provider: string, model: string, efforts: Record<string, string> }} Route
 * @typedef {{
 *   mode: Mode,
 *   url: string,
 *   token: string | null,
 *   decideTimeoutMs: number,
 *   reportTimeoutMs: number,
 *   routeSubagents: boolean,
 *   models: Map<string, Route>,
 *   warnings: string[],
 * }} Config
 * @typedef {{ STEPROUTE_MODE?: string, STEPROUTE_URL?: string, STEPROUTE_TOKEN?: string }} Env
 */

/**
 * The effective configuration. The environment overrides the row config, which overrides
 * the defaults. A value that does not parse falls back to the safer default (an unknown
 * mode means shadow) and leaves a warning; nothing here throws, so a bad config can never
 * stop dsh from loading the plugin, let alone from starting.
 *
 * @param {unknown} raw the plugin row's config
 * @param {Env} [env]
 * @returns {Config}
 */
export function resolveConfig(raw, env = {}) {
  const o = raw && typeof raw === 'object' ? /** @type {Record<string, unknown>} */ (raw) : {}
  /** @type {string[]} */
  const warnings = []
  const modeRaw = nonEmpty(env.STEPROUTE_MODE) ?? nonEmpty(o.mode) ?? 'shadow'
  const modeText = modeRaw.trim().toLowerCase()
  /** @type {Mode} */
  let mode = 'shadow'
  if (/** @type {readonly string[]} */ (MODES).includes(modeText)) mode = /** @type {Mode} */ (modeText)
  else warnings.push(`unknown mode ${JSON.stringify(modeRaw.slice(0, 20))}; using shadow`)
  if (o.enabled === false) mode = 'off'

  let url = (nonEmpty(env.STEPROUTE_URL) ?? nonEmpty(o.serviceUrl) ?? DEFAULT_URL).trim().replace(/\/+$/, '')
  if (!/^http:\/\/(127\.0\.0\.1|localhost|\[::1\])(:\d+)?$/.test(url)) {
    warnings.push('service address must be http://127.0.0.1:<port> or http://localhost:<port>; using the default')
    url = DEFAULT_URL
  }
  return {
    mode,
    url,
    token: nonEmpty(env.STEPROUTE_TOKEN) ?? null,
    decideTimeoutMs: bounded(o.decideTimeoutMs, 200, 20, 2000),
    reportTimeoutMs: bounded(o.reportTimeoutMs, 500, 50, 5000),
    routeSubagents: o.routeSubagents !== false,
    models: parseModels(o.models, warnings),
    warnings,
  }
}

/**
 * @param {unknown} raw
 * @param {string[]} warnings
 * @returns {Map<string, Route>}
 */
function parseModels(raw, warnings) {
  /** @type {Map<string, Route>} */
  const out = new Map()
  const source = raw === undefined || raw === null ? DEFAULT_MODELS : raw
  if (typeof source !== 'object' || Array.isArray(source)) {
    warnings.push('models must be a mapping of catalog ids; using the default mapping')
    return parseModels(undefined, warnings)
  }
  for (const [id, value] of Object.entries(/** @type {Record<string, unknown>} */ (source))) {
    const v = /** @type {Record<string, unknown>} */ (value && typeof value === 'object' ? value : {})
    const provider = nonEmpty(v.provider)
    const model = nonEmpty(v.model)
    if (!provider || !model) {
      warnings.push(`models.${id.slice(0, 40)}: needs provider and model; ignored`)
      continue
    }
    /** @type {Record<string, string>} */
    const efforts = {}
    if (v.efforts && typeof v.efforts === 'object') {
      for (const [k, e] of Object.entries(/** @type {Record<string, unknown>} */ (v.efforts))) {
        if (typeof e === 'string' && e) efforts[k] = e
      }
    }
    out.set(id, { provider, model, efforts })
  }
  return out
}

/**
 * The catalog id of a dsh route, by the mapping; the dsh model id when it maps to nothing.
 * @param {Config} cfg
 * @param {string | undefined} provider
 * @param {string | undefined} model
 * @returns {string | null}
 */
export function catalogIdOf(cfg, provider, model) {
  if (!model) return null
  for (const [id, r] of cfg.models) if (r.provider === provider && r.model === model) return id
  return model
}

/**
 * A dsh reasoning effort in steproute's vocabulary (docs/SCHEMA.md), by the route's effort
 * mapping read backwards; null when the mapping does not name it.
 * @param {Config} cfg
 * @param {string | null} catalogId
 * @param {string | undefined} effort
 * @returns {string | null}
 */
export function steprouteEffortOf(cfg, catalogId, effort) {
  if (!effort || !catalogId) return null
  const r = cfg.models.get(catalogId)
  if (!r) return null
  for (const [ours, theirs] of Object.entries(r.efforts)) if (theirs === effort) return ours
  return null
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
 * The Standard Schema object cordis validates a row's config with (cordis/src/fiber.ts:50-61;
 * a plain object is refused there, docs/user/develop/basic/config.md:45). It never reports
 * issues: a value it cannot use is replaced by its default in resolveConfig, with a warning,
 * so a typo in the patch cannot fail the plugin's load.
 */
export const ConfigSchema = {
  '~standard': {
    version: 1,
    vendor: 'steproute',
    /** @param {unknown} value */
    validate(value) {
      return { value: value && typeof value === 'object' ? value : {} }
    },
  },
}
