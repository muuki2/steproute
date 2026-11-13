// @ts-check
// steproute plugin for DeepSeek Harness (dsh) 0.2.2-rc.3: a cordis plugin in plain
// JavaScript, loaded by dsh as it is (no build step, no imports from dsh). docs/DSH.md.
//
// Every listener below either observes (emit events) or, in a waterfall, calls `next()`
// first and returns its result unchanged; the plugin's own work runs inside the collector,
// which catches everything. Nothing the plugin does can make a dsh step fail.

import { ConfigSchema, PLUGIN_VERSION } from './lib/config.js'
import { createCollector } from './lib/collector.js'

export const name = 'steproute'
export const Config = ConfigSchema

/**
 * @param {any} ctx the cordis context dsh hands the plugin
 * @param {unknown} config the row's config from the patch
 */
export function apply(ctx, config) {
  /** @type {{ info: (t: string) => void, warn: (t: string) => void } | null} */
  let logger = null
  try {
    logger = typeof ctx.logger === 'function' ? ctx.logger('steproute') : null
  } catch {
    logger = null
  }
  const collector = createCollector(config, {
    env: {
      STEPROUTE_MODE: process.env.STEPROUTE_MODE,
      STEPROUTE_URL: process.env.STEPROUTE_URL,
      STEPROUTE_TOKEN: process.env.STEPROUTE_TOKEN,
    },
    // Provider routes with a registered adapter (dsh-llm/src/index.ts:493-500), so a
    // decision is never applied to a route dsh cannot call.
    providers: () => {
      const llm = typeof ctx.get === 'function' ? ctx.get('llm') : undefined
      if (!llm || typeof llm.listProviders !== 'function') return null
      return llm.listProviders().map((/** @type {any} */ p) => String(p.id))
    },
  })
  const cfg = collector._debug.cfg
  const say = (/** @type {string} */ text) => {
    try {
      logger?.info(text)
    } catch {
      // no logger
    }
  }
  say(`steproute ${PLUGIN_VERSION}: ${cfg.mode} mode, service ${cfg.url}`)
  for (const w of cfg.warnings) say(`steproute config: ${w}`)
  if (cfg.mode === 'off') return

  ctx.on('agent/created', (/** @type {any} */ payload) => {
    collector.agentCreated(payload)
    return undefined
  })
  ctx.on('agent/disposed', (/** @type {any} */ payload) => {
    collector.agentDisposed(payload)
  })
  ctx.on('agent/pre-step', async (/** @type {any} */ payload, /** @type {() => Promise<any>} */ next) => {
    const decision = await next()
    collector.preStep(payload, decision)
    return decision
  })
  ctx.on(
    'tools/post-execute',
    async (/** @type {any} */ exec, /** @type {any} */ result, /** @type {() => Promise<any>} */ next) => {
      const decision = await next()
      await collector.postExecute(exec, result)
      return decision
    },
  )
  ctx.on('session/event', (/** @type {any} */ session, /** @type {any} */ event) => {
    collector.sessionEvent(session, event)
  })
  ctx.on('session/flush', (/** @type {any} */ session) => collector.sessionFlush(session))
  if (typeof ctx.effect === 'function') {
    ctx.effect(() => () => say(collector.statusText()), 'steproute summary')
  }
}
