// @ts-check
// HTTP to the local steproute service through the host (`$.http.fetch`, reached through the
// Host adapter register.js builds), with a hard timeout. `$.http.fetch` has no timeout of its
// own below 30 seconds, and the time a hook spends inside a `$` call does not count against
// its budget, so without this race a hung service would hold a subagent spawn. Never rejects.

/**
 * What the collector needs from Claude Code, as closures over a hook's `$` (the mod
 * validator allows `$` only at call sites, `$.noun.event(...)`, so `$` itself is never
 * passed around). Tests give a fake.
 * @typedef {{
 *   fetch: (url: string, init: { method: string, headers: Record<string, string>, body?: string }) => Promise<{ status: number, text: string }>,
 *   sleep?: (ms: number, options: { signal: AbortSignal }) => Promise<unknown>,
 *   status?: (text: string) => void,
 *   log?: (text: string) => void,
 *   sessionId?: () => Promise<string>,
 *   version?: () => Promise<{ version: string }>,
 *   registerCommand?: (spec: { name: string, description: string }) => Promise<unknown>,
 * }} Host
 */

/**
 * @typedef {{ kind: 'http', status: number, body: any }
 *   | { kind: 'timeout' }
 *   | { kind: 'unreachable' }} Reply
 */

const TIMEOUT = Symbol('timeout')

/**
 * Wait `ms`, ending early (quietly) when `signal` aborts. Uses the host's clock (`$.clock.sleep`)
 * when it has one, else the environment's timer.
 * @param {Host} host
 * @param {number} ms
 * @param {AbortSignal} signal
 * @returns {Promise<unknown>}
 */
export function sleep(host, ms, signal) {
  try {
    if (host && typeof host.sleep === 'function') {
      return Promise.resolve(host.sleep(ms, { signal })).catch(() => undefined)
    }
  } catch {
    // fall through to the environment's timer
  }
  return new Promise((resolve) => {
    const t = setTimeout(resolve, ms)
    signal.addEventListener('abort', () => {
      clearTimeout(t)
      resolve(undefined)
    })
  })
}

/**
 * One request, raced against a timer.
 * @param {Host} host
 * @param {string} url
 * @param {{ method: string, headers: Record<string, string>, body?: string }} init
 * @param {number} ms
 * @returns {Promise<Reply>}
 */
async function once(host, url, init, ms) {
  const stop = new AbortController()
  const timer = sleep(host, ms, stop.signal).then(() => TIMEOUT)
  /** @type {Promise<Reply>} */
  const call = Promise.resolve()
    .then(() => host.fetch(url, init))
    .then(
      (res) => {
        let body = null
        try {
          body = res && typeof res.text === 'string' && res.text ? JSON.parse(res.text) : null
        } catch {
          body = null
        }
        return { kind: /** @type {const} */ ('http'), status: Number(res && res.status), body }
      },
      () => ({ kind: /** @type {const} */ ('unreachable') }),
    )
  const winner = await Promise.race([call, timer])
  stop.abort()
  return winner === TIMEOUT ? { kind: 'timeout' } : /** @type {Reply} */ (winner)
}

/**
 * POST `body` as JSON to `base + path` within `timeoutMs` in all. A refusal that comes back
 * fast (a kept-alive connection the service had closed) is retried once in the time left.
 *
 * @param {Host} host
 * @param {{ url: string, token: string | null }} cfg
 * @param {string} path
 * @param {unknown} body
 * @param {number} timeoutMs
 * @param {() => number} now
 * @returns {Promise<Reply>}
 */
export async function postJson(host, cfg, path, body, timeoutMs, now) {
  /** @type {Record<string, string>} */
  const headers = { 'Content-Type': 'application/json', Accept: 'application/json' }
  if (cfg.token) headers.Authorization = `Bearer ${cfg.token}`
  const init = { method: 'POST', headers, body: JSON.stringify(body) }
  const deadline = now() + timeoutMs
  /** @type {Reply} */
  let reply = { kind: 'unreachable' }
  for (let attempt = 0; attempt < 2; attempt++) {
    const left = deadline - now()
    if (left <= 0) return { kind: 'timeout' }
    const started = now()
    reply = await once(host, cfg.url + path, init, left)
    if (reply.kind !== 'unreachable' || now() - started >= left / 2) break
  }
  return reply
}
