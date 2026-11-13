// @ts-check
// POST JSON to the local steproute service with a hard timeout. Never rejects.
//
// node:http on purpose, not the global fetch: dsh installs its own fetch dispatcher for
// HTTP(S)_PROXY before plugins mount (dsh-http-proxy/src/install.ts), and this plugin must
// only ever talk to loopback. config.js refuses any service address that is not loopback.

import http from 'node:http'

/**
 * @typedef {{ kind: 'http', status: number, body: any }
 *   | { kind: 'timeout' }
 *   | { kind: 'unreachable' }} Reply
 * @typedef {(url: string, token: string | null, path: string, body: unknown, timeoutMs: number) => Promise<Reply>} Post
 */

const agent = new http.Agent({ keepAlive: true, maxSockets: 4, keepAliveMsecs: 10_000 })

/**
 * One request, at most `ms` long.
 * @param {URL} url
 * @param {Record<string, string>} headers
 * @param {string} payload
 * @param {number} ms
 * @returns {Promise<Reply>}
 */
function once(url, headers, payload, ms) {
  return new Promise((resolve) => {
    let done = false
    /** @type {ReturnType<typeof setTimeout> | undefined} */
    let timer
    /** @param {Reply} r */
    const finish = (r) => {
      if (done) return
      done = true
      clearTimeout(timer)
      resolve(r)
    }
    let req
    try {
      req = http.request(
        { host: url.hostname.replace(/^\[|\]$/g, ''), port: url.port || 80, path: url.pathname, method: 'POST', headers, agent },
        (res) => {
          /** @type {Buffer[]} */
          const chunks = []
          let size = 0
          res.on('data', (c) => {
            size += c.length
            if (size <= 4 * 1024 * 1024) chunks.push(c)
          })
          res.on('end', () => {
            let body = null
            try {
              const text = Buffer.concat(chunks).toString('utf8')
              body = text ? JSON.parse(text) : null
            } catch {
              body = null
            }
            finish({ kind: 'http', status: Number(res.statusCode), body })
          })
          res.on('error', () => finish({ kind: 'unreachable' }))
        },
      )
    } catch {
      finish({ kind: 'unreachable' })
      return
    }
    timer = setTimeout(() => {
      finish({ kind: 'timeout' })
      req.destroy()
    }, ms)
    timer.unref?.()
    req.on('error', () => finish({ kind: 'unreachable' }))
    req.end(payload)
  })
}

/**
 * POST `body` as JSON to `base + path` within `timeoutMs` in all. A refusal that comes back
 * fast (a kept-alive connection the service had closed) is retried once in the time left.
 * @type {Post}
 */
export async function postJson(base, token, path, body, timeoutMs) {
  let url
  let payload
  try {
    url = new URL(base + path)
    payload = JSON.stringify(body)
  } catch {
    return { kind: 'unreachable' }
  }
  /** @type {Record<string, string>} */
  const headers = {
    'Content-Type': 'application/json',
    Accept: 'application/json',
    'Content-Length': String(Buffer.byteLength(payload)),
  }
  if (token) headers.Authorization = `Bearer ${token}`
  const deadline = Date.now() + timeoutMs
  /** @type {Reply} */
  let reply = { kind: 'unreachable' }
  for (let attempt = 0; attempt < 2; attempt++) {
    const left = deadline - Date.now()
    if (left <= 0) return { kind: 'timeout' }
    const started = Date.now()
    reply = await once(url, headers, payload, left)
    if (reply.kind !== 'unreachable' || Date.now() - started >= left / 2) break
  }
  return reply
}
