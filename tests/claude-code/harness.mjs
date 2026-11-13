// Test harness for the Claude Code plugin's hooks module: a fake engine that loads
// register.js the way Claude Code does (register(on, options)), a fake `$`, and a loopback
// HTTP server standing in for `steproute serve`. No Claude Code, no network beyond 127.0.0.1.

import http from 'node:http'
import { register } from '../../plugins/claude-code/hooks/register.mjs'

/**
 * A programmable fake service. `answer(path, body)` returns
 * `{ status, body, delayMs }` (defaults: 200, a decision or the outcome echoed).
 */
export async function fakeService(answer = () => ({})) {
  ids = 0 // decision ids d1, d2, ... per service
  const requests = []
  const server = http.createServer((req, res) => {
    let raw = ''
    req.on('data', (c) => (raw += c))
    req.on('end', () => {
      let body = null
      try {
        body = JSON.parse(raw)
      } catch {
        body = null
      }
      requests.push({ path: req.url, method: req.method, headers: req.headers, body })
      const a = answer(req.url, body, requests.length) || {}
      const status = a.status ?? 200
      const out = a.body !== undefined ? a.body : defaultBody(req.url, body, requests.length)
      const send = () => {
        if (res.destroyed) return
        res.writeHead(status, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify(out))
      }
      if (a.delayMs) setTimeout(send, a.delayMs)
      else send()
    })
  })
  await new Promise((r) => server.listen(0, '127.0.0.1', r))
  const url = `http://127.0.0.1:${server.address().port}`
  return {
    url,
    requests,
    decides: () => requests.filter((r) => r.path === '/decide'),
    reports: () => requests.filter((r) => r.path === '/report'),
    close: () =>
      new Promise((r) => {
        server.closeAllConnections?.()
        server.close(r)
      }),
  }
}

let ids = 0
export function decisionFor(state, action = { model: 'claude-sonnet-5-5', effort: 'medium', escalate_to: null }, policy = 'rules') {
  ids += 1
  return {
    decision_id: `d${ids}`,
    ts: 1,
    state: { ...state, task_text: null },
    candidates: [action],
    action,
    policy,
    propensity: 1.0,
  }
}

function defaultBody(path, body) {
  if (path === '/decide') return decisionFor(body)
  return body
}

/** The fake `$` a hook receives. */
export function fakeDollar({ env = {}, fetch: fetchImpl } = {}) {
  const status = []
  const logs = []
  const commands = []
  const $ = {
    http: {
      fetch:
        fetchImpl ??
        (async (url, init) => {
          const r = await fetch(url, { method: init.method, headers: init.headers, body: init.body })
          return { status: r.status, ok: r.ok, headers: {}, text: await r.text() }
        }),
    },
    clock: {
      sleep: (ms, o = {}) =>
        new Promise((resolve, reject) => {
          const t = setTimeout(resolve, ms)
          o.signal?.addEventListener('abort', () => {
            clearTimeout(t)
            reject(new Error('aborted'))
          })
        }),
    },
    ui: { status: (t) => status.push(t), log: (t) => logs.push(t) },
    env: { get: async (name) => env[name] },
    session: { id: async () => 'sess-1', version: async () => ({ version: '2.1.323' }) },
    command: { register: async (spec) => commands.push(spec) },
  }
  return { $, status, logs, commands }
}

function matches(matcher, e) {
  if (!matcher) return true
  return Object.entries(matcher).every(([k, v]) => {
    const got = e[k]
    if (v instanceof RegExp) return typeof got === 'string' && v.test(got)
    if (Array.isArray(v)) return v.includes(got)
    return got === v
  })
}

/** Load the plugin as the engine does and return a driver for its hooks. */
export function loadPlugin(options = {}, dollar = fakeDollar()) {
  const hooks = []
  const on = (event, a, b) => {
    const [matcher, hook] = b === undefined ? [undefined, a] : [a, b]
    hooks.push({ event, matcher, hook })
    return { catch: () => undefined }
  }
  register(on, options)
  const find = (event, e) => hooks.find((h) => h.event === event && matches(h.matcher, e))

  const engine = {
    hooks,
    $: dollar.$,
    dollar,
    /** Fire a plain event; `bottom(e)` is the engine's own behaviour. */
    async fire(event, e, bottom = async (x) => x) {
      const h = find(event, e)
      if (!h) return bottom(e)
      return h.hook(dollar.$, Object.freeze({ ...e }), bottom)
    },
    async start() {
      return engine.fire('session.start', { source: 'startup' }, async () => ({}))
    },
    /** Spawn; resolves with what the bottom saw and what the hook returned. */
    async spawn(e, result = { model: 'claude-haiku-5-5', agentId: 'a1' }) {
      let seen = null
      const res = await engine.fire('agent.spawn', e, async (x) => {
        seen = x
        return typeof result === 'function' ? result(x) : result
      })
      return { seen, res }
    },
    /** One model request of `agentId`; resolves with the input the bottom saw. */
    async step(agentId, usage, extra = {}) {
      const e = Object.freeze({ turnId: 't', index: 0, model: 'claude-haiku-5-5', messageCount: 1, agentId, ...extra })
      const h = find('turn.step', e)
      let seen = null
      const result = { turnId: 't', index: 0, answer: '', toolUses: [], stopReason: 'end_turn', usage }
      async function* bottom(x) {
        seen = x
        yield { kind: 'text', index: 0, text: 'hi' }
        return result
      }
      if (!h) return { seen: e, result }
      const gen = h.hook(dollar.$, e, bottom)
      let r = await gen.next()
      while (!r.done) r = await gen.next()
      return { seen, result: r.value }
    },
    async tool(agentId, tool, input, res) {
      return engine.fire('tool.call', { tool, tool_use_id: 'tu', agentId, ...input }, async () => res)
    },
    async complete(agentId, usage, extra = {}) {
      return engine.fire(
        'turn.complete',
        { answer: 'done', durationMs: 1234, isAborted: false, turnId: 't', agentId, usage, reason: 'answer', ...extra },
        async () => ({ text: 'done' }),
      )
    },
    async end() {
      return engine.fire('session.end', { reason: 'other' }, async () => ({}))
    },
    async command() {
      return engine.fire('command.run', { command: 'steproute', args: '' }, async () => ({ text: 'unanswered' }))
    },
  }
  return engine
}

export const usage = (input, output, cacheRead = 0, cacheWrite = 0, model = 'claude-haiku-5-5') => ({
  input_tokens: input,
  output_tokens: output,
  cache_read_input_tokens: cacheRead,
  cache_creation_input_tokens: cacheWrite,
  model,
})

export const SPAWN = Object.freeze({
  tool_use_id: 'toolu_1',
  prompt: 'Summarise notes/a.txt in one sentence.',
  description: 'summarise a',
  subagentType: 'general-purpose',
  provider: { plugin: 'engine', tier: 'core' },
  parentModel: 'claude-haiku-5-5',
  background: false,
  fork: false,
})

export const wait = (ms) => new Promise((r) => setTimeout(r, ms))
