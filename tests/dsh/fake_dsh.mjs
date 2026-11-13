// A stand-in for the parts of dsh 0.2.2-rc.3 the steproute plugin touches, driving the
// plugin's listeners in the order dsh-agent-loop/src/agent.ts and dsh-tools/src/batch.ts
// call them. No dsh code is loaded; docs/DSH.md, "Tests".
//
// One step, as the loop runs it:
//   1. selection = agent.model.current()                   (agent.ts:294, synchronous)
//   2. agent/pre-step waterfall                           (agent.ts:302-305)
//   3. step/start                                          (agent.ts:367)
//   4. user/message for the step, request/header if the config changed (agent.ts:465-497, 676-697)
//   5. assistant/message with message.source = the request's provider/model (agent.ts:572-583)
//   6. per tool call: tool/call, then (in model order) post-execute, then tool/result
//                                                          (batch.ts:138-150, 243-265)
//   7. step/end                                             (agent.ts:390)
// A turn: turn/start, its steps, turn/end.

import { createCollector } from '../../plugins/dsh/lib/collector.js'

/** A cordis-like context that records listeners, so index.js can be loaded too. */
export function fakeContext(providers = ['deepseek-official']) {
  /** @type {Map<string, Function[]>} */
  const listeners = new Map()
  const lines = []
  return {
    lines,
    listeners,
    on(name, fn) {
      if (!listeners.has(name)) listeners.set(name, [])
      listeners.get(name).push(fn)
      return () => {}
    },
    get(name) {
      return name === 'llm' ? { listProviders: () => providers.map((id) => ({ id, name: id })) } : undefined
    },
    logger() {
      return { info: (t) => lines.push(t), warn: (t) => lines.push(t) }
    },
    effect() {},
  }
}

/** Adapts a collector to the listener calls dsh would make (index.js does the same). */
export function collectorHost(rawConfig, deps) {
  const c = createCollector(rawConfig, deps)
  return {
    collector: c,
    emit: (session, event) => c.sessionEvent(session, event),
    created: (payload) => c.agentCreated(payload),
    disposed: (payload) => c.agentDisposed(payload),
    preStep: async (payload, next) => {
      const d = await next()
      c.preStep(payload, d)
      return d
    },
    postExecute: async (exec, result, next) => {
      const d = await next()
      await c.postExecute(exec, result)
      return d
    },
    flush: (session) => c.sessionFlush(session),
  }
}

/** Host shape built from a fakeContext after index.js's apply() ran. */
export function contextHost(ctx) {
  const one = (name) => (...args) => {
    let r
    for (const fn of ctx.listeners.get(name) ?? []) r = fn(...args)
    return r
  }
  return {
    emit: one('session/event'),
    created: one('agent/created'),
    disposed: one('agent/disposed'),
    preStep: async (payload, next) => {
      const fns = ctx.listeners.get('agent/pre-step') ?? []
      return fns.length ? fns[0](payload, next) : next()
    },
    postExecute: async (exec, result, next) => {
      const fns = ctx.listeners.get('tools/post-execute') ?? []
      return fns.length ? fns[0](exec, result, next) : next()
    },
    flush: async (session) => one('session/flush')(session),
  }
}

export class FakeAgent {
  /**
   * @param {*} host collectorHost(...) or contextHost(...)
   * @param {{ id?: string, selection?: any, origin?: string, parentSession?: string }} [opts]
   */
  constructor(host, opts = {}) {
    this.host = host
    this.id = opts.id ?? 'sess-1'
    this.session = {
      id: this.id,
      header: opts.origin ? { origin: opts.origin, parentSession: opts.parentSession } : { origin: 'user' },
    }
    this.log = []
    this.seq = 0
    this.selection = opts.selection ?? { provider: 'deepseek-official', model: 'deepseek-flash', reasoningEffort: 'high' }
    this.lastHeader = null
    this.turn = 0
    this.selects = []
    const agent = this
    this.model = {
      current: () => ({ ...agent.selection }),
      select: (sel) => {
        const same =
          sel.provider === agent.selection.provider &&
          sel.model === agent.selection.model &&
          (sel.reasoningEffort ?? null) === (agent.selection.reasoningEffort ?? null)
        if (same) return
        agent.selection = { ...sel }
        agent.selects.push({ ...sel })
        agent.append('model/selection', { ...sel })
      },
    }
  }

  append(type, data) {
    const event = { type, seq: this.seq++, time: Date.now(), data }
    this.log.push(event)
    this.host.emit(this.session, event)
    return event
  }

  async create() {
    await this.host.created({ agent: this, source: 'startup' })
  }

  /** The user (not the plugin) picks a model, as a UI picker would. */
  userSelects(sel) {
    this.model.select(sel)
  }

  /**
   * One turn: `steps` is a list of replies; every reply but the last should ask for tools.
   * @param {string} prompt
   * @param {Array<{ usage?: any, tools?: Array<{ name: string, args?: any, text?: string, isError?: boolean, code?: string }>, approvals?: string[], retries?: number }>} steps
   * @param {{ endKind?: string, tools?: string[] }} [opts]
   */
  async runTurn(prompt, steps, opts = {}) {
    const turn = ++this.turn
    this.append('turn/start', { turn })
    const record = []
    for (let i = 0; i < steps.length; i++) {
      const step = i + 1
      const reply = steps[i]
      const selection = this.model.current()
      const messages = step === 1 ? [{ role: 'user', source: { kind: 'user' }, content: [{ type: 'text', text: prompt }] }] : []
      const decision = await this.host.preStep(
        { agent: this, messages, turn, step, signal: new AbortController().signal, modelSelection: Object.freeze({ ...selection }) },
        async () => ({ kind: 'enter', messages }),
      )
      if (decision.kind !== 'enter') break
      this.append('step/start', { turn, step })
      for (const m of decision.messages) this.append('user/message', m)
      const header = {
        config: { ...selection },
        tools: (opts.tools ?? ['read', 'bash', 'edit']).map((name) => ({ name })),
      }
      if (JSON.stringify(header) !== JSON.stringify(this.lastHeader)) {
        this.append('request/header', { header, reason: this.lastHeader ? 'change' : 'initial' })
        this.lastHeader = header
      }
      for (let r = 0; r < (reply.retries ?? 0); r++) {
        this.append('assistant/attempt', { turn, step, stream: [] })
        this.append('llm/retry', { turn, step, provider: selection.provider, retry: r + 1, delayMs: 0 })
      }
      const calls = (reply.tools ?? []).map((t, k) => ({ ...t, id: `call-${turn}-${step}-${k}` }))
      this.append('assistant/message', {
        turn,
        step,
        message: {
          role: 'assistant',
          source: { provider: selection.provider, model: selection.model },
          content: [
            { type: 'text', text: 'working' },
            ...calls.map((c) => ({ type: 'tool-call', id: c.id, name: c.name, arguments: JSON.stringify(c.args ?? {}) })),
          ],
        },
        stream: [],
        usage: reply.usage ?? { inputTokens: 100, outputTokens: 20, cacheReadTokens: 0 },
      })
      for (const c of calls) this.append('tool/call', { turn, step, callId: c.id, name: c.name, arguments: JSON.stringify(c.args ?? {}) })
      const approvals = reply.approvals ?? []
      for (let k = 0; k < calls.length; k++) {
        const c = calls[k]
        if (approvals[k]) {
          this.append('approval/asked', { id: `ap-${c.id}`, toolName: c.name, callId: c.id })
          this.append('approval/decided', { id: `ap-${c.id}`, outcome: approvals[k] })
        }
        const result = { content: [{ type: 'text', text: c.text ?? 'ok' }], isError: c.isError === true }
        const exec = { callId: c.id, name: c.name, arguments: c.args ?? {}, agent: this, signal: new AbortController().signal }
        const started = Date.now()
        await this.host.postExecute(exec, result, async () => ({ kind: 'accept' }))
        record.push({ step, postExecuteMs: Date.now() - started })
        this.append('tool/result', {
          turn,
          step,
          message: { role: 'tool', toolCallId: c.id, source: { kind: 'tool', callId: c.id }, content: result.content, isError: result.isError || undefined },
          ...(c.code ? { error: { name: 'ToolError', code: c.code } } : {}),
        })
      }
      this.append('step/end', { turn, step })
    }
    const endKind = opts.endKind ?? 'completed'
    this.append('turn/end', { turn, reason: { kind: endKind } })
    return record
  }

  async flush() {
    await this.host.flush(this.session)
  }
}

/** A scripted stand-in for `postJson`: answers /decide and /report, and records the bodies. */
export function fakeService(opts = {}) {
  const decisions = []
  const reports = []
  let n = 0
  const svc = {
    decisions,
    reports,
    /** model/effort the next /decide answers with (or a function of the state). */
    action: opts.action ?? { model: 'deepseek-v4-pro', effort: 'low', escalate_to: null },
    /** Replace to script failures: (path, body) => Reply | Promise<Reply> | undefined. */
    override: null,
    delayMs: 0,
    async post(_url, _token, path, body, timeoutMs) {
      if (svc.override) {
        const r = await svc.override(path, body, timeoutMs)
        if (r) return r
      }
      if (svc.delayMs) {
        if (svc.delayMs >= timeoutMs) {
          await new Promise((r) => setTimeout(r, timeoutMs))
          return { kind: 'timeout' }
        }
        await new Promise((r) => setTimeout(r, svc.delayMs))
      }
      if (path === '/decide') {
        const action = typeof svc.action === 'function' ? svc.action(body) : svc.action
        const d = { decision_id: `d${++n}`, ts: 0, state: body, candidates: [], action, policy: opts.policy ?? 'rules', propensity: 1 }
        decisions.push(d)
        return { kind: 'http', status: 200, body: d }
      }
      reports.push(body)
      return { kind: 'http', status: 200, body }
    },
  }
  return svc
}

export const settle = () => new Promise((r) => setTimeout(r, 5))
