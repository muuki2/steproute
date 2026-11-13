// @ts-check
// The steproute collector for dsh: one decision per agent step, one outcome per step,
// joined by decision_id. index.js wires these handlers to dsh's events; nothing here imports
// dsh, so tests drive it with a fake. docs/DSH.md.
//
// A dsh step is one model request plus the tool calls it asked for (dsh-agent-loop
// src/agent.ts: `step/start`, the request, `tools.execute`, `step/end`). The loop reads the
// next step's model with `agent.model.current()` synchronously at the start of pre-step
// (agent.ts:294), so a decision can only be applied if it is in hand before that read:
//
// - after a tool batch, the decision for the next step is asked for inside
//   `tools/post-execute` (an awaited waterfall that runs before the batch settles) and
//   applied there with `agent.model.select()`;
// - the first step of a turn follows nothing awaitable, so its decision is asked for (in
//   `agent/pre-step`, without waiting) and logged, but never applied. So is any other step
//   that no tool batch preceded (a step after steering).
//
// Tool calls of a batch commit in model order, each one's post-execute just before its
// `tool/result` (dsh-tools/src/batch.ts:138-150), so the post-execute of the last call sees
// every other result of the batch in the log: that call asks for the decision.
//
// Rules every handler keeps: never throw into dsh, never hold a step longer than the
// configured timeout, change nothing unless the mode is `route`, report observed facts only.

import { postJson } from './client.js'
import { BUILT_FOR_DSH, PLUGIN_VERSION, catalogIdOf, resolveConfig, steprouteEffortOf } from './config.js'
import {
  MAX_PATHS,
  MAX_TASK_TEXT,
  actionLabel,
  count,
  extractPaths,
  isTestCommand,
  pathsOfToolArgs,
  routeLabel,
  sameSelection,
  shellCommandOf,
  shellVerdict,
  SHELL_TOOLS,
  textOf,
  toolCallCount,
  usageOf,
} from './state.js'

export const HARNESS = 'dsh'
/** Agents (sessions) tracked at once; past it the least recently active idle one is dropped. */
export const MAX_AGENTS = 64
/** Tool calls remembered per step (for test results and paths). */
export const MAX_CALLS_PER_STEP = 128
/** Outcomes kept for resending while the service is away. */
export const MAX_RETRY = 64
/** After a timeout, refusal or 500: skip the service this long (ms). */
export const BACKOFF_MS = 30_000
/** After 400, 401, 403, 404, 405, 413, 415 (a configuration or client problem). */
export const CONFIG_BACKOFF_MS = 300_000
/** Longest wait for unsent outcomes when dsh flushes an idle session (headless exit). */
export const FLUSH_WAIT_MS = 1000
/** Delay samples kept for the status text. */
const MAX_SAMPLES = 256

const DENIED_CODES = new Set(['APPROVAL_REJECTED', 'APPROVAL_CANCELLED', 'APPROVAL_UNAVAILABLE'])
const ABORTED_CODES = new Set(['ABORTED', 'ABORTED_BEFORE_DISPATCH'])
const INFRA_CODES = new Set(['TOOL_CONTEXT_UNAVAILABLE', 'TOOL_EXECUTOR_UNAVAILABLE'])

/**
 * @typedef {import('./config.js').Config} Config
 * @typedef {import('./client.js').Post} Post
 * @typedef {{ provider: string, model: string, reasoningEffort?: string }} Selection
 * @typedef {{
 *   role: 'turn_start' | 'after_tools' | 'unplanned',
 *   decisionId: string | null,
 *   recommended: { model: string, effort: string } | null,
 *   fallback: string | null,
 *   policy: string | null,
 *   target: Selection | null,
 *   applied: boolean,
 *   reason: string,
 *   decideMs: number | null,
 *   waitMs: number | null,
 *   late: boolean,
 *   applyDone: boolean,
 *   stepIdx: number,
 *   settled: Promise<void>,
 * }} Plan
 * @typedef {{
 *   turn: number, step: number, plan: Plan | null, selection: Selection | null,
 *   startedAt: number, endedAt: number | null,
 *   requests: number, attempts: number,
 *   usage: { input: number, output: number, cacheRead: number, cacheWrite: number, reasoning: number },
 *   executed: { provider: string, model: string } | null, effort: string | null,
 *   toolCallsAsked: number, toolCalls: number, toolErrors: number, denied: number, aborted: number,
 *   approvals: { asked: number, allowed: number, rejected: number, cancelled: number, unavailable: number },
 *   testRuns: number, testsPassed: boolean | null, llmRetries: number, interrupted: boolean,
 *   calls: Map<string, { name: string, command: string | null, background: boolean }>,
 *   resultChars: number, resultText: string, results: number, paths: string[], previousModel: string | null,
 * }} Step
 * @typedef {{
 *   sessionId: string, agent: any, subagent: boolean, parentSession: string | null,
 *   baseline: Selection | null, pluginSel: Selection | null, selecting: boolean,
 *   turn: number, turnOpen: boolean, stepCount: number,
 *   tools: string[], headerEffort: string | null, last: { prompt: number, output: number, provider: string, model: string, effort: string | null } | null,
 *   userText: string,
 *   upcoming: { turn: number, step: number, plan: Plan | null, selection: Selection | null } | null,
 *   next: (Plan & { turn: number, afterStep: number }) | null,
 *   cur: Step | null, done: Step | null, lastActive: number,
 * }} Track
 */

/**
 * @param {unknown} rawConfig the plugin row's config
 * @param {{ env?: import('./config.js').Env, post?: Post, now?: () => number,
 *   providers?: () => string[] | null, log?: (text: string) => void }} [deps]
 */
export function createCollector(rawConfig, deps = {}) {
  const now = deps.now ?? (() => Date.now())
  // Delays are measured on the monotonic clock, to a hundredth of a millisecond.
  const hr = () => performance.now()
  const ms = (/** @type {number} */ since) => Math.round((hr() - since) * 100) / 100
  const post = deps.post ?? postJson
  const providers = deps.providers ?? (() => null)
  const log = deps.log ?? (() => {})
  /** @type {Config} */
  const cfg = resolveConfig(rawConfig, deps.env ?? {})
  /** @type {Map<string, Track>} */
  const tracks = new Map()
  /** @type {Record<string, unknown>[]} */
  const retry = []
  /** @type {Set<Promise<unknown>>} */
  const inflight = new Set()
  let backoffUntil = 0
  /** @type {string | null} */
  let lastProblem = null
  const stats = {
    steps: 0,
    decided: 0,
    applied: 0,
    restored: 0,
    reported: 0,
    dropped: 0,
    sensitive: 0,
    skipped: 0,
    failOpen: 0,
    pluginErrors: 0,
    /** Time a step waited inside tools/post-execute for its decision (the added delay). */
    waitMs: /** @type {number[]} */ ([]),
    decideMs: /** @type {number[]} */ ([]),
  }

  /** @param {string} where */
  function contained(where) {
    stats.pluginErrors += 1
    lastProblem = `plugin error in ${where}`
  }

  // --- service calls --------------------------------------------------------------------

  /** @param {number} ms @param {string} problem */
  function backOff(ms, problem) {
    backoffUntil = now() + ms
    lastProblem = problem
  }

  /**
   * @param {string} path
   * @param {unknown} body
   * @param {number} timeoutMs
   */
  async function call(path, body, timeoutMs) {
    const reply = await post(cfg.url, cfg.token, path, body, timeoutMs)
    if (reply.kind === 'timeout') backOff(BACKOFF_MS, 'service timed out')
    else if (reply.kind === 'unreachable') backOff(BACKOFF_MS, 'service unreachable')
    else if ([400, 401, 403, 404, 405, 411, 413, 415].includes(reply.status)) {
      backOff(CONFIG_BACKOFF_MS, `service refused (${reply.status})`)
    } else if (reply.status >= 500 && reply.status !== 503) backOff(BACKOFF_MS, `service error (${reply.status})`)
    else if (reply.status === 200) lastProblem = null
    return reply
  }

  // --- tracks ---------------------------------------------------------------------------

  /**
   * @param {string} sessionId
   * @param {any} [agent]
   * @param {any} [session]
   * @returns {Track}
   */
  function trackOf(sessionId, agent, session) {
    let t = tracks.get(sessionId)
    // The session header says whether this agent is a subagent (`origin: 'subagent'`, with
    // `parentSession`; dsh-tool-subagent/src/model-selection-settings.ts:63).
    const sess = agent && agent.session ? agent.session : session
    const header = sess && sess.header && typeof sess.header === 'object' ? sess.header : null
    if (!t) {
      t = {
        sessionId,
        agent: agent ?? null,
        subagent: false,
        parentSession: null,
        baseline: null,
        pluginSel: null,
        selecting: false,
        turn: 0,
        turnOpen: false,
        stepCount: 0,
        tools: [],
        headerEffort: null,
        last: null,
        userText: '',
        upcoming: null,
        next: null,
        cur: null,
        done: null,
        lastActive: now(),
      }
      tracks.set(sessionId, t)
      evict()
    }
    if (agent && !t.agent) t.agent = agent
    if (header && header.origin === 'subagent') {
      t.subagent = true
      if (typeof header.parentSession === 'string') t.parentSession = header.parentSession
    }
    t.lastActive = now()
    return t
  }

  function evict() {
    while (tracks.size > MAX_AGENTS) {
      let victim = null
      for (const t of tracks.values()) {
        if (t.turnOpen) continue
        if (!victim || t.lastActive < victim.lastActive) victim = t
      }
      victim = victim ?? tracks.values().next().value ?? null
      if (!victim) return
      closeTrack(victim, 'evicted')
    }
  }

  /** @param {Track} t @param {string} reason */
  function closeTrack(t, reason) {
    tracks.delete(t.sessionId)
    if (t.done) finish(t, t.done, 'continued')
    if (t.cur) finish(t, t.cur, reason)
    dropPlan(t.upcoming && t.upcoming.plan, reason)
    dropPlan(t.next, reason)
    t.done = t.cur = t.upcoming = t.next = null
  }

  /** @param {any} agent */
  function currentOf(agent) {
    try {
      const c = agent && agent.model && typeof agent.model.current === 'function' ? agent.model.current() : undefined
      return c && typeof c.provider === 'string' && typeof c.model === 'string' ? selectionOf(c) : null
    } catch {
      return null
    }
  }

  /** @param {any} s @returns {Selection} */
  function selectionOf(s) {
    /** @type {Selection} */
    const out = { provider: String(s.provider), model: String(s.model) }
    if (typeof s.reasoningEffort === 'string') out.reasoningEffort = s.reasoningEffort
    return out
  }

  /**
   * agent.model.select(), with the plugin's own model/selection event told apart from the
   * user's (the append publishes synchronously inside select, dsh-session/src/index.ts:960-990).
   * @param {Track} t
   * @param {Selection} sel
   */
  function select(t, sel) {
    t.selecting = true
    try {
      t.agent.model.select(sel)
      return true
    } catch {
      return false
    } finally {
      t.selecting = false
    }
  }

  /**
   * Fail open: put back the user's own selection if the plugin replaced it.
   * @param {Track} t
   */
  function restoreBaseline(t) {
    if (cfg.mode !== 'route' || !t.pluginSel || !t.agent) return false
    const base = t.baseline
    t.pluginSel = null
    if (!base || sameSelection(currentOf(t.agent), base)) return false
    const ok = select(t, base)
    if (ok) stats.restored += 1
    return ok
  }

  // --- the state ------------------------------------------------------------------------

  /**
   * The StepState for the step about to run (docs/SCHEMA.md).
   *
   * Cache: dsh's providers cache per model, and a model switch also injects a
   * "[model changed ...]" notice and a new request header (dsh-agent model-selection.ts:
   * 133-171; the headless persona names the model in the system prompt). The owner is
   * therefore the route of the previous request, and its whole prompt is the cached prefix;
   * the cost model then prices any other model as a full miss.
   *
   * @param {Track} t
   * @param {Plan['role']} role
   * @param {Selection | null} selection the route in effect for the step
   * @param {string} text text for sensitive detection (never stored by default)
   * @param {string[]} paths
   * @param {number} extraChars characters added since the previous request
   * @param {number} stepIdx
   */
  function buildState(t, role, selection, text, paths, extraChars, stepIdx) {
    const last = t.last
    const ownerId = last ? catalogIdOf(cfg, last.provider, last.model) : null
    const kind = t.subagent ? 'subagent' : role === 'after_tools' ? 'tool_followup' : 'main'
    const prompt = (last ? last.prompt + last.output : 0) + Math.ceil(extraChars / 4)
    /** @type {Record<string, unknown>} */
    const features = {
      step_role: role,
      agent_kind: t.subagent ? 'subagent' : 'main',
      dsh_turn: t.turn,
      selected_route: routeLabel(selection),
      selected_effort: selection ? selection.reasoningEffort ?? null : null,
      cache_owner_route: last ? routeLabel(last) : null,
      prompt_tokens_basis: last ? 'previous_request_prompt_plus_output_plus_new_chars_div_4' : 'new_chars_div_4_no_system_prompt',
      routing_mode: cfg.mode,
      plugin_version: PLUGIN_VERSION,
      built_for_dsh: BUILT_FOR_DSH,
    }
    if (t.parentSession) features.parent_session = t.parentSession
    if (paths.length) features.paths = paths.slice(0, MAX_PATHS)
    return {
      harness: HARNESS,
      session_id: t.sessionId,
      step_idx: stepIdx,
      kind,
      prompt_tokens: Math.max(prompt, 1),
      task_id: null,
      cached_prefix_tokens: last ? Math.min(last.prompt, prompt) : 0,
      current_model: ownerId,
      current_effort: last ? steprouteEffortOf(cfg, ownerId, last.effort ?? undefined) : null,
      tools: t.tools.slice(0, 256),
      sensitive: false,
      source: 'personal',
      task_text: text ? text.slice(0, MAX_TASK_TEXT) : null,
      features,
    }
  }

  // --- decisions ------------------------------------------------------------------------

  /**
   * Ask /decide; resolve the plan's fields. Never rejects.
   * @param {Track} t
   * @param {Plan} plan
   * @param {Record<string, unknown>} state
   */
  async function decide(t, plan, state) {
    const started = hr()
    const reply = await call('/decide', state, cfg.decideTimeoutMs)
    plan.decideMs = ms(started)
    remember(stats.decideMs, plan.decideMs)
    if (reply.kind !== 'http') {
      plan.reason = `service_${reply.kind}`
      return
    }
    if (reply.status === 503) {
      const sensitive = Boolean(reply.body && reply.body.sensitive)
      if (sensitive) stats.sensitive += 1
      plan.reason = sensitive ? 'sensitive_no_decision' : 'no_decision'
      return
    }
    const d = reply.body
    if (reply.status !== 200 || !d || typeof d.decision_id !== 'string' || !d.action) {
      plan.reason = `service_${reply.status}`
      return
    }
    stats.decided += 1
    plan.decisionId = d.decision_id
    plan.policy = typeof d.policy === 'string' ? d.policy : null
    plan.recommended = { model: String(d.action.model), effort: String(d.action.effort) }
    plan.fallback = d.action.escalate_to ? actionLabel(d.action.escalate_to) : null
    plan.reason = routeReason(t, plan)
  }

  /**
   * Whether route mode can apply the plan's decision, and the dsh selection for it. Sets
   * `plan.target` when it can; returns `routed` then.
   * @param {Track} t
   * @param {Plan} plan
   */
  function routeReason(t, plan) {
    if (cfg.mode !== 'route') return 'shadow'
    if (plan.role !== 'after_tools') return plan.role === 'turn_start' ? 'turn_start' : 'no_tool_batch'
    if (plan.policy && plan.policy.startsWith('fail_open:')) return 'fail_open'
    if (t.subagent && !cfg.routeSubagents) return 'subagent_not_routed'
    if (!t.agent) return 'no_agent'
    const rec = plan.recommended
    const route = rec ? cfg.models.get(rec.model) : undefined
    if (!rec || !route) return 'unmapped_model'
    const effort = route.efforts[rec.effort]
    if (effort === undefined) return 'unmapped_effort'
    let registered = null
    try {
      registered = providers()
    } catch {
      registered = null
    }
    if (registered && !registered.includes(route.provider)) return 'provider_not_registered'
    plan.target = { provider: route.provider, model: route.model, reasoningEffort: effort }
    return 'routed'
  }

  /**
   * @param {Plan['role']} role
   * @param {number} stepIdx
   * @returns {Plan}
   */
  function newPlan(role, stepIdx) {
    return {
      role,
      decisionId: null,
      recommended: null,
      fallback: null,
      policy: null,
      target: null,
      applied: false,
      reason: '',
      decideMs: null,
      waitMs: null,
      late: false,
      applyDone: false,
      stepIdx,
      settled: Promise.resolve(),
    }
  }

  /**
   * Ask for the decision for the step after the current tool batch, and wait for it at most
   * until the decide timeout. Called from every root `tools/post-execute`; only the batch's
   * last call (in model order) asks, since by then every other result is in the log.
   * @param {Track} t
   * @param {any} agent
   * @param {any} result this call's result, not yet in the log
   */
  async function betweenSteps(t, agent, result) {
    const cur = t.cur
    if (!cur || cfg.mode === 'off') return
    if (cur.toolCallsAsked > 0 && cur.results < cur.toolCallsAsked - 1) return
    if (t.next && t.next.turn === cur.turn && t.next.afterStep === cur.step) {
      await waitFor(t.next)
      return
    }
    if (now() < backoffUntil) {
      stats.skipped += 1
      // The service is away: nothing is decided, and route mode falls back to the user's model.
      if (restoreBaseline(t)) stats.failOpen += 1
      return
    }
    const plan = /** @type {Plan & { turn: number, afterStep: number, startedAt: number }} */ (
      Object.assign(newPlan('after_tools', t.stepCount), { turn: cur.turn, afterStep: cur.step, startedAt: now() })
    )
    t.stepCount += 1
    t.next = plan
    const selection = currentOf(agent)
    const own = textOf(result && result.content)
    const text = (cur.resultText + (cur.resultText && own ? '\n' : '') + own).slice(0, MAX_TASK_TEXT)
    const paths = cur.paths.slice()
    const state = buildState(t, 'after_tools', selection, text, paths, cur.resultChars + own.length, plan.stepIdx)
    plan.settled = decide(t, plan, state).then(
      () => applyPlan(t, plan),
      () => contained('decide'),
    )
    // Shadow mode applies nothing, so the step does not wait for the answer.
    if (cfg.mode !== 'route') return
    await waitFor(plan)
    // No decision in hand when the step must start: it runs on the user's model.
    if (plan.late && restoreBaseline(t)) stats.failOpen += 1
  }

  /**
   * @param {Track} t
   * @param {Plan} plan
   */
  function applyPlan(t, plan) {
    try {
      plan.applyDone = true
      if (plan.late) {
        if (plan.reason === 'routed') plan.reason = 'late'
        return
      }
      if (plan.reason === 'routed' && plan.target) {
        if (select(t, plan.target)) {
          plan.applied = true
          t.pluginSel = plan.target
          stats.applied += 1
        } else {
          plan.reason = 'select_failed'
        }
      } else if (cfg.mode === 'route' && plan.reason !== 'shadow') {
        // No usable decision for this step: the user's model runs it.
        if (restoreBaseline(t)) stats.failOpen += 1
      }
    } catch {
      contained('apply')
    }
  }

  /**
   * Wait for a between-steps plan, at most until its deadline; past it the plan is late and
   * its decision, if it still comes, is logged but never applied.
   * @param {Plan & { startedAt?: number }} plan
   */
  async function waitFor(plan) {
    const started = hr()
    const left = (plan.startedAt ?? now()) + cfg.decideTimeoutMs + 10 - now()
    if (left > 0) {
      /** @type {ReturnType<typeof setTimeout> | undefined} */
      let timer
      await Promise.race([plan.settled, new Promise((resolve) => (timer = setTimeout(resolve, left)))])
      clearTimeout(timer)
    }
    if (!plan.applyDone) plan.late = true
    const waited = ms(started)
    plan.waitMs = Math.round(((plan.waitMs ?? 0) + waited) * 100) / 100
    remember(stats.waitMs, waited)
  }

  // --- handlers -------------------------------------------------------------------------

  /** `agent/created`: remember the user's selection before the plugin changes anything. */
  function agentCreated(/** @type {any} */ payload) {
    try {
      const agent = payload && payload.agent
      if (!agent || cfg.mode === 'off') return
      const t = trackOf(String(agent.id), agent)
      if (!t.baseline) t.baseline = currentOf(agent)
    } catch {
      contained('agent/created')
    }
  }

  /** `agent/disposed`: report what is open and forget the agent. */
  function agentDisposed(/** @type {any} */ payload) {
    try {
      const agent = payload && payload.agent
      const t = agent ? tracks.get(String(agent.id)) : undefined
      if (t) closeTrack(t, 'session_end')
    } catch {
      contained('agent/disposed')
    }
  }

  /**
   * `agent/pre-step`, after the rest of the waterfall: `modelSelection` is the selection this
   * step was fixed with (agent.ts:294,303). A step after a tool batch takes the plan decided
   * in post-execute; the first step of a turn gets a decision that is logged, not applied.
   * @param {any} payload
   * @param {any} decision the waterfall's result
   */
  function preStep(payload, decision) {
    try {
      if (cfg.mode === 'off' || !payload || !payload.agent) return
      const agent = payload.agent
      const t = trackOf(String(agent.id), agent)
      const selection = payload.modelSelection ? selectionOf(payload.modelSelection) : null
      if (!t.baseline && !t.pluginSel && selection) t.baseline = selection
      const turn = count(payload.turn)
      const step = count(payload.step)
      if (t.upcoming) dropPlan(t.upcoming.plan, 'no_step')
      t.upcoming = null
      const entered = decision && decision.kind === 'enter' && Array.isArray(decision.messages)
      let plan = null
      if (t.next && t.next.turn === turn) {
        plan = t.next
        t.next = null
        if (!entered) {
          dropPlan(plan, 'step_rejected')
          return
        }
      } else if (entered && (step > 1 || decision.messages.length > 0)) {
        // The first step of a turn (or a step after steering): its selection was read before
        // any awaitable hook, so the decision cannot change it. Ask without waiting.
        if (now() < backoffUntil) stats.skipped += 1
        else {
          let text = ''
          for (const m of decision.messages) {
            if (m && m.source && m.source.kind === 'user') text += (text ? '\n' : '') + textOf(m.content)
          }
          if (text) t.userText = text.slice(0, MAX_TASK_TEXT)
          const p = newPlan(step <= 1 ? 'turn_start' : 'unplanned', t.stepCount)
          t.stepCount += 1
          let chars = 0
          for (const m of decision.messages) chars += textOf(m && m.content).length
          const state = buildState(t, p.role, selection, text, extractPaths(text), chars, p.stepIdx)
          p.settled = decide(t, p, state).catch(() => contained('decide'))
          plan = p
        }
      }
      if (entered) t.upcoming = { turn, step, plan, selection }
    } catch {
      contained('agent/pre-step')
    }
  }

  /**
   * Root `tools/post-execute`, after the rest of the waterfall.
   * @param {any} exec
   * @param {any} result the dispatch result the waterfall received
   */
  async function postExecute(exec, result) {
    try {
      if (cfg.mode === 'off' || !exec || exec.parent !== undefined || !exec.agent) return
      const t = trackOf(String(exec.agent.id), exec.agent)
      await betweenSteps(t, exec.agent, result)
    } catch {
      contained('tools/post-execute')
    }
  }

  /**
   * `session/flush` (awaited by dsh; the headless runner flushes before it exits): for an
   * idle session, wait briefly for outcomes still on their way.
   * @param {any} session
   */
  async function sessionFlush(session) {
    try {
      const t = session ? tracks.get(String(session.id)) : undefined
      if (t && t.turnOpen) return
      if (t && t.done) {
        finish(t, t.done, 'session_end')
        t.done = null
      }
      if (retry.length && now() >= backoffUntil) flushRetry()
      if (inflight.size) {
        /** @type {ReturnType<typeof setTimeout> | undefined} */
        let timer
        await Promise.race([
          Promise.allSettled([...inflight]),
          new Promise((resolve) => (timer = setTimeout(resolve, FLUSH_WAIT_MS))),
        ])
        clearTimeout(timer)
      }
    } catch {
      contained('session/flush')
    }
  }

  /**
   * `session/event`: the session log as it is appended (dsh-session/src/index.ts:66-79).
   * Observe only; never append from here (a nested append throws, index.ts:405-418).
   * @param {any} session
   * @param {any} event
   */
  function sessionEvent(session, event) {
    try {
      if (cfg.mode === 'off' || !session || !event || typeof event.type !== 'string') return
      const t = trackOf(String(session.id), undefined, session)
      const d = event.data ?? {}
      switch (event.type) {
        case 'turn/start':
          t.turn = count(d.turn)
          t.turnOpen = true
          break
        case 'step/start': {
          if (t.done) {
            finish(t, t.done, 'continued')
            t.done = null
          }
          const up = t.upcoming && t.upcoming.turn === count(d.turn) && t.upcoming.step === count(d.step) ? t.upcoming : null
          t.upcoming = null
          t.cur = newStep(count(d.turn), count(d.step), up ? up.plan : null, up ? up.selection : null, t)
          if (typeof event.time === 'number') t.cur.startedAt = event.time
          stats.steps += 1
          break
        }
        case 'step/end':
          if (t.cur) {
            t.cur.endedAt = event.time ?? now()
            t.done = t.cur
            t.cur = null
          }
          break
        case 'turn/end': {
          t.turnOpen = false
          const kind = d.reason && typeof d.reason.kind === 'string' ? d.reason.kind : 'unknown'
          if (t.cur) {
            finish(t, t.cur, `turn_${kind}`)
            t.cur = null
          }
          if (t.done) {
            finish(t, t.done, `turn_${kind}`)
            t.done = null
          }
          if (t.upcoming) dropPlan(t.upcoming.plan, 'no_step')
          dropPlan(t.next, 'no_step')
          t.upcoming = null
          t.next = null
          break
        }
        case 'request/header': {
          const h = d.header ?? {}
          if (Array.isArray(h.tools)) {
            t.tools = h.tools.map((x) => (x && typeof x.name === 'string' ? x.name : '')).filter(Boolean)
          }
          // Logged only when the call config changes (agent.ts:676-697), so it is kept.
          if (h.config) t.headerEffort = typeof h.config.reasoningEffort === 'string' ? h.config.reasoningEffort : null
          break
        }
        case 'model/selection':
          // Not the plugin's own: the user (or another plugin) chose a model; that is the
          // choice to fall back to from now on.
          if (!t.selecting && typeof d.provider === 'string' && typeof d.model === 'string') {
            t.baseline = selectionOf(d)
            t.pluginSel = null
          }
          break
        case 'assistant/message':
          onAssistant(t, d)
          break
        case 'assistant/attempt':
          if (t.cur) t.cur.attempts += 1
          break
        case 'llm/retry':
          if (t.cur) t.cur.llmRetries += 1
          break
        case 'tool/call':
          onToolCall(t, d)
          break
        case 'tool/result':
          onToolResult(t, d)
          break
        case 'approval/asked':
          if (t.cur) t.cur.approvals.asked += 1
          break
        case 'approval/decided': {
          const a = t.cur ? t.cur.approvals : null
          if (a && d.outcome === 'allowed-once') a.allowed += 1
          else if (a && d.outcome === 'rejected') a.rejected += 1
          else if (a && d.outcome === 'cancelled') a.cancelled += 1
          else if (a && d.outcome === 'unavailable') a.unavailable += 1
          break
        }
        default:
          break
      }
    } catch {
      contained('session/event')
    }
  }

  /**
   * @param {number} turn
   * @param {number} step
   * @param {Plan | null} plan
   * @param {Selection | null} selection
   * @param {Track} t
   * @returns {Step}
   */
  function newStep(turn, step, plan, selection, t) {
    return {
      turn,
      step,
      plan,
      selection,
      startedAt: now(),
      endedAt: null,
      requests: 0,
      attempts: 0,
      usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, reasoning: 0 },
      executed: null,
      effort: null,
      toolCallsAsked: 0,
      toolCalls: 0,
      toolErrors: 0,
      denied: 0,
      aborted: 0,
      approvals: { asked: 0, allowed: 0, rejected: 0, cancelled: 0, unavailable: 0 },
      testRuns: 0,
      testsPassed: null,
      llmRetries: 0,
      interrupted: false,
      calls: new Map(),
      resultChars: 0,
      resultText: '',
      results: 0,
      paths: [],
      previousModel: t.last ? routeLabel(t.last) : null,
    }
  }

  /** @param {Track} t @param {any} d */
  function onAssistant(t, d) {
    const s = t.cur
    if (!s) return
    s.requests += 1
    const u = usageOf(d.usage)
    if (u) {
      s.usage.input += u.input
      s.usage.output += u.output
      s.usage.cacheRead += u.cacheRead
      s.usage.cacheWrite += u.cacheWrite
      s.usage.reasoning += u.reasoning
    }
    if (d.interrupted === true) s.interrupted = true
    s.effort = t.headerEffort
    s.toolCallsAsked += toolCallCount(d.message)
    // message.source is {provider, model} of the request that produced it (agent.ts:572-579):
    // the route that actually ran.
    const src = d.message && d.message.source
    if (src && typeof src.model === 'string') {
      s.executed = { provider: String(src.provider), model: src.model }
      t.last = {
        prompt: u ? u.input + u.cacheRead + u.cacheWrite : 0,
        output: u ? u.output : 0,
        provider: String(src.provider),
        model: src.model,
        effort: s.effort,
      }
    }
  }

  /** @param {Track} t @param {any} d */
  function onToolCall(t, d) {
    const s = t.cur
    if (!s || typeof d.callId !== 'string') return
    s.toolCalls += 1
    const name = typeof d.name === 'string' ? d.name : ''
    const shell = SHELL_TOOLS.has(name) ? shellCommandOf(d.arguments) : { command: null, background: false }
    if (s.calls.size < MAX_CALLS_PER_STEP) s.calls.set(d.callId, { name, command: shell.command, background: shell.background })
    if (s.paths.length < MAX_PATHS) pathsOfToolArgs(d.arguments, s.paths)
  }

  /** @param {Track} t @param {any} d */
  function onToolResult(t, d) {
    const s = t.cur
    if (!s) return
    const msg = d.message ?? {}
    const text = textOf(msg.content)
    s.results += 1
    s.resultChars += text.length
    if (s.resultText.length < MAX_TASK_TEXT) s.resultText = (s.resultText + (s.resultText ? '\n' : '') + text).slice(0, MAX_TASK_TEXT)
    const code = d.error && typeof d.error.code === 'string' ? d.error.code : null
    if (msg.isError === true) {
      if (code && DENIED_CODES.has(code)) s.denied += 1
      else if (code && ABORTED_CODES.has(code)) s.aborted += 1
      else if (!(code && INFRA_CODES.has(code))) s.toolErrors += 1
    }
    const callId = typeof msg.toolCallId === 'string' ? msg.toolCallId : msg.source && msg.source.callId
    const c = typeof callId === 'string' ? s.calls.get(callId) : undefined
    if (c && c.command && !c.background && isTestCommand(c.command) && msg.isError !== true) {
      const verdict = shellVerdict(text)
      if (verdict !== null) {
        s.testRuns += 1
        s.testsPassed = verdict
      }
    }
  }

  /** @param {Plan | null | undefined} plan @param {string} why */
  function dropPlan(plan, why) {
    if (!plan) return
    const p = plan
    track(
      p.settled.then(async () => {
        if (!p.decisionId) return
        await send(
          outcomeOf(cfg, p, null, why, now()),
        )
      }),
    )
  }

  /**
   * Report a finished step (after its decision settles). Never rejects.
   * @param {Track} t
   * @param {Step} s
   * @param {string} endReason
   */
  function finish(t, s, endReason) {
    const plan = s.plan
    if (!plan) return
    track(
      plan.settled.then(async () => {
        if (!plan.decisionId) return
        await send(outcomeOf(cfg, plan, s, endReason, now()))
      }),
    )
  }

  /** @param {Promise<unknown>} p */
  function track(p) {
    const q = p.catch(() => contained('report'))
    inflight.add(q)
    q.finally(() => inflight.delete(q))
  }

  /** @param {Record<string, unknown>} outcome */
  async function send(outcome) {
    if (now() < backoffUntil) {
      keep(outcome)
      return
    }
    const reply = await call('/report', outcome, cfg.reportTimeoutMs)
    if (reply.kind === 'http' && reply.status === 200) {
      stats.reported += 1
      if (retry.length) flushRetry()
    } else if (reply.kind !== 'http' || reply.status === 503 || reply.status >= 500) {
      keep(outcome)
    } else {
      stats.dropped += 1 // 4xx: resending the same body cannot help
    }
  }

  /** @param {Record<string, unknown>} outcome */
  function keep(outcome) {
    retry.push(outcome)
    while (retry.length > MAX_RETRY) {
      retry.shift()
      stats.dropped += 1
    }
  }

  function flushRetry() {
    for (const outcome of retry.splice(0, retry.length)) track(send(outcome))
  }

  function statusText() {
    if (cfg.mode === 'off') return 'steproute: off'
    const w = [...stats.waitMs].sort((a, b) => a - b)
    let text = `steproute: ${cfg.mode} · ${stats.decided} decisions · ${stats.reported} outcomes`
    if (cfg.mode === 'route') text += ` · ${stats.applied} applied · ${stats.restored} restored`
    if (w.length) text += ` · step wait median ${w[Math.floor(w.length / 2)]} ms`
    if (lastProblem) text += ` · ${lastProblem}`
    for (const warning of cfg.warnings) text += ` · config: ${warning}`
    return text
  }

  return {
    agentCreated,
    agentDisposed,
    preStep,
    postExecute,
    sessionFlush,
    sessionEvent,
    statusText,
    log,
    /** For tests: the live state. */
    _debug: { tracks, retry, inflight, stats, cfg, get backoffUntil() { return backoffUntil } },
  }
}

/**
 * The Outcome for one step (docs/SCHEMA.md). `s` is null for a decision whose step never ran.
 * @param {Config} cfg
 * @param {Plan} plan
 * @param {Step | null} s
 * @param {string} endReason
 * @param {number} nowMs
 * @returns {Record<string, unknown>}
 */
export function outcomeOf(cfg, plan, s, endReason, nowMs) {
  const rec = plan.recommended
  const executedId = s && s.executed ? catalogIdOf(cfg, s.executed.provider, s.executed.model) : null
  const executedEffort = s && s.effort ? steprouteEffortOf(cfg, executedId, s.effort) : null
  /** @type {Record<string, unknown>} */
  const signals = {
    routing_mode: cfg.mode,
    step_role: plan.role,
    applied: plan.applied,
    apply_reason: plan.reason,
    recommended_model: rec ? rec.model : null,
    recommended_effort: rec ? rec.effort : null,
    decide_ms: plan.decideMs,
    // Time the step waited in tools/post-execute for this decision: the delay the plugin added.
    plugin_wait_ms: plan.waitMs,
    end_reason: endReason,
    step_ran: s !== null,
  }
  if (plan.fallback) signals.fallback_not_run = plan.fallback
  if (s) {
    Object.assign(signals, {
      dsh_turn: s.turn,
      dsh_step: s.step,
      selected_route: routeLabel(s.selection),
      selected_effort: s.selection ? s.selection.reasoningEffort ?? null : null,
      executed_route: routeLabel(s.executed),
      executed_effort: s.effort,
      selection_ran: Boolean(s.selection && s.executed && s.selection.provider === s.executed.provider && s.selection.model === s.executed.model),
      model_matches_recommendation: rec !== null && executedId === rec.model,
      effort_matches_recommendation: rec !== null && executedEffort === rec.effort,
      model_switched: s.previousModel !== null && s.executed !== null && s.previousModel !== routeLabel(s.executed),
      requests: s.requests,
      failed_attempts: s.attempts,
      llm_retry_count: s.llmRetries,
      tool_calls_requested: s.toolCallsAsked,
      tool_call_count: s.toolCalls,
      tool_error_count: s.toolErrors,
      tool_denied_count: s.denied,
      tool_aborted_count: s.aborted,
      approvals_asked: s.approvals.asked,
      approvals_allowed: s.approvals.allowed,
      approvals_rejected: s.approvals.rejected,
      approvals_cancelled: s.approvals.cancelled,
      approvals_unavailable: s.approvals.unavailable,
      test_runs: s.testRuns,
      reasoning_tokens: s.usage.reasoning,
    })
    if (s.interrupted) signals.interrupted = true
    // Labelling signals (docs/LABELS.md): facts only. A tool call of this step that failed
    // (not a denial, not a cancellation) is `tool_error`; the last test command it ran is
    // `tests_passed`; an approval the user rejected is `user_rejected` (dsh words it "the user
    // rejected tool ...", dsh-tools/src/pipeline.ts:467-468). Provider retries (llm/retry) are
    // transport failures, not a verdict on the step, so they are not `retry_count`.
    if (s.toolErrors > 0) signals.tool_error = true
    if (s.testsPassed !== null) signals.tests_passed = s.testsPassed
    if (s.approvals.rejected > 0) signals.user_rejected = true
  }
  const u = s ? s.usage : { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
  const latency = s && s.endedAt !== null ? Math.max(0, s.endedAt - s.startedAt) : null
  return {
    decision_id: plan.decisionId,
    ts: nowMs / 1000,
    success: null,
    signals,
    final_model: executedId,
    escalated: false,
    input_tokens: u.input,
    output_tokens: u.output,
    cache_read_tokens: u.cacheRead,
    cache_write_tokens: u.cacheWrite,
    latency_ms: latency,
  }
}

/**
 * @param {number[]} list
 * @param {number} value
 */
function remember(list, value) {
  list.push(value)
  if (list.length > MAX_SAMPLES) list.shift()
}
