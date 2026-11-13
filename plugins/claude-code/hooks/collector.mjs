// @ts-check
// The steproute collector for Claude Code: one decision per subagent spawn, one outcome per
// subagent run. register.js wires these handlers to the mod events; everything here takes
// the host interface `$` as an argument, so tests drive it with a fake. docs/CLAUDE-CODE.md.
//
// Rules every handler keeps:
// - never throw into the host, and never hold an event longer than the configured timeout;
// - change nothing unless the mode is `route`, and then only the spawn's model and the
//   effort of that subagent's own requests;
// - report what was observed, never a success nobody checked.

import { postJson, sleep } from './client.mjs'
import {
  actionLabel,
  buildState,
  isClaudeModel,
  isTestCommand,
  normaliseModel,
  resolveConfig,
  STEP_EFFORTS,
} from './state.mjs'

/** Open subagent runs kept at once; past it the oldest is reported as `evicted`. */
export const MAX_OPEN = 256
/** Finished runs remembered so a resumed run (SendMessage) adds to its own outcome. */
export const MAX_FINISHED = 128
/** Outcomes kept for resending while the service is away. */
export const MAX_RETRY = 64
/** Distinct executed models remembered per run. */
const MAX_MODELS = 8
/** After a timeout, refusal or 500: skip the service this long (ms). */
export const BACKOFF_MS = 30_000
/** After 401, 403, 404 or 405 (a configuration problem): skip it this long (ms). */
export const CONFIG_BACKOFF_MS = 300_000
/** Longest wait for unsent outcomes at session end (its hooks get 1.5 s in all). */
export const END_FLUSH_MS = 1000

/**
 * @typedef {import('./state.mjs').Config} Config
 * @typedef {import('./client.mjs').Host} Host
 * @typedef {{
 *   decisionId: string | null,
 *   recommended: { model: string, effort: string } | null,
 *   fallback: string | null,
 *   policy: string | null,
 *   apply: boolean,
 *   reason: string,
 *   decideMs: number | null,
 * }} Plan
 * @typedef {{ input: number, output: number, cacheRead: number, cacheWrite: number }} Usage
 * @typedef {{
 *   decisionId: string,
 *   agentId: string,
 *   plan: Plan,
 *   applied: boolean,
 *   mode: string,
 *   subagentType: string | null,
 *   background: boolean,
 *   spawnModel: string | null,
 *   effortTarget: string | null,
 *   effortSteps: number,
 *   stepEffort: string | null,
 *   usage: Usage,
 *   requests: number,
 *   models: string[],
 *   lastModel: string | null,
 *   turnUsage: Usage | null,
 *   turnModel: string | null,
 *   toolCalls: number,
 *   toolErrors: number,
 *   denied: number,
 *   testRuns: number,
 *   testsPassed: boolean | null,
 *   durationMs: number,
 *   runs: number,
 *   resumed: number,
 *   endReason: string | null,
 * }} Run
 */

/** @returns {Usage} */
const zero = () => ({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0 })

/** @param {unknown} n */
const count = (n) => (typeof n === 'number' && Number.isFinite(n) && n > 0 ? Math.floor(n) : 0)

/**
 * @param {Usage} into
 * @param {any} u an API usage object
 */
function addUsage(into, u) {
  into.input += count(u.input_tokens)
  into.output += count(u.output_tokens)
  into.cacheRead += count(u.cache_read_input_tokens)
  into.cacheWrite += count(u.cache_creation_input_tokens)
}

/**
 * @param {Record<string, unknown> | undefined} options the plugin's userConfig values
 * @param {{ now?: () => number }} [deps]
 */
export function createCollector(options, deps = {}) {
  const now = deps.now ?? (() => Date.now())
  /** @type {Config} */
  let cfg = resolveConfig(options)
  const ctx = { sessionId: /** @type {string | null} */ (null), ccVersion: /** @type {string | null} */ (null) }
  let stepIdx = 0
  /** @type {Map<string, Run>} */
  const open = new Map()
  /** @type {Map<string, Run>} */
  const finished = new Map()
  /** @type {Record<string, unknown>[]} */
  const retry = []
  /** @type {Set<Promise<void>>} */
  const inflight = new Set()
  let backoffUntil = 0
  /** @type {string | null} */
  let lastProblem = null
  const stats = {
    spawns: 0,
    decided: 0,
    applied: 0,
    reported: 0,
    unsent: 0,
    sensitive: 0,
    skipped: 0,
    decideMs: /** @type {number[]} */ ([]),
  }

  /** @param {Host} host */
  function showStatus(host) {
    try {
      host.status?.(statusText())
    } catch {
      // no surface (claude -p, the SDK): nothing to show
    }
  }

  function statusText() {
    if (cfg.mode === 'off') return 'steproute: off'
    let text = `steproute: ${cfg.mode} · ${stats.decided} logged`
    if (cfg.mode === 'route') text += ` · ${stats.applied} routed`
    if (lastProblem) text += ` · ${lastProblem}`
    return text
  }

  /** @param {number} ms @param {string} problem */
  function backOff(ms, problem) {
    backoffUntil = now() + ms
    lastProblem = problem
  }

  /**
   * @param {Host} host
   * @param {string} path
   * @param {unknown} body
   * @param {number} timeoutMs
   */
  async function call(host, path, body, timeoutMs) {
    const reply = await postJson(host, cfg, path, body, timeoutMs, now)
    if (reply.kind === 'timeout') backOff(BACKOFF_MS, 'service timed out')
    else if (reply.kind === 'unreachable') backOff(BACKOFF_MS, 'service unreachable')
    else if ([401, 403, 404, 405].includes(reply.status)) {
      backOff(CONFIG_BACKOFF_MS, `service refused (${reply.status})`)
    } else if (reply.status >= 500 && reply.status !== 503) {
      backOff(BACKOFF_MS, `service error (${reply.status})`)
    } else if (reply.status === 200) lastProblem = null
    return reply
  }

  // --- spawn ----------------------------------------------------------------------------

  /**
   * @param {Host} host
   * @param {import('./state.mjs').SpawnInput} e
   * @returns {Promise<Plan>}
   */
  async function decide(host, e) {
    /** @type {Plan} */
    const plan = {
      decisionId: null,
      recommended: null,
      fallback: null,
      policy: null,
      apply: false,
      reason: '',
      decideMs: null,
    }
    if (cfg.mode === 'off') return { ...plan, reason: 'off' }
    if (e.fork) return { ...plan, reason: 'fork' }
    if (e.isTeammate) return { ...plan, reason: 'teammate' }
    stats.spawns += 1
    if (now() < backoffUntil) {
      stats.skipped += 1
      return { ...plan, reason: 'service_backoff' }
    }
    if (ctx.sessionId === null) await loadSession(host)
    const started = now()
    const state = buildState(e, ctx, stepIdx++, cfg.mode)
    const reply = await call(host, '/decide', state, cfg.decideTimeoutMs)
    plan.decideMs = now() - started
    remember(stats.decideMs, plan.decideMs, 256)
    if (reply.kind !== 'http') return { ...plan, reason: `service_${reply.kind}` }
    if (reply.status === 503) {
      const sensitive = Boolean(reply.body && reply.body.sensitive)
      if (sensitive) stats.sensitive += 1
      return { ...plan, reason: sensitive ? 'sensitive_no_decision' : 'no_decision' }
    }
    const d = reply.body
    if (reply.status !== 200 || !d || typeof d.decision_id !== 'string' || !d.action) {
      if (reply.status !== 200) lastProblem = `service answered ${reply.status}`
      return { ...plan, reason: `service_${reply.status}` }
    }
    stats.decided += 1
    plan.decisionId = d.decision_id
    plan.policy = typeof d.policy === 'string' ? d.policy : null
    plan.recommended = { model: String(d.action.model), effort: String(d.action.effort) }
    plan.fallback = d.action.escalate_to ? actionLabel(d.action.escalate_to) : null
    plan.reason = routeReason(e, plan)
    plan.apply = plan.reason === 'routed'
    return plan
  }

  /**
   * Why route mode does or does not apply this decision; `routed` when it does.
   * @param {import('./state.mjs').SpawnInput} e
   * @param {Plan} plan
   */
  function routeReason(e, plan) {
    if (cfg.mode !== 'route') return 'shadow'
    if (plan.policy && plan.policy.startsWith('fail_open:')) return 'fail_open'
    if (!plan.recommended || !isClaudeModel(plan.recommended.model)) return 'not_claude_model'
    if (typeof e.model === 'string' && e.model) return 'explicit_model'
    const type = typeof e.subagentType === 'string' ? e.subagentType : ''
    if (cfg.routeTypes !== '*' && !cfg.routeTypes.includes(type)) return 'agent_type_not_routed'
    return 'routed'
  }

  /**
   * The `agent.spawn` hook.
   * @param {Host} host
   * @param {any} e
   * @param {(e: any) => Promise<any>} next
   */
  async function spawn(host, e, next) {
    if (e && e.resume) {
      try {
        resume(e.resume.agentId)
      } catch {
        // bookkeeping only
      }
      return next(e)
    }
    /** @type {Plan} */
    let plan
    try {
      plan = await decide(host, e)
    } catch {
      plan = {
        decisionId: null,
        recommended: null,
        fallback: null,
        policy: null,
        apply: false,
        reason: 'plugin_error',
        decideMs: null,
      }
    }
    let res
    if (plan.apply && plan.recommended) {
      try {
        res = await next({ ...e, model: plan.recommended.model })
      } catch {
        // The engine refused the model (availableModels, say): start with its own choice.
        plan.apply = false
        plan.reason = 'model_refused'
        res = await next(e)
      }
    } else {
      res = await next(e)
    }
    try {
      started(host, plan, e, res)
    } catch {
      // bookkeeping only
    }
    return res
  }

  /**
   * @param {Host} host
   * @param {Plan} plan
   * @param {any} e
   * @param {any} res
   */
  function started(host, plan, e, res) {
    if (plan.apply) stats.applied += 1
    showStatus(host)
    if (!plan.decisionId) return
    /** @type {Run} */
    const run = {
      decisionId: plan.decisionId,
      agentId: res && typeof res.agentId === 'string' ? res.agentId : '',
      plan,
      applied: plan.apply,
      mode: cfg.mode,
      subagentType: typeof e.subagentType === 'string' ? e.subagentType : null,
      background: e.background === true,
      spawnModel: res && typeof res.model === 'string' ? res.model : null,
      effortTarget:
        plan.apply && plan.recommended && STEP_EFFORTS.includes(plan.recommended.effort)
          ? plan.recommended.effort
          : null,
      effortSteps: 0,
      stepEffort: null,
      usage: zero(),
      requests: 0,
      models: [],
      lastModel: null,
      turnUsage: null,
      turnModel: null,
      toolCalls: 0,
      toolErrors: 0,
      denied: 0,
      testRuns: 0,
      testsPassed: null,
      durationMs: 0,
      runs: 0,
      resumed: 0,
      endReason: null,
    }
    if (!run.agentId || (res && res.deny)) {
      // Nothing started (a hook refused it, or a remote agent with no local loop): close
      // the decision with what is known so it does not dangle.
      run.endReason = res && res.deny ? 'spawn_denied' : 'no_local_loop'
      report(host, run)
      return
    }
    open.set(run.agentId, run)
    while (open.size > MAX_OPEN) {
      const [oldest] = open.values()
      open.delete(oldest.agentId)
      oldest.endReason = 'evicted'
      report(host, oldest)
    }
  }

  /** @param {string} agentId */
  function resume(agentId) {
    const run = finished.get(agentId)
    if (!run) return
    finished.delete(agentId)
    run.resumed += 1
    open.set(agentId, run)
  }

  // --- per-request usage and tools --------------------------------------------------------

  /**
   * The `turn.step` input to send down: the routed subagent's effort, else `e` unchanged.
   * @param {any} e
   */
  function stepInput(e) {
    try {
      const run = e && e.agentId ? open.get(e.agentId) : undefined
      if (run) run.stepEffort = typeof e.effort === 'string' ? e.effort : e.effort === undefined ? null : String(e.effort)
      if (run && run.effortTarget && e.effort !== run.effortTarget) {
        run.effortSteps += 1
        return { ...e, effort: run.effortTarget }
      }
    } catch {
      // fall through
    }
    return e
  }

  /**
   * @param {any} e the `turn.step` input
   * @param {any} result the step's result
   */
  function stepResult(e, result) {
    try {
      const run = e && e.agentId ? open.get(e.agentId) : undefined
      if (!run || !result) return
      run.requests += 1
      const u = result.usage
      if (!u) return
      addUsage(run.usage, u)
      if (typeof u.model === 'string' && u.model) {
        run.lastModel = u.model
        if (!run.models.includes(u.model) && run.models.length < MAX_MODELS) run.models.push(u.model)
      }
    } catch {
      // bookkeeping only
    }
  }

  /**
   * The `tool.call` hook for a subagent's own tools (register.js keeps the spawning tools
   * out of its matcher, so no subagent ever starts inside this hook).
   * @param {any} e
   * @param {(e: any) => Promise<any>} next
   */
  async function toolCall(e, next) {
    const run = e && e.agentId ? open.get(e.agentId) : undefined
    if (!run) return next(e)
    const res = await next(e)
    try {
      run.toolCalls += 1
      if (res && res.deny) run.denied += 1
      else if (res && res.isError) run.toolErrors += 1
      if (e.tool === 'Bash' && isTestCommand(e.command) && res && !res.deny) {
        const r = res.result
        const finishedHere = !(r && typeof r === 'object' && (r.interrupted || r.backgroundTaskId))
        if (res.isError || finishedHere) {
          run.testRuns += 1
          run.testsPassed = !res.isError
        }
      }
    } catch {
      // bookkeeping only
    }
    return res
  }

  // --- end of a run -----------------------------------------------------------------------

  /**
   * The `turn.complete` hook: a subagent's run ended (each run of its loop is one turn).
   * @param {Host} host
   * @param {any} e
   * @param {(e: any) => Promise<any>} next
   */
  async function turnComplete(host, e, next) {
    const result = await next(e)
    try {
      const run = e && e.agentId ? open.get(e.agentId) : undefined
      if (run) {
        open.delete(run.agentId)
        run.runs += 1
        run.durationMs += count(e.durationMs)
        run.endReason = typeof e.reason === 'string' ? e.reason : 'answer'
        if (e.usage) {
          run.turnUsage = run.turnUsage ?? zero()
          addUsage(run.turnUsage, e.usage)
          if (typeof e.usage.model === 'string') run.turnModel = e.usage.model
        }
        finished.set(run.agentId, run)
        while (finished.size > MAX_FINISHED) {
          const [oldest] = finished.keys()
          finished.delete(oldest)
        }
        await bounded(host, report(host, run), cfg.reportTimeoutMs)
      }
    } catch {
      // bookkeeping only
    }
    return result
  }

  /**
   * The outcome record for a run (docs/SCHEMA.md, Outcome).
   * @param {Run} run
   */
  function outcomeOf(run) {
    const viaSteps = run.requests > 0
    const u = viaSteps ? run.usage : run.turnUsage ?? zero()
    const executedRaw = (viaSteps ? run.lastModel : run.turnModel) ?? run.spawnModel
    const executed = normaliseModel(executedRaw)
    const rec = run.plan.recommended
    /** @type {Record<string, unknown>} */
    const signals = {
      routing_mode: run.mode,
      applied: run.applied,
      apply_reason: run.plan.reason,
      recommended_model: rec ? rec.model : null,
      recommended_effort: rec ? rec.effort : null,
      executed_model: executed,
      executed_models: run.models.map((m) => normaliseModel(m)),
      spawn_model: run.spawnModel,
      model_matches_recommendation: rec !== null && executed === normaliseModel(rec.model),
      subagent_type: run.subagentType,
      background: run.background,
      end_reason: run.endReason,
      requests: run.requests,
      usage_source: viaSteps ? 'turn.step' : run.turnUsage ? 'turn.complete' : 'none',
      tool_call_count: run.toolCalls,
      tool_error_count: run.toolErrors,
      tool_denied_count: run.denied,
      billing: cfg.billing,
      decide_ms: run.plan.decideMs,
    }
    if (run.turnUsage && viaSteps) {
      const t = run.turnUsage
      signals.turn_usage_matches =
        t.input === u.input && t.output === u.output && t.cacheRead === u.cacheRead && t.cacheWrite === u.cacheWrite
    }
    if (run.plan.fallback) signals.fallback_not_run = run.plan.fallback
    if (run.effortTarget) signals.effort_rewrites = run.effortSteps
    // The effort the engine had set for the subagent's last request, before any rewrite.
    if (run.requests > 0) signals.engine_effort = run.stepEffort
    if (run.resumed) signals.resumed_runs = run.resumed
    // Labelling signals (docs/LABELS.md): facts only. A tool call that errored is
    // `tool_error`; a test command's result is `tests_passed`. Nothing else is claimed.
    if (run.toolErrors > 0) signals.tool_error = true
    if (run.testsPassed !== null) signals.tests_passed = run.testsPassed
    /** @type {Record<string, unknown>} */
    const outcome = {
      decision_id: run.decisionId,
      ts: now() / 1000,
      success: null,
      signals,
      final_model: executed,
      escalated: false,
      input_tokens: u.input,
      output_tokens: u.output,
      cache_read_tokens: u.cacheRead,
      cache_write_tokens: u.cacheWrite,
      latency_ms: run.durationMs > 0 ? run.durationMs : null,
    }
    // On a subscription a request has no per-request charge. The API-price equivalent is
    // recomputable from the tokens and final_model; it is not written as money spent.
    if (cfg.billing === 'subscription') outcome.cost_usd = 0
    return outcome
  }

  /**
   * Send the run's outcome; keep it for later if the service is away. Never rejects.
   * @param {Host} host
   * @param {Run} run
   * @returns {Promise<void>}
   */
  function report(host, run) {
    const outcome = outcomeOf(run)
    const p = send(host, outcome)
    inflight.add(p)
    p.finally(() => inflight.delete(p))
    return p
  }

  /**
   * @param {Host} host
   * @param {Record<string, unknown>} outcome
   * @returns {Promise<void>}
   */
  async function send(host, outcome) {
    try {
      if (now() < backoffUntil) {
        keep(outcome)
        return
      }
      const reply = await call(host, '/report', outcome, cfg.reportTimeoutMs)
      if (reply.kind === 'http' && reply.status === 200) {
        stats.reported += 1
        flushRetry(host)
      } else if (reply.kind !== 'http' || reply.status === 503 || reply.status >= 500) {
        keep(outcome)
      } else {
        stats.unsent += 1 // 4xx: resending the same body cannot help
      }
    } catch {
      keep(outcome)
    }
  }

  /** @param {Record<string, unknown>} outcome */
  function keep(outcome) {
    retry.push(outcome)
    while (retry.length > MAX_RETRY) {
      retry.shift()
      stats.unsent += 1
    }
  }

  /** @param {Host} host */
  function flushRetry(host) {
    const batch = retry.splice(0, retry.length)
    for (const outcome of batch) {
      const p = send(host, outcome)
      inflight.add(p)
      p.finally(() => inflight.delete(p))
    }
  }

  /**
   * Wait for `p`, at most `ms`.
   * @param {Host} host
   * @param {Promise<unknown>} p
   * @param {number} ms
   */
  async function bounded(host, p, ms) {
    const stop = new AbortController()
    await Promise.race([p, sleep(host, ms, stop.signal)])
    stop.abort()
  }

  // --- session ----------------------------------------------------------------------------

  /** @param {Host} host */
  async function loadSession(host) {
    try {
      ctx.sessionId = host.sessionId ? String(await host.sessionId()) : 'unknown'
    } catch {
      ctx.sessionId = ctx.sessionId ?? 'unknown'
    }
    try {
      const v = host.version ? await host.version() : null
      ctx.ccVersion = v && typeof v.version === 'string' ? v.version : null
    } catch {
      ctx.ccVersion = null
    }
  }

  /**
   * The `session.start` hook, after the engine's own start. `env` holds the STEPROUTE_*
   * variables register.js read (the validator wants each name spelled as a literal there).
   * @param {Host} host
   * @param {import('./state.mjs').EnvOverrides} env
   */
  async function sessionStart(host, env) {
    cfg = resolveConfig(options, env)
    await loadSession(host)
    try {
      await host.registerCommand?.({
        name: 'steproute',
        description: 'steproute: the mode, the service and what was logged this session',
      })
    } catch {
      // no command surface
    }
    try {
      host.log?.(statusText())
    } catch {
      // no log
    }
    showStatus(host)
  }

  /**
   * The `session.end` hook: report runs still open, then wait briefly for unsent outcomes.
   * @param {Host} host
   * @param {any} e
   * @param {(e: any) => Promise<any>} next
   */
  async function sessionEnd(host, e, next) {
    try {
      for (const run of [...open.values()]) {
        open.delete(run.agentId)
        run.endReason = 'session_end'
        report(host, run)
      }
      if (retry.length && now() >= backoffUntil) flushRetry(host)
      if (inflight.size) await bounded(host, Promise.allSettled([...inflight]), END_FLUSH_MS)
    } catch {
      // nothing more to do
    }
    return next(e)
  }

  function commandText() {
    const ms = [...stats.decideMs].sort((a, b) => a - b)
    const median = ms.length ? ms[Math.floor(ms.length / 2)] : null
    const lines = [
      `steproute ${cfg.mode} mode, service ${cfg.url}`,
      cfg.mode === 'shadow'
        ? 'Decisions and outcomes are logged; every subagent keeps the model Claude Code chose.'
        : cfg.mode === 'route'
          ? 'Decisions are logged and applied when a subagent starts (model, then effort).'
          : 'Off: nothing is logged or changed.',
      `subagent spawns ${stats.spawns}, decisions logged ${stats.decided}, applied ${stats.applied}, ` +
        `outcomes stored ${stats.reported}, waiting ${retry.length}, dropped ${stats.unsent}`,
      `sensitive spawns with no decision ${stats.sensitive}, skipped while the service was away ${stats.skipped}`,
      `open runs ${open.size}`,
      median === null ? 'decision delay: none measured yet' : `decision delay: median ${median} ms, max ${ms[ms.length - 1]} ms over ${ms.length}`,
    ]
    if (lastProblem) lines.push(`last problem: ${lastProblem}`)
    for (const w of cfg.warnings) lines.push(`config: ${w}`)
    return lines.join('\n')
  }

  return {
    spawn,
    stepInput,
    stepResult,
    toolCall,
    turnComplete,
    sessionStart,
    sessionEnd,
    commandText,
    statusText,
    /** For tests: the live state. */
    _debug: { open, finished, retry, inflight, stats, ctx, get cfg() { return cfg } },
  }
}

/**
 * @param {number[]} list
 * @param {number} value
 * @param {number} max
 */
function remember(list, value, max) {
  list.push(value)
  if (list.length > max) list.shift()
}
