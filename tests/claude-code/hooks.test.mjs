// Unit tests for the Claude Code plugin's hooks module (plugins/claude-code/hooks).
// Run: node --test tests/claude-code/   (scripts/ci.sh does, when Node is installed)

import assert from 'node:assert/strict'
import { afterEach, describe, test } from 'node:test'
import { MAX_OPEN } from '../../plugins/claude-code/hooks/collector.mjs'
import {
  buildState,
  extractPaths,
  isClaudeModel,
  isTestCommand,
  normaliseModel,
  resolveConfig,
} from '../../plugins/claude-code/hooks/state.mjs'
import { decisionFor, fakeDollar, fakeService, loadPlugin, SPAWN, usage, wait } from './harness.mjs'

let services = []
async function service(answer) {
  const s = await fakeService(answer)
  services.push(s)
  return s
}
afterEach(async () => {
  await Promise.all(services.map((s) => s.close()))
  services = []
})

/** A plugin pointed at `svc`, started. */
async function plugin(svc, options = {}, env = {}) {
  const p = loadPlugin({ service_url: svc.url, ...options }, fakeDollar({ env }))
  await p.start()
  return p
}

describe('shadow mode (the default)', () => {
  test('logs a decision, changes nothing, reports usage as observed', async () => {
    const svc = await service()
    const p = await plugin(svc)
    const { seen } = await p.spawn(SPAWN)
    assert.equal(seen.model, undefined, 'the spawn reaches the engine unchanged')
    assert.deepEqual(seen, SPAWN)
    assert.equal(svc.decides().length, 1)
    const state = svc.decides()[0].body
    assert.equal(state.harness, 'claude-code')
    assert.equal(state.kind, 'subagent')
    assert.equal(state.session_id, 'sess-1')
    assert.equal(state.cached_prefix_tokens, 0)
    assert.equal(state.current_model, null)
    assert.equal(state.task_text, SPAWN.prompt)
    assert.deepEqual(state.features.paths, ['notes/a.txt'])

    const step1 = await p.step('a1', usage(10, 20, 300, 40))
    assert.equal(step1.seen.effort, undefined, 'shadow mode rewrites no effort')
    await p.step('a1', usage(5, 7, 340, 0))
    await p.complete('a1', usage(15, 27, 640, 40))
    const [r] = svc.reports()
    assert.ok(r, 'one outcome reported')
    const o = r.body
    assert.equal(o.decision_id, 'd1')
    assert.equal(o.success, null, 'no evidence, no label')
    assert.deepEqual(
      [o.input_tokens, o.output_tokens, o.cache_read_tokens, o.cache_write_tokens],
      [15, 27, 640, 40],
    )
    assert.equal(o.final_model, 'claude-haiku-5-5')
    assert.equal(o.cost_usd, 0, 'subscription: no per-request charge')
    assert.equal(o.escalated, false)
    assert.equal(o.latency_ms, 1234)
    assert.equal(o.signals.applied, false)
    assert.equal(o.signals.apply_reason, 'shadow')
    assert.equal(o.signals.recommended_model, 'claude-sonnet-5-5')
    assert.equal(o.signals.model_matches_recommendation, false)
    assert.equal(o.signals.requests, 2)
    assert.equal(o.signals.usage_source, 'turn.step')
    assert.equal(o.signals.turn_usage_matches, true)
    assert.equal(o.signals.billing, 'subscription')
    assert.equal('tool_error' in o.signals, false)
    assert.equal(typeof o.signals.decide_ms, 'number')
  })

  test('the status line names the mode', async () => {
    const svc = await service()
    const p = await plugin(svc)
    await p.spawn(SPAWN)
    assert.match(p.dollar.status.at(-1), /^steproute: shadow · 1 logged/)
    const { text } = await p.command()
    assert.match(text, /steproute shadow mode/)
    assert.match(text, /decisions logged 1/)
  })
})

describe('route mode', () => {
  test('applies the model at the spawn and the effort on that subagent only', async () => {
    const svc = await service()
    const p = await plugin(svc, { mode: 'route' })
    const { seen } = await p.spawn(SPAWN, (x) => ({ model: x.model, agentId: 'a1' }))
    assert.equal(seen.model, 'claude-sonnet-5-5')
    const sub = await p.step('a1', usage(1, 2, 0, 0, 'claude-sonnet-5-5'), { effort: 'high' })
    assert.equal(sub.seen.effort, 'medium')
    const main = await p.step(undefined, usage(1, 2), { effort: 'high' })
    assert.equal(main.seen.effort, 'high', 'the main thread is never touched')
    const other = await p.step('zz', usage(1, 2), { effort: 'high' })
    assert.equal(other.seen.effort, 'high', 'an untracked subagent is never touched')
    await p.complete('a1', usage(1, 2, 0, 0, 'claude-sonnet-5-5'))
    const o = svc.reports()[0].body
    assert.equal(o.signals.applied, true)
    assert.equal(o.signals.apply_reason, 'routed')
    assert.equal(o.signals.model_matches_recommendation, true)
    assert.equal(o.signals.effort_rewrites, 1)
    assert.equal(o.signals.engine_effort, 'high')
    assert.match(p.dollar.status.at(-1), /route · 1 logged · 1 routed/)
  })

  const cases = [
    ['an explicit per-call model', { ...SPAWN, model: 'opus' }, undefined, 'explicit_model'],
    ['an agent type that may pin its own model', { ...SPAWN, subagentType: 'my-reviewer' }, undefined, 'agent_type_not_routed'],
    ['a fail-open decision', SPAWN, 'fail_open:timeout', 'fail_open'],
  ]
  for (const [name, spawn, policy, reason] of cases) {
    test(`leaves ${name} alone, and still logs it`, async () => {
      const svc = await service((path, body) =>
        path === '/decide' && policy ? { body: decisionFor(body, undefined, policy) } : {},
      )
      const p = await plugin(svc, { mode: 'route' })
      const { seen } = await p.spawn(spawn)
      assert.deepEqual(seen, spawn)
      await p.complete('a1', usage(1, 1))
      assert.equal(svc.reports()[0].body.signals.apply_reason, reason)
      assert.equal(svc.reports()[0].body.signals.applied, false)
    })
  }

  test('never applies a model Claude Code cannot run', async () => {
    const svc = await service((path, body) =>
      path === '/decide' ? { body: decisionFor(body, { model: 'deepseek-flash', effort: 'low', escalate_to: null }) } : {},
    )
    const p = await plugin(svc, { mode: 'route' })
    const { seen } = await p.spawn(SPAWN)
    assert.deepEqual(seen, SPAWN)
    await p.complete('a1', usage(1, 1))
    assert.equal(svc.reports()[0].body.signals.apply_reason, 'not_claude_model')
  })

  test('route_agent_types "*" routes every type', async () => {
    const svc = await service()
    const p = await plugin(svc, { mode: 'route', route_agent_types: '*' })
    const { seen } = await p.spawn({ ...SPAWN, subagentType: 'my-reviewer' })
    assert.equal(seen.model, 'claude-sonnet-5-5')
  })

  test('a model the engine refuses falls back to its own choice', async () => {
    const svc = await service()
    const p = await plugin(svc, { mode: 'route' })
    const calls = []
    const res = await p.fire('agent.spawn', SPAWN, async (x) => {
      calls.push(x)
      if (x.model) throw new Error('model not available')
      return { model: 'claude-haiku-5-5', agentId: 'a1' }
    })
    assert.equal(res.agentId, 'a1')
    assert.equal(calls.length, 2)
    assert.equal(calls[1].model, undefined)
    await p.complete('a1', usage(1, 1))
    assert.equal(svc.reports()[0].body.signals.apply_reason, 'model_refused')
  })

  test('a fallback step is recorded, never run', async () => {
    const action = {
      model: 'claude-haiku-5-5',
      effort: 'low',
      escalate_to: { model: 'claude-sonnet-5-5', effort: 'low', escalate_to: null },
    }
    const svc = await service((path, body) => (path === '/decide' ? { body: decisionFor(body, action) } : {}))
    const p = await plugin(svc, { mode: 'route' })
    const { seen } = await p.spawn(SPAWN)
    assert.equal(seen.model, 'claude-haiku-5-5')
    await p.complete('a1', usage(1, 1))
    const o = svc.reports()[0].body
    assert.equal(o.escalated, false)
    assert.equal(o.signals.fallback_not_run, 'claude-sonnet-5-5 (low)')
  })
})

describe('fail-open', () => {
  test('a slow service: the spawn goes ahead unchanged within the timeout, then backs off', async () => {
    const svc = await service(() => ({ delayMs: 1000 }))
    const p = await plugin(svc, { mode: 'route', decide_timeout_ms: 60 })
    const t0 = Date.now()
    const { seen } = await p.spawn(SPAWN)
    const took = Date.now() - t0
    assert.deepEqual(seen, SPAWN)
    assert.ok(took < 400, `spawn held ${took} ms`)
    await p.spawn({ ...SPAWN, tool_use_id: 'toolu_2' }, { model: 'm', agentId: 'a2' })
    await wait(20)
    assert.equal(svc.decides().length, 1, 'no second call while backing off')
    assert.match(p.dollar.status.at(-1), /service timed out/)
    await p.complete('a1', usage(1, 1))
    assert.equal(svc.reports().length, 0, 'nothing to report without a decision')
  })

  test('an unreachable service', async () => {
    const svc = await service()
    const url = svc.url
    await svc.close()
    services = []
    const p = loadPlugin({ service_url: url, mode: 'route' })
    await p.start()
    const { seen } = await p.spawn(SPAWN)
    assert.deepEqual(seen, SPAWN)
    assert.match(p.dollar.status.at(-1), /unreachable/)
  })

  test('a fetch that throws synchronously never reaches the host', async () => {
    const dollar = fakeDollar({
      fetch: () => {
        throw new Error('boom')
      },
    })
    const p = loadPlugin({ mode: 'route' }, dollar)
    await p.start()
    const { seen, res } = await p.spawn(SPAWN)
    assert.deepEqual(seen, SPAWN)
    assert.equal(res.agentId, 'a1')
  })

  for (const status of [400, 401, 403, 404, 405, 411, 413, 415, 500]) {
    test(`HTTP ${status} from /decide: unchanged, nothing reported`, async () => {
      const svc = await service((path) =>
        path === '/decide' ? { status, body: { error: 'x', message: 'y' } } : {},
      )
      const p = await plugin(svc, { mode: 'route' })
      const { seen } = await p.spawn(SPAWN)
      assert.deepEqual(seen, SPAWN)
      await p.complete('a1', usage(1, 1))
      assert.equal(svc.reports().length, 0)
      await p.spawn({ ...SPAWN, tool_use_id: 'toolu_2' }, { model: 'm', agentId: 'a2' })
      const backsOff = [401, 403, 404, 405, 500].includes(status)
      assert.equal(svc.decides().length, backsOff ? 1 : 2)
    })
  }

  for (const sensitive of [true, false]) {
    test(`HTTP 503 with sensitive=${sensitive}: Claude Code's own choice, nothing logged`, async () => {
      const svc = await service((path) =>
        path === '/decide' ? { status: 503, body: { error: 'no_action_allowed', message: 'm', sensitive } } : {},
      )
      const p = await plugin(svc, { mode: 'route' })
      const { seen } = await p.spawn(SPAWN)
      assert.deepEqual(seen, SPAWN)
      await p.complete('a1', usage(1, 1))
      assert.equal(svc.reports().length, 0)
      const { text } = await p.command()
      assert.match(text, new RegExp(`sensitive spawns with no decision ${sensitive ? 1 : 0}`))
      await p.spawn(SPAWN)
      assert.equal(svc.decides().length, 2, 'a 503 is an answer, not an outage')
    })
  }

  test('an outcome the service cannot store is kept and resent', async () => {
    let failReports = true
    const svc = await service((path) => (path === '/report' && failReports ? { status: 503, body: { error: 'store_unavailable' } } : {}))
    const p = await plugin(svc)
    await p.spawn(SPAWN)
    await p.complete('a1', usage(3, 4))
    assert.equal(svc.reports().length, 1)
    failReports = false
    await p.spawn({ ...SPAWN, tool_use_id: 'toolu_2' }, { model: 'claude-haiku-5-5', agentId: 'a2' })
    await p.complete('a2', usage(5, 6))
    await wait(50)
    const ids = svc.reports().map((r) => r.body.decision_id)
    assert.deepEqual([...ids].sort(), ['d1', 'd1', 'd2'])
    assert.equal(ids.filter((i) => i === 'd1').length, 2, 'd1 was resent')
  })
})

describe('usage accounting', () => {
  test('per-request usage is summed per subagent; parallel subagents stay apart', async () => {
    const svc = await service()
    const p = await plugin(svc)
    await p.spawn(SPAWN, { model: 'claude-haiku-5-5', agentId: 'a1' })
    await p.spawn({ ...SPAWN, tool_use_id: 'toolu_2', background: true }, { model: 'claude-haiku-5-5', agentId: 'a2' })
    await p.step('a1', usage(1, 1, 100, 10))
    await p.step('a2', usage(2, 2, 200, 20))
    await p.step('a1', usage(1, 1, 100, 0))
    await p.step(undefined, usage(1000, 1000), {}) // the main thread is not counted
    await p.complete('a2', usage(2, 2, 200, 20))
    await p.complete('a1', usage(2, 2, 200, 10))
    const byId = Object.fromEntries(svc.reports().map((r) => [r.body.decision_id, r.body]))
    assert.deepEqual([byId.d1.input_tokens, byId.d1.cache_read_tokens, byId.d1.cache_write_tokens], [2, 200, 10])
    assert.deepEqual([byId.d2.input_tokens, byId.d2.cache_read_tokens, byId.d2.cache_write_tokens], [2, 200, 20])
    assert.equal(byId.d2.signals.background, true)
  })

  test('with no turn.step seen, the turn totals are used and said so', async () => {
    const svc = await service()
    const p = await plugin(svc)
    await p.spawn(SPAWN)
    await p.complete('a1', usage(7, 8, 9, 10, 'claude-haiku-5-5-20261101'))
    const o = svc.reports()[0].body
    assert.deepEqual([o.input_tokens, o.output_tokens, o.cache_read_tokens, o.cache_write_tokens], [7, 8, 9, 10])
    assert.equal(o.signals.usage_source, 'turn.complete')
    assert.equal(o.final_model, 'claude-haiku-5-5')
  })

  test('the executed model is what the API reported, not the recommendation', async () => {
    const svc = await service()
    const p = await plugin(svc)
    await p.spawn(SPAWN)
    await p.step('a1', usage(1, 1, 0, 0, 'claude-opus-5-5'))
    await p.complete('a1', usage(1, 1, 0, 0, 'claude-opus-5-5'))
    const o = svc.reports()[0].body
    assert.equal(o.final_model, 'claude-opus-5-5')
    assert.equal(o.signals.recommended_model, 'claude-sonnet-5-5')
    assert.deepEqual(o.signals.executed_models, ['claude-opus-5-5'])
  })

  test('a resumed subagent: no new decision, no model change, the outcome grows', async () => {
    const svc = await service()
    const p = await plugin(svc, { mode: 'route' })
    await p.spawn(SPAWN, (x) => ({ model: x.model, agentId: 'a1' }))
    await p.step('a1', usage(1, 1, 0, 0, 'claude-sonnet-5-5'))
    await p.complete('a1', usage(1, 1, 0, 0, 'claude-sonnet-5-5'))
    const resume = { ...SPAWN, prompt: 'more please', resume: { agentId: 'a1' } }
    const { seen } = await p.spawn(resume, { model: 'claude-sonnet-5-5', agentId: 'a1' })
    assert.deepEqual(seen, resume, 'a resumed run is passed through untouched')
    assert.equal(svc.decides().length, 1)
    await p.step('a1', usage(2, 2, 0, 0, 'claude-sonnet-5-5'))
    await p.complete('a1', usage(2, 2, 0, 0, 'claude-sonnet-5-5'))
    const reports = svc.reports().map((r) => r.body)
    assert.equal(reports.length, 2)
    assert.equal(reports[1].decision_id, reports[0].decision_id, 'the later outcome replaces the first')
    assert.equal(reports[1].input_tokens, 3)
    assert.equal(reports[1].signals.resumed_runs, 1)
    assert.equal(reports[1].latency_ms, 2468)
  })

  test('tool errors and test results from the subagent own tools', async () => {
    const svc = await service()
    const p = await plugin(svc)
    await p.spawn(SPAWN)
    await p.tool('a1', 'Read', { file_path: 'x' }, { isError: true, result: 'no such file', text: 'no such file' })
    await p.tool('a1', 'Bash', { command: 'uv run pytest -q' }, { isError: true, result: 'Exit code 1' })
    await p.tool('a1', 'Bash', { command: 'uv run pytest -q' }, { result: { stdout: 'ok', stderr: '', interrupted: false } })
    await p.tool('a1', 'Edit', { file_path: 'x' }, { deny: 'not allowed' })
    await p.tool(undefined, 'Bash', { command: 'pytest' }, { isError: true, result: 'main thread' })
    await p.complete('a1', usage(1, 1))
    const s = svc.reports()[0].body.signals
    assert.equal(s.tool_call_count, 4)
    assert.equal(s.tool_error_count, 2)
    assert.equal(s.tool_denied_count, 1)
    assert.equal(s.tool_error, true)
    assert.equal(s.tests_passed, true, 'the last test run decides')
  })

  test('the tool.call matcher keeps spawning tools out', () => {
    const p = loadPlugin()
    const h = p.hooks.find((x) => x.event === 'tool.call')
    for (const t of ['Agent', 'Task', 'SendMessage', 'Workflow', 'TeamCreate']) assert.equal(h.matcher.tool.test(t), false, t)
    for (const t of ['Read', 'Bash', 'AgentLike', 'mcp__x__Agent']) assert.equal(h.matcher.tool.test(t), true, t)
  })

  test('open runs are bounded; the oldest is reported as evicted', async () => {
    const svc = await service()
    const p = await plugin(svc)
    for (let i = 0; i <= MAX_OPEN; i++) {
      await p.spawn({ ...SPAWN, tool_use_id: `t${i}` }, { model: 'claude-haiku-5-5', agentId: `a${i}` })
    }
    await wait(50)
    const evicted = svc.reports().map((r) => r.body)
    assert.equal(evicted.length, 1)
    assert.equal(evicted[0].signals.end_reason, 'evicted')
  })

  test('session end reports runs still open', async () => {
    const svc = await service()
    const p = await plugin(svc)
    await p.spawn(SPAWN)
    await p.step('a1', usage(4, 5))
    await p.end()
    const o = svc.reports()[0].body
    assert.equal(o.signals.end_reason, 'session_end')
    assert.equal(o.input_tokens, 4)
    assert.equal(o.success, null)
  })

  test('a denied spawn closes its decision', async () => {
    const svc = await service()
    const p = await plugin(svc)
    await p.spawn(SPAWN, { deny: 'refused by a policy mod' })
    await wait(50)
    assert.equal(svc.reports()[0].body.signals.end_reason, 'spawn_denied')
  })
})

describe('sensitive tasks and privacy', () => {
  test('the prompt text and paths go to /decide for detection; the service decides', async () => {
    const svc = await service((path) =>
      path === '/decide' ? { status: 503, body: { error: 'no_action_allowed', message: 'm', sensitive: true } } : {},
    )
    const p = await plugin(svc, { mode: 'route' })
    const prompt = 'Check that ~/.ssh/id_ed25519 and config/.env are not committed.'
    const { seen } = await p.spawn({ ...SPAWN, prompt })
    assert.equal(seen.model, undefined)
    const state = svc.decides()[0].body
    assert.equal(state.task_text, prompt)
    assert.deepEqual(state.features.paths, ['~/.ssh/id_ed25519', 'config/.env'])
    assert.equal(state.sensitive, false, 'the plugin does not judge; the service does')
  })

  test('long prompts are cut before sending', () => {
    const s = buildState({ ...SPAWN, prompt: 'x'.repeat(150_000) }, { sessionId: 's', ccVersion: null }, 0, 'shadow')
    assert.equal(s.task_text.length, 100_000)
    assert.equal(s.features.task_text_truncated, true)
    assert.equal(s.prompt_tokens, 37_500)
  })
})

describe('configuration', () => {
  test('off mode makes no request and changes nothing', async () => {
    const svc = await service()
    const p = await plugin(svc, { mode: 'off' })
    const { seen } = await p.spawn(SPAWN)
    await p.step('a1', usage(1, 1))
    await p.complete('a1', usage(1, 1))
    assert.deepEqual(seen, SPAWN)
    assert.equal(svc.requests.length, 0)
    assert.equal(p.dollar.status.at(-1), 'steproute: off')
  })

  test('the environment overrides the plugin options', async () => {
    const svc = await service()
    const p = await plugin(svc, { mode: 'shadow' }, { STEPROUTE_MODE: 'route', STEPROUTE_TOKEN: 'tok' })
    const { seen } = await p.spawn(SPAWN)
    assert.equal(seen.model, 'claude-sonnet-5-5')
    assert.equal(svc.decides()[0].headers.authorization, 'Bearer tok')
  })

  test('a mistyped mode means shadow, with a warning', () => {
    const cfg = resolveConfig({ mode: 'rout' })
    assert.equal(cfg.mode, 'shadow')
    assert.match(cfg.warnings[0], /unknown mode/)
    assert.equal(resolveConfig({}, { mode: ' Route ' }).mode, 'route')
    assert.equal(resolveConfig({ service_url: 'not a url' }).url, 'http://127.0.0.1:8787')
    assert.equal(resolveConfig({ decide_timeout_ms: 1e9 }).decideTimeoutMs, 2000)
    assert.equal(resolveConfig({ billing: 'api' }).billing, 'api')
  })

  test('billing "api" leaves cost_usd for the service to price', async () => {
    const svc = await service()
    const p = await plugin(svc, { billing: 'api' })
    await p.spawn(SPAWN)
    await p.complete('a1', usage(1, 1))
    assert.equal('cost_usd' in svc.reports()[0].body, false)
  })

  test('forks and teammates are not routed or logged', async () => {
    const svc = await service()
    const p = await plugin(svc, { mode: 'route' })
    await p.spawn({ ...SPAWN, fork: true })
    await p.spawn({ ...SPAWN, isTeammate: true })
    assert.equal(svc.decides().length, 0)
  })
})

describe('helpers', () => {
  test('extractPaths', () => {
    assert.deepEqual(extractPaths('read src/app.py and README.md, then .env. See https://x.y/z e.g. this'), [
      'src/app.py',
      'README.md',
      '.env',
    ])
    assert.equal(extractPaths(Array.from({ length: 40 }, (_, i) => `d${i}/f`).join(' ')).length, 32)
  })

  test('isClaudeModel', () => {
    for (const m of ['claude-sonnet-5-5', 'haiku', 'claude-fable-5-2']) assert.ok(isClaudeModel(m), m)
    for (const m of ['deepseek-flash', 'qwen3-coder:30b', '', null]) assert.ok(!isClaudeModel(m), String(m))
  })

  test('normaliseModel', () => {
    assert.equal(normaliseModel('claude-opus-5-5[1m]'), 'claude-opus-5-5')
    assert.equal(normaliseModel('claude-haiku-5-5-20261101'), 'claude-haiku-5-5')
    assert.equal(normaliseModel(undefined), null)
  })

  test('isTestCommand', () => {
    for (const c of ['pytest -q', 'uv run pytest', 'npm test', 'npm run test', 'go test ./...', 'cargo test', 'python -m unittest']) {
      assert.ok(isTestCommand(c), c)
    }
    for (const c of ['cat pytest.ini', 'ls tests', 'echo testing']) assert.ok(!isTestCommand(c), c)
  })
})
