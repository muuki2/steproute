// Unit tests of the steproute dsh plugin with a fake dsh and a fake service: no dsh, no
// network. Run: node --test tests/dsh/*.test.mjs (scripts/ci.sh does).

import assert from 'node:assert/strict'
import http from 'node:http'
import { test } from 'node:test'

import * as plugin from '../../plugins/dsh/index.js'
import { postJson } from '../../plugins/dsh/lib/client.js'
import { MAX_AGENTS } from '../../plugins/dsh/lib/collector.js'
import { resolveConfig } from '../../plugins/dsh/lib/config.js'
import { extractPaths, isTestCommand, shellVerdict } from '../../plugins/dsh/lib/state.js'
import { FakeAgent, collectorHost, contextHost, fakeContext, fakeService, settle } from './fake_dsh.mjs'

const READ = { name: 'read', args: { path: 'src/app.py' }, text: 'print(1)\n' }

function setup(config = {}, opts = {}) {
  const svc = fakeService(opts)
  const host = collectorHost(config, {
    post: svc.post,
    providers: () => opts.providers ?? ['deepseek-official'],
  })
  const agent = new FakeAgent(host, opts.agent)
  return { svc, host, agent, c: host.collector }
}

async function done(agent) {
  await agent.flush()
  await settle()
}

test('shadow mode logs a decision and an outcome per step and changes nothing', async () => {
  const { svc, agent, c } = setup()
  await agent.create()
  await agent.runTurn('fix the bug', [{ tools: [READ] }, { tools: [READ] }, {}])
  await done(agent)
  assert.equal(svc.decisions.length, 3)
  assert.equal(svc.reports.length, 3)
  assert.deepEqual(agent.selects, [])
  const ids = new Set(svc.decisions.map((d) => d.decision_id))
  for (const r of svc.reports) assert.ok(ids.has(r.decision_id))
  assert.deepEqual(svc.reports.map((r) => r.signals.apply_reason), ['shadow', 'shadow', 'shadow'])
  assert.deepEqual(svc.decisions.map((d) => d.state.kind), ['main', 'tool_followup', 'tool_followup'])
  assert.deepEqual(svc.decisions.map((d) => d.state.step_idx), [0, 1, 2])
  for (const r of svc.reports) {
    assert.equal(r.signals.executed_route, 'deepseek-official/deepseek-flash')
    assert.equal(r.final_model, 'deepseek-flash')
    assert.equal(r.signals.applied, false)
    assert.equal(r.success, null)
  }
  assert.equal(c._debug.stats.applied, 0)
})

test('shadow mode never makes a step wait, even for a slow service', async () => {
  const { svc, agent } = setup({ decideTimeoutMs: 300 })
  svc.delayMs = 120
  await agent.create()
  const record = await agent.runTurn('t', [{ tools: [READ] }, { tools: [READ] }, {}])
  await new Promise((r) => setTimeout(r, 200))
  await done(agent)
  for (const r of record) assert.ok(r.postExecuteMs < 20, `${r.postExecuteMs}`)
  assert.equal(svc.decisions.length, 3)
  assert.equal(svc.reports.length, 3)
  assert.equal(svc.reports[1].signals.plugin_wait_ms, null)
})

test('route mode applies the decision before the next step and records what ran', async () => {
  const { svc, agent } = setup({ mode: 'route' }, { action: { model: 'deepseek-v4-pro', effort: 'none', escalate_to: null } })
  await agent.create()
  await agent.runTurn('task', [{ tools: [READ] }, {}])
  await done(agent)
  assert.deepEqual(agent.selects[0], { provider: 'deepseek-official', model: 'deepseek-v4-pro', reasoningEffort: 'off' })
  const [first, second] = svc.reports
  assert.equal(first.signals.apply_reason, 'turn_start')
  assert.equal(first.signals.applied, false)
  assert.equal(first.final_model, 'deepseek-flash')
  assert.equal(second.signals.apply_reason, 'routed')
  assert.equal(second.signals.applied, true)
  assert.equal(second.signals.selected_route, 'deepseek-official/deepseek-v4-pro')
  assert.equal(second.signals.executed_route, 'deepseek-official/deepseek-v4-pro')
  assert.equal(second.signals.executed_effort, 'off')
  assert.equal(second.signals.selection_ran, true)
  assert.equal(second.signals.model_matches_recommendation, true)
  assert.equal(second.signals.effort_matches_recommendation, true)
  assert.equal(second.signals.model_switched, true)
  assert.equal(second.final_model, 'deepseek-v4-pro')
  // The routed model stays selected after the turn: the next turn starts on it, warm, rather
  // than switching back to the user's model and away again (docs/DSH.md, decision 1).
  assert.deepEqual(agent.selection, { provider: 'deepseek-official', model: 'deepseek-v4-pro', reasoningEffort: 'off' })
  await agent.runTurn('next', [{}])
  await done(agent)
  assert.equal(svc.reports[2].signals.executed_route, 'deepseek-official/deepseek-v4-pro')
  assert.equal(svc.reports[2].signals.model_switched, false)
})

test('the first step of a turn is decided but never applied', async () => {
  const { svc, agent } = setup({ mode: 'route' })
  await agent.create()
  await agent.runTurn('one-shot answer', [{}])
  await done(agent)
  assert.equal(svc.decisions.length, 1)
  assert.equal(svc.decisions[0].state.task_text, 'one-shot answer')
  assert.equal(svc.reports[0].signals.apply_reason, 'turn_start')
  assert.deepEqual(agent.selects, [])
})

test('the state after a switch names the model that ran as the cache owner', async () => {
  const { svc, agent } = setup({ mode: 'route' }, { action: { model: 'deepseek-v4-pro', effort: 'high', escalate_to: null } })
  await agent.create()
  await agent.runTurn('t', [
    { tools: [READ], usage: { inputTokens: 1000, cacheReadTokens: 4000, outputTokens: 50 } },
    { tools: [READ], usage: { inputTokens: 5100, cacheReadTokens: 0, outputTokens: 60 } },
    {},
  ])
  await done(agent)
  const [, s2, s3] = svc.decisions.map((d) => d.state)
  // Step 2's decision: the previous request ran on flash with a 5000-token prompt.
  assert.equal(s2.current_model, 'deepseek-flash')
  assert.equal(s2.current_effort, 'high')
  assert.equal(s2.cached_prefix_tokens, 5000)
  assert.equal(s2.prompt_tokens, 5000 + 50 + Math.ceil('print(1)\n'.length / 4))
  assert.deepEqual(s2.tools, ['read', 'bash', 'edit'])
  // Step 3's: the switch made v4-pro the owner; its whole 5100-token prompt was a miss.
  assert.equal(s3.current_model, 'deepseek-v4-pro')
  assert.equal(s3.cached_prefix_tokens, 5100)
  assert.equal(svc.reports[1].cache_read_tokens, 0)
  assert.equal(svc.reports[1].input_tokens, 5100)
})

test('usage, tool errors, denials, approvals, tests and retries map to signals', async () => {
  const { svc, agent } = setup()
  await agent.create()
  await agent.runTurn('t', [
    {
      usage: { inputTokens: 10, outputTokens: 5, cacheReadTokens: 90, cacheWriteTokens: 0, reasoningTokens: 3 },
      retries: 2,
      tools: [
        { name: 'read', args: { path: 'missing.txt' }, text: 'Error: not found', isError: true, code: 'ENOENT' },
        { name: 'bash', args: { command: 'rm -rf build' }, text: 'Error: the user rejected tool "bash"', isError: true, code: 'APPROVAL_REJECTED' },
        { name: 'bash', args: { command: 'python -m pytest -q' }, text: '1 failed\n[exit code: 1]' },
        { name: 'bash', args: { command: 'sleep 9' }, text: 'Error: aborted', isError: true, code: 'ABORTED' },
      ],
      approvals: [null, 'rejected', 'allowed-once'],
    },
    { tools: [{ name: 'bash', args: { command: 'npm test' }, text: 'all good' }] },
    {},
  ])
  await done(agent)
  const [a, b, c] = svc.reports.map((r) => r.signals)
  assert.equal(svc.reports[0].input_tokens, 10)
  assert.equal(svc.reports[0].output_tokens, 5)
  assert.equal(svc.reports[0].cache_read_tokens, 90)
  assert.equal(a.reasoning_tokens, 3)
  assert.equal(a.tool_call_count, 4)
  assert.equal(a.tool_error_count, 1)
  assert.equal(a.tool_denied_count, 1)
  assert.equal(a.tool_aborted_count, 1)
  assert.equal(a.tool_error, true)
  assert.equal(a.user_rejected, true)
  assert.equal(a.approvals_rejected, 1)
  assert.equal(a.approvals_allowed, 1)
  assert.equal(a.user_approved, undefined)
  assert.equal(a.tests_passed, false)
  assert.equal(a.llm_retry_count, 2)
  assert.equal(a.failed_attempts, 2)
  assert.equal(a.retry_count, undefined)
  assert.equal(a.retried, undefined)
  assert.equal(b.tests_passed, true)
  assert.equal(b.tool_error, undefined)
  // Nothing checked the final answer: no labelling signal at all.
  for (const k of ['tool_error', 'tests_passed', 'user_rejected', 'user_approved']) assert.equal(c[k], undefined)
  assert.equal(c.end_reason, 'turn_completed')
  assert.equal(a.end_reason, 'continued')
  // Paths from the tool arguments travel with the next decision for sensitive detection.
  assert.ok(svc.decisions[1].state.features.paths.includes('missing.txt'))
})

test('a decision that misses the timeout is logged but never applied', async () => {
  const { svc, agent } = setup({ mode: 'route', decideTimeoutMs: 30 })
  await agent.create()
  let release
  svc.override = (path) => {
    if (path !== '/decide' || svc.decisions.length !== 1) return undefined
    // The second decision (between steps) answers after the step's wait is over.
    return new Promise((r) => (release = r)).then(() => undefined)
  }
  const turn = agent.runTurn('t', [{ tools: [READ] }, {}])
  await new Promise((r) => setTimeout(r, 120))
  release()
  await turn
  await done(agent)
  assert.deepEqual(agent.selects, [])
  const late = svc.reports.find((r) => r.signals.step_role === 'after_tools')
  assert.equal(late.signals.applied, false)
  assert.equal(late.signals.apply_reason, 'late')
  assert.equal(late.final_model, 'deepseek-flash')
})

test('fail-open: the service going away mid-turn puts the user model back and stops asking', async () => {
  const { svc, agent, c } = setup({ mode: 'route' })
  await agent.create()
  // Decisions 1 (turn start) and 2 (before step 2) are answered; then the service is gone
  // and every call is refused, as a killed process's port refuses it.
  svc.override = () => (svc.decisions.length >= 2 ? { kind: 'unreachable' } : undefined)
  const record = await agent.runTurn('t', [{ tools: [READ] }, { tools: [READ] }, { tools: [READ] }, {}])
  await done(agent)
  assert.equal(agent.selects[0].model, 'deepseek-v4-pro') // step 2 was routed
  assert.equal(agent.selects[1].model, 'deepseek-flash') // fail-open before step 3
  assert.equal(c._debug.stats.failOpen >= 1, true)
  assert.ok(c._debug.backoffUntil > Date.now())
  assert.ok(c.statusText().includes('service unreachable'))
  // After the first refusal nothing waits: the steps went on without the service.
  for (const r of record.slice(2)) assert.ok(r.postExecuteMs < 20, `${r.postExecuteMs}`)
  // The outcome that could not be sent is kept for resending.
  assert.ok(c._debug.retry.length >= 1)
  assert.ok(c._debug.retry.some((o) => o.signals.executed_route === 'deepseek-official/deepseek-v4-pro'))
})

for (const [status, body, reason] of [
  [503, { error: 'no_fallback_configured', sensitive: false }, 'no_decision'],
  [503, { error: 'no_action_allowed', sensitive: true }, 'sensitive_no_decision'],
  [500, { error: 'internal_error' }, 'service_500'],
  [400, { error: 'bad_request' }, 'service_400'],
  [401, { error: 'unauthorized' }, 'service_401'],
]) {
  test(`status ${status} ${body.error}: no decision, the step keeps the user model`, async () => {
    const { svc, agent, c } = setup({ mode: 'route' })
    await agent.create()
    svc.override = (path) => (path === '/decide' ? { kind: 'http', status, body } : undefined)
    await agent.runTurn('t', [{ tools: [READ] }, {}])
    await done(agent)
    assert.deepEqual(agent.selects, [])
    assert.equal(svc.reports.length, 0) // no decision, nothing to join
    if (status === 503) assert.equal(c._debug.backoffUntil, 0)
    else assert.ok(c._debug.backoffUntil > Date.now())
    if (reason === 'sensitive_no_decision') assert.equal(c._debug.stats.sensitive, 2)
  })
}

test('a timeout backs off: the next steps do not wait at all', async () => {
  const { svc, agent, c } = setup({ mode: 'route', decideTimeoutMs: 40 })
  await agent.create()
  svc.delayMs = 1000
  const record = await agent.runTurn('t', [{ tools: [READ] }, { tools: [READ] }, { tools: [READ] }, {}])
  await done(agent)
  const waits = record.map((r) => r.postExecuteMs)
  assert.ok(waits.slice(1).every((ms) => ms < 20), `later steps waited ${waits}`)
  assert.equal(c._debug.stats.skipped >= 2, true)
})

test('a sensitive step: its tool output goes to the service for detection, never kept by the plugin', async () => {
  const { svc, agent } = setup({ mode: 'route' })
  await agent.create()
  svc.override = (path, body) =>
    path === '/decide' && /API_KEY=/.test(body.task_text ?? '') ? { kind: 'http', status: 503, body: { error: 'no_action_allowed', sensitive: true } } : undefined
  await agent.runTurn('t', [{ tools: [{ name: 'read', args: { path: '.env' }, text: 'API_KEY=placeholder' }] }, {}])
  await done(agent)
  assert.deepEqual(agent.selects, [])
  const sent = JSON.stringify(svc.reports)
  assert.equal(sent.includes('API_KEY'), false)
})

test('decisions naming a model dsh cannot run are not applied', async () => {
  for (const [action, providers, reason] of [
    [{ model: 'claude-opus-5-5', effort: 'medium' }, ['deepseek-official'], 'unmapped_model'],
    [{ model: 'deepseek-v4-pro', effort: 'medium' }, ['deepseek-official'], 'unmapped_effort'],
    [{ model: 'deepseek-v4-pro', effort: 'low' }, ['some-other-route'], 'provider_not_registered'],
  ]) {
    const { svc, agent } = setup({ mode: 'route' }, { action: { ...action, escalate_to: null }, providers })
    await agent.create()
    await agent.runTurn('t', [{ tools: [READ] }, {}])
    await done(agent)
    assert.deepEqual(agent.selects, [], reason)
    assert.equal(svc.reports[1].signals.apply_reason, reason)
  }
})

test('a fail-open decision from the service is not applied', async () => {
  const { svc, agent } = setup({ mode: 'route' }, { policy: 'fail_open:timeout' })
  await agent.create()
  await agent.runTurn('t', [{ tools: [READ] }, {}])
  await done(agent)
  assert.deepEqual(agent.selects, [])
  assert.equal(svc.reports[1].signals.apply_reason, 'fail_open')
})

test('a cascade fallback is recorded as not run', async () => {
  const action = { model: 'deepseek-flash', effort: 'low', escalate_to: { model: 'deepseek-v4-pro', effort: 'high', escalate_to: null } }
  const { svc, agent } = setup({}, { action })
  await agent.create()
  await agent.runTurn('t', [{}])
  await done(agent)
  assert.equal(svc.reports[0].signals.fallback_not_run, 'deepseek-v4-pro (high)')
  assert.equal(svc.reports[0].escalated, false)
})

test("the user's own model change becomes the model to fall back to", async () => {
  const { svc, agent } = setup({ mode: 'route' })
  await agent.create()
  agent.userSelects({ provider: 'deepseek-official', model: 'deepseek-v4-pro', reasoningEffort: 'max' })
  svc.action = { model: 'deepseek-flash', effort: 'low', escalate_to: null }
  await agent.runTurn('t', [{ tools: [READ] }, {}])
  await done(agent)
  assert.equal(svc.reports[1].signals.executed_route, 'deepseek-official/deepseek-flash')
  // The service goes away: the step after the next tool batch falls back to the user's pick.
  svc.override = () => ({ kind: 'unreachable' })
  await agent.runTurn('again', [{ tools: [READ] }, {}])
  await done(agent)
  assert.deepEqual(agent.selection, { provider: 'deepseek-official', model: 'deepseek-v4-pro', reasoningEffort: 'max' })
})

test('subagents are tracked as their own sessions and can be left alone', async () => {
  const { svc, host } = setup({ mode: 'route', routeSubagents: false })
  const child = new FakeAgent(host, { id: 'child-1', origin: 'subagent', parentSession: 'sess-1' })
  await child.create()
  await child.runTurn('sub task', [{ tools: [READ] }, {}])
  await done(child)
  assert.equal(svc.decisions[0].state.kind, 'subagent')
  assert.equal(svc.decisions[0].state.features.parent_session, 'sess-1')
  assert.equal(svc.reports[1].signals.apply_reason, 'subagent_not_routed')
  assert.deepEqual(child.selects, [])
})

test('off mode calls nothing', async () => {
  const { svc, agent } = setup({ mode: 'off' })
  await agent.create()
  await agent.runTurn('t', [{ tools: [READ] }, {}])
  await done(agent)
  assert.equal(svc.decisions.length + svc.reports.length, 0)
})

test('memory stays bounded over many sessions', async () => {
  const { host, c } = setup()
  for (let i = 0; i < MAX_AGENTS + 20; i++) {
    const a = new FakeAgent(host, { id: `s${i}` })
    await a.create()
    await a.runTurn('t', [{}])
  }
  await settle()
  assert.ok(c._debug.tracks.size <= MAX_AGENTS)
})

test('a turn that errors reports its last step with the turn end reason', async () => {
  const { svc, agent } = setup()
  await agent.create()
  await agent.runTurn('t', [{ tools: [READ] }], { endKind: 'error' })
  await done(agent)
  assert.equal(svc.reports.length, 2)
  assert.equal(svc.reports[0].signals.end_reason, 'turn_error')
  // The decision taken for a step that never came is closed as not run.
  assert.equal(svc.reports[1].signals.step_ran, false)
  assert.equal(svc.reports[1].signals.end_reason, 'no_step')
})

test('config: bad values fall back safely, the environment wins', () => {
  const c = resolveConfig({ mode: 'turbo', serviceUrl: 'http://example.com:80', decideTimeoutMs: 'x', models: 7 })
  assert.equal(c.mode, 'shadow')
  assert.equal(c.url, 'http://127.0.0.1:8789')
  assert.equal(c.decideTimeoutMs, 200)
  assert.ok(c.models.has('deepseek-flash'))
  assert.equal(c.warnings.length, 3)
  const e = resolveConfig({ mode: 'shadow' }, { STEPROUTE_MODE: 'route', STEPROUTE_URL: 'http://localhost:9000/' })
  assert.equal(e.mode, 'route')
  assert.equal(e.url, 'http://localhost:9000')
  assert.equal(resolveConfig({ enabled: false }).mode, 'off')
  assert.deepEqual(plugin.Config['~standard'].validate({ mode: 'x' }), { value: { mode: 'x' } })
})

test('helpers: test commands, shell verdicts, paths', () => {
  assert.ok(isTestCommand('cd app && pytest -x'))
  assert.ok(isTestCommand('npm run test'))
  assert.ok(!isTestCommand('cat pytest.ini'))
  assert.equal(shellVerdict('ok\n'), true)
  assert.equal(shellVerdict('boom\n[exit code: 2]'), false)
  assert.equal(shellVerdict('...\n[timed out after 1000ms]'), null)
  assert.equal(shellVerdict('[still running after 5ms; moved to background job j1]'), null)
  assert.deepEqual(extractPaths('read ~/.ssh/id_rsa and .env, see https://x.y/z'), ['~/.ssh/id_rsa', '.env'])
})

test('index.js: listeners pass every waterfall result through, even on malformed input', async () => {
  const ctx = fakeContext()
  const svc = fakeService()
  const saved = process.env.STEPROUTE_URL
  plugin.apply(ctx, { mode: 'shadow' })
  assert.ok(ctx.lines[0].startsWith('steproute 0.1.0: shadow mode'))
  for (const name of ['agent/created', 'agent/pre-step', 'tools/post-execute', 'session/event', 'session/flush', 'agent/disposed']) {
    assert.ok(ctx.listeners.has(name), name)
  }
  const pre = ctx.listeners.get('agent/pre-step')[0]
  const marker = { kind: 'enter', messages: [] }
  assert.equal(await pre(null, async () => marker), marker)
  assert.equal(await pre({ agent: { id: 1, model: { current: () => { throw new Error('x') } } } }, async () => marker), marker)
  const post = ctx.listeners.get('tools/post-execute')[0]
  const accept = { kind: 'accept' }
  assert.equal(await post(undefined, undefined, async () => accept), accept)
  assert.equal(await post({ agent: {} }, null, async () => accept), accept)
  ctx.listeners.get('session/event')[0](null, null)
  ctx.listeners.get('session/event')[0]({ id: 's' }, { type: 'tool/result', data: null })
  // An error from dsh's own chain is dsh's, and is not swallowed.
  await assert.rejects(pre({}, async () => { throw new Error('dsh failed') }), /dsh failed/)
  assert.equal(saved, process.env.STEPROUTE_URL)
  void svc
})

test('index.js: the status lines go to stderr only when asked', () => {
  const written = []
  const original = process.stderr.write
  process.stderr.write = (chunk) => (written.push(String(chunk)), true)
  try {
    plugin.apply(fakeContext(), { mode: 'shadow' })
    const quiet = written.length
    process.env.STEPROUTE_STATUS = 'stderr'
    plugin.apply(fakeContext(), { mode: 'route' })
    delete process.env.STEPROUTE_STATUS
    plugin.apply(fakeContext(), { mode: 'shadow', statusToStderr: true })
    assert.equal(quiet, 0)
    assert.deepEqual(
      written.map((l) => l.trim()),
      ['steproute 0.1.0: route mode, service http://127.0.0.1:8789', 'steproute 0.1.0: shadow mode, service http://127.0.0.1:8789'],
    )
  } finally {
    process.stderr.write = original
    delete process.env.STEPROUTE_STATUS
  }
})

test('index.js end to end on the fake context', async () => {
  const ctx = fakeContext()
  // Point the plugin at a port nothing listens on: every call is refused, fast.
  plugin.apply(ctx, { mode: 'route', serviceUrl: 'http://127.0.0.1:9' })
  const agent = new FakeAgent(contextHost(ctx))
  await agent.create()
  const record = await agent.runTurn('t', [{ tools: [READ] }, { tools: [READ] }, {}])
  await agent.flush()
  assert.deepEqual(agent.selects, [])
  for (const r of record) assert.ok(r.postExecuteMs < 200)
})

test('client: 200, refused, and a hung service within the timeout', async () => {
  const server = http.createServer((req, res) => {
    if (req.url === '/hang') return
    let body = ''
    req.on('data', (c) => (body += c))
    req.on('end', () => {
      res.writeHead(200, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ echo: JSON.parse(body), auth: req.headers.authorization ?? null }))
    })
  })
  await new Promise((r) => server.listen(0, '127.0.0.1', r))
  const base = `http://127.0.0.1:${server.address().port}`
  try {
    const ok = await postJson(base, 'tok', '/decide', { a: 1 }, 500)
    assert.deepEqual(ok, { kind: 'http', status: 200, body: { echo: { a: 1 }, auth: 'Bearer tok' } })
    const t0 = Date.now()
    assert.deepEqual(await postJson(base, null, '/hang', {}, 80), { kind: 'timeout' })
    assert.ok(Date.now() - t0 < 300)
  } finally {
    server.closeAllConnections()
    await new Promise((r) => server.close(r))
  }
  assert.deepEqual(await postJson(base, null, '/decide', {}, 200), { kind: 'unreachable' })
})
