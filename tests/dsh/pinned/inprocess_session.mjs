// An in-process dsh session on the real dsh 0.2.2-rc.3 libraries, with the steproute plugin
// mounted next to them. It composes only the core rows of dsh-base (session, projections,
// agent registry, LLM runtime, system prompt, tools, native executor, agent loop) plus three
// local test plugins: a scripted model adapter registered as route `deepseek-official`, one
// read-only tool, and the steproute plugin itself. Nothing here opens a network connection
// except the plugin's loopback calls to `steproute serve`; no API key exists anywhere.
//
//   node inprocess_session.mjs <service-url> <mode> [--kill-after-step N] [--pid PID]
//
// Prints one JSON line: what the adapter was asked to run per request (provider, model,
// effort), the session log's assistant/message sources and usage, and the plugin's status.
// scripts/dsh_inprocess_session.sh starts the service, runs this and checks the store.

import { Context } from '@deepseek-ai/cordis'
import { LlmAdapter } from '@deepseek-ai/dsh-llm'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { installModelSelection } from '@deepseek-ai/dsh-agent'
import * as steproute from '../../../plugins/dsh/index.js'

const [url, mode, ...rest] = process.argv.slice(2)
const killAfter = rest.includes('--kill-after-step') ? Number(rest[rest.indexOf('--kill-after-step') + 1]) : null
const killPid = rest.includes('--pid') ? Number(rest[rest.indexOf('--pid') + 1]) : null
process.env.STEPROUTE_URL = url
process.env.STEPROUTE_MODE = mode

/** Scripted replies: two tool rounds, then an answer, per turn. */
const requests = []
let killed = false
const EFFORTS = ['off', 'low', 'high', 'max'].map((id) => ({ id, name: id }))
class ScriptedAdapter extends LlmAdapter {
  // The reasoning efforts dsh-llm-deepseek declares for its models (model-info.ts:6-31).
  resolveModel(provider, model) {
    return Promise.resolve({
      provider,
      id: model,
      name: model,
      context: { contextWindow: 1_048_576 },
      reasoning: { efforts: EFFORTS, defaultEffort: 'high' },
    })
  }

  async *stream(options) {
    const n = requests.length
    requests.push({ provider: options.provider, model: options.model, reasoningEffort: options.reasoningEffort ?? null })
    if (killAfter !== null && !killed && n >= killAfter && killPid) {
      // Kill the service mid-session, between two model requests.
      process.kill(killPid, 'SIGKILL')
      killed = true
    }
    const turnStep = n % 3
    const usage = { inputTokens: 900 + 100 * n, outputTokens: 30 + n, cacheReadTokens: n === 0 ? 0 : 800 }
    if (turnStep < 2) {
      const id = `call-${n}`
      const args = JSON.stringify({ name: turnStep === 0 ? 'alpha' : 'beta' })
      yield { type: 'block-start', index: 0, blockType: 'tool-call' }
      yield { type: 'tool-call-delta', index: 0, id, name: 'read_note', argumentsDelta: args }
      yield { type: 'block-end', index: 0, block: { type: 'tool-call', id, name: 'read_note', arguments: args } }
      yield { type: 'usage', usage }
      yield { type: 'finish', reason: { kind: 'tool-calls' } }
    } else {
      const text = 'Both notes are short reminders.'
      yield { type: 'block-start', index: 0, blockType: 'text' }
      yield { type: 'text-delta', index: 0, text }
      yield { type: 'block-end', index: 0, block: { type: 'text', text } }
      yield { type: 'usage', usage }
      yield { type: 'finish', reason: { kind: 'stop' } }
    }
  }
}

const adapterPlugin = {
  name: 'scripted-adapter',
  inject: ['llm'],
  apply(ctx) {
    ctx.llm.registerAdapter(['deepseek-official'], new ScriptedAdapter())
  },
}

const toolPlugin = {
  name: 'read-note-tool',
  inject: ['tools'],
  apply(ctx) {
    ctx.tools.register(
      defineTool({
        name: 'read_note',
        description: 'Read a short note by name.',
        parameters: { name: { type: 'string', required: true, description: 'note name' } },
        output: { schema: { type: 'string' }, render: (_a, v) => [{ type: 'text', text: v }] },
        async execute(args) {
          return `note ${args.name}: water the plants`
        },
      }),
    )
  },
}

const mod = async (name) => {
  const m = await import(name)
  return m.default ?? m
}

const app = new Context()
const plugins = [
  ['@deepseek-ai/dsh-session', {}],
  ['@deepseek-ai/dsh-session-projection', {}],
  ['@deepseek-ai/dsh-agent', {}],
  ['@deepseek-ai/dsh-llm', {}],
  ['@deepseek-ai/dsh-system-prompt', { personaPrefix: 'You are a coding agent powered by the {{model}} model.' }],
  ['@deepseek-ai/dsh-tools', {}],
  ['@deepseek-ai/dsh-tool-executor-native', {}],
  ['@deepseek-ai/dsh-agent-loop', { agents: [] }],
]
for (const [name, config] of plugins) app.plugin(await mod(name), config)
app.plugin(adapterPlugin)
app.plugin(toolPlugin)
app.plugin(steproute, {})

const waitFor = async (pred, what) => {
  for (let i = 0; i < 200; i++) {
    if (pred()) return
    await new Promise((r) => setTimeout(r, 10))
  }
  throw new Error(`timed out waiting for ${what}`)
}
await waitFor(() => app.get('agents') && app.get('sessions') && app.get('tools') && app.get('llm'), 'services')
await new Promise((r) => setTimeout(r, 50))

const agents = app.get('agents')
const sessions = app.get('sessions')
const { agent } = await agents.create({
  sessionId: 'steproute-inprocess-1',
  meta: { cwd: process.cwd() },
  agentOptions: { provider: 'deepseek-official', model: 'deepseek-flash', reasoningEffort: 'high' },
  setup: (agentCtx) => {
    installModelSelection(agentCtx)
  },
})
const started = Date.now()
const errors = []
app.on('agent/error', ({ error }) => errors.push(String(error && error.message ? error.message : error)))
for (const prompt of ['Read the notes alpha and beta, then summarise them.', 'Do it once more.']) {
  agent.followup({ role: 'user', id: `m-${Date.now()}-${Math.random()}`, source: { kind: 'user' }, content: [{ type: 'text', text: prompt }] })
  await agent.whenIdle()
}
await sessions.flush(agent.session)
const log = agent.session.log ?? []
const events = (typeof agent.session.events === 'function' ? agent.session.events() : log).map((e) => e)
const assistant = events
  .filter((e) => e.type === 'assistant/message')
  .map((e) => ({ turn: e.data.turn, step: e.data.step, source: e.data.message.source, usage: e.data.usage }))
const selections = events.filter((e) => e.type === 'model/selection').map((e) => e.data)
const notices = events
  .filter((e) => e.type === 'user/message' && (e.data.message ?? e.data)?.source?.kind === 'model-selection')
  .map((e) => (e.data.message ?? e.data).content[0].text)
const turnEnds = events.filter((e) => e.type === 'turn/end').map((e) => e.data.reason.kind)
console.log(
  JSON.stringify({
    mode,
    requests,
    assistant,
    selections,
    notices,
    turnEnds,
    errors,
    killed,
    ms: Date.now() - started,
    eventTypes: [...new Set(events.map((e) => e.type))],
  }),
)
await app.dispose?.()
process.exit(0)
