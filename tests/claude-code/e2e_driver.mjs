// Drives the plugin's hooks against a real `steproute serve`: three subagent spawns with
// per-request usage, as a scripted Claude Code session would produce them. Used by
// tests/test_claude_code_plugin.py; prints one JSON line with what the engine saw.
//
//   node tests/claude-code/e2e_driver.mjs <service url> <shadow|route>

import { fakeDollar, loadPlugin, SPAWN, usage } from './harness.mjs'

const [url, mode = 'shadow'] = process.argv.slice(2)
const p = loadPlugin({ service_url: url, mode, decide_timeout_ms: 2000 }, fakeDollar())
await p.start()

const tasks = [
  ['a1', 'Summarise notes/alpha.txt in one sentence.', false],
  ['a2', 'Summarise notes/beta.txt in one sentence.', true],
  ['a3', 'Summarise notes/gamma.txt in one sentence.', false],
]
const seen = []
for (const [agentId, prompt, background] of tasks) {
  const { seen: e } = await p.spawn(
    { ...SPAWN, tool_use_id: `toolu_${agentId}`, prompt, background, subagentType: 'Explore' },
    (x) => ({ model: x.model ?? 'claude-haiku-5-5', agentId }),
  )
  seen.push(e.model ?? null)
}
for (const [agentId] of tasks) {
  const model = seen[tasks.findIndex((t) => t[0] === agentId)] ?? 'claude-haiku-5-5'
  await p.step(agentId, usage(3, 40, 0, 4000, model))
  await p.tool(agentId, 'Read', { file_path: 'notes/x.txt' }, { result: { type: 'text' } })
  await p.step(agentId, usage(5, 60, 4000, 120, model))
  await p.complete(agentId, usage(8, 100, 4000, 4120, model))
}
await p.end()
const { text } = await p.command()
console.log(JSON.stringify({ seen, status: p.dollar.status.at(-1), command: text }))
