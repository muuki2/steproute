// Drives the dsh plugin (index.js on a fake cordis context, with its real HTTP client) through
// a fake dsh loop against a real `steproute serve`. Used by tests/test_dsh_plugin.py.
//
//   node e2e_driver.mjs <service-url> <mode> [--pause]
//
// Runs one turn of three steps (two tool batches, then an answer). With --pause it then
// prints "PAUSED" and waits for a line on stdin (the test stops the service meanwhile)
// before a second turn. Prints one JSON line with what ran.

import { createInterface } from 'node:readline'

import * as plugin from '../../plugins/dsh/index.js'
import { FakeAgent, contextHost, fakeContext } from './fake_dsh.mjs'

const [url, mode, flag] = process.argv.slice(2)
const ctx = fakeContext()
plugin.apply(ctx, { mode, serviceUrl: url })
const agent = new FakeAgent(contextHost(ctx))
await agent.create()

const READ = (path) => ({ name: 'read', args: { path }, text: `contents of ${path}\n`.repeat(20) })
const turn = () =>
  agent.runTurn('Read a.txt and b.txt, then summarise them.', [
    { tools: [READ('a.txt')], usage: { inputTokens: 2000, outputTokens: 40, cacheReadTokens: 0 } },
    { tools: [READ('b.txt')], usage: { inputTokens: 300, outputTokens: 35, cacheReadTokens: 2000 } },
    { usage: { inputTokens: 250, outputTokens: 60, cacheReadTokens: 2300 } },
  ])

const record = [...(await turn())]
await agent.flush()
if (flag === '--pause') {
  console.log('PAUSED')
  const rl = createInterface({ input: process.stdin })
  await new Promise((resolve) => rl.once('line', resolve))
  rl.close()
  record.push(...(await turn()))
  await agent.flush()
}
const ran = agent.log.filter((e) => e.type === 'assistant/message').map((e) => e.data.message.source.model)
console.log(JSON.stringify({ ran, selects: agent.selects, record, lines: ctx.lines }))
