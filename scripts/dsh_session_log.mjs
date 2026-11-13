// Summarise a dsh session log (session.v4.jsonl.zstd, or plain .jsonl) as dsh itself wrote it:
// per step, the model selection in force, the request header's route and effort, and the
// route and usage of the assistant message. Used by scripts/dsh_session.sh to show from dsh's
// own log which model ran. Needs Node 22.15+ (zlib zstd); no dependencies.
//
//   node scripts/dsh_session_log.mjs DSH_HOME
//
// Finds every session log under DSH_HOME/sessions and prints one JSON object per log.

import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'
import zlib from 'node:zlib'

const home = process.argv[2]
if (!home) {
  console.error('usage: node scripts/dsh_session_log.mjs DSH_HOME')
  process.exit(2)
}

/** @param {string} dir @returns {string[]} */
function find(dir) {
  const out = []
  let entries = []
  try {
    entries = readdirSync(dir)
  } catch {
    return out
  }
  for (const name of entries) {
    const p = join(dir, name)
    if (statSync(p).isDirectory()) out.push(...find(p))
    else if (/^session\.v\d+\.jsonl(\.zstd)?$/.test(name)) out.push(p)
  }
  return out
}

const logs = find(join(home, 'sessions'))
if (logs.length === 0) {
  console.error(`no session log under ${join(home, 'sessions')}`)
  process.exit(1)
}
for (const file of logs) {
  const raw = readFileSync(file)
  const text = file.endsWith('.zstd') ? zlib.zstdDecompressSync(raw).toString('utf8') : raw.toString('utf8')
  const lines = text.split('\n').filter(Boolean).map((l) => JSON.parse(l))
  const steps = []
  let selection = null
  let header = null
  let cur = null
  const types = new Map()
  for (const e of lines) {
    if (!e || typeof e.type !== 'string') continue
    types.set(e.type, (types.get(e.type) ?? 0) + 1)
    const d = e.data ?? {}
    if (e.type === 'model/selection') {
      selection = d
      // Selected during this step (between its tool calls): applies from the next step on.
      if (cur) cur.selected_for_next = d
    } else if (e.type === 'request/header') header = d.header?.config ?? null
    else if (e.type === 'step/start') {
      cur = { turn: d.turn, step: d.step, selection_before: selection, header: null, ran: null, usage: null, tools: [] }
      steps.push(cur)
    } else if (e.type === 'assistant/message' && cur) {
      cur.header = header && { provider: header.provider, model: header.model, reasoningEffort: header.reasoningEffort ?? null }
      cur.ran = d.message?.source ? { provider: d.message.source.provider, model: d.message.source.model } : null
      cur.usage = d.usage ?? null
    } else if (e.type === 'tool/call' && cur) cur.tools.push(d.name)
  }
  console.log(
    JSON.stringify({
      file: file.slice(home.length),
      events: lines.length,
      eventTypes: Object.fromEntries(types),
      selections: lines.filter((e) => e.type === 'model/selection').map((e) => e.data),
      notices: lines
        .filter((e) => e.type === 'user/message' && (e.data?.message ?? e.data)?.source?.kind === 'model-selection')
        .map((e) => (e.data.message ?? e.data).content?.[0]?.text),
      ignorable: lines.filter((e) => e.ignorable).map((e) => e.type),
      steps,
      turnEnds: lines.filter((e) => e.type === 'turn/end').map((e) => e.data.reason?.kind),
    }),
  )
}
