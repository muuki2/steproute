// @ts-check
// The steproute mod's hooks module: wires Claude Code's events to the collector.
// Plain JavaScript, no build step, no dependencies. docs/CLAUDE-CODE.md explains what each
// hook does and what it never does.
//
// The mod validator accepts `$` only at a call site (`$.noun.event(...)`), so each hook hands
// the collector a small adapter of closures (collector.js, `Host`) instead of `$` itself.

import { createCollector } from './collector.mjs'

// Tools whose call starts or wakes an agent. The tool.call hook must not wrap them: an agent
// started inside a hook's `next` is nested in that hook, and the nesting hides the agent's
// own events from it.
const NOT_SPAWNING = /^(?!(?:Agent|Task|SendMessage|Workflow|TeamCreate)$)/

/** @type {import('claude-code').Register} */
export const register = (on, options) => {
  const c = createCollector(/** @type {Record<string, unknown>} */ (options))

  on('session.start', async ($, e, next) => {
    const started = await next(e)
    /** @type {import('./state.mjs').EnvOverrides} */
    const env = {}
    try {
      env.mode = await $.env.get('STEPROUTE_MODE')
      env.url = await $.env.get('STEPROUTE_URL')
      env.token = await $.env.get('STEPROUTE_TOKEN')
    } catch {
      // keep the userConfig options
    }
    await c.sessionStart(
      {
        fetch: (url, init) => $.http.fetch(url, init),
        sleep: (ms, o) => $.clock.sleep(ms, o),
        status: (text) => $.ui.status(text),
        log: (text) => $.ui.log(text, { to: 'debug' }),
        sessionId: () => $.session.id(),
        version: () => $.session.version(),
        registerCommand: (spec) => $.command.register(spec),
      },
      env,
    )
    return started
  })

  // Before a subagent starts: ask /decide, and in route mode choose the model.
  on('agent.spawn', ($, e, next) =>
    c.spawn(
      {
        fetch: (url, init) => $.http.fetch(url, init),
        sleep: (ms, o) => $.clock.sleep(ms, o),
        status: (text) => $.ui.status(text),
        sessionId: () => $.session.id(),
        version: () => $.session.version(),
      },
      e,
      next,
    ),
  )

  // Every model request: a routed subagent's effort goes down, its usage comes back.
  on('turn.step', async function* ($, e, next) {
    const result = yield* next(c.stepInput(e))
    c.stepResult(e, result)
    return result
  })

  // A subagent's tool results: tool errors and test runs.
  on('tool.call', { tool: NOT_SPAWNING }, ($, e, next) => c.toolCall(e, next))

  // The end of a subagent's run: its outcome is reported.
  on('turn.complete', ($, e, next) =>
    c.turnComplete(
      {
        fetch: (url, init) => $.http.fetch(url, init),
        sleep: (ms, o) => $.clock.sleep(ms, o),
      },
      e,
      next,
    ),
  )

  on('session.end', ($, e, next) =>
    c.sessionEnd(
      {
        fetch: (url, init) => $.http.fetch(url, init),
        sleep: (ms, o) => $.clock.sleep(ms, o),
      },
      e,
      next,
    ),
  )

  on('command.run', { command: 'steproute' }, () => ({ text: c.commandText() }))
}
