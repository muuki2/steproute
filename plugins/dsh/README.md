# steproute-dsh

The steproute collector for DeepSeek Harness (dsh) **0.2.2-rc.3**: one routing decision and one
outcome per agent step, sent to a local `steproute serve`. Shadow mode (log only) is the
default; route mode applies decisions between steps. Full documentation: `docs/DSH.md` in the
steproute repository.

## Privacy

**dsh's default setup uploads your session log to DeepSeek, and this plugin cannot stop that.**
The `session-log-deepseek` plugin of dsh's base bundle sends your prompts, the model's answers
and tool results inside every request to the DeepSeek API; `plugin-package-inventory-deepseek`
adds the list of your plugins; and every such request carries an anonymous user id and the
session id. steproute's `sensitive` routing chooses models; it does not see or block any of
this. To turn the two uploads off, put this in your profile's `cordis.patch.yml`:

```yaml
- id: session-log-deepseek
  disabled: true
- id: plugin-package-inventory-deepseek
  disabled: true
```

This plugin itself talks only to the steproute service on loopback (`127.0.0.1` or
`localhost`; any other address is refused) and writes nothing to the session log except dsh's
own model selection in route mode.

## Quick start

```bash
steproute serve --config plugins/dsh/service/steproute.toml       # leave it running
dsh headless --patch ./steproute.yml "your task"
```

with `steproute.yml`:

```yaml
- insert:
    - id: steproute
      name: /absolute/path/to/steproute/plugins/dsh/index.js
```

`STEPROUTE_MODE=route` switches routing on for one run; `STEPROUTE_MODE=off` switches the
plugin off.
