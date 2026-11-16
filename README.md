# steproute

steproute is an open router that picks, for each step of an LLM agent run, which model and effort level to use. It chooses the cheapest action whose predicted chance of success is at least a threshold: `argmin expected_cost subject to P(success) >= tau`. Cost is computed from prices, token counts and prompt-cache state. The chance of success is meant to be learned from recorded step outcomes.

Python 3.11 or later, MIT licence. Version 0.0.1.

## Current state (16 Nov 2026)

**Works and is tested** (with mocks, loopback servers and synthetic data):

- A trace schema, an append-only trace store, outcome labelling and privacy rules.
- A model catalog (Claude, DeepSeek, Ollama models), a cache-aware cost model and a rule-based baseline policy.
- A local decision service, an OpenAI-compatible proxy with a shadow mode, an MCP server, a Claude Code plugin and a dsh plugin.
- A 150-task SWE-bench Verified task set with a 20-task pilot, and a replay runner with spend caps.
- An offline evaluator that compares a policy with every single model and effort configuration, and a LightGBM predictor.

**Run on real providers:**

- The pilot replay (14 to 16 Nov 2026): 20 tasks under 21 model and effort configurations, 420 runs, $12.85 of paid spend. `docs/results/pilot.md` has the cost and success of each configuration and the caveats. Five of the 21 results are dominated by a limit of the test setup, so read the caveats before using a number.

**Not run yet:**

- The larger replay grid. No run beyond the pilot is approved, so there is no dataset release.
- The LightGBM predictor has only been fitted on synthetic data. Its model card says so.
- Whether routing beats the best single configuration is unknown. The pilot is too small to say.

**Planned, not built:** the graph model for new models, the external baselines, the dataset release, online updates.

## Install and check

```
uv sync --extra dev --extra proxy --extra mcp --extra ml
scripts/ci.sh
```

`scripts/ci.sh` is the whole local pipeline: ruff, mypy, pytest on Python 3.11 and 3.12, then a package build. The project has no hosted CI. The plugin tests need Node and are skipped with a message without it.

## Try the proxy in shadow mode

In shadow mode the proxy sends each request to the model the client asked for, unchanged, and logs what the router would have chosen. This example talks to a mock upstream on loopback, so nothing is spent and no key is needed. Run it from the repository root.

Write the files:

```
mkdir -p runs/shadow-demo
cat > runs/shadow-demo/mock.py <<'EOF'
import time
from steproute.collectors.mock_upstream import MockUpstream

print(MockUpstream().start().base_url, flush=True)
time.sleep(900)
EOF
cat > runs/shadow-demo/service.toml <<'EOF'
[store]
path = "traces"
EOF
```

Start the mock. It prints its address, for example `http://127.0.0.1:56474/v1`:

```
uv run python runs/shadow-demo/mock.py
```

In a second terminal, write `runs/shadow-demo/proxy.toml` with that address as `base_url`, then start the proxy:

```
cat > runs/shadow-demo/proxy.toml <<'EOF'
[server]
host = "127.0.0.1"
port = 8788

[proxy]
mode = "shadow"

[decisions]
source = "inprocess"
service_config = "service.toml"

[[upstream]]
name = "mock"
provider = "openai"
base_url = "http://127.0.0.1:56474/v1"
local = true
models = ["qwen3.5:9b", "gpt-oss:20b", "qwen3-coder:30b"]
EOF
uv run steproute proxy --config runs/shadow-demo/proxy.toml
```

The proxy prints a few warnings about models without an upstream, then `listening on http://127.0.0.1:8788/v1 (mode shadow)`. In a third terminal:

```
for msg in hello again; do
  curl -s -i http://127.0.0.1:8788/v1/chat/completions \
    -H 'Content-Type: application/json' -H 'X-Steproute-Session: demo' \
    -d '{"model": "qwen3-coder:30b", "messages": [{"role": "user", "content": "'$msg'"}]}'
done
uv run steproute traces stats runs/shadow-demo/traces
```

Each response carries `X-Steproute-Mode: shadow` and a decision id. After two requests, `traces stats` reports `decisions: 2` and `outcomes: 2`. It reports `joined: 0` by design: in shadow mode the outcome belongs to the client's model, not to the router's recommendation (`docs/PROXY.md`, section 7).

To use it with a real local model, point `base_url` at Ollama's `/v1` endpoint (`http://127.0.0.1:11434/v1`) and see `docs/PROXY.md` for the full configuration. That path, and any paid provider, is not covered by the steps above. `runs/` is ignored by git.

## Where to read next

- `docs/DESIGN.md`: the design overview, with what is built, what is unverified and what is planned.
- `docs/ISSUES.md`: the issue list, milestones and decisions.
- Per-area documents: `docs/COST-MODEL.md`, `docs/CATALOG.md`, `docs/POLICY.md`, `docs/RULES.md`, `docs/PROXY.md`, `docs/SERVICE.md`, `docs/MCP.md`, `docs/CLAUDE-CODE.md`, `docs/DSH.md`, `docs/TASKSET.md`, `docs/REPLAY.md`, `docs/EVALUATION.md`, `docs/MODEL-CARD-TABULAR.md`, `docs/PRIVACY.md`, `docs/SCHEMA.md`.
- `docs/INTEGRATION-NOTES.md`: what upstream docs and source actually say, with sources.
- `CONTRIBUTING.md`: how to contribute.

The core package (`steproute.core`) has no third-party runtime dependencies.
