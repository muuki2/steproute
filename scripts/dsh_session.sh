#!/usr/bin/env bash
# A scripted real dsh session (the pinned CLI, `dsh headless`) with the steproute plugin, for
# acceptance criteria 1 and 2 of issue 15 (docs/DSH.md, "Check it works"). Opt-in: it needs
# the pinned dsh install (tests/dsh/pinned, `npm ci --ignore-scripts`) and Node 24.9 or
# later (the engines field of @deepseek-ai/dsh 0.2.2-rc.3); scripts/ci.sh does not run it.
#
#   scripts/dsh_session.sh shadow|route SCRATCH_DIR [KILL_AFTER_SECONDS]
#
# SCRATCH_DIR must not exist yet and must be outside this repository. The script:
#   1. makes SCRATCH_DIR/{home,dsh-home,agents,work} and two small files in work/;
#   2. starts the mock upstream (scripts/dsh_mock_upstream.py) and `steproute serve` with the
#      plugin's service config, both on loopback;
#   3. writes SCRATCH_DIR/session.patch.yml, which turns off every outbound path found in
#      dsh 0.2.2-rc.3 (docs/INTEGRATION-NOTES.md section 13), points the DeepSeek route at
#      the mock, and inserts the plugin row;
#   4. runs `dsh headless --patch session.patch.yml --json "<task>"` with HOME, DSH_HOME and
#      DSH_AGENTS_HOME inside SCRATCH_DIR and a clean environment;
#   5. kills the service KILL_AFTER_SECONDS (default 1.5; "none" keeps it) into the session;
#   6. checks that dsh exited 0 and the store (scripts/check_dsh_traces.py).
# No API key exists: the DeepSeek route reads its key from STEPROUTE_MOCK_UPSTREAM_KEY, a
# placeholder the mock never checks.
set -euo pipefail

mode=${1:-}
scratch=${2:-}
kill_after=${3:-1.5}
usage="usage: $0 shadow|route SCRATCH_DIR [KILL_AFTER_SECONDS|none]"
case "$mode" in shadow | route) ;; *) echo "$usage" >&2; exit 2 ;; esac
[ -n "$scratch" ] || { echo "$usage" >&2; exit 2; }

repo=$(cd "$(dirname "$0")/.." && pwd)
dsh="$repo/tests/dsh/pinned/node_modules/@deepseek-ai/dsh/lib/bin.js"
[ -f "$dsh" ] || { echo "the pinned dsh is not installed: (cd tests/dsh/pinned && npm ci --ignore-scripts)" >&2; exit 2; }
node -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a>24||(a===24&&b>=9)?0:1)' || {
  echo "dsh 0.2.2-rc.3 needs Node 24.9 or later (found $(node --version)); not running" >&2
  exit 3
}
[ ! -e "$scratch" ] || { echo "$scratch already exists" >&2; exit 2; }
mkdir -p "$scratch"/{home,dsh-home,agents,work}
scratch=$(cd "$scratch" && pwd)
case "$scratch/" in "$repo"/*) echo "SCRATCH_DIR must be outside the repository" >&2; exit 2 ;; esac
printf 'Alpha is a small coastal town known for its lighthouse.\n' > "$scratch/work/a.txt"
printf 'Beta describes a bicycle repair shop that opens at dawn.\n' > "$scratch/work/b.txt"

pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2> /dev/null || true; wait "$p" 2> /dev/null || true; done; }
trap cleanup EXIT
py=$(uv run --project "$repo" --no-sync python -c 'import sys; print(sys.executable)')
serve_bin=$(dirname "$py")/steproute

wait_line() { # file prefix
  for _ in $(seq 1 100); do grep -q "$2" "$1" 2> /dev/null && return 0; sleep 0.1; done
  return 1
}
"$py" "$repo/scripts/dsh_mock_upstream.py" --out "$scratch/upstream.json" > "$scratch/mock.out" 2> "$scratch/mock.err" &
mock_pid=$!
pids+=("$mock_pid")
wait_line "$scratch/mock.out" 'mock upstream:' || { cat "$scratch/mock.err" >&2; exit 1; }
mock_url=$(sed -n 's/^mock upstream: //p' "$scratch/mock.out")
"$serve_bin" serve --config "$repo/plugins/dsh/service/steproute.toml" --port 0 \
  --store "$scratch/traces" > "$scratch/serve.out" 2> "$scratch/serve.err" &
serve_pid=$!
pids+=("$serve_pid")
wait_line "$scratch/serve.out" 'listening on' || { cat "$scratch/serve.err" >&2; exit 1; }
url=$(sed -n 's/^steproute serve: listening on //p' "$scratch/serve.out")
echo "mock upstream $mock_url, service $url (pid $serve_pid), store $scratch/traces"

cat > "$scratch/session.patch.yml" << EOF
# Outbound paths of dsh 0.2.2-rc.3, all off (docs/INTEGRATION-NOTES.md section 13).
- id: session-log-deepseek
  disabled: true
- id: plugin-package-inventory-deepseek
  disabled: true
- id: session-telemetry-otel
  disabled: true
- id: session-title-llm
  disabled: true
- id: tool-web
  disabled: true
- id: web-search-deepseek
  disabled: true
- id: web-fetch-http
  disabled: true
- id: web-deepseek-api
  disabled: true
- id: tool-bash
  disabled: true
- id: tool-jobs
  disabled: true
- id: llm-deepseek-account
  disabled: true
- id: llm-pi-ai
  disabled: true
- id: plugin-manager
  disabled: true
# The only model route: the mock upstream on loopback, with a placeholder key reference.
- id: llm-deepseek
  config:
    baseURL: $mock_url
    apiKeyEnv: STEPROUTE_MOCK_UPSTREAM_KEY
- insert:
    - id: steproute
      name: $repo/plugins/dsh/index.js
      config:
        mode: $mode
        serviceUrl: $url
EOF

task='Read a.txt and then b.txt in the current directory with the read tool, one file per step, then summarise both in one sentence.'
echo "dsh $(node "$dsh" --version 2> /dev/null || echo '?'), node $(node --version), mode $mode"
(cd "$scratch/work" && env -i HOME="$scratch/home" DSH_HOME="$scratch/dsh-home" DSH_AGENTS_HOME="$scratch/agents" \
  PATH="$PATH" LANG="${LANG:-en_US.UTF-8}" TERM=dumb \
  DSH_TELEMETRY_DISABLED=1 DSH_TELEMETRY_MODE=DISABLED DSH_TELEMETRY_OTLP_URL=http://127.0.0.1:9/v1/logs \
  STEPROUTE_MOCK_UPSTREAM_KEY=placeholder-for-the-mock \
  node "$dsh" headless --patch "$scratch/session.patch.yml" --json "$task" \
  > "$scratch/dsh.jsonl" 2> "$scratch/dsh.err") &
dsh_pid=$!
if [ "$kill_after" != none ]; then
  sleep "$kill_after"
  kill -9 "$serve_pid" 2> /dev/null && echo "killed the service at ${kill_after}s"
fi
set +e
wait "$dsh_pid"
status=$?
set -e
echo "dsh exit status: $status"
cleanup
trap - EXIT
echo "upstream requests:"
cat "$scratch/upstream.json"
uv run --project "$repo" --no-sync steproute traces stats "$scratch/traces"
check_args=(--mode "$mode" --expect 3)
[ "$kill_after" = none ] || check_args=(--mode "$mode" --expect 1 --allow-unjoined)
uv run --project "$repo" --no-sync python "$repo/scripts/check_dsh_traces.py" "$scratch/traces" "${check_args[@]}"
[ "$status" -eq 0 ]
