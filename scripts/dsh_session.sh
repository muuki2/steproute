#!/usr/bin/env bash
# A scripted real dsh session (the pinned CLI, `dsh headless`) with the steproute plugin, for
# acceptance criteria 1 and 2 of issue 15 (docs/DSH.md, "Check it works"). Opt-in: it needs
# the pinned dsh install (tests/dsh/pinned, `npm ci --ignore-scripts`) and Node 24.9 or later
# first on PATH (the engines field of @deepseek-ai/dsh 0.2.2-rc.3). scripts/ci.sh does not run
# it. macOS only: dsh runs inside `sandbox-exec` with scripts/dsh-loopback-only.sb, which
# blocks every outbound connection that is not to loopback.
#
#   PATH=/path/to/node-v24.x-darwin-arm64/bin:$PATH \
#     scripts/dsh_session.sh shadow|route SCRATCH_DIR [KILL_AT_REQUEST|none]
#
# SCRATCH_DIR must not exist yet and must be outside this repository. The script:
#   1. makes SCRATCH_DIR/{home,dsh-home,agents,work} and two small files in work/;
#   2. starts the mock upstream (scripts/dsh_mock_upstream.py) and `steproute serve` with the
#      plugin's service config, both on loopback and outside the sandbox;
#   3. proves the sandbox profile first (scripts/dsh_sandbox_check.sh: an external host must
#      be unreachable from inside it, the mock reachable) and stops if it does not hold;
#   4. writes SCRATCH_DIR/session.patch.yml, which turns off every outbound path found in
#      dsh 0.2.2-rc.3 (docs/INTEGRATION-NOTES.md section 13), points the DeepSeek route at
#      the mock, and inserts the plugin row;
#   5. runs `dsh headless --patch session.patch.yml --json "<task>"` inside the sandbox, with
#      HOME, DSH_HOME and DSH_AGENTS_HOME inside SCRATCH_DIR and a clean environment;
#   6. kills the service with SIGKILL when the mock receives model request KILL_AT_REQUEST of
#      the session, before answering it (default 2; "none" keeps it running): the mock does
#      the kill itself, so it lands at that exact point;
#   7. prints dsh's own session log per step (scripts/dsh_session_log.mjs), the requests the
#      mock received, the sandbox's denials, and checks the store (scripts/check_dsh_traces.py).
# No API key exists: the DeepSeek route reads its key from STEPROUTE_MOCK_UPSTREAM_KEY, a
# placeholder the mock never checks.
set -euo pipefail

mode=${1:-}
scratch=${2:-}
kill_at=${3:-2}
usage="usage: PATH=<node 24.9+ bin>:\$PATH $0 shadow|route SCRATCH_DIR [KILL_AT_REQUEST|none]"
case "$mode" in shadow | route) ;; *) echo "$usage" >&2; exit 2 ;; esac
[ -n "$scratch" ] || { echo "$usage" >&2; exit 2; }

repo=$(cd "$(dirname "$0")/.." && pwd)
dsh="$repo/tests/dsh/pinned/node_modules/@deepseek-ai/dsh/lib/bin.js"
profile="$repo/scripts/dsh-loopback-only.sb"
[ -f "$dsh" ] || { echo "the pinned dsh is not installed: (cd tests/dsh/pinned && npm ci --ignore-scripts)" >&2; exit 2; }
node -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a>24||(a===24&&b>=9)?0:1)' || {
  echo "dsh 0.2.2-rc.3 needs Node 24.9 or later (found $(node --version)); not running." >&2
  echo "Put a Node 24 build's bin directory first on PATH for this command only, e.g." >&2
  echo "  PATH=/path/to/node-v24.x-darwin-arm64/bin:\$PATH $0 $*" >&2
  exit 3
}
[ -x /usr/bin/sandbox-exec ] || { echo "/usr/bin/sandbox-exec is missing; dsh is never run unsandboxed" >&2; exit 3; }
[ ! -e "$scratch" ] || { echo "$scratch already exists" >&2; exit 2; }
mkdir -p "$scratch"/{home,dsh-home,agents,work}
scratch=$(cd "$scratch" && pwd)
case "$scratch/" in "$repo"/*) echo "SCRATCH_DIR must be outside the repository" >&2; exit 2 ;; esac
printf 'Alpha is a small coastal town known for its lighthouse.\n' > "$scratch/work/a.txt"
printf 'Beta describes a bicycle repair shop that opens at dawn.\n' > "$scratch/work/b.txt"

pids=()
cleanup() { for p in ${pids[@]+"${pids[@]}"}; do kill "$p" 2> /dev/null || true; wait "$p" 2> /dev/null || true; done; }
trap cleanup EXIT
py=$(uv run --project "$repo" --no-sync python -c 'import sys; print(sys.executable)')
serve_bin=$(dirname "$py")/steproute

wait_line() { # file pattern
  for _ in $(seq 1 100); do grep -q "$2" "$1" 2> /dev/null && return 0; sleep 0.1; done
  return 1
}
"$py" "$repo/scripts/dsh_mock_upstream.py" --out "$scratch/upstream.json" --count-file "$scratch/requests.count" \
  --kill-plan "$scratch/kill.plan" \
  > "$scratch/mock.out" 2> "$scratch/mock.err" &
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

# The sandbox must hold before dsh runs in it. The check's own requests to the mock are
# counted, so the counter restarts afterwards.
# One external address, resolved outside the sandbox, so the check also covers a connection
# that needs no name lookup.
ext_ip=$("$py" -c 'import socket; print(socket.gethostbyname("nodejs.org"))' 2> /dev/null || true)
"$repo/scripts/dsh_sandbox_check.sh" "$mock_url" nodejs.org "$ext_ip" > "$scratch/sandbox-check.txt" 2>&1 || {
  cat "$scratch/sandbox-check.txt" >&2
  echo "the sandbox does not block outbound connections; dsh was not run" >&2
  exit 3
}
tail -1 "$scratch/sandbox-check.txt"
offset=$(cat "$scratch/requests.count" 2> /dev/null || echo 0)
[ "$kill_at" = none ] || echo "$((offset + kill_at)) $serve_pid" > "$scratch/kill.plan"

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
echo "node $(command -v node) $(node --version), dsh $(node -p "require('$repo/tests/dsh/pinned/node_modules/@deepseek-ai/dsh/package.json').version"), mode $mode"
started=$(date '+%Y-%m-%d %H:%M:%S')
(cd "$scratch/work" && env -i HOME="$scratch/home" DSH_HOME="$scratch/dsh-home" DSH_AGENTS_HOME="$scratch/agents" \
  PATH="$PATH" LANG="${LANG:-en_US.UTF-8}" TERM=dumb TMPDIR="$scratch/home" \
  DSH_TELEMETRY_DISABLED=1 DSH_TELEMETRY_MODE=DISABLED DSH_TELEMETRY_OTLP_URL=http://127.0.0.1:9/v1/logs \
  STEPROUTE_MOCK_UPSTREAM_KEY=placeholder-for-the-mock STEPROUTE_STATUS=stderr \
  /usr/bin/sandbox-exec -f "$profile" \
  node "$dsh" headless --patch "$scratch/session.patch.yml" --json "$task" \
  > "$scratch/dsh.jsonl" 2> "$scratch/dsh.err") &
dsh_pid=$!
set +e
wait "$dsh_pid"
status=$?
set -e
echo "dsh exit status: $status"
grep 'killed pid\|kill plan' "$scratch/mock.err" || true
cleanup
trap - EXIT

echo "== dsh stderr (last lines)"
tail -15 "$scratch/dsh.err" || true
echo "== network denials by the sandbox since the session started (system log)"
/usr/bin/log show --style compact --start "$started" --predicate 'sender == "Sandbox" AND eventMessage CONTAINS "network-outbound"' \
  2> /dev/null | grep 'deny' | tail -20 > "$scratch/sandbox-denials.txt" || true
if [ -s "$scratch/sandbox-denials.txt" ]; then cat "$scratch/sandbox-denials.txt"; else echo "(none found)"; fi
echo "== dsh's own session log, per step"
node "$repo/scripts/dsh_session_log.mjs" "$scratch/dsh-home" | tee "$scratch/session-log-summary.jsonl"
echo "== requests the mock received"
cat "$scratch/upstream.json"
echo
echo "== trace store"
uv run --project "$repo" --no-sync steproute traces stats "$scratch/traces"
check_args=(--mode "$mode" --expect 3)
[ "$kill_at" = none ] || check_args=(--mode "$mode" --expect 1 --allow-unjoined)
uv run --project "$repo" --no-sync python "$repo/scripts/check_dsh_traces.py" "$scratch/traces" "${check_args[@]}"
[ "$status" -eq 0 ]
