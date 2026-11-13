#!/usr/bin/env bash
# A scripted Claude Code session with three subagent spawns, for acceptance criterion 2 of
# issue 14 (docs/CLAUDE-CODE.md, "Check it works"). It makes REAL model requests on whatever
# login Claude Code has; run it only when you mean to.
#
#   scripts/claude_code_session.sh shadow|route SCRATCH_DIR
#
# SCRATCH_DIR must not exist yet and must be outside this repository. The script:
#   1. writes three small text files there and copies the plugin there;
#   2. starts `steproute serve` with the plugin's service config, storing to SCRATCH_DIR/traces;
#   3. runs one non-interactive session (`claude -p`) on the cheapest model, with only the
#      Agent and Read tools, the plugin loaded for this run only, no session saved, no MCP
#      servers, and a clean environment (no API key is passed on);
#   4. stops the service and checks the store (scripts/check_claude_code_traces.py).
# Claude Code's own JSON result (with modelUsage per model) is kept in SCRATCH_DIR/result.json.
set -euo pipefail

mode=${1:-}
scratch=${2:-}
case "$mode" in shadow | route) ;; *) echo "usage: $0 shadow|route SCRATCH_DIR" >&2; exit 2 ;; esac
[ -n "$scratch" ] || { echo "usage: $0 shadow|route SCRATCH_DIR" >&2; exit 2; }

repo=$(cd "$(dirname "$0")/.." && pwd)
[ ! -e "$scratch" ] || { echo "$scratch already exists" >&2; exit 2; }
mkdir -p "$scratch"
scratch=$(cd "$scratch" && pwd)
case "$scratch/" in "$repo"/*) echo "SCRATCH_DIR must be outside the repository" >&2; exit 2 ;; esac
command -v claude > /dev/null || { echo "claude is not on PATH" >&2; exit 2; }

mkdir -p "$scratch/work"
printf 'Alpha is a small coastal town known for its lighthouse and its Friday fish market.\n' > "$scratch/work/a.txt"
printf 'Beta describes a bicycle repair shop that opens at dawn and closes at noon.\n' > "$scratch/work/b.txt"
printf 'Gamma is a note about watering tomato plants twice a week in summer.\n' > "$scratch/work/c.txt"
cp -R "$repo/plugins/claude-code" "$scratch/plugin"

steproute() { uv run --project "$repo" --no-sync steproute "$@"; }
# The service is started from the environment's own executable, not through `uv run`, so
# that $! is the service itself and stopping it leaves nothing behind.
serve_bin=$(uv run --project "$repo" --no-sync python -c 'import shutil; print(shutil.which("steproute"))')

"$serve_bin" serve --config "$repo/plugins/claude-code/service/steproute.toml" --port 0 \
  --store "$scratch/traces" > "$scratch/serve.out" 2> "$scratch/serve.err" &
serve_pid=$!
trap 'kill "$serve_pid" 2> /dev/null || true; wait "$serve_pid" 2> /dev/null || true' EXIT
for _ in $(seq 1 100); do
  grep -q 'listening on' "$scratch/serve.out" 2> /dev/null && break
  sleep 0.1
done
url=$(sed -n 's/^steproute serve: listening on //p' "$scratch/serve.out")
[ -n "$url" ] || { echo "steproute serve did not start:" >&2; cat "$scratch/serve.err" >&2; exit 1; }
echo "service: $url (store $scratch/traces)"

prompt='Use the Agent tool three times to start three subagents, one per file: a.txt, b.txt and c.txt in the current directory. Use the general-purpose agent type and do not pass a model parameter. Each subagent must read its one file with the Read tool and reply with a one-sentence summary. When all three have answered, reply with the three sentences.'

echo "claude: mode $mode, Claude Code $(claude --version)"
set +e
(cd "$scratch/work" && env -i HOME="$HOME" USER="${USER:-}" LOGNAME="${LOGNAME:-}" \
  PATH="$PATH" LANG="${LANG:-en_US.UTF-8}" TERM=dumb \
  STEPROUTE_MODE="$mode" STEPROUTE_URL="$url" \
  claude -p "$prompt" \
  --plugin-dir "$scratch/plugin" \
  --model haiku \
  --tools Agent,Read \
  --allowedTools Agent,Read \
  --strict-mcp-config \
  --no-session-persistence \
  --output-format json \
  --debug-file "$scratch/debug.log" \
  > "$scratch/result.json" 2> "$scratch/claude.err")
status=$?
set -e
echo "claude exit status: $status"
sleep 1
kill "$serve_pid" 2> /dev/null || true
wait "$serve_pid" 2> /dev/null || true
trap - EXIT

steproute traces stats "$scratch/traces"
uv run --project "$repo" --no-sync python "$repo/scripts/check_claude_code_traces.py" \
  "$scratch/traces" --mode "$mode"
