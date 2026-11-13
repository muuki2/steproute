#!/usr/bin/env bash
# An in-process dsh session on the real dsh 0.2.2-rc.3 libraries, with the steproute plugin
# loaded and the service killed part-way through (docs/DSH.md, "Check it works"). Opt-in: it
# needs the pinned dsh install (tests/dsh/pinned, `npm ci --ignore-scripts`), so scripts/ci.sh
# does not run it. No model provider is involved: a scripted adapter answers as route
# `deepseek-official`, and nothing opens a connection except the plugin's loopback calls.
#
#   scripts/dsh_inprocess_session.sh shadow|route SCRATCH_DIR [KILL_AFTER_REQUEST]
#
# SCRATCH_DIR must not exist yet and must be outside this repository. KILL_AFTER_REQUEST
# (default 2) is the model request after which the service gets SIGKILL; "none" keeps it.
# HOME for the session is SCRATCH_DIR/home, so nothing is read from or written to yours.
set -euo pipefail

mode=${1:-}
scratch=${2:-}
kill_after=${3:-2}
case "$mode" in shadow | route) ;; *) echo "usage: $0 shadow|route SCRATCH_DIR [KILL_AFTER_REQUEST|none]" >&2; exit 2 ;; esac
[ -n "$scratch" ] || { echo "usage: $0 shadow|route SCRATCH_DIR [KILL_AFTER_REQUEST|none]" >&2; exit 2; }

repo=$(cd "$(dirname "$0")/.." && pwd)
pinned="$repo/tests/dsh/pinned"
[ -d "$pinned/node_modules/@deepseek-ai/dsh-agent-loop" ] || {
  echo "the pinned dsh is not installed: (cd tests/dsh/pinned && npm ci --ignore-scripts)" >&2
  exit 2
}
[ ! -e "$scratch" ] || { echo "$scratch already exists" >&2; exit 2; }
mkdir -p "$scratch/home"
scratch=$(cd "$scratch" && pwd)
case "$scratch/" in "$repo"/*) echo "SCRATCH_DIR must be outside the repository" >&2; exit 2 ;; esac

serve_bin=$(uv run --project "$repo" --no-sync python -c 'import shutil; print(shutil.which("steproute"))')
"$serve_bin" serve --config "$repo/plugins/dsh/service/steproute.toml" --port 0 \
  --store "$scratch/traces" > "$scratch/serve.out" 2> "$scratch/serve.err" &
serve_pid=$!
trap 'kill "$serve_pid" 2> /dev/null || true; wait "$serve_pid" 2> /dev/null || true' EXIT
for _ in $(seq 1 100); do
  grep -q 'listening on' "$scratch/serve.out" 2> /dev/null && break
  sleep 0.1
done
url=$(sed -n 's/^steproute serve: listening on //p' "$scratch/serve.out")
[ -n "$url" ] || { echo "steproute serve did not start:" >&2; cat "$scratch/serve.err" >&2; exit 1; }
echo "service: $url (pid $serve_pid, store $scratch/traces)"

kill_args=()
[ "$kill_after" = none ] || kill_args=(--kill-after-step "$kill_after" --pid "$serve_pid")
echo "node $(node --version), mode $mode, kill after request ${kill_after}"
set +e
(cd "$pinned" && env -i HOME="$scratch/home" PATH="$PATH" LANG="${LANG:-en_US.UTF-8}" \
  node inprocess_session.mjs "$url" "$mode" ${kill_args[@]+"${kill_args[@]}"} \
  > "$scratch/session.json" 2> "$scratch/session.err")
status=$?
set -e
echo "session exit status: $status"
cat "$scratch/session.json"
# Criterion 2: every turn of the session completed, whatever happened to the service.
uv run --project "$repo" --no-sync python -c '
import json, sys
d = json.load(open(sys.argv[1]))
ends, errors = d["turnEnds"], d["errors"]
print("turn ends:", ends, "errors:", errors, "service killed:", d["killed"])
print("ran per request:", [(r["model"], r["reasoningEffort"]) for r in d["requests"]])
sys.exit(0 if ends and all(e == "completed" for e in ends) and not errors else 1)
' "$scratch/session.json"
kill "$serve_pid" 2> /dev/null || true
wait "$serve_pid" 2> /dev/null || true
trap - EXIT

uv run --project "$repo" --no-sync steproute traces stats "$scratch/traces"
check_args=(--mode "$mode" --expect 6)
[ "$kill_after" = none ] || check_args=(--mode "$mode" --expect 1 --allow-unjoined)
uv run --project "$repo" --no-sync python "$repo/scripts/check_dsh_traces.py" \
  "$scratch/traces" "${check_args[@]}"
[ "$status" -eq 0 ]
