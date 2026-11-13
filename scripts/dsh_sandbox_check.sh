#!/usr/bin/env bash
# Prove the loopback-only sandbox profile (scripts/dsh-loopback-only.sb) works before any dsh
# session runs inside it: from inside the sandbox, requests to an external host (by name and
# by address) must fail, and a request to the local mock upstream must succeed. Uses curl and
# the Node on PATH (put Node 24 first on PATH, as scripts/dsh_session.sh needs).
#
#   scripts/dsh_sandbox_check.sh MOCK_URL [EXTERNAL_HOST] [EXTERNAL_IP]
#
# Exits 0 only when every external attempt failed and every loopback attempt succeeded.
set -uo pipefail

mock=${1:?usage: $0 MOCK_URL [EXTERNAL_HOST] [EXTERNAL_IP]}
host=${2:-nodejs.org}
ip=${3:-}
profile="$(cd "$(dirname "$0")" && pwd)/dsh-loopback-only.sb"
sb() { /usr/bin/sandbox-exec -f "$profile" "$@"; }
ok=1

expect_fail() { # label, command...
  local label=$1
  shift
  if out=$(sb "$@" 2>&1); then
    echo "FAIL  $label: succeeded inside the sandbox"
    ok=0
  else
    echo "ok    $label: blocked ($(printf '%s' "$out" | tr '\n' ' ' | cut -c1-140))"
  fi
}
expect_ok() {
  local label=$1
  shift
  if out=$(sb "$@" 2>&1); then
    echo "ok    $label: reached ($(printf '%s' "$out" | tr '\n' ' ' | cut -c1-100))"
  else
    echo "FAIL  $label: blocked inside the sandbox ($(printf '%s' "$out" | tr '\n' ' ' | cut -c1-140))"
    ok=0
  fi
}

body='{"model":"deepseek-flash","max_tokens":8,"messages":[{"role":"user","content":"ping"}]}'
fetch_js='const u=process.argv[1]; const o=process.argv[2]?{method:"POST",headers:{"content-type":"application/json"},body:process.argv[2]}:{}; fetch(u,{...o,signal:AbortSignal.timeout(8000)}).then(async r=>{console.log(r.status, (await r.text()).slice(0,60))}).catch(e=>{console.error(String(e.cause?.code??e.cause??e));process.exit(1)})'

echo "profile: $profile"
echo "node: $(command -v node) $(node --version)"
expect_fail "curl https://$host" curl -sS -m 8 -o /dev/null "https://$host/"
expect_fail "node fetch https://$host" node -e "$fetch_js" "https://$host/"
if [ -n "$ip" ]; then
  expect_fail "curl http://$ip (by address)" curl -sS -m 8 -o /dev/null "http://$ip/"
  expect_fail "node fetch http://$ip (by address)" node -e "$fetch_js" "http://$ip/"
fi
expect_ok "curl $mock/v1/messages" curl -sS -m 8 -H 'Content-Type: application/json' -d "$body" "$mock/v1/messages"
expect_ok "node fetch $mock/v1/messages" node -e "$fetch_js" "$mock/v1/messages" "$body"
[ "$ok" = 1 ] && echo "sandbox check: PASSED" || echo "sandbox check: FAILED"
[ "$ok" = 1 ]
