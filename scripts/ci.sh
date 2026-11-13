#!/usr/bin/env bash
# Local CI pipeline for steproute. Everything runs on this machine; the
# repository uses no hosted CI.
#
#   scripts/ci.sh            lint, type-check and test on every supported
#                            Python, then build the wheel and smoke-test it
#   scripts/ci.sh --report   the same, then publish the result as a "local-ci"
#                            commit status on HEAD so pull requests show it
#
# CI_PYTHONS overrides the interpreter list:  CI_PYTHONS=3.12 scripts/ci.sh
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

report=0
case "${1:-}" in
  "") ;;
  --report) report=1 ;;
  *) echo "usage: scripts/ci.sh [--report]" >&2; exit 2 ;;
esac

pythons=${CI_PYTHONS:-"3.11 3.12"}
first=${pythons%% *}
head=$(git rev-parse HEAD)
dirty=$(git status --porcelain)
marker=.ci/passed
stamp="$head $pythons"

step() { printf '\n== %s\n' "$*"; }

# Each interpreter gets its own environment under .ci/, so .venv is left alone.
in_env() {
  local py=$1
  shift
  UV_PROJECT_ENVIRONMENT=".ci/py$py" uv run --no-sync "$@"
}

pipeline() {
  local py dist
  for py in $pythons; do
    step "python $py: install from uv.lock"
    UV_PROJECT_ENVIRONMENT=".ci/py$py" uv sync --locked --extra dev --python "$py" --quiet
  done

  step "ruff"
  in_env "$first" ruff check .
  in_env "$first" ruff format --check .

  for py in $pythons; do
    step "python $py: mypy"
    in_env "$py" mypy
    step "python $py: pytest"
    in_env "$py" pytest -q
  done

  step "build: the wheel installs and runs outside the source tree"
  dist=$(mktemp -d)
  uv build --quiet --out-dir "$dist"
  (cd "$dist" && uv run --isolated --no-project --python "$first" \
    --with ./steproute-*.whl steproute --version)
  (cd "$dist" && uv run --isolated --no-project --python "$first" \
    --with ./steproute-*.whl python -c \
    "from steproute.core import load_rule_config, RuleConfig; assert load_rule_config() == RuleConfig()")
  rm -rf "$dist"
}

if [ "$report" = 1 ] && [ -n "$dirty" ]; then
  echo "ci: --report needs a clean working tree; commit or stash first" >&2
  exit 2
fi

if [ "$report" = 1 ] && [ -f "$marker" ] && [ "$(cat "$marker")" = "$stamp" ]; then
  # This exact commit already passed with a clean tree (usually in the
  # pre-push hook a moment ago), so there is nothing new to run.
  echo "ci: $head already passed; reporting without re-running"
  status=0
else
  rm -f "$marker"
  set +e
  (set -e; pipeline)
  status=$?
  set -e
  if [ "$status" -eq 0 ] && [ -z "$dirty" ]; then
    echo "$stamp" > "$marker"
  fi
fi

echo
if [ "$status" -eq 0 ]; then
  echo "ci: PASSED (python $pythons)"
else
  echo "ci: FAILED" >&2
fi

if [ "$report" = 1 ]; then
  state=success
  [ "$status" -eq 0 ] || state=failure
  if gh api "repos/{owner}/{repo}/statuses/$head" -f state="$state" \
    -f context=local-ci -f description="scripts/ci.sh, python $pythons" > /dev/null; then
    echo "ci: published local-ci=$state for $head"
  else
    echo "ci: could not publish the status; is $head pushed and gh signed in?" >&2
    exit 1
  fi
fi

exit "$status"
