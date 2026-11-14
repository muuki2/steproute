#!/usr/bin/env bash
# Regenerate the checked-in synthetic reports for the external baselines (issue 22).
#
#   scripts/synthetic_baselines_report.sh
#
# Writes docs/synthetic/baselines/{paretobandit,graphrouter,routellm}/{report.json,report.svg}.
# The data is the evaluator's synthetic generator (the same grid, seed and split as
# scripts/synthetic_tabular_report.sh), so every output is stamped "SYNTHETIC DATA. NOT A
# RESULT." Nothing is downloaded and no model file is kept. Needs the baselines and ml extras
# (installed from uv.lock); a few minutes on a laptop CPU. Not part of scripts/ci.sh.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

out=docs/synthetic/baselines
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
run() { uv run --locked --extra ml --extra baselines "$@"; }

run python -m steproute.eval.synthetic --out "$work/data" --tasks 1000 --seed 0 > /dev/null
mkdir -p "$out"
run python -m steproute.baselines all --data "$work/data/grid.jsonl" \
  --catalog "$work/data/catalog.toml" --rules "$work/data/rules.toml" --out-dir "$out" --seed 0
