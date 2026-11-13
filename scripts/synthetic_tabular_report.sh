#!/usr/bin/env bash
# Regenerate the checked-in synthetic report for the tabular predictor (issue 21).
#
#   scripts/synthetic_tabular_report.sh
#
# Writes docs/synthetic/tabular-predictor/{report.json,report.svg,calibration.json,
# calibration.md}. The data is the evaluator's synthetic generator, so every output is stamped
# "SYNTHETIC DATA. NOT A RESULT." The trained model goes to a temporary directory and is
# deleted: no model file is checked in. Needs the ml extra (installed from uv.lock).
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

out=docs/synthetic/tabular-predictor
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
run() { uv run --locked --extra ml "$@"; }

run python -m steproute.eval.synthetic --out "$work/data" --tasks 1000 --seed 0 > /dev/null
run steproute ml train --data "$work/data/grid.jsonl" --catalog "$work/data/catalog.toml" \
  --out "$work/model" --seed 0
mkdir -p "$out"
run steproute ml evaluate --model "$work/model" --data "$work/data/grid.jsonl" \
  --catalog "$work/data/catalog.toml" --rules "$work/data/rules.toml" --out-dir "$out" --seed 0
