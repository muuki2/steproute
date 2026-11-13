"""Arguments of ``steproute eval``. Kept apart from the evaluator so building the CLI parser
imports nothing else."""

from __future__ import annotations

import argparse
from pathlib import Path

__all__ = ["add_arguments"]


def add_arguments(p: argparse.ArgumentParser) -> None:
    # Defaults repeat EvalSettings'; tests/test_eval_cli.py checks they agree.
    p.add_argument(
        "--policy",
        action="append",
        required=True,
        metavar="SPEC",
        help="policy to evaluate (repeatable): fixed:EFFORT:MODEL, "
        "cascade:EFFORT:MODEL,EFFORT:MODEL, rules[:PATH], oracle (grid only, hindsight), "
        "python:MODULE:ATTR",
    )
    p.add_argument(
        "--data",
        action="append",
        required=True,
        type=Path,
        metavar="PATH",
        help="grid JSONL file, trace store directory or trace export file (repeatable)",
    )
    p.add_argument("--out", type=Path, required=True, help="JSON report to write")
    p.add_argument("--plot", type=Path, help="SVG plot to write (default: --out with .svg)")
    p.add_argument("--no-plot", action="store_true", help="do not write the plot")
    p.add_argument("--catalog", type=Path, help="catalog TOML (default: the built-in one)")
    p.add_argument("--split", help="grid: evaluate only tasks of this split (e.g. test)")
    p.add_argument(
        "--exclude-config",
        action="append",
        default=[],
        metavar="EFFORT:MODEL",
        help="grid: leave this configuration out of the frontier (repeatable)",
    )
    p.add_argument(
        "--logging-policy",
        metavar="SPEC",
        help="logged data: the policy that produced the logs, to check every propensity",
    )
    p.add_argument(
        "--reward-model",
        metavar="MODULE:ATTR",
        help="logged data: reward model for doubly robust (default: built-in, cross-fitted)",
    )
    p.add_argument("--confidence", type=float, default=0.95, help="interval level (0.95)")
    p.add_argument("--bootstrap", type=int, default=2000, help="bootstrap replicates (2000)")
    p.add_argument("--seed", type=int, default=0, help="bootstrap and fold seed (0)")
    p.add_argument("--min-ess", type=float, default=50.0, help="no headline below this ESS (50)")
    p.add_argument(
        "--min-clusters", type=int, default=100, help="no verdict below this many clusters (100)"
    )
    p.add_argument("--clip", type=float, help="cap importance weights (default: no cap)")
    p.add_argument(
        "--headline", choices=("dr", "snips", "ips"), default="dr", help="estimator (dr)"
    )
    p.add_argument(
        "--min-margin",
        type=float,
        default=0.0,
        help="smallest frontier margin, in success-rate units, that counts as a win (0)",
    )
    p.add_argument(
        "--cost-weight",
        type=float,
        help="also report and rank by utility = success - COST_WEIGHT * cost (USD)",
    )
