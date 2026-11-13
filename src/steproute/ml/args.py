"""Arguments of ``steproute ml``. Standard library only, so building the CLI parser imports
neither ``lightgbm`` nor ``numpy``."""

from __future__ import annotations

import argparse
from pathlib import Path

__all__ = ["add_arguments"]


def add_arguments(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="ml_command", metavar="<subcommand>", required=True)

    train = sub.add_parser(
        "train",
        help="Train the tabular predictor on a replay grid.",
        description="Train, calibrate and save the LightGBM predictor. Same grid, catalog, "
        "seed and library versions give the same model. See docs/MODEL-CARD-TABULAR.md.",
    )
    train.add_argument("--data", type=Path, required=True, help="replay grid JSONL")
    train.add_argument("--out", type=Path, required=True, help="model directory to write")
    train.add_argument("--catalog", type=Path, help="catalog TOML (default: the built-in one)")
    train.add_argument("--seed", type=int, default=0, help="seed for splits and LightGBM (0)")
    train.add_argument(
        "--probes", type=int, default=50, help="size of the probe set drawn from training (50)"
    )
    train.add_argument(
        "--calibration",
        choices=("auto", "platt", "isotonic", "none"),
        default="auto",
        help="calibration method, fitted on held-out tasks; auto picks by cross-validated "
        "log loss (auto)",
    )
    train.add_argument(
        "--max-history",
        type=int,
        default=1,
        help="longest failed history to train conditional cascade rows for (1; 0 = none)",
    )
    train.add_argument("--rounds", type=int, default=300, help="most boosting rounds (300)")
    train.add_argument(
        "--early-stopping",
        type=int,
        default=30,
        help="stop after this many rounds without gain on the tune tasks; 0 disables (30)",
    )
    train.add_argument("--learning-rate", type=float, default=0.05, help="learning rate (0.05)")
    train.add_argument("--leaves", type=int, default=15, help="leaves per tree (15)")
    train.add_argument("--min-leaf", type=int, default=40, help="smallest leaf, in rows (40)")
    train.add_argument(
        "--holdout-model",
        action="append",
        default=[],
        metavar="ID",
        help="give this model no training rows (cold start); repeatable",
    )
    train.add_argument(
        "--group-feature",
        metavar="KEY",
        help="task feature whose values must not be split across roles (a repository, a stratum)",
    )

    ev = sub.add_parser(
        "evaluate",
        help="Calibration metrics and the evaluator's report for a trained model.",
        description="Score a trained model on its own test tasks: calibration before and "
        "after, and the predictor inside ScoredPolicy against every single configuration and "
        "the rule baseline. Writes report.json, report.svg, calibration.json, calibration.md.",
    )
    ev.add_argument("--model", type=Path, required=True, help="model directory from `ml train`")
    ev.add_argument("--data", type=Path, required=True, help="replay grid JSONL (as in training)")
    ev.add_argument("--out-dir", type=Path, required=True, help="directory for the outputs")
    ev.add_argument("--catalog", type=Path, help="catalog TOML (default: the built-in one)")
    ev.add_argument(
        "--rules", type=Path, help="rules TOML for the rule baseline (default: built-in)"
    )
    ev.add_argument(
        "--tau",
        type=float,
        action="append",
        help="success threshold of the policy (repeatable; default 0.7 0.8 0.9)",
    )
    ev.add_argument(
        "--probes",
        type=int,
        help="use only this many probe results per configuration (default: all; 0 = cold start)",
    )
    ev.add_argument(
        "--holdout-model",
        action="append",
        default=[],
        metavar="ID",
        help="mark this model as held out in the per-configuration table; repeatable",
    )
    ev.add_argument("--bootstrap", type=int, default=2000, help="evaluator bootstrap replicates")
    ev.add_argument(
        "--metric-bootstrap", type=int, default=300, help="calibration bootstrap replicates (300)"
    )
    ev.add_argument("--seed", type=int, default=0, help="bootstrap seed (0)")
