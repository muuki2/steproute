"""``steproute ml train`` and ``steproute ml evaluate``. Needs the ``ml`` extra.

``steproute.cli`` registers the arguments from ``steproute.ml.args`` and imports this module
only when the command runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from steproute.core.catalog import Catalog, CatalogError, load_catalog, load_default_catalog
from steproute.core.rules import RuleConfigError
from steproute.eval.data import DataFormatError, load_grid
from steproute.eval.results import EvalSettings
from steproute.ml.stamp import STAMP

__all__ = ["run"]


def _err(msg: str) -> int:
    print(f"steproute: {msg}", file=sys.stderr)
    return 2


def _catalog(path: Path | None) -> Catalog:
    if path:
        return load_catalog(path, warn_unverified=False)
    return load_default_catalog(warn_unverified=False)


def run(args: argparse.Namespace) -> int:
    try:
        import lightgbm  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        return _err("the tabular predictor needs the [ml] extra: pip install 'steproute[ml]'")
    try:
        if args.ml_command == "train":
            return _train(args)
        return _evaluate(args)
    except (DataFormatError, CatalogError, RuleConfigError, OSError, ValueError) as exc:
        return _err(str(exc))


def _train(args: argparse.Namespace) -> int:
    from steproute.ml.model import TrainConfig, train

    grid = load_grid(args.data)
    cfg = TrainConfig(
        seed=args.seed,
        n_probes=args.probes,
        calibration=args.calibration,
        max_history=args.max_history,
        num_boost_round=args.rounds,
        early_stopping_rounds=args.early_stopping,
        learning_rate=args.learning_rate,
        num_leaves=args.leaves,
        min_data_in_leaf=args.min_leaf,
    )
    result = train(
        grid,
        _catalog(args.catalog),
        cfg,
        holdout_models=args.holdout_model,
        group_feature=args.group_feature,
    )
    path = result.model.save(args.out)
    s = result.summary
    if s["synthetic"]:
        print(STAMP)
    print(f"model: {path}")
    print(f"rows: {s['rows']}; tasks by role: {s['roles']}")
    print(f"boosting rounds kept: {s['best_iteration']}; calibration: {args.calibration}")
    print(f"data fingerprint: {s['data_fingerprint']}")
    if s["synthetic"]:
        print(STAMP)
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from steproute.ml.evaluate import calibration_markdown, calibration_report, evaluator_report
    from steproute.ml.model import TabularModel

    model = TabularModel.load(args.model)
    grid = load_grid(args.data)
    catalog = _catalog(args.catalog)
    if grid.sha256 != model.meta["data"]["grid_sha256"]:
        print(
            "steproute: warning: this grid is not the one the model was trained on; the "
            "test tasks are looked up by id",
            file=sys.stderr,
        )
    taus = args.tau or [0.7, 0.8, 0.9]
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    cal = calibration_report(
        model,
        grid,
        catalog,
        n_probes=args.probes,
        replicates=args.metric_bootstrap,
        seed=args.seed,
        holdout_models=args.holdout_model,
    )
    (out / "calibration.json").write_text(
        json.dumps(cal, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (out / "calibration.md").write_text(calibration_markdown(cal), encoding="utf-8")

    settings = EvalSettings(bootstrap=args.bootstrap, seed=args.seed)
    report, svg, _ = evaluator_report(
        model,
        grid,
        catalog,
        taus=taus,
        n_probes=args.probes,
        rules=None if args.rules is None else str(args.rules),
        settings=settings,
        command=["steproute", "ml", "evaluate", "..."],
    )
    from steproute.eval.report import write_report

    write_report(out / "report.json", report)
    (out / "report.svg").write_text(svg, encoding="utf-8")

    if cal.get("stamp"):
        print(cal["stamp"])
    for key, g in cal["groups"].items():
        u, c = g["uncalibrated"], g["calibrated"]
        print(
            f"{key}: ECE {u['ece']:.4f} -> {c['ece']:.4f}, Brier {u['brier']:.4f} -> "
            f"{c['brier']:.4f}, log loss {u['log_loss']:.4f} -> {c['log_loss']:.4f} "
            f"({g['n_rows']} rows, uncalibrated -> calibrated)"
        )
    for sentence in report["summary"]:
        print(f"  {sentence}")
    print(f"wrote {out}/report.json, report.svg, calibration.json, calibration.md")
    if cal.get("stamp"):
        print(cal["stamp"])
    return 0
