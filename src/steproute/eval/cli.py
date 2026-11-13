"""``steproute eval``: evaluate policies on a grid or on logged traces. Standard library only.

``steproute.cli`` registers the arguments from ``steproute.eval.args`` and imports this module
only when the command runs.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

from steproute.core.catalog import Catalog, CatalogError, load_catalog, load_default_catalog
from steproute.core.rules import RuleConfigError
from steproute.eval.data import DataFormatError, detect_kind, load_grid, load_logged
from steproute.eval.grid import evaluate_grid
from steproute.eval.ope import evaluate_logged
from steproute.eval.plot import write_svg
from steproute.eval.report import build_report, write_report
from steproute.eval.results import EvalSettings, Evaluation
from steproute.eval.targets import (
    OracleTarget,
    PolicySpecError,
    TargetPolicy,
    parse_action,
    parse_policy_spec,
)

__all__ = ["run"]


def _err(msg: str) -> int:
    print(f"steproute: {msg}", file=sys.stderr)
    return 2


def _import(spec: str) -> Any:
    module, sep, attr = spec.rpartition(":")
    if not sep or not module:
        raise PolicySpecError(f"expected MODULE:ATTR, got {spec!r}")
    obj = getattr(importlib.import_module(module), attr)
    return obj() if isinstance(obj, type) else obj


def run(args: argparse.Namespace) -> int:
    try:
        settings = EvalSettings(
            confidence=args.confidence,
            bootstrap=args.bootstrap,
            seed=args.seed,
            min_ess=args.min_ess,
            min_clusters=args.min_clusters,
            clip=args.clip,
            headline=args.headline,
            cost_weight=args.cost_weight,
            min_margin=args.min_margin,
        )
    except ValueError as exc:
        return _err(str(exc))

    cache: list[Catalog] = []

    def catalog() -> Catalog:
        if not cache:
            if args.catalog:
                cache.append(load_catalog(args.catalog, warn_unverified=False))
            else:
                cache.append(load_default_catalog(warn_unverified=False))
        return cache[0]

    try:
        policies: list[TargetPolicy] = [parse_policy_spec(s, catalog) for s in args.policy]
        logging_policy = (
            parse_policy_spec(args.logging_policy, catalog) if args.logging_policy else None
        )
        excluded = [(a.model, a.effort) for a in map(parse_action, args.exclude_config)]
        reward_model = _import(args.reward_model) if args.reward_model else None
    except (
        PolicySpecError, RuleConfigError, CatalogError, OSError, ImportError, AttributeError,
    ) as exc:  # fmt: skip
        return _err(str(exc))

    evaluations: list[Evaluation] = []
    for path in args.data:
        path = Path(path)
        try:
            if detect_kind(path) == "grid":
                ev = evaluate_grid(
                    load_grid(path), policies, settings, split=args.split,
                    exclude_configs=excluded,
                )  # fmt: skip
            else:
                usable = [p for p in policies if not isinstance(p, OracleTarget)]
                if len(usable) < len(policies):
                    print(
                        f"steproute: {path}: the oracle needs a grid; skipped on logged data",
                        file=sys.stderr,
                    )
                ev = evaluate_logged(
                    load_logged(path, catalog()), usable, settings,
                    logging_policy=logging_policy, reward_model=reward_model,
                )  # fmt: skip
                ev.provenance["catalog"] = str(args.catalog) if args.catalog else "built-in"
        except (DataFormatError, CatalogError, OSError) as exc:
            return _err(f"{path}: {exc}")
        evaluations.append(ev)

    report = build_report(evaluations, command=sys.argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.out, report)
    print(f"report: {args.out}")
    if not args.no_plot:
        target = args.plot or args.out.with_suffix(".svg")
        for p in write_svg(target, evaluations):
            print(f"plot: {p}")
    for ev in evaluations:
        print(f"\n{ev.provenance.get('source')} ({ev.kind} data, {ev.level} level)")
        for pol in ev.policies:
            print(f"  {pol.verdict.sentence}")
    return 0
