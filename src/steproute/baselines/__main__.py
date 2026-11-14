"""``python -m steproute.baselines``: train an external baseline on a grid and write its report.

    python -m steproute.baselines paretobandit --data grid.jsonl --catalog catalog.toml \\
        --rules rules.toml --out-dir out/

writes ``out/paretobandit/report.json`` and ``report.svg``: the baseline's policies, the rule
baseline and the hindsight oracle against every single configuration, on the ``test`` tasks of
the same split as the tabular predictor. ``all`` runs every baseline. A synthetic grid gives a
report stamped ``SYNTHETIC DATA. NOT A RESULT.``. Needs the ``baselines`` extra.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from pathlib import Path

BASELINES = ("paretobandit", "graphrouter", "routellm")
_NEEDS = {
    "paretobandit": ("pareto_bandit",),
    "graphrouter": ("torch", "torch_geometric"),
    "routellm": ("torch",),
}


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m steproute.baselines",
        description="Train an external baseline on a replay grid and write its evaluator report.",
    )
    ap.add_argument("baseline", choices=(*BASELINES, "all"))
    ap.add_argument("--data", type=Path, required=True, help="replay grid (JSON Lines)")
    ap.add_argument("--catalog", type=Path, help="catalog TOML (default: the built-in one)")
    ap.add_argument("--rules", type=Path, help="rules TOML for the rule baseline")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0, help="split, training and bootstrap seed (0)")
    ap.add_argument("--probes", type=int, default=50, help="probe tasks, as the tabular model (50)")
    ap.add_argument("--bootstrap", type=int, default=2000, help="bootstrap replicates (2000)")
    ap.add_argument(
        "--exclude-config",
        action="append",
        default=[],
        metavar="EFFORT:MODEL",
        help="leave a configuration out of the arms and the frontier (repeatable), "
        "e.g. one that ran on only part of a sparse grid",
    )
    ap.add_argument(
        "--pair",
        help="routellm only: WEAK,STRONG as EFFORT:MODEL each (default: cheapest, strongest)",
    )
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    names = BASELINES if args.baseline == "all" else (args.baseline,)
    for name in names:
        for mod in _NEEDS[name]:
            try:
                importlib.import_module(mod)
            except ImportError:
                print(
                    f"steproute: {name} needs the [baselines] extra ({mod} is missing): "
                    "uv sync --extra baselines",
                    file=sys.stderr,
                )
                return 2

    from steproute.baselines.common import BaselineData, ContextEncoder, baseline_report
    from steproute.core.catalog import load_catalog, load_default_catalog
    from steproute.eval.data import load_grid
    from steproute.eval.report import write_report
    from steproute.eval.results import EvalSettings
    from steproute.eval.targets import parse_action
    from steproute.ml.stamp import STAMP

    grid = load_grid(args.data)
    catalog = (
        load_catalog(args.catalog, warn_unverified=False)
        if args.catalog
        else load_default_catalog(warn_unverified=False)
    )
    excluded = [(a.model, a.effort) for a in map(parse_action, args.exclude_config)]
    data = BaselineData(
        grid, catalog, seed=args.seed, n_probes=args.probes, exclude_configs=excluded
    )
    encoder = ContextEncoder(catalog).fit([data.state(t) for t in data.tasks("train")])
    settings = EvalSettings(bootstrap=args.bootstrap, seed=args.seed)
    for name in names:
        module = importlib.import_module(f"steproute.baselines.{name}")
        kwargs: dict[str, object] = {"seed": args.seed}
        if name == "routellm" and args.pair:
            weak, _, strong = args.pair.partition(",")
            kwargs["pair"] = (weak.strip(), strong.strip())
        policies, provenance = module.build_policies(data, encoder, **kwargs)
        report, svg, _ = baseline_report(
            data,
            policies,
            provenance,
            rules=None if args.rules is None else str(args.rules),
            settings=settings,
            command=["python", "-m", "steproute.baselines", name, "..."],
        )
        out = args.out_dir / name
        out.mkdir(parents=True, exist_ok=True)
        write_report(out / "report.json", report)
        (out / "report.svg").write_text(svg, encoding="utf-8")
        if data.synthetic:
            print(STAMP)
        for sentence in report["summary"]:
            print(f"  {sentence}")
        print(f"wrote {out}/report.json, report.svg")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
