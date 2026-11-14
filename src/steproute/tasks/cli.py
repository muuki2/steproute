"""``steproute tasks``: list the task set, check a patch, fetch dataset rows, estimate the grid."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from steproute.core.catalog import Catalog, CatalogError, load_catalog, load_default_catalog
from steproute.core.offpeak import offpeak_profile
from steproute.tasks.check import (
    BASELINE_PATCH,
    DEFAULT_ROWS,
    Backend,
    CheckError,
    LocalBackend,
    SwebenchBackend,
    fetch_rows,
    find_harness,
    load_rows,
    run_check,
)
from steproute.tasks.estimate import (
    PLANS,
    SCENARIOS,
    build_plans,
    estimate_plan,
    load_assumptions,
    months_needed,
)
from steproute.tasks.manifest import (
    ManifestError,
    TaskManifest,
    load_default_manifest,
    load_manifest,
)

__all__ = ["add_tasks"]

MONTHLY_CAP = {"anthropic": 80.0, "deepseek": 10.0}  # decision D3 (DeepSeek: $10 in total)
#: Providers funded once, not monthly (maintainer decision, 13 Nov 2026), and run off-peak only.
TOTAL_BUDGET = {"deepseek": 10.0}


def add_tasks(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    tasks = sub.add_parser("tasks", help="Benchmark tasks: list, check a patch, estimate cost.")
    tsub = tasks.add_subparsers(dest="tasks_command", metavar="<subcommand>", required=True)

    ls = tsub.add_parser("list", help="List the tasks of a manifest.")
    ls.add_argument("--manifest", type=Path, help="manifest TOML (default: the built-in set)")
    ls.add_argument("--pilot", action="store_true", help="only the pilot slice")
    ls.set_defaults(func=_list)

    chk = tsub.add_parser("check", help="Apply a patch to a task and run its check.")
    chk.add_argument("task_id")
    chk.add_argument(
        "--patch",
        required=True,
        help="patch file, 'gold' (the reference fix), 'empty', or 'baseline' (no code change, "
        "but the tests really run: the image check that must fail; the swebench harness skips "
        "an 'empty' patch without running anything)",
    )
    chk.add_argument("--manifest", type=Path, help="manifest TOML (default: the built-in set)")
    chk.add_argument(
        "--rows", type=Path, default=DEFAULT_ROWS, help=f"dataset rows (default {DEFAULT_ROWS})"
    )
    chk.add_argument(
        "--harness",
        help="command that runs swebench.harness.run_evaluation (default: the pinned "
        "environment of scripts/install_swebench.sh when installed, else `python -m ...`)",
    )
    chk.add_argument("--timeout", type=float, help="seconds for the check (default: the task's)")
    chk.add_argument("--keep", action="store_true", help="keep the workspace or harness logs")
    chk.add_argument("--json", action="store_true", help="print the result as JSON")
    chk.set_defaults(func=_check)

    fetch = tsub.add_parser("fetch", help="Download the SWE-bench Verified rows (about 9 MB).")
    fetch.add_argument("--out", type=Path, default=DEFAULT_ROWS)
    fetch.add_argument("--dataset", default="SWE-bench/SWE-bench_Verified")
    fetch.set_defaults(func=_fetch)

    est = tsub.add_parser("estimate", help="Estimate the spend of a replay plan.")
    est.add_argument("--plan", choices=(*PLANS, "all"), default="all")
    est.add_argument("--tasks", type=int, default=150, help="tasks in the full set")
    est.add_argument("--pilot-tasks", type=int, default=20)
    est.add_argument("--assumptions", type=Path, help="assumptions TOML (default: built-in)")
    est.add_argument("--catalog", type=Path, help="catalog TOML (default: built-in)")
    est.add_argument(
        "--no-cache", action="store_true", help="Anthropic runs set no cache breakpoints"
    )
    est.add_argument("--per-config", action="store_true", help="also print every configuration")
    est.set_defaults(func=_estimate)


def _manifest(path: Path | None) -> TaskManifest | None:
    try:
        return load_manifest(path) if path else load_default_manifest()
    except (ManifestError, OSError) as exc:
        print(f"steproute: {exc}", file=sys.stderr)
        return None


def _list(args: argparse.Namespace) -> int:
    m = _manifest(args.manifest)
    if m is None:
        return 2
    tasks = m.pilot() if args.pilot else m.tasks
    for t in tasks:
        fits = {True: "fits32k", False: "long", None: "-"}[t.fits32k]
        print(f"{t.id}  {t.stratum or '-'}  {fits}{'  pilot' if t.pilot else ''}")
    print(f"{len(tasks)} task(s) in {m.name}", file=sys.stderr)
    return 0


def _check(args: argparse.Namespace) -> int:
    m = _manifest(args.manifest)
    if m is None:
        return 2
    try:
        task = m.get(args.task_id)
    except KeyError:
        print(f"steproute: no task {args.task_id!r} in {m.name}", file=sys.stderr)
        return 2
    try:
        backend: Backend
        if task.backend == "local":
            backend = LocalBackend(keep_workspace=args.keep)
        else:
            if not args.rows.exists():
                print(
                    f"steproute: no dataset rows at {args.rows}; run `steproute tasks fetch`",
                    file=sys.stderr,
                )
                return 2
            harness = shlex.split(args.harness) if args.harness else find_harness()
            backend = SwebenchBackend(load_rows(args.rows), harness=harness, keep_logs=args.keep)
        if args.patch == "empty":
            patch = ""
        elif args.patch == "baseline":
            patch = BASELINE_PATCH
        elif args.patch == "gold":
            if isinstance(backend, SwebenchBackend):
                patch = backend.gold_patch(task)
            elif task.gold_patch is not None:
                patch = task.gold_patch.read_text(encoding="utf-8")
            else:
                print(f"steproute: task {task.id!r} has no gold patch", file=sys.stderr)
                return 2
        else:
            patch = Path(args.patch).read_text(encoding="utf-8")
        result = run_check(task, patch, backend, timeout_s=args.timeout)
    except (CheckError, OSError) as exc:
        print(f"steproute: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result.to_dict()))
    else:
        print(f"{result.status.upper()}  {task.id}: {result.detail} ({result.duration_ms:.0f} ms)")
    return {"passed": 0, "error": 3}.get(result.status, 1)


def _fetch(args: argparse.Namespace) -> int:
    try:
        n = fetch_rows(args.out, dataset=args.dataset)
    except OSError as exc:
        print(f"steproute: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {n} rows to {args.out}")
    return 0


def _estimate(args: argparse.Namespace) -> int:
    try:
        catalog = load_catalog(args.catalog) if args.catalog else load_default_catalog()
        assumptions = load_assumptions(args.assumptions)
    except (CatalogError, OSError, ValueError, KeyError) as exc:
        print(f"steproute: {exc}", file=sys.stderr)
        return 2
    plans = build_plans(catalog, args.tasks, args.pilot_tasks)
    names = PLANS if args.plan == "all" else (args.plan,)
    # The same catalog at off-peak prices, for providers that run only off-peak.
    off_catalog = Catalog(tuple(offpeak_profile(m) or m for m in catalog.models))
    print("USD by provider: low / central / high; months at the monthly cap (D3)")
    print(
        "DeepSeek runs only off-peak (half price): its line is at peak (list) prices, the "
        f"next line at the scheduled off-peak price, against its ${TOTAL_BUDGET['deepseek']:g} "
        "total budget.\n"
    )
    for name in names:
        est = estimate_plan(plans[name], catalog, assumptions, caching=not args.no_cache)
        off = estimate_plan(plans[name], off_catalog, assumptions, caching=not args.no_cache)
        print(f"{name}: {est.plan.description}")
        for prov in sorted(est.usd):
            usd = est.usd[prov]
            cells = " / ".join(f"${usd[s]:,.0f}" for s in SCENARIOS)
            line = f"  {prov:10} {est.runs[prov]:5} runs  {cells}"
            if prov in TOTAL_BUDGET:
                line += "  at peak (list) prices"
            elif prov in MONTHLY_CAP:
                months = [months_needed(usd[s], MONTHLY_CAP[prov]) for s in SCENARIOS]
                line += f"  months {months[0]} / {months[1]} / {months[2]}"
            if prov in TOTAL_BUDGET:
                o = off.usd[prov]
                budget = TOTAL_BUDGET[prov]
                fits = "fits" if o["central"] <= budget else "does NOT fit"
                print(line)
                line = (
                    f"  {'':10} {'':5}       "
                    + " / ".join(f"${o[s]:,.2f}" for s in SCENARIOS)
                    + f"  scheduled off-peak; central {fits} in the ${budget:g} total"
                )
            hours = est.hours[prov]
            if prov == "ollama":
                line += f"  ({hours['central']:,.0f} h of local compute, central)"
            print(line)
        if args.per_config:
            for model, effort, n, ests in est.per_config:
                c = ests["central"]
                print(
                    f"    {model:20} {effort:7} x{n:<4} ${c.usd:.3f}/run central "
                    f"(${ests['low'].usd:.3f}-${ests['high'].usd:.3f}), {c.steps:.0f} steps, "
                    f"{c.input_tokens / 1e6:.2f}M in, {c.output_tokens / 1e3:.1f}K out"
                )
        print()
    return 0
