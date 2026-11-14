"""``steproute replay``: plan, preflight, run, resume, status, report. See ``docs/REPLAY.md``.

Every command is a dry run unless it is ``run`` or ``resume`` with ``--execute``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from steproute.core.catalog import Catalog, load_catalog, load_default_catalog
from steproute.replay.config import ReplayConfig, ReplayConfigError, load_replay_config
from steproute.replay.plan import (
    NAMED_PLANS,
    PlanError,
    ReplayPlan,
    build_named_plan,
    load_plan_file,
)
from steproute.tasks.estimate import load_assumptions
from steproute.tasks.manifest import (
    ManifestError,
    TaskManifest,
    load_default_manifest,
    load_manifest,
)

__all__ = ["add_replay"]


def add_replay(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    rp = sub.add_parser(
        "replay",
        help="Run the replay grid (dry run unless --execute).",
        description="Run each task under one fixed (model, effort) for the whole task, through "
        "the proxy, under per-provider monthly caps. Without --execute nothing runs. "
        "See docs/REPLAY.md.",
    )
    rsub = rp.add_subparsers(dest="replay_command", metavar="<subcommand>", required=True)

    def common(p: argparse.ArgumentParser, *, plan: bool = True) -> None:
        if plan:
            p.add_argument(
                "--plan", required=True, help=f"{' or '.join(NAMED_PLANS)}, or a plan TOML file"
            )
            p.add_argument(
                "--manifest", type=Path, help="task manifest (default: the built-in set)"
            )
            p.add_argument("--assumptions", type=Path, help="cost assumptions (default: built-in)")
        p.add_argument("--config", type=Path, help="replay TOML (default: safe built-in defaults)")
        p.add_argument("--out", type=Path, required=True, help="output directory of the run")

    p = rsub.add_parser("plan", help="Print the plan, estimated spend, caps and month-to-date.")
    common(p)
    p.set_defaults(func=_plan)

    def env_file(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--env-file",
            type=Path,
            help="NAME=value lines (optional `export`, quotes, # comments) loaded into this "
            "process's environment; a variable already set wins. Values are never shown.",
        )

    p = rsub.add_parser("preflight", help="Check what a real run needs; changes nothing.")
    common(p)
    env_file(p)
    p.set_defaults(func=_preflight)

    p = rsub.add_parser("run", help="Dry run by default; --execute runs it for real.")
    common(p)
    env_file(p)
    p.add_argument("--execute", action="store_true", help="really run (this spends money)")
    p.set_defaults(func=_run)

    p = rsub.add_parser("resume", help="Continue the plan recorded in --out (dry run by default).")
    common(p, plan=False)
    env_file(p)
    p.add_argument("--execute", action="store_true", help="really run (this spends money)")
    p.set_defaults(func=_resume)

    p = rsub.add_parser("status", help="Runs finished, in flight, and spend per month.")
    common(p, plan=False)
    p.set_defaults(func=_status)

    p = rsub.add_parser("report", help="Write the pilot report from a finished pilot's outputs.")
    common(p, plan=False)
    p.add_argument("--manifest", type=Path, help="task manifest (default: the one recorded)")
    p.add_argument(
        "--md", type=Path, default=Path("docs/results/pilot.md"), help="where to write the report"
    )
    p.set_defaults(func=_report)


def _err(msg: str) -> int:
    print(f"steproute replay: {msg}", file=sys.stderr)
    return 2


def _load(args: argparse.Namespace) -> tuple[ReplayConfig, Catalog, TaskManifest]:
    config = load_replay_config(args.config)
    catalog = load_catalog(config.catalog_path) if config.catalog_path else load_default_catalog()
    manifest = load_manifest(args.manifest) if args.manifest else load_default_manifest()
    return config, catalog, manifest


def _build_plan(
    spec: str, manifest: TaskManifest, catalog: Catalog, assumptions: Path | None
) -> ReplayPlan:
    a = load_assumptions(assumptions)
    if spec in NAMED_PLANS:
        return build_named_plan(spec, manifest, catalog, a)
    return load_plan_file(spec, manifest, catalog, a)


def _inputs(args: argparse.Namespace) -> dict[str, Any]:
    plan = args.plan if args.plan in NAMED_PLANS else str(Path(args.plan).resolve())
    return {
        "plan": plan,
        "manifest": str(args.manifest.resolve()) if args.manifest else None,
        "assumptions": str(args.assumptions.resolve()) if args.assumptions else None,
        "config": str(args.config.resolve()) if args.config else None,
    }


def _env(args: argparse.Namespace) -> list[str]:
    """Load ``--env-file`` into ``os.environ``; the values to redact (every value in the file
    and the value in force for each of its names)."""
    import os

    from steproute.replay.envfile import apply_env, load_env_file

    path = getattr(args, "env_file", None)
    if path is None:
        return []
    values = load_env_file(path)
    apply_env(values, os.environ)
    return [*values.values(), *(os.environ.get(n, "") for n in values)]


def _runner(args: argparse.Namespace) -> Any:
    from steproute.replay.runner import Runner, RunnerDeps

    secrets = _env(args)
    config, catalog, manifest = _load(args)
    plan = _build_plan(args.plan, manifest, catalog, args.assumptions)
    return Runner(
        plan,
        config=config,
        catalog=catalog,
        manifest=manifest,
        out=args.out,
        inputs=_inputs(args),
        deps=RunnerDeps(secrets=secrets),
    )


def _guard(fn: Any, args: argparse.Namespace) -> int:
    from steproute.replay.envfile import EnvFileError
    from steproute.replay.journal import LockHeld
    from steproute.replay.runner import MissingKey

    try:
        return int(fn(args))
    except EnvFileError as exc:
        return _err(str(exc))  # never carries a value
    except (ReplayConfigError, PlanError, ManifestError, OSError, ValueError) as exc:
        return _err(str(exc))
    except MissingKey:
        return 2
    except LockHeld as exc:
        return _err(str(exc))


def _plan(args: argparse.Namespace) -> int:
    return _guard(lambda a: _runner(a).run(execute=False), args)


def _run(args: argparse.Namespace) -> int:
    return _guard(lambda a: _runner(a).run(execute=a.execute), args)


def _recorded(args: argparse.Namespace) -> argparse.Namespace:
    path = args.out / "plan.json"
    if not path.exists():
        raise ValueError(f"{path} does not exist; start with `steproute replay run`")
    inputs = json.loads(path.read_text("utf-8")).get("inputs") or {}
    ns = argparse.Namespace(**vars(args))
    ns.plan = inputs.get("plan")
    if not ns.plan:
        raise ValueError(f"{path} does not record the plan's inputs")
    ns.manifest = getattr(args, "manifest", None) or (
        Path(inputs["manifest"]) if inputs.get("manifest") else None
    )
    ns.assumptions = Path(inputs["assumptions"]) if inputs.get("assumptions") else None
    if args.config is None and inputs.get("config"):
        ns.config = Path(inputs["config"])
    return ns


def _resume(args: argparse.Namespace) -> int:
    return _guard(lambda a: _runner(_recorded(a)).run(execute=a.execute), args)


def _preflight(args: argparse.Namespace) -> int:
    import os

    from steproute.replay.preflight import preflight

    def go(a: argparse.Namespace) -> int:
        _env(a)
        config, catalog, manifest = _load(a)
        plan = _build_plan(a.plan, manifest, catalog, a.assumptions)
        checks = preflight(plan, manifest, config, environ=os.environ, out=a.out)
        bad = 0
        for c in checks:
            mark = "ok  " if c.ok else ("FAIL" if c.required else "note")
            bad += (not c.ok) and c.required
            print(f"{mark}  {c.name:<18} {c.detail}")
        print(f"\npreflight: {'ready' if not bad else f'{bad} problem(s); a real run would fail'}")
        return 0 if not bad else 1

    return _guard(go, args)


def _status(args: argparse.Namespace) -> int:
    from steproute.replay.journal import Journal
    from steproute.replay.ledger import Ledger

    def go(a: argparse.Namespace) -> int:
        j = Journal.open(a.out / "runs.jsonl")
        led = Ledger(a.out / "ledger.jsonl")
        counts: dict[str, int] = {}
        for att in j.attempts.values():
            k = att.status or "in flight"
            counts[k] = counts.get(k, 0) + 1
        print(f"attempts: {len(j.attempts)}; cells finished: {sum(1 for _ in j.finished())}")
        for k, v in sorted(counts.items()):
            print(f"  {k:<18} {v}")
        for (prov, month), usd in sorted(led.by_month().items()):
            print(f"spend {prov:<10} {month}  ${usd:,.4f}")
        if led.open_intents():
            print(f"{len(led.open_intents())} unsettled call(s) charged at worst case")
        return 0

    return _guard(go, args)


def _report(args: argparse.Namespace) -> int:
    from steproute.replay.report import pilot_report

    def go(a: argparse.Namespace) -> int:
        rec = _recorded(argparse.Namespace(**{**vars(a), "plan": None}))
        config, catalog, manifest = _load(rec)
        assumptions = rec.assumptions
        pilot = build_named_plan("pilot", manifest, catalog, load_assumptions(assumptions))
        sparse = build_named_plan("sparse", manifest, catalog, load_assumptions(assumptions))
        md = pilot_report(
            a.out, catalog=catalog, manifest=manifest, pilot=pilot, sparse=sparse, caps=config.caps
        )
        a.md.parent.mkdir(parents=True, exist_ok=True)
        a.md.write_text(md, "utf-8")
        print(f"wrote {a.md}")
        return 0

    return _guard(go, args)
