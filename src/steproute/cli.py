"""Command-line entry point. Stdlib only; later issues add subcommands."""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from collections.abc import Sequence
from pathlib import Path

from steproute import __version__
from steproute.core.catalog import Catalog, CatalogError, load_catalog, load_default_catalog
from steproute.core.export import export_replay
from steproute.core.privacy import ScrubOptions
from steproute.core.schema import ModelProfile
from steproute.core.store import StoreStats, TraceStore, compute_stats
from steproute.core.toolcheck import DEFAULT_BASE_URL, Transport, check_model, urllib_transport
from steproute.tasks.cli import add_tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="steproute", description="Outcome-trained LLM step router."
    )
    parser.add_argument("--version", action="version", version=f"steproute {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    _add_traces(sub)
    _add_catalog(sub)
    _add_serve(sub)
    _add_proxy(sub)
    _add_mcp(sub)
    add_tasks(sub)
    return parser


def _add_traces(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """The ``traces`` command group; later issues add ``export`` etc. next to ``stats``."""
    traces = sub.add_parser("traces", help="Inspect trace stores.")
    tsub = traces.add_subparsers(dest="traces_command", metavar="<subcommand>", required=True)
    stats = tsub.add_parser("stats", help="Summarise a trace store directory.")
    stats.add_argument("path", type=Path, help="trace store directory")
    stats.set_defaults(func=_traces_stats)
    export = tsub.add_parser(
        "export", help="Export replay traces (personal traces are never exported)."
    )
    export.add_argument("path", type=Path, help="trace store directory")
    export.add_argument("--out", type=Path, required=True, help="output JSONL file")
    export.add_argument(
        "--scrub",
        action="store_true",
        help="drop text, hash ids, bucket timestamps, redact secret patterns",
    )
    export.add_argument(
        "--salt",
        help="salt for --scrub (default: $STEPROUTE_SCRUB_SALT, else random per export)",
    )
    export.set_defaults(func=_traces_export)


def _add_catalog(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """The ``catalog`` command group."""
    cat = sub.add_parser("catalog", help="Inspect and smoke-test the model catalog.")
    csub = cat.add_subparsers(dest="catalog_command", metavar="<subcommand>", required=True)
    show = csub.add_parser("show", help="List the models in a catalog.")
    check = csub.add_parser(
        "check", help="Tool-call smoke test against an OpenAI-compatible endpoint."
    )
    for p in (show, check):
        p.add_argument("--catalog", type=Path, help="catalog TOML (default: the built-in one)")
    show.set_defaults(func=_catalog_show)
    check.add_argument("--base-url", default=DEFAULT_BASE_URL, help="default: local Ollama")
    check.add_argument(
        "--model", action="append", metavar="ID", help="model id to check (repeatable)"
    )
    check.add_argument("--timeout", type=float, default=120.0, help="seconds per request")
    check.set_defaults(func=_catalog_check)


def _add_serve(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """The ``serve`` command: the local HTTP routing service."""
    serve = sub.add_parser("serve", help="Run the local routing service (HTTP API).")
    serve.add_argument("--config", type=Path, help="service config TOML (default: all defaults)")
    serve.add_argument("--host", help="bind address (default 127.0.0.1)")
    serve.add_argument("--port", type=int, help="port; 0 picks a free one and prints it")
    serve.add_argument("--store", type=Path, help="trace store directory")
    serve.set_defaults(func=_serve)


def _add_proxy(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """The ``proxy`` command: the OpenAI-compatible routing proxy (needs the [proxy] extra)."""
    proxy = sub.add_parser(
        "proxy", help="Run the OpenAI-compatible routing proxy (needs steproute[proxy])."
    )
    proxy.add_argument("--config", type=Path, required=True, help="proxy config TOML")
    proxy.add_argument("--host", help="bind address (default 127.0.0.1)")
    proxy.add_argument("--port", type=int, help="port; 0 picks a free one and prints it")
    proxy.add_argument("--mode", choices=("route", "shadow"), help="override [proxy] mode")
    proxy.set_defaults(func=_proxy)


def _proxy(args: argparse.Namespace) -> int:
    import dataclasses
    import logging

    from steproute.collectors.proxy.config import ProxyConfigError, load_proxy_config

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        config = load_proxy_config(args.config)
        overrides = {
            k: v
            for k, v in (("host", args.host), ("port", args.port), ("mode", args.mode))
            if v is not None
        }
        config = dataclasses.replace(config, **overrides)
        from steproute.collectors.proxy.server import run_proxy

        return run_proxy(config)
    except ImportError:
        print(
            "steproute: the proxy needs the [proxy] extra: pip install 'steproute[proxy]'",
            file=sys.stderr,
        )
        return 2
    except (ProxyConfigError, OSError) as exc:
        print(f"steproute: {exc}", file=sys.stderr)
        return 2


def _serve(args: argparse.Namespace) -> int:
    import dataclasses
    import logging

    from steproute.core.service import ServiceConfigError, load_service_config, serve

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        config = load_service_config(args.config)
        overrides = {
            k: v
            for k, v in (("host", args.host), ("port", args.port), ("store_path", args.store))
            if v is not None
        }
        config = dataclasses.replace(config, **overrides)
        return serve(config)
    except (ServiceConfigError, OSError) as exc:
        print(f"steproute: {exc}", file=sys.stderr)
        return 2


def _add_mcp(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """The ``mcp`` command: the stdio MCP server (needs the ``mcp`` extra)."""
    mcp = sub.add_parser(
        "mcp", help="Run the MCP server over stdio (route_task, report_outcome, list_models)."
    )
    mcp.add_argument("--config", type=Path, help="service config TOML, as for `serve`")
    mcp.add_argument("--store", type=Path, help="trace store directory")
    mcp.add_argument("--harness", default="mcp", help="harness name logged (default: mcp)")
    mcp.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        metavar="MINUTES",
        help="close decisions left unreported this long with success=None (default 30)",
    )
    for word, default in (("small", 2000), ("medium", 8000), ("large", 32000)):
        mcp.add_argument(
            f"--{word}-tokens",
            type=int,
            default=default,
            metavar="N",
            help=f"prompt tokens assumed for est_size={word!r} (placeholder; default {default})",
        )
    mcp.set_defaults(func=_mcp)


def _mcp(args: argparse.Namespace) -> int:
    import dataclasses
    import importlib.util
    import logging

    if importlib.util.find_spec("mcp") is None:
        print(
            "steproute: the MCP server needs the 'mcp' extra: pip install 'steproute[mcp]'",
            file=sys.stderr,
        )
        return 2
    from steproute.collectors.mcp_server import Advisor, McpConfig, McpConfigError, run_stdio
    from steproute.core.service import ServiceConfigError, build_service, load_service_config

    # stdout carries the protocol; every log line goes to stderr.
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config = load_service_config(args.config)
        if args.store is not None:
            config = dataclasses.replace(config, store_path=args.store)
        service, _token = build_service(config)
        mcp_config = McpConfig(
            harness=args.harness,
            timeout_s=args.timeout * 60.0,
            size_tokens={
                "small": args.small_tokens,
                "medium": args.medium_tokens,
                "large": args.large_tokens,
            },
        )
    except (ServiceConfigError, McpConfigError, OSError) as exc:
        print(f"steproute: {exc}", file=sys.stderr)
        return 2
    run_stdio(Advisor(service, mcp_config))
    return 0


def _load(path: Path | None) -> Catalog | None:
    try:
        return load_catalog(path) if path else load_default_catalog()
    except (CatalogError, OSError) as exc:
        print(f"steproute: {exc}", file=sys.stderr)
        return None


def format_catalog(models: Sequence[ModelProfile]) -> str:
    rows = [("id", "provider", "tier", "efforts", "flags")]
    for m in models:
        flags = [
            f
            for f, on in (
                ("local", m.local),
                ("holdout", m.holdout),
                ("no-tools", not m.tools),
                ("verified" if m.verified else "unverified", True),
            )
            if on
        ]
        rows.append((m.id, m.provider, str(m.tier), ",".join(m.efforts), ",".join(flags)))
    widths = [max(len(r[i]) for r in rows) for i in range(5)]
    return "\n".join(
        "  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in rows
    )


def _catalog_show(args: argparse.Namespace) -> int:
    catalog = _load(args.catalog)
    if catalog is None:
        return 2
    print(format_catalog(list(catalog)))
    return 0


def _catalog_check(args: argparse.Namespace) -> int:
    catalog = _load(args.catalog)
    if catalog is None:
        return 2
    transport: Transport = getattr(args, "transport", None) or urllib_transport
    if args.model:
        try:
            models = [catalog.get(i) for i in args.model]
        except KeyError as exc:
            print(f"steproute: model {exc.args[0]!r} is not in the catalog", file=sys.stderr)
            return 2
        cloud = [m.id for m in models if not m.local]
        if cloud:
            print(
                f"steproute: {', '.join(cloud)} is not local; cloud models are checked "
                "through the proxy, not by this command",
                file=sys.stderr,
            )
            return 2
    else:
        models = list(catalog.local_models())
    failed = 0
    for m in models:
        r = check_model(m.id, base_url=args.base_url, transport=transport, timeout=args.timeout)
        print(f"{'PASS' if r.ok else 'FAIL'}  {m.id}: {r.detail}")
        if r.unreachable:
            print(
                f"steproute: cannot reach {args.base_url}; is Ollama running? "
                "(stopping; nothing was concluded about the remaining models)",
                file=sys.stderr,
            )
            return 3
        if not r.ok:
            failed += 1
            print(
                f'      set tools = false for "{m.id}" in the catalog '
                "to exclude it from agent tasks"
            )
    return 1 if failed else 0


def format_stats(s: StoreStats) -> str:
    lines = [
        f"decisions: {s.decisions}",
        f"outcomes: {s.outcomes}",
        f"joined: {s.joined}",
        f"join rate: {s.join_rate:.1%}",
        f"skipped lines: {s.skipped_lines}",
        f"total cost: ${s.total_cost_usd:.4f}",
        f"outcomes without cost: {s.outcomes_without_cost}",
        f"superseded outcomes: {s.superseded_outcomes}",
        "success rate by first action (model, effort):",
    ]
    if not s.success:
        lines.append("  (none)")
    for (model, effort), (ok, n) in s.success.items():
        lines.append(f"  {model} {effort}: {ok / n:.1%} (n={n})")
    return "\n".join(lines)


def _traces_stats(args: argparse.Namespace) -> int:
    path: Path = args.path
    if path.exists() and not path.is_dir():
        print(f"steproute: not a directory: {path}", file=sys.stderr)
        return 2
    print(format_stats(compute_stats(TraceStore(path))))
    return 0


def _traces_export(args: argparse.Namespace) -> int:
    path: Path = args.path
    if not path.is_dir():
        print(f"steproute: not a directory: {path}", file=sys.stderr)
        return 2
    scrub = None
    if args.scrub:
        salt = args.salt or os.environ.get("STEPROUTE_SCRUB_SALT") or secrets.token_hex(16)
        scrub = ScrubOptions(salt=salt)
    elif args.salt:
        print("steproute: --salt has no effect without --scrub", file=sys.stderr)
    summary = export_replay(TraceStore(path), args.out, scrub=scrub)
    print(
        f"exported {summary.exported} records "
        f"({summary.decisions} decisions, {summary.outcomes} outcomes); "
        f"withheld {summary.withheld} (personal or unknown source)"
    )
    return 0


def main(argv: Sequence[str] | None = None, *, transport: Transport | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.transport = transport
    if args.command is None:
        parser.print_help()
        return 0
    return int(args.func(args))
