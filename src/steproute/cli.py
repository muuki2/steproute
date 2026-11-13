"""Command-line entry point. Stdlib only; later issues add subcommands."""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from collections.abc import Sequence
from pathlib import Path

from steproute import __version__
from steproute.core.export import export_replay
from steproute.core.privacy import ScrubOptions
from steproute.core.store import StoreStats, TraceStore, compute_stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="steproute", description="Outcome-trained LLM step router."
    )
    parser.add_argument("--version", action="version", version=f"steproute {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    _add_traces(sub)
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return int(args.func(args))
