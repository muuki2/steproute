"""Command-line entry point. Stdlib only; later issues add subcommands."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from steproute import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="steproute", description="Outcome-trained LLM step router."
    )
    parser.add_argument("--version", action="version", version=f"steproute {__version__}")
    parser.add_subparsers(dest="command", metavar="<command>")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
    return 0
