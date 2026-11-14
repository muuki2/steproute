"""``steproute dataset build`` and ``steproute dataset check``. See ``docs/DATASET.md``.

``steproute.cli`` registers the arguments from here and calls ``run`` when the command runs;
the Parquet library is imported only then.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from steproute.core.catalog import CatalogError, load_catalog, load_default_catalog
from steproute.dataset.splits import DEFAULT_FRACTIONS, DEFAULT_SEED
from steproute.tasks.manifest import ManifestError, load_default_manifest, load_manifest

__all__ = ["add_arguments", "run"]


def add_arguments(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="dataset_command", metavar="<subcommand>", required=True)
    b = sub.add_parser(
        "build",
        help="Build a release directory from replay output directories.",
        description="Read replay output directories and write a release: scrubbed traces as "
        "Parquet, the JSON Schema, fixed splits by task, a datasheet and a manifest with "
        "checksums. Refuses (and writes nothing) on any record that is not a replay record, "
        "a run directory with an unknown plan or schema version, or a record that fails the "
        "privacy scrubber. Nothing is uploaded.",
    )
    b.add_argument(
        "--from",
        dest="sources",
        action="append",
        type=Path,
        required=True,
        metavar="RUN_DIR",
        help="a replay output directory (repeatable)",
    )
    b.add_argument("--out", type=Path, required=True, help="release directory (must not exist)")
    b.add_argument("--manifest", type=Path, help="task manifest (default: the built-in set)")
    b.add_argument("--catalog", type=Path, help="catalog TOML (default: the built-in one)")
    b.add_argument("--seed", type=int, default=DEFAULT_SEED, help="split seed")
    b.add_argument(
        "--fractions",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VALIDATION", "TEST"),
        default=[
            DEFAULT_FRACTIONS["train"],
            DEFAULT_FRACTIONS["validation"],
            DEFAULT_FRACTIONS["test"],
        ],
        help="share of tasks per split (normalised)",
    )
    b.add_argument("--salt", help="salt for hashed ids (default: derived from the sources)")
    b.add_argument("--licence", default="MIT", help="licence named in the datasheet")
    b.add_argument(
        "--allow-mock",
        action="store_true",
        help="accept run directories made against the mock upstream (tests and fixtures only)",
    )
    b.set_defaults(func=run)
    c = sub.add_parser(
        "check",
        help="Verify a release: checksums, splits, and the datasheet's summary tables.",
    )
    c.add_argument("release", type=Path, help="release directory")
    c.set_defaults(func=run)


def _err(msg: str) -> int:
    print(f"steproute: {msg}", file=sys.stderr)
    return 2


def run(args: argparse.Namespace) -> int:
    from steproute.dataset import (
        MissingParquetSupport,
        ReleaseError,
        ReleaseRefused,
        build_release,
        load,
    )

    try:
        if args.dataset_command == "check":
            rel = load(args.release)
            if rel.summary_block() != rel.datasheet_block():
                return _err(f"{args.release}: the datasheet's summary tables differ from the data")
            print(
                f"ok: {len(rel.runs)} runs, {len(rel.steps)} steps; checksums, splits and "
                "summary tables agree"
            )
            return 0
        try:
            manifest = load_manifest(args.manifest) if args.manifest else load_default_manifest()
            catalog = (
                load_catalog(args.catalog, warn_unverified=False)
                if args.catalog
                else load_default_catalog(warn_unverified=False)
            )
        except (ManifestError, CatalogError, OSError) as exc:
            return _err(str(exc))
        train, val, test = args.fractions
        summary = build_release(
            args.sources,
            args.out,
            manifest=manifest,
            catalog=catalog,
            seed=args.seed,
            fractions={"train": train, "validation": val, "test": test},
            salt=args.salt,
            licence=args.licence,
            allow_mock=args.allow_mock,
        )
    except MissingParquetSupport as exc:
        return _err(str(exc))
    except (ReleaseRefused, ReleaseError) as exc:
        return _err(f"refused: {exc}\nnothing was written")
    print(
        f"wrote {summary.out}: {summary.runs} runs, {summary.steps} steps, {summary.files} files. "
        "Nothing was uploaded; publishing is a manual step (docs/DATASET.md)."
    )
    return 0
