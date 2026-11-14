"""Read a release back. Needs the ``dataset`` extra (pyarrow) for the Parquet files."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from steproute.dataset import tables
from steproute.dataset.source import sha256_file
from steproute.dataset.splits import HOLDOUT_SPLIT, TASK_SPLITS, all_splits
from steproute.dataset.summary import extract_block, render_block

__all__ = ["Release", "ReleaseError", "SplitData", "load"]

Row = dict[str, Any]


class ReleaseError(ValueError):
    """The directory is not an intact release."""


@dataclass
class SplitData:
    """One split: its runs, its steps (decision joined to outcome) and its task ids."""

    name: str
    runs: list[Row]
    steps: list[Row]

    @property
    def tasks(self) -> set[str]:
        return {r["task_id"] for r in self.runs}


@dataclass
class Release:
    path: Path
    manifest: dict[str, Any]
    splits: dict[str, SplitData]
    #: every task of the task set: ``{"stratum", "split"}``
    tasks: dict[str, dict[str, Any]]
    datasheet: str = field(repr=False, default="")

    @property
    def runs(self) -> list[Row]:
        return [r for s in self.splits.values() for r in s.runs]

    @property
    def steps(self) -> list[Row]:
        return [r for s in self.splits.values() for r in s.steps]

    def summary_block(self) -> str:
        """The datasheet's summary tables, recomputed from the data in this release."""
        return render_block(self.runs, self.steps, self.tasks)

    def datasheet_block(self) -> str:
        """The summary tables as written in ``DATASHEET.md``."""
        return extract_block(self.datasheet)


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, ValueError) as exc:
        raise ReleaseError(f"{path}: unreadable ({exc})") from None


def load(path: str | Path, *, verify: bool = True) -> Release:
    """Load the release in ``path``. With ``verify`` (the default) every file is checked
    against the checksums in ``manifest.json`` and no unlisted file may be present.

    Raises ``ReleaseError`` on a damaged release, ``MissingParquetSupport`` without pyarrow.
    """
    root = Path(path)
    manifest = _json(root / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("format") != "steproute-dataset":
        raise ReleaseError(f"{root}: not a steproute dataset release")
    version = str(manifest.get("format_version"))
    if not version.startswith("0."):
        raise ReleaseError(f"{root}: unknown release format version {version!r}")
    files: Mapping[str, Mapping[str, Any]] = manifest.get("files", {})
    if verify:
        for rel, info in files.items():
            p = root / rel
            if not p.is_file():
                raise ReleaseError(f"{p}: listed in manifest.json but missing")
            if sha256_file(p) != info["sha256"]:
                raise ReleaseError(f"{p}: checksum differs from manifest.json")
        present = {
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.is_file() and p.name != "manifest.json"
        }
        extra = sorted(present - set(files))
        if extra:
            raise ReleaseError(f"{root}: files not listed in manifest.json: {', '.join(extra)}")

    splits_doc = _json(root / "splits.json")
    tasks: dict[str, dict[str, Any]] = splits_doc["tasks"]
    runs = tables.read_rows(root / "data" / "runs.parquet", "runs")
    splits: dict[str, SplitData] = {}
    for name in all_splits():
        steps = tables.read_rows(root / "data" / f"steps-{name}.parquet", "steps")
        for s in steps:
            if s["source"] != "replay":
                raise ReleaseError(f"{root}: step {s['decision_id']} is not a replay record")
            if s["split"] != name:
                raise ReleaseError(f"{root}: step {s['decision_id']} is in the wrong file")
        splits[name] = SplitData(name, [r for r in runs if r["split"] == name], steps)

    owner: dict[str, str] = {}
    for name in TASK_SPLITS:
        for tid in splits[name].tasks:
            if owner.setdefault(tid, name) != name:
                raise ReleaseError(f"{root}: task {tid} is in both {owner[tid]} and {name}")
    for name in TASK_SPLITS:
        for tid in splits[name].tasks:
            if tasks[tid]["split"] != name:
                raise ReleaseError(
                    f"{root}: task {tid} is in {name} but splits.json says otherwise"
                )
    for r in splits[HOLDOUT_SPLIT].runs:
        if tasks[r["task_id"]]["split"] != r["task_split"]:
            raise ReleaseError(f"{root}: run {r['run_id']} has a task_split that differs")
    datasheet = (root / "DATASHEET.md").read_text("utf-8")
    return Release(root, manifest, splits, tasks, datasheet)
