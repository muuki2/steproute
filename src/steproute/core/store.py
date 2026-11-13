"""Append-only JSONL trace store. Standard library only.

Layout: ``path`` is a directory holding one file per UTC day of the write time,
``traces-YYYY-MM-DD.jsonl``. Each line is ``{"type": ..., "v": SCHEMA_VERSION, "data": ...}``.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from steproute.core.schema import SCHEMA_VERSION, Decision, Outcome

try:  # pragma: no cover - platform dependent
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

__all__ = ["CorruptLineWarning", "StoreStats", "TraceStore", "compute_stats"]

Record = Decision | Outcome


class CorruptLineWarning(UserWarning):
    """A trace line was unreadable or not a valid record; it was skipped."""


def _day_name(ts: float) -> str:
    return f"traces-{datetime.fromtimestamp(ts, UTC):%Y-%m-%d}.jsonl"


class TraceStore:
    """Append-only store of decisions and outcomes under a directory."""

    def __init__(
        self, path: str | os.PathLike[str], clock: Callable[[], float] = time.time
    ) -> None:
        self.path = Path(path)
        self._clock = clock
        #: Number of lines skipped by the most recent completed read.
        self.skipped_lines = 0

    # --- writing ---

    def log_decision(self, decision: Decision) -> None:
        self._append("decision", decision.to_dict())

    def log_outcome(self, outcome: Outcome) -> None:
        self._append("outcome", outcome.to_dict())

    def _append(self, kind: str, data: dict[str, Any]) -> None:
        text = json.dumps({"type": kind, "v": SCHEMA_VERSION, "data": data}, separators=(",", ":"))
        line = text.encode("utf-8") + b"\n"
        self.path.mkdir(parents=True, exist_ok=True)
        target = self.path / _day_name(self._clock())
        fd = os.open(target, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                if self._torn_tail(target):
                    line = b"\n" + line
                view = memoryview(line)
                while view:
                    n = os.write(fd, view)
                    view = view[n:]
            finally:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    @staticmethod
    def _torn_tail(target: Path) -> bool:
        """True if the file is non-empty and does not end with a newline."""
        with open(target, "rb") as f:
            f.seek(0, os.SEEK_END)
            if f.tell() == 0:
                return False
            f.seek(-1, os.SEEK_END)
            return f.read(1) != b"\n"

    # --- reading ---

    def files(self) -> list[Path]:
        if not self.path.is_dir():
            return []
        return sorted(self.path.glob("traces-????-??-??.jsonl"))

    def records(self) -> Iterator[Record]:
        """Yield typed records in file order, skipping and warning about bad lines."""
        skipped = 0
        self.skipped_lines = 0
        for file in self.files():
            with open(file, "rb") as f:
                for lineno, raw in enumerate(f, start=1):
                    if not raw.strip():
                        continue
                    try:
                        rec = self._parse(raw)
                    except Exception as exc:  # any parse failure is a skip
                        skipped += 1
                        self.skipped_lines = skipped
                        warnings.warn(
                            f"{file}:{lineno}: skipped unreadable line ({exc})",
                            CorruptLineWarning,
                            stacklevel=2,
                        )
                        continue
                    yield rec
        self.skipped_lines = skipped

    @staticmethod
    def _parse(raw: bytes) -> Record:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("line is not a JSON object")
        kind = obj.get("type")
        data: Any = obj.get("data")
        if kind == "decision":
            return Decision.from_dict(data)
        if kind == "outcome":
            return Outcome.from_dict(data)
        raise ValueError(f"unknown record type {kind!r}")

    def _collect(self) -> tuple[dict[str, Decision], dict[str, Outcome], list[Outcome]]:
        decisions: dict[str, Decision] = {}
        outcomes: dict[str, Outcome] = {}
        all_outcomes: list[Outcome] = []
        for rec in self.records():
            if isinstance(rec, Decision):
                decisions.setdefault(rec.decision_id, rec)
            else:
                all_outcomes.append(rec)
                outcomes[rec.decision_id] = rec
        return decisions, outcomes, all_outcomes

    def joined(self) -> Iterator[tuple[Decision, Outcome | None]]:
        """Decisions in order, each with its last outcome (or None)."""
        decisions, outcomes, _ = self._collect()
        for did, dec in decisions.items():
            yield dec, outcomes.get(did)


@dataclass(frozen=True)
class StoreStats:
    decisions: int = 0
    outcomes: int = 0
    joined: int = 0
    skipped_lines: int = 0
    total_cost_usd: float = 0.0
    outcomes_without_cost: int = 0
    #: (model, effort) -> (successes, counted outcomes), sorted by key.
    success: dict[tuple[str, str], tuple[int, int]] = field(default_factory=dict)

    @property
    def join_rate(self) -> float:
        return self.joined / self.decisions if self.decisions else 0.0


def compute_stats(store: TraceStore) -> StoreStats:
    """Summarise a store. Cost sums every outcome record read (including superseded ones)."""
    decisions, outcomes, all_outcomes = store._collect()
    cost = 0.0
    no_cost = 0
    for o in all_outcomes:
        if o.cost_usd is None:
            no_cost += 1
        else:
            cost += o.cost_usd
    tally: dict[tuple[str, str], list[int]] = {}
    joined = 0
    for did, dec in decisions.items():
        out = outcomes.get(did)
        if out is None:
            continue
        joined += 1
        if out.success is None:
            continue
        t = tally.setdefault((dec.action.model, dec.action.effort), [0, 0])
        t[0] += int(out.success)
        t[1] += 1
    return StoreStats(
        decisions=len(decisions),
        outcomes=len(all_outcomes),
        joined=joined,
        skipped_lines=store.skipped_lines,
        total_cost_usd=cost,
        outcomes_without_cost=no_cost,
        success={k: (v[0], v[1]) for k, v in sorted(tally.items())},
    )
