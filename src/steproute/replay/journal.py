"""The run journal and the output-directory lock. Standard library only.

``runs.jsonl`` is append-only and ``fsync``-ed like the ledger. A run writes a ``start`` line
before its first call and an ``end`` line when it is over. Resume is driven only by this
file: a cell (task, model, effort, repeat) is **finished** when its latest attempt has an end
line with a terminal status. An attempt with a start and no end was in flight when the process
died; ``Journal.close_interrupted`` closes it as ``interrupted`` and the cell is run again
(``docs/REPLAY.md``, "Resume").
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from steproute.replay.ledger import append_durable, read_lines

try:  # pragma: no cover - platform dependent
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

__all__ = [
    "DEFERRED",
    "LABELLED",
    "RETRYABLE",
    "STATUSES",
    "Attempt",
    "Journal",
    "LockHeld",
    "OutputLock",
]

#: Run statuses. Labelled outcomes (the model did it): ``passed`` (``success = true``) and
#: ``failed``, ``run_cap``, ``step_limit``, ``context_limit``, ``wall_time`` (``false``).
#: Unlabelled (``success = null``, never a model failure): ``provider_error``, ``infra_error``,
#: ``interrupted`` (all retried, with a bound), ``blocked_sensitive`` and ``config_mismatch``
#: (never retried: the same thing would happen again).
#: Deferred (``success = null``, the cell stays pending, not retried in this invocation and
#: not counted against ``max_retries``; a later resume runs it again): ``budget_stop`` (the run
#: reached a reduced budget given from the provider's last headroom) and ``balance_stop`` (the
#: provider said the account is out of funds).
LABELLED = ("passed", "failed", "run_cap", "step_limit", "context_limit", "wall_time")
RETRYABLE = ("provider_error", "infra_error", "interrupted")
DEFERRED = ("budget_stop", "balance_stop")
STATUSES = (*LABELLED, *RETRYABLE, *DEFERRED, "blocked_sensitive", "config_mismatch")

CellKey = tuple[str, str, str, int]  # task_id, model, effort, repeat


class LockHeld(RuntimeError):
    """Another runner holds this output directory."""


class OutputLock:
    """An exclusive, non-blocking ``flock`` on ``<out>/.lock``. The kernel drops it when the
    process dies, so a crash never leaves the directory locked."""

    def __init__(self, out: Path) -> None:
        self.path = out / ".lock"
        self._fd: int | None = None

    def acquire(self) -> OutputLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        if fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                holder = os.read(fd, 64).decode(errors="replace").strip() or "unknown"
                os.close(fd)
                raise LockHeld(
                    f"another replay runner (pid {holder}) is using {self.path.parent}; "
                    "two runners on one directory could both spend. Stop the other one first."
                ) from None
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        os.fsync(fd)
        self._fd = fd
        return self

    def release(self) -> None:
        if self._fd is not None:
            if fcntl is not None:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> OutputLock:
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.release()


@dataclass
class Attempt:
    run_id: str
    cell: CellKey
    attempt: int
    session_id: str
    started: float
    start: dict[str, Any]
    end: dict[str, Any] | None = None

    @property
    def status(self) -> str | None:
        return None if self.end is None else str(self.end.get("status"))

    @property
    def terminal(self) -> bool:
        return self.status is not None and self.status not in (*RETRYABLE, *DEFERRED)


@dataclass
class Journal:
    path: Path
    attempts: dict[str, Attempt] = field(default_factory=dict)
    skipped_lines: int = 0

    @classmethod
    def open(cls, path: Path) -> Journal:
        j = cls(path)
        j.reload()
        return j

    def reload(self) -> None:
        rows, bad = read_lines(self.path)
        attempts: dict[str, Attempt] = {}
        for r in rows:
            try:
                if r.get("kind") == "start":
                    c = r["cell"]
                    key: CellKey = (
                        str(c["task_id"]),
                        str(c["model"]),
                        str(c["effort"]),
                        int(c.get("repeat", 0)),
                    )
                    a = Attempt(
                        str(r["run_id"]),
                        key,
                        int(r["attempt"]),
                        str(r["session_id"]),
                        float(r["ts"]),
                        dict(r),
                    )
                    attempts[a.run_id] = a
                elif r.get("kind") == "end":
                    a2 = attempts.get(str(r["run_id"]))
                    if a2 is None:
                        bad += 1
                    elif a2.end is None:
                        a2.end = dict(r)
                else:
                    bad += 1
            except (KeyError, TypeError, ValueError):
                bad += 1
        self.attempts = attempts
        self.skipped_lines = bad

    def record_start(self, record: Mapping[str, Any]) -> Attempt:
        rec = {"kind": "start", "ts": time.time(), **record}
        append_durable(self.path, rec)
        c = rec["cell"]
        a = Attempt(
            str(rec["run_id"]),
            (str(c["task_id"]), str(c["model"]), str(c["effort"]), int(c.get("repeat", 0))),
            int(rec["attempt"]),
            str(rec["session_id"]),
            float(rec["ts"]),
            dict(rec),
        )
        self.attempts[a.run_id] = a
        return a

    def record_end(self, run_id: str, record: Mapping[str, Any]) -> None:
        a = self.attempts[run_id]
        if a.end is not None:
            raise RuntimeError(f"run {run_id} already ended")
        rec = {"kind": "end", "ts": time.time(), "run_id": run_id, **record}
        append_durable(self.path, rec)
        a.end = dict(rec)

    def by_cell(self) -> dict[CellKey, list[Attempt]]:
        out: dict[CellKey, list[Attempt]] = {}
        for a in self.attempts.values():
            out.setdefault(a.cell, []).append(a)
        for v in out.values():
            v.sort(key=lambda a: a.attempt)
        return out

    def in_flight(self) -> list[Attempt]:
        return [a for a in self.attempts.values() if a.end is None]

    def close_interrupted(self, spent: Mapping[str, float]) -> list[Attempt]:
        """Close every attempt that has no end line as ``interrupted``. Its spend is already in
        the ledger (``spent`` maps run id to it); it is recorded here for the report."""
        closed = []
        for a in self.in_flight():
            self.record_end(
                a.run_id,
                {
                    "status": "interrupted",
                    "success": None,
                    "cost_usd": round(spent.get(a.run_id, 0.0), 6),
                    "reason": "the runner stopped while this run was in flight; it is re-run",
                },
            )
            closed.append(a)
        return closed

    def finished(self) -> Iterator[Attempt]:
        """The latest attempt of every finished cell."""
        for attempts in self.by_cell().values():
            last = attempts[-1]
            if last.terminal:
                yield last
