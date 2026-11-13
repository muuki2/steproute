"""The spend ledger: the single source of truth for the monthly caps. Standard library only.

An append-only JSON Lines file. Every paid call is written twice:

1. an ``intent`` line with the call's worst-case cost, written and ``fsync``-ed **before** the
   call is made;
2. a ``settle`` line with the realised cost (the provider's usage priced with ``price_usage``),
   or the worst case again when the usage was missing, after the call.

Month-to-date spend is the settled amount of every settled intent plus the worst case of
every intent that was never settled. So a crash at any point counts against the cap at least
what was really spent:

* killed after the intent and before the call, or during the call: the worst case is charged;
* a torn intent line (killed while writing it): the call was never made (calls start only
  after the intent is on disk), and the torn line is skipped;
* a torn settle line: the intent stays open and its worst case is charged.

The month is the **UTC** calendar month of the intent's timestamp (``month_of``), so a run
does not change months with the local time zone or daylight saving, and a call that started
before midnight UTC on the last day counts in the month it started.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:  # pragma: no cover - platform dependent
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

__all__ = ["CapExceeded", "CapGuard", "Intent", "Ledger", "append_durable", "month_of"]

LEDGER_VERSION = 1


def month_of(ts: float) -> str:
    """``YYYY-MM`` of a Unix timestamp, in UTC."""
    return f"{datetime.fromtimestamp(ts, UTC):%Y-%m}"


def _torn_tail(path: Path) -> bool:
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() == 0:
            return False
        f.seek(-1, os.SEEK_END)
        return f.read(1) != b"\n"


def append_durable(path: Path, record: Mapping[str, Any]) -> None:
    """Append one JSON line and ``fsync`` it. A torn last line (a crash mid-write) is closed
    with a newline first, so it stays one unreadable line and never merges with this one."""
    line = json.dumps(record, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            if _torn_tail(path):
                line = b"\n" + line
            view = memoryview(line)
            while view:
                n = os.write(fd, view)
                view = view[n:]
            os.fsync(fd)
        finally:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def read_lines(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Every readable JSON object line, and the number of unreadable lines skipped."""
    out: list[dict[str, Any]] = []
    bad = 0
    if not path.exists():
        return out, 0
    with open(path, "rb") as f:
        for raw in f:
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except ValueError:
                bad += 1
                continue
            if isinstance(obj, dict):
                out.append(obj)
            else:
                bad += 1
    return out, bad


@dataclass
class Intent:
    id: str
    ts: float
    month: str
    provider: str
    worst_usd: float
    run_id: str
    settled_usd: float | None = None
    flags: list[str] = field(default_factory=list)

    @property
    def charged_usd(self) -> float:
        return self.worst_usd if self.settled_usd is None else self.settled_usd


class Ledger:
    """Month-to-date spend per provider. One writer: the runner holding the output lock."""

    def __init__(self, path: str | os.PathLike[str], *, clock: Callable[[], float] = time.time):
        self.path = Path(path)
        self._clock = clock
        self._lock = threading.Lock()
        self.intents: dict[str, Intent] = {}
        self.skipped_lines = 0
        self.reload()

    def reload(self) -> None:
        rows, bad = read_lines(self.path)
        intents: dict[str, Intent] = {}
        for r in rows:
            try:
                if r.get("kind") == "intent":
                    i = Intent(
                        id=str(r["id"]),
                        ts=float(r["ts"]),
                        month=str(r["month"]),
                        provider=str(r["provider"]),
                        worst_usd=float(r["usd"]),
                        run_id=str(r.get("run_id", "")),
                    )
                    intents[i.id] = i
                elif r.get("kind") == "settle":
                    i2 = intents.get(str(r["id"]))
                    if i2 is not None:
                        i2.settled_usd = float(r["usd"])
                        i2.flags = [str(f) for f in r.get("flags", [])]
                else:
                    bad += 1
            except (KeyError, TypeError, ValueError):
                bad += 1
        with self._lock:
            self.intents = intents
            self.skipped_lines = bad

    # --- writing ---

    def intend(self, provider: str, worst_usd: float, run_id: str) -> Intent:
        """Record, durably, that a call costing at most ``worst_usd`` is about to be made."""
        ts = self._clock()
        i = Intent(uuid.uuid4().hex, ts, month_of(ts), provider, float(worst_usd), run_id)
        append_durable(
            self.path,
            {
                "v": LEDGER_VERSION,
                "kind": "intent",
                "id": i.id,
                "ts": ts,
                "month": i.month,
                "provider": provider,
                "usd": i.worst_usd,
                "run_id": run_id,
            },
        )
        with self._lock:
            self.intents[i.id] = i
        return i

    def settle(self, intent: Intent, usd: float, flags: Iterable[str] = ()) -> None:
        """Record the realised cost of an intended call."""
        fl = sorted(set(flags))
        append_durable(
            self.path,
            {
                "v": LEDGER_VERSION,
                "kind": "settle",
                "id": intent.id,
                "ts": self._clock(),
                "usd": float(usd),
                "flags": fl,
            },
        )
        with self._lock:
            intent.settled_usd = float(usd)
            intent.flags = fl

    # --- reading ---

    def spent(self, provider: str, month: str | None = None) -> float:
        """Month-to-date spend (the current UTC month unless ``month`` is given)."""
        m = month or month_of(self._clock())
        with self._lock:
            return sum(
                i.charged_usd
                for i in self.intents.values()
                if i.provider == provider and i.month == m
            )

    def by_month(self) -> dict[tuple[str, str], float]:
        out: dict[tuple[str, str], float] = {}
        with self._lock:
            for i in self.intents.values():
                key = (i.provider, i.month)
                out[key] = out.get(key, 0.0) + i.charged_usd
        return out

    def run_spent(self, run_id: str) -> float:
        with self._lock:
            return sum(i.charged_usd for i in self.intents.values() if i.run_id == run_id)

    def open_intents(self) -> list[Intent]:
        """Intents never settled (a crash or kill during the call): charged at worst case."""
        with self._lock:
            return [i for i in self.intents.values() if i.settled_usd is None]

    def flagged(self) -> list[Intent]:
        with self._lock:
            return [i for i in self.intents.values() if i.flags]


class CapExceeded(Exception):
    """Starting the run could pass a monthly cap."""


class CapGuard:
    """Admits a run only if month-to-date spend, plus what every run in flight could still
    spend, plus this run's worst-case bound fits under the provider's monthly cap.

    The check runs under one lock, so parallel runs can never jointly pass a cap: each
    admitted run holds a reservation of ``bound - spent so far`` until it ends.
    """

    def __init__(self, ledger: Ledger, caps: Mapping[str, float]) -> None:
        self.ledger = ledger
        self.caps = dict(caps)
        self._lock = threading.Lock()
        self._held: dict[str, tuple[str, float]] = {}  # run_id -> (provider, bound)

    def _outstanding(self, provider: str) -> float:
        total = 0.0
        for run_id, (prov, bound) in self._held.items():
            if prov == provider:
                total += max(0.0, bound - self.ledger.run_spent(run_id))
        return total

    def headroom(self, provider: str) -> float:
        with self._lock:
            cap = self.caps[provider]
            return cap - self.ledger.spent(provider) - self._outstanding(provider)

    def admit(self, provider: str, run_id: str, bound: float) -> None:
        """Reserve ``bound`` for ``run_id`` or raise ``CapExceeded``."""
        with self._lock:
            cap = self.caps.get(provider)
            if cap is None:
                raise CapExceeded(f"no monthly cap is configured for paid provider {provider!r}")
            used = self.ledger.spent(provider)
            held = self._outstanding(provider)
            if used + held + bound > cap:
                raise CapExceeded(
                    f"{provider}: month-to-date ${used:.4f} + in flight ${held:.4f} + this run's "
                    f"bound ${bound:.4f} would pass the ${cap:g} monthly cap"
                )
            self._held[run_id] = (provider, bound)

    def release(self, run_id: str) -> None:
        with self._lock:
            self._held.pop(run_id, None)
