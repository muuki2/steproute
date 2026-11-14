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
    #: ``peak``, ``offpeak`` or ``flat`` (``steproute.core.offpeak``): the price the call was
    #: settled at. ``None`` on an open intent and on lines written before issue 57.
    period: str | None = None
    #: The same usage priced at the catalog (peak, list) price. Equal to ``settled_usd``
    #: except for an off-peak call.
    list_usd: float | None = None

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
                        i2.period = str(r["period"]) if r.get("period") else None
                        lu = r.get("list_usd")
                        i2.list_usd = float(lu) if lu is not None else i2.settled_usd
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

    def settle(
        self,
        intent: Intent,
        usd: float,
        flags: Iterable[str] = (),
        *,
        period: str | None = None,
        list_usd: float | None = None,
    ) -> None:
        """Record the realised cost of an intended call: ``usd`` is what is billed (the price
        of the call's ``period``); ``list_usd`` the same usage at the catalog's peak price."""
        fl = sorted(set(flags))
        rec: dict[str, Any] = {
            "v": LEDGER_VERSION,
            "kind": "settle",
            "id": intent.id,
            "ts": self._clock(),
            "usd": float(usd),
            "flags": fl,
        }
        if period is not None:
            rec["period"] = period
        if list_usd is not None:
            rec["list_usd"] = float(list_usd)
        append_durable(self.path, rec)
        with self._lock:
            intent.settled_usd = float(usd)
            intent.flags = fl
            intent.period = period
            intent.list_usd = float(usd) if list_usd is None else float(list_usd)

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

    def spent_total(self, provider: str) -> float:
        """All-time spend of ``provider`` in this ledger, every month."""
        with self._lock:
            return sum(i.charged_usd for i in self.intents.values() if i.provider == provider)

    def by_period(self) -> dict[tuple[str, str], tuple[float, float]]:
        """``(provider, period) -> (billed, at list price)`` over settled calls. Lines written
        before periods were recorded count as ``unknown``."""
        out: dict[tuple[str, str], tuple[float, float]] = {}
        with self._lock:
            for i in self.intents.values():
                if i.settled_usd is None:
                    continue
                key = (i.provider, i.period or "unknown")
                b, lst = out.get(key, (0.0, 0.0))
                out[key] = (b + i.settled_usd, lst + (i.list_usd or i.settled_usd))
        return out

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

    def run_list(self, run_id: str) -> float:
        """A run's spend with every call at the catalog's peak (list) price; an open intent
        counts at its worst case, as in ``run_spent``."""
        with self._lock:
            return sum(
                i.charged_usd if i.list_usd is None else i.list_usd
                for i in self.intents.values()
                if i.run_id == run_id
            )

    def open_intents(self) -> list[Intent]:
        """Intents never settled (a crash or kill during the call): charged at worst case."""
        with self._lock:
            return [i for i in self.intents.values() if i.settled_usd is None]

    def flagged(self) -> list[Intent]:
        with self._lock:
            return [i for i in self.intents.values() if i.flags]


class CapExceeded(Exception):
    """Starting the run could pass a monthly cap or a total budget."""


class CapGuard:
    """Admits a run only if spend so far, plus what every run in flight could still spend,
    plus this run's reservation fits under the provider's monthly cap (month-to-date spend)
    and under its total budget (all-time spend in this ledger), when it has one.

    The check runs under one lock, so parallel runs can never jointly pass a cap: each
    admitted run holds a reservation of ``reservation - spent so far`` until it ends.

    ``admit(..., least=x)``: when the full ``bound`` does not fit but at least ``x`` does, the
    run is admitted with a reservation of exactly the headroom left (a reduced budget). The
    caller must then stop the run early enough that it cannot spend more than that.
    """

    def __init__(
        self,
        ledger: Ledger,
        caps: Mapping[str, float],
        totals: Mapping[str, float] | None = None,
    ) -> None:
        self.ledger = ledger
        self.caps = dict(caps)
        self.totals = dict(totals or {})
        self._lock = threading.Lock()
        self._held: dict[str, tuple[str, float]] = {}  # run_id -> (provider, reservation)

    def _outstanding(self, provider: str) -> float:
        total = 0.0
        for run_id, (prov, bound) in self._held.items():
            if prov == provider:
                total += max(0.0, bound - self.ledger.run_spent(run_id))
        return total

    def _limits(self, provider: str) -> list[tuple[str, float, float]]:
        """``(what, limit, spent)`` for the total budget, if set, and the monthly cap (on a tie
        the total is named: it is the one that does not reset)."""
        out = []
        if provider in self.totals:
            out.append(("all-time", self.totals[provider], self.ledger.spent_total(provider)))
        out.append(("month-to-date", self.caps[provider], self.ledger.spent(provider)))
        return out

    def headroom(self, provider: str) -> float:
        with self._lock:
            held = self._outstanding(provider)
            return min(lim - used for _, lim, used in self._limits(provider)) - held

    def admit(
        self, provider: str, run_id: str, bound: float, *, least: float | None = None
    ) -> float:
        """Reserve ``bound`` for ``run_id`` (or, with ``least``, the headroom left when that is
        at least ``least``) and return the reservation; otherwise raise ``CapExceeded``."""
        with self._lock:
            if self.caps.get(provider) is None:
                raise CapExceeded(f"no monthly cap is configured for paid provider {provider!r}")
            held = self._outstanding(provider)
            limits = self._limits(provider)
            room = min(lim - used for _, lim, used in limits) - held
            grant = bound
            if bound > room:
                if least is not None and room >= least:
                    grant = room
                else:
                    what, lim, used = min(limits, key=lambda x: x[1] - x[2])
                    kind = "monthly cap" if what == "month-to-date" else "total budget"
                    need = bound if least is None else least
                    raise CapExceeded(
                        f"{provider}: {what} ${used:.4f} + in flight ${held:.4f} + this run's "
                        f"{'bound' if least is None else 'smallest budget'} ${need:.4f} would "
                        f"pass the ${lim:g} {kind}"
                    )
            self._held[run_id] = (provider, grant)
            return grant

    def release(self, run_id: str) -> None:
        with self._lock:
            self._held.pop(run_id, None)
