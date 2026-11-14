"""Time-of-day pricing: peak hours as data, and the off-peak price of a model. Standard library.

A model whose catalog entry has a ``[model.meta.peak_hours]`` table is billed at its catalog
prices (``price_in``, ``price_out``) inside the listed windows and at its off-peak prices
(``meta.price_in_offpeak``, ``meta.price_out_offpeak``) outside them. A model without the
table is ``flat``: one price at every hour. See ``docs/CATALOG.md``, "Peak hours".

Rules this module encodes (``docs/COST-MODEL.md``, "Time of day"):

* Windows are in UTC, on named weekdays, ``start`` inclusive and ``end`` exclusive.
* Public holidays are treated as ordinary days. A provider may bill a holiday off-peak; we
  cannot verify a holiday calendar, so we never assume the discount.
* A call is priced off-peak only if both its start and its end are off-peak; a call that
  touches a peak window anywhere is priced at peak. (A call that starts and ends off-peak
  but spans a whole window is impossible for the windows we have: none is shorter than
  the longest call the runner allows. ``call_period`` still checks for it.)
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from steproute.core.schema import ModelProfile

__all__ = [
    "DAYS",
    "FLAT",
    "OFFPEAK",
    "PEAK",
    "PeakHours",
    "PeakHoursError",
    "PeakWindow",
    "call_period",
    "fmt_instant",
    "offpeak_profile",
    "parse_peak_hours",
    "peak_hours",
    "price_for_period",
]

PEAK = "peak"
OFFPEAK = "offpeak"
FLAT = "flat"

DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_HHMM = re.compile(r"^([01]\d|2[0-4]):([0-5]\d)$")
_KEYS = {"windows", "timezone", "source", "checked", "note"}


class PeakHoursError(ValueError):
    """A ``peak_hours`` table is malformed."""


@dataclass(frozen=True)
class PeakWindow:
    """One peak window: ``[start, end)`` minutes after midnight UTC on each of ``days``
    (0 = Monday). It never crosses midnight; write two windows for that."""

    days: frozenset[int]
    start: int
    end: int


@dataclass(frozen=True)
class PeakHours:
    windows: tuple[PeakWindow, ...]
    source: str = ""
    checked: str = ""
    note: str = ""

    # --- the intervals around an instant ---

    def _intervals(
        self, ts: float, days_before: int = 1, days_after: int = 9
    ) -> list[tuple[float, float]]:
        """Peak intervals ``[start, end)`` in Unix seconds overlapping the days around ``ts``,
        sorted and merged (two windows that touch are one interval)."""
        day0 = datetime.fromtimestamp(ts, UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        raw: list[tuple[float, float]] = []
        for k in range(-days_before, days_after + 1):
            d = day0 + timedelta(days=k)
            base = d.timestamp()
            for w in self.windows:
                if d.weekday() in w.days:
                    raw.append((base + 60 * w.start, base + 60 * w.end))
        raw.sort()
        merged: list[tuple[float, float]] = []
        for s, e in raw:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return merged

    def is_peak(self, ts: float) -> bool:
        return any(s <= ts < e for s, e in self._intervals(ts))

    def period(self, ts: float) -> str:
        return PEAK if self.is_peak(ts) else OFFPEAK

    def offpeak_end(self, ts: float) -> float:
        """When the off-peak window around ``ts`` ends (the next peak start), or ``ts`` itself
        when ``ts`` is in peak hours. ``inf`` when there are no windows."""
        if not self.windows:
            return math.inf
        for s, e in self._intervals(ts):
            if s <= ts < e:
                return ts
            if s > ts:
                return s
        return math.inf  # pragma: no cover - every week has a window when windows exist

    def offpeak_left(self, ts: float) -> float:
        """Seconds of off-peak left from ``ts`` (0 in peak hours, ``inf`` without windows)."""
        return self.offpeak_end(ts) - ts

    def next_offpeak_start(self, ts: float) -> float:
        """``ts`` when it is off-peak; otherwise the end of the peak hours it is in."""
        for s, e in self._intervals(ts):
            if s <= ts < e:
                return e
        return ts

    def next_window(self, ts: float, margin_s: float = 0.0) -> tuple[float, float]:
        """The first off-peak stretch at or after ``ts`` that stays open for ``margin_s``:
        ``(start, end)``. ``end`` may be ``inf``."""
        t = ts
        for _ in range(64):
            t = self.next_offpeak_start(t)
            end = self.offpeak_end(t)
            if end - t >= margin_s:
                return t, end
            t = end
        raise PeakHoursError(f"no off-peak stretch of {margin_s:g} s in the next weeks")

    def table(self) -> list[tuple[str, str, str]]:
        """``(days, start, end)`` rows for display, times as ``HH:MM`` UTC."""
        out = []
        for w in self.windows:
            days = ", ".join(DAYS[d].capitalize() for d in sorted(w.days))
            out.append((days, _hhmm(w.start), _hhmm(w.end)))
        return out


def _hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _minutes(v: Any, where: str) -> int:
    if not isinstance(v, str) or not _HHMM.match(v):
        raise PeakHoursError(f'{where} must be a time as "HH:MM" (UTC), got {v!r}')
    h, m = (int(x) for x in v.split(":"))
    total = 60 * h + m
    if total > 24 * 60:
        raise PeakHoursError(f"{where} must not be later than 24:00")
    return total


def parse_peak_hours(table: Any) -> PeakHours:
    """Validate a ``peak_hours`` table (see ``docs/CATALOG.md``)."""
    if not isinstance(table, Mapping):
        raise PeakHoursError("peak_hours must be a table")
    unknown = sorted(set(table) - _KEYS)
    if unknown:
        raise PeakHoursError(f"peak_hours: unknown key(s) {unknown}")
    if table.get("timezone", "UTC") != "UTC":
        raise PeakHoursError('peak_hours: timezone must be "UTC" (write the windows in UTC)')
    for k in ("source", "checked"):
        if not isinstance(table.get(k), str) or not table.get(k):
            raise PeakHoursError(f"peak_hours: {k} is required (where the hours come from, when)")
    raw = table.get("windows")
    if not isinstance(raw, Sequence) or isinstance(raw, str) or not raw:
        raise PeakHoursError("peak_hours: windows must be a non-empty array of tables")
    windows = []
    for i, w in enumerate(raw):
        where = f"peak_hours.windows[{i}]"
        if not isinstance(w, Mapping):
            raise PeakHoursError(f"{where} must be a table")
        bad = sorted(set(w) - {"days", "start", "end"})
        if bad:
            raise PeakHoursError(f"{where}: unknown key(s) {bad}")
        days = w.get("days")
        if (
            not isinstance(days, Sequence)
            or isinstance(days, str)
            or not days
            or any(d not in DAYS for d in days)
        ):
            raise PeakHoursError(f"{where}.days must list days from {list(DAYS)}")
        start = _minutes(w.get("start"), f"{where}.start")
        end = _minutes(w.get("end"), f"{where}.end")
        if end <= start:
            raise PeakHoursError(f"{where}: end must be after start (split a window at midnight)")
        windows.append(PeakWindow(frozenset(DAYS.index(d) for d in days), start, end))
    note = table.get("note", "")
    return PeakHours(tuple(windows), str(table["source"]), str(table["checked"]), str(note))


def peak_hours(profile: ModelProfile) -> PeakHours | None:
    """The model's peak hours, or ``None`` for a flat-priced model."""
    t = profile.meta.get("peak_hours")
    return None if t is None else parse_peak_hours(t)


def _price(meta: Mapping[str, Any], key: str, model: str) -> float:
    v = meta.get(key)
    if isinstance(v, bool) or not isinstance(v, int | float) or v < 0:
        raise PeakHoursError(f"{model}: meta.{key} must be a non-negative number with peak_hours")
    return float(v)


def offpeak_profile(profile: ModelProfile) -> ModelProfile | None:
    """The profile with off-peak prices, or ``None`` for a flat-priced model.

    ``meta.price_in_offpeak`` and ``meta.price_out_offpeak`` replace the prices. The cache-hit
    ratio is ``meta.cache_hit_price_offpeak / price_in_offpeak`` when the cache-hit price is
    given, otherwise ``cache_read_mult`` unchanged. Price tiers are not supported together
    with off-peak prices (no provider has both) and are refused.
    """
    if peak_hours(profile) is None:
        return None
    meta = profile.meta
    pin = _price(meta, "price_in_offpeak", profile.id)
    pout = _price(meta, "price_out_offpeak", profile.id)
    if profile.price_tiers:
        raise PeakHoursError(f"{profile.id}: off-peak prices with price tiers are not supported")
    read_mult = profile.cache_read_mult
    if "cache_hit_price_offpeak" in meta and pin > 0:
        read_mult = _price(meta, "cache_hit_price_offpeak", profile.id) / pin
    return dataclasses.replace(profile, price_in=pin, price_out=pout, cache_read_mult=read_mult)


def call_period(hours: PeakHours | None, start: float, end: float) -> str:
    """``flat`` without peak hours; ``offpeak`` only if ``[start, end]`` touches no peak hour;
    otherwise ``peak``."""
    if hours is None:
        return FLAT
    if hours.is_peak(start) or hours.is_peak(end):
        return PEAK
    if hours.offpeak_end(start) <= end:
        return PEAK  # a whole peak window inside the call
    return OFFPEAK


def price_for_period(profile: ModelProfile, period: str) -> ModelProfile:
    """The profile to price a call of ``period`` with: off-peak prices only for ``offpeak``."""
    if period == OFFPEAK:
        off = offpeak_profile(profile)
        if off is not None:
            return off
    return profile


def fmt_instant(ts: float) -> str:
    """``2026-11-16 10:00 UTC (11:00 CET local)``: UTC first, then this machine's local time."""
    if math.isinf(ts):
        return "never"
    utc = datetime.fromtimestamp(ts, UTC)
    local = utc.astimezone()
    same_day = local.date() == utc.date()
    lfmt = "%H:%M" if same_day else "%Y-%m-%d %H:%M"
    return (
        f"{utc:%Y-%m-%d %H:%M} UTC ({local.strftime(lfmt)} {local.tzname() or ''} local)".replace(
            "  ", " "
        )
    )
