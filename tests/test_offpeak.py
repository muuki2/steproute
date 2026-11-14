"""Peak hours as data, and time-of-day prices (issue 57). Fake instants only."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from steproute.core.catalog import Catalog, CatalogError, load_default_catalog
from steproute.core.cost import AttemptUsage, price_usage
from steproute.core.offpeak import (
    FLAT,
    OFFPEAK,
    PEAK,
    PeakHoursError,
    call_period,
    fmt_instant,
    offpeak_profile,
    parse_peak_hours,
    peak_hours,
    price_for_period,
)


def ts(d: int, h: int, m: int = 0, s: float = 0.0) -> float:
    """An instant in the week of Monday 16 November 2026 (d=16 Monday ... d=22 Sunday)."""
    return datetime(2026, 11, d, h, m, tzinfo=UTC).timestamp() + s


CAT = load_default_catalog()
FLASH = CAT.get("deepseek-flash")
PRO = CAT.get("deepseek-v4-pro")
H = peak_hours(FLASH)
assert H is not None


def test_the_catalog_records_deepseeks_windows_with_source_and_date() -> None:
    assert H.table() == [
        ("Mon, Tue, Wed, Thu, Fri", "01:00", "04:00"),
        ("Mon, Tue, Wed, Thu, Fri", "06:00", "10:00"),
    ]
    assert H.source == "https://api-docs.deepseek.com/quick_start/pricing"
    assert H.checked == "2026-11-13"
    assert peak_hours(PRO) == H
    for m in CAT.models:
        if m.provider != "deepseek":
            assert peak_hours(m) is None  # flat


@pytest.mark.parametrize(
    ("instant", "period"),
    [
        (ts(16, 0, 59, 59.9), OFFPEAK),
        (ts(16, 1), PEAK),  # first window: start inclusive
        (ts(16, 3, 59, 59.9), PEAK),
        (ts(16, 4), OFFPEAK),  # end exclusive
        (ts(16, 5, 59, 59.9), OFFPEAK),
        (ts(16, 6), PEAK),  # second window
        (ts(16, 9, 59, 59.9), PEAK),
        (ts(16, 10), OFFPEAK),
        (ts(20, 9), PEAK),  # Friday
        (ts(20, 23, 59), OFFPEAK),
        (ts(21, 2), OFFPEAK),  # Saturday
        (ts(21, 7), OFFPEAK),
        (ts(22, 2), OFFPEAK),  # Sunday
        (ts(22, 8), OFFPEAK),
        (ts(23, 1), PEAK),  # Monday again
    ],
)
def test_peak_and_off_peak_at_every_boundary(instant: float, period: str) -> None:
    assert H.period(instant) == period


def test_when_the_window_ends_and_the_next_one_starts() -> None:
    # Off-peak at 00:00 Monday: the window ends at 01:00.
    assert H.offpeak_end(ts(16, 0)) == ts(16, 1)
    assert H.offpeak_left(ts(16, 0)) == 3600
    # In the gap 04:00-06:00 it ends at 06:00.
    assert H.offpeak_end(ts(16, 4, 30)) == ts(16, 6)
    # After 10:00 it ends at 01:00 the next day.
    assert H.offpeak_end(ts(16, 10)) == ts(17, 1)
    # Friday after 10:00: open the whole weekend, until Monday 01:00.
    assert H.offpeak_end(ts(20, 12)) == ts(23, 1)
    # In peak hours: zero left, and the next off-peak starts at the end of the window.
    assert H.offpeak_left(ts(16, 2)) == 0
    assert H.next_offpeak_start(ts(16, 2)) == ts(16, 4)
    assert H.next_offpeak_start(ts(16, 7)) == ts(16, 10)
    assert H.next_offpeak_start(ts(16, 11)) == ts(16, 11)  # already off-peak


def test_next_window_with_a_margin_skips_the_short_gap() -> None:
    # 04:00-06:00 is exactly two hours: enough for a 2 h margin, not for more.
    assert H.next_window(ts(16, 2), 7200) == (ts(16, 4), ts(16, 6))
    assert H.next_window(ts(16, 2), 7201) == (ts(16, 10), ts(17, 1))
    # From 00:30 on Monday with a 2 h margin: not now (30 min left), not 04:00 + 1 s.
    assert H.next_window(ts(16, 0, 30), 7200)[0] == ts(16, 4)
    assert H.next_window(ts(16, 4, 0, 1), 7200)[0] == ts(16, 10)
    # Friday evening: the whole weekend.
    assert H.next_window(ts(20, 18), 7200) == (ts(20, 18), ts(23, 1))


@pytest.mark.parametrize(
    ("start", "end", "period"),
    [
        (ts(16, 0, 50), ts(16, 0, 59), OFFPEAK),
        (ts(16, 0, 59), ts(16, 1, 0, 1), PEAK),  # crosses into the first window
        (ts(16, 3, 59), ts(16, 4, 1), PEAK),  # crosses out of it
        (ts(16, 5, 59), ts(16, 6, 0, 1), PEAK),  # into the second
        (ts(16, 9, 59), ts(16, 10, 1), PEAK),  # out of it
        (ts(16, 4, 1), ts(16, 5, 59), OFFPEAK),  # inside the gap
        (ts(16, 0, 30), ts(16, 4, 30), PEAK),  # spans a whole window
        (ts(20, 23, 59), ts(21, 0, 1), OFFPEAK),  # Friday into Saturday
        (ts(22, 23, 59), ts(23, 0, 1), OFFPEAK),  # Sunday into Monday
        (ts(23, 0, 59), ts(23, 1, 0, 1), PEAK),  # weekend into Monday's window
    ],
)
def test_a_call_is_off_peak_only_if_it_starts_and_ends_off_peak(
    start: float, end: float, period: str
) -> None:
    assert call_period(H, start, end) == period


def test_a_model_without_windows_is_flat() -> None:
    haiku = CAT.get("claude-haiku-5-5")
    assert call_period(peak_hours(haiku), ts(16, 2), ts(16, 3)) == FLAT
    assert offpeak_profile(haiku) is None
    assert price_for_period(haiku, OFFPEAK) is haiku


def test_off_peak_prices_are_half_for_misses_hits_and_output() -> None:
    for p in (FLASH, PRO):
        off = offpeak_profile(p)
        assert off is not None
        u = AttemptUsage(input_tokens=1e6, cache_read_tokens=2e6, output_tokens=5e5)
        peak, low = price_usage(p, u), price_usage(off, u)
        assert low.input_usd == pytest.approx(peak.input_usd / 2)
        assert low.cache_read_usd == pytest.approx(peak.cache_read_usd / 2)
        assert low.output_usd == pytest.approx(peak.output_usd / 2)
        assert price_for_period(p, PEAK) is p
    # the cache-hit prices in the catalog are what the ratio gives
    off = offpeak_profile(FLASH)
    assert off is not None and off.price_in * off.cache_read_mult == pytest.approx(0.003)


def _cat_with(peak: object, **meta: object) -> dict[str, object]:
    m = FLASH.to_dict()
    m["meta"] = {**m["meta"], **meta, "peak_hours": peak}
    return {"model": [m]}


def test_bad_peak_hours_are_catalog_errors() -> None:
    good = dict(FLASH.meta["peak_hours"])
    bad_cases = [
        {**good, "timezone": "Asia/Shanghai"},
        {**good, "windows": [{"days": ["mon"], "start": "04:00", "end": "01:00"}]},
        {**good, "windows": [{"days": ["monday"], "start": "01:00", "end": "04:00"}]},
        {**good, "windows": [{"days": ["mon"], "start": "1:00", "end": "04:00"}]},
        {**good, "windows": []},
        {k: v for k, v in good.items() if k != "source"},
        {**good, "extra": 1},
    ]
    for case in bad_cases:
        with pytest.raises(CatalogError, match="peak_hours"):
            Catalog.from_dict(_cat_with(case), warn_unverified=False)
    with pytest.raises(CatalogError, match="price_in_offpeak"):
        Catalog.from_dict(_cat_with(good, price_in_offpeak="cheap"), warn_unverified=False)
    with pytest.raises(PeakHoursError):
        parse_peak_hours("01:00-04:00")


def test_fmt_instant_shows_utc_and_local() -> None:
    text = fmt_instant(ts(16, 10))
    assert text.startswith("2026-11-16 10:00 UTC (") and "local)" in text
    assert fmt_instant(math.inf) == "never"
