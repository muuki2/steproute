"""The spend ledger: the single source of truth for the monthly caps and the total budgets.
Standard library only.

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

**One ledger for every output directory** (issue 60). The runner writes to one shared file
(``resolve_ledger_path``: by default ``$XDG_STATE_HOME/steproute/ledger.jsonl``, else
``~/.local/state/steproute/ledger.jsonl``), so the caps and totals count every run of the
user, whatever its output directory. Every line carries the run's output directory (``out``),
and the runner also appends each intent and settle line to ``<out>/ledger.jsonl`` (the
*mirror*): the per-directory view that reports read. The shared file is written first, so the
mirror never holds a line the shared ledger lacks, except a directory written before the
shared ledger existed; ``Ledger.import_from`` brings such a file in, once, by intent id.

**Across processes.** ``CapGuard.admit`` runs under an exclusive ``flock`` on
``<ledger>.lock`` and re-reads the file first, so it sees every runner's spend and
reservations. An admitted run's reservation is a ``reserve`` line, and its end a ``release``
line. Each reservation names its *owner*: a lock file ``<ledger>.owners/<id>.lock`` on which
the runner holds a ``flock`` for as long as it lives. The kernel drops that lock when the
process dies, however it dies, so a reservation whose owner file can be locked by someone else
belongs to a dead runner and is released (a ``release`` line saying so). A reservation whose
owner file is missing or cannot be checked stays held: when in doubt, the money is held.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:  # pragma: no cover - platform dependent
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

__all__ = [
    "LEDGER_ENV",
    "CapExceeded",
    "CapGuard",
    "ImportResult",
    "Intent",
    "Ledger",
    "Reservation",
    "UnknownSpend",
    "append_durable",
    "default_ledger_path",
    "month_of",
    "resolve_ledger_path",
]

LEDGER_VERSION = 1
#: The environment variable that moves the shared ledger (below ``[ledger] path``).
LEDGER_ENV = "STEPROUTE_LEDGER"

_OWNER_ID = re.compile(r"[0-9a-f]{32}")


def month_of(ts: float) -> str:
    """``YYYY-MM`` of a Unix timestamp, in UTC."""
    return f"{datetime.fromtimestamp(ts, UTC):%Y-%m}"


def default_ledger_path(environ: Mapping[str, str]) -> Path:
    """``$XDG_STATE_HOME/steproute/ledger.jsonl`` when that is set (and absolute, as the XDG
    specification requires), else ``~/.local/state/steproute/ledger.jsonl``. A state
    directory, not a cache: the ledger is a record that must not be cleared."""
    state = environ.get("XDG_STATE_HOME", "")
    if state and os.path.isabs(state):
        base = Path(state)
    else:
        home = environ.get("HOME")
        base = (Path(home) if home else Path.home()) / ".local" / "state"
    return base / "steproute" / "ledger.jsonl"


def resolve_ledger_path(
    *,
    explicit: Path | None = None,
    configured: Path | None = None,
    environ: Mapping[str, str],
) -> tuple[Path, str]:
    """The shared ledger and where that choice came from: ``--ledger``, then ``[ledger] path``
    in the replay config, then ``$STEPROUTE_LEDGER``, then ``default_ledger_path``."""
    if explicit is not None:
        return explicit.expanduser().absolute(), "--ledger"
    if configured is not None:
        return configured, "[ledger] path in the config"
    env = environ.get(LEDGER_ENV)
    if env:
        return Path(env).expanduser().absolute(), f"${LEDGER_ENV}"
    return default_ledger_path(environ), "the default per-user state directory"


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
    #: The output directory of the run (absolute path); empty on lines written before the
    #: shared ledger (issue 60) and not imported.
    out: str = ""

    @property
    def charged_usd(self) -> float:
        return self.worst_usd if self.settled_usd is None else self.settled_usd


@dataclass
class Reservation:
    """What an admitted run may still spend is ``usd`` minus what it has spent, held against
    the caps until the run ends (a ``release`` line) or its owner is found dead."""

    run_id: str
    provider: str
    usd: float
    owner: str
    ts: float
    out: str = ""
    released: bool = False


@dataclass(frozen=True)
class ImportResult:
    intents: int
    settles: int
    already: int
    orphans: int
    unreadable: int


class UnknownSpend(RuntimeError):
    """An output directory holds spend that the shared ledger does not know."""


class Ledger:
    """Spend per provider, month and run, read from ``path``; written by the runners.

    ``out`` tags every line this object writes with an output directory, and ``mirror`` is
    that directory's own ``ledger.jsonl``: each intent and settle line is appended to it too,
    after the shared file. Reading methods reflect what was read at the last ``refresh``
    (``CapGuard`` refreshes under the cross-process lock before every admission).
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        clock: Callable[[], float] = time.time,
        out: str | os.PathLike[str] | None = None,
        mirror: str | os.PathLike[str] | None = None,
    ):
        self.path = Path(path)
        self.out = str(Path(out).absolute()) if out is not None else None
        self.mirror = Path(mirror) if mirror is not None else None
        if self.mirror is not None and self.mirror.absolute() == self.path.absolute():
            self.mirror = None  # the shared ledger configured inside the directory: one file
        self._clock = clock
        self._lock = threading.Lock()
        self.intents: dict[str, Intent] = {}
        self.reservations: dict[str, Reservation] = {}
        self._records: dict[tuple[str, str], dict[str, Any]] = {}
        self._offset = 0
        self._bad = 0
        self._torn = False
        self._owner_id: str | None = None
        self._owner_fd: int | None = None
        self.reload()

    @property
    def lock_path(self) -> Path:
        return self.path.with_name(self.path.name + ".lock")

    @property
    def owners_dir(self) -> Path:
        return self.path.with_name(self.path.name + ".owners")

    @property
    def skipped_lines(self) -> int:
        """Unreadable lines: torn writes (and an unterminated last line)."""
        with self._lock:
            return self._bad + int(self._torn)

    # --- reading the file ---

    def _reset(self) -> None:
        self.intents = {}
        self.reservations = {}
        self._records = {}
        self._offset = 0
        self._bad = 0
        self._torn = False

    def reload(self) -> None:
        """Read the whole file again."""
        with self._lock:
            self._reset()
            self._read_new()

    def refresh(self) -> None:
        """Read what other writers appended since the last read."""
        with self._lock:
            self._read_new()

    def _read_new(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "rb") as f:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # never half of a line being written
            try:
                if os.fstat(f.fileno()).st_size < self._offset:
                    self._reset()  # the file was replaced: read it from the start
                f.seek(self._offset)
                data = f.read()
            finally:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        end = data.rfind(b"\n") + 1
        # An unterminated tail is a torn write (writers hold the lock while writing): it
        # stays unread, and the next append closes it into a line of its own.
        self._torn = bool(data[end:].strip())
        self._offset += end
        for raw in data[:end].split(b"\n"):
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except ValueError:
                self._bad += 1
                continue
            if not isinstance(obj, dict) or not self._apply(obj):
                self._bad += 1

    def _apply(self, r: dict[str, Any]) -> bool:
        """Apply one line to the state; ``False`` when it is not a valid ledger line."""
        kind = r.get("kind")
        try:
            if kind == "intent":
                i = Intent(
                    id=str(r["id"]),
                    ts=float(r["ts"]),
                    month=str(r["month"]),
                    provider=str(r["provider"]),
                    worst_usd=float(r["usd"]),
                    run_id=str(r.get("run_id", "")),
                    out=str(r.get("out", "")),
                )
                self.intents.setdefault(i.id, i)
                self._records.setdefault(("intent", i.id), r)
            elif kind == "settle":
                rid = str(r["id"])
                usd = float(r["usd"])
                lu = r.get("list_usd")
                list_usd = float(lu) if lu is not None else usd
                i2 = self.intents.get(rid)
                if i2 is not None:
                    i2.settled_usd = usd
                    i2.flags = [str(f) for f in r.get("flags", [])]
                    i2.period = str(r["period"]) if r.get("period") else None
                    i2.list_usd = list_usd
                self._records.setdefault(("settle", rid), r)
            elif kind == "reserve":
                res = Reservation(
                    run_id=str(r["run_id"]),
                    provider=str(r["provider"]),
                    usd=float(r["usd"]),
                    owner=str(r.get("owner", "")),
                    ts=float(r["ts"]),
                    out=str(r.get("out", "")),
                )
                self.reservations.setdefault(res.run_id, res)
            elif kind == "release":
                held = self.reservations.get(str(r["run_id"]))
                if held is not None:
                    held.released = True
            elif kind == "import":
                pass  # the record of an import; its lines are ordinary intents and settles
            else:
                return False
        except (KeyError, TypeError, ValueError):
            return False
        return True

    # --- writing ---

    def _write(self, rec: dict[str, Any], *, mirror: bool) -> dict[str, Any]:
        if self.out is not None:
            rec["out"] = self.out
        append_durable(self.path, rec)  # the shared ledger first: it is the one the caps read
        if mirror and self.mirror is not None:
            append_durable(self.mirror, rec)
        return rec

    def intend(self, provider: str, worst_usd: float, run_id: str) -> Intent:
        """Record, durably, that a call costing at most ``worst_usd`` is about to be made."""
        ts = self._clock()
        i = Intent(uuid.uuid4().hex, ts, month_of(ts), provider, float(worst_usd), run_id)
        i.out = self.out or ""
        rec = self._write(
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
            mirror=True,
        )
        with self._lock:
            self.intents[i.id] = i
            self._records.setdefault(("intent", i.id), rec)
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
        self._write(rec, mirror=True)
        with self._lock:
            intent.settled_usd = float(usd)
            intent.flags = fl
            intent.period = period
            intent.list_usd = float(usd) if list_usd is None else float(list_usd)
            self.intents[intent.id] = intent
            self._records.setdefault(("settle", intent.id), rec)

    # --- reservations and their owners ---

    def _own(self) -> str:
        """This object's owner id, with its lock file held for as long as it lives."""
        with self._lock:
            if self._owner_id is None:
                oid = uuid.uuid4().hex
                self.owners_dir.mkdir(parents=True, exist_ok=True)
                fd = os.open(self.owners_dir / f"{oid}.lock", os.O_RDWR | os.O_CREAT, 0o644)
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                os.write(fd, str(os.getpid()).encode())
                os.fsync(fd)
                self._owner_id, self._owner_fd = oid, fd
            return self._owner_id

    def owner_state(self, owner: str) -> str:
        """``self``, ``alive`` (its lock is held), ``dead`` (its lock file exists and nobody
        holds it) or ``unknown`` (no lock file, an unreadable one, or no ``flock`` here).
        Only ``dead`` frees a reservation."""
        if owner and owner == self._owner_id:
            return "self"
        if fcntl is None or not _OWNER_ID.fullmatch(owner):
            return "unknown"
        try:
            fd = os.open(self.owners_dir / f"{owner}.lock", os.O_RDWR)
        except OSError:
            return "unknown"
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return "alive"
            except OSError:
                return "unknown"
            fcntl.flock(fd, fcntl.LOCK_UN)
            return "dead"
        finally:
            os.close(fd)

    def reserve(self, provider: str, run_id: str, usd: float) -> Reservation:
        """Record, durably, that ``run_id`` holds ``usd`` of ``provider``'s headroom."""
        owner = self._own()
        ts = self._clock()
        self._write(
            {
                "v": LEDGER_VERSION,
                "kind": "reserve",
                "id": run_id,
                "run_id": run_id,
                "ts": ts,
                "provider": provider,
                "usd": float(usd),
                "owner": owner,
                "pid": os.getpid(),
            },
            mirror=False,
        )
        res = Reservation(run_id, provider, float(usd), owner, ts, self.out or "")
        with self._lock:
            self.reservations[run_id] = res
        return res

    def release(self, run_id: str, reason: str = "run ended") -> None:
        self._write(
            {
                "v": LEDGER_VERSION,
                "kind": "release",
                "id": run_id,
                "run_id": run_id,
                "ts": self._clock(),
                "reason": reason,
            },
            mirror=False,
        )
        with self._lock:
            held = self.reservations.get(run_id)
            if held is not None:
                held.released = True

    def live_reservations(self, provider: str | None = None) -> list[Reservation]:
        """Reservations not released and whose owner is not known to be dead."""
        with self._lock:
            held = [
                r
                for r in self.reservations.values()
                if not r.released and (provider is None or r.provider == provider)
            ]
        return [r for r in held if self.owner_state(r.owner) != "dead"]

    def stale_reservations(self) -> list[Reservation]:
        """Reservations left by a runner that died without releasing them."""
        with self._lock:
            held = [r for r in self.reservations.values() if not r.released]
        return [r for r in held if self.owner_state(r.owner) == "dead"]

    def close(self) -> None:
        """Release this object's reservations still held and drop its owner lock."""
        with self._lock:
            own = [
                r.run_id
                for r in self.reservations.values()
                if not r.released and r.owner == self._owner_id
            ]
        for run_id in own:
            self.release(run_id, reason="runner ended")
        with self._lock:
            fd, oid = self._owner_fd, self._owner_id
            self._owner_fd = self._owner_id = None
        if fd is not None and oid is not None:
            with contextlib.suppress(OSError):
                (self.owners_dir / f"{oid}.lock").unlink()
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    @contextlib.contextmanager
    def exclusive(self) -> Iterator[None]:
        """One holder at a time across processes (and threads): ``flock`` on ``<ledger>.lock``.
        Not reentrant."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    # --- output directories ---

    def unknown_spend(self, local: Path) -> list[Intent]:
        """Intents in ``local`` (an output directory's ``ledger.jsonl``) that this ledger does
        not hold: spend the caps would not count."""
        other = Ledger(local)
        with self._lock:
            return [i for i in other.intents.values() if i.id not in self.intents]

    def repair_mirror(self) -> int:
        """Append to the mirror the lines of this output directory it is missing (a crash
        between the shared write and the mirror write). Returns how many."""
        if self.mirror is None or self.out is None:
            return 0
        rows, _ = read_lines(self.mirror)
        have = {(str(r.get("kind")), str(r.get("id"))) for r in rows}
        with self._lock:
            missing = [
                rec
                for key, rec in self._records.items()
                if rec.get("out") == self.out and key not in have
            ]
        for rec in missing:
            append_durable(self.mirror, rec)
        return len(missing)

    def import_from(self, source: Path, *, out: str | os.PathLike[str]) -> ImportResult:
        """Bring the intent and settle lines of an output directory's own ledger into this
        one, tagged with ``out``. Idempotent: a line already here (same kind and intent id)
        is skipped. Lines written before ``period`` was recorded are taken as they are."""
        tag = str(Path(out).absolute())
        rows, unreadable = read_lines(source)
        added_i = added_s = already = orphans = 0
        with self.exclusive():
            self.refresh()
            for r in rows:
                kind = r.get("kind")
                if kind not in ("intent", "settle"):
                    unreadable += 1
                    continue
                key = (str(kind), str(r.get("id")))
                if key in self._records:
                    already += 1
                    continue
                if kind == "settle" and key[1] not in self.intents:
                    orphans += 1  # its intent is torn or missing: never a call that ran
                    continue
                rec = {**r, "out": tag, "imported": True}
                with self._lock:
                    ok = self._apply(dict(rec))
                if not ok:
                    unreadable += 1
                    continue
                append_durable(self.path, rec)
                with self._lock:
                    self._records[key] = rec
                if kind == "intent":
                    added_i += 1
                else:
                    added_s += 1
            if added_i or added_s:
                append_durable(
                    self.path,
                    {
                        "v": LEDGER_VERSION,
                        "kind": "import",
                        "id": uuid.uuid4().hex,
                        "ts": self._clock(),
                        "out": tag,
                        "source": str(source.absolute()),
                        "intents": added_i,
                        "settles": added_s,
                    },
                )
        return ImportResult(added_i, added_s, already, orphans, unreadable)

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

    def by_out(self) -> dict[tuple[str, str], float]:
        """``(output directory, provider) -> charged``, every month."""
        out: dict[tuple[str, str], float] = {}
        with self._lock:
            for i in self.intents.values():
                key = (i.out or "(untagged)", i.provider)
                out[key] = out.get(key, 0.0) + i.charged_usd
        return out

    def charged_by_run(self) -> dict[str, float]:
        out: dict[str, float] = {}
        with self._lock:
            for i in self.intents.values():
                out[i.run_id] = out.get(i.run_id, 0.0) + i.charged_usd
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
    and under its total budget (all-time spend in the ledger), when it has one.

    The check runs under one lock across threads and processes (``Ledger.exclusive``), after
    re-reading the ledger, so parallel runs, in this runner or in another one on the same
    ledger, can never jointly pass a cap: each admitted run holds a reservation of
    ``reservation - spent so far`` until it ends, written to the ledger.

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
        self._held: dict[str, tuple[str, float]] = {}  # this guard's runs: (provider, usd)
        #: Reservations of dead runners released by this guard (for the runner to report).
        self.reclaimed: list[Reservation] = []

    def _outstanding(self, provider: str) -> float:
        runs = self.ledger.charged_by_run()
        return sum(
            max(0.0, r.usd - runs.get(r.run_id, 0.0))
            for r in self.ledger.live_reservations(provider)
        )

    def _reclaim(self) -> None:
        for r in self.ledger.stale_reservations():
            self.ledger.release(r.run_id, reason="its runner ended without releasing it")
            self.reclaimed.append(r)

    def _limits(self, provider: str) -> list[tuple[str, float, float]]:
        """``(what, limit, spent)`` for the total budget, if set, and the monthly cap (on a tie
        the total is named: it is the one that does not reset)."""
        out = []
        if provider in self.totals:
            out.append(("all-time", self.totals[provider], self.ledger.spent_total(provider)))
        out.append(("month-to-date", self.caps[provider], self.ledger.spent(provider)))
        return out

    def headroom(self, provider: str) -> float:
        with self._lock, self.ledger.exclusive():
            self.ledger.refresh()
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
            with self.ledger.exclusive():
                self.ledger.refresh()
                self._reclaim()
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
                            f"{provider}: {what} ${used:.4f} + in flight (every runner on the "
                            f"ledger) ${held:.4f} + this run's "
                            f"{'bound' if least is None else 'smallest budget'} ${need:.4f} "
                            f"would pass the ${lim:g} {kind}"
                        )
                self.ledger.reserve(provider, run_id, grant)
                self._held[run_id] = (provider, grant)
                return grant

    def release(self, run_id: str) -> None:
        with self._lock:
            if self._held.pop(run_id, None) is not None:
                self.ledger.release(run_id)
