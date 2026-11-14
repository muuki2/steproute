"""The grid runner: resume, isolation, concurrency, caps. See ``docs/REPLAY.md``.

What is safe by default:

* ``execute=False`` (the default) prints the plan and touches nothing.
* A real run takes the output directory's lock first; a second runner on it exits.
* Every paid provider in the plan needs its key (DeepSeek explicitly) before anything runs.
* Before each paid run, ``CapGuard.admit`` checks month-to-date spend (the ledger), plus what
  runs in flight could still spend, plus this run's worst-case bound, against the monthly cap
  and the provider's total budget. When less than a full bound is left (but at least
  ``min_run_budget_usd`` plus one worst-case call), the run gets that smaller budget and ends
  as ``budget_stop`` (pending, not a failure) if it reaches it.
* A provider with ``offpeak_only`` (DeepSeek by default) starts a run only when its off-peak
  hours stay open for the start margin, and its client never sends a call in peak hours.
* Each run has its own proxy, session and nonce; the configuration that executed is checked
  on every call, and a mismatch stops the whole runner.

Standard library only (plus the proxy's ``[proxy]`` extra when a run calls a model).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sys
import threading
import time
import urllib.request
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TextIO

from steproute import __version__
from steproute.core.catalog import Catalog
from steproute.core.offpeak import PeakHours, PeakHoursError, fmt_instant, peak_hours
from steproute.core.policy import Policy
from steproute.core.privacy import PrivacyConfig
from steproute.core.store import TraceStore
from steproute.eval.data import GridRun, GridTask, write_grid
from steproute.replay.agent import AgentResult, BashAgent, RunStopped, Scaffold
from steproute.replay.config import ReplayConfig
from steproute.replay.journal import DEFERRED, LABELLED, RETRYABLE, Attempt, Journal, OutputLock
from steproute.replay.ledger import CapExceeded, CapGuard, Ledger, month_of
from steproute.replay.plan import ReplayPlan, RunSpec
from steproute.replay.proxyrun import (
    PAUSE_CHUNK_S,
    ConfigMismatch,
    ProxyChatClient,
    Redactor,
    RunProxy,
    worst_call_usd,
)
from steproute.replay.workspace import DockerWorkspace, LocalWorkspace, Workspace, WorkspaceError
from steproute.tasks.check import (
    Backend,
    CheckError,
    CheckResult,
    LocalBackend,
    SwebenchBackend,
    load_rows,
)
from steproute.tasks.manifest import Task, TaskManifest

__all__ = [
    "EXIT_CAP",
    "EXIT_FATAL",
    "EXIT_INTERRUPTED",
    "MissingKey",
    "Runner",
    "RunnerDeps",
    "describe_plan",
    "missing_keys",
    "run_bound",
    "write_grids",
]

EXIT_CAP = 4
EXIT_FATAL = 3
EXIT_INTERRUPTED = 130

#: Statuses with a labelled outcome, and their ``success``.
_SUCCESS = {s: (s == "passed") for s in LABELLED}


class MissingKey(RuntimeError):
    """A paid provider in the plan has no key configured. Nothing was run."""


class LocalModels(Protocol):
    def load(self, model: str) -> None: ...

    def unload_all(self) -> None: ...


class OllamaModels:
    """One local model loaded at a time: unload the previous one (``keep_alive = 0``)."""

    def __init__(self, base_url: str) -> None:
        self.base = base_url.rstrip("/").removesuffix("/v1")
        self.loaded: str | None = None

    def _unload(self, model: str) -> None:
        data = json.dumps({"model": model, "keep_alive": 0}).encode()
        req = urllib.request.Request(
            f"{self.base}/api/generate", data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                r.read()
        except OSError:
            pass  # best effort: Ollama also evicts on its own with OLLAMA_MAX_LOADED_MODELS=1

    def load(self, model: str) -> None:
        if self.loaded is not None and self.loaded != model:
            self._unload(self.loaded)
        self.loaded = model

    def unload_all(self) -> None:
        if self.loaded is not None:
            self._unload(self.loaded)
            self.loaded = None


class NoLocalModels:
    def load(self, model: str) -> None:
        return None

    def unload_all(self) -> None:
        return None


@dataclass
class RunnerDeps:
    """Everything the runner touches outside itself, so tests can replace it."""

    scaffold: Callable[[], Scaffold] | None = None
    workspace: Callable[[Task], Workspace] | None = None
    backend: Callable[[Task], Backend] | None = None
    task_text: Callable[[Task], str] | None = None
    local_models: LocalModels | None = None
    #: Test hook: replaces the run's ``FixedPolicy`` (to prove a mismatch is caught).
    policy: Callable[[RunSpec, Catalog], Policy] | None = None
    clock: Callable[[], float] = time.time
    #: The timer each run measures its timeout and its pauses with.
    monotonic: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    environ: Mapping[str, str] = field(default_factory=lambda: os.environ)
    out: TextIO = field(default_factory=lambda: sys.stdout)
    #: Values to scrub from everything the runner writes, besides the configured keys (every
    #: value loaded from ``--env-file``).
    secrets: Sequence[str] = ()


def run_bound(spec: RunSpec, catalog: Catalog, config: ReplayConfig) -> float:
    """The most one run can spend: the per-run cap is checked after every call, so a run can
    pass it by at most one call, whose worst case is ``worst_call_usd``. Local runs: 0."""
    if spec.local:
        return 0.0
    wc = worst_call_usd(catalog.get(spec.model), config.limits, local=False)
    return config.limits.run_cap_usd + wc


def worst_call(spec: RunSpec, catalog: Catalog, config: ReplayConfig) -> float:
    """The most one call of this run can cost, at the catalog (peak) price."""
    if spec.local:
        return 0.0
    return worst_call_usd(catalog.get(spec.model), config.limits, local=False)


def missing_keys(plan: ReplayPlan, config: ReplayConfig, environ: Mapping[str, str]) -> list[str]:
    """Messages for every paid provider in the plan without a usable key (names only)."""
    out = []
    for prov in plan.paid_providers():
        env = config.provider(prov).api_key_env
        if env is None:
            out.append(f"{prov}: no api_key_env is configured in [providers.{prov}]")
        elif not environ.get(env):
            out.append(f"{prov}: the environment variable {env} is not set")
    return out


def _fmt(usd: float) -> str:
    return f"${usd:,.2f}"


def describe_plan(
    plan: ReplayPlan,
    catalog: Catalog,
    config: ReplayConfig,
    *,
    ledger: Ledger | None,
    journal: Journal | None,
    environ: Mapping[str, str],
    now: float,
) -> str:
    """The plan, the estimated spend per provider with its range, the caps, month-to-date
    spend and the key status (set or not set, never the value)."""
    done = {a.cell for a in journal.finished()} if journal else set()
    pending = [s for s in plan.specs if s.cell not in done]
    lines = [
        f"plan {plan.name}: {plan.description}",
        f"  runs: {len(plan.specs)} in the plan, {len(plan.specs) - len(pending)} finished, "
        f"{len(pending)} to run",
        "",
        "  order (cheapest first; configuration-major):",
    ]
    by_conf: dict[tuple[str, str, bool], list[RunSpec]] = {}
    for s in pending:
        by_conf.setdefault((s.model, s.effort, s.holdout), []).append(s)
    for i, ((model, effort, holdout), ss) in enumerate(by_conf.items(), 1):
        est = ss[0].est_usd
        tag = "  [holdout split]" if holdout else ""
        stage = f"  stage {ss[0].stage}" if plan.gate else ""
        lines.append(
            f"   {i:2d}. {model:<18} {effort:<6} {len(ss):4d} runs  "
            f"~{_fmt(est[0] * len(ss))} ({_fmt(est[1] * len(ss))}-{_fmt(est[2] * len(ss))})"
            f"{stage}{tag}"
        )
    lines += ["", "  estimated spend of the runs to go (central, low-high) and caps:"]
    month = month_of(now)
    for prov in plan.providers():
        ss = [s for s in pending if s.provider == prov]
        c, lo, hi = (sum(s.est_usd[k] for s in ss) for k in range(3))
        cap = config.cap(prov)
        if cap is None:
            lines.append(f"   {prov:<10} {len(ss):4d} runs  free (local)")
            continue
        mtd = ledger.spent(prov, month) if ledger else 0.0
        bound = max((run_bound(s, catalog, config) for s in ss), default=0.0)
        sched = config.schedule(prov)
        timed = [s for s in ss if peak_hours(catalog.get(s.model)) is not None]
        offpeak = sched.offpeak_only and bool(ss) and len(timed) == len(ss)
        price = "at peak (list) prices" if offpeak else ""
        lines.append(
            f"   {prov:<10} {len(ss):4d} runs  ~{_fmt(c)} ({_fmt(lo)}-{_fmt(hi)}) {price}".rstrip()
            + f"  cap {_fmt(cap)}/month, month-to-date {month} {_fmt(mtd)}, "
            f"worst case per run {_fmt(bound)}"
        )
        room = cap - mtd
        total = config.total(prov)
        if total is not None:
            all_time = ledger.spent_total(prov) if ledger else 0.0
            room = min(room, total - all_time)
            lines.append(
                f"   {'':<10} total budget {_fmt(total)} (all months), spent so far "
                f"{_fmt(all_time)}, left {_fmt(total - all_time)}"
            )
        if offpeak:
            oc, olo, ohi = (sum((s.est_offpeak_usd or s.est_usd)[k] for s in ss) for k in range(3))
            lines.append(
                f"   {'':<10} scheduled off-peak only: ~{_fmt(oc)} ({_fmt(olo)}-{_fmt(ohi)}) "
                f"at off-peak prices, against {_fmt(room)} left"
                + ("; the central estimate fits" if oc <= room else "; does NOT fit")
            )
            ph = peak_hours(catalog.get(timed[0].model))
            assert ph is not None
            margin = config.start_margin(prov)
            period = ph.period(now)
            lines.append(
                f"   {'':<10} now {fmt_instant(now)} is {period.replace('offpeak', 'off-peak')}"
                + (
                    f", off-peak until {fmt_instant(ph.offpeak_end(now))}"
                    if period == "offpeak"
                    else ""
                )
            )
            start, end = ph.next_window(now, margin)
            lines.append(
                f"   {'':<10} next start (off-peak with {margin / 3600:g} h ahead): "
                f"{fmt_instant(start)}, window closes {fmt_instant(end)}"
            )
            hi_cmp = ohi
        else:
            hi_cmp = hi
        if hi_cmp > room:
            lines.append(
                f"   {'':<10} the high estimate does not fit; the budget stops {prov} first "
                "(cheapest first leaves a usable partial matrix; unrun cells stay pending)"
            )
    lines += ["", "  keys (values are never shown):"]
    for prov in plan.paid_providers():
        env = config.provider(prov).api_key_env
        state = "set" if env and environ.get(env) else "NOT SET"
        lines.append(f"   {prov:<10} {env or '(none configured)'}: {state}")
    lim = config.limits
    lines += [
        "",
        f"  limits: ${lim.run_cap_usd:g} per run, {lim.step_limit} steps, "
        f"{lim.max_prompt_tokens:,} prompt tokens per paid call, "
        f"{lim.max_output_tokens:,} output tokens per call, "
        f"local context {lim.local_context_cap:,}",
        f"  concurrency: {config.paid_concurrency} paid runs in parallel; local runs one at a "
        "time, one model loaded",
    ]
    if plan.gate:
        g = plan.gate
        lines.append(
            f"  stage gate: after stage 1, if {g.provider} month-to-date + r x the rest of the "
            f"central estimate > {_fmt(g.threshold_usd)}, stage 2 {g.provider} cells are cut to "
            f"{len(g.reduced_tasks)} tasks (other providers are not cut)"
        )
    return "\n".join(lines)


# --- the grid ----------------------------------------------------------------------------


def _retryable(attempts: Sequence[Attempt]) -> int:
    """Attempts that count against ``max_retries``: a deferred one (``budget_stop``,
    ``balance_stop``) does not."""
    return sum(1 for a in attempts if a.status not in DEFERRED)


def _done(attempts: Sequence[Attempt], max_retries: int) -> bool:
    last = attempts[-1]
    if last.terminal:
        return True
    return last.status in RETRYABLE and _retryable(attempts) > max_retries


def write_grids(
    out: Path,
    journal: Journal,
    catalog: Catalog,
    manifest: TaskManifest,
    *,
    max_retries: int,
    header: Mapping[str, Any] | None = None,
) -> tuple[int, int]:
    """``grid.jsonl`` (the pool) and ``grid-holdout.jsonl`` (holdout models), in the
    evaluator's contract, rebuilt from the journal. One run line per finished cell: its last
    attempt. Returns the number of run lines in each."""
    pool: list[GridRun] = []
    held: list[GridRun] = []
    for cell, attempts in journal.by_cell().items():
        if not _done(attempts, max_retries):
            continue
        last = attempts[-1]
        end = last.end or {}
        task_id, model, effort, repeat = cell
        earlier = sum(float((a.end or {}).get("cost_usd") or 0.0) for a in attempts[:-1])
        run = GridRun(
            task_id=task_id,
            model=model,
            effort=effort,
            success=end.get("success"),
            cost_usd=end.get("cost_usd"),
            latency_ms=end.get("latency_ms"),
            run_id=last.run_id,
            session_id=last.session_id,
            repeat=repeat,
            ts=float(end.get("ts", last.started)),
            meta={
                "status": end.get("status"),
                "reason": end.get("reason"),
                "attempts": len(attempts),
                "spent_on_earlier_attempts_usd": round(earlier, 6),
                "steps": end.get("steps"),
                "calls_missing_usage": end.get("calls_missing_usage", 0),
                "scaffold": end.get("scaffold"),
                # ``cost_usd`` above is what was billed (off-peak calls at off-peak prices);
                # this is the same usage at the catalog's peak (list) price.
                "list_cost_usd": end.get("list_cost_usd", end.get("cost_usd")),
                "calls_by_period": end.get("calls_by_period", {}),
            },
        )
        holdout = model in catalog and catalog.get(model).holdout
        (held if holdout else pool).append(run)
    order = {t.id: i for i, t in enumerate(manifest)}

    def tasks_for(runs: Iterable[GridRun]) -> list[GridTask]:
        ids = sorted({r.task_id for r in runs}, key=lambda t: order.get(t, 0))
        out_t = []
        for tid in ids:
            feats: dict[str, Any] = {}
            if tid in manifest:
                t = manifest.get(tid)
                feats = {"stratum": t.stratum, "fits32k": t.fits32k, "pilot": t.pilot}
            out_t.append(GridTask(tid, None, None, feats))
        return out_t

    hdr = {"runner": f"steproute replay {__version__}", **(header or {})}
    for name, runs, split in (
        ("grid.jsonl", pool, "pool"),
        ("grid-holdout.jsonl", held, "holdout"),
    ):
        path = out / name
        if not runs:
            continue
        tmp = path.with_suffix(".jsonl.tmp")
        write_grid(tmp, runs, tasks_for(runs), {**hdr, "split": split})
        os.replace(tmp, path)
    return len(pool), len(held)


# --- the runner --------------------------------------------------------------------------


@dataclass
class _Lane:
    """The paid specs not started yet, in plan order."""

    specs: list[RunSpec]


class Runner:
    def __init__(
        self,
        plan: ReplayPlan,
        *,
        config: ReplayConfig,
        catalog: Catalog,
        manifest: TaskManifest,
        out: Path,
        deps: RunnerDeps | None = None,
        inputs: Mapping[str, Any] | None = None,
    ) -> None:
        self.inputs = dict(inputs or {})
        self.plan = plan
        self.config = config
        self.catalog = catalog
        self.manifest = manifest
        self.out = Path(out)
        self.deps = deps or RunnerDeps()
        self.stop = threading.Event()
        self.fatal: str | None = None
        self.halted: dict[str, str] = {}  # provider -> why its remaining runs are skipped
        self._cv = threading.Condition()
        self._in_flight: dict[str, int] = {}
        keys = [
            self.deps.environ.get(p.api_key_env, "")
            for p in config.providers.values()
            if p.api_key_env
        ]
        self.redact = Redactor([*keys, *self.deps.secrets])
        self._announced: set[tuple[str, float]] = set()
        self.ledger: Ledger | None = None
        self.journal: Journal | None = None
        self.guard: CapGuard | None = None
        self.stores: dict[str, TraceStore] = {}
        self._rows_cache: Mapping[str, Mapping[str, Any]] | None = None

    def say(self, text: str) -> None:
        print(self.redact(text), file=self.deps.out, flush=True)

    # --- identity of the plan in this directory ---

    def _plan_identity(self) -> dict[str, Any]:
        cells = [list(s.cell) for s in self.plan.specs]
        digest = hashlib.sha256(json.dumps(cells).encode()).hexdigest()
        loopback = []
        for prov in self.plan.paid_providers():
            url = self.config.provider(prov).base_url or ""
            loopback.append(url.startswith(("http://127.0.0.1", "http://localhost")))
        return {
            "plan": self.plan.name,
            "cells_sha256": digest,
            "cells": len(cells),
            "manifest": self.manifest.name,
            "mock": bool(loopback) and all(loopback),
            "inputs": self.inputs,
            "meta": dict(self.plan.meta),
        }

    def _check_plan_file(self) -> None:
        path = self.out / "plan.json"
        ident = self._plan_identity()
        if path.exists():
            old = json.loads(path.read_text("utf-8"))
            if (
                old.get("plan") == ident["plan"]
                and old.get("cells_sha256") != ident["cells_sha256"]
            ):
                raise RuntimeError(
                    f"{path}: plan {ident['plan']!r} has changed since this directory was "
                    "started (different cells); use a new output directory"
                )
            plans = old.get("plans", {})
        else:
            plans = {}
        plans[ident["plan"]] = ident
        doc = {**ident, "plans": plans}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=2) + "\n", "utf-8")
        os.replace(tmp, path)

    # --- the run ---

    def run(self, *, execute: bool = False) -> int:
        now = self.deps.clock()
        journal = (
            Journal.open(self.out / "runs.jsonl") if (self.out / "runs.jsonl").exists() else None
        )
        ledger = (
            Ledger(self.out / "ledger.jsonl", clock=self.deps.clock)
            if (self.out / "ledger.jsonl").exists()
            else None
        )
        missing = missing_keys(self.plan, self.config, self.deps.environ)
        text = describe_plan(
            self.plan,
            self.catalog,
            self.config,
            ledger=ledger,
            journal=journal,
            environ=self.deps.environ,
            now=now,
        )
        if not execute:
            self.say(text)
            self.say("\nDRY RUN: nothing was run and nothing was spent. Add --execute to run.")
            for m in missing:
                self.say(f"  a real run would stop here: {m}")
            return 0
        if missing:
            self.say(text)
            for m in missing:
                self.say(f"replay: {m}")
            if "deepseek" in self.plan.paid_providers() and any(
                m.startswith("deepseek") for m in missing
            ):
                self.say(
                    "replay: DeepSeek configurations are required (decision D3) and no DeepSeek "
                    "key is configured. Ask the maintainer for the key; nothing was run."
                )
            raise MissingKey("; ".join(missing))
        for prov in self.plan.paid_providers():
            if self.config.cap(prov) is None:
                raise MissingKey(f"paid provider {prov!r} has no monthly cap in [caps]")
        with OutputLock(self.out):
            return self._run_locked(text)

    def _run_locked(self, text: str) -> int:
        self._check_plan_file()
        self.ledger = Ledger(self.out / "ledger.jsonl", clock=self.deps.clock)
        # Always present after a real run: an empty ledger is the record that no paid call
        # was made (a local-only run used to leave no ledger at all).
        self.ledger.path.touch(exist_ok=True)
        self.journal = Journal.open(self.out / "runs.jsonl")
        provs = self.plan.paid_providers()
        self.guard = CapGuard(
            self.ledger,
            {p: c for p in provs if (c := self.config.cap(p)) is not None},
            {p: b for p in provs if (b := self.config.total(p)) is not None},
        )
        spent = {a.run_id: self.ledger.run_spent(a.run_id) for a in self.journal.in_flight()}
        for a in self.journal.close_interrupted(spent):
            self.say(
                f"resume: run {a.run_id} ({a.cell[1]} {a.cell[2]} on {a.cell[0]}) was in flight "
                f"when the runner stopped; its ${spent.get(a.run_id, 0.0):.4f} is in the ledger, "
                "and the cell is run again"
            )
        if self.ledger.open_intents():
            n = len(self.ledger.open_intents())
            self.say(f"resume: {n} unsettled call(s) in the ledger are charged at their worst case")
        if self.ledger.skipped_lines:
            self.say(
                f"ledger: {self.ledger.skipped_lines} unreadable line(s) skipped (torn writes)"
            )
        self.say(text)
        self.say("")
        self.say("REAL RUN: starting. Ctrl-C stops after the calls in flight; resume continues.")
        pending = self._pending()
        local = [s for s in pending if s.local]
        paid = [s for s in pending if not s.local]
        threads = [threading.Thread(target=self._local_lane, args=(local,), daemon=True)]
        threads.append(threading.Thread(target=self._paid_lane, args=(paid,), daemon=True))
        for t in threads:
            t.start()
        try:
            for t in threads:
                while t.is_alive():
                    t.join(0.2)
        except KeyboardInterrupt:
            self.stop.set()
            self.say("replay: stopping; waiting for runs in flight to record themselves")
            for t in threads:
                t.join()
            self._finish()
            return EXIT_INTERRUPTED
        self._finish()
        if self.fatal:
            self.say(f"replay: STOPPED: {self.fatal}")
            return EXIT_FATAL
        left = self._pending()
        if left:
            for prov, why in sorted(self.halted.items()):
                self.say(f"replay: {prov} stopped: {why}")
            assert self.journal is not None
            cells = self.journal.by_cell()
            by_prov: dict[str, int] = {}
            deferred = 0
            for s in left:
                by_prov[s.provider] = by_prov.get(s.provider, 0) + 1
                att = cells.get(s.cell) or []
                deferred += bool(att) and att[-1].status in DEFERRED
            parts = ", ".join(f"{p} {n}" for p, n in sorted(by_prov.items()))
            self.say(
                f"replay: {len(left)} cell(s) pending ({parts}); {deferred} of them stopped early "
                "for budget or balance. Pending cells are not failures and are not in the grid; "
                "`steproute replay resume` continues"
            )
            return EXIT_CAP
        return 0

    def _finish(self) -> None:
        assert self.journal is not None
        pool, held = write_grids(
            self.out,
            self.journal,
            self.catalog,
            self.manifest,
            max_retries=self.config.limits.max_retries,
            header={"plan": self.plan.name, "manifest": self.manifest.name},
        )
        if self.deps.local_models is not None:
            self.deps.local_models.unload_all()
        self.say(f"replay: wrote {pool} run(s) to grid.jsonl and {held} to grid-holdout.jsonl")
        if self.ledger is not None:
            for (prov, month), usd in sorted(self.ledger.by_month().items()):
                self.say(f"replay: ledger {prov} {month}: {_fmt(usd)}")
            for (prov, period), (billed, listed) in sorted(self.ledger.by_period().items()):
                if period in ("peak", "offpeak"):
                    self.say(
                        f"replay: ledger {prov} {period} calls: billed {_fmt(billed)} "
                        f"(at peak list price {_fmt(listed)})"
                    )

    def _pending(self) -> list[RunSpec]:
        assert self.journal is not None
        cells = self.journal.by_cell()
        out = []
        reduced = self._gate_cut()
        for s in self.plan.specs:
            attempts = cells.get(s.cell)
            if attempts and _done(attempts, self.config.limits.max_retries):
                continue
            if reduced is not None and self._cut_by_gate(s, reduced):
                continue
            out.append(s)
        return out

    def _cut_by_gate(self, spec: RunSpec, reduced: set[str]) -> bool:
        """Whether a gate that cut drops this cell. Only the gate's own provider is cut: the
        gate protects that provider's cap, and every other provider has its own cap or budget."""
        gate = self.plan.gate
        return (
            gate is not None
            and spec.stage >= 2
            and spec.provider == gate.provider
            and spec.task_id not in reduced
        )

    # --- the pilot's stage gate ---

    def _gate_cut(self) -> set[str] | None:
        path = self.out / "gate.json"
        if not path.exists():
            return None
        g = json.loads(path.read_text("utf-8"))
        return set(g["tasks"]) if g.get("cut") else None

    def _evaluate_gate(self) -> None:
        gate = self.plan.gate
        assert gate is not None and self.journal is not None and self.ledger is not None
        path = self.out / "gate.json"
        if path.exists():
            return
        cells = self.journal.by_cell()
        stage1 = [s for s in self.plan.specs if s.stage == 1 and s.provider == gate.provider]
        rest = [s for s in self.plan.specs if s.stage >= 2 and s.provider == gate.provider]
        real = est = 0.0
        unfinished = 0
        for s in stage1:
            attempts = cells.get(s.cell)
            if not attempts or not _done(attempts, self.config.limits.max_retries):
                unfinished += 1
                continue
            real += float((attempts[-1].end or {}).get("cost_usd") or 0.0)
            est += s.est_usd[0]
        remaining = sum(s.est_usd[0] for s in rest)
        mtd = self.ledger.spent(gate.provider)
        r = real / est if est > 0 else None
        if unfinished or r is None:
            cut, why = True, f"{unfinished} stage-1 run(s) did not finish; cutting to be safe"
        else:
            projected = mtd + r * remaining
            cut = projected > gate.threshold_usd
            why = (
                f"r = {r:.2f}; month-to-date {_fmt(mtd)} + r x {_fmt(remaining)} = "
                f"{_fmt(projected)} {'>' if cut else '<='} {_fmt(gate.threshold_usd)}"
            )
        doc = {"cut": cut, "reason": why, "r": r, "tasks": list(gate.reduced_tasks)}
        path.write_text(json.dumps(doc, indent=2) + "\n", "utf-8")
        verdict = f"CUT stage 2 {gate.provider} cells to the 10-task subset" if cut else "pass"
        self.say(f"stage gate: {verdict} ({why})")

    # --- lanes ---

    def _local_lane(self, specs: list[RunSpec]) -> None:
        models = self.deps.local_models
        if models is None:
            url = self.config.provider("ollama").base_url
            models = OllamaModels(url) if self.config.manage_ollama and url else NoLocalModels()
            self.deps.local_models = models
        for s in specs:
            while not self.stop.is_set():
                models.load(s.model)
                if not self._retry(s, self._run_cell(s)):
                    break

    def _paid_lane(self, specs: list[RunSpec]) -> None:
        stages = sorted({s.stage for s in specs})
        for stage in stages:
            if self.stop.is_set():
                return
            if stage >= 2 and self.plan.gate is not None:
                self._evaluate_gate()
                cut = self._gate_cut()
                if cut is not None:
                    specs = [s for s in specs if not self._cut_by_gate(s, cut)]
            lane = _Lane([s for s in specs if s.stage == stage])
            workers = [
                threading.Thread(target=self._paid_worker, args=(lane,), daemon=True)
                for _ in range(self.config.paid_concurrency)
            ]
            for w in workers:
                w.start()
            for w in workers:
                w.join()

    # --- off-peak schedule ---

    def _hours(self, spec: RunSpec) -> PeakHours | None:
        """The model's peak hours when its provider runs off-peak only, else ``None``."""
        if spec.local or not self.config.schedule(spec.provider).offpeak_only:
            return None
        return peak_hours(self.catalog.get(spec.model))

    def _gate(self, spec: RunSpec, now: float) -> float | None:
        """``None`` when the run may start now; otherwise when its next off-peak stretch of at
        least the start margin begins."""
        hours = self._hours(spec)
        if hours is None:
            return None
        margin = self.config.start_margin(spec.provider)
        left = hours.offpeak_left(now)
        if left > 0 and left >= margin:
            return None
        return hours.next_window(now, max(margin, 1.0))[0]

    def _announce(self, provider: str, until: float, others: bool) -> None:
        key = (provider, until)
        if key in self._announced:
            return
        self._announced.add(key)
        margin = self.config.start_margin(provider)
        tail = (
            "; other providers' runs continue meanwhile"
            if others
            else "; nothing else to run, waiting"
        )
        self.say(
            f"schedule: {provider} runs only off-peak with at least {margin / 3600:g} h of "
            f"off-peak ahead; the next start is {fmt_instant(until)}{tail}"
        )

    def _pick(self, lane: _Lane) -> tuple[RunSpec, str, float] | float | None:
        """Under the lock: the next spec this worker may start (holding its reservation), or
        when to try again (a float: an instant to sleep until; ``-1.0``: wait for a run in
        flight), or ``None`` when nothing is left. Within a provider, specs keep plan order: a
        provider whose next spec cannot start is passed over as a whole, and other providers'
        specs are taken meanwhile."""
        assert self.guard is not None
        now = self.deps.clock()
        passed: set[str] = set()
        gated: dict[str, float] = {}
        for s in list(lane.specs):
            if self.stop.is_set():
                return None
            if s.provider in self.halted or s.provider in passed:
                continue
            try:
                opens = self._gate(s, now)
            except PeakHoursError as exc:
                self.halted[s.provider] = f"schedule: {exc}"
                self.say(f"schedule: {s.provider}: {exc}; its runs are left pending")
                continue
            if opens is not None:
                gated[s.provider] = opens
                passed.add(s.provider)
                continue
            bound = run_bound(s, self.catalog, self.config)
            wc = worst_call(s, self.catalog, self.config)
            run_id = uuid.uuid4().hex[:16]
            try:
                least = None if s.local else self.config.limits.min_run_budget_usd + wc
                grant = self.guard.admit(s.provider, run_id, bound, least=least)
            except CapExceeded as exc:
                if self._in_flight.get(s.provider, 0) > 0:
                    passed.add(s.provider)  # a run in flight may free its reservation
                    continue
                self.halted[s.provider] = f"budget: {exc}"
                self.say(f"cap: {exc}; no further {s.provider} runs now (they stay pending)")
                continue
            lane.specs.remove(s)
            self._in_flight[s.provider] = self._in_flight.get(s.provider, 0) + 1
            for prov, until in gated.items():
                self._announce(prov, until, others=True)
            if grant < bound:
                self.say(
                    f"budget: {s.provider} has {_fmt(grant)} of headroom left, less than a full "
                    f"run's {_fmt(bound)}; {s.model} {s.effort} on {s.task_id} runs with a "
                    f"budget of {_fmt(grant - wc)} and stops as budget_stop if it reaches it"
                )
            # A full grant is the per-run cap exactly (no float residue from the subtraction).
            return s, run_id, grant - wc if grant < bound else self.config.limits.run_cap_usd
        lane.specs[:] = [s for s in lane.specs if s.provider not in self.halted]
        if not lane.specs:
            return None
        busy = any(n > 0 for n in self._in_flight.values())
        for prov, until in gated.items():
            others = busy or any(s.provider not in gated for s in lane.specs)
            self._announce(prov, until, others)
        if gated and not busy and all(s.provider in gated for s in lane.specs):
            return min(gated.values())
        return -1.0

    def _next(self, lane: _Lane) -> tuple[RunSpec, str, float] | None:
        """The next paid spec this worker may start with its budget, holding its reservation,
        or ``None`` when nothing is left. Waits while a refusal could clear when a run in
        flight ends, and sleeps until off-peak when only gated runs are left."""
        while not self.stop.is_set():
            with self._cv:
                got = self._pick(lane)
                if got is None or isinstance(got, tuple):
                    return got
                if got < 0:
                    self._cv.wait(1.0)
                    continue
            until = got
            while not self.stop.is_set() and (now := self.deps.clock()) < until:
                self.deps.sleep(min(PAUSE_CHUNK_S, until - now))
        return None

    def _paid_worker(self, lane: _Lane) -> None:
        while not self.stop.is_set():
            taken = self._next(lane)
            if taken is None:
                return
            spec, run_id, budget = taken
            status = "infra_error"
            try:
                status = self._run_cell(spec, run_id=run_id, budget=budget)
            finally:
                assert self.guard is not None
                with self._cv:
                    self.guard.release(run_id)
                    self._in_flight[spec.provider] -= 1
                    if status == "balance_stop" and spec.provider not in self.halted:
                        self.halted[spec.provider] = (
                            "the provider reported an insufficient balance (HTTP 402); its "
                            "cells stay pending. Add funds, then resume."
                        )
                        self.say(f"replay: {spec.provider} halted: {self.halted[spec.provider]}")
                    if self._retry(spec, status):
                        lane.specs.insert(0, spec)  # next, through the cap check again
                    self._cv.notify_all()

    def _retry(self, spec: RunSpec, status: str) -> bool:
        """Whether to run the cell again now: after a provider or infrastructure error, up to
        ``max_retries`` more attempts, after a pause. An interruption is retried on resume."""
        if status not in ("provider_error", "infra_error") or self.stop.is_set():
            return False
        assert self.journal is not None
        attempts = _retryable(self.journal.by_cell().get(spec.cell, []))
        if attempts > self.config.limits.max_retries:
            return False
        self.deps.sleep(self.config.limits.call_backoff_s * attempts)
        return True

    # --- one cell ---

    def _store(self, spec: RunSpec) -> TraceStore:
        name = "traces-holdout" if spec.holdout else "traces"
        with self._cv:
            if name not in self.stores:
                self.stores[name] = TraceStore(self.out / name, privacy=PrivacyConfig())
            return self.stores[name]

    def _run_cell(
        self, spec: RunSpec, *, run_id: str | None = None, budget: float | None = None
    ) -> str:
        """Run one cell. ``budget``: the run's stop threshold (``None``: the per-run cap)."""
        assert self.journal is not None
        run_id = run_id or uuid.uuid4().hex[:16]
        cap = self.config.limits.run_cap_usd
        budget = None if spec.local or budget is None or budget >= cap else budget
        with self._cv:
            attempts = self.journal.by_cell().get(spec.cell, [])
            start = self.journal.record_start(
                {
                    "run_id": run_id,
                    "cell": {
                        "task_id": spec.task_id,
                        "model": spec.model,
                        "effort": spec.effort,
                        "repeat": spec.repeat,
                    },
                    "attempt": len(attempts) + 1,
                    "session_id": f"replay-{run_id}",
                    "provider": spec.provider,
                    "split": spec.split,
                    "plan": self.plan.name,
                    "bound_usd": round(
                        run_bound(spec, self.catalog, self.config)
                        if budget is None
                        else budget + worst_call(spec, self.catalog, self.config),
                        6,
                    ),
                    "budget_usd": round(cap if budget is None else budget, 6),
                }
            )
        t0 = time.monotonic()
        try:
            end = self._execute(spec, start, budget)
        except ConfigMismatch as exc:
            self.fatal = str(exc)
            self.stop.set()
            end = {"status": "config_mismatch", "success": None, "reason": str(exc)}
        except Exception as exc:  # anything unexpected is infrastructure, never the model
            end = {
                "status": "infra_error",
                "success": None,
                "reason": f"runner error: {type(exc).__name__}: {exc}",
            }
        assert self.ledger is not None
        end.setdefault(
            "cost_usd", round(self.ledger.run_spent(run_id), 6) if not spec.local else 0.0
        )
        end.setdefault("list_cost_usd", end["cost_usd"])
        end.setdefault("latency_ms", round((time.monotonic() - t0) * 1000, 1))
        end["reason"] = self.redact(str(end.get("reason", "")))
        rdir = self.out / "runs" / run_id
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "result.json").write_text(
            self.redact(json.dumps({"run_id": run_id, **start.start, **end}, indent=2, default=str))
            + "\n",
            "utf-8",
        )
        with self._cv:
            self.journal.record_end(run_id, end)
        self.say(
            f"run {run_id}: {spec.model} {spec.effort} on {spec.task_id}: {end['status']} "
            f"(${float(end.get('cost_usd') or 0):.4f}, {end.get('steps', 0)} steps)"
        )
        if spec.provider == "anthropic":
            self._cache_canary()
        return str(end["status"])

    def _cache_canary(self) -> None:
        lim = self.config.limits
        if lim.cache_canary_runs <= 0 or self.journal is None:
            return
        read = total = runs = 0
        for a in self.journal.attempts.values():
            end = a.end or {}
            if a.start.get("provider") != "anthropic" or int(end.get("calls", 0)) < 5:
                continue
            u = end.get("usage") or {}
            runs += 1
            read += int(u.get("cache_read", 0))
            total += (
                int(u.get("input", 0)) + int(u.get("cache_read", 0)) + int(u.get("cache_write", 0))
            )
        if runs >= lim.cache_canary_runs and total > 0 and read / total < lim.min_cache_read_share:
            with self._cv:
                if "anthropic" not in self.halted:
                    self.halted["anthropic"] = (
                        f"cache canary: only {read / total:.0%} of input tokens were read from the "
                        f"cache over {runs} runs (minimum {lim.min_cache_read_share:.0%}); prompt "
                        "caching is not reaching Anthropic, so every run costs about 3 times the "
                        "estimate. Fix caching, then resume."
                    )
                    self.say(f"replay: {self.halted['anthropic']}")

    def _task_text(self, task: Task) -> str:
        if self.deps.task_text is not None:
            return self.deps.task_text(task)
        if task.prompt:
            return task.prompt
        row = self._rows().get(task.id)
        if row is None or not isinstance(row.get("problem_statement"), str):
            raise WorkspaceError(f"task {task.id!r} has no problem statement in [check] rows")
        return str(row["problem_statement"])

    def _rows(self) -> Mapping[str, Mapping[str, Any]]:
        """The SWE-bench dataset rows (problem statements, tests), loaded once."""
        with self._cv:
            if self._rows_cache is None:
                if self.config.rows is None:
                    raise CheckError("[check] rows is not set; run `steproute tasks fetch`")
                self._rows_cache = load_rows(self.config.rows)
            return self._rows_cache

    def _workspace(self, task: Task) -> Workspace:
        if self.deps.workspace is not None:
            return self.deps.workspace(task)
        if self.config.environment == "local" or task.backend == "local":
            return LocalWorkspace(task)
        if not task.image:
            raise WorkspaceError(f"task {task.id!r} has no image")
        return DockerWorkspace(task.image, docker=self.config.docker)

    def _backend(self, task: Task) -> Backend:
        if self.deps.backend is not None:
            return self.deps.backend(task)
        if task.backend == "local":
            return LocalBackend()
        return SwebenchBackend(self._rows(), harness=self.config.harness)

    def _execute(self, spec: RunSpec, start: Attempt, budget: float | None) -> dict[str, Any]:
        assert self.ledger is not None
        run_id = start.run_id
        task = self.manifest.get(spec.task_id)
        rdir = self.out / "runs" / run_id
        rdir.mkdir(parents=True, exist_ok=True)
        scaffold = (
            self.deps.scaffold()
            if self.deps.scaffold
            else BashAgent(
                step_limit=self.config.limits.step_limit,
                command_timeout_s=self.config.limits.command_timeout_s,
                max_observation_chars=self.config.limits.max_observation_chars,
            )
        )
        try:
            text = self._task_text(task)
            ws = self._workspace(task)
        except (WorkspaceError, CheckError, OSError, KeyError) as exc:
            return {"status": "infra_error", "success": None, "reason": f"workspace: {exc}"}
        policy = self.deps.policy(spec, self.catalog) if self.deps.policy else None
        result: AgentResult | None = None
        status = reason = ""
        check: CheckResult | None = None
        try:
            with RunProxy(
                spec,
                catalog=self.catalog,
                store=self._store(spec),
                provider=self.config.provider(spec.provider),
                environ=self.deps.environ,
                policy=policy,
            ) as rp:
                client = ProxyChatClient(
                    spec=spec,
                    proxy=rp,
                    session_id=start.session_id,
                    run_id=run_id,
                    limits=self.config.limits,
                    worst_call=worst_call_usd(
                        self.catalog.get(spec.model), self.config.limits, local=spec.local
                    ),
                    ledger=self.ledger,
                    stop=self.stop,
                    monotonic=self.deps.monotonic,
                    sleep=self.deps.sleep,
                    profile=self.catalog.get(spec.model),
                    hours=peak_hours(self.catalog.get(spec.model)),
                    offpeak_only=self.config.schedule(spec.provider).offpeak_only,
                    call_margin_s=self.config.schedule(spec.provider).call_margin_s,
                    clock=self.deps.clock,
                    budget_usd=budget,
                    notify=self.say,
                )
                try:
                    result = scaffold.run(text, ws, client, nonce=secrets.token_hex(8))
                except RunStopped as stopped:
                    status, reason = stopped.status, stopped.reason
                if result is not None:
                    (rdir / "trajectory.json").write_text(
                        self.redact(json.dumps(result.messages, indent=1)), "utf-8"
                    )
                    try:
                        (rdir / "workspace.diff").write_text(self.redact(ws.diff()), "utf-8")
                    except WorkspaceError:
                        pass
                    if result.exit == "step_limit":
                        status, reason = "step_limit", f"{result.steps} steps"
                    elif result.exit == "no_submission":
                        status, reason = "failed", "no submission (replies without a tool call)"
                    else:
                        patch = result.submission or ""
                        (rdir / "patch.diff").write_text(self.redact(patch), "utf-8")
                        try:
                            check = self._backend(task).check(task, patch)
                        except (CheckError, OSError) as exc:
                            status, reason = "infra_error", f"check: {exc}"
                        else:
                            (rdir / "check.json").write_text(
                                self.redact(json.dumps(check.to_dict(), indent=2)), "utf-8"
                            )
                            if check.tests_passed is None:
                                status, reason = "infra_error", f"check harness: {check.detail}"
                            else:
                                status = "passed" if check.tests_passed else "failed"
                                reason = f"check {check.status}: {check.detail}"
                success = _SUCCESS.get(status)
                signals: dict[str, Any] = {"replay_status": status, "replay_run_id": run_id}
                if check is not None and check.tests_passed is not None:
                    signals.update(check.signals())
                dids = [c.decision_id for c in client.calls if c.decision_id]
                rp.label(dids, signals, success)
                # A call that got an error answer produced no step: never a model outcome.
                # ``retried = False`` overrides the proxy's retry signal, so it stays unlabelled.
                failed = {**signals, "replay_call_failed": True, "retried": False}
                failed.pop("tests_passed", None)
                rp.label(client.failed_decisions, failed, None)
        finally:
            ws.close()
        calls = client.calls
        periods: dict[str, int] = {}
        for c in calls:
            periods[c.period] = periods.get(c.period, 0) + 1
        return {
            "status": status,
            "success": success,
            "reason": reason,
            "steps": result.steps if result else len(calls),
            "calls": len(calls),
            "calls_missing_usage": sum(1 for c in calls if c.flags),
            "usage": {
                "input": sum(c.input_tokens for c in calls),
                "cache_read": sum(c.cache_read_tokens for c in calls),
                "cache_write": sum(c.cache_write_tokens for c in calls),
                "output": sum(c.output_tokens for c in calls),
            },
            "check_status": check.status if check else None,
            "scaffold": scaffold.name,
            # Billed (``cost_usd``, from the ledger) and the same usage at peak list price.
            "list_cost_usd": round(self.ledger.run_list(run_id), 6) if not spec.local else 0.0,
            "calls_by_period": periods,
            "paused_s": round(client.paused_s, 1),
        }
