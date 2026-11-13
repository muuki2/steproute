"""Data contracts the evaluator reads. Standard library only. See ``docs/EVALUATION.md``.

Two kinds of data, at two levels, deliberately kept apart:

- **The replay grid** (task level, full information). One line per run of one task under one
  fixed ``(model, effort)`` for the whole task. Issue 18 writes it, issue 19 publishes it.
  ``load_grid`` reads it; ``write_grid`` writes it.
- **Logged traces** (step level, bandit feedback). Decisions and outcomes from the trace store
  (``steproute.core.store``) or a ``steproute traces export`` file. ``load_logged`` reads them.

A grid file is JSON Lines in the trace store's line shape: ``{"type": ..., "v": ..., "data":
{...}}`` with ``type`` one of ``header`` (optional, at most one), ``task`` (optional, one per
task) or ``run`` (one per run).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import warnings
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from steproute.core.catalog import Catalog
from steproute.core.cost import CostModelError, realized_cost
from steproute.core.policy import CandidateSetMismatch, rebuild_candidates
from steproute.core.schema import EFFORTS, Action, Decision, Outcome, StepState
from steproute.core.store import TraceStore

__all__ = [
    "GRID_FORMAT",
    "GRID_VERSION",
    "Cell",
    "DataFormatError",
    "GridData",
    "GridRun",
    "GridTask",
    "LoggedData",
    "LoggedRow",
    "config_key",
    "detect_kind",
    "load_grid",
    "load_logged",
    "logged_from_pairs",
    "task_state",
    "write_grid",
]

GRID_FORMAT = "steproute-grid"
#: Version of the grid line format. Readers accept any ``0.x`` and ignore unknown keys.
GRID_VERSION = "0.1"

ConfigKey = tuple[str, str]


class DataFormatError(ValueError):
    """A data file cannot be read as the format it claims to be."""


def config_key(action: Action) -> ConfigKey:
    """``(model, effort)`` of a single-step action."""
    return (action.model, action.effort)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _opt_number(value: Any, where: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DataFormatError(f"{where} must be a number or null, got {value!r}")
    v = float(value)
    if not (math.isfinite(v) and v >= 0.0):
        raise DataFormatError(f"{where} must be finite and >= 0, got {value!r}")
    return v


def _opt_str(value: Any, where: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise DataFormatError(f"{where} must be a string or null, got {value!r}")
    return value


# --- the grid ----------------------------------------------------------------------------


@dataclass(frozen=True)
class GridRun:
    """One run of one task under one fixed ``(model, effort)`` for the whole task.

    ``success`` is the task check's result; ``None`` means the run has no usable label
    (harness error, timeout before the check): it is not a failure. ``cost_usd`` is the
    realized cost of the whole run; ``None`` means unknown. ``latency_ms`` is the run's
    wall-clock time. ``session_id`` links the run to its per-step traces in the trace store
    (``StepState.session_id``); ``run_id`` is unique per run. ``repeat`` numbers repeated runs
    of the same cell from 0.
    """

    task_id: str
    model: str
    effort: str
    success: bool | None
    cost_usd: float | None
    latency_ms: float | None = None
    run_id: str | None = None
    session_id: str | None = None
    repeat: int = 0
    ts: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def config(self) -> ConfigKey:
        return (self.model, self.effort)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "model": self.model,
            "effort": self.effort,
            "success": self.success,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "repeat": self.repeat,
            "ts": self.ts,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GridRun:
        for key in ("task_id", "model", "effort", "success", "cost_usd"):
            if key not in data:
                raise DataFormatError(f"run is missing required key {key!r}")
        task_id, model, effort = data["task_id"], data["model"], data["effort"]
        if not isinstance(task_id, str) or not task_id:
            raise DataFormatError(f"task_id must be a non-empty string, got {task_id!r}")
        if not isinstance(model, str) or not model:
            raise DataFormatError(f"model must be a non-empty string, got {model!r}")
        if effort not in EFFORTS:
            raise DataFormatError(f"unknown effort {effort!r}; expected one of {EFFORTS}")
        success = data["success"]
        if success is not None and not isinstance(success, bool):
            raise DataFormatError(f"success must be true, false or null, got {success!r}")
        repeat = data.get("repeat", 0)
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 0:
            raise DataFormatError(f"repeat must be an integer >= 0, got {repeat!r}")
        ts = data.get("ts")
        if ts is not None and (isinstance(ts, bool) or not isinstance(ts, int | float)):
            raise DataFormatError(f"ts must be a number or null, got {ts!r}")
        meta = data.get("meta", {})
        if not isinstance(meta, dict):
            raise DataFormatError("meta must be an object")
        return cls(
            task_id=task_id,
            model=model,
            effort=effort,
            success=success,
            cost_usd=_opt_number(data["cost_usd"], "cost_usd"),
            latency_ms=_opt_number(data.get("latency_ms"), "latency_ms"),
            run_id=_opt_str(data.get("run_id"), "run_id"),
            session_id=_opt_str(data.get("session_id"), "session_id"),
            repeat=repeat,
            ts=None if ts is None else float(ts),
            meta=dict(meta),
        )


@dataclass(frozen=True)
class GridTask:
    """Per-task metadata: the split it belongs to and what a router sees at task start.

    ``state`` is the routing state at the start of the task (a ``StepState``); a task-level
    policy is asked for its decision in this state, and that decision is applied to the whole
    task. When absent, ``task_state`` builds a minimal one. ``features`` holds anything a
    later predictor joins on (issue 21 and 24): it is passed through untouched.
    """

    task_id: str
    split: str | None = None
    state: StepState | None = None
    features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "split": self.split,
            "state": None if self.state is None else self.state.to_dict(),
            "features": dict(self.features),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GridTask:
        task_id = data.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise DataFormatError(f"task_id must be a non-empty string, got {task_id!r}")
        state_d = data.get("state")
        state = None
        if state_d is not None:
            if not isinstance(state_d, dict):
                raise DataFormatError("state must be an object or null")
            try:
                state = StepState.from_dict(state_d)
            except (TypeError, ValueError) as exc:
                raise DataFormatError(f"task {task_id}: invalid state: {exc}") from None
        features = data.get("features", {})
        if not isinstance(features, dict):
            raise DataFormatError("features must be an object")
        return cls(
            task_id=task_id,
            split=_opt_str(data.get("split"), "split"),
            state=state,
            features=dict(features),
        )


def task_state(task: GridTask) -> StepState:
    """The routing state a task-level policy is asked to decide in.

    ``task.state`` when present; otherwise a minimal replay state with the task's features
    and ``prompt_tokens`` taken from ``features["prompt_tokens"]`` (0 if absent).
    """
    if task.state is not None:
        return task.state
    prompt = task.features.get("prompt_tokens", 0)
    if isinstance(prompt, bool) or not isinstance(prompt, int) or prompt < 0:
        prompt = 0
    return StepState(
        harness="replay",
        session_id=task.task_id,
        step_idx=0,
        kind="main",
        prompt_tokens=prompt,
        task_id=task.task_id,
        source="replay",
        features=dict(task.features),
    )


@dataclass(frozen=True)
class Cell:
    """All runs of one task under one configuration, summarised.

    ``success`` is the mean of the labelled runs (1.0 or 0.0 for a single run), ``cost`` and
    ``latency`` the means of the runs that report them. ``complete`` means every run is
    labelled and costed, which is what the headline evaluation requires.
    """

    runs: int
    labelled: int
    costed: int
    success: float | None
    cost: float | None
    latency: float | None

    @property
    def complete(self) -> bool:
        return self.runs > 0 and self.labelled == self.runs and self.costed == self.runs


@dataclass
class GridData:
    """A loaded grid. ``tasks`` covers every task id seen in task or run lines."""

    source: str
    runs: tuple[GridRun, ...]
    tasks: dict[str, GridTask]
    header: dict[str, Any] = field(default_factory=dict)
    skipped_lines: int = 0
    sha256: str | None = None

    def configs(self) -> tuple[ConfigKey, ...]:
        """Every ``(model, effort)`` that has at least one run, sorted."""
        return tuple(sorted({r.config for r in self.runs}))

    def cells(self) -> dict[tuple[str, ConfigKey], Cell]:
        groups: dict[tuple[str, ConfigKey], list[GridRun]] = {}
        for r in self.runs:
            groups.setdefault((r.task_id, r.config), []).append(r)
        out: dict[tuple[str, ConfigKey], Cell] = {}
        for key, runs in groups.items():
            lab = [float(r.success) for r in runs if r.success is not None]
            cst = [r.cost_usd for r in runs if r.cost_usd is not None]
            lat = [r.latency_ms for r in runs if r.latency_ms is not None]
            out[key] = Cell(
                runs=len(runs),
                labelled=len(lab),
                costed=len(cst),
                success=sum(lab) / len(lab) if lab else None,
                cost=sum(cst) / len(cst) if cst else None,
                latency=sum(lat) / len(lat) if lat else None,
            )
        return out

    def with_imputed(self, success: bool) -> GridData:
        """A copy in which every unlabelled run is labelled ``success`` (sensitivity only)."""
        runs = tuple(
            r if r.success is not None else dataclasses.replace(r, success=success)
            for r in self.runs
        )
        return GridData(self.source, runs, self.tasks, self.header, self.skipped_lines, self.sha256)


class UnknownGridKeyWarning(UserWarning):
    """A grid line has a key this version does not know; it was ignored."""


_RUN_KEYS = set(GridRun.__dataclass_fields__)
_TASK_KEYS = set(GridTask.__dataclass_fields__)


def _lines(path: Path) -> Iterator[tuple[int, Any]]:
    with open(path, "rb") as f:
        for lineno, raw in enumerate(f, start=1):
            if not raw.strip():
                continue
            try:
                yield lineno, json.loads(raw)
            except ValueError:
                yield lineno, None


def load_grid(path: str | os.PathLike[str]) -> GridData:
    """Read a grid file. Unreadable or invalid lines are skipped, warned about and counted.

    Raises ``DataFormatError`` if the file holds no run at all, if a header names another
    format or a major version this reader does not know, or if a run line is duplicated
    (same task, configuration and repeat).
    """
    p = Path(path)
    runs: list[GridRun] = []
    tasks: dict[str, GridTask] = {}
    header: dict[str, Any] = {}
    skipped = 0
    seen: set[tuple[str, ConfigKey, int]] = set()
    warned: set[str] = set()
    for lineno, obj in _lines(p):
        try:
            if not isinstance(obj, dict) or not isinstance(obj.get("data"), dict):
                raise DataFormatError("not a {type, v, data} object")
            kind, data = obj.get("type"), obj["data"]
            version = str(obj.get("v", GRID_VERSION))
            if not version.startswith("0."):
                raise DataFormatError(f"unsupported grid version {version!r}")
            if kind == "header":
                if data.get("format", GRID_FORMAT) != GRID_FORMAT:
                    raise DataFormatError(f"{p}: not a {GRID_FORMAT} file")
                header = dict(data)
                continue
            known = _RUN_KEYS if kind == "run" else _TASK_KEYS if kind == "task" else None
            if known is None:
                raise DataFormatError(f"unknown line type {kind!r}")
            for key in sorted(set(data) - known - warned):
                warned.add(key)
                warnings.warn(
                    f"{p}: ignoring unknown key {key!r}", UnknownGridKeyWarning, stacklevel=2
                )
            if kind == "task":
                t = GridTask.from_dict(data)
                tasks[t.task_id] = t
                continue
            run = GridRun.from_dict(data)
            key3 = (run.task_id, run.config, run.repeat)
            if key3 in seen:
                raise DataFormatError(
                    f"{p}:{lineno}: duplicate run for task {run.task_id!r}, "
                    f"{run.model} ({run.effort}), repeat {run.repeat}"
                )
            seen.add(key3)
            runs.append(run)
        except DataFormatError as exc:
            if "duplicate run" in str(exc) or "not a steproute-grid" in str(exc):
                raise
            skipped += 1
            warnings.warn(f"{p}:{lineno}: skipped line ({exc})", UserWarning, stacklevel=2)
    if not runs:
        raise DataFormatError(f"{p}: no run lines; is this a grid file?")
    for r in runs:
        tasks.setdefault(r.task_id, GridTask(r.task_id))
    return GridData(str(path), tuple(runs), tasks, header, skipped, _sha256(p))


def write_grid(
    path: str | os.PathLike[str],
    runs: Iterable[GridRun],
    tasks: Iterable[GridTask] = (),
    header: Mapping[str, Any] | None = None,
) -> None:
    """Write a grid file (header, then task lines, then run lines)."""

    def line(kind: str, data: Mapping[str, Any]) -> str:
        return json.dumps({"type": kind, "v": GRID_VERSION, "data": data}, separators=(",", ":"))

    with open(path, "w", encoding="utf-8") as f:
        f.write(line("header", {"format": GRID_FORMAT, **(header or {})}) + "\n")
        for t in tasks:
            f.write(line("task", t.to_dict()) + "\n")
        for r in runs:
            f.write(line("run", r.to_dict()) + "\n")


# --- logged traces -----------------------------------------------------------------------


@dataclass(frozen=True)
class LoggedRow:
    """One logged decision with what the evaluator needs from its outcome.

    ``cluster`` is the resampling unit: the task id when the state has one (sessions of one
    task are not independent), else the session id. ``candidates`` is the logging policy's
    candidate set when it is known (explicit, or rebuilt and verified against a catalog),
    else ``None``. ``cost_source`` is ``reported``, ``tokens`` (priced from the outcome's
    token counts with the catalog) or ``missing``.
    """

    decision: Decision
    outcome: Outcome | None
    cluster: str
    success: bool | None
    cost: float | None
    cost_source: str
    candidates: tuple[Action, ...] | None

    @property
    def state(self) -> StepState:
        return self.decision.state

    @property
    def action(self) -> Action:
        return self.decision.action

    @property
    def propensity(self) -> float:
        return self.decision.propensity


@dataclass
class LoggedData:
    source: str
    rows: tuple[LoggedRow, ...]
    files: tuple[str, ...]
    skipped_lines: int = 0
    orphan_outcomes: int = 0
    candidate_issues: dict[str, int] = field(default_factory=dict)
    sha256: str | None = None


def _read_trace_file(path: Path) -> tuple[list[Decision], dict[str, Outcome], int, int]:
    decisions: dict[str, Decision] = {}
    outcomes: dict[str, Outcome] = {}
    skipped = 0
    for lineno, obj in _lines(path):
        try:
            if not isinstance(obj, dict):
                raise ValueError("not a JSON object")
            kind, data = obj.get("type"), obj.get("data")
            if not isinstance(data, dict):
                raise ValueError("no data object")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if kind == "decision":
                    d = Decision.from_dict(data)
                    decisions.setdefault(d.decision_id, d)
                elif kind == "outcome":
                    o = Outcome.from_dict(data)
                    outcomes[o.decision_id] = o  # the last outcome wins, as in the store
                else:
                    raise ValueError(f"unknown record type {kind!r}")
        except (TypeError, ValueError) as exc:
            skipped += 1
            warnings.warn(f"{path}:{lineno}: skipped line ({exc})", UserWarning, stacklevel=2)
    orphans = sum(1 for k in outcomes if k not in decisions)
    return list(decisions.values()), outcomes, skipped, orphans


def _cost(
    decision: Decision, outcome: Outcome | None, catalog: Catalog | None
) -> tuple[float | None, str]:
    if outcome is None:
        return None, "missing"
    if outcome.cost_usd is not None:
        if math.isfinite(outcome.cost_usd) and outcome.cost_usd >= 0:
            return float(outcome.cost_usd), "reported"
        return None, "missing"
    if catalog is None or outcome.escalated or decision.action.escalate_to is not None:
        return None, "missing"
    model = outcome.final_model or decision.action.model
    if model not in catalog:
        return None, "missing"
    if outcome.input_tokens + outcome.output_tokens + outcome.cache_read_tokens == 0:
        return None, "missing"
    try:
        return realized_cost(outcome, catalog.get(model)), "tokens"
    except CostModelError:
        return None, "missing"


def load_logged(path: str | os.PathLike[str], catalog: Catalog | None = None) -> LoggedData:
    """Read logged decisions and outcomes from a trace store directory or an export file.

    The last outcome per decision wins (the store's rule). A decision without an outcome is
    kept, unlabelled and uncosted, so it is counted rather than silently dropped. Candidate
    sets are rebuilt with ``catalog`` when a decision recorded only a fingerprint; a mismatch
    leaves ``candidates = None`` and is counted in ``candidate_issues``.
    """
    p = Path(path)
    if p.is_dir():
        store = TraceStore(p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            decisions_d, outcomes, _ = store._collect()
        decisions = list(decisions_d.values())
        skipped = store.skipped_lines
        orphans = sum(1 for k in outcomes if k not in decisions_d)
        files = tuple(str(f) for f in store.files())
        digest = None
    else:
        decisions, outcomes, skipped, orphans = _read_trace_file(p)
        files = (str(p),)
        digest = _sha256(p)
    rows, issues = _rows(decisions, outcomes, catalog)
    return LoggedData(str(path), rows, files, skipped, orphans, issues, digest)


def logged_from_pairs(
    pairs: Iterable[tuple[Decision, Outcome | None]],
    catalog: Catalog | None = None,
    source: str = "<memory>",
) -> LoggedData:
    """``LoggedData`` from in-memory decisions and outcomes (tests, generators)."""
    decisions: list[Decision] = []
    outcomes: dict[str, Outcome] = {}
    for d, o in pairs:
        decisions.append(d)
        if o is not None:
            outcomes[d.decision_id] = o
    rows, issues = _rows(decisions, outcomes, catalog)
    return LoggedData(source, rows, (), 0, 0, issues, None)


def _rows(
    decisions: Iterable[Decision], outcomes: Mapping[str, Outcome], catalog: Catalog | None
) -> tuple[tuple[LoggedRow, ...], dict[str, int]]:
    issues: dict[str, int] = {}
    rows: list[LoggedRow] = []
    for d in decisions:
        o = outcomes.get(d.decision_id)
        cands: tuple[Action, ...] | None
        if d.candidate_set is None:
            cands = tuple(d.candidates) if d.candidates else None
            if cands is None:
                issues["no_candidates_logged"] = issues.get("no_candidates_logged", 0) + 1
        elif catalog is None:
            cands = tuple(d.candidates) if d.candidates else None
            if cands is None:
                issues["no_catalog_to_rebuild"] = issues.get("no_catalog_to_rebuild", 0) + 1
        else:
            try:
                cands = rebuild_candidates(d, catalog)
            except CandidateSetMismatch:
                cands = None
                issues["fingerprint_mismatch"] = issues.get("fingerprint_mismatch", 0) + 1
        cost, how = _cost(d, o, catalog)
        cluster = f"task:{d.state.task_id}" if d.state.task_id else f"session:{d.state.session_id}"
        rows.append(
            LoggedRow(
                decision=d,
                outcome=o,
                cluster=cluster,
                success=None if o is None else o.success,
                cost=cost,
                cost_source=how,
                candidates=cands,
            )
        )
    return tuple(rows), issues


def detect_kind(path: str | os.PathLike[str]) -> str:
    """``"grid"`` or ``"traces"``: a directory is a trace store; a file is judged by the
    ``type`` of its first readable line."""
    p = Path(path)
    if p.is_dir():
        return "traces"
    if not p.is_file():
        raise DataFormatError(f"{p}: no such file or directory")
    for _, obj in _lines(p):
        if isinstance(obj, dict):
            kind = obj.get("type")
            if kind in ("header", "task", "run"):
                return "grid"
            if kind in ("decision", "outcome"):
                return "traces"
    raise DataFormatError(f"{p}: cannot tell whether this is a grid or a trace file")
