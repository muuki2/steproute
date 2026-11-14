"""Reading and checking one replay output directory. Standard library only.

Everything here refuses (``ReleaseRefused``) instead of skipping: a release is a published
artefact, so a record that cannot be vouched for stops the build. The message names the file,
the line and the record id. Nothing is written by this module.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from steproute.core.catalog import Catalog
from steproute.core.privacy import redact_secrets
from steproute.core.schema import SCHEMA_VERSION, Decision, Outcome, UnknownFieldWarning
from steproute.core.store import TraceStore
from steproute.eval.data import DataFormatError, GridRun, load_grid
from steproute.tasks.manifest import TaskManifest

__all__ = ["ReleaseRefused", "Source", "read_source"]

PLAN_KEYS = ("plan", "cells_sha256", "manifest")


class ReleaseRefused(Exception):
    """The release cannot be built. The message says which file and record, and why."""


def _refuse(where: Path | str, why: str, record: str | None = None) -> ReleaseRefused:
    rec = f" (record {record})" if record else ""
    return ReleaseRefused(f"{where}{rec}: {why}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class StepRecord:
    decision: Decision
    outcome: Outcome | None
    where: str  # "file:line" of the decision
    outcome_where: str | None = None


@dataclass
class Source:
    """One replay output directory, validated."""

    label: str
    plan_sha256: str
    plans: dict[str, dict[str, Any]]
    mock: bool
    grid_sha256: dict[str, str]
    runs: list[GridRun] = field(default_factory=list)
    holdout_runs: list[GridRun] = field(default_factory=list)
    #: session id -> its decisions (joined to the last outcome), pool and holdout stores
    steps: dict[str, list[StepRecord]] = field(default_factory=dict)
    #: counts of what was left out, for the datasheet
    superseded_outcomes: int = 0
    duplicate_decisions: int = 0
    unmatched_decisions: int = 0
    unmatched_outcomes: int = 0


def _read_plan(
    directory: Path, manifest: TaskManifest, allow_mock: bool
) -> tuple[str, dict[str, Any], bool]:
    path = directory / "plan.json"
    if not path.is_file():
        raise _refuse(path, "no plan.json: the plan identity of this run directory is unknown")
    try:
        doc = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError) as exc:
        raise _refuse(path, f"plan.json is unreadable ({exc})") from None
    if not isinstance(doc, dict):
        raise _refuse(path, "plan.json is not a JSON object")
    plans = doc.get("plans")
    entries = [doc] if not isinstance(plans, dict) else [v for v in plans.values()]
    if isinstance(plans, dict) and doc.get("plan") not in plans:
        entries.append(doc)
    identities: dict[str, dict[str, Any]] = {}
    mock = False
    for entry in entries:
        if not isinstance(entry, dict) or not all(
            isinstance(entry.get(k), str) and entry.get(k) for k in PLAN_KEYS
        ):
            raise _refuse(path, f"plan identity is incomplete (needs {', '.join(PLAN_KEYS)})")
        if entry["manifest"] != manifest.name:
            raise _refuse(
                path,
                f"plan {entry['plan']!r} was made for task set {entry['manifest']!r}, "
                f"not {manifest.name!r}",
            )
        mock = mock or bool(entry.get("mock"))
        identities[entry["plan"]] = {
            "plan": entry["plan"],
            "cells_sha256": entry["cells_sha256"],
            "cells": entry.get("cells"),
            "manifest": entry["manifest"],
            "mock": bool(entry.get("mock")),
        }
    if mock and not allow_mock:
        raise _refuse(
            path,
            "the plan ran against a mock upstream (mock = true); a real release must not "
            "contain mock data (use --allow-mock only for fixtures and tests)",
        )
    return sha256_file(path), identities, mock


def _grid(path: Path, holdout: bool, catalog: Catalog, manifest: TaskManifest) -> list[GridRun]:
    if not path.is_file():
        return []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            grid = load_grid(path)
    except (DataFormatError, UserWarning) as exc:
        raise _refuse(path, f"the grid cannot be read ({exc})") from None
    fmt = grid.header.get("split")
    if fmt not in (None, "holdout" if holdout else "pool"):
        raise _refuse(
            path, f"header says split {fmt!r}; expected {'holdout' if holdout else 'pool'}"
        )
    for r in grid.runs:
        who = f"run {r.run_id or '?'}"
        if r.model not in catalog:
            raise _refuse(path, f"model {r.model!r} is not in the catalog", who)
        if catalog.get(r.model).holdout != holdout:
            raise _refuse(
                path,
                f"model {r.model!r} is {'not ' if holdout else ''}the held-out model but "
                f"is in the {'holdout' if holdout else 'pool'} grid",
                who,
            )
        if r.task_id not in manifest:
            raise _refuse(path, f"task {r.task_id!r} is not in task set {manifest.name!r}", who)
        if manifest.get(r.task_id).broken:
            raise _refuse(
                path,
                f"task {r.task_id!r} is marked broken in the task set; its labels are wrong",
                who,
            )
        if not r.run_id:
            raise _refuse(path, "run without a run_id", f"{r.task_id}/{r.model}")
    return list(grid.runs)


def _lines(directory: Path) -> Iterator[tuple[Path, int, dict[str, Any]]]:
    for file in TraceStore(directory).files():
        with open(file, "rb") as f:
            for lineno, raw in enumerate(f, start=1):
                if not raw.strip():
                    continue
                try:
                    obj = json.loads(raw)
                except ValueError:
                    raise _refuse(
                        f"{file}:{lineno}", "unreadable line (a torn write from a crash?)"
                    ) from None
                if not isinstance(obj, dict) or not isinstance(obj.get("data"), dict):
                    raise _refuse(f"{file}:{lineno}", "not a {type, v, data} record")
                if obj.get("v") != SCHEMA_VERSION:
                    raise _refuse(
                        f"{file}:{lineno}",
                        f"unknown schema version {obj.get('v')!r}; "
                        f"this tool reads {SCHEMA_VERSION!r}",
                        str(obj["data"].get("decision_id")),
                    )
                yield file, lineno, obj


def _read_store(
    directory: Path, holdout: bool, catalog: Catalog, src: Source
) -> dict[str, list[tuple[Decision, str]]]:
    """Decisions of one store by session id, with their last outcome attached to ``src``."""
    decisions: dict[str, tuple[Decision, str]] = {}
    outcomes: dict[str, tuple[Outcome, str]] = {}
    for file, lineno, obj in _lines(directory):
        where = f"{file}:{lineno}"
        data = obj["data"]
        rid = str(data.get("decision_id"))
        # Replay output is redacted when it is written; a pattern match here is a leak upstream.
        text = json.dumps(data, sort_keys=True)
        if redact_secrets(text) != text:
            raise _refuse(
                where, "a secret pattern matches the record; it fails the privacy scrubber", rid
            )
        kind = obj.get("type")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", UnknownFieldWarning)
                if kind == "decision":
                    dec = Decision.from_dict(data)
                elif kind == "outcome":
                    out = Outcome.from_dict(data)
                else:
                    raise ValueError(f"unknown record type {kind!r}")
        except (ValueError, TypeError, KeyError, UnknownFieldWarning) as exc:
            raise _refuse(where, f"invalid record ({exc})", rid) from None
        if kind == "decision":
            if dec.state.source != "replay":
                raise _refuse(
                    where,
                    f"not a replay record (source={dec.state.source!r}); only benchmark replays "
                    "are published (decision D5)",
                    dec.decision_id,
                )
            for a in (dec.action, *dec.candidates):
                for step in a.chain():
                    if step.model not in catalog:
                        raise _refuse(where, f"model {step.model!r} is not in the catalog", rid)
            if catalog.get(dec.action.model).holdout != holdout:
                raise _refuse(
                    where,
                    f"model {dec.action.model!r} is in the wrong store "
                    f"({'traces-holdout' if holdout else 'traces'})",
                    rid,
                )
            if dec.decision_id in decisions:
                src.duplicate_decisions += 1
                continue
            decisions[dec.decision_id] = (dec, where)
        else:
            if out.decision_id in outcomes:
                src.superseded_outcomes += 1
            outcomes[out.decision_id] = (out, where)
    for did, (_, where) in outcomes.items():
        if did not in decisions:
            raise _refuse(where, "outcome without a decision: its source is unknown", did)
    by_session: dict[str, list[tuple[Decision, str]]] = {}
    for dec, where in decisions.values():
        by_session.setdefault(dec.state.session_id, []).append((dec, where))
        src.steps.setdefault(dec.state.session_id, [])
    # attach outcomes now, keeping the order of the decisions
    for sid, items in by_session.items():
        for dec, where in items:
            hit = outcomes.get(dec.decision_id)
            src.steps[sid].append(
                StepRecord(dec, hit[0] if hit else None, where, hit[1] if hit else None)
            )
    return by_session


def read_source(
    directory: Path,
    label: str,
    manifest: TaskManifest,
    catalog: Catalog,
    *,
    allow_mock: bool = False,
) -> Source:
    """Read and validate one replay output directory."""
    if not directory.is_dir():
        raise _refuse(directory, "not a directory")
    plan_sha, plans, mock = _read_plan(directory, manifest, allow_mock)
    pool = _grid(directory / "grid.jsonl", False, catalog, manifest)
    held = _grid(directory / "grid-holdout.jsonl", True, catalog, manifest)
    if not pool and not held:
        raise _refuse(directory, "no grid.jsonl or grid-holdout.jsonl: nothing to release")
    src = Source(
        label=label,
        plan_sha256=plan_sha,
        plans=plans,
        mock=mock,
        grid_sha256={
            n: sha256_file(directory / n)
            for n in ("grid.jsonl", "grid-holdout.jsonl")
            if (directory / n).is_file()
        },
        runs=pool,
        holdout_runs=held,
    )
    sessions = {r.session_id: r for r in (*pool, *held) if r.session_id}
    held_ids = {r.run_id for r in held}
    for store, is_holdout in (("traces", False), ("traces-holdout", True)):
        for sid, items in _read_store(directory / store, is_holdout, catalog, src).items():
            run = sessions.get(sid)
            if run is None or (run.run_id in held_ids) != is_holdout:
                # a superseded attempt: its run is not in the grid any more
                src.unmatched_decisions += len(items)
                src.unmatched_outcomes += sum(
                    1 for rec in src.steps.pop(sid, []) if rec.outcome is not None
                )
                continue
            for dec, where in items:
                if dec.state.task_id != run.task_id or (dec.action.model, dec.action.effort) != (
                    run.model,
                    run.effort,
                ):
                    raise _refuse(
                        where,
                        f"the step is for {dec.state.task_id!r} / {dec.action.label()} but its "
                        f"run is {run.task_id!r} / {run.model} ({run.effort})",
                        dec.decision_id,
                    )
    return src
