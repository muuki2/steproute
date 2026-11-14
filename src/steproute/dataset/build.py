"""Build a release directory from replay output directories.

All checks run before the first byte is written; the files are written to ``<out>.partial``
and the directory is renamed into place at the end, so a refused or failed build leaves
nothing behind.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from steproute import __version__
from steproute.core.catalog import Catalog
from steproute.core.privacy import (
    ScrubOptions,
    keyed_hash,
    redact_secrets,
    scrub_decision,
    scrub_outcome,
)
from steproute.core.schema import SCHEMA_VERSION
from steproute.core.schema_gen import generate_json_schema_text
from steproute.dataset import tables
from steproute.dataset.datasheet import DatasheetInput, render_datasheet
from steproute.dataset.source import ReleaseRefused, Source, StepRecord, read_source, sha256_file
from steproute.dataset.splits import (
    DEFAULT_FRACTIONS,
    DEFAULT_SEED,
    HOLDOUT_SPLIT,
    all_splits,
    assign_splits,
)
from steproute.eval.data import GRID_VERSION, GridRun
from steproute.tasks.manifest import TaskManifest

__all__ = [
    "RELEASE_FORMAT",
    "RELEASE_VERSION",
    "BuildSummary",
    "build_release",
]

RELEASE_FORMAT = "steproute-dataset"
RELEASE_VERSION = "0.1"

Row = dict[str, Any]


@dataclass(frozen=True)
class BuildSummary:
    out: Path
    runs: int
    steps: int
    files: int


def _bucket(ts: float | None, size: float) -> float | None:
    if ts is None or not math.isfinite(ts):
        return None
    return math.floor(ts / size) * size


def _scalar(v: Any) -> bool:
    return v is None or isinstance(v, bool | int | float)


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, int | float) and not isinstance(v, bool) else None


def _derived_salt(sources: Sequence[Source]) -> str:
    h = hashlib.sha256(b"steproute-release-salt")
    for s in sorted(sources, key=lambda s: s.plan_sha256):
        h.update(s.plan_sha256.encode())
        for name in sorted(s.grid_sha256):
            h.update(s.grid_sha256[name].encode())
    return h.hexdigest()


def _step_row(
    rec: StepRecord,
    run: GridRun,
    meta: Mapping[str, Any],
    opts: ScrubOptions,
    report: dict[str, Any],
) -> Row:
    d, o = rec.decision, rec.outcome
    if d.state.task_text:
        report["task_text_removed"] += 1
    for k, v in d.state.features.items():
        if not _scalar(v):
            report["dropped_features"][k] = report["dropped_features"].get(k, 0) + 1
    sd = scrub_decision(d, opts)
    state = dataclasses.replace(sd.state, task_id=d.state.task_id)  # public benchmark id
    sd = dataclasses.replace(sd, state=state)
    texts = [json.dumps(sd.to_dict(), sort_keys=True)]
    so = None
    if o is not None:
        for k, v in o.signals.items():
            if not _scalar(v):
                report["dropped_signals"][k] = report["dropped_signals"].get(k, 0) + 1
        so = scrub_outcome(o, opts)
        texts.append(json.dumps(so.to_dict(), sort_keys=True))
    for text in texts:
        if sd.state.task_text is not None or redact_secrets(text) != text:
            raise ReleaseRefused(
                f"{rec.where} (record {d.decision_id}): the scrubbed record still fails the "
                "privacy scrubber"
            )
    report["decisions_kept"] += 1
    report["outcomes_kept"] += int(so is not None)
    report["decisions_without_outcome"] += int(so is None)
    return {
        "decision_id": sd.decision_id,
        "run_id": run.run_id,
        "session_id": sd.state.session_id,
        "task_id": run.task_id,
        "stratum": meta["stratum"],
        "task_split": meta["task_split"],
        "split": meta["split"],
        "ts": sd.ts,
        "step_idx": sd.state.step_idx,
        "kind": sd.state.kind,
        "harness": sd.state.harness,
        "source": sd.state.source,
        "prompt_tokens": sd.state.prompt_tokens,
        "cached_prefix_tokens": sd.state.cached_prefix_tokens,
        "sensitive": sd.state.sensitive,
        "tools": list(sd.state.tools),
        "model": sd.action.model,
        "effort": sd.action.effort,
        "policy": sd.policy,
        "propensity": sd.propensity,
        "candidates_json": json.dumps([a.to_dict() for a in sd.candidates], sort_keys=True),
        "features_json": json.dumps(sd.state.features, sort_keys=True),
        "has_outcome": so is not None,
        "success": None if so is None else so.success,
        "final_model": None if so is None else so.final_model,
        "escalated": None if so is None else so.escalated,
        "input_tokens": None if so is None else so.input_tokens,
        "output_tokens": None if so is None else so.output_tokens,
        "cache_read_tokens": None if so is None else so.cache_read_tokens,
        "cache_write_tokens": None if so is None else so.cache_write_tokens,
        "cost_usd": None if so is None else so.cost_usd,
        "latency_ms": None if so is None else so.latency_ms,
        "signals_json": "{}" if so is None else json.dumps(so.signals, sort_keys=True),
    }


def _run_row(run: GridRun, meta: Mapping[str, Any], opts: ScrubOptions) -> Row:
    m = run.meta
    steps = m.get("steps")
    attempts = m.get("attempts")
    return {
        "run_id": run.run_id,
        "session_id": None
        if run.session_id is None
        else keyed_hash(opts.salt, "session:" + run.session_id),
        "task_id": run.task_id,
        "stratum": meta["stratum"],
        "task_split": meta["task_split"],
        "split": meta["split"],
        "model": run.model,
        "effort": run.effort,
        "repeat": run.repeat,
        "success": run.success,
        "status": m.get("status") if isinstance(m.get("status"), str) else None,
        "steps": steps if isinstance(steps, int) and not isinstance(steps, bool) else None,
        "attempts": attempts
        if isinstance(attempts, int) and not isinstance(attempts, bool)
        else None,
        "cost_usd": run.cost_usd,
        "list_cost_usd": _num(m.get("list_cost_usd")),
        "latency_ms": run.latency_ms,
        "ts": _bucket(run.ts, opts.bucket_seconds),
    }


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, "utf-8")


def build_release(
    sources: Sequence[Path],
    out: Path,
    *,
    manifest: TaskManifest,
    catalog: Catalog,
    seed: int = DEFAULT_SEED,
    fractions: Mapping[str, float] = DEFAULT_FRACTIONS,
    salt: str | None = None,
    licence: str = "MIT",
    allow_mock: bool = False,
    bucket_seconds: float = 3600.0,
) -> BuildSummary:
    """Build the release in ``out`` (which must not exist). Raises ``ReleaseRefused``."""
    if not sources:
        raise ReleaseRefused("no run directory given (--from)")
    if len({Path(s).resolve() for s in sources}) != len(sources):
        raise ReleaseRefused("the same run directory was given twice")
    tables.check_available()
    out = Path(out)
    if out.exists():
        raise ReleaseRefused(f"{out}: already exists; a release is never overwritten")
    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        raise ReleaseRefused(f"{partial}: left over from an earlier build; remove it first")

    # --- read and check everything; nothing is written yet ---
    srcs = [
        read_source(Path(p), f"source-{i}", manifest, catalog, allow_mock=allow_mock)
        for i, p in enumerate(sources, start=1)
    ]
    split_of = assign_splits([(t.id, t.stratum) for t in manifest], seed=seed, fractions=fractions)
    task_info = {t.id: {"stratum": t.stratum, "split": split_of[t.id]} for t in manifest}
    opts = ScrubOptions(salt=salt or _derived_salt(srcs), bucket_seconds=bucket_seconds)

    seen_ids: set[str] = set()
    seen_cells: set[tuple[str, str, str, int]] = set()
    run_rows: list[tuple[GridRun, bool, Source]] = []
    for s in srcs:
        for holdout, grid_runs in ((False, s.runs), (True, s.holdout_runs)):
            for r in grid_runs:
                cell = (r.task_id, r.model, r.effort, r.repeat)
                if r.run_id in seen_ids or cell in seen_cells:
                    raise ReleaseRefused(
                        f"{s.label}: run {r.run_id} ({r.task_id}, {r.model} {r.effort}, "
                        f"repeat {r.repeat}) appears in more than one source"
                    )
                seen_ids.add(str(r.run_id))
                seen_cells.add(cell)
                run_rows.append((r, holdout, s))

    report: dict[str, Any] = {
        "decisions_read": 0,
        "decisions_kept": 0,
        "outcomes_read": 0,
        "outcomes_kept": 0,
        "decisions_without_outcome": 0,
        "task_text_removed": 0,
        "dropped_features": {},
        "dropped_signals": {},
        "secrets_redacted": 0,  # a match in a source record refuses the build instead
        "superseded_outcomes": sum(s.superseded_outcomes for s in srcs),
        "unmatched_decisions": sum(s.unmatched_decisions for s in srcs),
        "unmatched_outcomes": sum(s.unmatched_outcomes for s in srcs),
        "duplicate_decisions": sum(s.duplicate_decisions for s in srcs),
        "bucket_seconds": bucket_seconds,
    }
    runs: list[Row] = []
    steps: list[Row] = []
    # (steps are filled below; the read counts are completed after that)
    key = lambda r: (r["task_id"], r["model"], r["effort"], r["repeat"])  # noqa: E731
    for r, holdout, s in run_rows:
        info = task_info[r.task_id]
        meta = {
            "stratum": info["stratum"],
            "task_split": info["split"],
            "split": HOLDOUT_SPLIT if holdout else info["split"],
        }
        runs.append(_run_row(r, meta, opts))
        for rec in s.steps.get(r.session_id or "", []):
            steps.append(_step_row(rec, r, meta, opts, report))
    runs.sort(key=key)
    steps.sort(
        key=lambda r: (
            r["task_id"],
            r["model"],
            r["effort"],
            r["run_id"],
            r["step_idx"],
            r["decision_id"],
        )
    )
    if not runs:
        raise ReleaseRefused("no runs to release")
    report["decisions_read"] = (
        report["decisions_kept"] + report["unmatched_decisions"] + report["duplicate_decisions"]
    )
    report["outcomes_read"] = (
        report["outcomes_kept"] + report["superseded_outcomes"] + report["unmatched_outcomes"]
    )

    sources_doc = [
        {
            "label": s.label,
            "plan_sha256": s.plan_sha256,
            "plans": list(s.plans.values()),
            "grid_sha256": s.grid_sha256,
            "mock": s.mock,
        }
        for s in srcs
    ]
    mock = any(s.mock for s in srcs)
    taskset = {
        k: manifest.meta.get(k)
        for k in ("name", "benchmark", "dataset", "dataset_revision", "split", "seed", "full_size")
        if k in manifest.meta
    }
    taskset["tasks"] = len(manifest)

    # --- write to the partial directory, then rename ---
    try:
        partial.mkdir(parents=True)
        for split in all_splits():
            tables.write_rows(
                partial / "data" / f"steps-{split}.parquet",
                "steps",
                [r for r in steps if r["split"] == split],
            )
        tables.write_rows(partial / "data" / "runs.parquet", "runs", runs)
        _write(partial / "schema" / "trace.schema.json", generate_json_schema_text())
        _write(
            partial / "schema" / "tables.json",
            json.dumps(tables.table_docs(), indent=2) + "\n",
        )
        _write(
            partial / "splits.json",
            json.dumps(
                {
                    "seed": seed,
                    "fractions": dict(fractions),
                    "holdout_split": HOLDOUT_SPLIT,
                    "tasks": task_info,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        _write(
            partial / "DATASHEET.md",
            render_datasheet(
                DatasheetInput(
                    runs=runs,
                    steps=steps,
                    tasks=task_info,
                    catalog=catalog,
                    taskset=taskset,
                    split_seed=seed,
                    fractions=fractions,
                    licence=licence,
                    mock=mock,
                    sources=sources_doc,
                    scrub=report,
                    versions={"steproute": __version__, "schema": SCHEMA_VERSION},
                )
            ),
        )
        files = {
            p.relative_to(partial).as_posix(): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
            for p in sorted(partial.rglob("*"))
            if p.is_file()
        }
        doc = {
            "format": RELEASE_FORMAT,
            "format_version": RELEASE_VERSION,
            "steproute_version": __version__,
            "schema_version": SCHEMA_VERSION,
            "grid_version": GRID_VERSION,
            "taskset": taskset,
            "split": {"seed": seed, "fractions": dict(fractions), "by": "task"},
            "scrub": {
                "bucket_seconds": bucket_seconds,
                "salt": "supplied" if salt else "derived from the source plan and grid checksums",
                "report": report,
            },
            "licence": licence,
            "mock": mock,
            "sources": sources_doc,
            "counts": {
                "runs": len(runs),
                "steps": len(steps),
                "by_split": {
                    s: {
                        "runs": sum(1 for r in runs if r["split"] == s),
                        "steps": sum(1 for r in steps if r["split"] == s),
                    }
                    for s in all_splits()
                },
            },
            "files": files,
        }
        _write(partial / "manifest.json", json.dumps(doc, indent=2, sort_keys=True) + "\n")
        partial.rename(out)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return BuildSummary(out, len(runs), len(steps), len(files) + 1)
