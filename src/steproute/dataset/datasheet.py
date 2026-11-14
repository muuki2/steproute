"""The datasheet (Markdown), generated from the release's data. Standard library only.

Prose that does not depend on the data is fixed here; every number is computed. The summary
tables sit between markers so the loader can recompute and compare them.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from steproute.core.catalog import Catalog
from steproute.core.labels import DEFAULT_LABEL_CONFIG, SIGNAL_KEYS, TIERS
from steproute.dataset.splits import HOLDOUT_SPLIT
from steproute.dataset.summary import render_block

__all__ = ["DatasheetInput", "render_datasheet"]

_CONFIDENCE = {
    "task_check": DEFAULT_LABEL_CONFIG.confidence_task_check,
    "error": DEFAULT_LABEL_CONFIG.confidence_error,
    "user": DEFAULT_LABEL_CONFIG.confidence_user,
    "retry": DEFAULT_LABEL_CONFIG.confidence_retry,
    "judge": DEFAULT_LABEL_CONFIG.confidence_judge,
}


@dataclass
class DatasheetInput:
    runs: Sequence[Mapping[str, Any]]
    steps: Sequence[Mapping[str, Any]]
    tasks: Mapping[str, Mapping[str, Any]]
    catalog: Catalog
    taskset: Mapping[str, Any]
    split_seed: int
    fractions: Mapping[str, float]
    licence: str
    mock: bool
    sources: Sequence[Mapping[str, Any]]
    scrub: Mapping[str, Any]
    versions: Mapping[str, str] = field(default_factory=dict)


def _day(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%Y-%m-%d")


def _models_table(inp: DatasheetInput) -> str:
    efforts: dict[str, set[str]] = {}
    for r in inp.runs:
        efforts.setdefault(r["model"], set()).add(r["effort"])
    order = ["none", "low", "medium", "high", "xhigh", "max"]
    rows = [
        "| model | provider | where it ran | role | efforts in the data | catalog entry checked |"
    ]
    rows.append("|---|---|---|---|---|---|")
    for m in inp.catalog:
        if m.id not in efforts:
            continue
        eff = ", ".join(sorted(efforts[m.id], key=order.index))
        where = "local (Ollama)" if m.local else "provider API"
        role = "held-out model (own split)" if m.holdout else "pool"
        checked = str(m.meta.get("checked", "unknown"))
        rows.append(f"| `{m.id}` | {m.provider} | {where} | {role} | {eff} | {checked} |")
    return "\n".join(rows)


def _scrub_lines(scrub: Mapping[str, Any]) -> str:
    def keys(d: Mapping[str, int]) -> str:
        return ", ".join(f"`{k}` ({n})" for k, n in sorted(d.items())) or "none"

    return "\n".join(
        [
            f"- Decisions read: {scrub['decisions_read']}; kept: {scrub['decisions_kept']}.",
            f"- Outcomes read: {scrub['outcomes_read']}; kept: {scrub['outcomes_kept']} (the last "
            f"outcome of each decision; {scrub['superseded_outcomes']} earlier outcomes of the "
            "same decision were left out).",
            f"- Left out because they belong to an attempt that is no longer in the run grid "
            f"(a superseded attempt): {scrub['unmatched_decisions']} decisions, "
            f"{scrub['unmatched_outcomes']} outcomes. Duplicate decision lines: "
            f"{scrub['duplicate_decisions']}.",
            f"- Decisions without an outcome: {scrub['decisions_without_outcome']}.",
            f"- Prompt or task text removed: {scrub['task_text_removed']} decisions carried text "
            "(replay traces normally carry none).",
            f"- Feature keys dropped (non-scalar values): {keys(scrub['dropped_features'])}.",
            f"- Signal keys dropped (non-scalar values): {keys(scrub['dropped_signals'])}.",
            f"- Secret patterns found and redacted: {scrub['secrets_redacted']} (a match in a "
            "source record stops the build, so this is zero in a published release).",
            "- Decision ids and session ids replaced by keyed hashes; timestamps rounded down "
            f"to {scrub['bucket_seconds']:g} s buckets. Task ids are kept: they are public "
            "benchmark ids and the join key to the task set.",
        ]
    )


def render_datasheet(inp: DatasheetInput) -> str:
    runs, steps = inp.runs, inp.steps
    ts = [r["ts"] for r in runs if r["ts"] is not None]
    when = f"from {_day(min(ts))} to {_day(max(ts))} (UTC)" if ts else "at an unknown time"
    n_models = len({r["model"] for r in runs})
    unlabelled = sum(1 for r in runs if r["success"] is None)
    ts_name = inp.taskset
    frac = ", ".join(f"{k} {v:g}" for k, v in inp.fractions.items())
    out: list[str] = ["# Datasheet: steproute replay dataset", ""]
    if inp.mock:
        out += [
            "> **MOCK DATA.** Every run in this release came from a mock upstream used to test "
            "the tooling. It is not a measurement of any model.",
            "",
        ]
    out += [
        "Outcome-labelled routing traces from benchmark replays: each task of a fixed task set "
        "was run to completion under one fixed model and effort, and every step of the run was "
        "logged with the task-level result. Built with steproute "
        f"{inp.versions.get('steproute', '?')} (trace schema {inp.versions.get('schema', '?')}). "
        "How to read it: `docs/DATASET.md`.",
        "",
        "## Summary",
        "",
        render_block(runs, steps, inp.tasks),
        "",
        "## Collection method",
        "",
        f"- Task set: `{ts_name.get('name')}` ({ts_name.get('benchmark', 'unknown benchmark')}, "
        f"dataset revision `{ts_name.get('dataset_revision', 'unknown')}`), "
        f"{len(inp.tasks)} working tasks. Only task identifiers are published, not problem "
        "statements, patches or tests.",
        f"- Runs were made {when} with the steproute replay runner: a linear-history "
        "bash agent, one tool, the model and effort fixed for the whole task, calls routed "
        "through the steproute proxy, which logs a decision and an outcome per call.",
        "- The task label comes from the task's own check (the official harness for SWE-bench "
        "tasks) and is copied onto every step of the run.",
        f"- {len(runs)} runs, {len(steps)} steps, {n_models} models, "
        f"{len(inp.sources)} source run director{'y' if len(inp.sources) == 1 else 'ies'}.",
        "- Only records with `source = replay` are included (decision D5). Traces of personal "
        "use are never part of a release, and the build stops if it meets one.",
        "",
        "## Models, versions and dates",
        "",
        _models_table(inp),
        "",
        "The model id is the identifier the provider was called with (a provider alias can move "
        "to a newer snapshot, so the id alone does not pin the weights). The date is when the "
        "catalog entry (prices, limits, effort levels) was last checked, not a release date.",
        "",
        "## Splits",
        "",
        f"Splits are by task: no task is in two splits. Fractions: {frac}; seed "
        f"{inp.split_seed}. Tasks are ordered within each stratum by a keyed hash of the id and "
        "cut in order, so the split depends on the task set only, never on which runs exist. "
        f"`{HOLDOUT_SPLIT}` holds every run of the held-out model on every task (these runs "
        "are never in train, validation or test); their `task_split` column keeps the split "
        "the task has for the other models.",
        "",
        "## Label definitions",
        "",
        "`success` on a step is the task-level result of its run: true if the task check "
        "passed, false if it failed, null if the check gave no result (harness error, "
        f"timeout before the check). {unlabelled} of {len(runs)} runs are unlabelled; null is "
        "not a failure. The general labelling rules of steproute (`docs/LABELS.md`), strongest "
        "tier first, are:",
        "",
        "| tier | placeholder confidence |",
        "|---|---|",
        *[f"| `{t}` | {_CONFIDENCE[t]:g} |" for t in TIERS],
        "",
        "Replay labels are always `task_check:tests_passed`. The confidence values only encode "
        "an order and are not calibrated. Signal keys known to the labeller: "
        + ", ".join(f"`{k}`" for k in SIGNAL_KEYS)
        + ".",
        "",
        "## What scrubbing removed",
        "",
        _scrub_lines(inp.scrub),
        "",
        "## Known biases and limits",
        "",
        "- **One benchmark, one language.** SWE-bench Verified is Python bug-fixing in a few "
        "open-source repositories. Routing behaviour on other work (refactoring, other "
        "languages, long-lived sessions) is not covered. The benchmark is public and may "
        "overlap the training data of some of the models.",
        "- **Strata are estimates.** Difficulty strata come from other models' public results, "
        "not from this data.",
        "- **One scaffold.** All runs use the same minimal bash agent. A model's success here "
        "depends on that scaffold and prompt.",
        "- **The label is per task.** Every step carries the task's result, so a step is "
        "labelled success even if it was a wasted step. Step-level credit is not measured.",
        "- **Small and uneven.** The number of runs per cell is small, usually one repeat, so "
        "success rates per cell have wide intervals. Cells were chosen to fit a budget, not at "
        "random, and not every model ran on every task (see the summary).",
        "- **Local models run at a reduced context** (32K) and may fail by running out of "
        "context rather than by lacking ability.",
        "- **Costs mix prices.** `cost_usd` is as billed (off-peak where it applied); "
        "`list_cost_usd` is the same usage at list price and is the one to compare models on.",
        "- **Time buckets.** Timestamps are rounded to buckets, so within-bucket order is "
        "given only by `run_id` and `step_idx`.",
        "",
        "## Licence and use",
        "",
        f"Released under the {inp.licence} licence. Task identifiers refer to SWE-bench "
        "Verified, whose task data and the upstream repositories have their own licences; none "
        "of that content is included here. Model outputs are not included either.",
        "",
        "## Provenance",
        "",
        "| source | plan | plan identity (cells sha256) | mock |",
        "|---|---|---|---|",
    ]
    for s in inp.sources:
        for p in s["plans"]:
            out.append(
                f"| {s['label']} | `{p['plan']}` | `{p['cells_sha256'][:16]}` | "
                f"{'yes' if p['mock'] else 'no'} |"
            )
    out += ["", "Checksums of every file are in `manifest.json`.", ""]
    return "\n".join(out)
