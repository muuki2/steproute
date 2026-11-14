"""The datasheet's summary tables, computed from the release's rows. Standard library only.

The builder renders them from the rows it is about to write; the loader renders them again
from the rows it reads back from Parquet. The two must agree to the character, which is what
the loader test checks.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from steproute.dataset.splits import UNSTRATIFIED, all_splits

__all__ = ["BEGIN", "END", "extract_block", "render_block", "run_table", "task_table"]

Row = Mapping[str, Any]

BEGIN = "<!-- summary:begin -->"
END = "<!-- summary:end -->"


def _md(header: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _rate(passed: int, labelled: int) -> str:
    return f"{passed / labelled:.3f}" if labelled else "n/a"


def run_table(runs: Sequence[Row], steps: Sequence[Row]) -> str:
    """Per split, model and effort: tasks, runs, labelled runs, passed runs, success rate and
    steps. A run is labelled when the task check gave a result (``success`` is not null)."""
    order = {s: i for i, s in enumerate(all_splits())}
    tasks: dict[tuple[str, str, str], set[str]] = {}
    n_runs: Counter[tuple[str, str, str]] = Counter()
    labelled: Counter[tuple[str, str, str]] = Counter()
    passed: Counter[tuple[str, str, str]] = Counter()
    for r in runs:
        k = (r["split"], r["model"], r["effort"])
        tasks.setdefault(k, set()).add(r["task_id"])
        n_runs[k] += 1
        if r["success"] is not None:
            labelled[k] += 1
            passed[k] += int(bool(r["success"]))
    n_steps: Counter[tuple[str, str, str]] = Counter(
        (s["split"], s["model"], s["effort"]) for s in steps
    )
    keys = sorted(set(n_runs) | set(n_steps), key=lambda k: (order.get(k[0], 99), k[1], k[2]))
    body: list[list[str]] = []
    for split in all_splits():
        part = [k for k in keys if k[0] == split]
        if not part:
            continue
        for k in part:
            body.append(
                [
                    k[0],
                    k[1],
                    k[2],
                    str(len(tasks.get(k, ()))),
                    str(n_runs[k]),
                    str(labelled[k]),
                    str(passed[k]),
                    _rate(passed[k], labelled[k]),
                    str(n_steps[k]),
                ]
            )
        all_tasks = set().union(*(tasks.get(k, set()) for k in part))
        body.append(
            [
                split,
                "(all)",
                "",
                str(len(all_tasks)),
                str(sum(n_runs[k] for k in part)),
                str(sum(labelled[k] for k in part)),
                str(sum(passed[k] for k in part)),
                _rate(sum(passed[k] for k in part), sum(labelled[k] for k in part)),
                str(sum(n_steps[k] for k in part)),
            ]
        )
    header = ["split", "model", "effort", "tasks", "runs", "labelled", "passed", "success", "steps"]
    return _md(header, body)


def task_table(tasks: Mapping[str, Row], present: Iterable[str] | None = None) -> str:
    """Tasks per split and stratum. ``tasks`` maps task id to ``{"stratum", "split"}``; with
    ``present`` only those task ids are counted."""
    keep = set(tasks) if present is None else set(present)
    known = ("easy", "medium", "hard", "unsolved")
    found = {str(t["stratum"] or UNSTRATIFIED) for t in tasks.values()}
    strata = [s for s in known if s in found] + sorted(found - set(known))
    counts: Counter[tuple[str, str]] = Counter()
    for tid, t in tasks.items():
        if tid in keep:
            counts[(str(t["split"]), str(t["stratum"] or UNSTRATIFIED))] += 1
    body = []
    for split in ("train", "validation", "test"):
        row = [counts[(split, st)] for st in strata]
        body.append([split, *map(str, row), str(sum(row))])
    return _md(["split", *strata, "total"], body)


def render_block(runs: Sequence[Row], steps: Sequence[Row], tasks: Mapping[str, Row]) -> str:
    """The generated part of the datasheet, between ``BEGIN`` and ``END`` markers."""
    in_data = {r["task_id"] for r in runs}
    parts = [
        BEGIN,
        "",
        "Runs and steps per split, model and effort. `success` is passed runs over labelled "
        "runs (a run the task check did not label is not a failure and is not counted).",
        "",
        run_table(runs, steps),
        "",
        "Tasks per split and stratum, in the fixed split of the whole task set:",
        "",
        task_table(tasks),
        "",
        "Tasks per split and stratum that have at least one run in this release:",
        "",
        task_table(tasks, in_data),
        "",
        END,
    ]
    return "\n".join(parts)


def extract_block(text: str) -> str:
    """The summary block of a datasheet, markers included."""
    start = text.index(BEGIN)
    end = text.index(END, start) + len(END)
    return text[start:end]
