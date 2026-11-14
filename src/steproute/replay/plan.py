"""Run plans: which (task, model, effort) cells run, in which order, with what estimate.

The named plans are built from the task manifest, the catalog and ``steproute.tasks.estimate``
(the code behind every number in ``docs/TASKSET.md``), so they cannot drift from the
document:

* ``pilot``: the 20 pilot tasks x ``PILOT_EFFORTS`` plus every local configuration, with the
  stage gate of ``docs/TASKSET.md`` section 5.
* ``sparse`` (option B, approved at Gate A): all tasks x every local configuration and each
  paid model at its default effort, plus the other paid efforts on a stratum-balanced sweep
  of 30 tasks (the pilot's 20 first, so finished pilot cells are reused).

Order (issue 18): cheapest first. Local models first (one model at a time), then Haiku, then
Sonnet and DeepSeek, then Opus, then Fable; within a model, lower effort first; within a
configuration, tasks in manifest order (configuration-major, so a cap hit leaves whole
columns of the matrix). Holdout models run in a separate pass and go to a separate split.
Standard library only.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from steproute.core.catalog import Catalog
from steproute.core.schema import effort_rank
from steproute.tasks.estimate import Assumptions, build_plans, estimate_run
from steproute.tasks.manifest import Task, TaskManifest

__all__ = [
    "NAMED_PLANS",
    "ORDER_GROUPS",
    "PlanError",
    "PilotGate",
    "ReplayPlan",
    "RunSpec",
    "build_named_plan",
    "load_plan_file",
    "balanced_counts",
    "order_key",
    "stratum_pick",
]

NAMED_PLANS = ("pilot", "sparse")

#: Cheapest-first groups for paid models (issue 18). A paid model not listed here runs after
#: every listed one, by input price.
ORDER_GROUPS: Mapping[str, int] = {
    "claude-haiku-5-5": 1,
    "claude-sonnet-5-5": 2,
    "deepseek-flash": 2,
    "deepseek-v4-pro": 2,
    "claude-opus-5-5": 3,
    "claude-fable-5-2": 4,
}
_UNLISTED_GROUP = 9


class PlanError(ValueError):
    """The plan cannot be built (unknown task, model or effort)."""


@dataclass(frozen=True)
class RunSpec:
    """One cell to run: a task under one fixed configuration for the whole task."""

    task_id: str
    model: str
    effort: str
    provider: str
    local: bool
    holdout: bool
    repeat: int = 0
    stage: int = 1
    #: Central, low and high estimated cost of one run (``steproute tasks estimate``).
    est_usd: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def cell(self) -> tuple[str, str, str, int]:
        return (self.task_id, self.model, self.effort, self.repeat)

    @property
    def lane(self) -> str:
        return "local" if self.local else "paid"

    @property
    def split(self) -> str:
        return "holdout" if self.holdout else "pool"


@dataclass(frozen=True)
class PilotGate:
    """``docs/TASKSET.md`` section 5, as a check. After every stage-1 run is finished:
    ``r = realised / estimated`` over stage 1's paid runs of ``provider``; if month-to-date
    spend plus ``r`` times the remaining central estimate passes ``threshold_usd``, the
    remaining paid cells are cut to ``reduced_tasks`` (stratum-balanced 3, 3, 3, 1)."""

    provider: str = "anthropic"
    threshold_usd: float = 70.0
    reduced_tasks: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReplayPlan:
    name: str
    description: str
    specs: tuple[RunSpec, ...]
    tasks: tuple[Task, ...]
    gate: PilotGate | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def providers(self) -> tuple[str, ...]:
        return tuple(sorted({s.provider for s in self.specs}))

    def paid_providers(self) -> tuple[str, ...]:
        return tuple(sorted({s.provider for s in self.specs if not s.local}))

    def configs(self) -> tuple[tuple[str, str], ...]:
        seen: dict[tuple[str, str], None] = {}
        for s in self.specs:
            seen.setdefault((s.model, s.effort), None)
        return tuple(seen)


def order_key(spec: RunSpec, catalog: Catalog, task_pos: Mapping[str, int]) -> tuple[Any, ...]:
    """Sort key: holdout pass last; then stage; then the cheapest-first group; then input
    price and model id; then effort (lower first); then task order; then repeat."""
    p = catalog.get(spec.model)
    if spec.local:
        group = 0
    else:
        group = ORDER_GROUPS.get(spec.model, _UNLISTED_GROUP)
    return (
        int(spec.holdout),
        spec.stage,
        group,
        p.tier if spec.local else p.price_in,
        spec.model,
        effort_rank(spec.effort),
        task_pos.get(spec.task_id, 0),
        spec.repeat,
    )


def _estimate(
    catalog: Catalog, assumptions: Assumptions | None, model: str, effort: str
) -> tuple[float, float, float]:
    if assumptions is None:
        return (0.0, 0.0, 0.0)
    p = catalog.get(model)
    return (
        estimate_run(p, effort, assumptions, "central").usd,
        estimate_run(p, effort, assumptions, "low").usd,
        estimate_run(p, effort, assumptions, "high").usd,
    )


def _specs(
    tasks: Sequence[Task],
    configs: Sequence[tuple[str, str]],
    catalog: Catalog,
    assumptions: Assumptions | None,
    *,
    repeats: int = 1,
    stage_of: Mapping[tuple[str, str], int] | None = None,
) -> list[RunSpec]:
    for t in tasks:
        if t.broken is not None:
            raise PlanError(f"task {t.id!r} is marked broken ({t.broken}); a plan cannot run it")
    out = []
    for model, effort in configs:
        if model not in catalog:
            raise PlanError(f"model {model!r} is not in the catalog")
        p = catalog.get(model)
        if effort not in p.efforts:
            raise PlanError(f"{model} has no effort {effort!r} (it has {list(p.efforts)})")
        est = _estimate(catalog, assumptions, model, effort)
        stage = (stage_of or {}).get((model, effort), 1)
        for t in tasks:
            for r in range(repeats):
                out.append(
                    RunSpec(t.id, model, effort, p.provider, p.local, p.holdout, r, stage, est)
                )
    return out


def _ordered(specs: list[RunSpec], catalog: Catalog, tasks: Sequence[Task]) -> tuple[RunSpec, ...]:
    pos = {t.id: i for i, t in enumerate(tasks)}
    seen: dict[tuple[str, str, str, int], RunSpec] = {}
    for s in specs:
        seen.setdefault(s.cell, s)  # a cell in two blocks runs once
    return tuple(sorted(seen.values(), key=lambda s: order_key(s, catalog, pos)))


def stratum_pick(
    tasks: Sequence[Task], counts: Mapping[str, int], *, first: Sequence[Task] = ()
) -> tuple[Task, ...]:
    """``counts[stratum]`` tasks per stratum, taking ``first`` before the rest, each in
    manifest order. Deterministic."""
    chosen: list[Task] = []
    for stratum, n in counts.items():
        pool = [t for t in first if (t.stratum or "?") == stratum]
        pool += [t for t in tasks if (t.stratum or "?") == stratum and t not in pool]
        chosen += pool[:n]
    order = {t.id: i for i, t in enumerate(tasks)}
    return tuple(sorted(chosen, key=lambda t: order.get(t.id, 0)))


def balanced_counts(tasks: Sequence[Task], n: int) -> dict[str, int]:
    """``n`` split over the strata in proportion to their size in ``tasks`` (largest
    remainder). For the 150-task set and ``n = 30``: easy 8, medium 9, hard 10, unsolved 3."""
    sizes: dict[str, int] = {}
    for t in tasks:
        sizes[t.stratum or "?"] = sizes.get(t.stratum or "?", 0) + 1
    total = sum(sizes.values())
    exact = {k: n * v / total for k, v in sizes.items()}
    out = {k: int(x) for k, x in exact.items()}
    for k in sorted(exact, key=lambda k: exact[k] - out[k], reverse=True)[: n - sum(out.values())]:
        out[k] += 1
    return out


def build_named_plan(
    name: str,
    manifest: TaskManifest,
    catalog: Catalog,
    assumptions: Assumptions | None,
) -> ReplayPlan:
    """``pilot`` or ``sparse``, from ``steproute.tasks.estimate.build_plans``."""
    if name not in NAMED_PLANS:
        raise PlanError(f"unknown plan {name!r}; expected one of {NAMED_PLANS} or a plan file")
    all_tasks = tuple(manifest)
    pilot = manifest.pilot()
    est_plans = build_plans(catalog, n_tasks=len(all_tasks), pilot_tasks=len(pilot))
    source = est_plans[name]
    if name == "pilot":
        (block,) = source.blocks
        stage_of = {
            (m, e): 1
            if catalog.get(m).local
            or m == "claude-haiku-5-5"
            or (m, e) == ("claude-sonnet-5-5", "low")
            else 2
            for m, e in block.configs
        }
        specs = _specs(pilot, block.configs, catalog, assumptions, stage_of=stage_of)
        reduced = stratum_pick(pilot, {"easy": 3, "medium": 3, "hard": 3, "unsolved": 1})
        return ReplayPlan(
            "pilot",
            source.description,
            _ordered(specs, catalog, all_tasks),
            pilot,
            PilotGate(reduced_tasks=tuple(t.id for t in reduced)),
            {"manifest": manifest.name},
        )
    specs = []
    sweep_n = next(b.tasks for b in source.blocks if b.tasks != len(all_tasks))
    sweep = stratum_pick(all_tasks, balanced_counts(all_tasks, sweep_n), first=pilot)
    for block in source.blocks:
        if block.tasks == len(all_tasks):
            tasks: Sequence[Task] = all_tasks
        elif block.tasks == len(sweep):
            tasks = sweep
        else:
            raise PlanError(f"sparse block of {block.tasks} tasks does not match the sweep")
        specs += _specs(tasks, block.configs, catalog, assumptions)
    return ReplayPlan(
        "sparse",
        source.description,
        _ordered(specs, catalog, all_tasks),
        all_tasks,
        None,
        {"manifest": manifest.name, "sweep_tasks": [t.id for t in sweep]},
    )


def load_plan_file(
    path: str | Path,
    manifest: TaskManifest,
    catalog: Catalog,
    assumptions: Assumptions | None,
) -> ReplayPlan:
    """A custom plan (tests, ad-hoc batches)::

    name = "smoke"
    [[block]]
    tasks = ["a", "b"]          # or "all" or "pilot"
    configs = [["qwen3-coder:30b", "medium"]]
    repeats = 1
    """
    p = Path(path)
    data = tomllib.loads(p.read_text("utf-8"))
    extra = sorted(set(data) - {"name", "description", "block"})
    if extra:
        raise PlanError(f"{p}: unknown key(s) {extra}")
    specs: list[RunSpec] = []
    used: dict[str, Task] = {}
    for i, b in enumerate(data.get("block", [])):
        bad = sorted(set(b) - {"tasks", "configs", "repeats"})
        if bad:
            raise PlanError(f"{p}: block {i + 1}: unknown key(s) {bad}")
        sel = b.get("tasks", "all")
        if sel == "all":
            tasks = list(manifest)
        elif sel == "pilot":
            tasks = list(manifest.pilot())
        else:
            try:
                tasks = [manifest.get(str(t)) for t in sel]  # a broken one is refused by _specs
            except KeyError as exc:
                raise PlanError(f"{p}: block {i + 1}: unknown task {exc}") from None
        configs = [(str(c[0]), str(c[1])) for c in b.get("configs", [])]
        if not configs:
            raise PlanError(f"{p}: block {i + 1} has no configs")
        repeats = int(b.get("repeats", 1))
        specs += _specs(tasks, configs, catalog, assumptions, repeats=repeats)
        used.update({t.id: t for t in tasks})
    if not specs:
        raise PlanError(f"{p}: the plan has no [[block]]")
    order = [t for t in manifest if t.id in used]
    return ReplayPlan(
        str(data.get("name", p.stem)),
        str(data.get("description", f"plan file {p.name}")),
        _ordered(specs, catalog, order),
        tuple(order),
        None,
        {"manifest": manifest.name, "plan_file": p.name},
    )
