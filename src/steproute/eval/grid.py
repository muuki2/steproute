"""Full-information evaluation on the replay grid (task level). Standard library only.

Each task was run under every configuration on its own, so the outcome of any policy that
picks one configuration, or a cascade of configurations, *per task* can be read off the grid.
That is all this module evaluates. A policy is asked for its decision once, in the task's
start state (``task_state``), and the decision is applied to the whole task. Nothing here
can see switching inside a task: no run in the grid switches model mid-task.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from steproute.core.schema import Action
from steproute.eval.data import Cell, ConfigKey, GridData, task_state
from steproute.eval.frontier import Point, cost_at, envelope, success_at
from steproute.eval.results import (
    BEATS,
    NOT_SHOWN,
    Estimate,
    EvalSettings,
    Evaluation,
    PolicyResult,
    Verdict,
    decide_verdict,
    estimate_from,
    num,
)
from steproute.eval.stats import (
    cluster_bootstrap,
    derive_seed,
    dot,
    jackknife_counts,
    weighted_quantile,
)
from steproute.eval.targets import OracleTarget, TargetPolicy

__all__ = ["CASCADE_NOTE", "GRID_SCOPE", "TaskOutcome", "evaluate_grid", "simulate_action"]

GRID_SCOPE = (
    "Task-level, full-information evaluation on the replay grid. Each policy's decision at "
    "task start is applied to the whole task, as one configuration or a cascade of "
    "configurations. No run in the grid switches model within a task, so step-level routing, "
    "the switch tax and within-task switching are NOT measured here; they can only be "
    "estimated from logged traces with propensities."
)
CASCADE_NOTE = (
    "Cascades are simulated from independent single-configuration runs: a cascade succeeds if "
    "any step's run succeeded and pays for a step only when every earlier step failed. Each "
    "step's cost is that of a fresh run, so cache reuse between steps (and any benefit of "
    "seeing the earlier failure) is not in the data."
)
LATENCY_NOTE = (
    "Added latency is the time spent in cascade attempts before the one that produced the "
    "final result; routing-decision time is not in the grid and is not included."
)

_POLICY_KEYS = (
    "s", "c", "cps", "margin", "saving", "ret", "sret", "sw", "ch",
    "add50", "add95", "tot50", "tot95", "util",
)  # fmt: skip


@dataclass
class TaskOutcome:
    """Expected outcome of a policy on one task."""

    success: float
    cost: float
    switches: float
    changes: float
    #: (probability, added latency, total latency) per way the task can end; ``None`` if a
    #: run the policy can reach has no latency.
    atoms: list[tuple[float, float, float]] | None


def _key(a: Action) -> ConfigKey:
    return (a.model, a.effort)


def _label(k: ConfigKey) -> str:
    return f"{k[0]} ({k[1]})"


def simulate_action(action: Action, cells: Mapping[ConfigKey, Cell]) -> TaskOutcome | str:
    """Expected outcome of ``action`` on one task from that task's cells, or the reason it
    cannot be scored. Only the cells the cascade can reach are needed: a step after one
    whose runs all succeeded is never paid for, so its absence does not matter."""
    reach = 1.0
    succ = cost = switches = changes = 0.0
    lat_sum = 0.0
    lat_ok = True
    atoms: list[tuple[float, float, float]] = []
    chain = action.chain()
    n_sw = n_ch = 0
    prev: Action | None = None
    for j, step in enumerate(chain):
        cell = cells.get(_key(step))
        if cell is None:
            return f"no run of {step.label()}"
        if cell.costed < cell.runs:
            return f"uncosted run of {step.label()}"
        if cell.labelled < cell.runs:
            return f"unlabelled run of {step.label()}"
        assert cell.success is not None and cell.cost is not None
        if prev is not None:
            n_sw += step.model != prev.model
            n_ch += 1
        s = cell.success
        cost += reach * cell.cost
        p_end = reach * (1.0 if j == len(chain) - 1 else s)
        if cell.latency is None:
            lat_ok = False
        elif p_end > 0:
            atoms.append((p_end, lat_sum, lat_sum + cell.latency))
        switches += p_end * n_sw
        changes += p_end * n_ch
        succ += reach * s
        reach *= 1.0 - s
        lat_sum += cell.latency or 0.0
        prev = step
        if reach <= 0.0:
            break
    return TaskOutcome(succ, cost, switches, changes, atoms if lat_ok else None)


def _combine(parts: Sequence[tuple[float, TaskOutcome]]) -> TaskOutcome:
    atoms: list[tuple[float, float, float]] | None = []
    for p, o in parts:
        if o.atoms is None or atoms is None:
            atoms = None
        else:
            atoms.extend((p * q, a, t) for q, a, t in o.atoms)
    return TaskOutcome(
        sum(p * o.success for p, o in parts),
        sum(p * o.cost for p, o in parts),
        sum(p * o.switches for p, o in parts),
        sum(p * o.changes for p, o in parts),
        atoms,
    )


@dataclass
class _PolicyData:
    target: TargetPolicy
    outs: list[TaskOutcome | None]  # None: the grid cannot score the policy on that task
    reasons: dict[str, int]
    actions: dict[str, float]  # action label -> expected number of tasks


def _policy_data(
    target: TargetPolicy,
    grid: GridData,
    tasks: Sequence[str],
    by_task: Mapping[str, Mapping[ConfigKey, Cell]],
    configs: Sequence[ConfigKey],
) -> _PolicyData:
    outs: list[TaskOutcome | None] = []
    reasons: dict[str, int] = {}
    actions: dict[str, float] = {}

    def fail(reason: str) -> None:
        outs.append(None)
        reasons[reason] = reasons.get(reason, 0) + 1

    for t in tasks:
        cells = by_task.get(t, {})
        dist: tuple[tuple[Action, float], ...]
        if isinstance(target, OracleTarget):
            # Hindsight: the configuration with the highest success on this task, then the
            # cheapest, then label order.
            usable = [k for k in configs if k in cells and cells[k].complete]
            if not usable:
                fail("no configuration")
                continue
            best = min(usable, key=lambda k: (-(cells[k].success or 0.0), cells[k].cost or 0.0, k))
            dist = ((Action(best[0], best[1]), 1.0),)
        else:
            dist = target.distribution(task_state(grid.tasks[t]))
        if not dist:
            fail("the policy has no feasible action")
            continue
        parts: list[tuple[float, TaskOutcome]] = []
        bad = None
        for a, p in dist:
            o = simulate_action(a, cells)
            if isinstance(o, str):
                bad = o
                break
            parts.append((p, o))
        if bad is not None:
            fail(bad)
            continue
        for a, p in dist:
            actions[a.label()] = actions.get(a.label(), 0.0) + p
        outs.append(_combine(parts))
    return _PolicyData(target, outs, reasons, actions)


@dataclass
class _Vecs:
    """Per-task vectors of one policy, over the evaluated tasks."""

    mask: list[float]
    sp: list[float]
    cp: list[float]
    sw: list[float]
    ch: list[float]
    sk: list[float]  # the strongest configuration's success on the policy's tasks
    both: list[float]  # sp * sk
    added: list[tuple[float, int, float]]  # (latency, task index, probability), sorted
    total: list[tuple[float, int, float]]
    masked: dict[ConfigKey, tuple[list[float], list[float]]]  # config -> (success, cost)
    strongest: ConfigKey | None
    supported: int
    latency_tasks: int


@dataclass
class _Prepared:
    tasks: list[str]
    configs: list[ConfigKey]
    s: dict[ConfigKey, list[float]]
    c: dict[ConfigKey, list[float]]
    policies: list[_PolicyData]
    vecs: list[_Vecs]


def _vecs(pd: _PolicyData, configs: Sequence[ConfigKey], s: dict[ConfigKey, list[float]],
          c: dict[ConfigKey, list[float]]) -> _Vecs:  # fmt: skip
    n = len(pd.outs)
    mask = [0.0 if o is None else 1.0 for o in pd.outs]

    def m(v: list[float]) -> list[float]:
        return [a * b for a, b in zip(v, mask, strict=True)]

    masked = {k: (m(s[k]), m(c[k])) for k in configs}
    strongest = None
    if configs and sum(mask) > 0:
        # Highest success on the policy's tasks, then the cheapest, then label order.
        strongest = min(configs, key=lambda k: (-sum(masked[k][0]), sum(masked[k][1]), k))
    sp = [0.0 if o is None else o.success for o in pd.outs]
    sk = masked[strongest][0] if strongest is not None else [0.0] * n
    added: list[tuple[float, int, float]] = []
    total: list[tuple[float, int, float]] = []
    lat_tasks = 0
    for t, o in enumerate(pd.outs):
        if o is None or o.atoms is None:
            continue
        lat_tasks += 1
        for p, a, tt in o.atoms:
            added.append((a, t, p))
            total.append((tt, t, p))
    return _Vecs(
        mask=mask,
        sp=sp,
        cp=[0.0 if o is None else o.cost for o in pd.outs],
        sw=[0.0 if o is None else o.switches for o in pd.outs],
        ch=[0.0 if o is None else o.changes for o in pd.outs],
        sk=sk,
        both=[a * b for a, b in zip(sp, sk, strict=True)],
        added=sorted(added),
        total=sorted(total),
        masked=masked,
        strongest=strongest,
        supported=int(sum(mask)),
        latency_tasks=lat_tasks,
    )


def _prepare(
    grid: GridData,
    tasks: Sequence[str],
    configs: Sequence[ConfigKey],
    policies: Sequence[TargetPolicy],
) -> _Prepared:
    by_task: dict[str, dict[ConfigKey, Cell]] = {}
    for (t, k), cell in grid.cells().items():
        by_task.setdefault(t, {})[k] = cell
    s = {k: [by_task[t][k].success or 0.0 for t in tasks] for k in configs}
    c = {k: [by_task[t][k].cost or 0.0 for t in tasks] for k in configs}
    pdata = [_policy_data(p, grid, tasks, by_task, configs) for p in policies]
    vecs = [_vecs(pd, configs, s, c) for pd in pdata]
    return _Prepared(list(tasks), list(configs), s, c, pdata, vecs)


def _statistic(
    prep: _Prepared, cost_weight: float | None
) -> Callable[[Sequence[int]], dict[str, float]]:
    """Everything estimated, as a function of how often each task appears in a resample.
    With every count 1 it is the point estimate, so points and intervals cannot disagree
    about definitions."""

    def stat(counts: Sequence[int]) -> dict[str, float]:
        out: dict[str, float] = {}
        n = float(sum(counts))
        full_pts = []
        for k in prep.configs:
            ks, kc = dot(counts, prep.s[k]) / n, dot(counts, prep.c[k]) / n
            out[f"cfg|{_label(k)}|s"] = ks
            out[f"cfg|{_label(k)}|c"] = kc
            full_pts.append(Point(_label(k), kc, ks))
        full_env = envelope(full_pts)
        for i, v in enumerate(prep.vecs):
            pre = f"p{i}|"
            nm = n if v.supported == len(v.mask) else dot(counts, v.mask)
            if nm <= 0:
                for key in _POLICY_KEYS:
                    out[pre + key] = math.nan
                continue
            s = dot(counts, v.sp) / nm
            c = dot(counts, v.cp) / nm
            if v.supported == len(v.mask):
                env = full_env  # the policy is scored on every task: same frontier
            else:
                env = envelope(
                    [
                        Point(_label(k), dot(counts, mc) / nm, dot(counts, ms) / nm)
                        for k, (ms, mc) in v.masked.items()
                    ]
                )
            out[pre + "s"] = s
            out[pre + "c"] = c
            out[pre + "cps"] = c / s if s > 0 else math.inf
            out[pre + "margin"] = s - success_at(env, c)
            out[pre + "saving"] = cost_at(env, s) - c
            sk_sum = dot(counts, v.sk)
            out[pre + "ret"] = s * nm / sk_sum if sk_sum > 0 else math.nan
            out[pre + "sret"] = dot(counts, v.both) / sk_sum if sk_sum > 0 else math.nan
            out[pre + "sw"] = dot(counts, v.sw) / nm
            out[pre + "ch"] = dot(counts, v.ch) / nm
            for name, atoms in (("add", v.added), ("tot", v.total)):
                wa = [(x, counts[t] * p) for x, t, p in atoms if counts[t]]
                out[pre + name + "50"] = weighted_quantile(wa, 0.5) if wa else math.nan
                out[pre + name + "95"] = weighted_quantile(wa, 0.95) if wa else math.nan
            out[pre + "util"] = math.nan if cost_weight is None else s - cost_weight * c
        return out

    return stat


def _incomplete(
    grid: GridData, tasks: Sequence[str], configs: Sequence[ConfigKey]
) -> tuple[list[str], dict[str, int], dict[str, dict[str, int]], list[str]]:
    """Tasks complete for every configuration; counts of excluded tasks by reason; the same
    per configuration; tasks excluded only because some run is unlabelled."""
    cells = grid.cells()
    keep: list[str] = []
    only_unlabelled: list[str] = []
    by_reason = {"missing_run": 0, "unlabelled_run": 0, "uncosted_run": 0}
    per_cfg: dict[str, dict[str, int]] = {}
    for t in tasks:
        flags: set[str] = set()
        for k in configs:
            cell = cells.get((t, k))
            reason = None
            if cell is None:
                reason = "missing_run"
            elif cell.costed < cell.runs:
                reason = "uncosted_run"
            elif cell.labelled < cell.runs:
                reason = "unlabelled_run"
            if reason:
                flags.add(reason)
                d = per_cfg.setdefault(_label(k), {})
                d[reason] = d.get(reason, 0) + 1
        for r in flags:
            by_reason[r] += 1
        if not flags:
            keep.append(t)
        elif flags == {"unlabelled_run"}:
            only_unlabelled.append(t)
    return keep, by_reason, per_cfg, only_unlabelled


def evaluate_grid(
    grid: GridData,
    policies: Sequence[TargetPolicy],
    settings: EvalSettings | None = None,
    *,
    split: str | None = None,
    exclude_configs: Sequence[ConfigKey] = (),
) -> Evaluation:
    """Evaluate task-level policies on a grid against every single configuration.

    The headline task set is the tasks on which **every** configuration's runs are present,
    labelled and costed, so every policy and configuration is compared on the same tasks.
    Excluded tasks are counted by reason. Unlabelled runs also feed a sensitivity check
    (counted as all failures, then all successes) that can only weaken a verdict.
    """
    st = settings or EvalSettings()
    all_tasks = sorted(grid.tasks)
    in_split = [t for t in all_tasks if split is None or grid.tasks[t].split == split]
    excluded = set(exclude_configs)
    configs = [k for k in grid.configs() if k not in excluded]
    tasks, by_reason, per_cfg, only_unl = _incomplete(grid, in_split, configs)

    prep = _prepare(grid, tasks, configs, policies)
    stat = _statistic(prep, st.cost_weight)
    n = len(tasks)
    point = stat([1] * n) if n else {}
    samples = cluster_bootstrap(n, st.bootstrap, derive_seed(st.seed, "grid"), stat) if n else []
    jack = [stat(c) for c in jackknife_counts(n)] if n > 2 else []

    # Sensitivity: the tasks left out for unlabelled runs, with those runs counted as
    # failures, then as successes (point estimates only).
    sens: list[tuple[str, dict[str, float]]] = []
    if only_unl:
        ext = sorted([*tasks, *only_unl])
        for label, value in (("unlabelled_as_failure", False), ("unlabelled_as_success", True)):
            p2 = _prepare(grid.with_imputed(value), ext, configs, policies)
            sens.append((label, _statistic(p2, st.cost_weight)([1] * len(ext))))

    def est(key: str, n_: int, note: str | None = None) -> Estimate:
        v = point.get(key)
        if v is None or math.isnan(v):
            return Estimate(None, None, n=n_, note=note)
        return estimate_from(v, samples, key, st.confidence, jack, n=n_, note=note)

    cfg_rows = [
        (_label(k), est(f"cfg|{_label(k)}|s", n), est(f"cfg|{_label(k)}|c", n)) for k in configs
    ]

    results: list[PolicyResult] = []
    for i, pd in enumerate(prep.policies):
        pre = f"p{i}|"
        v = prep.vecs[i]
        sup = v.supported
        unsupported = n - sup
        strongest = None if v.strongest is None else _label(v.strongest)
        metrics: dict[str, Estimate] = {
            "success_rate": est(pre + "s", sup),
            "cost_per_task": est(pre + "c", sup),
            "cost_per_solved": est(
                pre + "cps", sup, "no task solved" if point.get(pre + "s") == 0 else None
            ),
            "quality_retention": est(
                pre + "ret", sup, None if strongest is None else f"versus {strongest}"
            ),
            "strongest_solved_retained": est(pre + "sret", sup),
            "switches_per_task": est(pre + "sw", sup),
            "config_changes_per_task": est(pre + "ch", sup),
            "added_latency_p50_ms": est(pre + "add50", v.latency_tasks),
            "added_latency_p95_ms": est(pre + "add95", v.latency_tasks),
            "total_latency_p50_ms": est(pre + "tot50", v.latency_tasks),
            "total_latency_p95_ms": est(pre + "tot95", v.latency_tasks),
            "frontier_margin": est(pre + "margin", sup),
            "cost_saving_at_equal_success": est(
                pre + "saving",
                sup,
                "above the frontier's top: no single configuration reaches this success rate"
                if point.get(pre + "saving") == math.inf
                else None,
            ),
        }
        if st.cost_weight is not None:
            metrics["utility"] = est(pre + "util", sup)
        gates: list[str] = []
        if n < st.min_clusters:
            gates.append(f"only {n} tasks could be evaluated; at least {st.min_clusters} needed")
        if unsupported:
            why = ", ".join(f"{r} ({c})" for r, c in sorted(pd.reasons.items()))
            gates.append(
                f"the grid cannot score the policy's choice on {unsupported} of {n} tasks: {why}"
            )
        # The configurations' points on this policy's own tasks (all tasks when supported).
        pts = [
            Point(_label(k), sum(mc) / max(sup, 1), sum(ms) / max(sup, 1))
            for k, (ms, mc) in v.masked.items()
        ]
        m = metrics["frontier_margin"]

        def verdict_with(g: list[str], _m: Estimate = m, _pts: list[Point] = pts,
                         _metrics: dict[str, Estimate] = metrics,
                         _pd: _PolicyData = pd) -> Verdict:  # fmt: skip
            return decide_verdict(
                _pd.target.name,
                "task",
                _metrics["cost_per_task"].value,
                _metrics["success_rate"].value,
                _m.value,
                _m.ci,
                g,
                _pts,
                st.confidence,
                hindsight=_pd.target.hindsight,
                min_margin=st.min_margin,
            )

        verdict = verdict_with(gates)
        sens_json: dict[str, object] = {}
        flips: list[str] = []
        for label, sp in sens:
            mv = sp.get(pre + "margin", math.nan)
            sens_json[label] = {"frontier_margin": num(mv), "success_rate": num(sp.get(pre + "s"))}
            if verdict.status != NOT_SHOWN:
                holds = mv > 0 if verdict.status == BEATS else mv <= 0
                if math.isnan(mv) or not holds:
                    flips.append(label.removeprefix("unlabelled_as_"))
        if flips:
            gates.append(
                f"the verdict changes when the {len(only_unl)} tasks with unlabelled runs are "
                f"included with those runs counted as {' or '.join(flips)}s"
            )
            verdict = verdict_with(gates)
        diagnostics: dict[str, object] = {
            "tasks_evaluated": sup,
            "tasks_unsupported": unsupported,
            "unsupported_reasons": dict(sorted(pd.reasons.items())),
            "strongest_configuration": strongest,
            "action_mix": {
                k: num(x, 4) for k, x in sorted(pd.actions.items(), key=lambda kv: -kv[1])
            },
            "latency_tasks": v.latency_tasks,
        }
        if sens_json:
            diagnostics["unlabelled_sensitivity"] = sens_json
        results.append(
            PolicyResult(pd.target.name, pd.target.hindsight, metrics, verdict, diagnostics)
        )

    runs = grid.runs
    notes = [CASCADE_NOTE, LATENCY_NOTE]
    if split is None:
        notes.append(
            "No split was selected, so every task is used. If any policy was tuned on these "
            "tasks its estimate is optimistic; evaluate on a held-out split (--split test)."
        )
    if only_unl:
        notes.append(
            f"{len(only_unl)} tasks have unlabelled runs and are left out of the headline; "
            "each policy's 'unlabelled_sensitivity' gives the margin with them included and "
            "those runs counted as failures, then as successes."
        )
    if any(r.repeat > 0 for r in runs):
        notes.append(
            "Some cells have repeated runs: a cell's success is the mean of its runs, and a "
            "cascade treats repeats of different configurations as independent given the task."
        )
    provenance: dict[str, object] = {
        "source": grid.source,
        "sha256": grid.sha256,
        "header": grid.header,
        "skipped_lines": grid.skipped_lines,
        "runs": len(runs),
        "unlabelled_runs": sum(1 for r in runs if r.success is None),
        "uncosted_runs": sum(1 for r in runs if r.cost_usd is None),
        "runs_without_latency": sum(1 for r in runs if r.latency_ms is None),
        "tasks_total": len(all_tasks),
        "split": split,
        "tasks_in_split": len(in_split),
        "configurations": [_label(k) for k in configs],
        "excluded_configurations": sorted(_label(k) for k in excluded),
        "tasks_evaluated": n,
        "tasks_excluded": len(in_split) - n,
        "tasks_excluded_by_reason": by_reason,
        "incomplete_cells_by_configuration": per_cfg,
    }
    return Evaluation(
        kind="grid",
        level="task",
        scope=GRID_SCOPE,
        provenance=provenance,
        configs=cfg_rows,
        frontier_excluded={},
        policies=results,
        notes=notes,
        settings=st,
    )
