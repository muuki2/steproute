"""Acceptance criterion 1 of issue 20, on synthetic data with a known ground truth.

- The evaluator ranks policies correctly (grid and logged data), and the known optimal
  policy comes first.
- Intervals cover the true value at the stated level, over many seeded repetitions.

Tolerance, per metric. A bootstrap interval at these sample sizes covers slightly less than
its nominal level: with 800 repetitions at 150 tasks the 95% grid intervals covered 93% for
cost metrics, 94% for the margin and 95% for success (docs/EVALUATION.md, "Coverage"). So
the base is 0.93, nominal minus that measured 2-point shortfall. With R repetitions the
observed coverage is binomial around it with standard error sqrt(0.93 * 0.07 / R), 0.018
for R = 200, and the test accepts 0.93 - 3 SE = 0.876 or more: a procedure that truly
covers 93% fails a metric by chance about once in 700. The mean coverage over all metrics
of one evaluator must also be at least 0.91, which catches a systematic shortfall that a
single loose per-metric bound would let through. Coverage above 0.995 is rejected too
(intervals too wide to be useful). The seeds are fixed, so the test cannot flake; the
bounds were set from the documented shortfall, not from these seeds.

Run ``python tests/test_eval_coverage.py`` to print the coverage tables.
"""

from __future__ import annotations

import math

from steproute.core.schema import Action
from steproute.eval.data import GridData, logged_from_pairs
from steproute.eval.grid import evaluate_grid
from steproute.eval.ope import evaluate_logged
from steproute.eval.results import BEATS, DOES_NOT_BEAT, NOT_SHOWN, EvalSettings
from steproute.eval.synthetic import (
    Truth,
    default_world,
    hint_policy,
    make_grid,
    make_logged,
    optimal_task_policy,
    true_logged_value,
    true_margin,
    true_task_value,
)
from steproute.eval.targets import FixedTarget, TargetPolicy

NOMINAL = 0.95
BASE = 0.93  # nominal minus the measured small-sample shortfall of the bootstrap
REPS = 200
LOWER = BASE - 3 * math.sqrt(BASE * (1 - BASE) / REPS)
POOLED = 0.91
UPPER = 0.995
COST_WEIGHT = 2.0  # success points per USD: the optimal policy maximises s - 2 c

WORLD = default_world()
CFGS = sorted(WORLD.configs, key=lambda c: c.ability)


def _grid(n: int, seed: int) -> GridData:
    runs, tasks = make_grid(WORLD, n, seed)
    return GridData(f"synthetic-{seed}", tuple(runs), {t.task_id: t for t in tasks})


def _config_truth() -> list[tuple[str, Truth]]:
    return [
        (f"{c.model} ({c.effort})", true_task_value(WORLD, FixedTarget(c.action))) for c in CFGS
    ]


def _grid_policies() -> list[TargetPolicy]:
    opt, _ = optimal_task_policy(WORLD, COST_WEIGHT)
    return [
        opt,
        FixedTarget(Action("local-small", "medium", Action("cloud-mid", "medium")), "local>mid"),
        FixedTarget(Action("cloud-small", "high", Action("cloud-big", "high")), "small>big"),
        hint_policy("inverted", lambda h: CFGS[len(CFGS) - 1 - h].action),
        hint_policy("uniform", lambda h: tuple((c.action, 1 / len(CFGS)) for c in CFGS)),
        FixedTarget(CFGS[3].action, "cloud-mid"),
    ]


def grid_coverage(reps: int = REPS, n: int = 150, boot: int = 200) -> dict[tuple[str, str], float]:
    pols = _grid_policies()[:2]
    cfg = _config_truth()
    truth = {}
    for p in pols:
        t = true_task_value(WORLD, p)
        truth[p.name] = {
            "success_rate": t.success,
            "cost_per_task": t.cost,
            "cost_per_solved": t.cost_per_solved,
            "frontier_margin": true_margin(t, cfg),
        }
    hits: dict[tuple[str, str], int] = {}
    for rep in range(reps):
        ev = evaluate_grid(_grid(n, 5000 + rep), pols, EvalSettings(bootstrap=boot, seed=rep))
        for p in ev.policies:
            for m, tv in truth[p.name].items():
                ci = p.metrics[m].ci
                assert ci is not None
                hits[(p.name, m)] = hits.get((p.name, m), 0) + (ci.lo <= tv <= ci.hi)
    return {k: v / reps for k, v in hits.items()}


def _logged_policies() -> list[TargetPolicy]:
    return [
        hint_policy("by-hint", lambda h: CFGS[min(h, 5)].action),
        hint_policy("mix", lambda h: ((CFGS[1].action, 0.5), (CFGS[min(h + 2, 5)].action, 0.5))),
    ]


def logged_coverage(
    reps: int = REPS, n: int = 200, boot: int = 200
) -> dict[tuple[str, str, str], float]:
    pols = _logged_policies()
    truth = {p.name: true_logged_value(WORLD, p) for p in pols}
    hits: dict[tuple[str, str, str], int] = {}
    ess: list[float] = []
    for rep in range(reps):
        data = logged_from_pairs(make_logged(WORLD, n, 1000 + rep))
        st = EvalSettings(bootstrap=boot, seed=rep)
        ev = evaluate_logged(data, pols, st, frontier=False)
        for p in ev.policies:
            ess.append(float(p.diagnostics["ess_success"]))  # type: ignore[arg-type]
            for est, ms in p.estimators.items():
                for m, tv in (("success_rate", truth[p.name].success),
                              ("cost_per_step", truth[p.name].cost)):  # fmt: skip
                    ci = ms[m].ci
                    assert ci is not None
                    k = (p.name, est, m)
                    hits[k] = hits.get(k, 0) + (ci.lo <= tv <= ci.hi)
    # Every repetition is above the headline ESS threshold, so this is coverage where the
    # report would give a headline number.
    assert min(ess) >= EvalSettings().min_ess
    return {k: v / reps for k, v in hits.items()}


def _check(cov: dict[tuple[str, ...], float], groups: list[tuple[str, ...]]) -> None:
    bad = {k: v for k, v in cov.items() if not LOWER <= v <= UPPER}
    assert not bad, f"coverage outside [{LOWER:.3f}, {UPPER}]: {bad}"
    for g in groups:
        vals = [v for k, v in cov.items() if all(part in k for part in g)]
        assert vals and sum(vals) / len(vals) >= POOLED, (g, vals)


def test_grid_intervals_cover_the_truth() -> None:
    cov = grid_coverage()
    assert len(cov) == 8  # 2 policies x 4 metrics
    _check(dict(cov), [()])


def test_off_policy_intervals_cover_the_truth() -> None:
    cov = logged_coverage()
    assert len(cov) == 12  # 2 policies x 3 estimators x 2 metrics
    _check(dict(cov), [("ips",), ("snips",), ("dr",)])


def _order(values: dict[str, float], min_gap: float) -> list[tuple[str, str]]:
    """Pairs (a, b) with a truly better than b by more than ``min_gap``."""
    return [(a, b) for a in values for b in values if values[a] - values[b] > min_gap]


def test_grid_ranks_policies_correctly() -> None:
    pols = _grid_policies()
    cfg = _config_truth()
    st = EvalSettings(bootstrap=100, cost_weight=COST_WEIGHT)
    ev = evaluate_grid(_grid(1500, 11), pols, st)
    est_u = {p.name: p.metrics["utility"].value for p in ev.policies}
    est_m = {p.name: p.metrics["frontier_margin"].value for p in ev.policies}
    true_u, true_m = {}, {}
    for p in pols:
        t = true_task_value(WORLD, p)
        true_u[p.name] = t.success - COST_WEIGHT * t.cost
        true_m[p.name] = true_margin(t, cfg)
    # The known optimum is first by utility, and every clearly ordered pair keeps its order.
    assert ev.ranking()[0]["name"] == "optimal"
    assert max(true_u, key=lambda k: true_u[k]) == "optimal"
    for a, b in _order(true_u, 0.02):
        assert est_u[a] > est_u[b], (a, b)  # type: ignore[operator]
    for a, b in _order(true_m, 0.02):
        assert est_m[a] > est_m[b], (a, b)  # type: ignore[operator]
    status = {p.name: p.verdict.status for p in ev.policies}
    assert status["optimal"] == BEATS
    assert status["inverted"] == DOES_NOT_BEAT
    assert status["uniform"] == DOES_NOT_BEAT  # a random mix of configurations never beats it
    assert status["cloud-mid"] == DOES_NOT_BEAT  # a single configuration ties, at best


def test_off_policy_ranks_policies_correctly() -> None:
    pols = [
        *_logged_policies(),
        FixedTarget(CFGS[3].action, "cloud-mid"),
        hint_policy("inverted", lambda h: CFGS[5 - h].action),
    ]
    st = EvalSettings(bootstrap=100, cost_weight=COST_WEIGHT)
    ev = evaluate_logged(logged_from_pairs(make_logged(WORLD, 1500, 21)), pols, st)
    true_u = {}
    for p in pols:
        t = true_logged_value(WORLD, p)
        true_u[p.name] = t.success - COST_WEIGHT * t.cost
    est_u = {
        p.name: p.estimators["dr"]["success_rate"].value  # type: ignore[operator]
        - COST_WEIGHT * p.estimators["dr"]["cost_per_step"].value  # type: ignore[operator]
        for p in ev.policies
    }
    for a, b in _order(true_u, 0.02):
        assert est_u[a] > est_u[b], (a, b)
    assert ev.ranking()[0]["name"] == max(true_u, key=lambda k: true_u[k])


def test_verdict_examples() -> None:
    """One policy that beats the frontier, one that does not, one where it cannot be shown."""
    opt, _ = optimal_task_policy(WORLD, COST_WEIGHT)
    inverted = hint_policy("inverted", lambda h: CFGS[5 - h].action)
    # A cascade with a real but small advantage (+3.1 points) that 200 tasks cannot resolve.
    small_edge = FixedTarget(Action("local-small", "medium", CFGS[5].action), "local>big-high")
    assert 0.02 < true_margin(true_task_value(WORLD, small_edge), _config_truth()) < 0.04
    ev = evaluate_grid(_grid(200, 3), [opt, inverted, small_edge], EvalSettings(bootstrap=500))
    assert [p.verdict.status for p in ev.policies] == [BEATS, DOES_NOT_BEAT, NOT_SHOWN]
    assert ev.policies[2].verdict.reasons == ("the margin's interval contains 0",)
    # A practical bar: the optimum beats the frontier by far more than 2 points, but not 30.
    bar = evaluate_grid(_grid(200, 3), [opt], EvalSettings(bootstrap=500, min_margin=0.02))
    assert bar.policies[0].verdict.status == BEATS
    assert "by the required 2.0 points" in bar.policies[0].verdict.sentence
    bar = evaluate_grid(_grid(200, 3), [opt], EvalSettings(bootstrap=500, min_margin=0.3))
    assert bar.policies[0].verdict.status == DOES_NOT_BEAT
    # Below min_clusters (100 tasks) no verdict is given at all, however large the margin.
    small = evaluate_grid(_grid(40, 4), [opt], EvalSettings(bootstrap=500))
    assert small.policies[0].verdict.status == NOT_SHOWN
    assert "only 40 tasks" in small.policies[0].verdict.reasons[0]
    for p in [*ev.policies, small.policies[0]]:
        print(p.verdict.sentence)


if __name__ == "__main__":  # pragma: no cover - prints the tables for docs/EVALUATION.md
    print(f"R = {REPS}, accepted [{LOWER:.3f}, {UPPER}]")
    for k, v in sorted(grid_coverage().items()):
        print("grid", *k, f"{v:.3f}")
    for k, v in sorted(logged_coverage().items()):
        print("logged", *k, f"{v:.3f}")
