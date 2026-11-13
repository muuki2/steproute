"""Off-policy evaluation on logged step-level bandit data. Standard library only.

Each logged step is one context (the logged state), one action and the propensity the
logging policy gave it. For a target policy ``pi`` the estimand is the **per-step value in
the logged state distribution**: the expected success and cost of a step routed by ``pi`` in
a state drawn as the logs draw them. States carry the cache owner the *logging* policy left
behind, so this is not the value of running ``pi`` for whole sessions (see
``docs/EVALUATION.md``, "Limitations").

Estimators (``w = pi(a_i | s_i) / mu_i``, optionally capped at ``clip``):

- ``ips``: ``sum(w r) / n``. Unbiased when propensities are right and support holds.
- ``snips``: ``sum(w r) / sum(w)``. Self-normalised; slightly biased, bounded, lower variance.
- ``dr``: ``sum(sum_a pi(a|s) q(s, a) + w (r - q(s, a_i))) / n`` with a reward model ``q``
  fitted out of fold (cross-fitting by cluster). Unbiased if the propensities are right
  whatever ``q`` is; ``q`` only changes the variance. The default headline.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from steproute.core.schema import Action, StepState
from steproute.eval.data import LoggedData, LoggedRow
from steproute.eval.frontier import Point, envelope, success_at
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
    Interval,
    cluster_bootstrap,
    cluster_index,
    cluster_sums,
    derive_seed,
    dot,
    jackknife_counts,
    kish_ess,
)
from steproute.eval.targets import Distribution, FixedTarget, TargetPolicy

__all__ = [
    "CellMeanModel",
    "ESTIMATORS",
    "FittableRewardModel",
    "LOGGED_SCOPE",
    "RewardModel",
    "evaluate_logged",
]

ESTIMATORS = ("ips", "snips", "dr")

LOGGED_SCOPE = (
    "Step-level off-policy evaluation on logged decisions with propensities. Each estimate is "
    "the expected success and cost of ONE step routed by the policy, in the states the logs "
    "contain (whose cache owner was set by the logging policy's earlier choices). It includes "
    "the switch tax of that step, but not the policy's effect on later steps' cache state; "
    "it is not the value of running the policy for whole sessions."
)

_PROPENSITY_RTOL = 1e-6


@runtime_checkable
class RewardModel(Protocol):
    """Predicted ``(success probability, expected cost in USD)`` of ``action`` in ``state``."""

    name: str

    def predict(self, state: StepState, action: Action) -> tuple[float, float]: ...


@runtime_checkable
class FittableRewardModel(Protocol):
    """A reward model the evaluator fits itself, out of fold (cross-fitting)."""

    name: str

    def fit(self, rows: Sequence[LoggedRow]) -> RewardModel: ...


@dataclass
class _FittedCellMean:
    name: str
    prior: float
    by_cell: dict[tuple[Action, str], tuple[float, float, float, float]]
    by_action: dict[Action, tuple[float, float, float, float]]
    glob: tuple[float, float]

    def predict(self, state: StepState, action: Action) -> tuple[float, float]:
        label = action
        gs, gc = self.glob
        a = self.by_action.get(label)
        # Shrink the action mean toward the global mean, then the cell mean toward that.
        if a is None:
            ms, mc = gs, gc
        else:
            ms = (a[0] + self.prior * gs) / (a[1] + self.prior)
            mc = (a[2] + self.prior * gc) / (a[3] + self.prior)
        cell = self.by_cell.get((label, state.kind))
        if cell is None:
            return ms, mc
        return (
            (cell[0] + self.prior * ms) / (cell[1] + self.prior),
            (cell[2] + self.prior * mc) / (cell[3] + self.prior),
        )


@dataclass
class CellMeanModel:
    """Built-in reward model: mean success and cost per (action, step kind), shrunk toward
    the action's mean and then the global mean with ``prior`` pseudo-observations. Crude on
    purpose: doubly robust estimates stay unbiased however wrong it is, so its only job is to
    cut variance where it can."""

    prior: float = 2.0
    name: str = "cell-mean(action, kind)"

    def fit(self, rows: Sequence[LoggedRow]) -> RewardModel:
        cell: dict[tuple[Action, str], list[float]] = {}
        act: dict[Action, list[float]] = {}
        ss = sn = cs = cn = 0.0
        for r in rows:
            label = r.action
            c_ = cell.setdefault((label, r.state.kind), [0.0, 0.0, 0.0, 0.0])
            a_ = act.setdefault(label, [0.0, 0.0, 0.0, 0.0])
            if r.success is not None:
                for acc in (c_, a_):
                    acc[0] += float(r.success)
                    acc[1] += 1
                ss += float(r.success)
                sn += 1
            if r.cost is not None:
                for acc in (c_, a_):
                    acc[2] += r.cost
                    acc[3] += 1
                cs += r.cost
                cn += 1
        glob = (ss / sn if sn else 0.5, cs / cn if cn else 0.0)
        return _FittedCellMean(
            self.name,
            self.prior,
            {k: (v[0], v[1], v[2], v[3]) for k, v in cell.items()},
            {k: (v[0], v[1], v[2], v[3]) for k, v in act.items()},
            glob,
        )


@dataclass
class _Row:
    row: LoggedRow
    cluster: int
    mu: float
    support: frozenset[Action] | None  # actions the logging policy could take; None: unknown
    fold: int


@dataclass
class _Target:
    target: TargetPolicy
    # per valid row
    w: list[float] = field(default_factory=list)
    clipped: list[float] = field(default_factory=list)  # (w - clip)_+
    dm_s: list[float] = field(default_factory=list)
    dm_c: list[float] = field(default_factory=list)
    q_s: list[float] = field(default_factory=list)
    q_c: list[float] = field(default_factory=list)
    unsupported: list[float] = field(default_factory=list)
    unknown: list[float] = field(default_factory=list)
    switch: list[float] = field(default_factory=list)
    infeasible: int = 0


@dataclass(frozen=True)
class _Diag:
    ess_success: float
    ess_cost: float
    unsupported_mass: float
    rows_with_unsupported_mass: int
    unknown_support_mass: float
    rows_with_unknown_support: int
    rows_matching_logged_action: int
    max_weight: float
    target_infeasible_rows: int
    clipped_rows: int
    clipped_weight_mass: float

    def to_json(self) -> dict[str, object]:
        return {k: (num(v, 6) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def _diagnose(t: _Target, rows: Sequence[_Row]) -> _Diag:
    n_rows = len(rows)
    return _Diag(
        ess_success=kish_ess(
            [w for w, x in zip(t.w, rows, strict=True) if x.row.success is not None]
        ),
        ess_cost=kish_ess([w for w, x in zip(t.w, rows, strict=True) if x.row.cost is not None]),
        unsupported_mass=sum(t.unsupported) / n_rows if n_rows else math.nan,
        rows_with_unsupported_mass=sum(1 for u in t.unsupported if u > 1e-12),
        unknown_support_mass=sum(t.unknown) / n_rows if n_rows else math.nan,
        rows_with_unknown_support=sum(1 for u in t.unknown if u > 1e-12),
        rows_matching_logged_action=sum(1 for w in t.w if w > 0),
        max_weight=max(t.w, default=0.0),
        target_infeasible_rows=t.infeasible,
        clipped_rows=sum(1 for o in t.clipped if o > 0),
        clipped_weight_mass=math.fsum(t.clipped),
    )


def _vectors(rows: Sequence[_Row], t: _Target, impute: bool | None) -> dict[str, list[float]]:
    """Per-row terms whose cluster sums give every estimator."""
    v: dict[str, list[float]] = {
        k: [] for k in ("nL", "wr", "wL", "dr", "nC", "wc", "wC", "drc", "sw", "one")
    }
    for i, rr in enumerate(rows):
        r = rr.row.success
        if r is None and impute is not None:
            r = impute
        w = t.w[i]
        if r is None:
            for k in ("nL", "wr", "wL", "dr"):
                v[k].append(0.0)
        else:
            rv = float(r)
            v["nL"].append(1.0)
            v["wr"].append(w * rv)
            v["wL"].append(w)
            v["dr"].append(t.dm_s[i] + w * (rv - t.q_s[i]))
        c = rr.row.cost
        if c is None:
            for k in ("nC", "wc", "wC", "drc"):
                v[k].append(0.0)
        else:
            v["nC"].append(1.0)
            v["wc"].append(w * c)
            v["wC"].append(w)
            v["drc"].append(t.dm_c[i] + w * (c - t.q_c[i]))
        v["sw"].append(t.switch[i])
        v["one"].append(1.0)
    return v


def _ratio(a: float, b: float) -> float:
    return a / b if b > 0 else math.nan


def _estimates(sums: Mapping[str, float]) -> dict[str, float]:
    """Every estimator from the sums of the per-row terms of ``_vectors``."""
    n_lab, n_cost = sums["nL"], sums["nC"]
    out = {
        "ips|s": _ratio(sums["wr"], n_lab),
        "snips|s": _ratio(sums["wr"], sums["wL"]),
        "dr|s": _ratio(sums["dr"], n_lab),
        "ips|c": _ratio(sums["wc"], n_cost),
        "snips|c": _ratio(sums["wc"], sums["wC"]),
        "dr|c": _ratio(sums["drc"], n_cost),
        "sw": _ratio(sums["sw"], sums["one"]),
    }
    for e in ESTIMATORS:
        s, c = out[f"{e}|s"], out[f"{e}|c"]
        out[f"{e}|cps"] = (
            math.nan if math.isnan(s) or math.isnan(c) else (c / s if s > 0 else math.inf)
        )
    return out


def evaluate_logged(
    data: LoggedData,
    policies: Sequence[TargetPolicy],
    settings: EvalSettings | None = None,
    *,
    logging_policy: TargetPolicy | None = None,
    reward_model: RewardModel | FittableRewardModel | None = None,
    frontier: bool = True,
) -> Evaluation:
    """Off-policy estimates of each target policy and of every single configuration seen in
    the logs, with cluster-bootstrap intervals, support and effective-sample-size checks.

    ``logging_policy``, when given, is used to recompute every logged propensity (mismatches
    are reported) and to know the logging policy's full support. Without it, support is
    taken to be the logged candidate set when the propensity is below 1, and the logged
    action alone when it is 1 (a deterministic logger).

    ``reward_model`` is either fitted by the evaluator out of fold (it has ``fit``; default
    ``CellMeanModel()``) or used as given (it has only ``predict``; it must not have been
    trained on these logs).

    ``frontier=False`` skips the single configurations (faster; every verdict is then
    "not shown").
    """
    st = settings or EvalSettings()
    model: RewardModel | FittableRewardModel = reward_model or CellMeanModel()

    # --- validate rows -----------------------------------------------------------------
    valid: list[LoggedRow] = []
    invalid = 0
    for r in data.rows:
        p = r.propensity
        if isinstance(p, bool) or not isinstance(p, int | float) or not (0.0 < p <= 1.0):
            invalid += 1
            continue
        valid.append(r)
    cidx, n_clusters = cluster_index([r.cluster for r in valid])

    mismatches = 0
    rows: list[_Row] = []
    for r, ci in zip(valid, cidx, strict=True):
        support: frozenset[Action] | None
        if logging_policy is not None:
            ldist = dict(logging_policy.distribution(r.state))
            mu = ldist.get(r.action, 0.0)
            if abs(mu - r.propensity) > _PROPENSITY_RTOL * max(1.0, r.propensity):
                mismatches += 1
            support = frozenset(a for a, q in ldist.items() if q > 0)
        elif r.propensity >= 1.0:
            support = frozenset({r.action})
        elif r.candidates is not None:
            support = frozenset(r.candidates) | {r.action}
        else:
            support = None
        fold = derive_seed(st.seed, "fold", r.cluster) % st.folds
        rows.append(_Row(r, ci, float(r.propensity), support, fold))

    # --- reward model, out of fold -------------------------------------------------------
    # Anything with ``fit`` is fitted here, out of fold; a model with only ``predict`` is
    # used as given.
    fittable = isinstance(model, FittableRewardModel)
    fitted: dict[int, RewardModel] = {}
    if fittable:
        assert isinstance(model, FittableRewardModel)
        for f in range(st.folds):
            fitted[f] = model.fit([x.row for x in rows if x.fold != f])
    else:
        assert isinstance(model, RewardModel)

    def predictor(x: _Row) -> RewardModel:
        if fittable:
            return fitted[x.fold]
        assert isinstance(model, RewardModel)
        return model

    # --- targets: policies, then every single configuration ----------------------------
    seen: set[Action] = set()
    for x in rows:
        seen.add(x.row.action)
        seen.update(x.row.candidates or ())
    singles = {(step.model, step.effort) for act in seen for step in act.chain()}
    config_targets = (
        [FixedTarget(Action(m, e), name=f"{m} ({e})") for m, e in sorted(singles)]
        if frontier
        else []
    )
    all_targets = [*policies, *config_targets]

    targets: list[_Target] = []
    for tp in all_targets:
        t = _Target(tp)
        for x in rows:
            dist: Distribution = tp.distribution(x.row.state)
            if not dist:
                t.infeasible += 1
            pred = predictor(x)
            p_logged = 0.0
            dm_s = dm_c = uns = unk = sw = 0.0
            cur = x.row.state.current_model
            for a, p in dist:
                qs, qc = pred.predict(x.row.state, a)
                dm_s += p * qs
                dm_c += p * qc
                if a == x.row.action:
                    p_logged = p
                elif x.support is None:
                    unk += p
                elif a not in x.support:
                    uns += p
                if cur is not None and a.model != cur:
                    sw += p
            if not dist:
                uns = 1.0  # the policy cannot act here; nothing in the logs can stand in
            w = p_logged / x.mu
            over = 0.0
            if st.clip is not None and w > st.clip:
                over = w - st.clip
                w = st.clip
            qs_a, qc_a = pred.predict(x.row.state, x.row.action)
            t.w.append(w)
            t.clipped.append(over)
            t.dm_s.append(dm_s)
            t.dm_c.append(dm_c)
            t.q_s.append(qs_a)
            t.q_c.append(qc_a)
            t.unsupported.append(uns)
            t.unknown.append(unk)
            t.switch.append(sw)
        targets.append(t)

    # --- bootstrap ----------------------------------------------------------------------
    sums_by_target = []
    for t in targets:
        vec = _vectors(rows, t, None)
        sums_by_target.append(
            {k: cluster_sums(v, [x.cluster for x in rows], n_clusters) for k, v in vec.items()}
        )
    n_pol = len(policies)
    head = st.headline

    def stat(counts: Sequence[int]) -> dict[str, float]:
        out: dict[str, float] = {}
        ests = [_estimates({k: dot(counts, v) for k, v in cs.items()}) for cs in sums_by_target]
        cfg_pts = []
        for j, e in enumerate(ests[n_pol:]):
            label = config_targets[j].name
            for k, val in e.items():
                out[f"cfg{j}|{k}"] = val
            if label in included:
                s, c = e[f"{head}|s"], e[f"{head}|c"]
                if not (math.isnan(s) or math.isnan(c)):
                    cfg_pts.append(Point(label, c, s))
        env = envelope(cfg_pts)
        strongest = max(cfg_pts, key=lambda pt: (pt.success, -pt.cost), default=None)
        for i, e in enumerate(ests[:n_pol]):
            for k, val in e.items():
                out[f"p{i}|{k}"] = val
            s, c = e[f"{head}|s"], e[f"{head}|c"]
            out[f"p{i}|margin"] = (
                math.nan if math.isnan(s) or math.isnan(c) else s - success_at(env, c)
            )
            out[f"p{i}|ret"] = (
                _ratio(s, strongest.success)
                if strongest is not None and not math.isnan(s)
                else math.nan
            )
            out[f"p{i}|util"] = (
                math.nan
                if st.cost_weight is None or math.isnan(s) or math.isnan(c)
                else s - st.cost_weight * c
            )
        return out

    # Which configurations may enter the frontier: enough ESS and support.
    included: set[str] = set()
    excluded: dict[str, str] = {}
    diag = [_diagnose(t, rows) for t in targets]
    for j, t in enumerate(targets[n_pol:]):
        d = diag[n_pol + j]
        label = t.target.name
        ess = min(d.ess_success, d.ess_cost)
        if ess < st.min_ess:
            excluded[label] = f"effective sample size {ess:.1f} < {st.min_ess:g}"
        elif d.unsupported_mass > st.support_tolerance:
            excluded[label] = (
                f"{100 * d.unsupported_mass:.1f}% of its probability is on actions "
                "the logging policy never took"
            )
        elif d.unknown_support_mass > st.support_tolerance:
            excluded[label] = "support cannot be checked"
        else:
            included.add(label)

    ones = [1] * n_clusters
    point = stat(ones) if n_clusters else {}
    samples = (
        cluster_bootstrap(n_clusters, st.bootstrap, derive_seed(st.seed, "logged"), stat)
        if n_clusters
        else []
    )
    jack = [stat(c) for c in jackknife_counts(n_clusters)] if n_clusters > 2 else []

    # Unlabelled sensitivity: unlabelled steps counted as failures, then successes.
    unlabelled = sum(1 for x in rows if x.row.success is None)
    sens: list[tuple[str, dict[str, float]]] = []
    if unlabelled:
        for label, value in (("unlabelled_as_failure", False), ("unlabelled_as_success", True)):
            imp_sums = []
            for t in targets:
                vec = _vectors(rows, t, value)
                imp_sums.append({k: math.fsum(v) for k, v in vec.items()})
            imp = [_estimates(cs) for cs in imp_sums]
            pts = [
                Point(config_targets[j].name, e[f"{head}|c"], e[f"{head}|s"])
                for j, e in enumerate(imp[n_pol:])
                if config_targets[j].name in included
                and not math.isnan(e[f"{head}|s"])
                and not math.isnan(e[f"{head}|c"])
            ]
            imp_env = envelope(pts)
            res: dict[str, float] = {}
            for i, e in enumerate(imp[:n_pol]):
                s_i, c_i = e[f"{head}|s"], e[f"{head}|c"]
                res[f"p{i}|s"] = s_i
                res[f"p{i}|margin"] = (
                    math.nan
                    if math.isnan(s_i) or math.isnan(c_i)
                    else s_i - success_at(imp_env, c_i)
                )
            sens.append((label, res))

    # --- clipping bias bounds -----------------------------------------------------------
    c_max = max((x.row.cost for x in rows if x.row.cost is not None), default=0.0)

    def clip_bound(t: _Target, metric: str) -> float:
        if st.clip is None:
            return 0.0
        if metric == "s":
            n_ = sum(1 for x in rows if x.row.success is not None)
            mass = sum(o for o, x in zip(t.clipped, rows, strict=True) if x.row.success is not None)
            return mass / n_ if n_ else 0.0
        n_ = sum(1 for x in rows if x.row.cost is not None)
        mass = sum(o for o, x in zip(t.clipped, rows, strict=True) if x.row.cost is not None)
        return mass * c_max / n_ if n_ else 0.0

    def est(
        pre: str,
        key: str,
        bound: float = 0.0,
        n: int | None = None,
        ess: float | None = None,
        note: str | None = None,
    ) -> Estimate:
        v = point.get(pre + key)
        if v is None or math.isnan(v):
            return Estimate(None, None, n, ess, note)
        e = estimate_from(v, samples, pre + key, st.confidence, jack, n=n, ess=ess, note=note)
        if bound and e.ci is not None and e.ci.estimable:
            e = Estimate(
                e.value,
                Interval(e.ci.lo - bound, e.ci.hi + bound, e.ci.undefined),
                e.n,
                e.ess,
                e.note,
            )
        return e

    n_lab = sum(1 for x in rows if x.row.success is not None)
    n_cost = sum(1 for x in rows if x.row.cost is not None)

    cfg_rows = []
    for j, t in enumerate(targets[n_pol:]):
        d = diag[n_pol + j]
        pre = f"cfg{j}|"
        cfg_rows.append(
            (
                t.target.name,
                est(pre, f"{head}|s", clip_bound(t, "s"), n=n_lab, ess=d.ess_success),
                est(pre, f"{head}|c", clip_bound(t, "c"), n=n_cost, ess=d.ess_cost),
            )
        )
    frontier_pts = [
        Point(lab, c.value, s.value)
        for lab, s, c in cfg_rows
        if lab in included and s.value is not None and c.value is not None
    ]
    strongest_pt = max(frontier_pts, key=lambda p: (p.success, -p.cost), default=None)

    results: list[PolicyResult] = []
    common_gates: list[str] = []
    if invalid:
        common_gates.append(f"{invalid} logged decisions have a missing or invalid propensity")
    if mismatches:
        common_gates.append(
            f"on {mismatches} rows the logged propensity disagrees with the logging policy given"
        )
    if n_clusters < st.min_clusters:
        common_gates.append(
            f"only {n_clusters} independent clusters (tasks or sessions); at least "
            f"{st.min_clusters} needed"
        )
    if not frontier:
        common_gates.append("the single-configuration frontier was not computed")
    elif not config_targets:
        common_gates.append("the logs name no single configuration to compare against")
    if excluded:
        common_gates.append(
            f"{len(excluded)} of {len(config_targets)} single configurations cannot be estimated "
            "from these logs, so the frontier is incomplete"
        )
    for i, t in enumerate(targets[:n_pol]):
        pre = f"p{i}|"
        d = diag[i]
        ess_s, ess_c = d.ess_success, d.ess_cost
        ess = min(ess_s, ess_c)
        gates = list(common_gates)
        low_ess = ess < st.min_ess
        if low_ess:
            gates.append(f"effective sample size {ess:.1f} is below {st.min_ess:g}")
        uns = d.unsupported_mass
        if uns > st.support_tolerance:
            gates.append(
                f"the policy puts {100 * uns:.1f}% of its probability on actions the logging "
                f"policy never took (on {d.rows_with_unsupported_mass} rows)"
            )
        unk = d.unknown_support_mass
        if unk > st.support_tolerance:
            gates.append(
                f"support cannot be checked for {100 * unk:.1f}% of the policy's probability "
                "(no candidate set: pass --catalog or --logging-policy)"
            )
        per_est: dict[str, dict[str, Estimate]] = {}
        for name in ESTIMATORS:
            per_est[name] = {
                "success_rate": est(pre, f"{name}|s", clip_bound(t, "s"), n=n_lab, ess=ess_s),
                "cost_per_step": est(pre, f"{name}|c", clip_bound(t, "c"), n=n_cost, ess=ess_c),
                "cost_per_solved": est(pre, f"{name}|cps", n=n_lab),
            }
        withheld = (
            f"withheld: effective sample size {ess:.1f} < {st.min_ess:g}; see 'estimators'"
            if low_ess
            else None
        )

        def headline(
            m: str, _w: str | None = withheld, _h: dict[str, Estimate] = per_est[head]
        ) -> Estimate:
            e = _h[m]
            return Estimate(None, None, e.n, e.ess, _w) if _w else e

        metrics: dict[str, Estimate] = {
            "success_rate": headline("success_rate"),
            "cost_per_step": headline("cost_per_step"),
            "cost_per_solved": headline("cost_per_solved"),
            "quality_retention": (
                Estimate(None, None, note=withheld or "no single configuration could be estimated")
                if withheld or strongest_pt is None
                else est(pre, "ret", note=f"versus {strongest_pt.label}")
            ),
            "strongest_solved_retained": Estimate(
                None,
                None,
                note="needs paired outcomes on the same step; logs have one action per step",
            ),
            "switch_rate_per_step": est(pre, "sw", n=len(rows)),
            "added_latency_p50_ms": Estimate(
                None, None, note="not estimable: logged outcomes do not split latency by attempt"
            ),
            "added_latency_p95_ms": Estimate(
                None, None, note="not estimable: logged outcomes do not split latency by attempt"
            ),
            "frontier_margin": (
                Estimate(None, None, note=withheld) if withheld else est(pre, "margin", n=n_lab)
            ),
        }
        if st.cost_weight is not None:
            metrics["utility"] = (
                Estimate(None, None, note=withheld) if withheld else est(pre, "util")
            )
        m = metrics["frontier_margin"]

        def verdict_with(g: list[str], _m: Estimate = m, _metrics: dict[str, Estimate] = metrics,
                         _t: _Target = t) -> Verdict:  # fmt: skip
            return decide_verdict(
                _t.target.name,
                "step",
                _metrics["cost_per_step"].value,
                _metrics["success_rate"].value,
                _m.value,
                _m.ci,
                g,
                frontier_pts,
                st.confidence,
                hindsight=_t.target.hindsight,
                min_margin=st.min_margin,
            )

        verdict = verdict_with(gates)
        sens_json: dict[str, object] = {}
        flips: list[str] = []
        for label, res in sens:
            mv = res.get(pre + "margin", math.nan)
            sens_json[label] = {"frontier_margin": num(mv), "success_rate": num(res.get(pre + "s"))}
            if verdict.status != NOT_SHOWN:
                holds = mv > 0 if verdict.status == BEATS else mv <= 0
                if math.isnan(mv) or not holds:
                    flips.append(label.removeprefix("unlabelled_as_"))
        if flips:
            gates.append(
                f"the verdict changes when the {unlabelled} unlabelled steps are counted as "
                f"{' or '.join(flips)}s"
            )
            verdict = verdict_with(gates)
        diagnostics = d.to_json()
        if st.clip is not None:
            diagnostics["clip_bias_bound"] = {
                "success_rate": num(clip_bound(t, "s")),
                "cost_per_step": num(clip_bound(t, "c")),
                "note": "intervals are widened by these bounds on both sides",
            }
        if sens_json:
            diagnostics["unlabelled_sensitivity"] = sens_json
        results.append(
            PolicyResult(t.target.name, t.target.hindsight, metrics, verdict, diagnostics, per_est)
        )

    by_source: dict[str, int] = {}
    for r in data.rows:
        by_source[r.state.source] = by_source.get(r.state.source, 0) + 1
    cost_sources: dict[str, int] = {}
    for r in data.rows:
        cost_sources[r.cost_source] = cost_sources.get(r.cost_source, 0) + 1
    logged_sw = [
        1.0
        if x.row.state.current_model is not None and x.row.action.model != x.row.state.current_model
        else 0.0
        for x in rows
    ]
    provenance: dict[str, object] = {
        "source": data.source,
        "files": list(data.files),
        "sha256": data.sha256,
        "skipped_lines": data.skipped_lines,
        "decisions": len(data.rows),
        "decisions_without_outcome": sum(1 for r in data.rows if r.outcome is None),
        "orphan_outcomes": data.orphan_outcomes,
        "unlabelled_steps": sum(1 for r in data.rows if r.success is None),
        "steps_without_cost": sum(1 for r in data.rows if r.cost is None),
        "cost_source": cost_sources,
        "invalid_propensity": invalid,
        "propensity_mismatches": mismatches if logging_policy is not None else None,
        "logging_policy_checked": None if logging_policy is None else logging_policy.name,
        "candidate_set_issues": data.candidate_issues,
        "rows_with_unknown_support": sum(1 for x in rows if x.support is None),
        "by_source": by_source,
        "logging_policies": sorted({r.decision.policy for r in data.rows}),
        "fail_open_rows": sum(1 for r in data.rows if r.decision.policy.startswith("fail_open:")),
        "clusters": n_clusters,
        "cluster_unit": "task_id when the state has one, else session_id",
        "logged_switch_rate": num(sum(logged_sw) / len(logged_sw)) if logged_sw else None,
        "reward_model": {
            "name": model.name,
            "fitting": (
                f"cross-fitted by cluster in {st.folds} folds: each row's prediction comes from "
                "a model fitted on the other folds"
                if fittable
                else "external: used as given; it must not have been trained on these logs"
            ),
        },
    }
    notes = [
        "Off-policy estimates assume the logged propensities are the true probabilities the "
        "logging policy gave each action. The report cannot check that unless the logging "
        "policy itself is supplied (--logging-policy).",
        "Intervals resample whole clusters (tasks, or sessions without a task id); the reward "
        "model is not refitted inside the bootstrap.",
    ]
    if unlabelled:
        notes.append(
            f"{unlabelled} steps are unlabelled. Success estimates use labelled steps only "
            "(this assumes labels are missing at random given state and action); each policy's "
            "'unlabelled_sensitivity' shows the margin with them counted as failures and as "
            "successes."
        )
    return Evaluation(
        kind="logged",
        level="step",
        scope=LOGGED_SCOPE,
        provenance=provenance,
        configs=cfg_rows,
        frontier_excluded=excluded,
        policies=results,
        notes=notes,
        settings=st,
    )
