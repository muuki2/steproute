"""Result types shared by both evaluators, the verdict rule and its plain sentence.

Standard library only. JSON encoding lives here too: infinities and NaN never reach the JSON
file; they become ``null`` next to an explicit flag.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from steproute.eval.frontier import Point, cost_at, envelope, success_at
from steproute.eval.stats import Interval, bca_interval, percentile_interval

__all__ = [
    "BEATS",
    "DOES_NOT_BEAT",
    "NOT_SHOWN",
    "Estimate",
    "Evaluation",
    "EvalSettings",
    "PolicyResult",
    "Verdict",
    "decide_verdict",
    "estimate_from",
    "num",
]

BEATS = "beats_frontier"
DOES_NOT_BEAT = "does_not_beat"
NOT_SHOWN = "not_shown"

#: Exact ties (a policy that *is* a frontier configuration) have margin 0 up to rounding.
TIE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class EvalSettings:
    """Knobs shared by both evaluators. Defaults are the documented ones."""

    #: Two-sided confidence level of every interval.
    confidence: float = 0.95
    #: Cluster bootstrap replicates.
    bootstrap: int = 2000
    #: Seeds the bootstrap and the cross-fitting folds.
    seed: int = 0
    #: Below this effective sample size an off-policy estimate gets no headline or verdict.
    min_ess: float = 50.0
    #: Below this many clusters (tasks, or sessions without a task id) nothing gets a verdict.
    #: Measured: the margin's 95% interval covers 82-91% at 25 tasks and 86-89% at 50, and
    #: 92-93% at 100 (docs/EVALUATION.md, "Coverage").
    min_clusters: int = 100
    #: Importance-weight cap; ``None`` (default) means no clipping.
    clip: float | None = None
    #: Off-policy headline estimator: ``dr``, ``snips`` or ``ips``.
    headline: str = "dr"
    #: Cross-fitting folds for the built-in reward model.
    folds: int = 5
    #: Largest mean target probability on unsupported actions that still allows a verdict.
    support_tolerance: float = 0.001
    #: If set, policies also get ``utility = success - cost_weight * cost`` (cost in USD).
    cost_weight: float | None = None
    #: Smallest frontier margin (success-rate units) worth calling a win. 0 means any margin
    #: whose interval excludes 0; set it to the smallest gain you would act on.
    min_margin: float = 0.0

    def __post_init__(self) -> None:
        if not 0.5 <= self.confidence < 1.0:
            raise ValueError("confidence must be in [0.5, 1)")
        if self.bootstrap < 100:
            raise ValueError("bootstrap must be at least 100 replicates")
        if self.headline not in ("dr", "snips", "ips"):
            raise ValueError("headline must be dr, snips or ips")
        if self.clip is not None and not self.clip >= 1.0:
            raise ValueError("clip must be >= 1 (a weight cap) or None")
        if self.folds < 2:
            raise ValueError("folds must be at least 2")

    def to_json(self) -> dict[str, Any]:
        return {
            "confidence": self.confidence,
            "bootstrap_replicates": self.bootstrap,
            "seed": self.seed,
            "min_ess": self.min_ess,
            "min_clusters": self.min_clusters,
            "clip": self.clip,
            "headline_estimator": self.headline,
            "crossfit_folds": self.folds,
            "support_tolerance": self.support_tolerance,
            "cost_weight": self.cost_weight,
            "min_margin": self.min_margin,
        }


def num(x: float | None, digits: int = 10) -> float | None:
    """JSON-safe number: ``None`` for missing, infinite or NaN."""
    if x is None or not math.isfinite(x):
        return None
    return round(x, digits)


@dataclass(frozen=True)
class Estimate:
    """A point estimate with its interval and sample information."""

    value: float | None
    ci: Interval | None = None
    n: int | None = None
    ess: float | None = None
    note: str | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"value": num(self.value)}
        if self.value is not None and math.isinf(self.value):
            out["unbounded"] = True
        if self.ci is not None:
            if self.ci.estimable:
                out["ci"] = [num(self.ci.lo), num(self.ci.hi)]
                if math.isinf(self.ci.hi) or math.isinf(self.ci.lo):
                    out["ci_unbounded"] = True
            else:
                out["ci"] = None
                out["ci_note"] = "not estimable: too many bootstrap resamples were undefined"
            if self.ci.undefined:
                out["undefined_replicates"] = self.ci.undefined
        if self.n is not None:
            out["n"] = self.n
        if self.ess is not None:
            out["ess"] = num(self.ess, 3)
        if self.note:
            out["note"] = self.note
        return out


@dataclass(frozen=True)
class Verdict:
    status: str
    reasons: tuple[str, ...]
    sentence: str

    def to_json(self) -> dict[str, Any]:
        return {"status": self.status, "reasons": list(self.reasons), "sentence": self.sentence}


@dataclass
class PolicyResult:
    name: str
    hindsight: bool
    metrics: dict[str, Estimate]
    verdict: Verdict
    diagnostics: dict[str, Any] = field(default_factory=dict)
    #: Estimators other than the headline, for the off-policy evaluator.
    estimators: dict[str, dict[str, Estimate]] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "hindsight": self.hindsight,
            "metrics": {k: v.to_json() for k, v in self.metrics.items()},
            "verdict": self.verdict.to_json(),
            "diagnostics": self.diagnostics,
        }
        if self.estimators:
            out["estimators"] = {
                est: {k: v.to_json() for k, v in ms.items()} for est, ms in self.estimators.items()
            }
        return out


@dataclass
class Evaluation:
    """One dataset evaluated against several policies."""

    kind: str  # "grid" or "logged"
    level: str  # "task" or "step"
    scope: str
    provenance: dict[str, Any]
    configs: list[tuple[str, Estimate, Estimate]]  # label, success, cost
    frontier_excluded: dict[str, str]
    policies: list[PolicyResult]
    notes: list[str]
    settings: EvalSettings

    def frontier_points(self) -> list[Point]:
        return [
            Point(label, c.value, s.value)
            for label, s, c in self.configs
            if s.value is not None and c.value is not None
        ]

    def ranking(self) -> list[dict[str, Any]]:
        """Policies by frontier margin, highest first; hindsight references excluded."""
        rows = []
        for p in self.policies:
            if p.hindsight:
                continue
            m = p.metrics.get("frontier_margin")
            u = p.metrics.get("utility")
            rows.append(
                {
                    "name": p.name,
                    "frontier_margin": None if m is None else num(m.value),
                    "utility": None if u is None else num(u.value),
                    "verdict": p.verdict.status,
                }
            )
        key = "utility" if self.settings.cost_weight is not None else "frontier_margin"

        def order(r: dict[str, Any]) -> float:
            v = r[key]
            return math.inf if v is None else -float(v)

        rows.sort(key=order)
        for i, r in enumerate(rows, start=1):
            r["rank"] = i
            r["ranked_by"] = key
        return rows

    def to_json(self) -> dict[str, Any]:
        pts = [p for p in self.frontier_points() if p.label not in self.frontier_excluded]
        env = envelope(pts)
        return {
            "kind": self.kind,
            "level": self.level,
            "scope": self.scope,
            "provenance": self.provenance,
            "frontier": {
                "definition": (
                    "upper concave envelope of the single-configuration points and the origin "
                    "(declining a unit); flat beyond the configuration with the highest success"
                ),
                "configurations": [
                    {"label": label, "success_rate": s.to_json(), "cost": c.to_json()}
                    for label, s, c in self.configs
                ],
                "envelope": [
                    {"label": p.label, "cost": num(p.cost), "success_rate": num(p.success)}
                    for p in env
                ],
                "excluded": self.frontier_excluded,
            },
            "policies": [p.to_json() for p in self.policies],
            "ranking": self.ranking(),
            "notes": self.notes,
        }


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _usd(x: float) -> str:
    return f"${x:.4g}"


def decide_verdict(
    name: str,
    unit: str,
    cost: float | None,
    success: float | None,
    margin_value: float | None,
    margin_ci: Interval | None,
    gates: Sequence[str],
    points: Sequence[Point],
    confidence: float,
    hindsight: bool = False,
    min_margin: float = 0.0,
) -> Verdict:
    """The verdict rule (``docs/EVALUATION.md``, "Beyond the frontier").

    ``beats_frontier`` when the lower end of the margin's interval is above ``min_margin``
    (default 0); ``does_not_beat`` when its upper end is at or below it (a tie with a
    frontier configuration does not beat it); ``not_shown`` otherwise, or whenever a gate
    (low effective sample size, unsupported actions, too few clusters, a verdict that depends
    on unlabelled outcomes) applies.
    """
    bar = min_margin + TIE_TOLERANCE
    need_txt = f" by the required {min_margin * 100:.1f} points" if min_margin > 0 else ""
    level = f"{round(100 * confidence)}%"
    tag = " (hindsight oracle: an upper reference that reads outcomes, not a policy)"
    who = name + (tag if hindsight else "")
    reasons = list(gates)
    if not reasons and (margin_value is None or margin_ci is None or not margin_ci.estimable):
        reasons.append("the frontier margin could not be estimated")
    if reasons or cost is None or success is None or margin_value is None or margin_ci is None:
        return Verdict(
            NOT_SHOWN,
            tuple(reasons),
            f"Not shown: this data cannot tell whether {who} beats every single "
            f"configuration ({'; '.join(reasons)}).",
        )
    env = envelope(points)
    at = success_at(env, cost)
    span = f"[{margin_ci.lo * 100:+.1f}, {margin_ci.hi * 100:+.1f}] points"
    where = (
        f"at {_usd(cost)} per {unit} it succeeds on {_pct(success)} of {unit}s, where the best "
        f"any single configuration or random mix of them reaches at that spend is {_pct(at)} "
        f"(margin {margin_value * 100:+.1f} points, {level} interval {span})"
    )
    if margin_ci.lo > bar:
        return Verdict(BEATS, (), f"{who} beats every single configuration{need_txt}: {where}.")
    if margin_ci.hi <= bar:
        need = cost_at(env, success)
        better = [p for p in points if p.cost <= cost and p.success >= success]
        detail = ""
        if better:
            b = min(better, key=lambda p: (p.cost, -p.success))
            detail = (
                f" {b.label} alone costs {_usd(b.cost)} per {unit} and succeeds on "
                f"{_pct(b.success)}."
            )
        elif math.isfinite(need):
            detail = f" The frontier reaches the same success rate at {_usd(need)} per {unit}."
        return Verdict(
            DOES_NOT_BEAT,
            (),
            f"{who} does not beat every single configuration{need_txt}: {where}.{detail}",
        )
    straddles = f"{min_margin * 100:.1f} points" if min_margin > 0 else "0"
    return Verdict(
        NOT_SHOWN,
        (f"the margin's interval contains {straddles}",),
        f"Not shown: this data cannot tell whether {who} beats every single configuration"
        f"{need_txt}; {where}, and the interval contains {straddles}.",
    )


def estimate_from(
    point: float | None,
    samples: Sequence[Mapping[str, float]],
    key: str,
    confidence: float,
    jack: Sequence[Mapping[str, float]] = (),
    **kw: Any,
) -> Estimate:
    """``point`` with its BCa interval from bootstrap ``samples`` and jackknife ``jack``
    (percentile interval when no jackknife is given)."""
    if point is None:
        return Estimate(None, None, **kw)
    if not samples:
        return Estimate(point, None, **kw)
    values = [s[key] for s in samples]
    if jack:
        ci = bca_interval(values, point, [j[key] for j in jack], confidence)
    else:
        ci = percentile_interval(values, confidence)
    return Estimate(point, ci, **kw)
