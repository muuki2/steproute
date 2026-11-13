"""The single-configuration frontier. Standard library only. See ``docs/EVALUATION.md``.

Points are ``(spend, score)``: mean cost per unit (task or step) and success rate. The
**frontier** is the upper concave envelope of the single-configuration points together with
the origin ``(0, 0)``: the best success rate reachable at each spend by assigning units at
random to single configurations, or to none (declining the unit costs nothing and solves
nothing). It is flat beyond the configuration with the highest success: no single
configuration buys more success with more money.

A policy lies **beyond** the frontier when its success rate is above the envelope at its
spend. The **margin** is that vertical gap, in success-rate units.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["Point", "cost_at", "envelope", "margin", "pareto", "success_at"]

ORIGIN = "decline (spend nothing, solve nothing)"


@dataclass(frozen=True)
class Point:
    label: str
    cost: float
    success: float


def pareto(points: Sequence[Point]) -> list[Point]:
    """Points no other point weakly dominates (no more cost and no less success, one strictly
    better), sorted by cost. Exact duplicates keep the first."""

    def beats(q: Point, p: Point) -> bool:
        return q.cost <= p.cost and q.success >= p.success

    out = [
        p
        for i, p in enumerate(points)
        if not any(
            beats(q, p) and ((q.cost, q.success) != (p.cost, p.success) or j < i)
            for j, q in enumerate(points)
            if j != i
        )
    ]
    return sorted(out, key=lambda p: (p.cost, p.success))


def envelope(points: Sequence[Point]) -> list[Point]:
    """Vertices of the frontier, by increasing cost, starting at the origin (or at a cost-0
    configuration) and ending at the cheapest configuration with the highest success."""
    if not points:
        return [Point(ORIGIN, 0.0, 0.0)]
    best = max(p.success for p in points)
    top_cost = min(p.cost for p in points if p.success == best)
    cands = [Point(ORIGIN, 0.0, 0.0), *(p for p in points if p.cost <= top_cost)]
    by_cost: dict[float, Point] = {}
    for p in cands:
        cur = by_cost.get(p.cost)
        if cur is None or p.success > cur.success:
            by_cost[p.cost] = p
    hull: list[Point] = []
    for p in sorted(by_cost.values(), key=lambda p: p.cost):
        while len(hull) >= 2:
            o, a = hull[-2], hull[-1]
            cross = (a.cost - o.cost) * (p.success - o.success) - (a.success - o.success) * (
                p.cost - o.cost
            )
            if cross >= 0:  # a is on or below the chord o-p: not a vertex of the upper hull
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def success_at(env: Sequence[Point], cost: float) -> float:
    """The frontier's success rate at ``cost`` (linear between vertices, flat at the end)."""
    if cost >= env[-1].cost:
        return env[-1].success
    for a, b in zip(env, env[1:], strict=False):
        if a.cost <= cost <= b.cost:
            if b.cost == a.cost:
                return max(a.success, b.success)
            t = (cost - a.cost) / (b.cost - a.cost)
            return a.success + t * (b.success - a.success)
    return env[0].success  # cost below the first vertex: only possible for cost < 0


def cost_at(env: Sequence[Point], success: float) -> float:
    """The least spend at which the frontier reaches ``success``; ``inf`` above its top."""
    if success > env[-1].success:
        return math.inf
    if success <= env[0].success:
        return env[0].cost
    for a, b in zip(env, env[1:], strict=False):
        if a.success <= success <= b.success:
            if b.success == a.success:
                return a.cost
            t = (success - a.success) / (b.success - a.success)
            return a.cost + t * (b.cost - a.cost)
    return env[-1].cost  # pragma: no cover - the envelope is non-decreasing


def margin(env: Sequence[Point], cost: float, success: float) -> float:
    """Success-rate points above (positive) or below (negative) the frontier at ``cost``."""
    return success - success_at(env, cost)
