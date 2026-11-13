"""Statistics helpers: cluster bootstrap, percentile intervals, quantiles, effective sample
size. Standard library only."""

from __future__ import annotations

import hashlib
import math
import operator
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist

__all__ = [
    "Interval",
    "bca_interval",
    "jackknife_counts",
    "cluster_bootstrap",
    "cluster_index",
    "cluster_sums",
    "derive_seed",
    "dot",
    "kish_ess",
    "percentile_interval",
    "quantile",
    "weighted_quantile",
]


def derive_seed(seed: int, *parts: str) -> int:
    """A 64-bit seed from ``seed`` and labels, so independent streams never share draws."""
    text = ":".join([str(seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(text, digest_size=8).digest(), "big")


def kish_ess(weights: Sequence[float]) -> float:
    """Kish's effective sample size ``(sum w)^2 / sum w^2``; 0 when every weight is 0."""
    s1 = math.fsum(weights)
    s2 = math.fsum(w * w for w in weights)
    return (s1 * s1) / s2 if s2 > 0 else 0.0


def cluster_index(clusters: Sequence[str]) -> tuple[list[int], int]:
    """Map cluster labels to ``0..C-1`` in order of first appearance."""
    ids: dict[str, int] = {}
    out = [ids.setdefault(c, len(ids)) for c in clusters]
    return out, len(ids)


def cluster_sums(values: Sequence[float], index: Sequence[int], n_clusters: int) -> list[float]:
    """Per-cluster sums of a per-unit quantity."""
    out = [0.0] * n_clusters
    for v, c in zip(values, index, strict=True):
        out[c] += v
    return out


def dot(counts: Sequence[int], values: Sequence[float]) -> float:
    # Plain summation: the bootstrap calls this millions of times and fsum is several times
    # slower; the rounding difference is far below any interval width.
    return float(sum(map(operator.mul, counts, values)))


def cluster_bootstrap(
    n_clusters: int,
    replicates: int,
    seed: int,
    statistic: Callable[[Sequence[int]], Mapping[str, float]],
) -> list[Mapping[str, float]]:
    """Resample whole clusters with replacement, ``replicates`` times.

    ``statistic`` receives the multiplicity of each cluster in the resample (a list of
    ``n_clusters`` counts summing to ``n_clusters``) and returns named values. Every caller
    with the same ``seed`` and ``n_clusters`` sees the same resamples, which is what makes
    comparisons between policies paired.
    """
    rng = random.Random(seed)
    population = range(n_clusters)
    out: list[Mapping[str, float]] = []
    for _ in range(replicates):
        counts = [0] * n_clusters
        for i in rng.choices(population, k=n_clusters):
            counts[i] += 1
        out.append(statistic(counts))
    return out


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (type 7) of already sorted values; infinities allowed."""
    n = len(sorted_values)
    if n == 0:
        return math.nan
    h = (n - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, n - 1)
    a, b = sorted_values[lo], sorted_values[hi]
    if a == b or h == lo:
        return a
    if math.isinf(a) or math.isinf(b):
        return b if h - lo >= 0.5 else a
    return a + (h - lo) * (b - a)


@dataclass(frozen=True)
class Interval:
    """A percentile bootstrap interval. ``lo``/``hi`` may be infinite (unbounded) or NaN
    (not estimable: too many replicates were undefined)."""

    lo: float
    hi: float
    undefined: int = 0

    @property
    def estimable(self) -> bool:
        return not (math.isnan(self.lo) or math.isnan(self.hi))


def percentile_interval(samples: Sequence[float], level: float) -> Interval:
    """Percentile interval at ``level``. NaN replicates (statistic undefined in that
    resample) are dropped; if more than half of the tail mass ``(1 - level) / 2`` is NaN the
    interval is not estimable."""
    finite = sorted(v for v in samples if not math.isnan(v))
    undefined = len(samples) - len(finite)
    alpha = 1.0 - level
    if not finite or undefined > 0.5 * (alpha / 2) * len(samples):
        return Interval(math.nan, math.nan, undefined)
    return Interval(quantile(finite, alpha / 2), quantile(finite, 1 - alpha / 2), undefined)


def jackknife_counts(n_clusters: int, max_groups: int = 100) -> list[list[int]]:
    """Cluster multiplicities for a (grouped) delete-one jackknife: clusters are dealt into at
    most ``max_groups`` groups and each count vector leaves one group out."""
    groups = min(n_clusters, max_groups)
    return [[0 if i % groups == g else 1 for i in range(n_clusters)] for g in range(groups)]


_NORMAL = NormalDist()


def bca_interval(
    samples: Sequence[float], point: float, jack: Sequence[float], level: float
) -> Interval:
    """Bias-corrected and accelerated (BCa) bootstrap interval.

    The bias correction ``z0`` comes from the share of resamples below the point estimate
    (ties count half); the acceleration ``a`` from the jackknife values' skewness. Both
    correct the percentile interval's known undercoverage for skewed statistics, such as
    mean cost or a ratio. Falls back to the percentile interval when either is undefined
    (every resample on one side of the point, or a constant jackknife).
    """
    finite = sorted(v for v in samples if not math.isnan(v))
    undefined = len(samples) - len(finite)
    alpha = 1.0 - level
    if not finite or undefined > 0.5 * (alpha / 2) * len(samples):
        return Interval(math.nan, math.nan, undefined)
    below = sum(1 for v in finite if v < point)
    equal = sum(1 for v in finite if v == point)
    p0 = (below + 0.5 * equal) / len(finite)
    jk = [v for v in jack if math.isfinite(v)]
    if not 0.0 < p0 < 1.0 or len(jk) < 3 or not math.isfinite(point):
        return percentile_interval(samples, level)
    z0 = _NORMAL.inv_cdf(p0)
    mean = sum(jk) / len(jk)
    d2 = sum((mean - v) ** 2 for v in jk)
    if d2 <= 0:
        return percentile_interval(samples, level)
    acc = sum((mean - v) ** 3 for v in jk) / (6.0 * d2**1.5)

    def adjusted(q: float) -> float:
        z = _NORMAL.inv_cdf(q)
        den = 1.0 - acc * (z0 + z)
        if den <= 0:
            return 1.0 if z > 0 else 0.0
        return _NORMAL.cdf(z0 + (z0 + z) / den)

    return Interval(
        quantile(finite, adjusted(alpha / 2)), quantile(finite, adjusted(1 - alpha / 2)), undefined
    )


def weighted_quantile(atoms: Sequence[tuple[float, float]], q: float) -> float:
    """Lower weighted quantile: the smallest value ``x`` with ``F(x) >= q`` under the
    distribution putting weight ``w`` on each ``(x, w)``. ``atoms`` must be sorted by value."""
    total = sum(w for _, w in atoms)
    if total <= 0:
        return math.nan
    target = q * total
    acc = 0.0
    for x, w in atoms:
        acc += w
        if acc >= target - 1e-12 * total:
            return x
    return atoms[-1][0]
