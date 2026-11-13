"""Statistics helpers of the evaluator."""

import math
import random

import pytest

from steproute.eval.stats import (
    bca_interval,
    cluster_bootstrap,
    cluster_index,
    jackknife_counts,
    kish_ess,
    percentile_interval,
    quantile,
    weighted_quantile,
)


def test_kish_ess() -> None:
    assert kish_ess([1, 1, 1, 1]) == 4
    assert kish_ess([0, 0, 2, 4]) == pytest.approx(36 / 20)
    assert kish_ess([0, 0]) == 0


def test_cluster_index_and_bootstrap_are_deterministic() -> None:
    idx, n = cluster_index(["b", "a", "b", "c"])
    assert idx == [0, 1, 0, 2] and n == 3
    seen = cluster_bootstrap(5, 100, 42, lambda c: {"n": sum(c), "c0": c[0]})
    assert all(s["n"] == 5 for s in seen)
    assert seen == cluster_bootstrap(5, 100, 42, lambda c: {"n": sum(c), "c0": c[0]})
    assert seen != cluster_bootstrap(5, 100, 43, lambda c: {"n": sum(c), "c0": c[0]})


def test_quantiles() -> None:
    assert quantile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert quantile([1.0, math.inf], 0.9) == math.inf
    assert weighted_quantile([(0.0, 1.0), (100.0, 3.0)], 0.5) == 100.0
    assert weighted_quantile([(0.0, 1.0), (100.0, 3.0)], 0.25) == 0.0


def test_percentile_interval_handles_undefined_replicates() -> None:
    ok = percentile_interval([float(i) for i in range(101)], 0.9)
    assert (ok.lo, ok.hi) == (pytest.approx(5.0), pytest.approx(95.0))
    many_nan = percentile_interval([math.nan] * 10 + [1.0] * 90, 0.95)
    assert not many_nan.estimable and many_nan.undefined == 10


def test_jackknife_groups() -> None:
    groups = jackknife_counts(250, max_groups=100)
    assert len(groups) == 100
    assert all(sum(g) in (247, 248) for g in groups)
    assert sum(1 - x for g in groups for x in g) == 250  # every cluster left out once


def test_bca_corrects_a_skewed_statistic() -> None:
    """For the mean of a right-skewed sample, BCa shifts the interval right of the
    percentile interval, which is the direction its undercoverage needs."""
    rng = random.Random(0)
    data = [rng.expovariate(1.0) ** 2 for _ in range(60)]
    n = len(data)

    def mean_of(counts: list[int] | tuple[int, ...]) -> float:
        return sum(c * x for c, x in zip(counts, data, strict=True)) / sum(counts)

    boots = [s["m"] for s in cluster_bootstrap(n, 2000, 1, lambda c: {"m": mean_of(c)})]
    jack = [mean_of(c) for c in jackknife_counts(n)]
    point = sum(data) / n
    pct = percentile_interval(boots, 0.95)
    bca = bca_interval(boots, point, jack, 0.95)
    assert bca.lo > pct.lo and bca.hi > pct.hi
    # Degenerate cases fall back to the percentile interval.
    assert bca_interval([1.0] * 200, 1.0, [1.0] * 10, 0.95) == percentile_interval(
        [1.0] * 200, 0.95
    )
