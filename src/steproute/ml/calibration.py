"""Probability calibration and calibration metrics. Standard library only.

Two calibrators are fitted on held-out data by ``fit_calibrator``:

- ``platt``: ``p = sigmoid(a * margin + b)`` where ``margin`` is the booster's raw score (the
  log-odds). Two parameters, so it is stable on a few hundred rows. This is the default.
- ``isotonic``: a non-decreasing piecewise-linear map from the raw probability, fitted by the
  pool-adjacent-violators algorithm. More flexible, needs more rows, and returns ties.
- ``none``: the identity (the booster's own probability).

A calibrator is plain data (``to_dict`` / ``from_dict``), so a saved model holds no pickle.
The metrics at the bottom (expected calibration error, Brier score, log loss, reliability
table, the gap near a threshold) are defined in ``docs/MODEL-CARD-TABULAR.md``.
"""

from __future__ import annotations

import bisect
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ECE_BINS",
    "Calibrator",
    "auc",
    "brier",
    "ece",
    "fit_calibrator",
    "log_loss",
    "reliability",
    "select_calibrator",
    "sigmoid",
    "window_gap",
]

#: Number of equal-width probability bins used by ``ece`` and ``reliability``.
ECE_BINS = 10
_EPS = 1e-12


def sigmoid(x: float) -> float:
    """Numerically safe logistic function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True)
class Calibrator:
    """Maps a raw margin (log-odds) to a calibrated probability."""

    kind: str = "none"
    a: float = 1.0
    b: float = 0.0
    xs: tuple[float, ...] = field(default_factory=tuple)
    ys: tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.kind not in ("none", "platt", "isotonic"):
            raise ValueError(f"unknown calibration method {self.kind!r}")
        if self.kind == "isotonic" and (not self.xs or len(self.xs) != len(self.ys)):
            raise ValueError("an isotonic calibrator needs matching, non-empty xs and ys")

    def __call__(self, margin: float) -> float:
        if self.kind == "none":
            return sigmoid(margin)
        if self.kind == "platt":
            return sigmoid(self.a * margin + self.b)
        p = sigmoid(margin)
        xs, ys = self.xs, self.ys
        if p <= xs[0]:
            return ys[0]
        if p >= xs[-1]:
            return ys[-1]
        i = bisect.bisect_right(xs, p)
        x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
        return y0 + (y1 - y0) * (p - x0) / (x1 - x0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "a": self.a,
            "b": self.b,
            "xs": list(self.xs),
            "ys": list(self.ys),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Calibrator:
        return cls(
            kind=str(data["kind"]),
            a=float(data["a"]),
            b=float(data["b"]),
            xs=tuple(float(v) for v in data.get("xs", ())),
            ys=tuple(float(v) for v in data.get("ys", ())),
        )


def _platt_loss(margins: Sequence[float], t: Sequence[float], a: float, b: float) -> float:
    total = 0.0
    for m, ti in zip(margins, t, strict=True):
        p = min(max(sigmoid(a * m + b), 1e-12), 1.0 - 1e-12)
        total -= ti * math.log(p) + (1.0 - ti) * math.log(1.0 - p)
    return total


def _fit_platt(margins: Sequence[float], y: Sequence[int]) -> Calibrator:
    # Platt's smoothed targets keep the fit finite when the classes are separable.
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    t_pos = (n_pos + 1.0) / (n_pos + 2.0)
    t_neg = 1.0 / (n_neg + 2.0)
    t = [t_pos if v else t_neg for v in y]
    a, b = 1.0, 0.0
    ridge = 1e-6
    for _ in range(100):
        g_a = g_b = h_aa = h_ab = h_bb = 0.0
        for m, ti in zip(margins, t, strict=True):
            p = sigmoid(a * m + b)
            d = p - ti
            w = max(p * (1.0 - p), 1e-12)
            g_a += d * m
            g_b += d
            h_aa += w * m * m
            h_ab += w * m
            h_bb += w
        h_aa += ridge
        h_bb += ridge
        det = h_aa * h_bb - h_ab * h_ab
        if det <= 0.0:
            break
        da = (h_bb * g_a - h_ab * g_b) / det
        db = (h_aa * g_b - h_ab * g_a) / det
        # Damped Newton step: halve until the loss does not rise.
        base = _platt_loss(margins, t, a, b)
        step = 1.0
        while step > 1e-4 and _platt_loss(margins, t, a - step * da, b - step * db) > base:
            step /= 2.0
        a -= step * da
        b -= step * db
        if abs(step * da) + abs(step * db) < 1e-10:
            break
    return Calibrator("platt", a=a, b=b)


def _fit_isotonic(margins: Sequence[float], y: Sequence[int]) -> Calibrator:
    order = sorted(range(len(y)), key=lambda i: margins[i])
    # Pool adjacent violators over blocks of [sum y, count, sum x].
    blocks: list[list[float]] = []
    for i in order:
        blocks.append([float(y[i]), 1.0, sigmoid(margins[i])])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]:
            top = blocks.pop()
            blocks[-1][0] += top[0]
            blocks[-1][1] += top[1]
            blocks[-1][2] += top[2]
    # A block of all failures or all successes would give probability exactly 0 or 1, and one
    # miss then costs infinite log loss: keep the values a rule-of-succession step inside.
    eps = 1.0 / (len(y) + 2.0)
    xs: list[float] = []
    ys: list[float] = []
    for s, c, sx in blocks:
        x = sx / c
        if xs and x <= xs[-1]:
            continue  # keep xs strictly increasing; the earlier block wins
        xs.append(x)
        ys.append(min(max(s / c, eps), 1.0 - eps))
    return Calibrator("isotonic", xs=tuple(xs), ys=tuple(ys))


def fit_calibrator(method: str, margins: Sequence[float], y: Sequence[int]) -> Calibrator:
    """Fit a calibrator on held-out ``(margin, label)`` pairs.

    ``margins`` are raw booster scores (log-odds); ``y`` is 0 or 1. With ``none``, fewer than
    20 rows, or a single class in ``y``, the identity is returned (a calibrator cannot be
    fitted from nothing, and a wrong one is worse than none).
    """
    if method not in ("none", "platt", "isotonic"):
        raise ValueError(f"unknown calibration method {method!r}")
    if len(margins) != len(y):
        raise ValueError("margins and labels differ in length")
    if method == "none" or len(y) < 20 or sum(y) in (0, len(y)):
        return Calibrator("none")
    if method == "platt":
        return _fit_platt(margins, y)
    return _fit_isotonic(margins, y)


def select_calibrator(
    margins: Sequence[float],
    y: Sequence[int],
    groups: Sequence[str],
    *,
    seed: int,
    folds: int = 5,
    min_gain: float = 0.001,
) -> tuple[Calibrator, dict[str, float]]:
    """Pick ``none``, ``platt`` or ``isotonic`` by cross-validated log loss, then fit it on all.

    Calibration is verified rather than assumed: a boosted model stopped early is often
    already well calibrated, and a fitted map then only adds noise. The calibration rows are
    split into ``folds`` folds **by group** (task), each method is fitted on the other folds
    and scored on this one, and a method replaces the identity only if it lowers the pooled
    held-out log loss by more than ``min_gain``. Returns the calibrator and the pooled
    held-out log loss of every method (for the saved metadata).
    """
    ids = sorted(set(groups))
    rng = random.Random(f"{seed}:calibration-folds")
    rng.shuffle(ids)
    fold_of = {g: i % folds for i, g in enumerate(ids)}
    if len(ids) < folds or len(y) < 40:
        return Calibrator("none"), {}
    losses: dict[str, float] = {}
    for method in ("none", "platt", "isotonic"):
        total = 0.0
        for f in range(folds):
            tr = [i for i, g in enumerate(groups) if fold_of[g] != f]
            te = [i for i, g in enumerate(groups) if fold_of[g] == f]
            cal = fit_calibrator(method, [margins[i] for i in tr], [y[i] for i in tr])
            total += log_loss([cal(margins[i]) for i in te], [y[i] for i in te]) * len(te)
        losses[method] = total / len(y)
    best = "none"
    for method in ("platt", "isotonic"):
        if losses[method] < losses[best] - (min_gain if best == "none" else 0.0):
            best = method
    return fit_calibrator(best, margins, y), losses


# --- metrics --------------------------------------------------------------------------------


def _bin(p: float, bins: int) -> int:
    return min(int(p * bins), bins - 1)


def brier(p: Sequence[float], y: Sequence[int]) -> float:
    """Mean squared error of the probabilities."""
    return sum((pi - yi) ** 2 for pi, yi in zip(p, y, strict=True)) / len(y)


def log_loss(p: Sequence[float], y: Sequence[int]) -> float:
    """Mean negative log-likelihood, probabilities clipped to ``[1e-12, 1 - 1e-12]``."""
    total = 0.0
    for pi, yi in zip(p, y, strict=True):
        q = min(max(pi, _EPS), 1.0 - _EPS)
        total -= math.log(q if yi else 1.0 - q)
    return total / len(y)


def reliability(
    p: Sequence[float], y: Sequence[int], bins: int = ECE_BINS
) -> list[dict[str, float | int | None]]:
    """One row per equal-width bin ``[i/bins, (i+1)/bins)`` (the last includes 1.0):
    ``count``, ``mean_predicted``, ``observed`` (None when empty)."""
    cnt = [0] * bins
    sp = [0.0] * bins
    sy = [0.0] * bins
    for pi, yi in zip(p, y, strict=True):
        b = _bin(pi, bins)
        cnt[b] += 1
        sp[b] += pi
        sy[b] += yi
    return [
        {
            "bin_low": i / bins,
            "bin_high": (i + 1) / bins,
            "count": cnt[i],
            "mean_predicted": sp[i] / cnt[i] if cnt[i] else None,
            "observed": sy[i] / cnt[i] if cnt[i] else None,
        }
        for i in range(bins)
    ]


def ece(p: Sequence[float], y: Sequence[int], bins: int = ECE_BINS) -> float:
    """Expected calibration error: ``sum_b (n_b / n) * |observed_b - mean_predicted_b|`` over
    ``bins`` equal-width bins of the predicted probability."""
    total = 0.0
    for row in reliability(p, y, bins):
        n_b = int(row["count"] or 0)
        if n_b:
            obs, pred = row["observed"], row["mean_predicted"]
            assert obs is not None and pred is not None
            total += n_b / len(y) * abs(obs - pred)
    return total


def window_gap(
    p: Sequence[float], y: Sequence[int], center: float, half_width: float = 0.05
) -> tuple[int, float | None]:
    """``(n, observed - mean predicted)`` over rows with ``|p - center| <= half_width``.

    The policy compares a success probability with a threshold, so this is the calibration
    that matters most. ``None`` when no row falls in the window.
    """
    rows = [(pi, yi) for pi, yi in zip(p, y, strict=True) if abs(pi - center) <= half_width]
    if not rows:
        return 0, None
    return len(rows), sum(yi for _, yi in rows) / len(rows) - sum(pi for pi, _ in rows) / len(rows)


def auc(p: Sequence[float], y: Sequence[int]) -> float | None:
    """Area under the ROC curve (ties count half); None if a class is missing."""
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(p)), key=lambda i: p[i])
    rank_sum = 0.0
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and p[order[j + 1]] == p[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        rank_sum += avg * sum(y[order[k]] for k in range(i, j + 1))
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
