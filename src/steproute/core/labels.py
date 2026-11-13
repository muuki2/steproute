"""Outcome labelling: turn raw ``Outcome.signals`` into the single ``Outcome.success`` label.

Precedence, strongest first (see ``docs/LABELS.md``): explicit task check, tool or compile
error, user rejection or rewrite, retry on the same step, judge score, nothing. Only the
highest tier that has usable evidence decides; lower tiers never override it. A value of the
wrong type, or a non-finite or out-of-range judge score, counts as absent. Absence of
evidence is ``None``, never ``False``.

Raw signals are never modified: ``label`` only reads them and ``apply_label`` leaves
``Outcome.signals`` exactly as it found it.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

from steproute.core.schema import Outcome

__all__ = [
    "COMPILE_ERROR",
    "DEFAULT_LABEL_CONFIG",
    "JUDGE_SCORE",
    "NO_SIGNALS",
    "RETRIED",
    "RETRY_COUNT",
    "SCHEMA_VALID",
    "SIGNAL_KEYS",
    "TESTS_PASSED",
    "TIERS",
    "TOOL_ERROR",
    "USER_APPROVED",
    "USER_REJECTED",
    "USER_REWROTE",
    "Label",
    "LabelConfig",
    "apply_label",
    "label",
]

# --- signal vocabulary (keys of ``Outcome.signals``) ----------------------------------------

TESTS_PASSED = "tests_passed"  # bool: test suite result; False is failure
SCHEMA_VALID = "schema_valid"  # bool: output validated against a schema; False is failure
TOOL_ERROR = "tool_error"  # bool: True is failure; False is no evidence
COMPILE_ERROR = "compile_error"  # bool: True is failure; False is no evidence
USER_REJECTED = "user_rejected"  # bool: True is failure
USER_REWROTE = "user_rewrote"  # bool: True is failure
USER_APPROVED = "user_approved"  # bool: True is success (loses to rejection or rewrite)
RETRIED = "retried"  # bool: True is failure (weak)
RETRY_COUNT = "retry_count"  # int: above 0 means the same as retried=True
JUDGE_SCORE = "judge_score"  # float in [0, 1]: success if at or above the threshold

#: Every key ``label`` reads. Other keys are ignored.
SIGNAL_KEYS: tuple[str, ...] = (
    TESTS_PASSED,
    SCHEMA_VALID,
    TOOL_ERROR,
    COMPILE_ERROR,
    USER_REJECTED,
    USER_REWROTE,
    USER_APPROVED,
    RETRIED,
    RETRY_COUNT,
    JUDGE_SCORE,
)

#: Tier names, strongest first. These are the prefixes of ``Label.reason``.
TIERS: tuple[str, ...] = ("task_check", "error", "user", "retry", "judge")

#: ``reason`` when nothing usable was found.
NO_SIGNALS = "no_signals"


class Label(NamedTuple):
    """Result of labelling: unpacks as ``(success, confidence, reason)``.

    ``success`` is ``None`` when there is no usable evidence. ``reason`` is a stable string
    ``"<tier>:<signal>"`` (for example ``task_check:tests_passed``) or ``"no_signals"``.
    """

    success: bool | None
    confidence: float
    reason: str


@dataclass(frozen=True)
class LabelConfig:
    """Confidence per tier and the judge threshold.

    The confidence numbers are untuned placeholders that only encode an ordering (they
    decrease down the precedence list). They are not measured and not calibrated
    probabilities. ``confidence_judge`` is the maximum for the judge tier; the actual value
    scales linearly with the distance of the score from ``judge_threshold``.
    """

    confidence_task_check: float = 0.95
    confidence_error: float = 0.85
    confidence_user: float = 0.70
    confidence_retry: float = 0.40
    confidence_judge: float = 0.30
    judge_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.judge_threshold <= 1.0:
            raise ValueError(f"judge_threshold must be in [0, 1], got {self.judge_threshold!r}")


DEFAULT_LABEL_CONFIG = LabelConfig()


# --- typed readers: a wrong type counts as absent -------------------------------------------


def _flag(signals: Mapping[str, Any], key: str) -> bool | None:
    v = signals.get(key)
    return v if isinstance(v, bool) else None


def _count(signals: Mapping[str, Any], key: str) -> int | None:
    v = signals.get(key)
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _score(signals: Mapping[str, Any], key: str) -> float | None:
    v = signals.get(key)
    if isinstance(v, bool) or not isinstance(v, int | float):
        return None
    f = float(v)
    return f if math.isfinite(f) and 0.0 <= f <= 1.0 else None


def label(signals: Mapping[str, Any], config: LabelConfig = DEFAULT_LABEL_CONFIG) -> Label:
    """Label a step from its raw signals. Never raises on odd values; reads, never mutates."""
    # Tier 1: explicit task check. Any False is failure, otherwise any True is success.
    checks = {k: _flag(signals, k) for k in (TESTS_PASSED, SCHEMA_VALID)}
    for key, val in checks.items():
        if val is False:
            return Label(False, config.confidence_task_check, f"task_check:{key}")
    for key, val in checks.items():
        if val is True:
            return Label(True, config.confidence_task_check, f"task_check:{key}")

    # Tier 2: tool or compile error. True is failure; False or absent is no evidence.
    for key in (TOOL_ERROR, COMPILE_ERROR):
        if _flag(signals, key) is True:
            return Label(False, config.confidence_error, f"error:{key}")

    # Tier 3: user. Rejection or rewrite beats approval.
    for key in (USER_REJECTED, USER_REWROTE):
        if _flag(signals, key) is True:
            return Label(False, config.confidence_user, f"user:{key}")
    if _flag(signals, USER_APPROVED) is True:
        return Label(True, config.confidence_user, f"user:{USER_APPROVED}")

    # Tier 4: retry on the same step. Weak, and only ever evidence of failure.
    if _flag(signals, RETRIED) is True:
        return Label(False, config.confidence_retry, f"retry:{RETRIED}")
    count = _count(signals, RETRY_COUNT)
    if count is not None and count > 0:
        return Label(False, config.confidence_retry, f"retry:{RETRY_COUNT}")

    # Tier 5: judge score. Confidence grows with distance from the threshold.
    score = _score(signals, JUDGE_SCORE)
    if score is not None:
        t = config.judge_threshold
        ok = score >= t
        span = (1.0 - t) if ok else t  # room between the threshold and the nearer extreme
        scale = abs(score - t) / span if span > 0 else 0.0
        return Label(ok, config.confidence_judge * scale, f"judge:{JUDGE_SCORE}")

    return Label(None, 0.0, NO_SIGNALS)


def apply_label(
    outcome: Outcome,
    config: LabelConfig = DEFAULT_LABEL_CONFIG,
    *,
    overwrite: bool = False,
) -> Outcome:
    """Return a new ``Outcome`` with ``success`` set from its signals.

    ``signals`` is copied unchanged (same keys and values; no label metadata is added). By
    default an existing non-None ``success`` is left alone. With ``overwrite=True`` it is
    recomputed from the signals; if the signals give no evidence the existing value is kept
    rather than erased. The input is not modified.
    """
    signals = dict(outcome.signals)
    if outcome.success is not None and not overwrite:
        return dataclasses.replace(outcome, signals=signals)
    success = label(signals, config).success
    if success is None:
        success = outcome.success
    return dataclasses.replace(outcome, success=success, signals=signals)
