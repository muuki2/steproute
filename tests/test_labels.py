"""Outcome labelling: precedence, absence, malformed values, application to an Outcome."""

from __future__ import annotations

import itertools
import math
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from steproute.core import Label, LabelConfig, Outcome, apply_label, label
from steproute.core import labels as L

# One representative way each tier says success / failure (absent: that tier cannot say it).
SAYS: dict[str, dict[bool, dict[str, Any]]] = {
    "task_check": {True: {"tests_passed": True}, False: {"tests_passed": False}},
    "error": {False: {"tool_error": True}},
    "user": {True: {"user_approved": True}, False: {"user_rejected": True}},
    "retry": {False: {"retried": True}},
    "judge": {True: {"judge_score": 0.9}, False: {"judge_score": 0.1}},
}

TIER_PAIRS = list(itertools.combinations(L.TIERS, 2))

# Every (higher, lower) pair where the lower tier disagrees with the higher one.
DISAGREE = [
    (hi, lo, hv, lv) for hi, lo in TIER_PAIRS for hv in SAYS[hi] for lv in SAYS[lo] if hv != lv
]


@pytest.mark.parametrize(("hi", "lo", "hv", "lv"), DISAGREE)
def test_higher_tier_wins_when_tiers_disagree(hi: str, lo: str, hv: bool, lv: bool) -> None:
    result = label({**SAYS[hi][hv], **SAYS[lo][lv]})
    assert result.success is hv
    assert result.reason.startswith(f"{hi}:")


@pytest.mark.parametrize(("hi", "lo"), TIER_PAIRS)
def test_higher_tier_decides_when_tiers_agree_too(hi: str, lo: str) -> None:
    for hv, lv in itertools.product(SAYS[hi], SAYS[lo]):
        result = label({**SAYS[hi][hv], **SAYS[lo][lv]})
        assert result.success is hv
        assert result.reason.startswith(f"{hi}:")


def test_disagreement_table_covers_every_tier_pair_and_both_directions() -> None:
    # error and retry can only say failure, so that one pair has no disagreement to test; the
    # agreement test above still pins which of the two decides.
    assert {(hi, lo) for hi, lo, _, _ in DISAGREE} == set(TIER_PAIRS) - {("error", "retry")}
    # Both directions wherever the higher tier can say either.
    assert {hv for hi, lo, hv, _ in DISAGREE if (hi, lo) == ("task_check", "judge")} == {
        True,
        False,
    }


def test_precedence_order_is_the_documented_one() -> None:
    assert L.TIERS == ("task_check", "error", "user", "retry", "judge")


def test_no_signals_is_none_never_false() -> None:
    assert label({}) == Label(None, 0.0, "no_signals")
    assert label({"unrelated": 1, "note": "x"}) == Label(None, 0.0, "no_signals")


@pytest.mark.parametrize(
    "signals",
    [
        {"tool_error": False},
        {"compile_error": False, "tool_error": False},
        {"user_rejected": False, "user_rewrote": False, "user_approved": False},
        {"retried": False, "retry_count": 0},
        {"tool_error": False, "retried": False, "retry_count": 0, "user_rejected": False},
    ],
)
def test_negative_flags_are_no_evidence(signals: dict[str, Any]) -> None:
    assert label(signals).success is None


def test_false_error_flag_does_not_block_lower_tiers() -> None:
    assert label({"tool_error": False, "judge_score": 0.2}).success is False
    assert label({"tool_error": False, "judge_score": 0.8}).success is True


@pytest.mark.parametrize(
    ("signals", "success", "reason"),
    [
        ({"tests_passed": True, "schema_valid": False}, False, "task_check:schema_valid"),
        ({"tests_passed": False, "schema_valid": True}, False, "task_check:tests_passed"),
        ({"tests_passed": True, "schema_valid": True}, True, "task_check:tests_passed"),
        ({"schema_valid": True}, True, "task_check:schema_valid"),
        ({"tool_error": True, "compile_error": False}, False, "error:tool_error"),
        ({"compile_error": True}, False, "error:compile_error"),
        ({"user_approved": True, "user_rejected": True}, False, "user:user_rejected"),
        ({"user_approved": True, "user_rewrote": True}, False, "user:user_rewrote"),
        ({"user_approved": True}, True, "user:user_approved"),
        ({"retried": True}, False, "retry:retried"),
        ({"retry_count": 2}, False, "retry:retry_count"),
        ({"retried": False, "retry_count": 3}, False, "retry:retry_count"),
    ],
)
def test_within_tier_rules(signals: dict[str, Any], success: bool, reason: str) -> None:
    result = label(signals)
    assert result.success is success
    assert result.reason == reason


@pytest.mark.parametrize(
    "signals",
    [
        {"tests_passed": "yes"},
        {"tests_passed": 1, "schema_valid": 0},
        {"tests_passed": None},
        {"tool_error": "true"},
        {"tool_error": 1},
        {"user_rejected": "yes", "user_approved": 1},
        {"retried": "yes"},
        {"retry_count": True},
        {"retry_count": "3"},
        {"retry_count": 2.0},
        {"judge_score": "0.9"},
        {"judge_score": True},
        {"judge_score": 1.5},
        {"judge_score": -0.1},
        {"judge_score": math.nan},
        {"judge_score": math.inf},
        {"judge_score": None},
        {"judge_score": [0.9]},
    ],
)
def test_malformed_values_count_as_absent(signals: dict[str, Any]) -> None:
    assert label(signals) == Label(None, 0.0, "no_signals")


def test_malformed_high_tier_falls_through_to_lower_tier() -> None:
    assert label({"tests_passed": "yes", "judge_score": 0.9}).reason == "judge:judge_score"


def test_judge_threshold_and_confidence_scaling() -> None:
    cap = LabelConfig().confidence_judge
    at = label({"judge_score": 0.5})
    assert at.success is True
    assert at.confidence == 0.0
    assert 0.0 < label({"judge_score": 0.6}).confidence < label({"judge_score": 1.0}).confidence
    assert label({"judge_score": 1.0}).confidence == cap
    assert label({"judge_score": 0.0}) == Label(False, cap, "judge:judge_score")
    assert label({"judge_score": 0.7}).success is True
    assert label({"judge_score": 0.7}, LabelConfig(judge_threshold=0.8)).success is False
    assert label({"judge_score": 1}).success is True  # in-range ints are fine


def test_judge_extreme_thresholds() -> None:
    cfg1 = LabelConfig(judge_threshold=1.0)
    assert label({"judge_score": 1.0}, cfg1) == Label(True, 0.0, "judge:judge_score")
    assert label({"judge_score": 0.0}, LabelConfig(judge_threshold=0.0)).success is True
    with pytest.raises(ValueError, match="judge_threshold"):
        LabelConfig(judge_threshold=1.5)


def test_default_confidences_decrease_down_the_precedence_order() -> None:
    c = LabelConfig()
    ordered = [
        c.confidence_task_check,
        c.confidence_error,
        c.confidence_user,
        c.confidence_retry,
        c.confidence_judge,
    ]
    assert ordered == sorted(ordered, reverse=True)
    assert len(set(ordered)) == len(ordered)
    assert label({"tests_passed": True}).confidence == c.confidence_task_check


def test_label_unpacks_as_three_tuple() -> None:
    success, confidence, reason = label({"tool_error": True})
    assert (success, reason) == (False, "error:tool_error")
    assert confidence > 0


def test_label_does_not_mutate_signals() -> None:
    signals: dict[str, Any] = {"tests_passed": True, "judge_score": 0.1, "extra": [1]}
    label(signals)
    assert signals == {"tests_passed": True, "judge_score": 0.1, "extra": [1]}


# --- apply_label ----------------------------------------------------------------------------


def _outcome(success: bool | None, signals: dict[str, Any]) -> Outcome:
    return Outcome(decision_id="d", ts=1.0, success=success, signals=signals)


def test_apply_label_sets_success_and_keeps_signals_exactly() -> None:
    out = _outcome(None, {"tests_passed": False, "judge_score": 0.9, "note": "x"})
    new = apply_label(out)
    assert new.success is False
    assert new.signals == {"tests_passed": False, "judge_score": 0.9, "note": "x"}
    assert out.success is None  # input untouched
    assert new.signals is not out.signals


def test_apply_label_without_evidence_stays_none() -> None:
    assert apply_label(_outcome(None, {})).success is None
    assert apply_label(_outcome(None, {"tool_error": False})).success is None


def test_apply_label_keeps_existing_value_by_default() -> None:
    assert apply_label(_outcome(True, {"tests_passed": False})).success is True
    assert apply_label(_outcome(False, {"tests_passed": True})).success is False


def test_apply_label_overwrite_recomputes() -> None:
    assert apply_label(_outcome(True, {"tests_passed": False}), overwrite=True).success is False


def test_apply_label_overwrite_never_erases_with_none() -> None:
    assert apply_label(_outcome(True, {}), overwrite=True).success is True


def test_apply_label_uses_config() -> None:
    cfg = LabelConfig(judge_threshold=0.9)
    assert apply_label(_outcome(None, {"judge_score": 0.8}), cfg).success is False


# --- property-based -------------------------------------------------------------------------

_json = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats() | st.text(),
    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.text(), inner, max_size=3),
    max_leaves=8,
)
_keys = st.sampled_from(L.SIGNAL_KEYS) | st.text()


@given(st.dictionaries(_keys, _json, max_size=8))
def test_arbitrary_signals_never_raise(signals: dict[str, Any]) -> None:
    result = label(signals)
    assert result.success is None or isinstance(result.success, bool)
    assert 0.0 <= result.confidence <= 1.0
    assert (result.success is None) == (result.reason == "no_signals")


@given(st.dictionaries(st.text().filter(lambda k: k not in L.SIGNAL_KEYS), _json, max_size=5))
def test_unknown_keys_only_is_none(signals: dict[str, Any]) -> None:
    assert label(signals).success is None


def test_empty_dict_is_always_none() -> None:
    for cfg in (LabelConfig(), LabelConfig(judge_threshold=0.0), LabelConfig(judge_threshold=1.0)):
        assert label({}, cfg) == Label(None, 0.0, "no_signals")
