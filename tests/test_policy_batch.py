"""The optional batch predictor path of ScoredPolicy: same answers, one call per decision."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from steproute.core import Action, BatchPredictor, ScoredPolicy, StepState
from steproute.core.catalog import load_default_catalog
from steproute.core.cost import CostModelError

DEFAULT = load_default_catalog(warn_unverified=False)

STATES = [
    StepState("proxy", "s0", 0, "main", 12_000),
    StepState(
        "proxy",
        "s1",
        3,
        "tool_followup",
        20_000,
        cached_prefix_tokens=18_000,
        current_model="claude-sonnet-5-5",
        current_effort="high",
        tools=["Bash"],
    ),
    StepState("proxy", "s2", 1, "plan", 3_000, sensitive=True),
]


def _p(state: StepState, step: Action) -> float:
    """A predictor that depends on the model, the effort and the failure history."""
    base = 0.3 + 0.1 * (len(step.model) % 5) / 5 + 0.05 * (step.effort == "high")
    return min(0.95, base + 0.1 * len(state.failed_attempts) + state.prompt_tokens / 1e6)


class PerCallOnly:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, state: StepState, action_step: Action) -> float:
        self.calls += 1
        return _p(state, action_step)


class Batched(PerCallOnly):
    def __init__(self) -> None:
        super().__init__()
        self.many_calls = 0
        self.many_sizes: list[int] = []

    def predict_many(self, pairs: Sequence[tuple[StepState, Action]]) -> list[float]:
        self.many_calls += 1
        self.many_sizes.append(len(pairs))
        return [_p(s, a) for s, a in pairs]


def test_batch_path_returns_exactly_what_the_per_call_path_returns() -> None:
    one = ScoredPolicy(PerCallOnly(), DEFAULT, 0.9, 0.05, seed=3)
    many = ScoredPolicy(Batched(), DEFAULT, 0.9, 0.05, seed=3)
    for state in STATES:
        assert many.score(state) == one.score(state)
        assert many.action_probabilities(state) == one.action_probabilities(state)
    a, b = one.decide(STATES[1]), many.decide(STATES[1])
    assert (a.action, a.propensity, a.predicted_success, a.predicted_cost) == (
        b.action,
        b.propensity,
        b.predicted_success,
        b.predicted_cost,
    )


def test_batch_predictor_is_called_once_per_decision() -> None:
    pred = Batched()
    policy = ScoredPolicy(pred, DEFAULT, 0.9, 0.0, seed=1)
    policy.decide(STATES[0])
    assert pred.many_calls == 1
    assert pred.calls == 0
    # One pair per distinct chain prefix: at depth 2 every candidate chain is a node.
    assert pred.many_sizes[0] == len(policy.score(STATES[0]))
    assert isinstance(pred, BatchPredictor)
    assert not isinstance(PerCallOnly(), BatchPredictor)


def test_predictor_without_predict_many_is_unchanged() -> None:
    pred = PerCallOnly()
    ScoredPolicy(pred, DEFAULT, 0.9, 0.0, seed=1).decide(STATES[0])
    assert pred.calls > 0


def test_wrong_length_and_bad_values_from_predict_many_are_refused() -> None:
    class Short(Batched):
        def predict_many(self, pairs: Sequence[tuple[StepState, Action]]) -> list[float]:
            return [0.5] * (len(pairs) - 1)

    class Bad(Batched):
        def predict_many(self, pairs: Sequence[tuple[StepState, Action]]) -> list[float]:
            return [1.5] * len(pairs)

    with pytest.raises(CostModelError, match="values for"):
        ScoredPolicy(Short(), DEFAULT, 0.9, 0.0).decide(STATES[0])
    with pytest.raises(CostModelError, match="expected"):
        ScoredPolicy(Bad(), DEFAULT, 0.9, 0.0).decide(STATES[0])
