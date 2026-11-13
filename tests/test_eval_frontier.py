"""The single-configuration frontier and the verdict rule (docs/EVALUATION.md)."""

import math

import pytest

from steproute.eval.frontier import Point, cost_at, envelope, margin, pareto, success_at
from steproute.eval.results import BEATS, DOES_NOT_BEAT, NOT_SHOWN, decide_verdict
from steproute.eval.stats import Interval

A = Point("a", 1.0, 0.2)
B = Point("b", 2.0, 0.8)
C = Point("c", 4.0, 0.9)
D = Point("d", 3.0, 0.5)  # dominated by b


def test_pareto_drops_dominated_and_duplicates() -> None:
    dup = Point("b2", 2.0, 0.8)
    assert [p.label for p in pareto([A, B, C, D, dup])] == ["a", "b", "c"]


def test_envelope_starts_at_origin_and_is_concave() -> None:
    env = envelope([A, B, C, D])
    # a (slope 0.2) lies below the chord from the origin to b (slope 0.4): not a vertex.
    assert [p.label for p in env][1:] == ["b", "c"]
    assert (env[0].cost, env[0].success) == (0.0, 0.0)
    slopes = [
        (q.success - p.success) / (q.cost - p.cost) for p, q in zip(env, env[1:], strict=False)
    ]
    assert slopes == sorted(slopes, reverse=True)


def test_success_and_cost_at_hand_values() -> None:
    env = envelope([A, B, C])
    assert success_at(env, 1.0) == pytest.approx(0.4)  # half of b, half declined
    assert success_at(env, 3.0) == pytest.approx(0.85)  # midway between b and c
    assert success_at(env, 10.0) == pytest.approx(0.9)  # flat beyond the top
    assert cost_at(env, 0.85) == pytest.approx(3.0)
    assert cost_at(env, 0.95) == math.inf
    assert margin(env, 1.0, 0.3) == pytest.approx(-0.1)


def test_free_configuration_anchors_the_frontier() -> None:
    free = Point("local", 0.0, 0.3)
    env = envelope([free, B])
    assert env[0].label == "local"
    assert success_at(env, 1.0) == pytest.approx(0.55)


def test_envelope_of_nothing_is_the_origin() -> None:
    env = envelope([])
    assert success_at(env, 5.0) == 0.0


def _verdict(lo: float, hi: float, gates: tuple[str, ...] = ()) -> str:
    return decide_verdict(
        "p", "task", 1.0, 0.5, (lo + hi) / 2, Interval(lo, hi), list(gates), [A, B, C], 0.95
    ).status


def test_verdict_rule() -> None:
    assert _verdict(0.01, 0.05) == BEATS
    assert _verdict(-0.05, -0.01) == DOES_NOT_BEAT
    assert _verdict(-0.01, 0.02) == NOT_SHOWN
    # A tie with a frontier configuration (margin exactly 0) does not beat it.
    assert _verdict(0.0, 0.0) == DOES_NOT_BEAT
    assert _verdict(-0.02, 0.0) == DOES_NOT_BEAT
    # Any gate forces "not shown", whatever the interval says.
    assert _verdict(0.1, 0.2, ("effective sample size 3 is below 50",)) == NOT_SHOWN


def test_verdict_sentences_are_plain() -> None:
    v = decide_verdict("p", "task", 1.0, 0.6, 0.2, Interval(0.1, 0.3), [], [A, B, C], 0.95)
    assert v.sentence.startswith("p beats every single configuration")
    v = decide_verdict("p", "task", 3.0, 0.5, -0.35, Interval(-0.4, -0.3), [], [A, B, C], 0.95)
    assert "does not beat every single configuration" in v.sentence
    assert "b alone costs" in v.sentence
    v = decide_verdict("p", "task", 1.0, 0.4, 0.0, Interval(-0.1, 0.1), [], [A, B, C], 0.95)
    assert v.sentence.startswith("Not shown: this data cannot tell")
    v = decide_verdict("p", "task", 1.0, 0.4, None, None, ["too few"], [A], 0.95)
    assert v.reasons == ("too few",) and "too few" in v.sentence


def test_unestimable_margin_is_not_shown() -> None:
    v = decide_verdict("p", "task", 1.0, 0.4, 0.1, Interval(math.nan, math.nan), [], [A, B], 0.95)
    assert v.status == NOT_SHOWN
    assert "could not be estimated" in v.reasons[0]
